
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\simplify.py ===
from __future__ import annotations

from typing import overload

from collections import defaultdict

from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core import (Basic, S, Add, Mul, Pow, Symbol, sympify,
                        expand_func, Function, Dummy, Expr, factor_terms,
                        expand_power_exp, Eq)
from sympy.core.exprtools import factor_nc
from sympy.core.parameters import global_parameters
from sympy.core.function import (expand_log, count_ops, _mexpand,
    nfloat, expand_mul, expand)
from sympy.core.numbers import Float, I, pi, Rational, equal_valued
from sympy.core.relational import Relational
from sympy.core.rules import Transform
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympify
from sympy.core.traversal import bottom_up as _bottom_up, walk as _walk
from sympy.functions import gamma, exp, sqrt, log, exp_polar, re
from sympy.functions.combinatorial.factorials import CombinatorialFunction
from sympy.functions.elementary.complexes import unpolarify, Abs, sign
from sympy.functions.elementary.exponential import ExpBase
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.piecewise import (Piecewise, piecewise_fold,
                                                  piecewise_simplify)
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.special.bessel import (BesselBase, besselj, besseli,
                                            besselk, bessely, jn)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Boolean
from sympy.matrices.expressions import (MatrixExpr, MatAdd, MatMul,
                                            MatPow, MatrixSymbol)
from sympy.polys import together, cancel, factor
from sympy.polys.numberfields.minpoly import _is_sum_surds, _minimal_polynomial_sq
from sympy.sets.sets import Set
from sympy.simplify.combsimp import combsimp
from sympy.simplify.cse_opts import sub_pre, sub_post
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import radsimp, fraction, collect_abs
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.trigsimp import trigsimp, exptrigsimp
from sympy.utilities.decorator import deprecated
from sympy.utilities.iterables import has_variety, sift, subsets, iterable
from sympy.utilities.misc import as_int

import mpmath


def separatevars(expr, symbols=[], dict=False, force=False):
    """
    Separates variables in an expression, if possible.  By
    default, it separates with respect to all symbols in an
    expression and collects constant coefficients that are
    independent of symbols.

    Explanation
    ===========

    If ``dict=True`` then the separated terms will be returned
    in a dictionary keyed to their corresponding symbols.
    By default, all symbols in the expression will appear as
    keys; if symbols are provided, then all those symbols will
    be used as keys, and any terms in the expression containing
    other symbols or non-symbols will be returned keyed to the
    string 'coeff'. (Passing None for symbols will return the
    expression in a dictionary keyed to 'coeff'.)

    If ``force=True``, then bases of powers will be separated regardless
    of assumptions on the symbols involved.

    Notes
    =====

    The order of the factors is determined by Mul, so that the
    separated expressions may not necessarily be grouped together.

    Although factoring is necessary to separate variables in some
    expressions, it is not necessary in all cases, so one should not
    count on the returned factors being factored.

    Examples
    ========

    >>> from sympy.abc import x, y, z, alpha
    >>> from sympy import separatevars, sin
    >>> separatevars((x*y)**y)
    (x*y)**y
    >>> separatevars((x*y)**y, force=True)
    x**y*y**y

    >>> e = 2*x**2*z*sin(y)+2*z*x**2
    >>> separatevars(e)
    2*x**2*z*(sin(y) + 1)
    >>> separatevars(e, symbols=(x, y), dict=True)
    {'coeff': 2*z, x: x**2, y: sin(y) + 1}
    >>> separatevars(e, [x, y, alpha], dict=True)
    {'coeff': 2*z, alpha: 1, x: x**2, y: sin(y) + 1}

    If the expression is not really separable, or is only partially
    separable, separatevars will do the best it can to separate it
    by using factoring.

    >>> separatevars(x + x*y - 3*x**2)
    -x*(3*x - y - 1)

    If the expression is not separable then expr is returned unchanged
    or (if dict=True) then None is returned.

    >>> eq = 2*x + y*sin(x)
    >>> separatevars(eq) == eq
    True
    >>> separatevars(2*x + y*sin(x), symbols=(x, y), dict=True) is None
    True

    """
    expr = sympify(expr)
    if dict:
        return _separatevars_dict(_separatevars(expr, force), symbols)
    else:
        return _separatevars(expr, force)


def _separatevars(expr, force):
    if isinstance(expr, Abs):
        arg = expr.args[0]
        if arg.is_Mul and not arg.is_number:
            s = separatevars(arg, dict=True, force=force)
            if s is not None:
                return Mul(*map(expr.func, s.values()))
            else:
                return expr

    if len(expr.free_symbols) < 2:
        return expr

    # don't destroy a Mul since much of the work may already be done
    if expr.is_Mul:
        args = list(expr.args)
        changed = False
        for i, a in enumerate(args):
            args[i] = separatevars(a, force)
            changed = changed or args[i] != a
        if changed:
            expr = expr.func(*args)
        return expr

    # get a Pow ready for expansion
    if expr.is_Pow and expr.base != S.Exp1:
        expr = Pow(separatevars(expr.base, force=force), expr.exp)

    # First try other expansion methods
    expr = expr.expand(mul=False, multinomial=False, force=force)

    _expr, reps = posify(expr) if force else (expr, {})
    expr = factor(_expr).subs(reps)

    if not expr.is_Add:
        return expr

    # Find any common coefficients to pull out
    args = list(expr.args)
    commonc = args[0].args_cnc(cset=True, warn=False)[0]
    for i in args[1:]:
        commonc &= i.args_cnc(cset=True, warn=False)[0]
    commonc = Mul(*commonc)
    commonc = commonc.as_coeff_Mul()[1]  # ignore constants
    commonc_set = commonc.args_cnc(cset=True, warn=False)[0]

    # remove them
    for i, a in enumerate(args):
        c, nc = a.args_cnc(cset=True, warn=False)
        c = c - commonc_set
        args[i] = Mul(*c)*Mul(*nc)
    nonsepar = Add(*args)

    if len(nonsepar.free_symbols) > 1:
        _expr = nonsepar
        _expr, reps = posify(_expr) if force else (_expr, {})
        _expr = (factor(_expr)).subs(reps)

        if not _expr.is_Add:
            nonsepar = _expr

    return commonc*nonsepar


def _separatevars_dict(expr, symbols):
    if symbols:
        if not all(t.is_Atom for t in symbols):
            raise ValueError("symbols must be Atoms.")
        symbols = list(symbols)
    elif symbols is None:
        return {'coeff': expr}
    else:
        symbols = list(expr.free_symbols)
        if not symbols:
            return None

    ret = {i: [] for i in symbols + ['coeff']}

    for i in Mul.make_args(expr):
        expsym = i.free_symbols
        intersection = set(symbols).intersection(expsym)
        if len(intersection) > 1:
            return None
        if len(intersection) == 0:
            # There are no symbols, so it is part of the coefficient
            ret['coeff'].append(i)
        else:
            ret[intersection.pop()].append(i)

    # rebuild
    for k, v in ret.items():
        ret[k] = Mul(*v)

    return ret


def posify(eq):
    """Return ``eq`` (with generic symbols made positive) and a
    dictionary containing the mapping between the old and new
    symbols.

    Explanation
    ===========

    Any symbol that has positive=None will be replaced with a positive dummy
    symbol having the same name. This replacement will allow more symbolic
    processing of expressions, especially those involving powers and
    logarithms.

    A dictionary that can be sent to subs to restore ``eq`` to its original
    symbols is also returned.

    >>> from sympy import posify, Symbol, log, solve
    >>> from sympy.abc import x
    >>> posify(x + Symbol('p', positive=True) + Symbol('n', negative=True))
    (_x + n + p, {_x: x})

    >>> eq = 1/x
    >>> log(eq).expand()
    log(1/x)
    >>> log(posify(eq)[0]).expand()
    -log(_x)
    >>> p, rep = posify(eq)
    >>> log(p).expand().subs(rep)
    -log(x)

    It is possible to apply the same transformations to an iterable
    of expressions:

    >>> eq = x**2 - 4
    >>> solve(eq, x)
    [-2, 2]
    >>> eq_x, reps = posify([eq, x]); eq_x
    [_x**2 - 4, _x]
    >>> solve(*eq_x)
    [2]
    """
    eq = sympify(eq)
    if not isinstance(eq, Basic) and iterable(eq):
        f = type(eq)
        eq = list(eq)
        syms = set()
        for e in eq:
            syms = syms.union(e.atoms(Symbol))
        reps = {}
        for s in syms:
            reps.update({v: k for k, v in posify(s)[1].items()})
        for i, e in enumerate(eq):
            eq[i] = e.subs(reps)
        return f(eq), {r: s for s, r in reps.items()}

    reps = {s: Dummy(s.name, positive=True, **s.assumptions0)
                 for s in eq.free_symbols if s.is_positive is None}
    eq = eq.subs(reps)
    return eq, {r: s for s, r in reps.items()}


def hypersimp(f, k):
    """Given combinatorial term f(k) simplify its consecutive term ratio
       i.e. f(k+1)/f(k).  The input term can be composed of functions and
       integer sequences which have equivalent representation in terms
       of gamma special function.

       Explanation
       ===========

       The algorithm performs three basic steps:

       1. Rewrite all functions in terms of gamma, if possible.

       2. Rewrite all occurrences of gamma in terms of products
          of gamma and rising factorial with integer,  absolute
          constant exponent.

       3. Perform simplification of nested fractions, powers
          and if the resulting expression is a quotient of
          polynomials, reduce their total degree.

       If f(k) is hypergeometric then as result we arrive with a
       quotient of polynomials of minimal degree. Otherwise None
       is returned.

       For more information on the implemented algorithm refer to:

       1. W. Koepf, Algorithms for m-fold Hypergeometric Summation,
          Journal of Symbolic Computation (1995) 20, 399-417
    """
    f = sympify(f)

    g = f.subs(k, k + 1) / f

    g = g.rewrite(gamma)
    if g.has(Piecewise):
        g = piecewise_fold(g)
        g = g.args[-1][0]
    g = expand_func(g)
    g = powsimp(g, deep=True, combine='exp')

    if g.is_rational_function(k):
        return simplify(g, ratio=S.Infinity)
    else:
        return None


def hypersimilar(f, g, k):
    """
    Returns True if ``f`` and ``g`` are hyper-similar.

    Explanation
    ===========

    Similarity in hypergeometric sense means that a quotient of
    f(k) and g(k) is a rational function in ``k``. This procedure
    is useful in solving recurrence relations.

    For more information see hypersimp().

    """
    f, g = list(map(sympify, (f, g)))

    h = (f/g).rewrite(gamma)
    h = h.expand(func=True, basic=False)

    return h.is_rational_function(k)


def signsimp(expr, evaluate=None):
    """Make all Add sub-expressions canonical wrt sign.

    Explanation
    ===========

    If an Add subexpression, ``a``, can have a sign extracted,
    as determined by could_extract_minus_sign, it is replaced
    with Mul(-1, a, evaluate=False). This allows signs to be
    extracted from powers and products.

    Examples
    ========

    >>> from sympy import signsimp, exp, symbols
    >>> from sympy.abc import x, y
    >>> i = symbols('i', odd=True)
    >>> n = -1 + 1/x
    >>> n/x/(-n)**2 - 1/n/x
    (-1 + 1/x)/(x*(1 - 1/x)**2) - 1/(x*(-1 + 1/x))
    >>> signsimp(_)
    0
    >>> x*n + x*-n
    x*(-1 + 1/x) + x*(1 - 1/x)
    >>> signsimp(_)
    0

    Since powers automatically handle leading signs

    >>> (-2)**i
    -2**i

    signsimp can be used to put the base of a power with an integer
    exponent into canonical form:

    >>> n**i
    (-1 + 1/x)**i

    By default, signsimp does not leave behind any hollow simplification:
    if making an Add canonical wrt sign didn't change the expression, the
    original Add is restored. If this is not desired then the keyword
    ``evaluate`` can be set to False:

    >>> e = exp(y - x)
    >>> signsimp(e) == e
    True
    >>> signsimp(e, evaluate=False)
    exp(-(x - y))

    """
    if evaluate is None:
        evaluate = global_parameters.evaluate
    expr = sympify(expr)
    if not isinstance(expr, (Expr, Relational)) or expr.is_Atom:
        return expr
    # get rid of an pre-existing unevaluation regarding sign
    e = expr.replace(lambda x: x.is_Mul and -(-x) != x, lambda x: -(-x))
    e = sub_post(sub_pre(e))
    if not isinstance(e, (Expr, Relational)) or e.is_Atom:
        return e
    if e.is_Add:
        rv = e.func(*[signsimp(a) for a in e.args])
        if not evaluate and isinstance(rv, Add
                ) and rv.could_extract_minus_sign():
            return Mul(S.NegativeOne, -rv, evaluate=False)
        return rv
    if evaluate:
        e = e.replace(lambda x: x.is_Mul and -(-x) != x, lambda x: -(-x))
    return e


@overload
def simplify(expr: Expr, **kwargs) -> Expr: ...
@overload
def simplify(expr: Boolean, **kwargs) -> Boolean: ...
@overload
def simplify(expr: Set, **kwargs) -> Set: ...
@overload
def simplify(expr: Basic, **kwargs) -> Basic: ...

def simplify(expr, ratio=1.7, measure=count_ops, rational=False, inverse=False, doit=True, **kwargs):
    """Simplifies the given expression.

    Explanation
    ===========

    Simplification is not a well defined term and the exact strategies
    this function tries can change in the future versions of SymPy. If
    your algorithm relies on "simplification" (whatever it is), try to
    determine what you need exactly  -  is it powsimp()?, radsimp()?,
    together()?, logcombine()?, or something else? And use this particular
    function directly, because those are well defined and thus your algorithm
    will be robust.

    Nonetheless, especially for interactive use, or when you do not know
    anything about the structure of the expression, simplify() tries to apply
    intelligent heuristics to make the input expression "simpler".  For
    example:

    >>> from sympy import simplify, cos, sin
    >>> from sympy.abc import x, y
    >>> a = (x + x**2)/(x*sin(y)**2 + x*cos(y)**2)
    >>> a
    (x**2 + x)/(x*sin(y)**2 + x*cos(y)**2)
    >>> simplify(a)
    x + 1

    Note that we could have obtained the same result by using specific
    simplification functions:

    >>> from sympy import trigsimp, cancel
    >>> trigsimp(a)
    (x**2 + x)/x
    >>> cancel(_)
    x + 1

    In some cases, applying :func:`simplify` may actually result in some more
    complicated expression. The default ``ratio=1.7`` prevents more extreme
    cases: if (result length)/(input length) > ratio, then input is returned
    unmodified.  The ``measure`` parameter lets you specify the function used
    to determine how complex an expression is.  The function should take a
    single argument as an expression and return a number such that if
    expression ``a`` is more complex than expression ``b``, then
    ``measure(a) > measure(b)``.  The default measure function is
    :func:`~.count_ops`, which returns the total number of operations in the
    expression.

    For example, if ``ratio=1``, ``simplify`` output cannot be longer
    than input.

    ::

        >>> from sympy import sqrt, simplify, count_ops, oo
        >>> root = 1/(sqrt(2)+3)

    Since ``simplify(root)`` would result in a slightly longer expression,
    root is returned unchanged instead::

       >>> simplify(root, ratio=1) == root
       True

    If ``ratio=oo``, simplify will be applied anyway::

        >>> count_ops(simplify(root, ratio=oo)) > count_ops(root)
        True

    Note that the shortest expression is not necessary the simplest, so
    setting ``ratio`` to 1 may not be a good idea.
    Heuristically, the default value ``ratio=1.7`` seems like a reasonable
    choice.

    You can easily define your own measure function based on what you feel
    should represent the "size" or "complexity" of the input expression.  Note
    that some choices, such as ``lambda expr: len(str(expr))`` may appear to be
    good metrics, but have other problems (in this case, the measure function
    may slow down simplify too much for very large expressions).  If you do not
    know what a good metric would be, the default, ``count_ops``, is a good
    one.

    For example:

    >>> from sympy import symbols, log
    >>> a, b = symbols('a b', positive=True)
    >>> g = log(a) + log(b) + log(a)*log(1/b)
    >>> h = simplify(g)
    >>> h
    log(a*b**(1 - log(a)))
    >>> count_ops(g)
    8
    >>> count_ops(h)
    5

    So you can see that ``h`` is simpler than ``g`` using the count_ops metric.
    However, we may not like how ``simplify`` (in this case, using
    ``logcombine``) has created the ``b**(log(1/a) + 1)`` term.  A simple way
    to reduce this would be to give more weight to powers as operations in
    ``count_ops``.  We can do this by using the ``visual=True`` option:

    >>> print(count_ops(g, visual=True))
    2*ADD + DIV + 4*LOG + MUL
    >>> print(count_ops(h, visual=True))
    2*LOG + MUL + POW + SUB

    >>> from sympy import Symbol, S
    >>> def my_measure(expr):
    ...     POW = Symbol('POW')
    ...     # Discourage powers by giving POW a weight of 10
    ...     count = count_ops(expr, visual=True).subs(POW, 10)
    ...     # Every other operation gets a weight of 1 (the default)
    ...     count = count.replace(Symbol, type(S.One))
    ...     return count
    >>> my_measure(g)
    8
    >>> my_measure(h)
    14
    >>> 15./8 > 1.7 # 1.7 is the default ratio
    True
    >>> simplify(g, measure=my_measure)
    -log(a)*log(b) + log(a) + log(b)

    Note that because ``simplify()`` internally tries many different
    simplification strategies and then compares them using the measure
    function, we get a completely different result that is still different
    from the input expression by doing this.

    If ``rational=True``, Floats will be recast as Rationals before simplification.
    If ``rational=None``, Floats will be recast as Rationals but the result will
    be recast as Floats. If rational=False(default) then nothing will be done
    to the Floats.

    If ``inverse=True``, it will be assumed that a composition of inverse
    functions, such as sin and asin, can be cancelled in any order.
    For example, ``asin(sin(x))`` will yield ``x`` without checking whether
    x belongs to the set where this relation is true. The default is
    False.

    Note that ``simplify()`` automatically calls ``doit()`` on the final
    expression. You can avoid this behavior by passing ``doit=False`` as
    an argument.

    Also, it should be noted that simplifying a boolean expression is not
    well defined. If the expression prefers automatic evaluation (such as
    :obj:`~.Eq()` or :obj:`~.Or()`), simplification will return ``True`` or
    ``False`` if truth value can be determined. If the expression is not
    evaluated by default (such as :obj:`~.Predicate()`), simplification will
    not reduce it and you should use :func:`~.refine` or :func:`~.ask`
    function. This inconsistency will be resolved in future version.

    See Also
    ========

    sympy.assumptions.refine.refine : Simplification using assumptions.
    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.
    """

    def shorter(*choices):
        """
        Return the choice that has the fewest ops. In case of a tie,
        the expression listed first is selected.
        """
        if not has_variety(choices):
            return choices[0]
        return min(choices, key=measure)

    def done(e):
        rv = e.doit() if doit else e
        return shorter(rv, collect_abs(rv))

    expr = sympify(expr, rational=rational)
    kwargs = {
        "ratio": kwargs.get('ratio', ratio),
        "measure": kwargs.get('measure', measure),
        "rational": kwargs.get('rational', rational),
        "inverse": kwargs.get('inverse', inverse),
        "doit": kwargs.get('doit', doit)}
    # no routine for Expr needs to check for is_zero
    if isinstance(expr, Expr) and expr.is_zero:
        return S.Zero if not expr.is_Number else expr

    _eval_simplify = getattr(expr, '_eval_simplify', None)
    if _eval_simplify is not None:
        return _eval_simplify(**kwargs)

    original_expr = expr = collect_abs(signsimp(expr))

    if not isinstance(expr, Basic) or not expr.args:  # XXX: temporary hack
        return expr

    if inverse and expr.has(Function):
        expr = inversecombine(expr)
        if not expr.args:  # simplified to atomic
            return expr

    # do deep simplification
    handled = Add, Mul, Pow, ExpBase
    expr = expr.replace(
        # here, checking for x.args is not enough because Basic has
        # args but Basic does not always play well with replace, e.g.
        # when simultaneous is True found expressions will be masked
        # off with a Dummy but not all Basic objects in an expression
        # can be replaced with a Dummy
        lambda x: isinstance(x, Expr) and x.args and not isinstance(
            x, handled),
        lambda x: x.func(*[simplify(i, **kwargs) for i in x.args]),
        simultaneous=False)
    if not isinstance(expr, handled):
        return done(expr)

    if not expr.is_commutative:
        expr = nc_simplify(expr)

    # TODO: Apply different strategies, considering expression pattern:
    # is it a purely rational function? Is there any trigonometric function?...
    # See also https://github.com/sympy/sympy/pull/185.

    # rationalize Floats
    floats = False
    if rational is not False and expr.has(Float):
        floats = True
        expr = nsimplify(expr, rational=True)

    expr = _bottom_up(expr, lambda w: getattr(w, 'normal', lambda: w)())
    expr = Mul(*powsimp(expr).as_content_primitive())
    _e = cancel(expr)
    expr1 = shorter(_e, _mexpand(_e).cancel())  # issue 6829
    expr2 = shorter(together(expr, deep=True), together(expr1, deep=True))

    if ratio is S.Infinity:
        expr = expr2
    else:
        expr = shorter(expr2, expr1, expr)
    if not isinstance(expr, Basic):  # XXX: temporary hack
        return expr

    expr = factor_terms(expr, sign=False)

    # must come before `Piecewise` since this introduces more `Piecewise` terms
    if expr.has(sign):
        expr = expr.rewrite(Abs)

    # Deal with Piecewise separately to avoid recursive growth of expressions
    if expr.has(Piecewise):
        # Fold into a single Piecewise
        expr = piecewise_fold(expr)
        # Apply doit, if doit=True
        expr = done(expr)
        # Still a Piecewise?
        if expr.has(Piecewise):
            # Fold into a single Piecewise, in case doit lead to some
            # expressions being Piecewise
            expr = piecewise_fold(expr)
            # kroneckersimp also affects Piecewise
            if expr.has(KroneckerDelta):
                expr = kroneckersimp(expr)
            # Still a Piecewise?
            if expr.has(Piecewise):
                # Do not apply doit on the segments as it has already
                # been done above, but simplify
                expr = piecewise_simplify(expr, deep=True, doit=False)
                # Still a Piecewise?
                if expr.has(Piecewise):
                    # Try factor common terms
                    expr = shorter(expr, factor_terms(expr))
                    # As all expressions have been simplified above with the
                    # complete simplify, nothing more needs to be done here
                    return expr

    # hyperexpand automatically only works on hypergeometric terms
    # Do this after the Piecewise part to avoid recursive expansion
    expr = hyperexpand(expr)

    if expr.has(KroneckerDelta):
        expr = kroneckersimp(expr)

    if expr.has(BesselBase):
        expr = besselsimp(expr)

    if expr.has(TrigonometricFunction, HyperbolicFunction):
        expr = trigsimp(expr, deep=True)

    if expr.has(log):
        expr = shorter(expand_log(expr, deep=True), logcombine(expr))

    if expr.has(CombinatorialFunction, gamma):
        # expression with gamma functions or non-integer arguments is
        # automatically passed to gammasimp
        expr = combsimp(expr)

    if expr.has(Sum):
        expr = sum_simplify(expr, **kwargs)

    if expr.has(Integral):
        expr = expr.xreplace({
            i: factor_terms(i) for i in expr.atoms(Integral)})

    if expr.has(Product):
        expr = product_simplify(expr, **kwargs)

    from sympy.physics.units import Quantity

    if expr.has(Quantity):
        from sympy.physics.units.util import quantity_simplify
        expr = quantity_simplify(expr)

    short = shorter(powsimp(expr, combine='exp', deep=True), powsimp(expr), expr)
    short = shorter(short, cancel(short))
    short = shorter(short, factor_terms(short), expand_power_exp(expand_mul(short)))
    if short.has(TrigonometricFunction, HyperbolicFunction, ExpBase, exp):
        short = exptrigsimp(short)

    # get rid of hollow 2-arg Mul factorization
    hollow_mul = Transform(
        lambda x: Mul(*x.args),
        lambda x:
        x.is_Mul and
        len(x.args) == 2 and
        x.args[0].is_Number and
        x.args[1].is_Add and
        x.is_commutative)
    expr = short.xreplace(hollow_mul)

    numer, denom = expr.as_numer_denom()
    if denom.is_Add:
        n, d = fraction(radsimp(1/denom, symbolic=False, max_terms=1))
        if n is not S.One:
            expr = (numer*n).expand()/d

    if expr.could_extract_minus_sign():
        n, d = fraction(expr)
        if d != 0:
            expr = signsimp(-n/(-d))

    if measure(expr) > ratio*measure(original_expr):
        expr = original_expr

    # restore floats
    if floats and rational is None:
        expr = nfloat(expr, exponent=False)

    return done(expr)


def sum_simplify(s, **kwargs):
    """Main function for Sum simplification"""
    if not isinstance(s, Add):
        s = s.xreplace({a: sum_simplify(a, **kwargs)
            for a in s.atoms(Add) if a.has(Sum)})
    s = expand(s)
    if not isinstance(s, Add):
        return s

    terms = s.args
    s_t = [] # Sum Terms
    o_t = [] # Other Terms

    for term in terms:
        sum_terms, other = sift(Mul.make_args(term),
            lambda i: isinstance(i, Sum), binary=True)
        if not sum_terms:
            o_t.append(term)
            continue
        other = [Mul(*other)]
        s_t.append(Mul(*(other + [s._eval_simplify(**kwargs) for s in sum_terms])))

    result = Add(sum_combine(s_t), *o_t)

    return result


def sum_combine(s_t):
    """Helper function for Sum simplification

       Attempts to simplify a list of sums, by combining limits / sum function's
       returns the simplified sum
    """
    used = [False] * len(s_t)

    for method in range(2):
        for i, s_term1 in enumerate(s_t):
            if not used[i]:
                for j, s_term2 in enumerate(s_t):
                    if not used[j] and i != j:
                        temp = sum_add(s_term1, s_term2, method)
                        if isinstance(temp, (Sum, Mul)):
                            s_t[i] = temp
                            s_term1 = s_t[i]
                            used[j] = True

    result = S.Zero
    for i, s_term in enumerate(s_t):
        if not used[i]:
            result = Add(result, s_term)

    return result

def factor_sum(self, limits=None, radical=False, clear=False, fraction=False, sign=True):
    """Return Sum with constant factors extracted.

    If ``limits`` is specified then ``self`` is the summand; the other
    keywords are passed to ``factor_terms``.

    Examples
    ========

    >>> from sympy import Sum
    >>> from sympy.abc import x, y
    >>> from sympy.simplify.simplify import factor_sum
    >>> s = Sum(x*y, (x, 1, 3))
    >>> factor_sum(s)
    y*Sum(x, (x, 1, 3))
    >>> factor_sum(s.function, s.limits)
    y*Sum(x, (x, 1, 3))
    """

    # XXX deprecate in favor of direct call to factor_terms
    kwargs = {"radical": radical, "clear": clear,
        "fraction": fraction, "sign": sign}
    expr = Sum(self, *limits) if limits else self
    return factor_terms(expr, **kwargs)


def sum_add(self, other, method=0):
    """Helper function for Sum simplification"""
    #we know this is something in terms of a constant * a sum
    #so we temporarily put the constants inside for simplification
    #then simplify the result
    def __refactor(val):
        args = Mul.make_args(val)
        sumv = next(x for x in args if isinstance(x, Sum))
        constant = Mul(*[x for x in args if x != sumv])
        return Sum(constant * sumv.function, *sumv.limits)

    if isinstance(self, Mul):
        rself = __refactor(self)
    else:
        rself = self

    if isinstance(other, Mul):
        rother = __refactor(other)
    else:
        rother = other

    if type(rself) is type(rother):
        if method == 0:
            if rself.limits == rother.limits:
                return factor_sum(Sum(rself.function + rother.function, *rself.limits))
        elif method == 1:
            if simplify(rself.function - rother.function) == 0:
                if len(rself.limits) == len(rother.limits) == 1:
                    i = rself.limits[0][0]
                    x1 = rself.limits[0][1]
                    y1 = rself.limits[0][2]
                    j = rother.limits[0][0]
                    x2 = rother.limits[0][1]
                    y2 = rother.limits[0][2]

                    if i == j:
                        if x2 == y1 + 1:
                            return factor_sum(Sum(rself.function, (i, x1, y2)))
                        elif x1 == y2 + 1:
                            return factor_sum(Sum(rself.function, (i, x2, y1)))

    return Add(self, other)


def product_simplify(s, **kwargs):
    """Main function for Product simplification"""
    terms = Mul.make_args(s)
    p_t = [] # Product Terms
    o_t = [] # Other Terms

    deep = kwargs.get('deep', True)
    for term in terms:
        if isinstance(term, Product):
            if deep:
                p_t.append(Product(term.function.simplify(**kwargs),
                                   *term.limits))
            else:
                p_t.append(term)
        else:
            o_t.append(term)

    used = [False] * len(p_t)

    for method in range(2):
        for i, p_term1 in enumerate(p_t):
            if not used[i]:
                for j, p_term2 in enumerate(p_t):
                    if not used[j] and i != j:
                        tmp_prod = product_mul(p_term1, p_term2, method)
                        if isinstance(tmp_prod, Product):
                            p_t[i] = tmp_prod
                            used[j] = True

    result = Mul(*o_t)

    for i, p_term in enumerate(p_t):
        if not used[i]:
            result = Mul(result, p_term)

    return result


def product_mul(self, other, method=0):
    """Helper function for Product simplification"""
    if type(self) is type(other):
        if method == 0:
            if self.limits == other.limits:
                return Product(self.function * other.function, *self.limits)
        elif method == 1:
            if simplify(self.function - other.function) == 0:
                if len(self.limits) == len(other.limits) == 1:
                    i = self.limits[0][0]
                    x1 = self.limits[0][1]
                    y1 = self.limits[0][2]
                    j = other.limits[0][0]
                    x2 = other.limits[0][1]
                    y2 = other.limits[0][2]

                    if i == j:
                        if x2 == y1 + 1:
                            return Product(self.function, (i, x1, y2))
                        elif x1 == y2 + 1:
                            return Product(self.function, (i, x2, y1))

    return Mul(self, other)


def _nthroot_solve(p, n, prec):
    """
     helper function for ``nthroot``
     It denests ``p**Rational(1, n)`` using its minimal polynomial
    """
    from sympy.solvers import solve
    while n % 2 == 0:
        p = sqrtdenest(sqrt(p))
        n = n // 2
    if n == 1:
        return p
    pn = p**Rational(1, n)
    x = Symbol('x')
    f = _minimal_polynomial_sq(p, n, x)
    if f is None:
        return None
    sols = solve(f, x)
    for sol in sols:
        if abs(sol - pn).n() < 1./10**prec:
            sol = sqrtdenest(sol)
            if _mexpand(sol**n) == p:
                return sol


def logcombine(expr, force=False):
    """
    Takes logarithms and combines them using the following rules:

    - log(x) + log(y) == log(x*y) if both are positive
    - a*log(x) == log(x**a) if x is positive and a is real

    If ``force`` is ``True`` then the assumptions above will be assumed to hold if
    there is no assumption already in place on a quantity. For example, if
    ``a`` is imaginary or the argument negative, force will not perform a
    combination but if ``a`` is a symbol with no assumptions the change will
    take place.

    Examples
    ========

    >>> from sympy import Symbol, symbols, log, logcombine, I
    >>> from sympy.abc import a, x, y, z
    >>> logcombine(a*log(x) + log(y) - log(z))
    a*log(x) + log(y) - log(z)
    >>> logcombine(a*log(x) + log(y) - log(z), force=True)
    log(x**a*y/z)
    >>> x,y,z = symbols('x,y,z', positive=True)
    >>> a = Symbol('a', real=True)
    >>> logcombine(a*log(x) + log(y) - log(z))
    log(x**a*y/z)

    The transformation is limited to factors and/or terms that
    contain logs, so the result depends on the initial state of
    expansion:

    >>> eq = (2 + 3*I)*log(x)
    >>> logcombine(eq, force=True) == eq
    True
    >>> logcombine(eq.expand(), force=True)
    log(x**2) + I*log(x**3)

    See Also
    ========

    posify: replace all symbols with symbols having positive assumptions
    sympy.core.function.expand_log: expand the logarithms of products
        and powers; the opposite of logcombine

    """

    def f(rv):
        if not (rv.is_Add or rv.is_Mul):
            return rv

        def gooda(a):
            # bool to tell whether the leading ``a`` in ``a*log(x)``
            # could appear as log(x**a)
            return (a is not S.NegativeOne and  # -1 *could* go, but we disallow
                (a.is_extended_real or force and a.is_extended_real is not False))

        def goodlog(l):
            # bool to tell whether log ``l``'s argument can combine with others
            a = l.args[0]
            return a.is_positive or force and a.is_nonpositive is not False

        other = []
        logs = []
        log1 = defaultdict(list)
        for a in Add.make_args(rv):
            if isinstance(a, log) and goodlog(a):
                log1[()].append(([], a))
            elif not a.is_Mul:
                other.append(a)
            else:
                ot = []
                co = []
                lo = []
                for ai in a.args:
                    if ai.is_Rational and ai < 0:
                        ot.append(S.NegativeOne)
                        co.append(-ai)
                    elif isinstance(ai, log) and goodlog(ai):
                        lo.append(ai)
                    elif gooda(ai):
                        co.append(ai)
                    else:
                        ot.append(ai)
                if len(lo) > 1:
                    logs.append((ot, co, lo))
                elif lo:
                    log1[tuple(ot)].append((co, lo[0]))
                else:
                    other.append(a)

        # if there is only one log in other, put it with the
        # good logs
        if len(other) == 1 and isinstance(other[0], log):
            log1[()].append(([], other.pop()))
        # if there is only one log at each coefficient and none have
        # an exponent to place inside the log then there is nothing to do
        if not logs and all(len(log1[k]) == 1 and log1[k][0] == [] for k in log1):
            return rv

        # collapse multi-logs as far as possible in a canonical way
        # TODO: see if x*log(a)+x*log(a)*log(b) -> x*log(a)*(1+log(b))?
        # -- in this case, it's unambiguous, but if it were were a log(c) in
        # each term then it's arbitrary whether they are grouped by log(a) or
        # by log(c). So for now, just leave this alone; it's probably better to
        # let the user decide
        for o, e, l in logs:
            l = list(ordered(l))
            e = log(l.pop(0).args[0]**Mul(*e))
            while l:
                li = l.pop(0)
                e = log(li.args[0]**e)
            c, l = Mul(*o), e
            if isinstance(l, log):  # it should be, but check to be sure
                log1[(c,)].append(([], l))
            else:
                other.append(c*l)

        # logs that have the same coefficient can multiply
        for k in list(log1.keys()):
            log1[Mul(*k)] = log(logcombine(Mul(*[
                l.args[0]**Mul(*c) for c, l in log1.pop(k)]),
                force=force), evaluate=False)

        # logs that have oppositely signed coefficients can divide
        for k in ordered(list(log1.keys())):
            if k not in log1:  # already popped as -k
                continue
            if -k in log1:
                # figure out which has the minus sign; the one with
                # more op counts should be the one
                num, den = k, -k
                if num.count_ops() > den.count_ops():
                    num, den = den, num
                other.append(
                    num*log(log1.pop(num).args[0]/log1.pop(den).args[0],
                            evaluate=False))
            else:
                other.append(k*log1.pop(k))

        return Add(*other)

    return _bottom_up(expr, f)


def inversecombine(expr):
    """Simplify the composition of a function and its inverse.

    Explanation
    ===========

    No attention is paid to whether the inverse is a left inverse or a
    right inverse; thus, the result will in general not be equivalent
    to the original expression.

    Examples
    ========

    >>> from sympy.simplify.simplify import inversecombine
    >>> from sympy import asin, sin, log, exp
    >>> from sympy.abc import x
    >>> inversecombine(asin(sin(x)))
    x
    >>> inversecombine(2*log(exp(3*x)))
    6*x
    """

    def f(rv):
        if isinstance(rv, log):
            if isinstance(rv.args[0], exp) or (rv.args[0].is_Pow and rv.args[0].base == S.Exp1):
                rv = rv.args[0].exp
        elif rv.is_Function and hasattr(rv, "inverse"):
            if (len(rv.args) == 1 and len(rv.args[0].args) == 1 and
               isinstance(rv.args[0], rv.inverse(argindex=1))):
                rv = rv.args[0].args[0]
        if rv.is_Pow and rv.base == S.Exp1:
            if isinstance(rv.exp, log):
                rv = rv.exp.args[0]
        return rv

    return _bottom_up(expr, f)


def kroneckersimp(expr):
    """
    Simplify expressions with KroneckerDelta.

    The only simplification currently attempted is to identify multiplicative cancellation:

    Examples
    ========

    >>> from sympy import KroneckerDelta, kroneckersimp
    >>> from sympy.abc import i
    >>> kroneckersimp(1 + KroneckerDelta(0, i) * KroneckerDelta(1, i))
    1
    """
    def args_cancel(args1, args2):
        for i1 in range(2):
            for i2 in range(2):
                a1 = args1[i1]
                a2 = args2[i2]
                a3 = args1[(i1 + 1) % 2]
                a4 = args2[(i2 + 1) % 2]
                if Eq(a1, a2) is S.true and Eq(a3, a4) is S.false:
                    return True
        return False

    def cancel_kronecker_mul(m):
        args = m.args
        deltas = [a for a in args if isinstance(a, KroneckerDelta)]
        for delta1, delta2 in subsets(deltas, 2):
            args1 = delta1.args
            args2 = delta2.args
            if args_cancel(args1, args2):
                return S.Zero * m # In case of oo etc
        return m

    if not expr.has(KroneckerDelta):
        return expr

    if expr.has(Piecewise):
        expr = expr.rewrite(KroneckerDelta)

    newexpr = expr
    expr = None

    while newexpr != expr:
        expr = newexpr
        newexpr = expr.replace(lambda e: isinstance(e, Mul), cancel_kronecker_mul)

    return expr


def besselsimp(expr):
    """
    Simplify bessel-type functions.

    Explanation
    ===========

    This routine tries to simplify bessel-type functions. Currently it only
    works on the Bessel J and I functions, however. It works by looking at all
    such functions in turn, and eliminating factors of "I" and "-1" (actually
    their polar equivalents) in front of the argument. Then, functions of
    half-integer order are rewritten using trigonometric functions and
    functions of integer order (> 1) are rewritten using functions
    of low order.  Finally, if the expression was changed, compute
    factorization of the result with factor().

    >>> from sympy import besselj, besseli, besselsimp, polar_lift, I, S
    >>> from sympy.abc import z, nu
    >>> besselsimp(besselj(nu, z*polar_lift(-1)))
    exp(I*pi*nu)*besselj(nu, z)
    >>> besselsimp(besseli(nu, z*polar_lift(-I)))
    exp(-I*pi*nu/2)*besselj(nu, z)
    >>> besselsimp(besseli(S(-1)/2, z))
    sqrt(2)*cosh(z)/(sqrt(pi)*sqrt(z))
    >>> besselsimp(z*besseli(0, z) + z*(besseli(2, z))/2 + besseli(1, z))
    3*z*besseli(0, z)/2
    """
    # TODO
    # - better algorithm?
    # - simplify (cos(pi*b)*besselj(b,z) - besselj(-b,z))/sin(pi*b) ...
    # - use contiguity relations?

    def replacer(fro, to, factors):
        factors = set(factors)

        def repl(nu, z):
            if factors.intersection(Mul.make_args(z)):
                return to(nu, z)
            return fro(nu, z)
        return repl

    def torewrite(fro, to):
        def tofunc(nu, z):
            return fro(nu, z).rewrite(to)
        return tofunc

    def tominus(fro):
        def tofunc(nu, z):
            return exp(I*pi*nu)*fro(nu, exp_polar(-I*pi)*z)
        return tofunc

    orig_expr = expr

    ifactors = [I, exp_polar(I*pi/2), exp_polar(-I*pi/2)]
    expr = expr.replace(
        besselj, replacer(besselj,
        torewrite(besselj, besseli), ifactors))
    expr = expr.replace(
        besseli, replacer(besseli,
        torewrite(besseli, besselj), ifactors))

    minusfactors = [-1, exp_polar(I*pi)]
    expr = expr.replace(
        besselj, replacer(besselj, tominus(besselj), minusfactors))
    expr = expr.replace(
        besseli, replacer(besseli, tominus(besseli), minusfactors))

    z0 = Dummy('z')

    def expander(fro):
        def repl(nu, z):
            if (nu % 1) == S.Half:
                return simplify(trigsimp(unpolarify(
                        fro(nu, z0).rewrite(besselj).rewrite(jn).expand(
                            func=True)).subs(z0, z)))
            elif nu.is_Integer and nu > 1:
                return fro(nu, z).expand(func=True)
            return fro(nu, z)
        return repl

    expr = expr.replace(besselj, expander(besselj))
    expr = expr.replace(bessely, expander(bessely))
    expr = expr.replace(besseli, expander(besseli))
    expr = expr.replace(besselk, expander(besselk))

    def _bessel_simp_recursion(expr):

        def _use_recursion(bessel, expr):
            while True:
                bessels = expr.find(lambda x: isinstance(x, bessel))
                try:
                    for ba in sorted(bessels, key=lambda x: re(x.args[0])):
                        a, x = ba.args
                        bap1 = bessel(a+1, x)
                        bap2 = bessel(a+2, x)
                        if expr.has(bap1) and expr.has(bap2):
                            expr = expr.subs(ba, 2*(a+1)/x*bap1 - bap2)
                            break
                    else:
                        return expr
                except (ValueError, TypeError):
                    return expr
        if expr.has(besselj):
            expr = _use_recursion(besselj, expr)
        if expr.has(bessely):
            expr = _use_recursion(bessely, expr)
        return expr

    expr = _bessel_simp_recursion(expr)
    if expr != orig_expr:
        expr = expr.factor()

    return expr


def nthroot(expr, n, max_len=4, prec=15):
    """
    Compute a real nth-root of a sum of surds.

    Parameters
    ==========

    expr : sum of surds
    n : integer
    max_len : maximum number of surds passed as constants to ``nsimplify``

    Algorithm
    =========

    First ``nsimplify`` is used to get a candidate root; if it is not a
    root the minimal polynomial is computed; the answer is one of its
    roots.

    Examples
    ========

    >>> from sympy.simplify.simplify import nthroot
    >>> from sympy import sqrt
    >>> nthroot(90 + 34*sqrt(7), 3)
    sqrt(7) + 3

    """
    expr = sympify(expr)
    n = sympify(n)
    p = expr**Rational(1, n)
    if not n.is_integer:
        return p
    if not _is_sum_surds(expr):
        return p
    surds = []
    coeff_muls = [x.as_coeff_Mul() for x in expr.args]
    for x, y in coeff_muls:
        if not x.is_rational:
            return p
        if y is S.One:
            continue
        if not (y.is_Pow and y.exp == S.Half and y.base.is_integer):
            return p
        surds.append(y)
    surds.sort()
    surds = surds[:max_len]
    if expr < 0 and n % 2 == 1:
        p = (-expr)**Rational(1, n)
        a = nsimplify(p, constants=surds)
        res = a if _mexpand(a**n) == _mexpand(-expr) else p
        return -res
    a = nsimplify(p, constants=surds)
    if _mexpand(a) is not _mexpand(p) and _mexpand(a**n) == _mexpand(expr):
        return _mexpand(a)
    expr = _nthroot_solve(expr, n, prec)
    if expr is None:
        return p
    return expr


def nsimplify(expr, constants=(), tolerance=None, full=False, rational=None,
    rational_conversion='base10'):
    """
    Find a simple representation for a number or, if there are free symbols or
    if ``rational=True``, then replace Floats with their Rational equivalents. If
    no change is made and rational is not False then Floats will at least be
    converted to Rationals.

    Explanation
    ===========

    For numerical expressions, a simple formula that numerically matches the
    given numerical expression is sought (and the input should be possible
    to evalf to a precision of at least 30 digits).

    Optionally, a list of (rationally independent) constants to
    include in the formula may be given.

    A lower tolerance may be set to find less exact matches. If no tolerance
    is given then the least precise value will set the tolerance (e.g. Floats
    default to 15 digits of precision, so would be tolerance=10**-15).

    With ``full=True``, a more extensive search is performed
    (this is useful to find simpler numbers when the tolerance
    is set low).

    When converting to rational, if rational_conversion='base10' (the default), then
    convert floats to rationals using their base-10 (string) representation.
    When rational_conversion='exact' it uses the exact, base-2 representation.

    Examples
    ========

    >>> from sympy import nsimplify, sqrt, GoldenRatio, exp, I, pi
    >>> nsimplify(4/(1+sqrt(5)), [GoldenRatio])
    -2 + 2*GoldenRatio
    >>> nsimplify((1/(exp(3*pi*I/5)+1)))
    1/2 - I*sqrt(sqrt(5)/10 + 1/4)
    >>> nsimplify(I**I, [pi])
    exp(-pi/2)
    >>> nsimplify(pi, tolerance=0.01)
    22/7

    >>> nsimplify(0.333333333333333, rational=True, rational_conversion='exact')
    6004799503160655/18014398509481984
    >>> nsimplify(0.333333333333333, rational=True)
    1/3

    See Also
    ========

    sympy.core.function.nfloat

    """
    try:
        return sympify(as_int(expr))
    except (TypeError, ValueError):
        pass
    expr = sympify(expr).xreplace({
        Float('inf'): S.Infinity,
        Float('-inf'): S.NegativeInfinity,
        })
    if expr is S.Infinity or expr is S.NegativeInfinity:
        return expr
    if rational or expr.free_symbols:
        return _real_to_rational(expr, tolerance, rational_conversion)

    # SymPy's default tolerance for Rationals is 15; other numbers may have
    # lower tolerances set, so use them to pick the largest tolerance if None
    # was given
    if tolerance is None:
        tolerance = 10**-min([15] +
             [mpmath.libmp.libmpf.prec_to_dps(n._prec)
             for n in expr.atoms(Float)])
    # XXX should prec be set independent of tolerance or should it be computed
    # from tolerance?
    prec = 30
    bprec = int(prec*3.33)

    constants_dict = {}
    for constant in constants:
        constant = sympify(constant)
        v = constant.evalf(prec)
        if not v.is_Float:
            raise ValueError("constants must be real-valued")
        constants_dict[str(constant)] = v._to_mpmath(bprec)

    exprval = expr.evalf(prec, chop=True)
    re, im = exprval.as_real_imag()

    # safety check to make sure that this evaluated to a number
    if not (re.is_Number and im.is_Number):
        return expr

    def nsimplify_real(x):
        orig = mpmath.mp.dps
        xv = x._to_mpmath(bprec)
        try:
            # We'll be happy with low precision if a simple fraction
            if not (tolerance or full):
                mpmath.mp.dps = 15
                rat = mpmath.pslq([xv, 1])
                if rat is not None:
                    return Rational(-int(rat[1]), int(rat[0]))
            mpmath.mp.dps = prec
            newexpr = mpmath.identify(xv, constants=constants_dict,
                tol=tolerance, full=full)
            if not newexpr:
                raise ValueError
            if full:
                newexpr = newexpr[0]
            expr = sympify(newexpr)
            if x and not expr:  # don't let x become 0
                raise ValueError
            if expr.is_finite is False and xv not in [mpmath.inf, mpmath.ninf]:
                raise ValueError
            return expr
        finally:
            # even though there are returns above, this is executed
            # before leaving
            mpmath.mp.dps = orig
    try:
        if re:
            re = nsimplify_real(re)
        if im:
            im = nsimplify_real(im)
    except ValueError:
        if rational is None:
            return _real_to_rational(expr, rational_conversion=rational_conversion)
        return expr

    rv = re + im*S.ImaginaryUnit
    # if there was a change or rational is explicitly not wanted
    # return the value, else return the Rational representation
    if rv != expr or rational is False:
        return rv
    return _real_to_rational(expr, rational_conversion=rational_conversion)


def _real_to_rational(expr, tolerance=None, rational_conversion='base10'):
    """
    Replace all reals in expr with rationals.

    Examples
    ========

    >>> from sympy.simplify.simplify import _real_to_rational
    >>> from sympy.abc import x

    >>> _real_to_rational(.76 + .1*x**.5)
    sqrt(x)/10 + 19/25

    If rational_conversion='base10', this uses the base-10 string. If
    rational_conversion='exact', the exact, base-2 representation is used.

    >>> _real_to_rational(0.333333333333333, rational_conversion='exact')
    6004799503160655/18014398509481984
    >>> _real_to_rational(0.333333333333333)
    1/3

    """
    expr = _sympify(expr)
    inf = Float('inf')
    p = expr
    reps = {}
    reduce_num = None
    if tolerance is not None and tolerance < 1:
        reduce_num = ceiling(1/tolerance)
    for fl in p.atoms(Float):
        key = fl
        if reduce_num is not None:
            r = Rational(fl).limit_denominator(reduce_num)
        elif (tolerance is not None and tolerance >= 1 and
                fl.is_Integer is False):
            r = Rational(tolerance*round(fl/tolerance)
                ).limit_denominator(int(tolerance))
        else:
            if rational_conversion == 'exact':
                r = Rational(fl)
                reps[key] = r
                continue
            elif rational_conversion != 'base10':
                raise ValueError("rational_conversion must be 'base10' or 'exact'")

            r = nsimplify(fl, rational=False)
            # e.g. log(3).n() -> log(3) instead of a Rational
            if fl and not r:
                r = Rational(fl)
            elif not r.is_Rational:
                if fl in (inf, -inf):
                    r = S.ComplexInfinity
                elif fl < 0:
                    fl = -fl
                    d = Pow(10, int(mpmath.log(fl)/mpmath.log(10)))
                    r = -Rational(str(fl/d))*d
                elif fl > 0:
                    d = Pow(10, int(mpmath.log(fl)/mpmath.log(10)))
                    r = Rational(str(fl/d))*d
                else:
                    r = S.Zero
        reps[key] = r
    return p.subs(reps, simultaneous=True)


def clear_coefficients(expr, rhs=S.Zero):
    """Return `p, r` where `p` is the expression obtained when Rational
    additive and multiplicative coefficients of `expr` have been stripped
    away in a naive fashion (i.e. without simplification). The operations
    needed to remove the coefficients will be applied to `rhs` and returned
    as `r`.

    Examples
    ========

    >>> from sympy.simplify.simplify import clear_coefficients
    >>> from sympy.abc import x, y
    >>> from sympy import Dummy
    >>> expr = 4*y*(6*x + 3)
    >>> clear_coefficients(expr - 2)
    (y*(2*x + 1), 1/6)

    When solving 2 or more expressions like `expr = a`,
    `expr = b`, etc..., it is advantageous to provide a Dummy symbol
    for `rhs` and  simply replace it with `a`, `b`, etc... in `r`.

    >>> rhs = Dummy('rhs')
    >>> clear_coefficients(expr, rhs)
    (y*(2*x + 1), _rhs/12)
    >>> _[1].subs(rhs, 2)
    1/6
    """
    was = None
    free = expr.free_symbols
    if expr.is_Rational:
        return (S.Zero, rhs - expr)
    while expr and was != expr:
        was = expr
        m, expr = (
            expr.as_content_primitive()
            if free else
            factor_terms(expr).as_coeff_Mul(rational=True))
        rhs /= m
        c, expr = expr.as_coeff_Add(rational=True)
        rhs -= c
    expr = signsimp(expr, evaluate = False)
    if expr.could_extract_minus_sign():
        expr = -expr
        rhs = -rhs
    return expr, rhs

def nc_simplify(expr, deep=True):
    '''
    Simplify a non-commutative expression composed of multiplication
    and raising to a power by grouping repeated subterms into one power.
    Priority is given to simplifications that give the fewest number
    of arguments in the end (for example, in a*b*a*b*c*a*b*c simplifying
    to (a*b)**2*c*a*b*c gives 5 arguments while a*b*(a*b*c)**2 has 3).
    If ``expr`` is a sum of such terms, the sum of the simplified terms
    is returned.

    Keyword argument ``deep`` controls whether or not subexpressions
    nested deeper inside the main expression are simplified. See examples
    below. Setting `deep` to `False` can save time on nested expressions
    that do not need simplifying on all levels.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.simplify.simplify import nc_simplify
    >>> a, b, c = symbols("a b c", commutative=False)
    >>> nc_simplify(a*b*a*b*c*a*b*c)
    a*b*(a*b*c)**2
    >>> expr = a**2*b*a**4*b*a**4
    >>> nc_simplify(expr)
    a**2*(b*a**4)**2
    >>> nc_simplify(a*b*a*b*c**2*(a*b)**2*c**2)
    ((a*b)**2*c**2)**2
    >>> nc_simplify(a*b*a*b + 2*a*c*a**2*c*a**2*c*a)
    (a*b)**2 + 2*(a*c*a)**3
    >>> nc_simplify(b**-1*a**-1*(a*b)**2)
    a*b
    >>> nc_simplify(a**-1*b**-1*c*a)
    (b*a)**(-1)*c*a
    >>> expr = (a*b*a*b)**2*a*c*a*c
    >>> nc_simplify(expr)
    (a*b)**4*(a*c)**2
    >>> nc_simplify(expr, deep=False)
    (a*b*a*b)**2*(a*c)**2

    '''
    if isinstance(expr, MatrixExpr):
        expr = expr.doit(inv_expand=False)
        _Add, _Mul, _Pow, _Symbol = MatAdd, MatMul, MatPow, MatrixSymbol
    else:
        _Add, _Mul, _Pow, _Symbol = Add, Mul, Pow, Symbol

    # =========== Auxiliary functions ========================
    def _overlaps(args):
        # Calculate a list of lists m such that m[i][j] contains the lengths
        # of all possible overlaps between args[:i+1] and args[i+1+j:].
        # An overlap is a suffix of the prefix that matches a prefix
        # of the suffix.
        # For example, let expr=c*a*b*a*b*a*b*a*b. Then m[3][0] contains
        # the lengths of overlaps of c*a*b*a*b with a*b*a*b. The overlaps
        # are a*b*a*b, a*b and the empty word so that m[3][0]=[4,2,0].
        # All overlaps rather than only the longest one are recorded
        # because this information helps calculate other overlap lengths.
        m = [[([1, 0] if a == args[0] else [0]) for a in args[1:]]]
        for i in range(1, len(args)):
            overlaps = []
            j = 0
            for j in range(len(args) - i - 1):
                overlap = []
                for v in m[i-1][j+1]:
                    if j + i + 1 + v < len(args) and args[i] == args[j+i+1+v]:
                        overlap.append(v + 1)
                overlap += [0]
                overlaps.append(overlap)
            m.append(overlaps)
        return m

    def _reduce_inverses(_args):
        # replace consecutive negative powers by an inverse
        # of a product of positive powers, e.g. a**-1*b**-1*c
        # will simplify to (a*b)**-1*c;
        # return that new args list and the number of negative
        # powers in it (inv_tot)
        inv_tot = 0 # total number of inverses
        inverses = []
        args = []
        for arg in _args:
            if isinstance(arg, _Pow) and arg.args[1].is_extended_negative:
                inverses = [arg**-1] + inverses
                inv_tot += 1
            else:
                if len(inverses) == 1:
                    args.append(inverses[0]**-1)
                elif len(inverses) > 1:
                    args.append(_Pow(_Mul(*inverses), -1))
                    inv_tot -= len(inverses) - 1
                inverses = []
                args.append(arg)
        if inverses:
            args.append(_Pow(_Mul(*inverses), -1))
            inv_tot -= len(inverses) - 1
        return inv_tot, tuple(args)

    def get_score(s):
        # compute the number of arguments of s
        # (including in nested expressions) overall
        # but ignore exponents
        if isinstance(s, _Pow):
            return get_score(s.args[0])
        elif isinstance(s, (_Add, _Mul)):
            return sum(get_score(a) for a in s.args)
        return 1

    def compare(s, alt_s):
        # compare two possible simplifications and return a
        # "better" one
        if s != alt_s and get_score(alt_s) < get_score(s):
            return alt_s
        return s
    # ========================================================

    if not isinstance(expr, (_Add, _Mul, _Pow)) or expr.is_commutative:
        return expr
    args = expr.args[:]
    if isinstance(expr, _Pow):
        if deep:
            return _Pow(nc_simplify(args[0]), args[1]).doit()
        else:
            return expr
    elif isinstance(expr, _Add):
        return _Add(*[nc_simplify(a, deep=deep) for a in args]).doit()
    else:
        # get the non-commutative part
        c_args, args = expr.args_cnc()
        com_coeff = Mul(*c_args)
        if not equal_valued(com_coeff, 1):
            return com_coeff*nc_simplify(expr/com_coeff, deep=deep)

    inv_tot, args = _reduce_inverses(args)
    # if most arguments are negative, work with the inverse
    # of the expression, e.g. a**-1*b*a**-1*c**-1 will become
    # (c*a*b**-1*a)**-1 at the end so can work with c*a*b**-1*a
    invert = False
    if inv_tot > len(args)/2:
        invert = True
        args = [a**-1 for a in args[::-1]]

    if deep:
        args = tuple(nc_simplify(a) for a in args)

    m = _overlaps(args)

    # simps will be {subterm: end} where `end` is the ending
    # index of a sequence of repetitions of subterm;
    # this is for not wasting time with subterms that are part
    # of longer, already considered sequences
    simps = {}

    post = 1
    pre = 1

    # the simplification coefficient is the number of
    # arguments by which contracting a given sequence
    # would reduce the word; e.g. in a*b*a*b*c*a*b*c,
    # contracting a*b*a*b to (a*b)**2 removes 3 arguments
    # while a*b*c*a*b*c to (a*b*c)**2 removes 6. It's
    # better to contract the latter so simplification
    # with a maximum simplification coefficient will be chosen
    max_simp_coeff = 0
    simp = None # information about future simplification

    for i in range(1, len(args)):
        simp_coeff = 0
        l = 0 # length of a subterm
        p = 0 # the power of a subterm
        if i < len(args) - 1:
            rep = m[i][0]
        start = i # starting index of the repeated sequence
        end = i+1 # ending index of the repeated sequence
        if i == len(args)-1 or rep == [0]:
            # no subterm is repeated at this stage, at least as
            # far as the arguments are concerned - there may be
            # a repetition if powers are taken into account
            if (isinstance(args[i], _Pow) and
                            not isinstance(args[i].args[0], _Symbol)):
                subterm = args[i].args[0].args
                l = len(subterm)
                if args[i-l:i] == subterm:
                    # e.g. a*b in a*b*(a*b)**2 is not repeated
                    # in args (= [a, b, (a*b)**2]) but it
                    # can be matched here
                    p += 1
                    start -= l
                if args[i+1:i+1+l] == subterm:
                    # e.g. a*b in (a*b)**2*a*b
                    p += 1
                    end += l
            if p:
                p += args[i].args[1]
            else:
                continue
        else:
            l = rep[0] # length of the longest repeated subterm at this point
            start -= l - 1
            subterm = args[start:end]
            p = 2
            end += l

        if subterm in simps and simps[subterm] >= start:
            # the subterm is part of a sequence that
            # has already been considered
            continue

        # count how many times it's repeated
        while end < len(args):
            if l in m[end-1][0]:
                p += 1
                end += l
            elif isinstance(args[end], _Pow) and args[end].args[0].args == subterm:
                # for cases like a*b*a*b*(a*b)**2*a*b
                p += args[end].args[1]
                end += 1
            else:
                break

        # see if another match can be made, e.g.
        # for b*a**2 in b*a**2*b*a**3 or a*b in
        # a**2*b*a*b

        pre_exp = 0
        pre_arg = 1
        if start - l >= 0 and args[start-l+1:start] == subterm[1:]:
            if isinstance(subterm[0], _Pow):
                pre_arg = subterm[0].args[0]
                exp = subterm[0].args[1]
            else:
                pre_arg = subterm[0]
                exp = 1
            if isinstance(args[start-l], _Pow) and args[start-l].args[0] == pre_arg:
                pre_exp = args[start-l].args[1] - exp
                start -= l
                p += 1
            elif args[start-l] == pre_arg:
                pre_exp = 1 - exp
                start -= l
                p += 1

        post_exp = 0
        post_arg = 1
        if end + l - 1 < len(args) and args[end:end+l-1] == subterm[:-1]:
            if isinstance(subterm[-1], _Pow):
                post_arg = subterm[-1].args[0]
                exp = subterm[-1].args[1]
            else:
                post_arg = subterm[-1]
                exp = 1
            if isinstance(args[end+l-1], _Pow) and args[end+l-1].args[0] == post_arg:
                post_exp = args[end+l-1].args[1] - exp
                end += l
                p += 1
            elif args[end+l-1] == post_arg:
                post_exp = 1 - exp
                end += l
                p += 1

        # Consider a*b*a**2*b*a**2*b*a:
        # b*a**2 is explicitly repeated, but note
        # that in this case a*b*a is also repeated
        # so there are two possible simplifications:
        # a*(b*a**2)**3*a**-1 or (a*b*a)**3
        # The latter is obviously simpler.
        # But in a*b*a**2*b**2*a**2 the simplifications are
        # a*(b*a**2)**2 and (a*b*a)**3*a in which case
        # it's better to stick with the shorter subterm
        if post_exp and exp % 2 == 0 and start > 0:
            exp = exp/2
            _pre_exp = 1
            _post_exp = 1
            if isinstance(args[start-1], _Pow) and args[start-1].args[0] == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = args[start-1].args[1] - exp
            elif args[start-1] == post_arg:
                _post_exp = post_exp + exp
                _pre_exp = 1 - exp
            if _pre_exp == 0 or _post_exp == 0:
                if not pre_exp:
                    start -= 1
                post_exp = _post_exp
                pre_exp = _pre_exp
                pre_arg = post_arg
                subterm = (post_arg**exp,) + subterm[:-1] + (post_arg**exp,)

        simp_coeff += end-start

        if post_exp:
            simp_coeff -= 1
        if pre_exp:
            simp_coeff -= 1

        simps[subterm] = end

        if simp_coeff > max_simp_coeff:
            max_simp_coeff = simp_coeff
            simp = (start, _Mul(*subterm), p, end, l)
            pre = pre_arg**pre_exp
            post = post_arg**post_exp

    if simp:
        subterm = _Pow(nc_simplify(simp[1], deep=deep), simp[2])
        pre = nc_simplify(_Mul(*args[:simp[0]])*pre, deep=deep)
        post = post*nc_simplify(_Mul(*args[simp[3]:]), deep=deep)
        simp = pre*subterm*post
        if pre != 1 or post != 1:
            # new simplifications may be possible but no need
            # to recurse over arguments
            simp = nc_simplify(simp, deep=False)
    else:
        simp = _Mul(*args)

    if invert:
        simp = _Pow(simp, -1)

    # see if factor_nc(expr) is simplified better
    if not isinstance(expr, MatrixExpr):
        f_expr = factor_nc(expr)
        if f_expr != expr:
            alt_simp = nc_simplify(f_expr, deep=deep)
            simp = compare(simp, alt_simp)
    else:
        simp = simp.doit(inv_expand=False)
    return simp


def dotprodsimp(expr, withsimp=False):
    """Simplification for a sum of products targeted at the kind of blowup that
    occurs during summation of products. Intended to reduce expression blowup
    during matrix multiplication or other similar operations. Only works with
    algebraic expressions and does not recurse into non.

    Parameters
    ==========

    withsimp : bool, optional
        Specifies whether a flag should be returned along with the expression
        to indicate roughly whether simplification was successful. It is used
        in ``MatrixArithmetic._eval_pow_by_recursion`` to avoid attempting to
        simplify an expression repetitively which does not simplify.
    """

    def count_ops_alg(expr):
        """Optimized count algebraic operations with no recursion into
        non-algebraic args that ``core.function.count_ops`` does. Also returns
        whether rational functions may be present according to negative
        exponents of powers or non-number fractions.

        Returns
        =======

        ops, ratfunc : int, bool
            ``ops`` is the number of algebraic operations starting at the top
            level expression (not recursing into non-alg children). ``ratfunc``
            specifies whether the expression MAY contain rational functions
            which ``cancel`` MIGHT optimize.
        """

        ops     = 0
        args    = [expr]
        ratfunc = False

        while args:
            a = args.pop()

            if not isinstance(a, Basic):
                continue

            if a.is_Rational:
                if a is not S.One: # -1/3 = NEG + DIV
                    ops += bool (a.p < 0) + bool (a.q != 1)

            elif a.is_Mul:
                if a.could_extract_minus_sign():
                    ops += 1
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a

                n, d = fraction(a)

                if n.is_Integer:
                    ops += 1 + bool (n < 0)
                    args.append(d) # won't be -Mul but could be Add

                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                        ratfunc=True

                    ops += 1
                    args.append(n) # could be -Mul

                else:
                    ops += len(a.args) - 1
                    args.extend(a.args)

            elif a.is_Add:
                laargs = len(a.args)
                negs   = 0

                for ai in a.args:
                    if ai.could_extract_minus_sign():
                        negs += 1
                        ai    = -ai
                    args.append(ai)

                ops += laargs - (negs != laargs) # -x - y = NEG + SUB

            elif a.is_Pow:
                ops += 1
                args.append(a.base)

                if not ratfunc:
                    ratfunc = a.exp.is_negative is not False

        return ops, ratfunc

    def nonalg_subs_dummies(expr, dummies):
        """Substitute dummy variables for non-algebraic expressions to avoid
        evaluation of non-algebraic terms that ``polys.polytools.cancel`` does.
        """

        if not expr.args:
            return expr

        if expr.is_Add or expr.is_Mul or expr.is_Pow:
            args = None

            for i, a in enumerate(expr.args):
                c = nonalg_subs_dummies(a, dummies)

                if c is a:
                    continue

                if args is None:
                    args = list(expr.args)

                args[i] = c

            if args is None:
                return expr

            return expr.func(*args)

        return dummies.setdefault(expr, Dummy())

    simplified = False # doesn't really mean simplified, rather "can simplify again"

    if isinstance(expr, Basic) and (expr.is_Add or expr.is_Mul or expr.is_Pow):
        expr2 = expr.expand(deep=True, modulus=None, power_base=False,
            power_exp=False, mul=True, log=False, multinomial=True, basic=False)

        if expr2 != expr:
            expr       = expr2
            simplified = True

        exprops, ratfunc = count_ops_alg(expr)

        if exprops >= 6: # empirically tested cutoff for expensive simplification
            if ratfunc:
                dummies = {}
                expr2   = nonalg_subs_dummies(expr, dummies)

                if expr2 is expr or count_ops_alg(expr2)[0] >= 6: # check again after substitution
                    expr3 = cancel(expr2)

                    if expr3 != expr2:
                        expr       = expr3.subs([(d, e) for e, d in dummies.items()])
                        simplified = True

        # very special case: x/(x-1) - 1/(x-1) -> 1
        elif (exprops == 5 and expr.is_Add and expr.args [0].is_Mul and
                expr.args [1].is_Mul and expr.args [0].args [-1].is_Pow and
                expr.args [1].args [-1].is_Pow and
                expr.args [0].args [-1].exp is S.NegativeOne and
                expr.args [1].args [-1].exp is S.NegativeOne):

            expr2    = together (expr)
            expr2ops = count_ops_alg(expr2)[0]

            if expr2ops < exprops:
                expr       = expr2
                simplified = True

        else:
            simplified = True

    return (expr, simplified) if withsimp else expr


bottom_up = deprecated(
    """
    Using bottom_up from the sympy.simplify.simplify submodule is
    deprecated.

    Instead, use bottom_up from the top-level sympy namespace, like

        sympy.bottom_up
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved",
)(_bottom_up)


# XXX: This function really should either be private API or exported in the
# top-level sympy/__init__.py
walk = deprecated(
    """
    Using walk from the sympy.simplify.simplify submodule is
    deprecated.

    Instead, use walk from sympy.core.traversal.walk
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved",
)(_walk)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\scoping.py ===
# orm/scoping.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .session import _S
from .session import Session
from .. import exc as sa_exc
from .. import util
from ..util import create_proxy_methods
from ..util import ScopedRegistry
from ..util import ThreadLocalRegistry
from ..util import warn
from ..util import warn_deprecated
from ..util.typing import Protocol

if TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _IdentityKeyType
    from ._typing import OrmExecuteOptionsParameter
    from .identity import IdentityMap
    from .interfaces import ORMOption
    from .mapper import Mapper
    from .query import Query
    from .query import RowReturningQuery
    from .session import _BindArguments
    from .session import _EntityBindKey
    from .session import _PKIdentityArgument
    from .session import _SessionBind
    from .session import sessionmaker
    from .session import SessionTransaction
    from ..engine import Connection
    from ..engine import CursorResult
    from ..engine import Engine
    from ..engine import Result
    from ..engine import Row
    from ..engine import RowMapping
    from ..engine.interfaces import _CoreAnyExecuteParams
    from ..engine.interfaces import _CoreSingleExecuteParams
    from ..engine.interfaces import CoreExecuteOptionsParameter
    from ..engine.result import ScalarResult
    from ..sql._typing import _ColumnsClauseArgument
    from ..sql._typing import _T0
    from ..sql._typing import _T1
    from ..sql._typing import _T2
    from ..sql._typing import _T3
    from ..sql._typing import _T4
    from ..sql._typing import _T5
    from ..sql._typing import _T6
    from ..sql._typing import _T7
    from ..sql._typing import _TypedColumnClauseArgument as _TCCA
    from ..sql.base import Executable
    from ..sql.dml import UpdateBase
    from ..sql.elements import ClauseElement
    from ..sql.roles import TypedColumnsClauseRole
    from ..sql.selectable import ForUpdateParameter
    from ..sql.selectable import TypedReturnsRows

_T = TypeVar("_T", bound=Any)


class QueryPropertyDescriptor(Protocol):
    """Describes the type applied to a class-level
    :meth:`_orm.scoped_session.query_property` attribute.

    .. versionadded:: 2.0.5

    """

    def __get__(self, instance: Any, owner: Type[_T]) -> Query[_T]: ...


_O = TypeVar("_O", bound=object)

__all__ = ["scoped_session"]


@create_proxy_methods(
    Session,
    ":class:`_orm.Session`",
    ":class:`_orm.scoping.scoped_session`",
    classmethods=["close_all", "object_session", "identity_key"],
    methods=[
        "__contains__",
        "__iter__",
        "add",
        "add_all",
        "begin",
        "begin_nested",
        "close",
        "reset",
        "commit",
        "connection",
        "delete",
        "execute",
        "expire",
        "expire_all",
        "expunge",
        "expunge_all",
        "flush",
        "get",
        "get_one",
        "get_bind",
        "is_modified",
        "bulk_save_objects",
        "bulk_insert_mappings",
        "bulk_update_mappings",
        "merge",
        "query",
        "refresh",
        "rollback",
        "scalar",
        "scalars",
    ],
    attributes=[
        "bind",
        "dirty",
        "deleted",
        "new",
        "identity_map",
        "is_active",
        "autoflush",
        "no_autoflush",
        "info",
    ],
)
class scoped_session(Generic[_S]):
    """Provides scoped management of :class:`.Session` objects.

    See :ref:`unitofwork_contextual` for a tutorial.

    .. note::

       When using :ref:`asyncio_toplevel`, the async-compatible
       :class:`_asyncio.async_scoped_session` class should be
       used in place of :class:`.scoped_session`.

    """

    _support_async: bool = False

    session_factory: sessionmaker[_S]
    """The `session_factory` provided to `__init__` is stored in this
    attribute and may be accessed at a later time.  This can be useful when
    a new non-scoped :class:`.Session` is needed."""

    registry: ScopedRegistry[_S]

    def __init__(
        self,
        session_factory: sessionmaker[_S],
        scopefunc: Optional[Callable[[], Any]] = None,
    ):
        """Construct a new :class:`.scoped_session`.

        :param session_factory: a factory to create new :class:`.Session`
         instances. This is usually, but not necessarily, an instance
         of :class:`.sessionmaker`.
        :param scopefunc: optional function which defines
         the current scope.   If not passed, the :class:`.scoped_session`
         object assumes "thread-local" scope, and will use
         a Python ``threading.local()`` in order to maintain the current
         :class:`.Session`.  If passed, the function should return
         a hashable token; this token will be used as the key in a
         dictionary in order to store and retrieve the current
         :class:`.Session`.

        """
        self.session_factory = session_factory

        if scopefunc:
            self.registry = ScopedRegistry(session_factory, scopefunc)
        else:
            self.registry = ThreadLocalRegistry(session_factory)

    @property
    def _proxied(self) -> _S:
        return self.registry()

    def __call__(self, **kw: Any) -> _S:
        r"""Return the current :class:`.Session`, creating it
        using the :attr:`.scoped_session.session_factory` if not present.

        :param \**kw: Keyword arguments will be passed to the
         :attr:`.scoped_session.session_factory` callable, if an existing
         :class:`.Session` is not present.  If the :class:`.Session` is present
         and keyword arguments have been passed,
         :exc:`~sqlalchemy.exc.InvalidRequestError` is raised.

        """
        if kw:
            if self.registry.has():
                raise sa_exc.InvalidRequestError(
                    "Scoped session is already present; "
                    "no new arguments may be specified."
                )
            else:
                sess = self.session_factory(**kw)
                self.registry.set(sess)
        else:
            sess = self.registry()
        if not self._support_async and sess._is_asyncio:
            warn_deprecated(
                "Using `scoped_session` with asyncio is deprecated and "
                "will raise an error in a future version. "
                "Please use `async_scoped_session` instead.",
                "1.4.23",
            )
        return sess

    def configure(self, **kwargs: Any) -> None:
        """reconfigure the :class:`.sessionmaker` used by this
        :class:`.scoped_session`.

        See :meth:`.sessionmaker.configure`.

        """

        if self.registry.has():
            warn(
                "At least one scoped session is already present. "
                " configure() can not affect sessions that have "
                "already been created."
            )

        self.session_factory.configure(**kwargs)

    def remove(self) -> None:
        """Dispose of the current :class:`.Session`, if present.

        This will first call :meth:`.Session.close` method
        on the current :class:`.Session`, which releases any existing
        transactional/connection resources still being held; transactions
        specifically are rolled back.  The :class:`.Session` is then
        discarded.   Upon next usage within the same scope,
        the :class:`.scoped_session` will produce a new
        :class:`.Session` object.

        """

        if self.registry.has():
            self.registry().close()
        self.registry.clear()

    def query_property(
        self, query_cls: Optional[Type[Query[_T]]] = None
    ) -> QueryPropertyDescriptor:
        """return a class property which produces a legacy
        :class:`_query.Query` object against the class and the current
        :class:`.Session` when called.

        .. legacy:: The :meth:`_orm.scoped_session.query_property` accessor
           is specific to the legacy :class:`.Query` object and is not
           considered to be part of :term:`2.0-style` ORM use.

        e.g.::

            from sqlalchemy.orm import QueryPropertyDescriptor
            from sqlalchemy.orm import scoped_session
            from sqlalchemy.orm import sessionmaker

            Session = scoped_session(sessionmaker())


            class MyClass:
                query: QueryPropertyDescriptor = Session.query_property()


            # after mappers are defined
            result = MyClass.query.filter(MyClass.name == "foo").all()

        Produces instances of the session's configured query class by
        default.  To override and use a custom implementation, provide
        a ``query_cls`` callable.  The callable will be invoked with
        the class's mapper as a positional argument and a session
        keyword argument.

        There is no limit to the number of query properties placed on
        a class.

        """

        class query:
            def __get__(s, instance: Any, owner: Type[_O]) -> Query[_O]:
                if query_cls:
                    # custom query class
                    return query_cls(owner, session=self.registry())  # type: ignore  # noqa: E501
                else:
                    # session's configured query class
                    return self.registry().query(owner)

        return query()

    # START PROXY METHODS scoped_session

    # code within this block is **programmatically,
    # statically generated** by tools/generate_proxy_methods.py

    def __contains__(self, instance: object) -> bool:
        r"""Return True if the instance is associated with this session.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        The instance may be pending or persistent within the Session for a
        result of True.


        """  # noqa: E501

        return self._proxied.__contains__(instance)

    def __iter__(self) -> Iterator[object]:
        r"""Iterate over all pending or persistent instances within this
        Session.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.


        """  # noqa: E501

        return self._proxied.__iter__()

    def add(self, instance: object, _warn: bool = True) -> None:
        r"""Place an object into this :class:`_orm.Session`.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Objects that are in the :term:`transient` state when passed to the
        :meth:`_orm.Session.add` method will move to the
        :term:`pending` state, until the next flush, at which point they
        will move to the :term:`persistent` state.

        Objects that are in the :term:`detached` state when passed to the
        :meth:`_orm.Session.add` method will move to the :term:`persistent`
        state directly.

        If the transaction used by the :class:`_orm.Session` is rolled back,
        objects which were transient when they were passed to
        :meth:`_orm.Session.add` will be moved back to the
        :term:`transient` state, and will no longer be present within this
        :class:`_orm.Session`.

        .. seealso::

            :meth:`_orm.Session.add_all`

            :ref:`session_adding` - at :ref:`session_basics`


        """  # noqa: E501

        return self._proxied.add(instance, _warn=_warn)

    def add_all(self, instances: Iterable[object]) -> None:
        r"""Add the given collection of instances to this :class:`_orm.Session`.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        See the documentation for :meth:`_orm.Session.add` for a general
        behavioral description.

        .. seealso::

            :meth:`_orm.Session.add`

            :ref:`session_adding` - at :ref:`session_basics`


        """  # noqa: E501

        return self._proxied.add_all(instances)

    def begin(self, nested: bool = False) -> SessionTransaction:
        r"""Begin a transaction, or nested transaction,
        on this :class:`.Session`, if one is not already begun.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        The :class:`_orm.Session` object features **autobegin** behavior,
        so that normally it is not necessary to call the
        :meth:`_orm.Session.begin`
        method explicitly. However, it may be used in order to control
        the scope of when the transactional state is begun.

        When used to begin the outermost transaction, an error is raised
        if this :class:`.Session` is already inside of a transaction.

        :param nested: if True, begins a SAVEPOINT transaction and is
         equivalent to calling :meth:`~.Session.begin_nested`. For
         documentation on SAVEPOINT transactions, please see
         :ref:`session_begin_nested`.

        :return: the :class:`.SessionTransaction` object.  Note that
         :class:`.SessionTransaction`
         acts as a Python context manager, allowing :meth:`.Session.begin`
         to be used in a "with" block.  See :ref:`session_explicit_begin` for
         an example.

        .. seealso::

            :ref:`session_autobegin`

            :ref:`unitofwork_transaction`

            :meth:`.Session.begin_nested`



        """  # noqa: E501

        return self._proxied.begin(nested=nested)

    def begin_nested(self) -> SessionTransaction:
        r"""Begin a "nested" transaction on this Session, e.g. SAVEPOINT.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        The target database(s) and associated drivers must support SQL
        SAVEPOINT for this method to function correctly.

        For documentation on SAVEPOINT
        transactions, please see :ref:`session_begin_nested`.

        :return: the :class:`.SessionTransaction` object.  Note that
         :class:`.SessionTransaction` acts as a context manager, allowing
         :meth:`.Session.begin_nested` to be used in a "with" block.
         See :ref:`session_begin_nested` for a usage example.

        .. seealso::

            :ref:`session_begin_nested`

            :ref:`pysqlite_serializable` - special workarounds required
            with the SQLite driver in order for SAVEPOINT to work
            correctly. For asyncio use cases, see the section
            :ref:`aiosqlite_serializable`.


        """  # noqa: E501

        return self._proxied.begin_nested()

    def close(self) -> None:
        r"""Close out the transactional resources and ORM objects used by this
        :class:`_orm.Session`.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        This expunges all ORM objects associated with this
        :class:`_orm.Session`, ends any transaction in progress and
        :term:`releases` any :class:`_engine.Connection` objects which this
        :class:`_orm.Session` itself has checked out from associated
        :class:`_engine.Engine` objects. The operation then leaves the
        :class:`_orm.Session` in a state which it may be used again.

        .. tip::

            In the default running mode the :meth:`_orm.Session.close`
            method **does not prevent the Session from being used again**.
            The :class:`_orm.Session` itself does not actually have a
            distinct "closed" state; it merely means
            the :class:`_orm.Session` will release all database connections
            and ORM objects.

            Setting the parameter :paramref:`_orm.Session.close_resets_only`
            to ``False`` will instead make the ``close`` final, meaning that
            any further action on the session will be forbidden.

        .. versionchanged:: 1.4  The :meth:`.Session.close` method does not
           immediately create a new :class:`.SessionTransaction` object;
           instead, the new :class:`.SessionTransaction` is created only if
           the :class:`.Session` is used again for a database operation.

        .. seealso::

            :ref:`session_closing` - detail on the semantics of
            :meth:`_orm.Session.close` and :meth:`_orm.Session.reset`.

            :meth:`_orm.Session.reset` - a similar method that behaves like
            ``close()`` with  the parameter
            :paramref:`_orm.Session.close_resets_only` set to ``True``.


        """  # noqa: E501

        return self._proxied.close()

    def reset(self) -> None:
        r"""Close out the transactional resources and ORM objects used by this
        :class:`_orm.Session`, resetting the session to its initial state.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        This method provides for same "reset-only" behavior that the
        :meth:`_orm.Session.close` method has provided historically, where the
        state of the :class:`_orm.Session` is reset as though the object were
        brand new, and ready to be used again.
        This method may then be useful for :class:`_orm.Session` objects
        which set :paramref:`_orm.Session.close_resets_only` to ``False``,
        so that "reset only" behavior is still available.

        .. versionadded:: 2.0.22

        .. seealso::

            :ref:`session_closing` - detail on the semantics of
            :meth:`_orm.Session.close` and :meth:`_orm.Session.reset`.

            :meth:`_orm.Session.close` - a similar method will additionally
            prevent re-use of the Session when the parameter
            :paramref:`_orm.Session.close_resets_only` is set to ``False``.

        """  # noqa: E501

        return self._proxied.reset()

    def commit(self) -> None:
        r"""Flush pending changes and commit the current transaction.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        When the COMMIT operation is complete, all objects are fully
        :term:`expired`, erasing their internal contents, which will be
        automatically re-loaded when the objects are next accessed. In the
        interim, these objects are in an expired state and will not function if
        they are :term:`detached` from the :class:`.Session`. Additionally,
        this re-load operation is not supported when using asyncio-oriented
        APIs. The :paramref:`.Session.expire_on_commit` parameter may be used
        to disable this behavior.

        When there is no transaction in place for the :class:`.Session`,
        indicating that no operations were invoked on this :class:`.Session`
        since the previous call to :meth:`.Session.commit`, the method will
        begin and commit an internal-only "logical" transaction, that does not
        normally affect the database unless pending flush changes were
        detected, but will still invoke event handlers and object expiration
        rules.

        The outermost database transaction is committed unconditionally,
        automatically releasing any SAVEPOINTs in effect.

        .. seealso::

            :ref:`session_committing`

            :ref:`unitofwork_transaction`

            :ref:`asyncio_orm_avoid_lazyloads`


        """  # noqa: E501

        return self._proxied.commit()

    def connection(
        self,
        bind_arguments: Optional[_BindArguments] = None,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> Connection:
        r"""Return a :class:`_engine.Connection` object corresponding to this
        :class:`.Session` object's transactional state.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Either the :class:`_engine.Connection` corresponding to the current
        transaction is returned, or if no transaction is in progress, a new
        one is begun and the :class:`_engine.Connection`
        returned (note that no
        transactional state is established with the DBAPI until the first
        SQL statement is emitted).

        Ambiguity in multi-bind or unbound :class:`.Session` objects can be
        resolved through any of the optional keyword arguments.   This
        ultimately makes usage of the :meth:`.get_bind` method for resolution.

        :param bind_arguments: dictionary of bind arguments.  May include
         "mapper", "bind", "clause", other custom arguments that are passed
         to :meth:`.Session.get_bind`.

        :param execution_options: a dictionary of execution options that will
         be passed to :meth:`_engine.Connection.execution_options`, **when the
         connection is first procured only**.   If the connection is already
         present within the :class:`.Session`, a warning is emitted and
         the arguments are ignored.

         .. seealso::

            :ref:`session_transaction_isolation`


        """  # noqa: E501

        return self._proxied.connection(
            bind_arguments=bind_arguments, execution_options=execution_options
        )

    def delete(self, instance: object) -> None:
        r"""Mark an instance as deleted.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        The object is assumed to be either :term:`persistent` or
        :term:`detached` when passed; after the method is called, the
        object will remain in the :term:`persistent` state until the next
        flush proceeds.  During this time, the object will also be a member
        of the :attr:`_orm.Session.deleted` collection.

        When the next flush proceeds, the object will move to the
        :term:`deleted` state, indicating a ``DELETE`` statement was emitted
        for its row within the current transaction.   When the transaction
        is successfully committed,
        the deleted object is moved to the :term:`detached` state and is
        no longer present within this :class:`_orm.Session`.

        .. seealso::

            :ref:`session_deleting` - at :ref:`session_basics`


        """  # noqa: E501

        return self._proxied.delete(instance)

    @overload
    def execute(
        self,
        statement: TypedReturnsRows[_T],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[_T]: ...

    @overload
    def execute(
        self,
        statement: UpdateBase,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> CursorResult[Any]: ...

    @overload
    def execute(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[Any]: ...

    def execute(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[Any]:
        r"""Execute a SQL expression construct.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Returns a :class:`_engine.Result` object representing
        results of the statement execution.

        E.g.::

            from sqlalchemy import select

            result = session.execute(select(User).where(User.id == 5))

        The API contract of :meth:`_orm.Session.execute` is similar to that
        of :meth:`_engine.Connection.execute`, the :term:`2.0 style` version
        of :class:`_engine.Connection`.

        .. versionchanged:: 1.4 the :meth:`_orm.Session.execute` method is
           now the primary point of ORM statement execution when using
           :term:`2.0 style` ORM usage.

        :param statement:
            An executable statement (i.e. an :class:`.Executable` expression
            such as :func:`_expression.select`).

        :param params:
            Optional dictionary, or list of dictionaries, containing
            bound parameter values.   If a single dictionary, single-row
            execution occurs; if a list of dictionaries, an
            "executemany" will be invoked.  The keys in each dictionary
            must correspond to parameter names present in the statement.

        :param execution_options: optional dictionary of execution options,
         which will be associated with the statement execution.  This
         dictionary can provide a subset of the options that are accepted
         by :meth:`_engine.Connection.execution_options`, and may also
         provide additional options understood only in an ORM context.

         .. seealso::

            :ref:`orm_queryguide_execution_options` - ORM-specific execution
            options

        :param bind_arguments: dictionary of additional arguments to determine
         the bind.  May include "mapper", "bind", or other custom arguments.
         Contents of this dictionary are passed to the
         :meth:`.Session.get_bind` method.

        :return: a :class:`_engine.Result` object.



        """  # noqa: E501

        return self._proxied.execute(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            _parent_execute_state=_parent_execute_state,
            _add_event=_add_event,
        )

    def expire(
        self, instance: object, attribute_names: Optional[Iterable[str]] = None
    ) -> None:
        r"""Expire the attributes on an instance.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Marks the attributes of an instance as out of date. When an expired
        attribute is next accessed, a query will be issued to the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire all objects in the :class:`.Session` simultaneously,
        use :meth:`Session.expire_all`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire` only makes sense for the specific
        case that a non-ORM SQL statement was emitted in the current
        transaction.

        :param instance: The instance to be refreshed.
        :param attribute_names: optional list of string attribute names
          indicating a subset of attributes to be expired.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`


        """  # noqa: E501

        return self._proxied.expire(instance, attribute_names=attribute_names)

    def expire_all(self) -> None:
        r"""Expires all persistent instances within this Session.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        When any attributes on a persistent instance is next accessed,
        a query will be issued using the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire individual objects and individual attributes
        on those objects, use :meth:`Session.expire`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire_all` is not usually needed,
        assuming the transaction is isolated.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`


        """  # noqa: E501

        return self._proxied.expire_all()

    def expunge(self, instance: object) -> None:
        r"""Remove the `instance` from this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        This will free all internal references to the instance.  Cascading
        will be applied according to the *expunge* cascade rule.


        """  # noqa: E501

        return self._proxied.expunge(instance)

    def expunge_all(self) -> None:
        r"""Remove all object instances from this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        This is equivalent to calling ``expunge(obj)`` on all objects in this
        ``Session``.


        """  # noqa: E501

        return self._proxied.expunge_all()

    def flush(self, objects: Optional[Sequence[Any]] = None) -> None:
        r"""Flush all the object changes to the database.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Writes out all pending object creations, deletions and modifications
        to the database as INSERTs, DELETEs, UPDATEs, etc.  Operations are
        automatically ordered by the Session's unit of work dependency
        solver.

        Database operations will be issued in the current transactional
        context and do not affect the state of the transaction, unless an
        error occurs, in which case the entire transaction is rolled back.
        You may flush() as often as you like within a transaction to move
        changes from Python to the database's transaction buffer.

        :param objects: Optional; restricts the flush operation to operate
          only on elements that are in the given collection.

          This feature is for an extremely narrow set of use cases where
          particular objects may need to be operated upon before the
          full flush() occurs.  It is not intended for general use.


        """  # noqa: E501

        return self._proxied.flush(objects=objects)

    def get(
        self,
        entity: _EntityBindKey[_O],
        ident: _PKIdentityArgument,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
    ) -> Optional[_O]:
        r"""Return an instance based on the given primary key identifier,
        or ``None`` if not found.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        E.g.::

            my_user = session.get(User, 5)

            some_object = session.get(VersionedFoo, (5, 10))

            some_object = session.get(VersionedFoo, {"id": 5, "version_id": 10})

        .. versionadded:: 1.4 Added :meth:`_orm.Session.get`, which is moved
           from the now legacy :meth:`_orm.Query.get` method.

        :meth:`_orm.Session.get` is special in that it provides direct
        access to the identity map of the :class:`.Session`.
        If the given primary key identifier is present
        in the local identity map, the object is returned
        directly from this collection and no SQL is emitted,
        unless the object has been marked fully expired.
        If not present,
        a SELECT is performed in order to locate the object.

        :meth:`_orm.Session.get` also will perform a check if
        the object is present in the identity map and
        marked as expired - a SELECT
        is emitted to refresh the object as well as to
        ensure that the row is still present.
        If not, :class:`~sqlalchemy.orm.exc.ObjectDeletedError` is raised.

        :param entity: a mapped class or :class:`.Mapper` indicating the
         type of entity to be loaded.

        :param ident: A scalar, tuple, or dictionary representing the
         primary key.  For a composite (e.g. multiple column) primary key,
         a tuple or dictionary should be passed.

         For a single-column primary key, the scalar calling form is typically
         the most expedient.  If the primary key of a row is the value "5",
         the call looks like::

            my_object = session.get(SomeClass, 5)

         The tuple form contains primary key values typically in
         the order in which they correspond to the mapped
         :class:`_schema.Table`
         object's primary key columns, or if the
         :paramref:`_orm.Mapper.primary_key` configuration parameter were
         used, in
         the order used for that parameter. For example, if the primary key
         of a row is represented by the integer
         digits "5, 10" the call would look like::

             my_object = session.get(SomeClass, (5, 10))

         The dictionary form should include as keys the mapped attribute names
         corresponding to each element of the primary key.  If the mapped class
         has the attributes ``id``, ``version_id`` as the attributes which
         store the object's primary key value, the call would look like::

            my_object = session.get(SomeClass, {"id": 5, "version_id": 10})

        :param options: optional sequence of loader options which will be
         applied to the query, if one is emitted.

        :param populate_existing: causes the method to unconditionally emit
         a SQL query and refresh the object with the newly loaded data,
         regardless of whether or not the object is already present.

        :param with_for_update: optional boolean ``True`` indicating FOR UPDATE
          should be used, or may be a dictionary containing flags to
          indicate a more specific set of FOR UPDATE flags for the SELECT;
          flags should match the parameters of
          :meth:`_query.Query.with_for_update`.
          Supersedes the :paramref:`.Session.refresh.lockmode` parameter.

        :param execution_options: optional dictionary of execution options,
         which will be associated with the query execution if one is emitted.
         This dictionary can provide a subset of the options that are
         accepted by :meth:`_engine.Connection.execution_options`, and may
         also provide additional options understood only in an ORM context.

         .. versionadded:: 1.4.29

         .. seealso::

            :ref:`orm_queryguide_execution_options` - ORM-specific execution
            options

        :param bind_arguments: dictionary of additional arguments to determine
         the bind.  May include "mapper", "bind", or other custom arguments.
         Contents of this dictionary are passed to the
         :meth:`.Session.get_bind` method.

         .. versionadded: 2.0.0rc1

        :return: The object instance, or ``None``.


        """  # noqa: E501

        return self._proxied.get(
            entity,
            ident,
            options=options,
            populate_existing=populate_existing,
            with_for_update=with_for_update,
            identity_token=identity_token,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
        )

    def get_one(
        self,
        entity: _EntityBindKey[_O],
        ident: _PKIdentityArgument,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
    ) -> _O:
        r"""Return exactly one instance based on the given primary key
        identifier, or raise an exception if not found.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Raises :class:`_exc.NoResultFound` if the query selects no rows.

        For a detailed documentation of the arguments see the
        method :meth:`.Session.get`.

        .. versionadded:: 2.0.22

        :return: The object instance.

        .. seealso::

            :meth:`.Session.get` - equivalent method that instead
              returns ``None`` if no row was found with the provided primary
              key


        """  # noqa: E501

        return self._proxied.get_one(
            entity,
            ident,
            options=options,
            populate_existing=populate_existing,
            with_for_update=with_for_update,
            identity_token=identity_token,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
        )

    def get_bind(
        self,
        mapper: Optional[_EntityBindKey[_O]] = None,
        *,
        clause: Optional[ClauseElement] = None,
        bind: Optional[_SessionBind] = None,
        _sa_skip_events: Optional[bool] = None,
        _sa_skip_for_implicit_returning: bool = False,
        **kw: Any,
    ) -> Union[Engine, Connection]:
        r"""Return a "bind" to which this :class:`.Session` is bound.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        The "bind" is usually an instance of :class:`_engine.Engine`,
        except in the case where the :class:`.Session` has been
        explicitly bound directly to a :class:`_engine.Connection`.

        For a multiply-bound or unbound :class:`.Session`, the
        ``mapper`` or ``clause`` arguments are used to determine the
        appropriate bind to return.

        Note that the "mapper" argument is usually present
        when :meth:`.Session.get_bind` is called via an ORM
        operation such as a :meth:`.Session.query`, each
        individual INSERT/UPDATE/DELETE operation within a
        :meth:`.Session.flush`, call, etc.

        The order of resolution is:

        1. if mapper given and :paramref:`.Session.binds` is present,
           locate a bind based first on the mapper in use, then
           on the mapped class in use, then on any base classes that are
           present in the ``__mro__`` of the mapped class, from more specific
           superclasses to more general.
        2. if clause given and ``Session.binds`` is present,
           locate a bind based on :class:`_schema.Table` objects
           found in the given clause present in ``Session.binds``.
        3. if ``Session.binds`` is present, return that.
        4. if clause given, attempt to return a bind
           linked to the :class:`_schema.MetaData` ultimately
           associated with the clause.
        5. if mapper given, attempt to return a bind
           linked to the :class:`_schema.MetaData` ultimately
           associated with the :class:`_schema.Table` or other
           selectable to which the mapper is mapped.
        6. No bind can be found, :exc:`~sqlalchemy.exc.UnboundExecutionError`
           is raised.

        Note that the :meth:`.Session.get_bind` method can be overridden on
        a user-defined subclass of :class:`.Session` to provide any kind
        of bind resolution scheme.  See the example at
        :ref:`session_custom_partitioning`.

        :param mapper:
          Optional mapped class or corresponding :class:`_orm.Mapper` instance.
          The bind can be derived from a :class:`_orm.Mapper` first by
          consulting the "binds" map associated with this :class:`.Session`,
          and secondly by consulting the :class:`_schema.MetaData` associated
          with the :class:`_schema.Table` to which the :class:`_orm.Mapper` is
          mapped for a bind.

        :param clause:
            A :class:`_expression.ClauseElement` (i.e.
            :func:`_expression.select`,
            :func:`_expression.text`,
            etc.).  If the ``mapper`` argument is not present or could not
            produce a bind, the given expression construct will be searched
            for a bound element, typically a :class:`_schema.Table`
            associated with
            bound :class:`_schema.MetaData`.

        .. seealso::

             :ref:`session_partitioning`

             :paramref:`.Session.binds`

             :meth:`.Session.bind_mapper`

             :meth:`.Session.bind_table`


        """  # noqa: E501

        return self._proxied.get_bind(
            mapper=mapper,
            clause=clause,
            bind=bind,
            _sa_skip_events=_sa_skip_events,
            _sa_skip_for_implicit_returning=_sa_skip_for_implicit_returning,
            **kw,
        )

    def is_modified(
        self, instance: object, include_collections: bool = True
    ) -> bool:
        r"""Return ``True`` if the given instance has locally
        modified attributes.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        This method retrieves the history for each instrumented
        attribute on the instance and performs a comparison of the current
        value to its previously flushed or committed value, if any.

        It is in effect a more expensive and accurate
        version of checking for the given instance in the
        :attr:`.Session.dirty` collection; a full test for
        each attribute's net "dirty" status is performed.

        E.g.::

            return session.is_modified(someobject)

        A few caveats to this method apply:

        * Instances present in the :attr:`.Session.dirty` collection may
          report ``False`` when tested with this method.  This is because
          the object may have received change events via attribute mutation,
          thus placing it in :attr:`.Session.dirty`, but ultimately the state
          is the same as that loaded from the database, resulting in no net
          change here.
        * Scalar attributes may not have recorded the previously set
          value when a new value was applied, if the attribute was not loaded,
          or was expired, at the time the new value was received - in these
          cases, the attribute is assumed to have a change, even if there is
          ultimately no net change against its database value. SQLAlchemy in
          most cases does not need the "old" value when a set event occurs, so
          it skips the expense of a SQL call if the old value isn't present,
          based on the assumption that an UPDATE of the scalar value is
          usually needed, and in those few cases where it isn't, is less
          expensive on average than issuing a defensive SELECT.

          The "old" value is fetched unconditionally upon set only if the
          attribute container has the ``active_history`` flag set to ``True``.
          This flag is set typically for primary key attributes and scalar
          object references that are not a simple many-to-one.  To set this
          flag for any arbitrary mapped column, use the ``active_history``
          argument with :func:`.column_property`.

        :param instance: mapped instance to be tested for pending changes.
        :param include_collections: Indicates if multivalued collections
         should be included in the operation.  Setting this to ``False`` is a
         way to detect only local-column based properties (i.e. scalar columns
         or many-to-one foreign keys) that would result in an UPDATE for this
         instance upon flush.


        """  # noqa: E501

        return self._proxied.is_modified(
            instance, include_collections=include_collections
        )

    def bulk_save_objects(
        self,
        objects: Iterable[object],
        return_defaults: bool = False,
        update_changed_only: bool = True,
        preserve_order: bool = True,
    ) -> None:
        r"""Perform a bulk save of the given list of objects.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        .. legacy::

            This method is a legacy feature as of the 2.0 series of
            SQLAlchemy.   For modern bulk INSERT and UPDATE, see
            the sections :ref:`orm_queryguide_bulk_insert` and
            :ref:`orm_queryguide_bulk_update`.

            For general INSERT and UPDATE of existing ORM mapped objects,
            prefer standard :term:`unit of work` data management patterns,
            introduced in the :ref:`unified_tutorial` at
            :ref:`tutorial_orm_data_manipulation`.  SQLAlchemy 2.0
            now uses :ref:`engine_insertmanyvalues` with modern dialects
            which solves previous issues of bulk INSERT slowness.

        :param objects: a sequence of mapped object instances.  The mapped
         objects are persisted as is, and are **not** associated with the
         :class:`.Session` afterwards.

         For each object, whether the object is sent as an INSERT or an
         UPDATE is dependent on the same rules used by the :class:`.Session`
         in traditional operation; if the object has the
         :attr:`.InstanceState.key`
         attribute set, then the object is assumed to be "detached" and
         will result in an UPDATE.  Otherwise, an INSERT is used.

         In the case of an UPDATE, statements are grouped based on which
         attributes have changed, and are thus to be the subject of each
         SET clause.  If ``update_changed_only`` is False, then all
         attributes present within each object are applied to the UPDATE
         statement, which may help in allowing the statements to be grouped
         together into a larger executemany(), and will also reduce the
         overhead of checking history on attributes.

        :param return_defaults: when True, rows that are missing values which
         generate defaults, namely integer primary key defaults and sequences,
         will be inserted **one at a time**, so that the primary key value
         is available.  In particular this will allow joined-inheritance
         and other multi-table mappings to insert correctly without the need
         to provide primary key values ahead of time; however,
         :paramref:`.Session.bulk_save_objects.return_defaults` **greatly
         reduces the performance gains** of the method overall.  It is strongly
         advised to please use the standard :meth:`_orm.Session.add_all`
         approach.

        :param update_changed_only: when True, UPDATE statements are rendered
         based on those attributes in each state that have logged changes.
         When False, all attributes present are rendered into the SET clause
         with the exception of primary key attributes.

        :param preserve_order: when True, the order of inserts and updates
         matches exactly the order in which the objects are given.   When
         False, common types of objects are grouped into inserts
         and updates, to allow for more batching opportunities.

        .. seealso::

            :doc:`queryguide/dml`

            :meth:`.Session.bulk_insert_mappings`

            :meth:`.Session.bulk_update_mappings`


        """  # noqa: E501

        return self._proxied.bulk_save_objects(
            objects,
            return_defaults=return_defaults,
            update_changed_only=update_changed_only,
            preserve_order=preserve_order,
        )

    def bulk_insert_mappings(
        self,
        mapper: Mapper[Any],
        mappings: Iterable[Dict[str, Any]],
        return_defaults: bool = False,
        render_nulls: bool = False,
    ) -> None:
        r"""Perform a bulk insert of the given list of mapping dictionaries.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        .. legacy::

            This method is a legacy feature as of the 2.0 series of
            SQLAlchemy.   For modern bulk INSERT and UPDATE, see
            the sections :ref:`orm_queryguide_bulk_insert` and
            :ref:`orm_queryguide_bulk_update`.  The 2.0 API shares
            implementation details with this method and adds new features
            as well.

        :param mapper: a mapped class, or the actual :class:`_orm.Mapper`
         object,
         representing the single kind of object represented within the mapping
         list.

        :param mappings: a sequence of dictionaries, each one containing the
         state of the mapped row to be inserted, in terms of the attribute
         names on the mapped class.   If the mapping refers to multiple tables,
         such as a joined-inheritance mapping, each dictionary must contain all
         keys to be populated into all tables.

        :param return_defaults: when True, the INSERT process will be altered
         to ensure that newly generated primary key values will be fetched.
         The rationale for this parameter is typically to enable
         :ref:`Joined Table Inheritance <joined_inheritance>` mappings to
         be bulk inserted.

         .. note:: for backends that don't support RETURNING, the
            :paramref:`_orm.Session.bulk_insert_mappings.return_defaults`
            parameter can significantly decrease performance as INSERT
            statements can no longer be batched.   See
            :ref:`engine_insertmanyvalues`
            for background on which backends are affected.

        :param render_nulls: When True, a value of ``None`` will result
         in a NULL value being included in the INSERT statement, rather
         than the column being omitted from the INSERT.   This allows all
         the rows being INSERTed to have the identical set of columns which
         allows the full set of rows to be batched to the DBAPI.  Normally,
         each column-set that contains a different combination of NULL values
         than the previous row must omit a different series of columns from
         the rendered INSERT statement, which means it must be emitted as a
         separate statement.   By passing this flag, the full set of rows
         are guaranteed to be batchable into one batch; the cost however is
         that server-side defaults which are invoked by an omitted column will
         be skipped, so care must be taken to ensure that these are not
         necessary.

         .. warning::

            When this flag is set, **server side default SQL values will
            not be invoked** for those columns that are inserted as NULL;
            the NULL value will be sent explicitly.   Care must be taken
            to ensure that no server-side default functions need to be
            invoked for the operation as a whole.

        .. seealso::

            :doc:`queryguide/dml`

            :meth:`.Session.bulk_save_objects`

            :meth:`.Session.bulk_update_mappings`


        """  # noqa: E501

        return self._proxied.bulk_insert_mappings(
            mapper,
            mappings,
            return_defaults=return_defaults,
            render_nulls=render_nulls,
        )

    def bulk_update_mappings(
        self, mapper: Mapper[Any], mappings: Iterable[Dict[str, Any]]
    ) -> None:
        r"""Perform a bulk update of the given list of mapping dictionaries.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        .. legacy::

            This method is a legacy feature as of the 2.0 series of
            SQLAlchemy.   For modern bulk INSERT and UPDATE, see
            the sections :ref:`orm_queryguide_bulk_insert` and
            :ref:`orm_queryguide_bulk_update`.  The 2.0 API shares
            implementation details with this method and adds new features
            as well.

        :param mapper: a mapped class, or the actual :class:`_orm.Mapper`
         object,
         representing the single kind of object represented within the mapping
         list.

        :param mappings: a sequence of dictionaries, each one containing the
         state of the mapped row to be updated, in terms of the attribute names
         on the mapped class.   If the mapping refers to multiple tables, such
         as a joined-inheritance mapping, each dictionary may contain keys
         corresponding to all tables.   All those keys which are present and
         are not part of the primary key are applied to the SET clause of the
         UPDATE statement; the primary key values, which are required, are
         applied to the WHERE clause.


        .. seealso::

            :doc:`queryguide/dml`

            :meth:`.Session.bulk_insert_mappings`

            :meth:`.Session.bulk_save_objects`


        """  # noqa: E501

        return self._proxied.bulk_update_mappings(mapper, mappings)

    def merge(
        self,
        instance: _O,
        *,
        load: bool = True,
        options: Optional[Sequence[ORMOption]] = None,
    ) -> _O:
        r"""Copy the state of a given instance into a corresponding instance
        within this :class:`.Session`.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        :meth:`.Session.merge` examines the primary key attributes of the
        source instance, and attempts to reconcile it with an instance of the
        same primary key in the session.   If not found locally, it attempts
        to load the object from the database based on primary key, and if
        none can be located, creates a new instance.  The state of each
        attribute on the source instance is then copied to the target
        instance.  The resulting target instance is then returned by the
        method; the original source instance is left unmodified, and
        un-associated with the :class:`.Session` if not already.

        This operation cascades to associated instances if the association is
        mapped with ``cascade="merge"``.

        See :ref:`unitofwork_merging` for a detailed discussion of merging.

        :param instance: Instance to be merged.
        :param load: Boolean, when False, :meth:`.merge` switches into
         a "high performance" mode which causes it to forego emitting history
         events as well as all database access.  This flag is used for
         cases such as transferring graphs of objects into a :class:`.Session`
         from a second level cache, or to transfer just-loaded objects
         into the :class:`.Session` owned by a worker thread or process
         without re-querying the database.

         The ``load=False`` use case adds the caveat that the given
         object has to be in a "clean" state, that is, has no pending changes
         to be flushed - even if the incoming object is detached from any
         :class:`.Session`.   This is so that when
         the merge operation populates local attributes and
         cascades to related objects and
         collections, the values can be "stamped" onto the
         target object as is, without generating any history or attribute
         events, and without the need to reconcile the incoming data with
         any existing related objects or collections that might not
         be loaded.  The resulting objects from ``load=False`` are always
         produced as "clean", so it is only appropriate that the given objects
         should be "clean" as well, else this suggests a mis-use of the
         method.
        :param options: optional sequence of loader options which will be
         applied to the :meth:`_orm.Session.get` method when the merge
         operation loads the existing version of the object from the database.

         .. versionadded:: 1.4.24


        .. seealso::

            :func:`.make_transient_to_detached` - provides for an alternative
            means of "merging" a single object into the :class:`.Session`


        """  # noqa: E501

        return self._proxied.merge(instance, load=load, options=options)

    @overload
    def query(self, _entity: _EntityType[_O]) -> Query[_O]: ...

    @overload
    def query(
        self, _colexpr: TypedColumnsClauseRole[_T]
    ) -> RowReturningQuery[Tuple[_T]]: ...

    # START OVERLOADED FUNCTIONS self.query RowReturningQuery 2-8

    # code within this block is **programmatically,
    # statically generated** by tools/generate_tuple_map_overloads.py

    @overload
    def query(
        self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1]
    ) -> RowReturningQuery[Tuple[_T0, _T1]]: ...

    @overload
    def query(
        self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2]
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

    @overload
    def query(
        self,
        __ent0: _TCCA[_T0],
        __ent1: _TCCA[_T1],
        __ent2: _TCCA[_T2],
        __ent3: _TCCA[_T3],
        __ent4: _TCCA[_T4],
        __ent5: _TCCA[_T5],
        __ent6: _TCCA[_T6],
        __ent7: _TCCA[_T7],
    ) -> RowReturningQuery[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]]: ...

    # END OVERLOADED FUNCTIONS self.query

    @overload
    def query(
        self, *entities: _ColumnsClauseArgument[Any], **kwargs: Any
    ) -> Query[Any]: ...

    def query(
        self, *entities: _ColumnsClauseArgument[Any], **kwargs: Any
    ) -> Query[Any]:
        r"""Return a new :class:`_query.Query` object corresponding to this
        :class:`_orm.Session`.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Note that the :class:`_query.Query` object is legacy as of
        SQLAlchemy 2.0; the :func:`_sql.select` construct is now used
        to construct ORM queries.

        .. seealso::

            :ref:`unified_tutorial`

            :ref:`queryguide_toplevel`

            :ref:`query_api_toplevel` - legacy API doc


        """  # noqa: E501

        return self._proxied.query(*entities, **kwargs)

    def refresh(
        self,
        instance: object,
        attribute_names: Optional[Iterable[str]] = None,
        with_for_update: ForUpdateParameter = None,
    ) -> None:
        r"""Expire and refresh attributes on the given instance.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        The selected attributes will first be expired as they would when using
        :meth:`_orm.Session.expire`; then a SELECT statement will be issued to
        the database to refresh column-oriented attributes with the current
        value available in the current transaction.

        :func:`_orm.relationship` oriented attributes will also be immediately
        loaded if they were already eagerly loaded on the object, using the
        same eager loading strategy that they were loaded with originally.

        .. versionadded:: 1.4 - the :meth:`_orm.Session.refresh` method
           can also refresh eagerly loaded attributes.

        :func:`_orm.relationship` oriented attributes that would normally
        load using the ``select`` (or "lazy") loader strategy will also
        load **if they are named explicitly in the attribute_names
        collection**, emitting a SELECT statement for the attribute using the
        ``immediate`` loader strategy.  If lazy-loaded relationships are not
        named in :paramref:`_orm.Session.refresh.attribute_names`, then
        they remain as "lazy loaded" attributes and are not implicitly
        refreshed.

        .. versionchanged:: 2.0.4  The :meth:`_orm.Session.refresh` method
           will now refresh lazy-loaded :func:`_orm.relationship` oriented
           attributes for those which are named explicitly in the
           :paramref:`_orm.Session.refresh.attribute_names` collection.

        .. tip::

            While the :meth:`_orm.Session.refresh` method is capable of
            refreshing both column and relationship oriented attributes, its
            primary focus is on refreshing of local column-oriented attributes
            on a single instance. For more open ended "refresh" functionality,
            including the ability to refresh the attributes on many objects at
            once while having explicit control over relationship loader
            strategies, use the
            :ref:`populate existing <orm_queryguide_populate_existing>` feature
            instead.

        Note that a highly isolated transaction will return the same values as
        were previously read in that same transaction, regardless of changes
        in database state outside of that transaction.   Refreshing
        attributes usually only makes sense at the start of a transaction
        where database rows have not yet been accessed.

        :param attribute_names: optional.  An iterable collection of
          string attribute names indicating a subset of attributes to
          be refreshed.

        :param with_for_update: optional boolean ``True`` indicating FOR UPDATE
          should be used, or may be a dictionary containing flags to
          indicate a more specific set of FOR UPDATE flags for the SELECT;
          flags should match the parameters of
          :meth:`_query.Query.with_for_update`.
          Supersedes the :paramref:`.Session.refresh.lockmode` parameter.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.expire_all`

            :ref:`orm_queryguide_populate_existing` - allows any ORM query
            to refresh objects as they would be loaded normally.


        """  # noqa: E501

        return self._proxied.refresh(
            instance,
            attribute_names=attribute_names,
            with_for_update=with_for_update,
        )

    def rollback(self) -> None:
        r"""Rollback the current transaction in progress.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        If no transaction is in progress, this method is a pass-through.

        The method always rolls back
        the topmost database transaction, discarding any nested
        transactions that may be in progress.

        .. seealso::

            :ref:`session_rollback`

            :ref:`unitofwork_transaction`


        """  # noqa: E501

        return self._proxied.rollback()

    @overload
    def scalar(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Optional[_T]: ...

    @overload
    def scalar(
        self,
        statement: Executable,
        params: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Any: ...

    def scalar(
        self,
        statement: Executable,
        params: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Any:
        r"""Execute a statement and return a scalar result.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Usage and parameters are the same as that of
        :meth:`_orm.Session.execute`; the return result is a scalar Python
        value.


        """  # noqa: E501

        return self._proxied.scalar(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )

    @overload
    def scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[_T]: ...

    @overload
    def scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[Any]: ...

    def scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[Any]:
        r"""Execute a statement and return the results as scalars.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        Usage and parameters are the same as that of
        :meth:`_orm.Session.execute`; the return result is a
        :class:`_result.ScalarResult` filtering object which
        will return single elements rather than :class:`_row.Row` objects.

        :return:  a :class:`_result.ScalarResult` object

        .. versionadded:: 1.4.24 Added :meth:`_orm.Session.scalars`

        .. versionadded:: 1.4.26 Added :meth:`_orm.scoped_session.scalars`

        .. seealso::

            :ref:`orm_queryguide_select_orm_entities` - contrasts the behavior
            of :meth:`_orm.Session.execute` to :meth:`_orm.Session.scalars`


        """  # noqa: E501

        return self._proxied.scalars(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )

    @property
    def bind(self) -> Optional[Union[Engine, Connection]]:
        r"""Proxy for the :attr:`_orm.Session.bind` attribute
        on behalf of the :class:`_orm.scoping.scoped_session` class.

        """  # noqa: E501

        return self._proxied.bind

    @bind.setter
    def bind(self, attr: Optional[Union[Engine, Connection]]) -> None:
        self._proxied.bind = attr

    @property
    def dirty(self) -> Any:
        r"""The set of all persistent instances considered dirty.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_orm.scoping.scoped_session` class.

        E.g.::

            some_mapped_object in session.dirty

        Instances are considered dirty when they were modified but not
        deleted.

        Note that this 'dirty' calculation is 'optimistic'; most
        attribute-setting or collection modification operations will
        mark an instance as 'dirty' and place it in this set, even if
        there is no net change to the attribute's value.  At flush
        time, the value of each attribute is compared to its
        previously saved value, and if there's no net change, no SQL
        operation will occur (this is a more expensive operation so
        it's only done at flush time).

        To check if an instance has actionable net changes to its
        attributes, use the :meth:`.Session.is_modified` method.


        """  # noqa: E501

        return self._proxied.dirty

    @property
    def deleted(self) -> Any:
        r"""The set of all instances marked as 'deleted' within this ``Session``

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_orm.scoping.scoped_session` class.

        """  # noqa: E501

        return self._proxied.deleted

    @property
    def new(self) -> Any:
        r"""The set of all instances marked as 'new' within this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_orm.scoping.scoped_session` class.

        """  # noqa: E501

        return self._proxied.new

    @property
    def identity_map(self) -> IdentityMap:
        r"""Proxy for the :attr:`_orm.Session.identity_map` attribute
        on behalf of the :class:`_orm.scoping.scoped_session` class.

        """  # noqa: E501

        return self._proxied.identity_map

    @identity_map.setter
    def identity_map(self, attr: IdentityMap) -> None:
        self._proxied.identity_map = attr

    @property
    def is_active(self) -> Any:
        r"""True if this :class:`.Session` not in "partial rollback" state.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_orm.scoping.scoped_session` class.

        .. versionchanged:: 1.4 The :class:`_orm.Session` no longer begins
           a new transaction immediately, so this attribute will be False
           when the :class:`_orm.Session` is first instantiated.

        "partial rollback" state typically indicates that the flush process
        of the :class:`_orm.Session` has failed, and that the
        :meth:`_orm.Session.rollback` method must be emitted in order to
        fully roll back the transaction.

        If this :class:`_orm.Session` is not in a transaction at all, the
        :class:`_orm.Session` will autobegin when it is first used, so in this
        case :attr:`_orm.Session.is_active` will return True.

        Otherwise, if this :class:`_orm.Session` is within a transaction,
        and that transaction has not been rolled back internally, the
        :attr:`_orm.Session.is_active` will also return True.

        .. seealso::

            :ref:`faq_session_rollback`

            :meth:`_orm.Session.in_transaction`


        """  # noqa: E501

        return self._proxied.is_active

    @property
    def autoflush(self) -> bool:
        r"""Proxy for the :attr:`_orm.Session.autoflush` attribute
        on behalf of the :class:`_orm.scoping.scoped_session` class.

        """  # noqa: E501

        return self._proxied.autoflush

    @autoflush.setter
    def autoflush(self, attr: bool) -> None:
        self._proxied.autoflush = attr

    @property
    def no_autoflush(self) -> Any:
        r"""Return a context manager that disables autoflush.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_orm.scoping.scoped_session` class.

        e.g.::

            with session.no_autoflush:

                some_object = SomeClass()
                session.add(some_object)
                # won't autoflush
                some_object.related_thing = session.query(SomeRelated).first()

        Operations that proceed within the ``with:`` block
        will not be subject to flushes occurring upon query
        access.  This is useful when initializing a series
        of objects which involve existing database queries,
        where the uncompleted object should not yet be flushed.


        """  # noqa: E501

        return self._proxied.no_autoflush

    @property
    def info(self) -> Any:
        r"""A user-modifiable dictionary.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_orm.scoping.scoped_session` class.

        The initial value of this dictionary can be populated using the
        ``info`` argument to the :class:`.Session` constructor or
        :class:`.sessionmaker` constructor or factory methods.  The dictionary
        here is always local to this :class:`.Session` and can be modified
        independently of all other :class:`.Session` objects.


        """  # noqa: E501

        return self._proxied.info

    @classmethod
    def close_all(cls) -> None:
        r"""Close *all* sessions in memory.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        .. deprecated:: 1.3 The :meth:`.Session.close_all` method is deprecated and will be removed in a future release.  Please refer to :func:`.session.close_all_sessions`.

        """  # noqa: E501

        return Session.close_all()

    @classmethod
    def object_session(cls, instance: object) -> Optional[Session]:
        r"""Return the :class:`.Session` to which an object belongs.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        This is an alias of :func:`.object_session`.


        """  # noqa: E501

        return Session.object_session(instance)

    @classmethod
    def identity_key(
        cls,
        class_: Optional[Type[Any]] = None,
        ident: Union[Any, Tuple[Any, ...]] = None,
        *,
        instance: Optional[Any] = None,
        row: Optional[Union[Row[Any], RowMapping]] = None,
        identity_token: Optional[Any] = None,
    ) -> _IdentityKeyType[Any]:
        r"""Return an identity key.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_orm.scoping.scoped_session` class.

        This is an alias of :func:`.util.identity_key`.


        """  # noqa: E501

        return Session.identity_key(
            class_=class_,
            ident=ident,
            instance=instance,
            row=row,
            identity_token=identity_token,
        )

    # END PROXY METHODS scoped_session


ScopedSession = scoped_session
"""Old name for backwards compatibility."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\mathml.py ===
"""
A MathML printer.
"""

from __future__ import annotations
from typing import Any

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import \
    precedence_traditional, PRECEDENCE, PRECEDENCE_TRADITIONAL
from sympy.printing.pretty.pretty_symbology import greek_unicode
from sympy.printing.printer import Printer, print_function

from mpmath.libmp import prec_to_dps, repr_dps, to_str as mlib_to_str


class MathMLPrinterBase(Printer):
    """Contains common code required for MathMLContentPrinter and
    MathMLPresentationPrinter.
    """

    _default_settings: dict[str, Any] = {
        "order": None,
        "encoding": "utf-8",
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_symbol_style": "plain",
        "mul_symbol": None,
        "root_notation": True,
        "symbol_names": {},
        "mul_symbol_mathml_numbers": '&#xB7;',
        "disable_split_super_sub": False,
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)
        from xml.dom.minidom import Document, Text

        self.dom = Document()

        # Workaround to allow strings to remain unescaped
        # Based on
        # https://stackoverflow.com/questions/38015864/python-xml-dom-minidom-\
        #                              please-dont-escape-my-strings/38041194
        class RawText(Text):
            def writexml(self, writer, indent='', addindent='', newl=''):
                if self.data:
                    writer.write('{}{}{}'.format(indent, self.data, newl))

        def createRawTextNode(data):
            r = RawText()
            r.data = data
            r.ownerDocument = self.dom
            return r

        self.dom.createTextNode = createRawTextNode

    def doprint(self, expr):
        """
        Prints the expression as MathML.
        """
        mathML = Printer._print(self, expr)
        unistr = mathML.toxml()
        xmlbstr = unistr.encode('ascii', 'xmlcharrefreplace')
        res = xmlbstr.decode()
        return res

    def _split_super_sub(self, name):
        if self._settings["disable_split_super_sub"]:
            return (name, [], [])
        else:
            return split_super_sub(name)


class MathMLContentPrinter(MathMLPrinterBase):
    """Prints an expression to the Content MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter4.html
    """
    printmethod = "_mathml_content"

    def mathml_tag(self, e):
        """Returns the MathML tag for an expression."""
        translate = {
            'Add': 'plus',
            'Mul': 'times',
            'Derivative': 'diff',
            'Number': 'cn',
            'int': 'cn',
            'Pow': 'power',
            'Max': 'max',
            'Min': 'min',
            'Abs': 'abs',
            'And': 'and',
            'Or': 'or',
            'Xor': 'xor',
            'Not': 'not',
            'Implies': 'implies',
            'Symbol': 'ci',
            'MatrixSymbol': 'ci',
            'RandomSymbol': 'ci',
            'Integral': 'int',
            'Sum': 'sum',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'cot': 'cot',
            'csc': 'csc',
            'sec': 'sec',
            'sinh': 'sinh',
            'cosh': 'cosh',
            'tanh': 'tanh',
            'coth': 'coth',
            'csch': 'csch',
            'sech': 'sech',
            'asin': 'arcsin',
            'asinh': 'arcsinh',
            'acos': 'arccos',
            'acosh': 'arccosh',
            'atan': 'arctan',
            'atanh': 'arctanh',
            'atan2': 'arctan',
            'acot': 'arccot',
            'acoth': 'arccoth',
            'asec': 'arcsec',
            'asech': 'arcsech',
            'acsc': 'arccsc',
            'acsch': 'arccsch',
            'log': 'ln',
            'Equality': 'eq',
            'Unequality': 'neq',
            'GreaterThan': 'geq',
            'LessThan': 'leq',
            'StrictGreaterThan': 'gt',
            'StrictLessThan': 'lt',
            'Union': 'union',
            'Intersection': 'intersect',
        }

        for cls in e.__class__.__mro__:
            n = cls.__name__
            if n in translate:
                return translate[n]
        # Not found in the MRO set
        n = e.__class__.__name__
        return n.lower()

    def _print_Mul(self, expr):

        if expr.could_extract_minus_sign():
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('minus'))
            x.appendChild(self._print_Mul(-expr))
            return x

        from sympy.simplify import fraction
        numer, denom = fraction(expr)

        if denom is not S.One:
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('divide'))
            x.appendChild(self._print(numer))
            x.appendChild(self._print(denom))
            return x

        coeff, terms = expr.as_coeff_mul()
        if coeff is S.One and len(terms) == 1:
            # XXX since the negative coefficient has been handled, I don't
            # think a coeff of 1 can remain
            return self._print(terms[0])

        if self.order != 'old':
            terms = Mul._from_args(terms).as_ordered_factors()

        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('times'))
        if coeff != 1:
            x.appendChild(self._print(coeff))
        for term in terms:
            x.appendChild(self._print(term))
        return x

    def _print_Add(self, expr, order=None):
        args = self._as_ordered_terms(expr, order=order)
        lastProcessed = self._print(args[0])
        plusNodes = []
        for arg in args[1:]:
            if arg.could_extract_minus_sign():
                # use minus
                x = self.dom.createElement('apply')
                x.appendChild(self.dom.createElement('minus'))
                x.appendChild(lastProcessed)
                x.appendChild(self._print(-arg))
                # invert expression since this is now minused
                lastProcessed = x
                if arg == args[-1]:
                    plusNodes.append(lastProcessed)
            else:
                plusNodes.append(lastProcessed)
                lastProcessed = self._print(arg)
                if arg == args[-1]:
                    plusNodes.append(self._print(arg))
        if len(plusNodes) == 1:
            return lastProcessed
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('plus'))
        while plusNodes:
            x.appendChild(plusNodes.pop(0))
        return x

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        root = self.dom.createElement('piecewise')
        for i, (e, c) in enumerate(expr.args):
            if i == len(expr.args) - 1 and c == True:
                piece = self.dom.createElement('otherwise')
                piece.appendChild(self._print(e))
            else:
                piece = self.dom.createElement('piece')
                piece.appendChild(self._print(e))
                piece.appendChild(self._print(c))
            root.appendChild(piece)
        return root

    def _print_MatrixBase(self, m):
        x = self.dom.createElement('matrix')
        for i in range(m.rows):
            x_r = self.dom.createElement('matrixrow')
            for j in range(m.cols):
                x_r.appendChild(self._print(m[i, j]))
            x.appendChild(x_r)
        return x

    def _print_Rational(self, e):
        if e.q == 1:
            # don't divide
            x = self.dom.createElement('cn')
            x.appendChild(self.dom.createTextNode(str(e.p)))
            return x
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('divide'))
        # numerator
        xnum = self.dom.createElement('cn')
        xnum.appendChild(self.dom.createTextNode(str(e.p)))
        # denominator
        xdenom = self.dom.createElement('cn')
        xdenom.appendChild(self.dom.createTextNode(str(e.q)))
        x.appendChild(xnum)
        x.appendChild(xdenom)
        return x

    def _print_Limit(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))

        x_1 = self.dom.createElement('bvar')
        x_2 = self.dom.createElement('lowlimit')
        x_1.appendChild(self._print(e.args[1]))
        x_2.appendChild(self._print(e.args[2]))

        x.appendChild(x_1)
        x.appendChild(x_2)
        x.appendChild(self._print(e.args[0]))
        return x

    def _print_ImaginaryUnit(self, e):
        return self.dom.createElement('imaginaryi')

    def _print_EulerGamma(self, e):
        return self.dom.createElement('eulergamma')

    def _print_GoldenRatio(self, e):
        """We use unicode #x3c6 for Greek letter phi as defined here
        https://www.w3.org/2003/entities/2007doc/isogrk1.html"""
        x = self.dom.createElement('cn')
        x.appendChild(self.dom.createTextNode("\N{GREEK SMALL LETTER PHI}"))
        return x

    def _print_Exp1(self, e):
        return self.dom.createElement('exponentiale')

    def _print_Pi(self, e):
        return self.dom.createElement('pi')

    def _print_Infinity(self, e):
        return self.dom.createElement('infinity')

    def _print_NaN(self, e):
        return self.dom.createElement('notanumber')

    def _print_EmptySet(self, e):
        return self.dom.createElement('emptyset')

    def _print_BooleanTrue(self, e):
        return self.dom.createElement('true')

    def _print_BooleanFalse(self, e):
        return self.dom.createElement('false')

    def _print_NegativeInfinity(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('minus'))
        x.appendChild(self.dom.createElement('infinity'))
        return x

    def _print_Integral(self, e):
        def lime_recur(limits):
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement(self.mathml_tag(e)))
            bvar_elem = self.dom.createElement('bvar')
            bvar_elem.appendChild(self._print(limits[0][0]))
            x.appendChild(bvar_elem)

            if len(limits[0]) == 3:
                low_elem = self.dom.createElement('lowlimit')
                low_elem.appendChild(self._print(limits[0][1]))
                x.appendChild(low_elem)
                up_elem = self.dom.createElement('uplimit')
                up_elem.appendChild(self._print(limits[0][2]))
                x.appendChild(up_elem)
            if len(limits[0]) == 2:
                up_elem = self.dom.createElement('uplimit')
                up_elem.appendChild(self._print(limits[0][1]))
                x.appendChild(up_elem)
            if len(limits) == 1:
                x.appendChild(self._print(e.function))
            else:
                x.appendChild(lime_recur(limits[1:]))
            return x

        limits = list(e.limits)
        limits.reverse()
        return lime_recur(limits)

    def _print_Sum(self, e):
        # Printer can be shared because Sum and Integral have the
        # same internal representation.
        return self._print_Integral(e)

    def _print_Symbol(self, sym):
        ci = self.dom.createElement(self.mathml_tag(sym))

        def join(items):
            if len(items) > 1:
                mrow = self.dom.createElement('mml:mrow')
                for i, item in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mml:mo')
                        mo.appendChild(self.dom.createTextNode(" "))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mml:mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mml:mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi

        # translate name, supers and subs to unicode characters
        def translate(s):
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s

        name, supers, subs = self._split_super_sub(sym.name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        mname = self.dom.createElement('mml:mi')
        mname.appendChild(self.dom.createTextNode(name))
        if not supers:
            if not subs:
                ci.appendChild(self.dom.createTextNode(name))
            else:
                msub = self.dom.createElement('mml:msub')
                msub.appendChild(mname)
                msub.appendChild(join(subs))
                ci.appendChild(msub)
        else:
            if not subs:
                msup = self.dom.createElement('mml:msup')
                msup.appendChild(mname)
                msup.appendChild(join(supers))
                ci.appendChild(msup)
            else:
                msubsup = self.dom.createElement('mml:msubsup')
                msubsup.appendChild(mname)
                msubsup.appendChild(join(subs))
                msubsup.appendChild(join(supers))
                ci.appendChild(msubsup)
        return ci

    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print_Pow(self, e):
        # Here we use root instead of power if the exponent is the reciprocal
        # of an integer
        if (self._settings['root_notation'] and e.exp.is_Rational
                and e.exp.p == 1):
            x = self.dom.createElement('apply')
            x.appendChild(self.dom.createElement('root'))
            if e.exp.q != 2:
                xmldeg = self.dom.createElement('degree')
                xmlcn = self.dom.createElement('cn')
                xmlcn.appendChild(self.dom.createTextNode(str(e.exp.q)))
                xmldeg.appendChild(xmlcn)
                x.appendChild(xmldeg)
            x.appendChild(self._print(e.base))
            return x

        x = self.dom.createElement('apply')
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        x.appendChild(self._print(e.base))
        x.appendChild(self._print(e.exp))
        return x

    def _print_Number(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    def _print_Float(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        repr_e = mlib_to_str(e._mpf_, repr_dps(e._prec))
        x.appendChild(self.dom.createTextNode(repr_e))
        return x

    def _print_Derivative(self, e):
        x = self.dom.createElement('apply')
        diff_symbol = self.mathml_tag(e)
        if requires_partial(e.expr):
            diff_symbol = 'partialdiff'
        x.appendChild(self.dom.createElement(diff_symbol))
        x_1 = self.dom.createElement('bvar')

        for sym, times in reversed(e.variable_count):
            x_1.appendChild(self._print(sym))
            if times > 1:
                degree = self.dom.createElement('degree')
                degree.appendChild(self._print(sympify(times)))
                x_1.appendChild(degree)

        x.appendChild(x_1)
        x.appendChild(self._print(e.expr))
        return x

    def _print_Function(self, e):
        x = self.dom.createElement("apply")
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Basic(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_AssocOp(self, e):
        x = self.dom.createElement('apply')
        x_1 = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(x_1)
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Relational(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement(self.mathml_tag(e)))
        x.appendChild(self._print(e.lhs))
        x.appendChild(self._print(e.rhs))
        return x

    def _print_list(self, seq):
        """MathML reference for the <list> element:
        https://www.w3.org/TR/MathML2/chapter4.html#contm.list"""
        dom_element = self.dom.createElement('list')
        for item in seq:
            dom_element.appendChild(self._print(item))
        return dom_element

    def _print_int(self, p):
        dom_element = self.dom.createElement(self.mathml_tag(p))
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        return dom_element

    _print_Implies = _print_AssocOp
    _print_Not = _print_AssocOp
    _print_Xor = _print_AssocOp

    def _print_FiniteSet(self, e):
        x = self.dom.createElement('set')
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Complement(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('setdiff'))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_ProductSet(self, e):
        x = self.dom.createElement('apply')
        x.appendChild(self.dom.createElement('cartesianproduct'))
        for arg in e.args:
            x.appendChild(self._print(arg))
        return x

    def _print_Lambda(self, e):
        # MathML reference for the lambda element:
        # https://www.w3.org/TR/MathML2/chapter4.html#id.4.2.1.7
        x = self.dom.createElement(self.mathml_tag(e))
        for arg in e.signature:
            x_1 = self.dom.createElement('bvar')
            x_1.appendChild(self._print(arg))
            x.appendChild(x_1)
        x.appendChild(self._print(e.expr))
        return x

    # XXX Symmetric difference is not supported for MathML content printers.


class MathMLPresentationPrinter(MathMLPrinterBase):
    """Prints an expression to the Presentation MathML markup language.

    References: https://www.w3.org/TR/MathML2/chapter3.html
    """
    printmethod = "_mathml_presentation"

    def mathml_tag(self, e):
        """Returns the MathML tag for an expression."""
        translate = {
            'Number': 'mn',
            'Limit': '&#x2192;',
            'Derivative': '&dd;',
            'int': 'mn',
            'Symbol': 'mi',
            'Integral': '&int;',
            'Sum': '&#x2211;',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'cot': 'cot',
            'asin': 'arcsin',
            'asinh': 'arcsinh',
            'acos': 'arccos',
            'acosh': 'arccosh',
            'atan': 'arctan',
            'atanh': 'arctanh',
            'acot': 'arccot',
            'atan2': 'arctan',
            'Equality': '=',
            'Unequality': '&#x2260;',
            'GreaterThan': '&#x2265;',
            'LessThan': '&#x2264;',
            'StrictGreaterThan': '>',
            'StrictLessThan': '<',
            'lerchphi': '&#x3A6;',
            'zeta': '&#x3B6;',
            'dirichlet_eta': '&#x3B7;',
            'elliptic_k': '&#x39A;',
            'lowergamma': '&#x3B3;',
            'uppergamma': '&#x393;',
            'gamma': '&#x393;',
            'totient': '&#x3D5;',
            'reduced_totient': '&#x3BB;',
            'primenu': '&#x3BD;',
            'primeomega': '&#x3A9;',
            'fresnels': 'S',
            'fresnelc': 'C',
            'LambertW': 'W',
            'Heaviside': '&#x398;',
            'BooleanTrue': 'True',
            'BooleanFalse': 'False',
            'NoneType': 'None',
            'mathieus': 'S',
            'mathieuc': 'C',
            'mathieusprime': 'S&#x2032;',
            'mathieucprime': 'C&#x2032;',
            'Lambda': 'lambda',
        }

        def mul_symbol_selection():
            if (self._settings["mul_symbol"] is None or
                    self._settings["mul_symbol"] == 'None'):
                return '&InvisibleTimes;'
            elif self._settings["mul_symbol"] == 'times':
                return '&#xD7;'
            elif self._settings["mul_symbol"] == 'dot':
                return '&#xB7;'
            elif self._settings["mul_symbol"] == 'ldot':
                return '&#x2024;'
            elif not isinstance(self._settings["mul_symbol"], str):
                raise TypeError
            else:
                return self._settings["mul_symbol"]
        for cls in e.__class__.__mro__:
            n = cls.__name__
            if n in translate:
                return translate[n]
        # Not found in the MRO set
        if e.__class__.__name__ == "Mul":
            return mul_symbol_selection()
        n = e.__class__.__name__
        return n.lower()

    def _l_paren(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('('))
        return mo

    def _r_paren(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(')'))
        return mo

    def _l_brace(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('{'))
        return mo

    def _r_brace(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('}'))
        return mo

    def _comma(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(','))
        return mo

    def _bar(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('|'))
        return mo

    def _semicolon(self):
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(';'))
        return mo

    def _paren_comma_separated(self, *args):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._l_paren())
        for i, arg in enumerate(args):
            if i:
                mrow.appendChild(self._comma())
            mrow.appendChild(self._print(arg))
        mrow.appendChild(self._r_paren())
        return mrow

    def _paren_bar_separated(self, *args):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._l_paren())
        for i, arg in enumerate(args):
            if i:
                mrow.appendChild(self._bar())
            mrow.appendChild(self._print(arg))
        mrow.appendChild(self._r_paren())
        return mrow

    def parenthesize(self, item, level, strict=False):
        prec_val = precedence_traditional(item)
        if (prec_val < level) or ((not strict) and prec_val <= level):
            mrow = self.dom.createElement('mrow')
            mrow.appendChild(self._l_paren())
            mrow.appendChild(self._print(item))
            mrow.appendChild(self._r_paren())
            return mrow
        return self._print(item)

    def _print_Mul(self, expr):

        def multiply(expr, mrow):
            from sympy.simplify import fraction
            numer, denom = fraction(expr)
            if denom is not S.One:
                frac = self.dom.createElement('mfrac')
                if self._settings["fold_short_frac"] and len(str(expr)) < 7:
                    frac.setAttribute('bevelled', 'true')
                xnum = self._print(numer)
                xden = self._print(denom)
                frac.appendChild(xnum)
                frac.appendChild(xden)
                mrow.appendChild(frac)
                return mrow

            coeff, terms = expr.as_coeff_mul()
            if coeff is S.One and len(terms) == 1:
                mrow.appendChild(self._print(terms[0]))
                return mrow
            if self.order != 'old':
                terms = Mul._from_args(terms).as_ordered_factors()

            if coeff != 1:
                x = self._print(coeff)
                y = self.dom.createElement('mo')
                y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                mrow.appendChild(x)
                mrow.appendChild(y)
            for term in terms:
                mrow.appendChild(self.parenthesize(term, PRECEDENCE['Mul']))
                if not term == terms[-1]:
                    y = self.dom.createElement('mo')
                    y.appendChild(self.dom.createTextNode(self.mathml_tag(expr)))
                    mrow.appendChild(y)
            return mrow
        mrow = self.dom.createElement('mrow')
        if expr.could_extract_minus_sign():
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode('-'))
            mrow.appendChild(x)
            mrow = multiply(-expr, mrow)
        else:
            mrow = multiply(expr, mrow)

        return mrow

    def _print_Add(self, expr, order=None):
        mrow = self.dom.createElement('mrow')
        args = self._as_ordered_terms(expr, order=order)
        mrow.appendChild(self._print(args[0]))
        for arg in args[1:]:
            if arg.could_extract_minus_sign():
                # use minus
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('-'))
                y = self._print(-arg)
                # invert expression since this is now minused
            else:
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode('+'))
                y = self._print(arg)
            mrow.appendChild(x)
            mrow.appendChild(y)

        return mrow

    def _print_MatrixBase(self, m):
        table = self.dom.createElement('mtable')
        for i in range(m.rows):
            x = self.dom.createElement('mtr')
            for j in range(m.cols):
                y = self.dom.createElement('mtd')
                y.appendChild(self._print(m[i, j]))
                x.appendChild(y)
            table.appendChild(x)
        mat_delim = self._settings["mat_delim"]
        if mat_delim == '':
            return table
        left = self.dom.createElement('mo')
        right = self.dom.createElement('mo')
        if mat_delim == "[":
            left.appendChild(self.dom.createTextNode("["))
            right.appendChild(self.dom.createTextNode("]"))
        else:
            left.appendChild(self.dom.createTextNode("("))
            right.appendChild(self.dom.createTextNode(")"))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(left)
        mrow.appendChild(table)
        mrow.appendChild(right)
        return mrow

    def _get_printed_Rational(self, e, folded=None):
        if e.p < 0:
            p = -e.p
        else:
            p = e.p
        x = self.dom.createElement('mfrac')
        if folded or self._settings["fold_short_frac"]:
            x.setAttribute('bevelled', 'true')
        x.appendChild(self._print(p))
        x.appendChild(self._print(e.q))
        if e.p < 0:
            mrow = self.dom.createElement('mrow')
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('-'))
            mrow.appendChild(mo)
            mrow.appendChild(x)
            return mrow
        else:
            return x

    def _print_Rational(self, e):
        if e.q == 1:
            # don't divide
            return self._print(e.p)

        return self._get_printed_Rational(e, self._settings["fold_short_frac"])

    def _print_Limit(self, e):
        mrow = self.dom.createElement('mrow')
        munder = self.dom.createElement('munder')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('lim'))

        x = self.dom.createElement('mrow')
        x_1 = self._print(e.args[1])
        arrow = self.dom.createElement('mo')
        arrow.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        x_2 = self._print(e.args[2])
        x.appendChild(x_1)
        x.appendChild(arrow)
        x.appendChild(x_2)

        munder.appendChild(mi)
        munder.appendChild(x)
        mrow.appendChild(munder)
        mrow.appendChild(self._print(e.args[0]))

        return mrow

    def _print_ImaginaryUnit(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&ImaginaryI;'))
        return x

    def _print_GoldenRatio(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x3A6;'))
        return x

    def _print_Exp1(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&ExponentialE;'))
        return x

    def _print_Pi(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&pi;'))
        return x

    def _print_Infinity(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x221E;'))
        return x

    def _print_NegativeInfinity(self, e):
        mrow = self.dom.createElement('mrow')
        y = self.dom.createElement('mo')
        y.appendChild(self.dom.createTextNode('-'))
        x = self._print_Infinity(e)
        mrow.appendChild(y)
        mrow.appendChild(x)
        return mrow

    def _print_HBar(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x210F;'))
        return x

    def _print_EulerGamma(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x3B3;'))
        return x

    def _print_TribonacciConstant(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('TribonacciConstant'))
        return x

    def _print_Dagger(self, e):
        msup = self.dom.createElement('msup')
        msup.appendChild(self._print(e.args[0]))
        msup.appendChild(self.dom.createTextNode('&#x2020;'))
        return msup

    def _print_Contains(self, e):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._print(e.args[0]))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2208;'))
        mrow.appendChild(mo)
        mrow.appendChild(self._print(e.args[1]))
        return mrow

    def _print_HilbertSpace(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x210B;'))
        return x

    def _print_ComplexSpace(self, e):
        msup = self.dom.createElement('msup')
        msup.appendChild(self.dom.createTextNode('&#x1D49E;'))
        msup.appendChild(self._print(e.args[0]))
        return msup

    def _print_FockSpace(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x2131;'))
        return x


    def _print_Integral(self, expr):
        intsymbols = {1: "&#x222B;", 2: "&#x222C;", 3: "&#x222D;"}

        mrow = self.dom.createElement('mrow')
        if len(expr.limits) <= 3 and all(len(lim) == 1 for lim in expr.limits):
            # Only up to three-integral signs exists
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode(intsymbols[len(expr.limits)]))
            mrow.appendChild(mo)
        else:
            # Either more than three or limits provided
            for lim in reversed(expr.limits):
                mo = self.dom.createElement('mo')
                mo.appendChild(self.dom.createTextNode(intsymbols[1]))
                if len(lim) == 1:
                    mrow.appendChild(mo)
                if len(lim) == 2:
                    msup = self.dom.createElement('msup')
                    msup.appendChild(mo)
                    msup.appendChild(self._print(lim[1]))
                    mrow.appendChild(msup)
                if len(lim) == 3:
                    msubsup = self.dom.createElement('msubsup')
                    msubsup.appendChild(mo)
                    msubsup.appendChild(self._print(lim[1]))
                    msubsup.appendChild(self._print(lim[2]))
                    mrow.appendChild(msubsup)
        # print function
        mrow.appendChild(self.parenthesize(expr.function, PRECEDENCE["Mul"],
                                           strict=True))
        # print integration variables
        for lim in reversed(expr.limits):
            d = self.dom.createElement('mo')
            d.appendChild(self.dom.createTextNode('&dd;'))
            mrow.appendChild(d)
            mrow.appendChild(self._print(lim[0]))
        return mrow

    def _print_Sum(self, e):
        limits = list(e.limits)
        subsup = self.dom.createElement('munderover')
        low_elem = self._print(limits[0][1])
        up_elem = self._print(limits[0][2])
        summand = self.dom.createElement('mo')
        summand.appendChild(self.dom.createTextNode(self.mathml_tag(e)))

        low = self.dom.createElement('mrow')
        var = self._print(limits[0][0])
        equal = self.dom.createElement('mo')
        equal.appendChild(self.dom.createTextNode('='))
        low.appendChild(var)
        low.appendChild(equal)
        low.appendChild(low_elem)

        subsup.appendChild(summand)
        subsup.appendChild(low)
        subsup.appendChild(up_elem)

        mrow = self.dom.createElement('mrow')
        mrow.appendChild(subsup)
        mrow.appendChild(self.parenthesize(e.function, precedence_traditional(e)))
        return mrow

    def _print_Symbol(self, sym, style='plain'):
        def join(items):
            if len(items) > 1:
                mrow = self.dom.createElement('mrow')
                for i, item in enumerate(items):
                    if i > 0:
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode(" "))
                        mrow.appendChild(mo)
                    mi = self.dom.createElement('mi')
                    mi.appendChild(self.dom.createTextNode(item))
                    mrow.appendChild(mi)
                return mrow
            else:
                mi = self.dom.createElement('mi')
                mi.appendChild(self.dom.createTextNode(items[0]))
                return mi

        # translate name, supers and subs to unicode characters
        def translate(s):
            if s in greek_unicode:
                return greek_unicode.get(s)
            else:
                return s

        name, supers, subs = self._split_super_sub(sym.name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        mname = self.dom.createElement('mi')
        mname.appendChild(self.dom.createTextNode(name))
        if len(supers) == 0:
            if len(subs) == 0:
                x = mname
            else:
                x = self.dom.createElement('msub')
                x.appendChild(mname)
                x.appendChild(join(subs))
        else:
            if len(subs) == 0:
                x = self.dom.createElement('msup')
                x.appendChild(mname)
                x.appendChild(join(supers))
            else:
                x = self.dom.createElement('msubsup')
                x.appendChild(mname)
                x.appendChild(join(subs))
                x.appendChild(join(supers))
        # Set bold font?
        if style == 'bold':
            x.setAttribute('mathvariant', 'bold')
        return x

    def _print_MatrixSymbol(self, sym):
        return self._print_Symbol(sym,
                                  style=self._settings['mat_symbol_style'])

    _print_RandomSymbol = _print_Symbol

    def _print_conjugate(self, expr):
        enc = self.dom.createElement('menclose')
        enc.setAttribute('notation', 'top')
        enc.appendChild(self._print(expr.args[0]))
        return enc

    def _print_operator_after(self, op, expr):
        row = self.dom.createElement('mrow')
        row.appendChild(self.parenthesize(expr, PRECEDENCE["Func"]))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode(op))
        row.appendChild(mo)
        return row

    def _print_factorial(self, expr):
        return self._print_operator_after('!', expr.args[0])

    def _print_factorial2(self, expr):
        return self._print_operator_after('!!', expr.args[0])

    def _print_binomial(self, expr):
        frac = self.dom.createElement('mfrac')
        frac.setAttribute('linethickness', '0')
        frac.appendChild(self._print(expr.args[0]))
        frac.appendChild(self._print(expr.args[1]))
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_paren())
        brac.appendChild(frac)
        brac.appendChild(self._r_paren())
        return brac

    def _print_Pow(self, e):
        # Here we use root instead of power if the exponent is the
        # reciprocal of an integer
        if (e.exp.is_Rational and abs(e.exp.p) == 1 and e.exp.q != 1 and
                self._settings['root_notation']):
            if e.exp.q == 2:
                x = self.dom.createElement('msqrt')
                x.appendChild(self._print(e.base))
            if e.exp.q != 2:
                x = self.dom.createElement('mroot')
                x.appendChild(self._print(e.base))
                x.appendChild(self._print(e.exp.q))
            if e.exp.p == -1:
                frac = self.dom.createElement('mfrac')
                frac.appendChild(self._print(1))
                frac.appendChild(x)
                return frac
            else:
                return x

        if e.exp.is_Rational and e.exp.q != 1:
            if e.exp.is_negative:
                top = self.dom.createElement('mfrac')
                top.appendChild(self._print(1))
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._get_printed_Rational(-e.exp,
                                    self._settings['fold_frac_powers']))
                top.appendChild(x)
                return top
            else:
                x = self.dom.createElement('msup')
                x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                x.appendChild(self._get_printed_Rational(e.exp,
                                    self._settings['fold_frac_powers']))
                return x

        if e.exp.is_negative:
                top = self.dom.createElement('mfrac')
                top.appendChild(self._print(1))
                if e.exp == -1:
                    top.appendChild(self._print(e.base))
                else:
                    x = self.dom.createElement('msup')
                    x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
                    x.appendChild(self._print(-e.exp))
                    top.appendChild(x)
                return top

        x = self.dom.createElement('msup')
        x.appendChild(self.parenthesize(e.base, PRECEDENCE['Pow']))
        x.appendChild(self._print(e.exp))
        return x

    def _print_Number(self, e):
        x = self.dom.createElement(self.mathml_tag(e))
        x.appendChild(self.dom.createTextNode(str(e)))
        return x

    def _print_AccumulationBounds(self, i):
        left = self.dom.createElement('mo')
        left.appendChild(self.dom.createTextNode('\u27e8'))
        right = self.dom.createElement('mo')
        right.appendChild(self.dom.createTextNode('\u27e9'))
        brac = self.dom.createElement('mrow')
        brac.appendChild(left)
        brac.appendChild(self._print(i.min))
        brac.appendChild(self._comma())
        brac.appendChild(self._print(i.max))
        brac.appendChild(right)
        return brac

    def _print_Derivative(self, e):

        if requires_partial(e.expr):
            d = '&#x2202;'
        else:
            d = self.mathml_tag(e)

        # Determine denominator
        m = self.dom.createElement('mrow')
        dim = 0  # Total diff dimension, for numerator
        for sym, num in reversed(e.variable_count):
            dim += num
            if num >= 2:
                x = self.dom.createElement('msup')
                xx = self.dom.createElement('mo')
                xx.appendChild(self.dom.createTextNode(d))
                x.appendChild(xx)
                x.appendChild(self._print(num))
            else:
                x = self.dom.createElement('mo')
                x.appendChild(self.dom.createTextNode(d))
            m.appendChild(x)
            y = self._print(sym)
            m.appendChild(y)

        mnum = self.dom.createElement('mrow')
        if dim >= 2:
            x = self.dom.createElement('msup')
            xx = self.dom.createElement('mo')
            xx.appendChild(self.dom.createTextNode(d))
            x.appendChild(xx)
            x.appendChild(self._print(dim))
        else:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(d))

        mnum.appendChild(x)
        mrow = self.dom.createElement('mrow')
        frac = self.dom.createElement('mfrac')
        frac.appendChild(mnum)
        frac.appendChild(m)
        mrow.appendChild(frac)

        # Print function
        mrow.appendChild(self._print(e.expr))

        return mrow

    def _print_Function(self, e):
        x = self.dom.createElement('mi')
        if self.mathml_tag(e) == 'log' and self._settings["ln_notation"]:
            x.appendChild(self.dom.createTextNode('ln'))
        else:
            x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(x)
        mrow.appendChild(self._paren_comma_separated(*e.args))
        return mrow

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=True)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_mathml_numbers']
        mrow = self.dom.createElement('mrow')
        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(mant))
            mrow.appendChild(mn)
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode(separator))
            mrow.appendChild(mo)
            msup = self.dom.createElement('msup')
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode("10"))
            msup.appendChild(mn)
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(exp))
            msup.appendChild(mn)
            mrow.appendChild(msup)
            return mrow
        elif str_real == "+inf":
            return self._print_Infinity(None)
        elif str_real == "-inf":
            return self._print_NegativeInfinity(None)
        else:
            mn = self.dom.createElement('mn')
            mn.appendChild(self.dom.createTextNode(str_real))
            return mn

    def _print_polylog(self, expr):
        mrow = self.dom.createElement('mrow')
        m = self.dom.createElement('msub')

        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('Li'))
        m.appendChild(mi)
        m.appendChild(self._print(expr.args[0]))
        mrow.appendChild(m)
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_paren())
        brac.appendChild(self._print(expr.args[1]))
        brac.appendChild(self._r_paren())
        mrow.appendChild(brac)
        return mrow

    def _print_Basic(self, e):
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        mrow.appendChild(self._paren_comma_separated(*e.args))
        return mrow

    def _print_Tuple(self, e):
        return self._paren_comma_separated(*e.args)

    def _print_Interval(self, i):
        right = self.dom.createElement('mo')
        if i.right_open:
            right.appendChild(self.dom.createTextNode(')'))
        else:
            right.appendChild(self.dom.createTextNode(']'))
        left = self.dom.createElement('mo')
        if i.left_open:
            left.appendChild(self.dom.createTextNode('('))
        else:
            left.appendChild(self.dom.createTextNode('['))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(left)
        mrow.appendChild(self._print(i.start))
        mrow.appendChild(self._comma())
        mrow.appendChild(self._print(i.end))
        mrow.appendChild(right)
        return mrow

    def _print_Abs(self, expr, exp=None):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._bar())
        mrow.appendChild(self._print(expr.args[0]))
        mrow.appendChild(self._bar())
        return mrow

    _print_Determinant = _print_Abs

    def _print_re_im(self, c, expr):
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_paren())
        brac.appendChild(self._print(expr))
        brac.appendChild(self._r_paren())
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(c))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(mi)
        mrow.appendChild(brac)
        return mrow

    def _print_re(self, expr, exp=None):
        return self._print_re_im('\u211C', expr.args[0])

    def _print_im(self, expr, exp=None):
        return self._print_re_im('\u2111', expr.args[0])

    def _print_AssocOp(self, e):
        mrow = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(mi)
        for arg in e.args:
            mrow.appendChild(self._print(arg))
        return mrow

    def _print_SetOp(self, expr, symbol, prec):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self.parenthesize(expr.args[0], prec))
        for arg in expr.args[1:]:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(symbol))
            y = self.parenthesize(arg, prec)
            mrow.appendChild(x)
            mrow.appendChild(y)
        return mrow

    def _print_Union(self, expr):
        prec = PRECEDENCE_TRADITIONAL['Union']
        return self._print_SetOp(expr, '&#x222A;', prec)

    def _print_Intersection(self, expr):
        prec = PRECEDENCE_TRADITIONAL['Intersection']
        return self._print_SetOp(expr, '&#x2229;', prec)

    def _print_Complement(self, expr):
        prec = PRECEDENCE_TRADITIONAL['Complement']
        return self._print_SetOp(expr, '&#x2216;', prec)

    def _print_SymmetricDifference(self, expr):
        prec = PRECEDENCE_TRADITIONAL['SymmetricDifference']
        return self._print_SetOp(expr, '&#x2206;', prec)

    def _print_ProductSet(self, expr):
        prec = PRECEDENCE_TRADITIONAL['ProductSet']
        return self._print_SetOp(expr, '&#x00d7;', prec)

    def _print_FiniteSet(self, s):
        return self._print_set(s.args)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_brace())
        for i, item in enumerate(items):
            if i:
                brac.appendChild(self._comma())
            brac.appendChild(self._print(item))
        brac.appendChild(self._r_brace())
        return brac

    _print_frozenset = _print_set

    def _print_LogOp(self, args, symbol):
        mrow = self.dom.createElement('mrow')
        if args[0].is_Boolean and not args[0].is_Not:
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(args[0]))
            brac.appendChild(self._r_paren())
            mrow.appendChild(brac)
        else:
            mrow.appendChild(self._print(args[0]))
        for arg in args[1:]:
            x = self.dom.createElement('mo')
            x.appendChild(self.dom.createTextNode(symbol))
            if arg.is_Boolean and not arg.is_Not:
                y = self.dom.createElement('mrow')
                y.appendChild(self._l_paren())
                y.appendChild(self._print(arg))
                y.appendChild(self._r_paren())
            else:
                y = self._print(arg)
            mrow.appendChild(x)
            mrow.appendChild(y)
        return mrow

    def _print_BasisDependent(self, expr):
        from sympy.vector import Vector

        if expr == expr.zero:
            # Not clear if this is ever called
            return self._print(expr.zero)
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        mrow = self.dom.createElement('mrow')
        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key = lambda x:x[0].__str__())
            for i, (k, v) in enumerate(inneritems):
                if v == 1:
                    if i: # No + for first item
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode('+'))
                        mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
                elif v == -1:
                    mo = self.dom.createElement('mo')
                    mo.appendChild(self.dom.createTextNode('-'))
                    mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
                else:
                    if i: # No + for first item
                        mo = self.dom.createElement('mo')
                        mo.appendChild(self.dom.createTextNode('+'))
                        mrow.appendChild(mo)
                    mbrac = self.dom.createElement('mrow')
                    mbrac.appendChild(self._l_paren())
                    mbrac.appendChild(self._print(v))
                    mbrac.appendChild(self._r_paren())
                    mrow.appendChild(mbrac)
                    mo = self.dom.createElement('mo')
                    mo.appendChild(self.dom.createTextNode('&InvisibleTimes;'))
                    mrow.appendChild(mo)
                    mrow.appendChild(self._print(k))
        return mrow


    def _print_And(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x2227;')

    def _print_Or(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x2228;')

    def _print_Xor(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x22BB;')

    def _print_Implies(self, expr):
        return self._print_LogOp(expr.args, '&#x21D2;')

    def _print_Equivalent(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        return self._print_LogOp(args, '&#x21D4;')

    def _print_Not(self, e):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xAC;'))
        mrow.appendChild(mo)
        if (e.args[0].is_Boolean):
            x = self.dom.createElement('mrow')
            x.appendChild(self._l_paren())
            x.appendChild(self._print(e.args[0]))
            x.appendChild(self._r_paren())
        else:
            x = self._print(e.args[0])
        mrow.appendChild(x)
        return mrow

    def _print_bool(self, e):
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        return mi

    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        return mi

    def _print_Range(self, s):
        dots = "\u2026"
        if s.start.is_infinite and s.stop.is_infinite:
            if s.step.is_positive:
                printset = dots, -1, 0, 1, dots
            else:
                printset = dots, 1, 0, -1, dots
        elif s.start.is_infinite:
            printset = dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite:
            it = iter(s)
            printset = next(it), next(it), dots
        elif len(s) > 4:
            it = iter(s)
            printset = next(it), next(it), dots, s[-1]
        else:
            printset = tuple(s)
        brac = self.dom.createElement('mrow')
        brac.appendChild(self._l_brace())
        for i, el in enumerate(printset):
            if i:
                brac.appendChild(self._comma())
            if el == dots:
                mi = self.dom.createElement('mi')
                mi.appendChild(self.dom.createTextNode(dots))
                brac.appendChild(mi)
            else:
                brac.appendChild(self._print(el))
        brac.appendChild(self._r_brace())
        return brac

    def _hprint_variadic_function(self, expr):
        args = sorted(expr.args, key=default_sort_key)
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode((str(expr.func)).lower()))
        mrow.appendChild(mo)
        mrow.appendChild(self._paren_comma_separated(*args))
        return mrow

    _print_Min = _print_Max = _hprint_variadic_function

    def _print_exp(self, expr):
        msup = self.dom.createElement('msup')
        msup.appendChild(self._print_Exp1(None))
        msup.appendChild(self._print(expr.args[0]))
        return msup

    def _print_Relational(self, e):
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(self._print(e.lhs))
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode(self.mathml_tag(e)))
        mrow.appendChild(x)
        mrow.appendChild(self._print(e.rhs))
        return mrow

    def _print_int(self, p):
        dom_element = self.dom.createElement(self.mathml_tag(p))
        dom_element.appendChild(self.dom.createTextNode(str(p)))
        return dom_element

    def _print_BaseScalar(self, e):
        msub = self.dom.createElement('msub')
        index, system = e._id
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._variable_names[index]))
        msub.appendChild(mi)
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._name))
        msub.appendChild(mi)
        return msub

    def _print_BaseVector(self, e):
        msub = self.dom.createElement('msub')
        index, system = e._id
        mover = self.dom.createElement('mover')
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._vector_names[index]))
        mover.appendChild(mi)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('^'))
        mover.appendChild(mo)
        msub.appendChild(mover)
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode(system._name))
        msub.appendChild(mi)
        return msub

    def _print_VectorZero(self, e):
        mover = self.dom.createElement('mover')
        mi = self.dom.createElement('mi')
        mi.setAttribute('mathvariant', 'bold')
        mi.appendChild(self.dom.createTextNode("0"))
        mover.appendChild(mi)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('^'))
        mover.appendChild(mo)
        return mover

    def _print_Cross(self, expr):
        mrow = self.dom.createElement('mrow')
        vec1 = expr._expr1
        vec2 = expr._expr2
        mrow.appendChild(self.parenthesize(vec1, PRECEDENCE['Mul']))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xD7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(vec2, PRECEDENCE['Mul']))
        return mrow

    def _print_Curl(self, expr):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xD7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Divergence(self, expr):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xB7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Dot(self, expr):
        mrow = self.dom.createElement('mrow')
        vec1 = expr._expr1
        vec2 = expr._expr2
        mrow.appendChild(self.parenthesize(vec1, PRECEDENCE['Mul']))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#xB7;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(vec2, PRECEDENCE['Mul']))
        return mrow

    def _print_Gradient(self, expr):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2207;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Laplacian(self, expr):
        mrow = self.dom.createElement('mrow')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2206;'))
        mrow.appendChild(mo)
        mrow.appendChild(self.parenthesize(expr._expr, PRECEDENCE['Mul']))
        return mrow

    def _print_Integers(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2124;'))
        return x

    def _print_Complexes(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2102;'))
        return x

    def _print_Reals(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x211D;'))
        return x

    def _print_Naturals(self, e):
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2115;'))
        return x

    def _print_Naturals0(self, e):
        sub = self.dom.createElement('msub')
        x = self.dom.createElement('mi')
        x.setAttribute('mathvariant', 'normal')
        x.appendChild(self.dom.createTextNode('&#x2115;'))
        sub.appendChild(x)
        sub.appendChild(self._print(S.Zero))
        return sub

    def _print_SingularityFunction(self, expr):
        shift = expr.args[0] - expr.args[1]
        power = expr.args[2]
        left = self.dom.createElement('mo')
        left.appendChild(self.dom.createTextNode('\u27e8'))
        right = self.dom.createElement('mo')
        right.appendChild(self.dom.createTextNode('\u27e9'))
        brac = self.dom.createElement('mrow')
        brac.appendChild(left)
        brac.appendChild(self._print(shift))
        brac.appendChild(right)
        sup = self.dom.createElement('msup')
        sup.appendChild(brac)
        sup.appendChild(self._print(power))
        return sup

    def _print_NaN(self, e):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('NaN'))
        return x

    def _print_number_function(self, e, name):
        # Print name_arg[0] for one argument or name_arg[0](arg[1])
        # for more than one argument
        sub = self.dom.createElement('msub')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode(name))
        sub.appendChild(mi)
        sub.appendChild(self._print(e.args[0]))
        if len(e.args) == 1:
            return sub
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(sub)
        mrow.appendChild(self._paren_comma_separated(*e.args[1:]))
        return mrow

    def _print_bernoulli(self, e):
        return self._print_number_function(e, 'B')

    _print_bell = _print_bernoulli

    def _print_catalan(self, e):
        return self._print_number_function(e, 'C')

    def _print_euler(self, e):
        return self._print_number_function(e, 'E')

    def _print_fibonacci(self, e):
        return self._print_number_function(e, 'F')

    def _print_lucas(self, e):
        return self._print_number_function(e, 'L')

    def _print_stieltjes(self, e):
        return self._print_number_function(e, '&#x03B3;')

    def _print_tribonacci(self, e):
        return self._print_number_function(e, 'T')

    def _print_ComplexInfinity(self, e):
        x = self.dom.createElement('mover')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x221E;'))
        x.appendChild(mo)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('~'))
        x.appendChild(mo)
        return x

    def _print_EmptySet(self, e):
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode('&#x2205;'))
        return x

    def _print_UniversalSet(self, e):
        x = self.dom.createElement('mo')
        x.appendChild(self.dom.createTextNode('&#x1D54C;'))
        return x

    def _print_Adjoint(self, expr):
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(mat))
            brac.appendChild(self._r_paren())
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x2020;'))
        sup.appendChild(mo)
        return sup

    def _print_Transpose(self, expr):
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(mat))
            brac.appendChild(self._r_paren())
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('T'))
        sup.appendChild(mo)
        return sup

    def _print_Inverse(self, expr):
        from sympy.matrices import MatrixSymbol
        mat = expr.arg
        sup = self.dom.createElement('msup')
        if not isinstance(mat, MatrixSymbol):
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(mat))
            brac.appendChild(self._r_paren())
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(mat))
        sup.appendChild(self._print(-1))
        return sup

    def _print_MatMul(self, expr):
        from sympy.matrices.expressions.matmul import MatMul

        x = self.dom.createElement('mrow')
        args = expr.args
        if isinstance(args[0], Mul):
            args = args[0].as_ordered_factors() + list(args[1:])
        else:
            args = list(args)

        if isinstance(expr, MatMul) and expr.could_extract_minus_sign():
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('-'))
            x.appendChild(mo)

        for arg in args[:-1]:
            x.appendChild(self.parenthesize(arg, precedence_traditional(expr),
                                            False))
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('&InvisibleTimes;'))
            x.appendChild(mo)
        x.appendChild(self.parenthesize(args[-1], precedence_traditional(expr),
                                        False))
        return x

    def _print_MatPow(self, expr):
        from sympy.matrices import MatrixSymbol
        base, exp = expr.base, expr.exp
        sup = self.dom.createElement('msup')
        if not isinstance(base, MatrixSymbol):
            brac = self.dom.createElement('mrow')
            brac.appendChild(self._l_paren())
            brac.appendChild(self._print(base))
            brac.appendChild(self._r_paren())
            sup.appendChild(brac)
        else:
            sup.appendChild(self._print(base))
        sup.appendChild(self._print(exp))
        return sup

    def _print_HadamardProduct(self, expr):
        x = self.dom.createElement('mrow')
        args = expr.args
        for arg in args[:-1]:
            x.appendChild(
                self.parenthesize(arg, precedence_traditional(expr), False))
            mo = self.dom.createElement('mo')
            mo.appendChild(self.dom.createTextNode('&#x2218;'))
            x.appendChild(mo)
        x.appendChild(
            self.parenthesize(args[-1], precedence_traditional(expr), False))
        return x

    def _print_ZeroMatrix(self, Z):
        x = self.dom.createElement('mn')
        x.appendChild(self.dom.createTextNode('&#x1D7D8'))
        return x

    def _print_OneMatrix(self, Z):
        x = self.dom.createElement('mn')
        x.appendChild(self.dom.createTextNode('&#x1D7D9'))
        return x

    def _print_Identity(self, I):
        x = self.dom.createElement('mi')
        x.appendChild(self.dom.createTextNode('&#x1D540;'))
        return x

    def _print_floor(self, e):
        left = self.dom.createElement('mo')
        left.appendChild(self.dom.createTextNode('\u230A'))
        right = self.dom.createElement('mo')
        right.appendChild(self.dom.createTextNode('\u230B'))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(left)
        mrow.appendChild(self._print(e.args[0]))
        mrow.appendChild(right)
        return mrow

    def _print_ceiling(self, e):
        left = self.dom.createElement('mo')
        left.appendChild(self.dom.createTextNode('\u2308'))
        right = self.dom.createElement('mo')
        right.appendChild(self.dom.createTextNode('\u2309'))
        mrow = self.dom.createElement('mrow')
        mrow.appendChild(left)
        mrow.appendChild(self._print(e.args[0]))
        mrow.appendChild(right)
        return mrow

    def _print_Lambda(self, e):
        mrow = self.dom.createElement('mrow')
        symbols = e.args[0]
        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(symbols)
        mrow.appendChild(self._l_paren())
        mrow.appendChild(symbols)
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('&#x21A6;'))
        mrow.appendChild(mo)
        mrow.appendChild(self._print(e.args[1]))
        mrow.appendChild(self._r_paren())
        return mrow

    def _print_tuple(self, e):
        return self._paren_comma_separated(*e)

    def _print_IndexedBase(self, e):
        return self._print(e.label)

    def _print_Indexed(self, e):
        x = self.dom.createElement('msub')
        x.appendChild(self._print(e.base))
        if len(e.indices) == 1:
            x.appendChild(self._print(e.indices[0]))
            return x
        x.appendChild(self._print(e.indices))
        return x

    def _print_MatrixElement(self, e):
        x = self.dom.createElement('msub')
        x.appendChild(self.parenthesize(e.parent, PRECEDENCE["Atom"], strict = True))
        brac = self.dom.createElement('mrow')
        for i, arg in enumerate(e.indices):
            if i:
                brac.appendChild(self._comma())
            brac.appendChild(self._print(arg))
        x.appendChild(brac)
        return x

    def _print_elliptic_f(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d5a5;'))
        x.appendChild(mi)
        x.appendChild(self._paren_bar_separated(*e.args))
        return x

    def _print_elliptic_e(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d5a4;'))
        x.appendChild(mi)
        x.appendChild(self._paren_bar_separated(*e.args))
        return x

    def _print_elliptic_pi(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('&#x1d6f1;'))
        x.appendChild(mi)
        y = self.dom.createElement('mrow')
        y.appendChild(self._l_paren())
        if len(e.args) == 2:
            n, m = e.args
            y.appendChild(self._print(n))
            y.appendChild(self._bar())
            y.appendChild(self._print(m))
        else:
            n, m, z = e.args
            y.appendChild(self._print(n))
            y.appendChild(self._semicolon())
            y.appendChild(self._print(m))
            y.appendChild(self._bar())
            y.appendChild(self._print(z))
        y.appendChild(self._r_paren())
        x.appendChild(y)
        return x

    def _print_Ei(self, e):
        x = self.dom.createElement('mrow')
        mi = self.dom.createElement('mi')
        mi.appendChild(self.dom.createTextNode('Ei'))
        x.appendChild(mi)
        x.appendChild(self._print(e.args))
        return x

    def _print_expint(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('E'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_jacobi(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:3]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[3:]))
        return x

    def _print_gegenbauer(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('C'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_chebyshevt(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('T'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_chebyshevu(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('U'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_legendre(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_assoc_legendre(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('P'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_laguerre(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('L'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x

    def _print_assoc_laguerre(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msubsup')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('L'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        y.appendChild(self._print(e.args[1:2]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[2:]))
        return x

    def _print_hermite(self, e):
        x = self.dom.createElement('mrow')
        y = self.dom.createElement('msub')
        mo = self.dom.createElement('mo')
        mo.appendChild(self.dom.createTextNode('H'))
        y.appendChild(mo)
        y.appendChild(self._print(e.args[0]))
        x.appendChild(y)
        x.appendChild(self._print(e.args[1:]))
        return x


@print_function(MathMLPrinterBase)
def mathml(expr, printer='content', **settings):
    """Returns the MathML representation of expr. If printer is presentation
    then prints Presentation MathML else prints content MathML.
    """
    if printer == 'presentation':
        return MathMLPresentationPrinter(settings).doprint(expr)
    else:
        return MathMLContentPrinter(settings).doprint(expr)


def print_mathml(expr, printer='content', **settings):
    """
    Prints a pretty representation of the MathML code for expr. If printer is
    presentation then prints Presentation MathML else prints content MathML.

    Examples
    ========

    >>> ##
    >>> from sympy import print_mathml
    >>> from sympy.abc import x
    >>> print_mathml(x+1) #doctest: +NORMALIZE_WHITESPACE
    <apply>
        <plus/>
        <ci>x</ci>
        <cn>1</cn>
    </apply>
    >>> print_mathml(x+1, printer='presentation')
    <mrow>
        <mi>x</mi>
        <mo>+</mo>
        <mn>1</mn>
    </mrow>

    """
    if printer == 'presentation':
        s = MathMLPresentationPrinter(settings)
    else:
        s = MathMLContentPrinter(settings)
    xml = s._print(sympify(expr))
    pretty_xml = xml.toprettyxml()

    print(pretty_xml)


# For backward compatibility
MathMLPrinter = MathMLContentPrinter

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\spin.py ===
"""Quantum mechanical angular momentum."""

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.numbers import int_valued
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.simplify import simplify
from sympy.matrices import zeros
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import pretty_symbol

from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.operator import (HermitianOperator, Operator,
                                            UnitaryOperator)
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import ComplexSpace, DirectSumHilbertSpace
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.qapply import qapply


__all__ = [
    'm_values',
    'Jplus',
    'Jminus',
    'Jx',
    'Jy',
    'Jz',
    'J2',
    'Rotation',
    'WignerD',
    'JxKet',
    'JxBra',
    'JyKet',
    'JyBra',
    'JzKet',
    'JzBra',
    'JzOp',
    'J2Op',
    'JxKetCoupled',
    'JxBraCoupled',
    'JyKetCoupled',
    'JyBraCoupled',
    'JzKetCoupled',
    'JzBraCoupled',
    'couple',
    'uncouple'
]


def m_values(j):
    j = sympify(j)
    size = 2*j + 1
    if not size.is_Integer or not size > 0:
        raise ValueError(
            'Only integer or half-integer values allowed for j, got: : %r' % j
        )
    return size, [j - i for i in range(int(2*j + 1))]


#-----------------------------------------------------------------------------
# Spin Operators
#-----------------------------------------------------------------------------


class SpinOpBase:
    """Base class for spin operators."""

    @classmethod
    def _eval_hilbert_space(cls, label):
        # We consider all j values so our space is infinite.
        return ComplexSpace(S.Infinity)

    @property
    def name(self):
        return self.args[0]

    def _print_contents(self, printer, *args):
        return '%s%s' % (self.name, self._coord)

    def _print_contents_pretty(self, printer, *args):
        a = stringPict(str(self.name))
        b = stringPict(self._coord)
        return self._print_subscript_pretty(a, b)

    def _print_contents_latex(self, printer, *args):
        return r'%s_%s' % ((self.name, self._coord))

    def _represent_base(self, basis, **options):
        j = options.get('j', S.Half)
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                result[p, q] = me
        return result

    def _apply_op(self, ket, orig_basis, **options):
        state = ket.rewrite(self.basis)
        # If the state has only one term
        if isinstance(state, State):
            ret = (hbar*state.m)*state
        # state is a linear combination of states
        elif isinstance(state, Sum):
            ret = self._apply_operator_Sum(state, **options)
        else:
            ret = qapply(self*state)
        if ret == self*state:
            raise NotImplementedError
        return ret.rewrite(orig_basis)

    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jx', **options)

    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jy', **options)

    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_op(ket, 'Jz', **options)

    def _apply_operator_TensorProduct(self, tp, **options):
        # Uncoupling operator is only easily found for coordinate basis spin operators
        # TODO: add methods for uncoupling operators
        if not isinstance(self, (JxOp, JyOp, JzOp)):
            raise NotImplementedError
        result = []
        for n in range(len(tp.args)):
            arg = []
            arg.extend(tp.args[:n])
            arg.append(self._apply_operator(tp.args[n]))
            arg.extend(tp.args[n + 1:])
            result.append(tp.__class__(*arg))
        return Add(*result).expand()

    # TODO: move this to qapply_Mul
    def _apply_operator_Sum(self, s, **options):
        new_func = qapply(self*s.function)
        if new_func == self*s.function:
            raise NotImplementedError
        return Sum(new_func, *s.limits)

    def _eval_trace(self, **options):
        #TODO: use options to use different j values
        #For now eval at default basis

        # is it efficient to represent each time
        # to do a trace?
        return self._represent_default_basis().trace()


class JplusOp(SpinOpBase, Operator):
    """The J+ operator."""

    _coord = '+'

    basis = 'Jz'

    def _eval_commutator_JminusOp(self, other):
        return 2*hbar*JzOp(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*JzKet(j, m + S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if m.is_Number and j.is_Number:
            if m >= j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m + S.One))*JzKetCoupled(j, m + S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*sqrt(j*(j + S.One) - mp*(mp + S.One))
        result *= KroneckerDelta(m, mp + 1)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) + I*JyOp(args[0])


class JminusOp(SpinOpBase, Operator):
    """The J- operator."""

    _coord = '-'

    basis = 'Jz'

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        m = ket.m
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*JzKet(j, m - S.One)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if m.is_Number and j.is_Number:
            if m <= -j:
                return S.Zero
        return hbar*sqrt(j*(j + S.One) - m*(m - S.One))*JzKetCoupled(j, m - S.One, jn, coupling)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*sqrt(j*(j + S.One) - mp*(mp - S.One))
        result *= KroneckerDelta(m, mp - 1)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0]) - I*JyOp(args[0])


class JxOp(SpinOpBase, HermitianOperator):
    """The Jx operator."""

    _coord = 'x'

    basis = 'Jx'

    def _eval_commutator_JyOp(self, other):
        return I*hbar*JzOp(self.name)

    def _eval_commutator_JzOp(self, other):
        return -I*hbar*JyOp(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKet(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKet(ket, **options)
        return (jp + jm)/Integer(2)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        return (jp + jm)/Integer(2)

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        jp = JplusOp(self.name)._represent_JzOp(basis, **options)
        jm = JminusOp(self.name)._represent_JzOp(basis, **options)
        return (jp + jm)/Integer(2)

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        return (JplusOp(args[0]) + JminusOp(args[0]))/2


class JyOp(SpinOpBase, HermitianOperator):
    """The Jy operator."""

    _coord = 'y'

    basis = 'Jy'

    def _eval_commutator_JzOp(self, other):
        return I*hbar*JxOp(self.name)

    def _eval_commutator_JxOp(self, other):
        return -I*hbar*J2Op(self.name)

    def _apply_operator_JzKet(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKet(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKet(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        jp = JplusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        jm = JminusOp(self.name)._apply_operator_JzKetCoupled(ket, **options)
        return (jp - jm)/(Integer(2)*I)

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        jp = JplusOp(self.name)._represent_JzOp(basis, **options)
        jm = JminusOp(self.name)._represent_JzOp(basis, **options)
        return (jp - jm)/(Integer(2)*I)

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        return (JplusOp(args[0]) - JminusOp(args[0]))/(2*I)


class JzOp(SpinOpBase, HermitianOperator):
    """The Jz operator."""

    _coord = 'z'

    basis = 'Jz'

    def _eval_commutator_JxOp(self, other):
        return I*hbar*JyOp(self.name)

    def _eval_commutator_JyOp(self, other):
        return -I*hbar*JxOp(self.name)

    def _eval_commutator_JplusOp(self, other):
        return hbar*JplusOp(self.name)

    def _eval_commutator_JminusOp(self, other):
        return -hbar*JminusOp(self.name)

    def matrix_element(self, j, m, jp, mp):
        result = hbar*mp
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)


class J2Op(SpinOpBase, HermitianOperator):
    """The J^2 operator."""

    _coord = '2'

    def _eval_commutator_JxOp(self, other):
        return S.Zero

    def _eval_commutator_JyOp(self, other):
        return S.Zero

    def _eval_commutator_JzOp(self, other):
        return S.Zero

    def _eval_commutator_JplusOp(self, other):
        return S.Zero

    def _eval_commutator_JminusOp(self, other):
        return S.Zero

    def _apply_operator_JxKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JxKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JyKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JyKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JzKet(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def _apply_operator_JzKetCoupled(self, ket, **options):
        j = ket.j
        return hbar**2*j*(j + 1)*ket

    def matrix_element(self, j, m, jp, mp):
        result = (hbar**2)*j*(j + 1)
        result *= KroneckerDelta(m, mp)
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _print_contents_pretty(self, printer, *args):
        a = prettyForm(str(self.name))
        b = prettyForm('2')
        return a**b

    def _print_contents_latex(self, printer, *args):
        return r'%s^2' % str(self.name)

    def _eval_rewrite_as_xyz(self, *args, **kwargs):
        return JxOp(args[0])**2 + JyOp(args[0])**2 + JzOp(args[0])**2

    def _eval_rewrite_as_plusminus(self, *args, **kwargs):
        a = args[0]
        return JzOp(a)**2 + \
            S.Half*(JplusOp(a)*JminusOp(a) + JminusOp(a)*JplusOp(a))


class Rotation(UnitaryOperator):
    """Wigner D operator in terms of Euler angles.

    Defines the rotation operator in terms of the Euler angles defined by
    the z-y-z convention for a passive transformation. That is the coordinate
    axes are rotated first about the z-axis, giving the new x'-y'-z' axes. Then
    this new coordinate system is rotated about the new y'-axis, giving new
    x''-y''-z'' axes. Then this new coordinate system is rotated about the
    z''-axis. Conventions follow those laid out in [1]_.

    Parameters
    ==========

    alpha : Number, Symbol
        First Euler Angle
    beta : Number, Symbol
        Second Euler angle
    gamma : Number, Symbol
        Third Euler angle

    Examples
    ========

    A simple example rotation operator:

        >>> from sympy import pi
        >>> from sympy.physics.quantum.spin import Rotation
        >>> Rotation(pi, 0, pi/2)
        R(pi,0,pi/2)

    With symbolic Euler angles and calculating the inverse rotation operator:

        >>> from sympy import symbols
        >>> a, b, c = symbols('a b c')
        >>> Rotation(a, b, c)
        R(a,b,c)
        >>> Rotation(a, b, c).inverse()
        R(-c,-b,-a)

    See Also
    ========

    WignerD: Symbolic Wigner-D function
    D: Wigner-D function
    d: Wigner small-d function

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    @classmethod
    def _eval_args(cls, args):
        args = QExpr._eval_args(args)
        if len(args) != 3:
            raise ValueError('3 Euler angles required, got: %r' % args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, label):
        # We consider all j values so our space is infinite.
        return ComplexSpace(S.Infinity)

    @property
    def alpha(self):
        return self.label[0]

    @property
    def beta(self):
        return self.label[1]

    @property
    def gamma(self):
        return self.label[2]

    def _print_operator_name(self, printer, *args):
        return 'R'

    def _print_operator_name_pretty(self, printer, *args):
        if printer._use_unicode:
            return prettyForm('\N{SCRIPT CAPITAL R}' + ' ')
        else:
            return prettyForm("R ")

    def _print_operator_name_latex(self, printer, *args):
        return r'\mathcal{R}'

    def _eval_inverse(self):
        return Rotation(-self.gamma, -self.beta, -self.alpha)

    @classmethod
    def D(cls, j, m, mp, alpha, beta, gamma):
        """Wigner D-function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        alpha : Number, Symbol
            First Euler angle of rotation
        beta : Number, Symbol
            Second Euler angle of rotation
        gamma : Number, Symbol
            Third Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> alpha, beta, gamma = symbols('alpha beta gamma')
            >>> Rotation.D(1, 1, 0,pi, pi/2,-pi)
            WignerD(1, 1, 0, pi, pi/2, -pi)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        return WignerD(j, m, mp, alpha, beta, gamma)

    @classmethod
    def d(cls, j, m, mp, beta):
        """Wigner small-d function.

        Returns an instance of the WignerD class corresponding to the Wigner-D
        function specified by the parameters with the alpha and gamma angles
        given as 0.

        Parameters
        ===========

        j : Number
            Total angular momentum
        m : Number
            Eigenvalue of angular momentum along axis after rotation
        mp : Number
            Eigenvalue of angular momentum along rotated axis
        beta : Number, Symbol
            Second Euler angle of rotation

        Examples
        ========

        Return the Wigner-D matrix element for a defined rotation, both
        numerical and symbolic:

            >>> from sympy.physics.quantum.spin import Rotation
            >>> from sympy import pi, symbols
            >>> beta = symbols('beta')
            >>> Rotation.d(1, 1, 0, pi/2)
            WignerD(1, 1, 0, 0, pi/2, 0)

        See Also
        ========

        WignerD: Symbolic Wigner-D function

        """
        return WignerD(j, m, mp, 0, beta, 0)

    def matrix_element(self, j, m, jp, mp):
        result = self.__class__.D(
            jp, m, mp, self.alpha, self.beta, self.gamma
        )
        result *= KroneckerDelta(j, jp)
        return result

    def _represent_base(self, basis, **options):
        j = sympify(options.get('j', S.Half))
        # TODO: move evaluation up to represent function/implement elsewhere
        evaluate = sympify(options.get('doit'))
        size, mvals = m_values(j)
        result = zeros(size, size)
        for p in range(size):
            for q in range(size):
                me = self.matrix_element(j, mvals[p], j, mvals[q])
                if evaluate:
                    result[p, q] = me.doit()
                else:
                    result[p, q] = me
        return result

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(basis, **options)

    def _apply_operator_uncoupled(self, state, ket, *, dummy=True, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z*state(j, mp))
            return Add(*s)
        else:
            if dummy:
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g)*state(j, mp), (mp, -j, j))

    def _apply_operator_JxKet(self, ket, **options):
        return self._apply_operator_uncoupled(JxKet, ket, **options)

    def _apply_operator_JyKet(self, ket, **options):
        return self._apply_operator_uncoupled(JyKet, ket, **options)

    def _apply_operator_JzKet(self, ket, **options):
        return self._apply_operator_uncoupled(JzKet, ket, **options)

    def _apply_operator_coupled(self, state, ket, *, dummy=True, **options):
        a = self.alpha
        b = self.beta
        g = self.gamma
        j = ket.j
        m = ket.m
        jn = ket.jn
        coupling = ket.coupling
        if j.is_number:
            s = []
            size = m_values(j)
            sz = size[1]
            for mp in sz:
                r = Rotation.D(j, m, mp, a, b, g)
                z = r.doit()
                s.append(z*state(j, mp, jn, coupling))
            return Add(*s)
        else:
            if dummy:
                mp = Dummy('mp')
            else:
                mp = symbols('mp')
            return Sum(Rotation.D(j, m, mp, a, b, g)*state(
                j, mp, jn, coupling), (mp, -j, j))

    def _apply_operator_JxKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JxKetCoupled, ket, **options)

    def _apply_operator_JyKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JyKetCoupled, ket, **options)

    def _apply_operator_JzKetCoupled(self, ket, **options):
        return self._apply_operator_coupled(JzKetCoupled, ket, **options)

class WignerD(Expr):
    r"""Wigner-D function

    The Wigner D-function gives the matrix elements of the rotation
    operator in the jm-representation. For the Euler angles `\alpha`,
    `\beta`, `\gamma`, the D-function is defined such that:

    .. math ::
        <j,m| \mathcal{R}(\alpha, \beta, \gamma ) |j',m'> = \delta_{jj'} D(j, m, m', \alpha, \beta, \gamma)

    Where the rotation operator is as defined by the Rotation class [1]_.

    The Wigner D-function defined in this way gives:

    .. math ::
        D(j, m, m', \alpha, \beta, \gamma) = e^{-i m \alpha} d(j, m, m', \beta) e^{-i m' \gamma}

    Where d is the Wigner small-d function, which is given by Rotation.d.

    The Wigner small-d function gives the component of the Wigner
    D-function that is determined by the second Euler angle. That is the
    Wigner D-function is:

    .. math ::
        D(j, m, m', \alpha, \beta, \gamma) = e^{-i m \alpha} d(j, m, m', \beta) e^{-i m' \gamma}

    Where d is the small-d function. The Wigner D-function is given by
    Rotation.D.

    Note that to evaluate the D-function, the j, m and mp parameters must
    be integer or half integer numbers.

    Parameters
    ==========

    j : Number
        Total angular momentum
    m : Number
        Eigenvalue of angular momentum along axis after rotation
    mp : Number
        Eigenvalue of angular momentum along rotated axis
    alpha : Number, Symbol
        First Euler angle of rotation
    beta : Number, Symbol
        Second Euler angle of rotation
    gamma : Number, Symbol
        Third Euler angle of rotation

    Examples
    ========

    Evaluate the Wigner-D matrix elements of a simple rotation:

        >>> from sympy.physics.quantum.spin import Rotation
        >>> from sympy import pi
        >>> rot = Rotation.D(1, 1, 0, pi, pi/2, 0)
        >>> rot
        WignerD(1, 1, 0, pi, pi/2, 0)
        >>> rot.doit()
        sqrt(2)/2

    Evaluate the Wigner-d matrix elements of a simple rotation

        >>> rot = Rotation.d(1, 1, 0, pi/2)
        >>> rot
        WignerD(1, 1, 0, 0, pi/2, 0)
        >>> rot.doit()
        -sqrt(2)/2

    See Also
    ========

    Rotation: Rotation operator

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    is_commutative = True

    def __new__(cls, *args, **hints):
        if not len(args) == 6:
            raise ValueError('6 parameters expected, got %s' % args)
        args = sympify(args)
        evaluate = hints.get('evaluate', False)
        if evaluate:
            return Expr.__new__(cls, *args)._eval_wignerd()
        return Expr.__new__(cls, *args)

    @property
    def j(self):
        return self.args[0]

    @property
    def m(self):
        return self.args[1]

    @property
    def mp(self):
        return self.args[2]

    @property
    def alpha(self):
        return self.args[3]

    @property
    def beta(self):
        return self.args[4]

    @property
    def gamma(self):
        return self.args[5]

    def _latex(self, printer, *args):
        if self.alpha == 0 and self.gamma == 0:
            return r'd^{%s}_{%s,%s}\left(%s\right)' % \
                (
                    printer._print(self.j), printer._print(
                        self.m), printer._print(self.mp),
                    printer._print(self.beta) )
        return r'D^{%s}_{%s,%s}\left(%s,%s,%s\right)' % \
            (
                printer._print(
                    self.j), printer._print(self.m), printer._print(self.mp),
                printer._print(self.alpha), printer._print(self.beta), printer._print(self.gamma) )

    def _pretty(self, printer, *args):
        top = printer._print(self.j)

        bot = printer._print(self.m)
        bot = prettyForm(*bot.right(','))
        bot = prettyForm(*bot.right(printer._print(self.mp)))

        pad = max(top.width(), bot.width())
        top = prettyForm(*top.left(' '))
        bot = prettyForm(*bot.left(' '))
        if pad > top.width():
            top = prettyForm(*top.right(' '*(pad - top.width())))
        if pad > bot.width():
            bot = prettyForm(*bot.right(' '*(pad - bot.width())))
        if self.alpha == 0 and self.gamma == 0:
            args = printer._print(self.beta)
            s = stringPict('d' + ' '*pad)
        else:
            args = printer._print(self.alpha)
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.beta)))
            args = prettyForm(*args.right(','))
            args = prettyForm(*args.right(printer._print(self.gamma)))

            s = stringPict('D' + ' '*pad)

        args = prettyForm(*args.parens())
        s = prettyForm(*s.above(top))
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.right(args))
        return s

    def doit(self, **hints):
        hints['evaluate'] = True
        return WignerD(*self.args, **hints)

    def _eval_wignerd(self):
        j = self.j
        m = self.m
        mp = self.mp
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        if alpha == 0 and beta == 0 and gamma == 0:
            return KroneckerDelta(m, mp)
        if not j.is_number:
            raise ValueError(
                'j parameter must be numerical to evaluate, got %s' % j)
        r = 0
        if beta == pi/2:
            # Varshalovich Equation (5), Section 4.16, page 113, setting
            # alpha=gamma=0.
            for k in range(2*j + 1):
                if k > j + mp or k > j - m or k < mp - m:
                    continue
                r += (S.NegativeOne)**k*binomial(j + mp, k)*binomial(j - mp, k + m - mp)
            r *= (S.NegativeOne)**(m - mp) / 2**j*sqrt(factorial(j + m) *
                    factorial(j - m) / (factorial(j + mp)*factorial(j - mp)))
        else:
            # Varshalovich Equation(5), Section 4.7.2, page 87, where we set
            # beta1=beta2=pi/2, and we get alpha=gamma=pi/2 and beta=phi+pi,
            # then we use the Eq. (1), Section 4.4. page 79, to simplify:
            # d(j, m, mp, beta+pi) = (-1)**(j-mp)*d(j, m, -mp, beta)
            # This happens to be almost the same as in Eq.(10), Section 4.16,
            # except that we need to substitute -mp for mp.
            size, mvals = m_values(j)
            for mpp in mvals:
                r += Rotation.d(j, m, mpp, pi/2).doit()*(cos(-mpp*beta) + I*sin(-mpp*beta))*\
                    Rotation.d(j, mpp, -mp, pi/2).doit()
            # Empirical normalization factor so results match Varshalovich
            # Tables 4.3-4.12
            # Note that this exact normalization does not follow from the
            # above equations
            r = r*I**(2*j - m - mp)*(-1)**(2*m)
            # Finally, simplify the whole expression
            r = simplify(r)
        r *= exp(-I*m*alpha)*exp(-I*mp*gamma)
        return r


Jx = JxOp('J')
Jy = JyOp('J')
Jz = JzOp('J')
J2 = J2Op('J')
Jplus = JplusOp('J')
Jminus = JminusOp('J')


#-----------------------------------------------------------------------------
# Spin States
#-----------------------------------------------------------------------------


class SpinState(State):
    """Base class for angular momentum states."""

    _label_separator = ','

    def __new__(cls, j, m):
        j = sympify(j)
        m = sympify(m)
        if j.is_number:
            if 2*j != int(2*j):
                raise ValueError(
                    'j must be integer or half-integer, got: %s' % j)
            if j < 0:
                raise ValueError('j must be >= 0, got: %s' % j)
        if m.is_number:
            if 2*m != int(2*m):
                raise ValueError(
                    'm must be integer or half-integer, got: %s' % m)
        if j.is_number and m.is_number:
            if abs(m) > j:
                raise ValueError('Allowed values for m are -j <= m <= j, got j, m: %s, %s' % (j, m))
            if int(j - m) != j - m:
                raise ValueError('Both j and m must be integer or half-integer, got j, m: %s, %s' % (j, m))
        return State.__new__(cls, j, m)

    @property
    def j(self):
        return self.label[0]

    @property
    def m(self):
        return self.label[1]

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2*label[0] + 1)

    def _represent_base(self, **options):
        j = self.j
        m = self.m
        alpha = sympify(options.get('alpha', 0))
        beta = sympify(options.get('beta', 0))
        gamma = sympify(options.get('gamma', 0))
        size, mvals = m_values(j)
        result = zeros(size, 1)
        # breaks finding angles on L930
        for p, mval in enumerate(mvals):
            if m.is_number:
                result[p, 0] = Rotation.D(
                    self.j, mval, self.m, alpha, beta, gamma).doit()
            else:
                result[p, 0] = Rotation.D(self.j, mval,
                                          self.m, alpha, beta, gamma)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBra, **options)
        return self._rewrite_basis(Jx, JxKet, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBra, **options)
        return self._rewrite_basis(Jy, JyKet, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBra, **options)
        return self._rewrite_basis(Jz, JzKet, **options)

    def _rewrite_basis(self, basis, evect, **options):
        from sympy.physics.quantum.represent import represent
        j = self.j
        args = self.args[2:]
        if j.is_number:
            if isinstance(self, CoupledSpinState):
                if j == int(j):
                    start = j**2
                else:
                    start = (2*j - 1)*(2*j + 1)/4
            else:
                start = 0
            vect = represent(self, basis=basis, **options)
            result = Add(
                *[vect[start + i]*evect(j, j - i, *args) for i in range(2*j + 1)])
            if isinstance(self, CoupledSpinState) and options.get('coupled') is False:
                return uncouple(result)
            return result
        else:
            i = 0
            mi = symbols('mi')
            # make sure not to introduce a symbol already in the state
            while self.subs(mi, 0) != self:
                i += 1
                mi = symbols('mi%d' % i)
                break
            # TODO: better way to get angles of rotation
            if isinstance(self, CoupledSpinState):
                test_args = (0, mi, (0, 0))
            else:
                test_args = (0, mi)
            if isinstance(self, Ket):
                angles = represent(
                    self.__class__(*test_args), basis=basis)[0].args[3:6]
            else:
                angles = represent(self.__class__(
                    *test_args), basis=basis)[0].args[0].args[3:6]
            if angles == (0, 0, 0):
                return self
            else:
                state = evect(j, mi, *args)
                lt = Rotation.D(j, mi, self.m, *angles)
                return Sum(lt*state, (mi, -j, j))

    def _eval_innerproduct_JxBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JxOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_JyBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JyOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_innerproduct_JzBra(self, bra, **hints):
        result = KroneckerDelta(self.j, bra.j)
        if bra.dual_class() is not self.__class__:
            result *= self._represent_JzOp(None)[bra.j - bra.m]
        else:
            result *= KroneckerDelta(
                self.j, bra.j)*KroneckerDelta(self.m, bra.m)
        return result

    def _eval_trace(self, bra, **hints):

        # One way to implement this method is to assume the basis set k is
        # passed.
        # Then we can apply the discrete form of Trace formula here
        # Tr(|i><j| ) = \Sum_k <k|i><j|k>
        #then we do qapply() on each each inner product and sum over them.

        # OR

        # Inner product of |i><j| = Trace(Outer Product).
        # we could just use this unless there are cases when this is not true

        return (bra*self).doit()


class JxKet(SpinState, Ket):
    """Eigenket of Jx.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JxBra

    @classmethod
    def coupled_class(self):
        return JxKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JxOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(**options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(beta=pi/2, **options)


class JxBra(SpinState, Bra):
    """Eigenbra of Jx.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JxKet

    @classmethod
    def coupled_class(self):
        return JxBraCoupled


class JyKet(SpinState, Ket):
    """Eigenket of Jy.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JyBra

    @classmethod
    def coupled_class(self):
        return JyKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JyOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(**options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), beta=-pi/2, gamma=pi/2, **options)


class JyBra(SpinState, Bra):
    """Eigenbra of Jy.

    See JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JyKet

    @classmethod
    def coupled_class(self):
        return JyBraCoupled


class JzKet(SpinState, Ket):
    """Eigenket of Jz.

    Spin state which is an eigenstate of the Jz operator. Uncoupled states,
    that is states representing the interaction of multiple separate spin
    states, are defined as a tensor product of states.

    Parameters
    ==========

    j : Number, Symbol
        Total spin angular momentum
    m : Number, Symbol
        Eigenvalue of the Jz spin operator

    Examples
    ========

    *Normal States:*

    Defining simple spin states, both numerical and symbolic:

        >>> from sympy.physics.quantum.spin import JzKet, JxKet
        >>> from sympy import symbols
        >>> JzKet(1, 0)
        |1,0>
        >>> j, m = symbols('j m')
        >>> JzKet(j, m)
        |j,m>

    Rewriting the JzKet in terms of eigenkets of the Jx operator:
    Note: that the resulting eigenstates are JxKet's

        >>> JzKet(1,1).rewrite("Jx")
        |1,-1>/2 - sqrt(2)*|1,0>/2 + |1,1>/2

    Get the vector representation of a state in terms of the basis elements
    of the Jx operator:

        >>> from sympy.physics.quantum.represent import represent
        >>> from sympy.physics.quantum.spin import Jx, Jz
        >>> represent(JzKet(1,-1), basis=Jx)
        Matrix([
        [      1/2],
        [sqrt(2)/2],
        [      1/2]])

    Apply innerproducts between states:

        >>> from sympy.physics.quantum.innerproduct import InnerProduct
        >>> from sympy.physics.quantum.spin import JxBra
        >>> i = InnerProduct(JxBra(1,1), JzKet(1,1))
        >>> i
        <1,1|1,1>
        >>> i.doit()
        1/2

    *Uncoupled States:*

    Define an uncoupled state as a TensorProduct between two Jz eigenkets:

        >>> from sympy.physics.quantum.tensorproduct import TensorProduct
        >>> j1,m1,j2,m2 = symbols('j1 m1 j2 m2')
        >>> TensorProduct(JzKet(1,0), JzKet(1,1))
        |1,0>x|1,1>
        >>> TensorProduct(JzKet(j1,m1), JzKet(j2,m2))
        |j1,m1>x|j2,m2>

    A TensorProduct can be rewritten, in which case the eigenstates that make
    up the tensor product is rewritten to the new basis:

        >>> TensorProduct(JzKet(1,1),JxKet(1,1)).rewrite('Jz')
        |1,1>x|1,-1>/2 + sqrt(2)*|1,1>x|1,0>/2 + |1,1>x|1,1>/2

    The represent method for TensorProduct's gives the vector representation of
    the state. Note that the state in the product basis is the equivalent of the
    tensor product of the vector representation of the component eigenstates:

        >>> represent(TensorProduct(JzKet(1,0),JzKet(1,1)))
        Matrix([
        [0],
        [0],
        [0],
        [1],
        [0],
        [0],
        [0],
        [0],
        [0]])
        >>> represent(TensorProduct(JzKet(1,1),JxKet(1,1)), basis=Jz)
        Matrix([
        [      1/2],
        [sqrt(2)/2],
        [      1/2],
        [        0],
        [        0],
        [        0],
        [        0],
        [        0],
        [        0]])

    See Also
    ========

    JzKetCoupled: Coupled eigenstates
    sympy.physics.quantum.tensorproduct.TensorProduct: Used to specify uncoupled states
    uncouple: Uncouples states given coupling parameters
    couple: Couples uncoupled states

    """

    @classmethod
    def dual_class(self):
        return JzBra

    @classmethod
    def coupled_class(self):
        return JzKetCoupled

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_base(beta=pi*Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_base(alpha=pi*Rational(3, 2), beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_base(**options)


class JzBra(SpinState, Bra):
    """Eigenbra of Jz.

    See the JzKet for the usage of spin eigenstates.

    See Also
    ========

    JzKet: Usage of spin states

    """

    @classmethod
    def dual_class(self):
        return JzKet

    @classmethod
    def coupled_class(self):
        return JzBraCoupled


# Method used primarily to create coupled_n and coupled_jn by __new__ in
# CoupledSpinState
# This same method is also used by the uncouple method, and is separated from
# the CoupledSpinState class to maintain consistency in defining coupling
def _build_coupled(jcoupling, length):
    n_list = [ [n + 1] for n in range(length) ]
    coupled_jn = []
    coupled_n = []
    for n1, n2, j_new in jcoupling:
        coupled_jn.append(j_new)
        coupled_n.append( (n_list[n1 - 1], n_list[n2 - 1]) )
        n_sort = sorted(n_list[n1 - 1] + n_list[n2 - 1])
        n_list[n_sort[0] - 1] = n_sort
    return coupled_n, coupled_jn


class CoupledSpinState(SpinState):
    """Base class for coupled angular momentum states."""

    def __new__(cls, j, m, jn, *jcoupling):
        # Check j and m values using SpinState
        SpinState(j, m)
        # Build and check coupling scheme from arguments
        if len(jcoupling) == 0:
            # Use default coupling scheme
            jcoupling = []
            for n in range(2, len(jn)):
                jcoupling.append( (1, n, Add(*[jn[i] for i in range(n)])) )
            jcoupling.append( (1, len(jn), j) )
        elif len(jcoupling) == 1:
            # Use specified coupling scheme
            jcoupling = jcoupling[0]
        else:
            raise TypeError("CoupledSpinState only takes 3 or 4 arguments, got: %s" % (len(jcoupling) + 3) )
        # Check arguments have correct form
        if not isinstance(jn, (list, tuple, Tuple)):
            raise TypeError('jn must be Tuple, list or tuple, got %s' %
                            jn.__class__.__name__)
        if not isinstance(jcoupling, (list, tuple, Tuple)):
            raise TypeError('jcoupling must be Tuple, list or tuple, got %s' %
                            jcoupling.__class__.__name__)
        if not all(isinstance(term, (list, tuple, Tuple)) for term in jcoupling):
            raise TypeError(
                'All elements of jcoupling must be list, tuple or Tuple')
        if not len(jn) - 1 == len(jcoupling):
            raise ValueError('jcoupling must have length of %d, got %d' %
                             (len(jn) - 1, len(jcoupling)))
        if not all(len(x) == 3 for x in jcoupling):
            raise ValueError('All elements of jcoupling must have length 3')
        # Build sympified args
        j = sympify(j)
        m = sympify(m)
        jn = Tuple( *[sympify(ji) for ji in jn] )
        jcoupling = Tuple( *[Tuple(sympify(
            n1), sympify(n2), sympify(ji)) for (n1, n2, ji) in jcoupling] )
        # Check values in coupling scheme give physical state
        if any(2*ji != int(2*ji) for ji in jn if ji.is_number):
            raise ValueError('All elements of jn must be integer or half-integer, got: %s' % jn)
        if any(n1 != int(n1) or n2 != int(n2) for (n1, n2, _) in jcoupling):
            raise ValueError('Indices in jcoupling must be integers')
        if any(n1 < 1 or n2 < 1 or n1 > len(jn) or n2 > len(jn) for (n1, n2, _) in jcoupling):
            raise ValueError('Indices must be between 1 and the number of coupled spin spaces')
        if any(2*ji != int(2*ji) for (_, _, ji) in jcoupling if ji.is_number):
            raise ValueError('All coupled j values in coupling scheme must be integer or half-integer')
        coupled_n, coupled_jn = _build_coupled(jcoupling, len(jn))
        jvals = list(jn)
        for n, (n1, n2) in enumerate(coupled_n):
            j1 = jvals[min(n1) - 1]
            j2 = jvals[min(n2) - 1]
            j3 = coupled_jn[n]
            if sympify(j1).is_number and sympify(j2).is_number and sympify(j3).is_number:
                if j1 + j2 < j3:
                    raise ValueError('All couplings must have j1+j2 >= j3, '
                        'in coupling number %d got j1,j2,j3: %d,%d,%d' % (n + 1, j1, j2, j3))
                if abs(j1 - j2) > j3:
                    raise ValueError("All couplings must have |j1+j2| <= j3, "
                        "in coupling number %d got j1,j2,j3: %d,%d,%d" % (n + 1, j1, j2, j3))
                if int_valued(j1 + j2):
                    pass
            jvals[min(n1 + n2) - 1] = j3
        if len(jcoupling) > 0 and jcoupling[-1][2] != j:
            raise ValueError('Last j value coupled together must be the final j of the state')
        # Return state
        return State.__new__(cls, j, m, jn, jcoupling)

    def _print_label(self, printer, *args):
        label = [printer._print(self.j), printer._print(self.m)]
        for i, ji in enumerate(self.jn, start=1):
            label.append('j%d=%s' % (
                i, printer._print(ji)
            ))
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            label.append('j(%s)=%s' % (
                ','.join(str(i) for i in sorted(n1 + n2)), printer._print(jn)
            ))
        return ','.join(label)

    def _print_label_pretty(self, printer, *args):
        label = [self.j, self.m]
        for i, ji in enumerate(self.jn, start=1):
            symb = 'j%d' % i
            symb = pretty_symbol(symb)
            symb = prettyForm(symb + '=')
            item = prettyForm(*symb.right(printer._print(ji)))
            label.append(item)
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join(pretty_symbol("j%d" % i)[-1] for i in sorted(n1 + n2))
            symb = prettyForm('j' + n + '=')
            item = prettyForm(*symb.right(printer._print(jn)))
            label.append(item)
        return self._print_sequence_pretty(
            label, self._label_separator, printer, *args
        )

    def _print_label_latex(self, printer, *args):
        label = [
            printer._print(self.j, *args),
            printer._print(self.m, *args)
        ]
        for i, ji in enumerate(self.jn, start=1):
            label.append('j_{%d}=%s' % (i, printer._print(ji, *args)) )
        for jn, (n1, n2) in zip(self.coupled_jn[:-1], self.coupled_n[:-1]):
            n = ','.join(str(i) for i in sorted(n1 + n2))
            label.append('j_{%s}=%s' % (n, printer._print(jn, *args)) )
        return self._label_separator.join(label)

    @property
    def jn(self):
        return self.label[2]

    @property
    def coupling(self):
        return self.label[3]

    @property
    def coupled_jn(self):
        return _build_coupled(self.label[3], len(self.label[2]))[1]

    @property
    def coupled_n(self):
        return _build_coupled(self.label[3], len(self.label[2]))[0]

    @classmethod
    def _eval_hilbert_space(cls, label):
        j = Add(*label[2])
        if j.is_number:
            return DirectSumHilbertSpace(*[ ComplexSpace(x) for x in range(int(2*j + 1), 0, -2) ])
        else:
            # TODO: Need hilbert space fix, see issue 5732
            # Desired behavior:
            #ji = symbols('ji')
            #ret = Sum(ComplexSpace(2*ji + 1), (ji, 0, j))
            # Temporary fix:
            return ComplexSpace(2*j + 1)

    def _represent_coupled_base(self, **options):
        evect = self.uncoupled_class()
        if not self.j.is_number:
            raise ValueError(
                'State must not have symbolic j value to represent')
        if not self.hilbert_space.dimension.is_number:
            raise ValueError(
                'State must not have symbolic j values to represent')
        result = zeros(self.hilbert_space.dimension, 1)
        if self.j == int(self.j):
            start = self.j**2
        else:
            start = (2*self.j - 1)*(1 + 2*self.j)/4
        result[start:start + 2*self.j + 1, 0] = evect(
            self.j, self.m)._represent_base(**options)
        return result

    def _eval_rewrite_as_Jx(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jx, JxBraCoupled, **options)
        return self._rewrite_basis(Jx, JxKetCoupled, **options)

    def _eval_rewrite_as_Jy(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jy, JyBraCoupled, **options)
        return self._rewrite_basis(Jy, JyKetCoupled, **options)

    def _eval_rewrite_as_Jz(self, *args, **options):
        if isinstance(self, Bra):
            return self._rewrite_basis(Jz, JzBraCoupled, **options)
        return self._rewrite_basis(Jz, JzKetCoupled, **options)


class JxKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JxBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JxKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(**options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(beta=pi/2, **options)


class JxBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jx.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JxKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JxBra


class JyKetCoupled(CoupledSpinState, Ket):
    """Coupled eigenket of Jy.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JyBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JyKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(gamma=pi/2, **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(**options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), beta=-pi/2, gamma=pi/2, **options)


class JyBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jy.

    See JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JyKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JyBra


class JzKetCoupled(CoupledSpinState, Ket):
    r"""Coupled eigenket of Jz

    Spin state that is an eigenket of Jz which represents the coupling of
    separate spin spaces.

    The arguments for creating instances of JzKetCoupled are ``j``, ``m``,
    ``jn`` and an optional ``jcoupling`` argument. The ``j`` and ``m`` options
    are the total angular momentum quantum numbers, as used for normal states
    (e.g. JzKet).

    The other required parameter in ``jn``, which is a tuple defining the `j_n`
    angular momentum quantum numbers of the product spaces. So for example, if
    a state represented the coupling of the product basis state
    `\left|j_1,m_1\right\rangle\times\left|j_2,m_2\right\rangle`, the ``jn``
    for this state would be ``(j1,j2)``.

    The final option is ``jcoupling``, which is used to define how the spaces
    specified by ``jn`` are coupled, which includes both the order these spaces
    are coupled together and the quantum numbers that arise from these
    couplings. The ``jcoupling`` parameter itself is a list of lists, such that
    each of the sublists defines a single coupling between the spin spaces. If
    there are N coupled angular momentum spaces, that is ``jn`` has N elements,
    then there must be N-1 sublists. Each of these sublists making up the
    ``jcoupling`` parameter have length 3. The first two elements are the
    indices of the product spaces that are considered to be coupled together.
    For example, if we want to couple `j_1` and `j_4`, the indices would be 1
    and 4. If a state has already been coupled, it is referenced by the
    smallest index that is coupled, so if `j_2` and `j_4` has already been
    coupled to some `j_{24}`, then this value can be coupled by referencing it
    with index 2. The final element of the sublist is the quantum number of the
    coupled state. So putting everything together, into a valid sublist for
    ``jcoupling``, if `j_1` and `j_2` are coupled to an angular momentum space
    with quantum number `j_{12}` with the value ``j12``, the sublist would be
    ``(1,2,j12)``, N-1 of these sublists are used in the list for
    ``jcoupling``.

    Note the ``jcoupling`` parameter is optional, if it is not specified, the
    default coupling is taken. This default value is to coupled the spaces in
    order and take the quantum number of the coupling to be the maximum value.
    For example, if the spin spaces are `j_1`, `j_2`, `j_3`, `j_4`, then the
    default coupling couples `j_1` and `j_2` to `j_{12}=j_1+j_2`, then,
    `j_{12}` and `j_3` are coupled to `j_{123}=j_{12}+j_3`, and finally
    `j_{123}` and `j_4` to `j=j_{123}+j_4`. The jcoupling value that would
    correspond to this is:

        ``((1,2,j1+j2),(1,3,j1+j2+j3))``

    Parameters
    ==========

    args : tuple
        The arguments that must be passed are ``j``, ``m``, ``jn``, and
        ``jcoupling``. The ``j`` value is the total angular momentum. The ``m``
        value is the eigenvalue of the Jz spin operator. The ``jn`` list are
        the j values of argular momentum spaces coupled together. The
        ``jcoupling`` parameter is an optional parameter defining how the spaces
        are coupled together. See the above description for how these coupling
        parameters are defined.

    Examples
    ========

    Defining simple spin states, both numerical and symbolic:

        >>> from sympy.physics.quantum.spin import JzKetCoupled
        >>> from sympy import symbols
        >>> JzKetCoupled(1, 0, (1, 1))
        |1,0,j1=1,j2=1>
        >>> j, m, j1, j2 = symbols('j m j1 j2')
        >>> JzKetCoupled(j, m, (j1, j2))
        |j,m,j1=j1,j2=j2>

    Defining coupled spin states for more than 2 coupled spaces with various
    coupling parameters:

        >>> JzKetCoupled(2, 1, (1, 1, 1))
        |2,1,j1=1,j2=1,j3=1,j(1,2)=2>
        >>> JzKetCoupled(2, 1, (1, 1, 1), ((1,2,2),(1,3,2)) )
        |2,1,j1=1,j2=1,j3=1,j(1,2)=2>
        >>> JzKetCoupled(2, 1, (1, 1, 1), ((2,3,1),(1,2,2)) )
        |2,1,j1=1,j2=1,j3=1,j(2,3)=1>

    Rewriting the JzKetCoupled in terms of eigenkets of the Jx operator:
    Note: that the resulting eigenstates are JxKetCoupled

        >>> JzKetCoupled(1,1,(1,1)).rewrite("Jx")
        |1,-1,j1=1,j2=1>/2 - sqrt(2)*|1,0,j1=1,j2=1>/2 + |1,1,j1=1,j2=1>/2

    The rewrite method can be used to convert a coupled state to an uncoupled
    state. This is done by passing coupled=False to the rewrite function:

        >>> JzKetCoupled(1, 0, (1, 1)).rewrite('Jz', coupled=False)
        -sqrt(2)*|1,-1>x|1,1>/2 + sqrt(2)*|1,1>x|1,-1>/2

    Get the vector representation of a state in terms of the basis elements
    of the Jx operator:

        >>> from sympy.physics.quantum.represent import represent
        >>> from sympy.physics.quantum.spin import Jx
        >>> from sympy import S
        >>> represent(JzKetCoupled(1,-1,(S(1)/2,S(1)/2)), basis=Jx)
        Matrix([
        [        0],
        [      1/2],
        [sqrt(2)/2],
        [      1/2]])

    See Also
    ========

    JzKet: Normal spin eigenstates
    uncouple: Uncoupling of coupling spin states
    couple: Coupling of uncoupled spin states

    """

    @classmethod
    def dual_class(self):
        return JzBraCoupled

    @classmethod
    def uncoupled_class(self):
        return JzKet

    def _represent_default_basis(self, **options):
        return self._represent_JzOp(None, **options)

    def _represent_JxOp(self, basis, **options):
        return self._represent_coupled_base(beta=pi*Rational(3, 2), **options)

    def _represent_JyOp(self, basis, **options):
        return self._represent_coupled_base(alpha=pi*Rational(3, 2), beta=pi/2, gamma=pi/2, **options)

    def _represent_JzOp(self, basis, **options):
        return self._represent_coupled_base(**options)


class JzBraCoupled(CoupledSpinState, Bra):
    """Coupled eigenbra of Jz.

    See the JzKetCoupled for the usage of coupled spin eigenstates.

    See Also
    ========

    JzKetCoupled: Usage of coupled spin states

    """

    @classmethod
    def dual_class(self):
        return JzKetCoupled

    @classmethod
    def uncoupled_class(self):
        return JzBra

#-----------------------------------------------------------------------------
# Coupling/uncoupling
#-----------------------------------------------------------------------------


def couple(expr, jcoupling_list=None):
    """ Couple a tensor product of spin states

    This function can be used to couple an uncoupled tensor product of spin
    states. All of the eigenstates to be coupled must be of the same class. It
    will return a linear combination of eigenstates that are subclasses of
    CoupledSpinState determined by Clebsch-Gordan angular momentum coupling
    coefficients.

    Parameters
    ==========

    expr : Expr
        An expression involving TensorProducts of spin states to be coupled.
        Each state must be a subclass of SpinState and they all must be the
        same class.

    jcoupling_list : list or tuple
        Elements of this list are sub-lists of length 2 specifying the order of
        the coupling of the spin spaces. The length of this must be N-1, where N
        is the number of states in the tensor product to be coupled. The
        elements of this sublist are the same as the first two elements of each
        sublist in the ``jcoupling`` parameter defined for JzKetCoupled. If this
        parameter is not specified, the default value is taken, which couples
        the first and second product basis spaces, then couples this new coupled
        space to the third product space, etc

    Examples
    ========

    Couple a tensor product of numerical states for two spaces:

        >>> from sympy.physics.quantum.spin import JzKet, couple
        >>> from sympy.physics.quantum.tensorproduct import TensorProduct
        >>> couple(TensorProduct(JzKet(1,0), JzKet(1,1)))
        -sqrt(2)*|1,1,j1=1,j2=1>/2 + sqrt(2)*|2,1,j1=1,j2=1>/2


    Numerical coupling of three spaces using the default coupling method, i.e.
    first and second spaces couple, then this couples to the third space:

        >>> couple(TensorProduct(JzKet(1,1), JzKet(1,1), JzKet(1,0)))
        sqrt(6)*|2,2,j1=1,j2=1,j3=1,j(1,2)=2>/3 + sqrt(3)*|3,2,j1=1,j2=1,j3=1,j(1,2)=2>/3

    Perform this same coupling, but we define the coupling to first couple
    the first and third spaces:

        >>> couple(TensorProduct(JzKet(1,1), JzKet(1,1), JzKet(1,0)), ((1,3),(1,2)) )
        sqrt(2)*|2,2,j1=1,j2=1,j3=1,j(1,3)=1>/2 - sqrt(6)*|2,2,j1=1,j2=1,j3=1,j(1,3)=2>/6 + sqrt(3)*|3,2,j1=1,j2=1,j3=1,j(1,3)=2>/3

    Couple a tensor product of symbolic states:

        >>> from sympy import symbols
        >>> j1,m1,j2,m2 = symbols('j1 m1 j2 m2')
        >>> couple(TensorProduct(JzKet(j1,m1), JzKet(j2,m2)))
        Sum(CG(j1, m1, j2, m2, j, m1 + m2)*|j,m1 + m2,j1=j1,j2=j2>, (j, m1 + m2, j1 + j2))

    """
    a = expr.atoms(TensorProduct)
    for tp in a:
        # Allow other tensor products to be in expression
        if not all(isinstance(state, SpinState) for state in tp.args):
            continue
        # If tensor product has all spin states, raise error for invalid tensor product state
        if not all(state.__class__ is tp.args[0].__class__ for state in tp.args):
            raise TypeError('All states must be the same basis')
        expr = expr.subs(tp, _couple(tp, jcoupling_list))
    return expr


def _couple(tp, jcoupling_list):
    states = tp.args
    coupled_evect = states[0].coupled_class()

    # Define default coupling if none is specified
    if jcoupling_list is None:
        jcoupling_list = []
        for n in range(1, len(states)):
            jcoupling_list.append( (1, n + 1) )

    # Check jcoupling_list valid
    if not len(jcoupling_list) == len(states) - 1:
        raise TypeError('jcoupling_list must be length %d, got %d' %
                        (len(states) - 1, len(jcoupling_list)))
    if not all( len(coupling) == 2 for coupling in jcoupling_list):
        raise ValueError('Each coupling must define 2 spaces')
    if any(n1 == n2 for n1, n2 in jcoupling_list):
        raise ValueError('Spin spaces cannot couple to themselves')
    if all(sympify(n1).is_number and sympify(n2).is_number for n1, n2 in jcoupling_list):
        j_test = [0]*len(states)
        for n1, n2 in jcoupling_list:
            if j_test[n1 - 1] == -1 or j_test[n2 - 1] == -1:
                raise ValueError('Spaces coupling j_n\'s are referenced by smallest n value')
            j_test[max(n1, n2) - 1] = -1

    # j values of states to be coupled together
    jn = [state.j for state in states]
    mn = [state.m for state in states]

    # Create coupling_list, which defines all the couplings between all
    # the spaces from jcoupling_list
    coupling_list = []
    n_list = [ [i + 1] for i in range(len(states)) ]
    for j_coupling in jcoupling_list:
        # Least n for all j_n which is coupled as first and second spaces
        n1, n2 = j_coupling
        # List of all n's coupled in first and second spaces
        j1_n = list(n_list[n1 - 1])
        j2_n = list(n_list[n2 - 1])
        coupling_list.append( (j1_n, j2_n) )
        # Set new j_n to be coupling of all j_n in both first and second spaces
        n_list[ min(n1, n2) - 1 ] = sorted(j1_n + j2_n)

    if all(state.j.is_number and state.m.is_number for state in states):
        # Numerical coupling
        # Iterate over difference between maximum possible j value of each coupling and the actual value
        diff_max = [ Add( *[ jn[n - 1] - mn[n - 1] for n in coupling[0] +
                         coupling[1] ] ) for coupling in coupling_list ]
        result = []
        for diff in range(diff_max[-1] + 1):
            # Determine available configurations
            n = len(coupling_list)
            tot = binomial(diff + n - 1, diff)

            for config_num in range(tot):
                diff_list = _confignum_to_difflist(config_num, diff, n)

                # Skip the configuration if non-physical
                # This is a lazy check for physical states given the loose restrictions of diff_max
                if any(d > m for d, m in zip(diff_list, diff_max)):
                    continue

                # Determine term
                cg_terms = []
                coupled_j = list(jn)
                jcoupling = []
                for (j1_n, j2_n), coupling_diff in zip(coupling_list, diff_list):
                    j1 = coupled_j[ min(j1_n) - 1 ]
                    j2 = coupled_j[ min(j2_n) - 1 ]
                    j3 = j1 + j2 - coupling_diff
                    coupled_j[ min(j1_n + j2_n) - 1 ] = j3
                    m1 = Add( *[ mn[x - 1] for x in j1_n] )
                    m2 = Add( *[ mn[x - 1] for x in j2_n] )
                    m3 = m1 + m2
                    cg_terms.append( (j1, m1, j2, m2, j3, m3) )
                    jcoupling.append( (min(j1_n), min(j2_n), j3) )
                # Better checks that state is physical
                if any(abs(term[5]) > term[4] for term in cg_terms):
                    continue
                if any(term[0] + term[2] < term[4] for term in cg_terms):
                    continue
                if any(abs(term[0] - term[2]) > term[4] for term in cg_terms):
                    continue
                coeff = Mul( *[ CG(*term).doit() for term in cg_terms] )
                state = coupled_evect(j3, m3, jn, jcoupling)
                result.append(coeff*state)
        return Add(*result)
    else:
        # Symbolic coupling
        cg_terms = []
        jcoupling = []
        sum_terms = []
        coupled_j = list(jn)
        for j1_n, j2_n in coupling_list:
            j1 = coupled_j[ min(j1_n) - 1 ]
            j2 = coupled_j[ min(j2_n) - 1 ]
            if len(j1_n + j2_n) == len(states):
                j3 = symbols('j')
            else:
                j3_name = 'j' + ''.join(["%s" % n for n in j1_n + j2_n])
                j3 = symbols(j3_name)
            coupled_j[ min(j1_n + j2_n) - 1 ] = j3
            m1 = Add( *[ mn[x - 1] for x in j1_n] )
            m2 = Add( *[ mn[x - 1] for x in j2_n] )
            m3 = m1 + m2
            cg_terms.append( (j1, m1, j2, m2, j3, m3) )
            jcoupling.append( (min(j1_n), min(j2_n), j3) )
            sum_terms.append((j3, m3, j1 + j2))
        coeff = Mul( *[ CG(*term) for term in cg_terms] )
        state = coupled_evect(j3, m3, jn, jcoupling)
        return Sum(coeff*state, *sum_terms)


def uncouple(expr, jn=None, jcoupling_list=None):
    """ Uncouple a coupled spin state

    Gives the uncoupled representation of a coupled spin state. Arguments must
    be either a spin state that is a subclass of CoupledSpinState or a spin
    state that is a subclass of SpinState and an array giving the j values
    of the spaces that are to be coupled

    Parameters
    ==========

    expr : Expr
        The expression containing states that are to be coupled. If the states
        are a subclass of SpinState, the ``jn`` and ``jcoupling`` parameters
        must be defined. If the states are a subclass of CoupledSpinState,
        ``jn`` and ``jcoupling`` will be taken from the state.

    jn : list or tuple
        The list of the j-values that are coupled. If state is a
        CoupledSpinState, this parameter is ignored. This must be defined if
        state is not a subclass of CoupledSpinState. The syntax of this
        parameter is the same as the ``jn`` parameter of JzKetCoupled.

    jcoupling_list : list or tuple
        The list defining how the j-values are coupled together. If state is a
        CoupledSpinState, this parameter is ignored. This must be defined if
        state is not a subclass of CoupledSpinState. The syntax of this
        parameter is the same as the ``jcoupling`` parameter of JzKetCoupled.

    Examples
    ========

    Uncouple a numerical state using a CoupledSpinState state:

        >>> from sympy.physics.quantum.spin import JzKetCoupled, uncouple
        >>> from sympy import S
        >>> uncouple(JzKetCoupled(1, 0, (S(1)/2, S(1)/2)))
        sqrt(2)*|1/2,-1/2>x|1/2,1/2>/2 + sqrt(2)*|1/2,1/2>x|1/2,-1/2>/2

    Perform the same calculation using a SpinState state:

        >>> from sympy.physics.quantum.spin import JzKet
        >>> uncouple(JzKet(1, 0), (S(1)/2, S(1)/2))
        sqrt(2)*|1/2,-1/2>x|1/2,1/2>/2 + sqrt(2)*|1/2,1/2>x|1/2,-1/2>/2

    Uncouple a numerical state of three coupled spaces using a CoupledSpinState state:

        >>> uncouple(JzKetCoupled(1, 1, (1, 1, 1), ((1,3,1),(1,2,1)) ))
        |1,-1>x|1,1>x|1,1>/2 - |1,0>x|1,0>x|1,1>/2 + |1,1>x|1,0>x|1,0>/2 - |1,1>x|1,1>x|1,-1>/2

    Perform the same calculation using a SpinState state:

        >>> uncouple(JzKet(1, 1), (1, 1, 1), ((1,3,1),(1,2,1)) )
        |1,-1>x|1,1>x|1,1>/2 - |1,0>x|1,0>x|1,1>/2 + |1,1>x|1,0>x|1,0>/2 - |1,1>x|1,1>x|1,-1>/2

    Uncouple a symbolic state using a CoupledSpinState state:

        >>> from sympy import symbols
        >>> j,m,j1,j2 = symbols('j m j1 j2')
        >>> uncouple(JzKetCoupled(j, m, (j1, j2)))
        Sum(CG(j1, m1, j2, m2, j, m)*|j1,m1>x|j2,m2>, (m1, -j1, j1), (m2, -j2, j2))

    Perform the same calculation using a SpinState state

        >>> uncouple(JzKet(j, m), (j1, j2))
        Sum(CG(j1, m1, j2, m2, j, m)*|j1,m1>x|j2,m2>, (m1, -j1, j1), (m2, -j2, j2))

    """
    a = expr.atoms(SpinState)
    for state in a:
        expr = expr.subs(state, _uncouple(state, jn, jcoupling_list))
    return expr


def _uncouple(state, jn, jcoupling_list):
    if isinstance(state, CoupledSpinState):
        jn = state.jn
        coupled_n = state.coupled_n
        coupled_jn = state.coupled_jn
        evect = state.uncoupled_class()
    elif isinstance(state, SpinState):
        if jn is None:
            raise ValueError("Must specify j-values for coupled state")
        if not isinstance(jn, (list, tuple)):
            raise TypeError("jn must be list or tuple")
        if jcoupling_list is None:
            # Use default
            jcoupling_list = []
            for i in range(1, len(jn)):
                jcoupling_list.append(
                    (1, 1 + i, Add(*[jn[j] for j in range(i + 1)])) )
        if not isinstance(jcoupling_list, (list, tuple)):
            raise TypeError("jcoupling must be a list or tuple")
        if not len(jcoupling_list) == len(jn) - 1:
            raise ValueError("Must specify 2 fewer coupling terms than the number of j values")
        coupled_n, coupled_jn = _build_coupled(jcoupling_list, len(jn))
        evect = state.__class__
    else:
        raise TypeError("state must be a spin state")
    j = state.j
    m = state.m
    coupling_list = []
    j_list = list(jn)

    # Create coupling, which defines all the couplings between all the spaces
    for j3, (n1, n2) in zip(coupled_jn, coupled_n):
        # j's which are coupled as first and second spaces
        j1 = j_list[n1[0] - 1]
        j2 = j_list[n2[0] - 1]
        # Build coupling list
        coupling_list.append( (n1, n2, j1, j2, j3) )
        # Set new value in j_list
        j_list[min(n1 + n2) - 1] = j3

    if j.is_number and m.is_number:
        diff_max = [ 2*x for x in jn ]
        diff = Add(*jn) - m

        n = len(jn)
        tot = binomial(diff + n - 1, diff)

        result = []
        for config_num in range(tot):
            diff_list = _confignum_to_difflist(config_num, diff, n)
            if any(d > p for d, p in zip(diff_list, diff_max)):
                continue

            cg_terms = []
            for coupling in coupling_list:
                j1_n, j2_n, j1, j2, j3 = coupling
                m1 = Add( *[ jn[x - 1] - diff_list[x - 1] for x in j1_n ] )
                m2 = Add( *[ jn[x - 1] - diff_list[x - 1] for x in j2_n ] )
                m3 = m1 + m2
                cg_terms.append( (j1, m1, j2, m2, j3, m3) )
            coeff = Mul( *[ CG(*term).doit() for term in cg_terms ] )
            state = TensorProduct(
                *[ evect(j, j - d) for j, d in zip(jn, diff_list) ] )
            result.append(coeff*state)
        return Add(*result)
    else:
        # Symbolic coupling
        m_str = "m1:%d" % (len(jn) + 1)
        mvals = symbols(m_str)
        cg_terms = [(j1, Add(*[mvals[n - 1] for n in j1_n]),
                     j2, Add(*[mvals[n - 1] for n in j2_n]),
                     j3, Add(*[mvals[n - 1] for n in j1_n + j2_n])) for j1_n, j2_n, j1, j2, j3 in coupling_list[:-1] ]
        cg_terms.append(*[(j1, Add(*[mvals[n - 1] for n in j1_n]),
                           j2, Add(*[mvals[n - 1] for n in j2_n]),
                           j, m) for j1_n, j2_n, j1, j2, j3 in [coupling_list[-1]] ])
        cg_coeff = Mul(*[CG(*cg_term) for cg_term in cg_terms])
        sum_terms = [ (m, -j, j) for j, m in zip(jn, mvals) ]
        state = TensorProduct( *[ evect(j, m) for j, m in zip(jn, mvals) ] )
        return Sum(cg_coeff*state, *sum_terms)


def _confignum_to_difflist(config_num, diff, list_len):
    # Determines configuration of diffs into list_len number of slots
    diff_list = []
    for n in range(list_len):
        prev_diff = diff
        # Number of spots after current one
        rem_spots = list_len - n - 1
        # Number of configurations of distributing diff among the remaining spots
        rem_configs = binomial(diff + rem_spots - 1, diff)
        while config_num >= rem_configs:
            config_num -= rem_configs
            diff -= 1
            rem_configs = binomial(diff + rem_spots - 1, diff)
        diff_list.append(prev_diff - diff)
    return diff_list

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\storage.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Storage (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page


class SerializedStorageKey(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> SerializedStorageKey:
        return cls(json)

    def __repr__(self):
        return 'SerializedStorageKey({})'.format(super().__repr__())


class StorageType(enum.Enum):
    '''
    Enum of possible storage types.
    '''
    COOKIES = "cookies"
    FILE_SYSTEMS = "file_systems"
    INDEXEDDB = "indexeddb"
    LOCAL_STORAGE = "local_storage"
    SHADER_CACHE = "shader_cache"
    WEBSQL = "websql"
    SERVICE_WORKERS = "service_workers"
    CACHE_STORAGE = "cache_storage"
    INTEREST_GROUPS = "interest_groups"
    SHARED_STORAGE = "shared_storage"
    STORAGE_BUCKETS = "storage_buckets"
    ALL_ = "all"
    OTHER = "other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class UsageForType:
    '''
    Usage for a storage type.
    '''
    #: Name of storage type.
    storage_type: StorageType

    #: Storage usage (bytes).
    usage: float

    def to_json(self):
        json = dict()
        json['storageType'] = self.storage_type.to_json()
        json['usage'] = self.usage
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            storage_type=StorageType.from_json(json['storageType']),
            usage=float(json['usage']),
        )


@dataclass
class TrustTokens:
    '''
    Pair of issuer origin and number of available (signed, but not used) Trust
    Tokens from that issuer.
    '''
    issuer_origin: str

    count: float

    def to_json(self):
        json = dict()
        json['issuerOrigin'] = self.issuer_origin
        json['count'] = self.count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            issuer_origin=str(json['issuerOrigin']),
            count=float(json['count']),
        )


class InterestGroupAuctionId(str):
    '''
    Protected audience interest group auction identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> InterestGroupAuctionId:
        return cls(json)

    def __repr__(self):
        return 'InterestGroupAuctionId({})'.format(super().__repr__())


class InterestGroupAccessType(enum.Enum):
    '''
    Enum of interest group access types.
    '''
    JOIN = "join"
    LEAVE = "leave"
    UPDATE = "update"
    LOADED = "loaded"
    BID = "bid"
    WIN = "win"
    ADDITIONAL_BID = "additionalBid"
    ADDITIONAL_BID_WIN = "additionalBidWin"
    TOP_LEVEL_BID = "topLevelBid"
    TOP_LEVEL_ADDITIONAL_BID = "topLevelAdditionalBid"
    CLEAR = "clear"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class InterestGroupAuctionEventType(enum.Enum):
    '''
    Enum of auction events.
    '''
    STARTED = "started"
    CONFIG_RESOLVED = "configResolved"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class InterestGroupAuctionFetchType(enum.Enum):
    '''
    Enum of network fetches auctions can do.
    '''
    BIDDER_JS = "bidderJs"
    BIDDER_WASM = "bidderWasm"
    SELLER_JS = "sellerJs"
    BIDDER_TRUSTED_SIGNALS = "bidderTrustedSignals"
    SELLER_TRUSTED_SIGNALS = "sellerTrustedSignals"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SharedStorageAccessScope(enum.Enum):
    '''
    Enum of shared storage access scopes.
    '''
    WINDOW = "window"
    SHARED_STORAGE_WORKLET = "sharedStorageWorklet"
    PROTECTED_AUDIENCE_WORKLET = "protectedAudienceWorklet"
    HEADER = "header"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SharedStorageAccessMethod(enum.Enum):
    '''
    Enum of shared storage access methods.
    '''
    ADD_MODULE = "addModule"
    CREATE_WORKLET = "createWorklet"
    SELECT_URL = "selectURL"
    RUN = "run"
    BATCH_UPDATE = "batchUpdate"
    SET_ = "set"
    APPEND = "append"
    DELETE = "delete"
    CLEAR = "clear"
    GET = "get"
    KEYS = "keys"
    VALUES = "values"
    ENTRIES = "entries"
    LENGTH = "length"
    REMAINING_BUDGET = "remainingBudget"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SharedStorageEntry:
    '''
    Struct for a single key-value pair in an origin's shared storage.
    '''
    key: str

    value: str

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['value'] = self.value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=str(json['key']),
            value=str(json['value']),
        )


@dataclass
class SharedStorageMetadata:
    '''
    Details for an origin's shared storage.
    '''
    #: Time when the origin's shared storage was last created.
    creation_time: network.TimeSinceEpoch

    #: Number of key-value pairs stored in origin's shared storage.
    length: int

    #: Current amount of bits of entropy remaining in the navigation budget.
    remaining_budget: float

    #: Total number of bytes stored as key-value pairs in origin's shared
    #: storage.
    bytes_used: int

    def to_json(self):
        json = dict()
        json['creationTime'] = self.creation_time.to_json()
        json['length'] = self.length
        json['remainingBudget'] = self.remaining_budget
        json['bytesUsed'] = self.bytes_used
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            creation_time=network.TimeSinceEpoch.from_json(json['creationTime']),
            length=int(json['length']),
            remaining_budget=float(json['remainingBudget']),
            bytes_used=int(json['bytesUsed']),
        )


@dataclass
class SharedStorageReportingMetadata:
    '''
    Pair of reporting metadata details for a candidate URL for ``selectURL()``.
    '''
    event_type: str

    reporting_url: str

    def to_json(self):
        json = dict()
        json['eventType'] = self.event_type
        json['reportingUrl'] = self.reporting_url
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            event_type=str(json['eventType']),
            reporting_url=str(json['reportingUrl']),
        )


@dataclass
class SharedStorageUrlWithMetadata:
    '''
    Bundles a candidate URL with its reporting metadata.
    '''
    #: Spec of candidate URL.
    url: str

    #: Any associated reporting metadata.
    reporting_metadata: typing.List[SharedStorageReportingMetadata]

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['reportingMetadata'] = [i.to_json() for i in self.reporting_metadata]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            reporting_metadata=[SharedStorageReportingMetadata.from_json(i) for i in json['reportingMetadata']],
        )


@dataclass
class SharedStorageAccessParams:
    '''
    Bundles the parameters for shared storage access events whose
    presence/absence can vary according to SharedStorageAccessType.
    '''
    #: Spec of the module script URL.
    #: Present only for SharedStorageAccessType.documentAddModule.
    script_source_url: typing.Optional[str] = None

    #: Name of the registered operation to be run.
    #: Present only for SharedStorageAccessType.documentRun and
    #: SharedStorageAccessType.documentSelectURL.
    operation_name: typing.Optional[str] = None

    #: The operation's serialized data in bytes (converted to a string).
    #: Present only for SharedStorageAccessType.documentRun and
    #: SharedStorageAccessType.documentSelectURL.
    serialized_data: typing.Optional[str] = None

    #: Array of candidate URLs' specs, along with any associated metadata.
    #: Present only for SharedStorageAccessType.documentSelectURL.
    urls_with_metadata: typing.Optional[typing.List[SharedStorageUrlWithMetadata]] = None

    #: Key for a specific entry in an origin's shared storage.
    #: Present only for SharedStorageAccessType.documentSet,
    #: SharedStorageAccessType.documentAppend,
    #: SharedStorageAccessType.documentDelete,
    #: SharedStorageAccessType.workletSet,
    #: SharedStorageAccessType.workletAppend,
    #: SharedStorageAccessType.workletDelete,
    #: SharedStorageAccessType.workletGet,
    #: SharedStorageAccessType.headerSet,
    #: SharedStorageAccessType.headerAppend, and
    #: SharedStorageAccessType.headerDelete.
    key: typing.Optional[str] = None

    #: Value for a specific entry in an origin's shared storage.
    #: Present only for SharedStorageAccessType.documentSet,
    #: SharedStorageAccessType.documentAppend,
    #: SharedStorageAccessType.workletSet,
    #: SharedStorageAccessType.workletAppend,
    #: SharedStorageAccessType.headerSet, and
    #: SharedStorageAccessType.headerAppend.
    value: typing.Optional[str] = None

    #: Whether or not to set an entry for a key if that key is already present.
    #: Present only for SharedStorageAccessType.documentSet,
    #: SharedStorageAccessType.workletSet, and
    #: SharedStorageAccessType.headerSet.
    ignore_if_present: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.script_source_url is not None:
            json['scriptSourceUrl'] = self.script_source_url
        if self.operation_name is not None:
            json['operationName'] = self.operation_name
        if self.serialized_data is not None:
            json['serializedData'] = self.serialized_data
        if self.urls_with_metadata is not None:
            json['urlsWithMetadata'] = [i.to_json() for i in self.urls_with_metadata]
        if self.key is not None:
            json['key'] = self.key
        if self.value is not None:
            json['value'] = self.value
        if self.ignore_if_present is not None:
            json['ignoreIfPresent'] = self.ignore_if_present
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_source_url=str(json['scriptSourceUrl']) if 'scriptSourceUrl' in json else None,
            operation_name=str(json['operationName']) if 'operationName' in json else None,
            serialized_data=str(json['serializedData']) if 'serializedData' in json else None,
            urls_with_metadata=[SharedStorageUrlWithMetadata.from_json(i) for i in json['urlsWithMetadata']] if 'urlsWithMetadata' in json else None,
            key=str(json['key']) if 'key' in json else None,
            value=str(json['value']) if 'value' in json else None,
            ignore_if_present=bool(json['ignoreIfPresent']) if 'ignoreIfPresent' in json else None,
        )


class StorageBucketsDurability(enum.Enum):
    RELAXED = "relaxed"
    STRICT = "strict"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class StorageBucket:
    storage_key: SerializedStorageKey

    #: If not specified, it is the default bucket of the storageKey.
    name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['storageKey'] = self.storage_key.to_json()
        if self.name is not None:
            json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            storage_key=SerializedStorageKey.from_json(json['storageKey']),
            name=str(json['name']) if 'name' in json else None,
        )


@dataclass
class StorageBucketInfo:
    bucket: StorageBucket

    id_: str

    expiration: network.TimeSinceEpoch

    #: Storage quota (bytes).
    quota: float

    persistent: bool

    durability: StorageBucketsDurability

    def to_json(self):
        json = dict()
        json['bucket'] = self.bucket.to_json()
        json['id'] = self.id_
        json['expiration'] = self.expiration.to_json()
        json['quota'] = self.quota
        json['persistent'] = self.persistent
        json['durability'] = self.durability.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            bucket=StorageBucket.from_json(json['bucket']),
            id_=str(json['id']),
            expiration=network.TimeSinceEpoch.from_json(json['expiration']),
            quota=float(json['quota']),
            persistent=bool(json['persistent']),
            durability=StorageBucketsDurability.from_json(json['durability']),
        )


class AttributionReportingSourceType(enum.Enum):
    NAVIGATION = "navigation"
    EVENT = "event"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class UnsignedInt64AsBase10(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UnsignedInt64AsBase10:
        return cls(json)

    def __repr__(self):
        return 'UnsignedInt64AsBase10({})'.format(super().__repr__())


class UnsignedInt128AsBase16(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> UnsignedInt128AsBase16:
        return cls(json)

    def __repr__(self):
        return 'UnsignedInt128AsBase16({})'.format(super().__repr__())


class SignedInt64AsBase10(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> SignedInt64AsBase10:
        return cls(json)

    def __repr__(self):
        return 'SignedInt64AsBase10({})'.format(super().__repr__())


@dataclass
class AttributionReportingFilterDataEntry:
    key: str

    values: typing.List[str]

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['values'] = [i for i in self.values]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=str(json['key']),
            values=[str(i) for i in json['values']],
        )


@dataclass
class AttributionReportingFilterConfig:
    filter_values: typing.List[AttributionReportingFilterDataEntry]

    #: duration in seconds
    lookback_window: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['filterValues'] = [i.to_json() for i in self.filter_values]
        if self.lookback_window is not None:
            json['lookbackWindow'] = self.lookback_window
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filter_values=[AttributionReportingFilterDataEntry.from_json(i) for i in json['filterValues']],
            lookback_window=int(json['lookbackWindow']) if 'lookbackWindow' in json else None,
        )


@dataclass
class AttributionReportingFilterPair:
    filters: typing.List[AttributionReportingFilterConfig]

    not_filters: typing.List[AttributionReportingFilterConfig]

    def to_json(self):
        json = dict()
        json['filters'] = [i.to_json() for i in self.filters]
        json['notFilters'] = [i.to_json() for i in self.not_filters]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filters=[AttributionReportingFilterConfig.from_json(i) for i in json['filters']],
            not_filters=[AttributionReportingFilterConfig.from_json(i) for i in json['notFilters']],
        )


@dataclass
class AttributionReportingAggregationKeysEntry:
    key: str

    value: UnsignedInt128AsBase16

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=str(json['key']),
            value=UnsignedInt128AsBase16.from_json(json['value']),
        )


@dataclass
class AttributionReportingEventReportWindows:
    #: duration in seconds
    start: int

    #: duration in seconds
    ends: typing.List[int]

    def to_json(self):
        json = dict()
        json['start'] = self.start
        json['ends'] = [i for i in self.ends]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            start=int(json['start']),
            ends=[int(i) for i in json['ends']],
        )


@dataclass
class AttributionReportingTriggerSpec:
    #: number instead of integer because not all uint32 can be represented by
    #: int
    trigger_data: typing.List[float]

    event_report_windows: AttributionReportingEventReportWindows

    def to_json(self):
        json = dict()
        json['triggerData'] = [i for i in self.trigger_data]
        json['eventReportWindows'] = self.event_report_windows.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            trigger_data=[float(i) for i in json['triggerData']],
            event_report_windows=AttributionReportingEventReportWindows.from_json(json['eventReportWindows']),
        )


class AttributionReportingTriggerDataMatching(enum.Enum):
    EXACT = "exact"
    MODULUS = "modulus"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AttributionReportingAggregatableDebugReportingData:
    key_piece: UnsignedInt128AsBase16

    #: number instead of integer because not all uint32 can be represented by
    #: int
    value: float

    types: typing.List[str]

    def to_json(self):
        json = dict()
        json['keyPiece'] = self.key_piece.to_json()
        json['value'] = self.value
        json['types'] = [i for i in self.types]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key_piece=UnsignedInt128AsBase16.from_json(json['keyPiece']),
            value=float(json['value']),
            types=[str(i) for i in json['types']],
        )


@dataclass
class AttributionReportingAggregatableDebugReportingConfig:
    key_piece: UnsignedInt128AsBase16

    debug_data: typing.List[AttributionReportingAggregatableDebugReportingData]

    #: number instead of integer because not all uint32 can be represented by
    #: int, only present for source registrations
    budget: typing.Optional[float] = None

    aggregation_coordinator_origin: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['keyPiece'] = self.key_piece.to_json()
        json['debugData'] = [i.to_json() for i in self.debug_data]
        if self.budget is not None:
            json['budget'] = self.budget
        if self.aggregation_coordinator_origin is not None:
            json['aggregationCoordinatorOrigin'] = self.aggregation_coordinator_origin
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key_piece=UnsignedInt128AsBase16.from_json(json['keyPiece']),
            debug_data=[AttributionReportingAggregatableDebugReportingData.from_json(i) for i in json['debugData']],
            budget=float(json['budget']) if 'budget' in json else None,
            aggregation_coordinator_origin=str(json['aggregationCoordinatorOrigin']) if 'aggregationCoordinatorOrigin' in json else None,
        )


@dataclass
class AttributionScopesData:
    values: typing.List[str]

    #: number instead of integer because not all uint32 can be represented by
    #: int
    limit: float

    max_event_states: float

    def to_json(self):
        json = dict()
        json['values'] = [i for i in self.values]
        json['limit'] = self.limit
        json['maxEventStates'] = self.max_event_states
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            values=[str(i) for i in json['values']],
            limit=float(json['limit']),
            max_event_states=float(json['maxEventStates']),
        )


@dataclass
class AttributionReportingSourceRegistration:
    time: network.TimeSinceEpoch

    #: duration in seconds
    expiry: int

    trigger_specs: typing.List[AttributionReportingTriggerSpec]

    #: duration in seconds
    aggregatable_report_window: int

    type_: AttributionReportingSourceType

    source_origin: str

    reporting_origin: str

    destination_sites: typing.List[str]

    event_id: UnsignedInt64AsBase10

    priority: SignedInt64AsBase10

    filter_data: typing.List[AttributionReportingFilterDataEntry]

    aggregation_keys: typing.List[AttributionReportingAggregationKeysEntry]

    trigger_data_matching: AttributionReportingTriggerDataMatching

    destination_limit_priority: SignedInt64AsBase10

    aggregatable_debug_reporting_config: AttributionReportingAggregatableDebugReportingConfig

    max_event_level_reports: int

    debug_key: typing.Optional[UnsignedInt64AsBase10] = None

    scopes_data: typing.Optional[AttributionScopesData] = None

    def to_json(self):
        json = dict()
        json['time'] = self.time.to_json()
        json['expiry'] = self.expiry
        json['triggerSpecs'] = [i.to_json() for i in self.trigger_specs]
        json['aggregatableReportWindow'] = self.aggregatable_report_window
        json['type'] = self.type_.to_json()
        json['sourceOrigin'] = self.source_origin
        json['reportingOrigin'] = self.reporting_origin
        json['destinationSites'] = [i for i in self.destination_sites]
        json['eventId'] = self.event_id.to_json()
        json['priority'] = self.priority.to_json()
        json['filterData'] = [i.to_json() for i in self.filter_data]
        json['aggregationKeys'] = [i.to_json() for i in self.aggregation_keys]
        json['triggerDataMatching'] = self.trigger_data_matching.to_json()
        json['destinationLimitPriority'] = self.destination_limit_priority.to_json()
        json['aggregatableDebugReportingConfig'] = self.aggregatable_debug_reporting_config.to_json()
        json['maxEventLevelReports'] = self.max_event_level_reports
        if self.debug_key is not None:
            json['debugKey'] = self.debug_key.to_json()
        if self.scopes_data is not None:
            json['scopesData'] = self.scopes_data.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            time=network.TimeSinceEpoch.from_json(json['time']),
            expiry=int(json['expiry']),
            trigger_specs=[AttributionReportingTriggerSpec.from_json(i) for i in json['triggerSpecs']],
            aggregatable_report_window=int(json['aggregatableReportWindow']),
            type_=AttributionReportingSourceType.from_json(json['type']),
            source_origin=str(json['sourceOrigin']),
            reporting_origin=str(json['reportingOrigin']),
            destination_sites=[str(i) for i in json['destinationSites']],
            event_id=UnsignedInt64AsBase10.from_json(json['eventId']),
            priority=SignedInt64AsBase10.from_json(json['priority']),
            filter_data=[AttributionReportingFilterDataEntry.from_json(i) for i in json['filterData']],
            aggregation_keys=[AttributionReportingAggregationKeysEntry.from_json(i) for i in json['aggregationKeys']],
            trigger_data_matching=AttributionReportingTriggerDataMatching.from_json(json['triggerDataMatching']),
            destination_limit_priority=SignedInt64AsBase10.from_json(json['destinationLimitPriority']),
            aggregatable_debug_reporting_config=AttributionReportingAggregatableDebugReportingConfig.from_json(json['aggregatableDebugReportingConfig']),
            max_event_level_reports=int(json['maxEventLevelReports']),
            debug_key=UnsignedInt64AsBase10.from_json(json['debugKey']) if 'debugKey' in json else None,
            scopes_data=AttributionScopesData.from_json(json['scopesData']) if 'scopesData' in json else None,
        )


class AttributionReportingSourceRegistrationResult(enum.Enum):
    SUCCESS = "success"
    INTERNAL_ERROR = "internalError"
    INSUFFICIENT_SOURCE_CAPACITY = "insufficientSourceCapacity"
    INSUFFICIENT_UNIQUE_DESTINATION_CAPACITY = "insufficientUniqueDestinationCapacity"
    EXCESSIVE_REPORTING_ORIGINS = "excessiveReportingOrigins"
    PROHIBITED_BY_BROWSER_POLICY = "prohibitedByBrowserPolicy"
    SUCCESS_NOISED = "successNoised"
    DESTINATION_REPORTING_LIMIT_REACHED = "destinationReportingLimitReached"
    DESTINATION_GLOBAL_LIMIT_REACHED = "destinationGlobalLimitReached"
    DESTINATION_BOTH_LIMITS_REACHED = "destinationBothLimitsReached"
    REPORTING_ORIGINS_PER_SITE_LIMIT_REACHED = "reportingOriginsPerSiteLimitReached"
    EXCEEDS_MAX_CHANNEL_CAPACITY = "exceedsMaxChannelCapacity"
    EXCEEDS_MAX_SCOPES_CHANNEL_CAPACITY = "exceedsMaxScopesChannelCapacity"
    EXCEEDS_MAX_TRIGGER_STATE_CARDINALITY = "exceedsMaxTriggerStateCardinality"
    EXCEEDS_MAX_EVENT_STATES_LIMIT = "exceedsMaxEventStatesLimit"
    DESTINATION_PER_DAY_REPORTING_LIMIT_REACHED = "destinationPerDayReportingLimitReached"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AttributionReportingSourceRegistrationTimeConfig(enum.Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class AttributionReportingAggregatableValueDictEntry:
    key: str

    #: number instead of integer because not all uint32 can be represented by
    #: int
    value: float

    filtering_id: UnsignedInt64AsBase10

    def to_json(self):
        json = dict()
        json['key'] = self.key
        json['value'] = self.value
        json['filteringId'] = self.filtering_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=str(json['key']),
            value=float(json['value']),
            filtering_id=UnsignedInt64AsBase10.from_json(json['filteringId']),
        )


@dataclass
class AttributionReportingAggregatableValueEntry:
    values: typing.List[AttributionReportingAggregatableValueDictEntry]

    filters: AttributionReportingFilterPair

    def to_json(self):
        json = dict()
        json['values'] = [i.to_json() for i in self.values]
        json['filters'] = self.filters.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            values=[AttributionReportingAggregatableValueDictEntry.from_json(i) for i in json['values']],
            filters=AttributionReportingFilterPair.from_json(json['filters']),
        )


@dataclass
class AttributionReportingEventTriggerData:
    data: UnsignedInt64AsBase10

    priority: SignedInt64AsBase10

    filters: AttributionReportingFilterPair

    dedup_key: typing.Optional[UnsignedInt64AsBase10] = None

    def to_json(self):
        json = dict()
        json['data'] = self.data.to_json()
        json['priority'] = self.priority.to_json()
        json['filters'] = self.filters.to_json()
        if self.dedup_key is not None:
            json['dedupKey'] = self.dedup_key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            data=UnsignedInt64AsBase10.from_json(json['data']),
            priority=SignedInt64AsBase10.from_json(json['priority']),
            filters=AttributionReportingFilterPair.from_json(json['filters']),
            dedup_key=UnsignedInt64AsBase10.from_json(json['dedupKey']) if 'dedupKey' in json else None,
        )


@dataclass
class AttributionReportingAggregatableTriggerData:
    key_piece: UnsignedInt128AsBase16

    source_keys: typing.List[str]

    filters: AttributionReportingFilterPair

    def to_json(self):
        json = dict()
        json['keyPiece'] = self.key_piece.to_json()
        json['sourceKeys'] = [i for i in self.source_keys]
        json['filters'] = self.filters.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key_piece=UnsignedInt128AsBase16.from_json(json['keyPiece']),
            source_keys=[str(i) for i in json['sourceKeys']],
            filters=AttributionReportingFilterPair.from_json(json['filters']),
        )


@dataclass
class AttributionReportingAggregatableDedupKey:
    filters: AttributionReportingFilterPair

    dedup_key: typing.Optional[UnsignedInt64AsBase10] = None

    def to_json(self):
        json = dict()
        json['filters'] = self.filters.to_json()
        if self.dedup_key is not None:
            json['dedupKey'] = self.dedup_key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filters=AttributionReportingFilterPair.from_json(json['filters']),
            dedup_key=UnsignedInt64AsBase10.from_json(json['dedupKey']) if 'dedupKey' in json else None,
        )


@dataclass
class AttributionReportingTriggerRegistration:
    filters: AttributionReportingFilterPair

    aggregatable_dedup_keys: typing.List[AttributionReportingAggregatableDedupKey]

    event_trigger_data: typing.List[AttributionReportingEventTriggerData]

    aggregatable_trigger_data: typing.List[AttributionReportingAggregatableTriggerData]

    aggregatable_values: typing.List[AttributionReportingAggregatableValueEntry]

    aggregatable_filtering_id_max_bytes: int

    debug_reporting: bool

    source_registration_time_config: AttributionReportingSourceRegistrationTimeConfig

    aggregatable_debug_reporting_config: AttributionReportingAggregatableDebugReportingConfig

    scopes: typing.List[str]

    debug_key: typing.Optional[UnsignedInt64AsBase10] = None

    aggregation_coordinator_origin: typing.Optional[str] = None

    trigger_context_id: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['filters'] = self.filters.to_json()
        json['aggregatableDedupKeys'] = [i.to_json() for i in self.aggregatable_dedup_keys]
        json['eventTriggerData'] = [i.to_json() for i in self.event_trigger_data]
        json['aggregatableTriggerData'] = [i.to_json() for i in self.aggregatable_trigger_data]
        json['aggregatableValues'] = [i.to_json() for i in self.aggregatable_values]
        json['aggregatableFilteringIdMaxBytes'] = self.aggregatable_filtering_id_max_bytes
        json['debugReporting'] = self.debug_reporting
        json['sourceRegistrationTimeConfig'] = self.source_registration_time_config.to_json()
        json['aggregatableDebugReportingConfig'] = self.aggregatable_debug_reporting_config.to_json()
        json['scopes'] = [i for i in self.scopes]
        if self.debug_key is not None:
            json['debugKey'] = self.debug_key.to_json()
        if self.aggregation_coordinator_origin is not None:
            json['aggregationCoordinatorOrigin'] = self.aggregation_coordinator_origin
        if self.trigger_context_id is not None:
            json['triggerContextId'] = self.trigger_context_id
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filters=AttributionReportingFilterPair.from_json(json['filters']),
            aggregatable_dedup_keys=[AttributionReportingAggregatableDedupKey.from_json(i) for i in json['aggregatableDedupKeys']],
            event_trigger_data=[AttributionReportingEventTriggerData.from_json(i) for i in json['eventTriggerData']],
            aggregatable_trigger_data=[AttributionReportingAggregatableTriggerData.from_json(i) for i in json['aggregatableTriggerData']],
            aggregatable_values=[AttributionReportingAggregatableValueEntry.from_json(i) for i in json['aggregatableValues']],
            aggregatable_filtering_id_max_bytes=int(json['aggregatableFilteringIdMaxBytes']),
            debug_reporting=bool(json['debugReporting']),
            source_registration_time_config=AttributionReportingSourceRegistrationTimeConfig.from_json(json['sourceRegistrationTimeConfig']),
            aggregatable_debug_reporting_config=AttributionReportingAggregatableDebugReportingConfig.from_json(json['aggregatableDebugReportingConfig']),
            scopes=[str(i) for i in json['scopes']],
            debug_key=UnsignedInt64AsBase10.from_json(json['debugKey']) if 'debugKey' in json else None,
            aggregation_coordinator_origin=str(json['aggregationCoordinatorOrigin']) if 'aggregationCoordinatorOrigin' in json else None,
            trigger_context_id=str(json['triggerContextId']) if 'triggerContextId' in json else None,
        )


class AttributionReportingEventLevelResult(enum.Enum):
    SUCCESS = "success"
    SUCCESS_DROPPED_LOWER_PRIORITY = "successDroppedLowerPriority"
    INTERNAL_ERROR = "internalError"
    NO_CAPACITY_FOR_ATTRIBUTION_DESTINATION = "noCapacityForAttributionDestination"
    NO_MATCHING_SOURCES = "noMatchingSources"
    DEDUPLICATED = "deduplicated"
    EXCESSIVE_ATTRIBUTIONS = "excessiveAttributions"
    PRIORITY_TOO_LOW = "priorityTooLow"
    NEVER_ATTRIBUTED_SOURCE = "neverAttributedSource"
    EXCESSIVE_REPORTING_ORIGINS = "excessiveReportingOrigins"
    NO_MATCHING_SOURCE_FILTER_DATA = "noMatchingSourceFilterData"
    PROHIBITED_BY_BROWSER_POLICY = "prohibitedByBrowserPolicy"
    NO_MATCHING_CONFIGURATIONS = "noMatchingConfigurations"
    EXCESSIVE_REPORTS = "excessiveReports"
    FALSELY_ATTRIBUTED_SOURCE = "falselyAttributedSource"
    REPORT_WINDOW_PASSED = "reportWindowPassed"
    NOT_REGISTERED = "notRegistered"
    REPORT_WINDOW_NOT_STARTED = "reportWindowNotStarted"
    NO_MATCHING_TRIGGER_DATA = "noMatchingTriggerData"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AttributionReportingAggregatableResult(enum.Enum):
    SUCCESS = "success"
    INTERNAL_ERROR = "internalError"
    NO_CAPACITY_FOR_ATTRIBUTION_DESTINATION = "noCapacityForAttributionDestination"
    NO_MATCHING_SOURCES = "noMatchingSources"
    EXCESSIVE_ATTRIBUTIONS = "excessiveAttributions"
    EXCESSIVE_REPORTING_ORIGINS = "excessiveReportingOrigins"
    NO_HISTOGRAMS = "noHistograms"
    INSUFFICIENT_BUDGET = "insufficientBudget"
    INSUFFICIENT_NAMED_BUDGET = "insufficientNamedBudget"
    NO_MATCHING_SOURCE_FILTER_DATA = "noMatchingSourceFilterData"
    NOT_REGISTERED = "notRegistered"
    PROHIBITED_BY_BROWSER_POLICY = "prohibitedByBrowserPolicy"
    DEDUPLICATED = "deduplicated"
    REPORT_WINDOW_PASSED = "reportWindowPassed"
    EXCESSIVE_REPORTS = "excessiveReports"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class RelatedWebsiteSet:
    '''
    A single Related Website Set object.
    '''
    #: The primary site of this set, along with the ccTLDs if there is any.
    primary_sites: typing.List[str]

    #: The associated sites of this set, along with the ccTLDs if there is any.
    associated_sites: typing.List[str]

    #: The service sites of this set, along with the ccTLDs if there is any.
    service_sites: typing.List[str]

    def to_json(self):
        json = dict()
        json['primarySites'] = [i for i in self.primary_sites]
        json['associatedSites'] = [i for i in self.associated_sites]
        json['serviceSites'] = [i for i in self.service_sites]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            primary_sites=[str(i) for i in json['primarySites']],
            associated_sites=[str(i) for i in json['associatedSites']],
            service_sites=[str(i) for i in json['serviceSites']],
        )


def get_storage_key_for_frame(
        frame_id: page.FrameId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SerializedStorageKey]:
    '''
    Returns a storage key given a frame id.

    :param frame_id:
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getStorageKeyForFrame',
        'params': params,
    }
    json = yield cmd_dict
    return SerializedStorageKey.from_json(json['storageKey'])


def clear_data_for_origin(
        origin: str,
        storage_types: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears storage for origin.

    :param origin: Security origin.
    :param storage_types: Comma separated list of StorageType to clear.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    params['storageTypes'] = storage_types
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearDataForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def clear_data_for_storage_key(
        storage_key: str,
        storage_types: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears storage for storage key.

    :param storage_key: Storage key.
    :param storage_types: Comma separated list of StorageType to clear.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    params['storageTypes'] = storage_types
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearDataForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def get_cookies(
        browser_context_id: typing.Optional[browser.BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[network.Cookie]]:
    '''
    Returns all browser cookies.

    :param browser_context_id: *(Optional)* Browser context to use when called on the browser endpoint.
    :returns: Array of cookie objects.
    '''
    params: T_JSON_DICT = dict()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getCookies',
        'params': params,
    }
    json = yield cmd_dict
    return [network.Cookie.from_json(i) for i in json['cookies']]


def set_cookies(
        cookies: typing.List[network.CookieParam],
        browser_context_id: typing.Optional[browser.BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets given cookies.

    :param cookies: Cookies to be set.
    :param browser_context_id: *(Optional)* Browser context to use when called on the browser endpoint.
    '''
    params: T_JSON_DICT = dict()
    params['cookies'] = [i.to_json() for i in cookies]
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setCookies',
        'params': params,
    }
    json = yield cmd_dict


def clear_cookies(
        browser_context_id: typing.Optional[browser.BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears cookies.

    :param browser_context_id: *(Optional)* Browser context to use when called on the browser endpoint.
    '''
    params: T_JSON_DICT = dict()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearCookies',
        'params': params,
    }
    json = yield cmd_dict


def get_usage_and_quota(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[float, float, bool, typing.List[UsageForType]]]:
    '''
    Returns usage and quota in bytes.

    :param origin: Security origin.
    :returns: A tuple with the following items:

        0. **usage** - Storage usage (bytes).
        1. **quota** - Storage quota (bytes).
        2. **overrideActive** - Whether or not the origin has an active storage quota override
        3. **usageBreakdown** - Storage usage per type (bytes).
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getUsageAndQuota',
        'params': params,
    }
    json = yield cmd_dict
    return (
        float(json['usage']),
        float(json['quota']),
        bool(json['overrideActive']),
        [UsageForType.from_json(i) for i in json['usageBreakdown']]
    )


def override_quota_for_origin(
        origin: str,
        quota_size: typing.Optional[float] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Override quota for the specified origin

    **EXPERIMENTAL**

    :param origin: Security origin.
    :param quota_size: *(Optional)* The quota size (in bytes) to override the original quota with. If this is called multiple times, the overridden quota will be equal to the quotaSize provided in the final call. If this is called without specifying a quotaSize, the quota will be reset to the default value for the specified origin. If this is called multiple times with different origins, the override will be maintained for each origin until it is disabled (called without a quotaSize).
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    if quota_size is not None:
        params['quotaSize'] = quota_size
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.overrideQuotaForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def track_cache_storage_for_origin(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Registers origin to be notified when an update occurs to its cache storage list.

    :param origin: Security origin.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.trackCacheStorageForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def track_cache_storage_for_storage_key(
        storage_key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Registers storage key to be notified when an update occurs to its cache storage list.

    :param storage_key: Storage key.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.trackCacheStorageForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def track_indexed_db_for_origin(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Registers origin to be notified when an update occurs to its IndexedDB.

    :param origin: Security origin.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.trackIndexedDBForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def track_indexed_db_for_storage_key(
        storage_key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Registers storage key to be notified when an update occurs to its IndexedDB.

    :param storage_key: Storage key.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.trackIndexedDBForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def untrack_cache_storage_for_origin(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Unregisters origin from receiving notifications for cache storage.

    :param origin: Security origin.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.untrackCacheStorageForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def untrack_cache_storage_for_storage_key(
        storage_key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Unregisters storage key from receiving notifications for cache storage.

    :param storage_key: Storage key.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.untrackCacheStorageForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def untrack_indexed_db_for_origin(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Unregisters origin from receiving notifications for IndexedDB.

    :param origin: Security origin.
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.untrackIndexedDBForOrigin',
        'params': params,
    }
    json = yield cmd_dict


def untrack_indexed_db_for_storage_key(
        storage_key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Unregisters storage key from receiving notifications for IndexedDB.

    :param storage_key: Storage key.
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.untrackIndexedDBForStorageKey',
        'params': params,
    }
    json = yield cmd_dict


def get_trust_tokens() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[TrustTokens]]:
    '''
    Returns the number of stored Trust Tokens per issuer for the
    current browsing context.

    **EXPERIMENTAL**

    :returns:
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getTrustTokens',
    }
    json = yield cmd_dict
    return [TrustTokens.from_json(i) for i in json['tokens']]


def clear_trust_tokens(
        issuer_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Removes all Trust Tokens issued by the provided issuerOrigin.
    Leaves other stored data, including the issuer's Redemption Records, intact.

    **EXPERIMENTAL**

    :param issuer_origin:
    :returns: True if any tokens were deleted, false otherwise.
    '''
    params: T_JSON_DICT = dict()
    params['issuerOrigin'] = issuer_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearTrustTokens',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['didDeleteTokens'])


def get_interest_group_details(
        owner_origin: str,
        name: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    Gets details for a named interest group.

    **EXPERIMENTAL**

    :param owner_origin:
    :param name:
    :returns: This largely corresponds to: https://wicg.github.io/turtledove/#dictdef-generatebidinterestgroup but has absolute expirationTime instead of relative lifetimeMs and also adds joiningOrigin.
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['name'] = name
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getInterestGroupDetails',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['details'])


def set_interest_group_tracking(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables/Disables issuing of interestGroupAccessed events.

    **EXPERIMENTAL**

    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setInterestGroupTracking',
        'params': params,
    }
    json = yield cmd_dict


def set_interest_group_auction_tracking(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables/Disables issuing of interestGroupAuctionEventOccurred and
    interestGroupAuctionNetworkRequestCreated.

    **EXPERIMENTAL**

    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setInterestGroupAuctionTracking',
        'params': params,
    }
    json = yield cmd_dict


def get_shared_storage_metadata(
        owner_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SharedStorageMetadata]:
    '''
    Gets metadata for an origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getSharedStorageMetadata',
        'params': params,
    }
    json = yield cmd_dict
    return SharedStorageMetadata.from_json(json['metadata'])


def get_shared_storage_entries(
        owner_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[SharedStorageEntry]]:
    '''
    Gets the entries in an given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getSharedStorageEntries',
        'params': params,
    }
    json = yield cmd_dict
    return [SharedStorageEntry.from_json(i) for i in json['entries']]


def set_shared_storage_entry(
        owner_origin: str,
        key: str,
        value: str,
        ignore_if_present: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets entry with ``key`` and ``value`` for a given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :param key:
    :param value:
    :param ignore_if_present: *(Optional)* If ```ignoreIfPresent```` is included and true, then only sets the entry if ````key``` doesn't already exist.
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['key'] = key
    params['value'] = value
    if ignore_if_present is not None:
        params['ignoreIfPresent'] = ignore_if_present
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setSharedStorageEntry',
        'params': params,
    }
    json = yield cmd_dict


def delete_shared_storage_entry(
        owner_origin: str,
        key: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes entry for ``key`` (if it exists) for a given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    :param key:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['key'] = key
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.deleteSharedStorageEntry',
        'params': params,
    }
    json = yield cmd_dict


def clear_shared_storage_entries(
        owner_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all entries for a given origin's shared storage.

    **EXPERIMENTAL**

    :param owner_origin:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.clearSharedStorageEntries',
        'params': params,
    }
    json = yield cmd_dict


def reset_shared_storage_budget(
        owner_origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Resets the budget for ``ownerOrigin`` by clearing all budget withdrawals.

    **EXPERIMENTAL**

    :param owner_origin:
    '''
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.resetSharedStorageBudget',
        'params': params,
    }
    json = yield cmd_dict


def set_shared_storage_tracking(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables/disables issuing of sharedStorageAccessed events.

    **EXPERIMENTAL**

    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setSharedStorageTracking',
        'params': params,
    }
    json = yield cmd_dict


def set_storage_bucket_tracking(
        storage_key: str,
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set tracking for a storage key's buckets.

    **EXPERIMENTAL**

    :param storage_key:
    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setStorageBucketTracking',
        'params': params,
    }
    json = yield cmd_dict


def delete_storage_bucket(
        bucket: StorageBucket
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes the Storage Bucket with the given storage key and bucket name.

    **EXPERIMENTAL**

    :param bucket:
    '''
    params: T_JSON_DICT = dict()
    params['bucket'] = bucket.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.deleteStorageBucket',
        'params': params,
    }
    json = yield cmd_dict


def run_bounce_tracking_mitigations() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Deletes state for sites identified as potential bounce trackers, immediately.

    **EXPERIMENTAL**

    :returns:
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.runBounceTrackingMitigations',
    }
    json = yield cmd_dict
    return [str(i) for i in json['deletedSites']]


def set_attribution_reporting_local_testing_mode(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    https://wicg.github.io/attribution-reporting-api/

    **EXPERIMENTAL**

    :param enabled: If enabled, noise is suppressed and reports are sent immediately.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setAttributionReportingLocalTestingMode',
        'params': params,
    }
    json = yield cmd_dict


def set_attribution_reporting_tracking(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables/disables issuing of Attribution Reporting events.

    **EXPERIMENTAL**

    :param enable:
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setAttributionReportingTracking',
        'params': params,
    }
    json = yield cmd_dict


def send_pending_attribution_reports() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,int]:
    '''
    Sends all pending Attribution Reports immediately, regardless of their
    scheduled report time.

    **EXPERIMENTAL**

    :returns: The number of reports that were sent.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.sendPendingAttributionReports',
    }
    json = yield cmd_dict
    return int(json['numSent'])


def get_related_website_sets() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[RelatedWebsiteSet]]:
    '''
    Returns the effective Related Website Sets in use by this profile for the browser
    session. The effective Related Website Sets will not change during a browser session.

    **EXPERIMENTAL**

    :returns:
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getRelatedWebsiteSets',
    }
    json = yield cmd_dict
    return [RelatedWebsiteSet.from_json(i) for i in json['sets']]


def get_affected_urls_for_third_party_cookie_metadata(
        first_party_url: str,
        third_party_urls: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Returns the list of URLs from a page and its embedded resources that match
    existing grace period URL pattern rules.
    https://developers.google.com/privacy-sandbox/cookies/temporary-exceptions/grace-period

    **EXPERIMENTAL**

    :param first_party_url: The URL of the page currently being visited.
    :param third_party_urls: The list of embedded resource URLs from the page.
    :returns: Array of matching URLs. If there is a primary pattern match for the first- party URL, only the first-party URL is returned in the array.
    '''
    params: T_JSON_DICT = dict()
    params['firstPartyUrl'] = first_party_url
    params['thirdPartyUrls'] = [i for i in third_party_urls]
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.getAffectedUrlsForThirdPartyCookieMetadata',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['matchedUrls']]


@event_class('Storage.cacheStorageContentUpdated')
@dataclass
class CacheStorageContentUpdated:
    '''
    A cache's contents have been modified.
    '''
    #: Origin to update.
    origin: str
    #: Storage key to update.
    storage_key: str
    #: Storage bucket to update.
    bucket_id: str
    #: Name of cache in origin.
    cache_name: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CacheStorageContentUpdated:
        return cls(
            origin=str(json['origin']),
            storage_key=str(json['storageKey']),
            bucket_id=str(json['bucketId']),
            cache_name=str(json['cacheName'])
        )


@event_class('Storage.cacheStorageListUpdated')
@dataclass
class CacheStorageListUpdated:
    '''
    A cache has been added/deleted.
    '''
    #: Origin to update.
    origin: str
    #: Storage key to update.
    storage_key: str
    #: Storage bucket to update.
    bucket_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CacheStorageListUpdated:
        return cls(
            origin=str(json['origin']),
            storage_key=str(json['storageKey']),
            bucket_id=str(json['bucketId'])
        )


@event_class('Storage.indexedDBContentUpdated')
@dataclass
class IndexedDBContentUpdated:
    '''
    The origin's IndexedDB object store has been modified.
    '''
    #: Origin to update.
    origin: str
    #: Storage key to update.
    storage_key: str
    #: Storage bucket to update.
    bucket_id: str
    #: Database to update.
    database_name: str
    #: ObjectStore to update.
    object_store_name: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IndexedDBContentUpdated:
        return cls(
            origin=str(json['origin']),
            storage_key=str(json['storageKey']),
            bucket_id=str(json['bucketId']),
            database_name=str(json['databaseName']),
            object_store_name=str(json['objectStoreName'])
        )


@event_class('Storage.indexedDBListUpdated')
@dataclass
class IndexedDBListUpdated:
    '''
    The origin's IndexedDB database list has been modified.
    '''
    #: Origin to update.
    origin: str
    #: Storage key to update.
    storage_key: str
    #: Storage bucket to update.
    bucket_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IndexedDBListUpdated:
        return cls(
            origin=str(json['origin']),
            storage_key=str(json['storageKey']),
            bucket_id=str(json['bucketId'])
        )


@event_class('Storage.interestGroupAccessed')
@dataclass
class InterestGroupAccessed:
    '''
    One of the interest groups was accessed. Note that these events are global
    to all targets sharing an interest group store.
    '''
    access_time: network.TimeSinceEpoch
    type_: InterestGroupAccessType
    owner_origin: str
    name: str
    #: For topLevelBid/topLevelAdditionalBid, and when appropriate,
    #: win and additionalBidWin
    component_seller_origin: typing.Optional[str]
    #: For bid or somethingBid event, if done locally and not on a server.
    bid: typing.Optional[float]
    bid_currency: typing.Optional[str]
    #: For non-global events --- links to interestGroupAuctionEvent
    unique_auction_id: typing.Optional[InterestGroupAuctionId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAccessed:
        return cls(
            access_time=network.TimeSinceEpoch.from_json(json['accessTime']),
            type_=InterestGroupAccessType.from_json(json['type']),
            owner_origin=str(json['ownerOrigin']),
            name=str(json['name']),
            component_seller_origin=str(json['componentSellerOrigin']) if 'componentSellerOrigin' in json else None,
            bid=float(json['bid']) if 'bid' in json else None,
            bid_currency=str(json['bidCurrency']) if 'bidCurrency' in json else None,
            unique_auction_id=InterestGroupAuctionId.from_json(json['uniqueAuctionId']) if 'uniqueAuctionId' in json else None
        )


@event_class('Storage.interestGroupAuctionEventOccurred')
@dataclass
class InterestGroupAuctionEventOccurred:
    '''
    An auction involving interest groups is taking place. These events are
    target-specific.
    '''
    event_time: network.TimeSinceEpoch
    type_: InterestGroupAuctionEventType
    unique_auction_id: InterestGroupAuctionId
    #: Set for child auctions.
    parent_auction_id: typing.Optional[InterestGroupAuctionId]
    #: Set for started and configResolved
    auction_config: typing.Optional[dict]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAuctionEventOccurred:
        return cls(
            event_time=network.TimeSinceEpoch.from_json(json['eventTime']),
            type_=InterestGroupAuctionEventType.from_json(json['type']),
            unique_auction_id=InterestGroupAuctionId.from_json(json['uniqueAuctionId']),
            parent_auction_id=InterestGroupAuctionId.from_json(json['parentAuctionId']) if 'parentAuctionId' in json else None,
            auction_config=dict(json['auctionConfig']) if 'auctionConfig' in json else None
        )


@event_class('Storage.interestGroupAuctionNetworkRequestCreated')
@dataclass
class InterestGroupAuctionNetworkRequestCreated:
    '''
    Specifies which auctions a particular network fetch may be related to, and
    in what role. Note that it is not ordered with respect to
    Network.requestWillBeSent (but will happen before loadingFinished
    loadingFailed).
    '''
    type_: InterestGroupAuctionFetchType
    request_id: network.RequestId
    #: This is the set of the auctions using the worklet that issued this
    #: request.  In the case of trusted signals, it's possible that only some of
    #: them actually care about the keys being queried.
    auctions: typing.List[InterestGroupAuctionId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> InterestGroupAuctionNetworkRequestCreated:
        return cls(
            type_=InterestGroupAuctionFetchType.from_json(json['type']),
            request_id=network.RequestId.from_json(json['requestId']),
            auctions=[InterestGroupAuctionId.from_json(i) for i in json['auctions']]
        )


@event_class('Storage.sharedStorageAccessed')
@dataclass
class SharedStorageAccessed:
    '''
    Shared storage was accessed by the associated page.
    The following parameters are included in all events.
    '''
    #: Time of the access.
    access_time: network.TimeSinceEpoch
    #: Enum value indicating the access scope.
    scope: SharedStorageAccessScope
    #: Enum value indicating the Shared Storage API method invoked.
    method: SharedStorageAccessMethod
    #: DevTools Frame Token for the primary frame tree's root.
    main_frame_id: page.FrameId
    #: Serialization of the origin owning the Shared Storage data.
    owner_origin: str
    #: Serialization of the site owning the Shared Storage data.
    owner_site: str
    #: The sub-parameters wrapped by ``params`` are all optional and their
    #: presence/absence depends on ``type``.
    params: SharedStorageAccessParams

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SharedStorageAccessed:
        return cls(
            access_time=network.TimeSinceEpoch.from_json(json['accessTime']),
            scope=SharedStorageAccessScope.from_json(json['scope']),
            method=SharedStorageAccessMethod.from_json(json['method']),
            main_frame_id=page.FrameId.from_json(json['mainFrameId']),
            owner_origin=str(json['ownerOrigin']),
            owner_site=str(json['ownerSite']),
            params=SharedStorageAccessParams.from_json(json['params'])
        )


@event_class('Storage.storageBucketCreatedOrUpdated')
@dataclass
class StorageBucketCreatedOrUpdated:
    bucket_info: StorageBucketInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> StorageBucketCreatedOrUpdated:
        return cls(
            bucket_info=StorageBucketInfo.from_json(json['bucketInfo'])
        )


@event_class('Storage.storageBucketDeleted')
@dataclass
class StorageBucketDeleted:
    bucket_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> StorageBucketDeleted:
        return cls(
            bucket_id=str(json['bucketId'])
        )


@event_class('Storage.attributionReportingSourceRegistered')
@dataclass
class AttributionReportingSourceRegistered:
    '''
    **EXPERIMENTAL**


    '''
    registration: AttributionReportingSourceRegistration
    result: AttributionReportingSourceRegistrationResult

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttributionReportingSourceRegistered:
        return cls(
            registration=AttributionReportingSourceRegistration.from_json(json['registration']),
            result=AttributionReportingSourceRegistrationResult.from_json(json['result'])
        )


@event_class('Storage.attributionReportingTriggerRegistered')
@dataclass
class AttributionReportingTriggerRegistered:
    '''
    **EXPERIMENTAL**


    '''
    registration: AttributionReportingTriggerRegistration
    event_level: AttributionReportingEventLevelResult
    aggregatable: AttributionReportingAggregatableResult

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AttributionReportingTriggerRegistered:
        return cls(
            registration=AttributionReportingTriggerRegistration.from_json(json['registration']),
            event_level=AttributionReportingEventLevelResult.from_json(json['eventLevel']),
            aggregatable=AttributionReportingAggregatableResult.from_json(json['aggregatable'])
        )