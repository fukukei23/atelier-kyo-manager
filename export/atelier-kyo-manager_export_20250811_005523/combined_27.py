
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\solvers.py ===
"""
This module contain solvers for all kinds of equations:

    - algebraic or transcendental, use solve()

    - recurrence, use rsolve()

    - differential, use dsolve()

    - nonlinear (numerically), use nsolve()
      (you will need a good starting point)

"""
from __future__ import annotations

from sympy.core import (S, Add, Symbol, Dummy, Expr, Mul)
from sympy.core.assumptions import check_assumptions
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand_mul, expand_log, Derivative,
                                 AppliedUndef, UndefinedFunction, nfloat,
                                 Function, expand_power_exp, _mexpand, expand,
                                 expand_func)
from sympy.core.logic import fuzzy_not, fuzzy_and
from sympy.core.numbers import Float, Rational, _illegal
from sympy.core.intfunc import integer_log, ilcm
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import And, BooleanAtom

from sympy.functions import (log, exp, LambertW, cos, sin, tan, acos, asin, atan,
                             Abs, re, im, arg, sqrt, atan2)
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.integrals.integrals import Integral
from sympy.ntheory.factor_ import divisors
from sympy.simplify import (simplify, collect, powsimp, posify,  # type: ignore
    powdenest, nsimplify, denom, logcombine, sqrtdenest, fraction,
    separatevars)
from sympy.simplify.sqrtdenest import sqrt_depth
from sympy.simplify.fu import TR1, TR2i, TR10, TR11
from sympy.strategies.rl import rebuild
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.matrices import Matrix, zeros
from sympy.polys import roots, cancel, factor, Poly
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy.polys.polytools import gcd
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import filldedent, debugf
from sympy.utilities.iterables import (connected_components,
    generate_bell, uniq, iterable, is_sequence, subsets, flatten, sift)
from sympy.utilities.decorator import conserve_mpmath_dps

from mpmath import findroot

from sympy.solvers.polysys import solve_poly_system

from types import GeneratorType
from collections import defaultdict
from itertools import combinations, product

import warnings


def recast_to_symbols(eqs, symbols):
    """
    Return (e, s, d) where e and s are versions of *eqs* and
    *symbols* in which any non-Symbol objects in *symbols* have
    been replaced with generic Dummy symbols and d is a dictionary
    that can be used to restore the original expressions.

    Examples
    ========

    >>> from sympy.solvers.solvers import recast_to_symbols
    >>> from sympy import symbols, Function
    >>> x, y = symbols('x y')
    >>> fx = Function('f')(x)
    >>> eqs, syms = [fx + 1, x, y], [fx, y]
    >>> e, s, d = recast_to_symbols(eqs, syms); (e, s, d)
    ([_X0 + 1, x, y], [_X0, y], {_X0: f(x)})

    The original equations and symbols can be restored using d:

    >>> assert [i.xreplace(d) for i in eqs] == eqs
    >>> assert [d.get(i, i) for i in s] == syms

    """
    if not iterable(eqs) and iterable(symbols):
        raise ValueError('Both eqs and symbols must be iterable')
    orig = list(symbols)
    symbols = list(ordered(symbols))
    swap_sym = {}
    i = 0
    for s in symbols:
        if not isinstance(s, Symbol) and s not in swap_sym:
            swap_sym[s] = Dummy('X%d' % i)
            i += 1
    new_f = []
    for i in eqs:
        isubs = getattr(i, 'subs', None)
        if isubs is not None:
            new_f.append(isubs(swap_sym))
        else:
            new_f.append(i)
    restore = {v: k for k, v in swap_sym.items()}
    return new_f, [swap_sym.get(i, i) for i in orig], restore


def _ispow(e):
    """Return True if e is a Pow or is exp."""
    return isinstance(e, Expr) and (e.is_Pow or isinstance(e, exp))


def _simple_dens(f, symbols):
    # when checking if a denominator is zero, we can just check the
    # base of powers with nonzero exponents since if the base is zero
    # the power will be zero, too. To keep it simple and fast, we
    # limit simplification to exponents that are Numbers
    dens = set()
    for d in denoms(f, symbols):
        if d.is_Pow and d.exp.is_Number:
            if d.exp.is_zero:
                continue  # foo**0 is never 0
            d = d.base
        dens.add(d)
    return dens


def denoms(eq, *symbols):
    """
    Return (recursively) set of all denominators that appear in *eq*
    that contain any symbol in *symbols*; if *symbols* are not
    provided then all denominators will be returned.

    Examples
    ========

    >>> from sympy.solvers.solvers import denoms
    >>> from sympy.abc import x, y, z

    >>> denoms(x/y)
    {y}

    >>> denoms(x/(y*z))
    {y, z}

    >>> denoms(3/x + y/z)
    {x, z}

    >>> denoms(x/2 + y/z)
    {2, z}

    If *symbols* are provided then only denominators containing
    those symbols will be returned:

    >>> denoms(1/x + 1/y + 1/z, y, z)
    {y, z}

    """

    pot = preorder_traversal(eq)
    dens = set()
    for p in pot:
        # Here p might be Tuple or Relational
        # Expr subtrees (e.g. lhs and rhs) will be traversed after by pot
        if not isinstance(p, Expr):
            continue
        den = denom(p)
        if den is S.One:
            continue
        dens.update(Mul.make_args(den))
    if not symbols:
        return dens
    elif len(symbols) == 1:
        if iterable(symbols[0]):
            symbols = symbols[0]
    return {d for d in dens if any(s in d.free_symbols for s in symbols)}


def checksol(f, symbol, sol=None, **flags):
    """
    Checks whether sol is a solution of equation f == 0.

    Explanation
    ===========

    Input can be either a single symbol and corresponding value
    or a dictionary of symbols and values. When given as a dictionary
    and flag ``simplify=True``, the values in the dictionary will be
    simplified. *f* can be a single equation or an iterable of equations.
    A solution must satisfy all equations in *f* to be considered valid;
    if a solution does not satisfy any equation, False is returned; if one or
    more checks are inconclusive (and none are False) then None is returned.

    Examples
    ========

    >>> from sympy import checksol, symbols
    >>> x, y = symbols('x,y')
    >>> checksol(x**4 - 1, x, 1)
    True
    >>> checksol(x**4 - 1, x, 0)
    False
    >>> checksol(x**2 + y**2 - 5**2, {x: 3, y: 4})
    True

    To check if an expression is zero using ``checksol()``, pass it
    as *f* and send an empty dictionary for *symbol*:

    >>> checksol(x**2 + x - x*(x + 1), {})
    True

    None is returned if ``checksol()`` could not conclude.

    flags:
        'numerical=True (default)'
           do a fast numerical check if ``f`` has only one symbol.
        'minimal=True (default is False)'
           a very fast, minimal testing.
        'warn=True (default is False)'
           show a warning if checksol() could not conclude.
        'simplify=True (default)'
           simplify solution before substituting into function and
           simplify the function before trying specific simplifications
        'force=True (default is False)'
           make positive all symbols without assumptions regarding sign.

    """
    from sympy.physics.units import Unit

    minimal = flags.get('minimal', False)

    if sol is not None:
        sol = {symbol: sol}
    elif isinstance(symbol, dict):
        sol = symbol
    else:
        msg = 'Expecting (sym, val) or ({sym: val}, None) but got (%s, %s)'
        raise ValueError(msg % (symbol, sol))

    if iterable(f):
        if not f:
            raise ValueError('no functions to check')
        return fuzzy_and(checksol(fi, sol, **flags) for fi in f)

    f = _sympify(f)

    if f.is_number:
        return f.is_zero

    if isinstance(f, Poly):
        f = f.as_expr()
    elif isinstance(f, (Eq, Ne)):
        if f.rhs in (S.true, S.false):
            f = f.reversed
        B, E = f.args
        if isinstance(B, BooleanAtom):
            f = f.subs(sol)
            if not f.is_Boolean:
                return
        elif isinstance(f, Eq):
            f = Add(f.lhs, -f.rhs, evaluate=False)

    if isinstance(f, BooleanAtom):
        return bool(f)
    elif not f.is_Relational and not f:
        return True

    illegal = set(_illegal)
    if any(sympify(v).atoms() & illegal for k, v in sol.items()):
        return False

    attempt = -1
    numerical = flags.get('numerical', True)
    while 1:
        attempt += 1
        if attempt == 0:
            val = f.subs(sol)
            if isinstance(val, Mul):
                val = val.as_independent(Unit)[0]
            if val.atoms() & illegal:
                return False
        elif attempt == 1:
            if not val.is_number:
                if not val.is_constant(*list(sol.keys()), simplify=not minimal):
                    return False
                # there are free symbols -- simple expansion might work
                _, val = val.as_content_primitive()
                val = _mexpand(val.as_numer_denom()[0], recursive=True)
        elif attempt == 2:
            if minimal:
                return
            if flags.get('simplify', True):
                for k in sol:
                    sol[k] = simplify(sol[k])
            # start over without the failed expanded form, possibly
            # with a simplified solution
            val = simplify(f.subs(sol))
            if flags.get('force', True):
                val, reps = posify(val)
                # expansion may work now, so try again and check
                exval = _mexpand(val, recursive=True)
                if exval.is_number:
                    # we can decide now
                    val = exval
        else:
            # if there are no radicals and no functions then this can't be
            # zero anymore -- can it?
            pot = preorder_traversal(expand_mul(val))
            seen = set()
            saw_pow_func = False
            for p in pot:
                if p in seen:
                    continue
                seen.add(p)
                if p.is_Pow and not p.exp.is_Integer:
                    saw_pow_func = True
                elif p.is_Function:
                    saw_pow_func = True
                elif isinstance(p, UndefinedFunction):
                    saw_pow_func = True
                if saw_pow_func:
                    break
            if saw_pow_func is False:
                return False
            if flags.get('force', True):
                # don't do a zero check with the positive assumptions in place
                val = val.subs(reps)
            nz = fuzzy_not(val.is_zero)
            if nz is not None:
                # issue 5673: nz may be True even when False
                # so these are just hacks to keep a false positive
                # from being returned

                # HACK 1: LambertW (issue 5673)
                if val.is_number and val.has(LambertW):
                    # don't eval this to verify solution since if we got here,
                    # numerical must be False
                    return None

                # add other HACKs here if necessary, otherwise we assume
                # the nz value is correct
                return not nz
            break
        if val.is_Rational:
            return val == 0
        if numerical and val.is_number:
            return (abs(val.n(18).n(12, chop=True)) < 1e-9) is S.true

    if flags.get('warn', False):
        warnings.warn("\n\tWarning: could not verify solution %s." % sol)
    # returns None if it can't conclude
    # TODO: improve solution testing


def solve(f, *symbols, **flags):
    r"""
    Algebraically solves equations and systems of equations.

    Explanation
    ===========

    Currently supported:
        - polynomial
        - transcendental
        - piecewise combinations of the above
        - systems of linear and polynomial equations
        - systems containing relational expressions
        - systems implied by undetermined coefficients

    Examples
    ========

    The default output varies according to the input and might
    be a list (possibly empty), a dictionary, a list of
    dictionaries or tuples, or an expression involving relationals.
    For specifics regarding different forms of output that may appear, see :ref:`solve_output`.
    Let it suffice here to say that to obtain a uniform output from
    `solve` use ``dict=True`` or ``set=True`` (see below).

        >>> from sympy import solve, Poly, Eq, Matrix, Symbol
        >>> from sympy.abc import x, y, z, a, b

    The expressions that are passed can be Expr, Equality, or Poly
    classes (or lists of the same); a Matrix is considered to be a
    list of all the elements of the matrix:

        >>> solve(x - 3, x)
        [3]
        >>> solve(Eq(x, 3), x)
        [3]
        >>> solve(Poly(x - 3), x)
        [3]
        >>> solve(Matrix([[x, x + y]]), x, y) == solve([x, x + y], x, y)
        True

    If no symbols are indicated to be of interest and the equation is
    univariate, a list of values is returned; otherwise, the keys in
    a dictionary will indicate which (of all the variables used in
    the expression(s)) variables and solutions were found:

        >>> solve(x**2 - 4)
        [-2, 2]
        >>> solve((x - a)*(y - b))
        [{a: x}, {b: y}]
        >>> solve([x - 3, y - 1])
        {x: 3, y: 1}
        >>> solve([x - 3, y**2 - 1])
        [{x: 3, y: -1}, {x: 3, y: 1}]

    If you pass symbols for which solutions are sought, the output will vary
    depending on the number of symbols you passed, whether you are passing
    a list of expressions or not, and whether a linear system was solved.
    Uniform output is attained by using ``dict=True`` or ``set=True``.

        >>> #### *** feel free to skip to the stars below *** ####
        >>> from sympy import TableForm
        >>> h = [None, ';|;'.join(['e', 's', 'solve(e, s)', 'solve(e, s, dict=True)',
        ... 'solve(e, s, set=True)']).split(';')]
        >>> t = []
        >>> for e, s in [
        ...         (x - y, y),
        ...         (x - y, [x, y]),
        ...         (x**2 - y, [x, y]),
        ...         ([x - 3, y -1], [x, y]),
        ...         ]:
        ...     how = [{}, dict(dict=True), dict(set=True)]
        ...     res = [solve(e, s, **f) for f in how]
        ...     t.append([e, '|', s, '|'] + [res[0], '|', res[1], '|', res[2]])
        ...
        >>> # ******************************************************* #
        >>> TableForm(t, headings=h, alignments="<")
        e              | s      | solve(e, s)  | solve(e, s, dict=True) | solve(e, s, set=True)
        ---------------------------------------------------------------------------------------
        x - y          | y      | [x]          | [{y: x}]               | ([y], {(x,)})
        x - y          | [x, y] | [(y, y)]     | [{x: y}]               | ([x, y], {(y, y)})
        x**2 - y       | [x, y] | [(x, x**2)]  | [{y: x**2}]            | ([x, y], {(x, x**2)})
        [x - 3, y - 1] | [x, y] | {x: 3, y: 1} | [{x: 3, y: 1}]         | ([x, y], {(3, 1)})

        * If any equation does not depend on the symbol(s) given, it will be
          eliminated from the equation set and an answer may be given
          implicitly in terms of variables that were not of interest:

            >>> solve([x - y, y - 3], x)
            {x: y}

    When you pass all but one of the free symbols, an attempt
    is made to find a single solution based on the method of
    undetermined coefficients. If it succeeds, a dictionary of values
    is returned. If you want an algebraic solutions for one
    or more of the symbols, pass the expression to be solved in a list:

        >>> e = a*x + b - 2*x - 3
        >>> solve(e, [a, b])
        {a: 2, b: 3}
        >>> solve([e], [a, b])
        {a: -b/x + (2*x + 3)/x}

    When there is no solution for any given symbol which will make all
    expressions zero, the empty list is returned (or an empty set in
    the tuple when ``set=True``):

        >>> from sympy import sqrt
        >>> solve(3, x)
        []
        >>> solve(x - 3, y)
        []
        >>> solve(sqrt(x) + 1, x, set=True)
        ([x], set())

    When an object other than a Symbol is given as a symbol, it is
    isolated algebraically and an implicit solution may be obtained.
    This is mostly provided as a convenience to save you from replacing
    the object with a Symbol and solving for that Symbol. It will only
    work if the specified object can be replaced with a Symbol using the
    subs method:

        >>> from sympy import exp, Function
        >>> f = Function('f')

        >>> solve(f(x) - x, f(x))
        [x]
        >>> solve(f(x).diff(x) - f(x) - x, f(x).diff(x))
        [x + f(x)]
        >>> solve(f(x).diff(x) - f(x) - x, f(x))
        [-x + Derivative(f(x), x)]
        >>> solve(x + exp(x)**2, exp(x), set=True)
        ([exp(x)], {(-sqrt(-x),), (sqrt(-x),)})

        >>> from sympy import Indexed, IndexedBase, Tuple
        >>> A = IndexedBase('A')
        >>> eqs = Tuple(A[1] + A[2] - 3, A[1] - A[2] + 1)
        >>> solve(eqs, eqs.atoms(Indexed))
        {A[1]: 1, A[2]: 2}

        * To solve for a function within a derivative, use :func:`~.dsolve`.

    To solve for a symbol implicitly, use implicit=True:

        >>> solve(x + exp(x), x)
        [-LambertW(1)]
        >>> solve(x + exp(x), x, implicit=True)
        [-exp(x)]

    It is possible to solve for anything in an expression that can be
    replaced with a symbol using :obj:`~sympy.core.basic.Basic.subs`:

        >>> solve(x + 2 + sqrt(3), x + 2)
        [-sqrt(3)]
        >>> solve((x + 2 + sqrt(3), x + 4 + y), y, x + 2)
        {y: -2 + sqrt(3), x + 2: -sqrt(3)}

        * Nothing heroic is done in this implicit solving so you may end up
          with a symbol still in the solution:

            >>> eqs = (x*y + 3*y + sqrt(3), x + 4 + y)
            >>> solve(eqs, y, x + 2)
            {y: -sqrt(3)/(x + 3), x + 2: -2*x/(x + 3) - 6/(x + 3) + sqrt(3)/(x + 3)}
            >>> solve(eqs, y*x, x)
            {x: -y - 4, x*y: -3*y - sqrt(3)}

        * If you attempt to solve for a number, remember that the number
          you have obtained does not necessarily mean that the value is
          equivalent to the expression obtained:

            >>> solve(sqrt(2) - 1, 1)
            [sqrt(2)]
            >>> solve(x - y + 1, 1)  # /!\ -1 is targeted, too
            [x/(y - 1)]
            >>> [_.subs(z, -1) for _ in solve((x - y + 1).subs(-1, z), 1)]
            [-x + y]

    **Additional Examples**

    ``solve()`` with check=True (default) will run through the symbol tags to
    eliminate unwanted solutions. If no assumptions are included, all possible
    solutions will be returned:

        >>> x = Symbol("x")
        >>> solve(x**2 - 1)
        [-1, 1]

    By setting the ``positive`` flag, only one solution will be returned:

        >>> pos = Symbol("pos", positive=True)
        >>> solve(pos**2 - 1)
        [1]

    When the solutions are checked, those that make any denominator zero
    are automatically excluded. If you do not want to exclude such solutions,
    then use the check=False option:

        >>> from sympy import sin, limit
        >>> solve(sin(x)/x)  # 0 is excluded
        [pi]

    If ``check=False``, then a solution to the numerator being zero is found
    but the value of $x = 0$ is a spurious solution since $\sin(x)/x$ has the well
    known limit (without discontinuity) of 1 at $x = 0$:

        >>> solve(sin(x)/x, check=False)
        [0, pi]

    In the following case, however, the limit exists and is equal to the
    value of $x = 0$ that is excluded when check=True:

        >>> eq = x**2*(1/x - z**2/x)
        >>> solve(eq, x)
        []
        >>> solve(eq, x, check=False)
        [0]
        >>> limit(eq, x, 0, '-')
        0
        >>> limit(eq, x, 0, '+')
        0

    **Solving Relationships**

    When one or more expressions passed to ``solve`` is a relational,
    a relational result is returned (and the ``dict`` and ``set`` flags
    are ignored):

        >>> solve(x < 3)
        (-oo < x) & (x < 3)
        >>> solve([x < 3, x**2 > 4], x)
        ((-oo < x) & (x < -2)) | ((2 < x) & (x < 3))
        >>> solve([x + y - 3, x > 3], x)
        (3 < x) & (x < oo) & Eq(x, 3 - y)

    Although checking of assumptions on symbols in relationals
    is not done, setting assumptions will affect how certain
    relationals might automatically simplify:

        >>> solve(x**2 > 4)
        ((-oo < x) & (x < -2)) | ((2 < x) & (x < oo))

        >>> r = Symbol('r', real=True)
        >>> solve(r**2 > 4)
        (2 < r) | (r < -2)

    There is currently no algorithm in SymPy that allows you to use
    relationships to resolve more than one variable. So the following
    does not determine that ``q < 0`` (and trying to solve for ``r``
    and ``q`` will raise an error):

        >>> from sympy import symbols
        >>> r, q = symbols('r, q', real=True)
        >>> solve([r + q - 3, r > 3], r)
        (3 < r) & Eq(r, 3 - q)

    You can directly call the routine that ``solve`` calls
    when it encounters a relational: :func:`~.reduce_inequalities`.
    It treats Expr like Equality.

        >>> from sympy import reduce_inequalities
        >>> reduce_inequalities([x**2 - 4])
        Eq(x, -2) | Eq(x, 2)

    If each relationship contains only one symbol of interest,
    the expressions can be processed for multiple symbols:

        >>> reduce_inequalities([0 <= x  - 1, y < 3], [x, y])
        (-oo < y) & (1 <= x) & (x < oo) & (y < 3)

    But an error is raised if any relationship has more than one
    symbol of interest:

        >>> reduce_inequalities([0 <= x*y  - 1, y < 3], [x, y])
        Traceback (most recent call last):
        ...
        NotImplementedError:
        inequality has more than one symbol of interest.

    **Disabling High-Order Explicit Solutions**

    When solving polynomial expressions, you might not want explicit solutions
    (which can be quite long). If the expression is univariate, ``CRootOf``
    instances will be returned instead:

        >>> solve(x**3 - x + 1)
        [-1/((-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)) -
        (-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3,
        -(-1/2 + sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3 -
        1/((-1/2 + sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)),
        -(3*sqrt(69)/2 + 27/2)**(1/3)/3 -
        1/(3*sqrt(69)/2 + 27/2)**(1/3)]
        >>> solve(x**3 - x + 1, cubics=False)
        [CRootOf(x**3 - x + 1, 0),
         CRootOf(x**3 - x + 1, 1),
         CRootOf(x**3 - x + 1, 2)]

    If the expression is multivariate, no solution might be returned:

        >>> solve(x**3 - x + a, x, cubics=False)
        []

    Sometimes solutions will be obtained even when a flag is False because the
    expression could be factored. In the following example, the equation can
    be factored as the product of a linear and a quadratic factor so explicit
    solutions (which did not require solving a cubic expression) are obtained:

        >>> eq = x**3 + 3*x**2 + x - 1
        >>> solve(eq, cubics=False)
        [-1, -1 + sqrt(2), -sqrt(2) - 1]

    **Solving Equations Involving Radicals**

    Because of SymPy's use of the principle root, some solutions
    to radical equations will be missed unless check=False:

        >>> from sympy import root
        >>> eq = root(x**3 - 3*x**2, 3) + 1 - x
        >>> solve(eq)
        []
        >>> solve(eq, check=False)
        [1/3]

    In the above example, there is only a single solution to the
    equation. Other expressions will yield spurious roots which
    must be checked manually; roots which give a negative argument
    to odd-powered radicals will also need special checking:

        >>> from sympy import real_root, S
        >>> eq = root(x, 3) - root(x, 5) + S(1)/7
        >>> solve(eq)  # this gives 2 solutions but misses a 3rd
        [CRootOf(7*x**5 - 7*x**3 + 1, 1)**15,
        CRootOf(7*x**5 - 7*x**3 + 1, 2)**15]
        >>> sol = solve(eq, check=False)
        >>> [abs(eq.subs(x,i).n(2)) for i in sol]
        [0.48, 0.e-110, 0.e-110, 0.052, 0.052]

    The first solution is negative so ``real_root`` must be used to see that it
    satisfies the expression:

        >>> abs(real_root(eq.subs(x, sol[0])).n(2))
        0.e-110

    If the roots of the equation are not real then more care will be
    necessary to find the roots, especially for higher order equations.
    Consider the following expression:

        >>> expr = root(x, 3) - root(x, 5)

    We will construct a known value for this expression at x = 3 by selecting
    the 1-th root for each radical:

        >>> expr1 = root(x, 3, 1) - root(x, 5, 1)
        >>> v = expr1.subs(x, -3)

    The ``solve`` function is unable to find any exact roots to this equation:

        >>> eq = Eq(expr, v); eq1 = Eq(expr1, v)
        >>> solve(eq, check=False), solve(eq1, check=False)
        ([], [])

    The function ``unrad``, however, can be used to get a form of the equation
    for which numerical roots can be found:

        >>> from sympy.solvers.solvers import unrad
        >>> from sympy import nroots
        >>> e, (p, cov) = unrad(eq)
        >>> pvals = nroots(e)
        >>> inversion = solve(cov, x)[0]
        >>> xvals = [inversion.subs(p, i) for i in pvals]

    Although ``eq`` or ``eq1`` could have been used to find ``xvals``, the
    solution can only be verified with ``expr1``:

        >>> z = expr - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z.subs(x, xi).n()) < 1e-9]
        []
        >>> z1 = expr1 - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z1.subs(x, xi).n()) < 1e-9]
        [-3.0]

    Parameters
    ==========

    f :
        - a single Expr or Poly that must be zero
        - an Equality
        - a Relational expression
        - a Boolean
        - iterable of one or more of the above

    symbols : (object(s) to solve for) specified as
        - none given (other non-numeric objects will be used)
        - single symbol
        - denested list of symbols
          (e.g., ``solve(f, x, y)``)
        - ordered iterable of symbols
          (e.g., ``solve(f, [x, y])``)

    flags :
        dict=True (default is False)
            Return list (perhaps empty) of solution mappings.
        set=True (default is False)
            Return list of symbols and set of tuple(s) of solution(s).
        exclude=[] (default)
            Do not try to solve for any of the free symbols in exclude;
            if expressions are given, the free symbols in them will
            be extracted automatically.
        check=True (default)
            If False, do not do any testing of solutions. This can be
            useful if you want to include solutions that make any
            denominator zero.
        numerical=True (default)
            Do a fast numerical check if *f* has only one symbol.
        minimal=True (default is False)
            A very fast, minimal testing.
        warn=True (default is False)
            Show a warning if ``checksol()`` could not conclude.
        simplify=True (default)
            Simplify all but polynomials of order 3 or greater before
            returning them and (if check is not False) use the
            general simplify function on the solutions and the
            expression obtained when they are substituted into the
            function which should be zero.
        force=True (default is False)
            Make positive all symbols without assumptions regarding sign.
        rational=True (default)
            Recast Floats as Rational; if this option is not used, the
            system containing Floats may fail to solve because of issues
            with polys. If rational=None, Floats will be recast as
            rationals but the answer will be recast as Floats. If the
            flag is False then nothing will be done to the Floats.
        manual=True (default is False)
            Do not use the polys/matrix method to solve a system of
            equations, solve them one at a time as you might "manually."
        implicit=True (default is False)
            Allows ``solve`` to return a solution for a pattern in terms of
            other functions that contain that pattern; this is only
            needed if the pattern is inside of some invertible function
            like cos, exp, etc.
        particular=True (default is False)
            Instructs ``solve`` to try to find a particular solution to
            a linear system with as many zeros as possible; this is very
            expensive.
        quick=True (default is False; ``particular`` must be True)
            Selects a fast heuristic to find a solution with many zeros
            whereas a value of False uses the very slow method guaranteed
            to find the largest number of zeros possible.
        cubics=True (default)
            Return explicit solutions when cubic expressions are encountered.
            When False, quartics and quintics are disabled, too.
        quartics=True (default)
            Return explicit solutions when quartic expressions are encountered.
            When False, quintics are disabled, too.
        quintics=True (default)
            Return explicit solutions (if possible) when quintic expressions
            are encountered.

    See Also
    ========

    rsolve: For solving recurrence relationships
    sympy.solvers.ode.dsolve: For solving differential equations

    """
    from .inequalities import reduce_inequalities

    # checking/recording flags
    ###########################################################################

    # set solver types explicitly; as soon as one is False
    # all the rest will be False
    hints = ('cubics', 'quartics', 'quintics')
    default = True
    for k in hints:
        default = flags.setdefault(k, bool(flags.get(k, default)))

    # allow solution to contain symbol if True:
    implicit = flags.get('implicit', False)

    # record desire to see warnings
    warn = flags.get('warn', False)

    # this flag will be needed for quick exits below, so record
    # now -- but don't record `dict` yet since it might change
    as_set = flags.get('set', False)

    # keeping track of how f was passed
    bare_f = not iterable(f)

    # check flag usage for particular/quick which should only be used
    # with systems of equations
    if flags.get('quick', None) is not None:
        if not flags.get('particular', None):
            raise ValueError('when using `quick`, `particular` should be True')
    if flags.get('particular', False) and bare_f:
        raise ValueError(filldedent("""
            The 'particular/quick' flag is usually used with systems of
            equations. Either pass your equation in a list or
            consider using a solver like `diophantine` if you are
            looking for a solution in integers."""))

    # sympify everything, creating list of expressions and list of symbols
    ###########################################################################

    def _sympified_list(w):
        return list(map(sympify, w if iterable(w) else [w]))
    f, symbols = (_sympified_list(w) for w in [f, symbols])

    # preprocess symbol(s)
    ###########################################################################

    ordered_symbols = None  # were the symbols in a well defined order?
    if not symbols:
        # get symbols from equations
        symbols = set().union(*[fi.free_symbols for fi in f])
        if len(symbols) < len(f):
            for fi in f:
                pot = preorder_traversal(fi)
                for p in pot:
                    if isinstance(p, AppliedUndef):
                        if not as_set:
                            flags['dict'] = True  # better show symbols
                        symbols.add(p)
                        pot.skip()  # don't go any deeper
        ordered_symbols = False
        symbols = list(ordered(symbols))  # to make it canonical
    else:
        if len(symbols) == 1 and iterable(symbols[0]):
            symbols = symbols[0]
        ordered_symbols = symbols and is_sequence(symbols,
                        include=GeneratorType)
        _symbols = list(uniq(symbols))
        if len(_symbols) != len(symbols):
            ordered_symbols = False
            symbols = list(ordered(symbols))
        else:
            symbols = _symbols

    # check for duplicates
    if len(symbols) != len(set(symbols)):
        raise ValueError('duplicate symbols given')
    # remove those not of interest
    exclude = flags.pop('exclude', set())
    if exclude:
        if isinstance(exclude, Expr):
            exclude = [exclude]
        exclude = set().union(*[e.free_symbols for e in sympify(exclude)])
        symbols = [s for s in symbols if s not in exclude]

    # preprocess equation(s)
    ###########################################################################

    # automatically ignore True values
    if isinstance(f, list):
        f = [s for s in f if s is not S.true]

    # handle canonicalization of equation types
    for i, fi in enumerate(f):
        if isinstance(fi, (Eq, Ne)):
            if 'ImmutableDenseMatrix' in [type(a).__name__ for a in fi.args]:
                fi = fi.lhs - fi.rhs
            else:
                L, R = fi.args
                if isinstance(R, BooleanAtom):
                    L, R = R, L
                if isinstance(L, BooleanAtom):
                    if isinstance(fi, Ne):
                        L = ~L
                    if R.is_Relational:
                        fi = ~R if L is S.false else R
                    elif R.is_Symbol:
                        return L
                    elif R.is_Boolean and (~R).is_Symbol:
                        return ~L
                    else:
                        raise NotImplementedError(filldedent('''
                            Unanticipated argument of Eq when other arg
                            is True or False.
                        '''))
                elif isinstance(fi, Eq):
                    fi = Add(fi.lhs, -fi.rhs, evaluate=False)
            f[i] = fi

        # *** dispatch and handle as a system of relationals
        # **************************************************
        if fi.is_Relational:
            if len(symbols) != 1:
                raise ValueError("can only solve for one symbol at a time")
            if warn and symbols[0].assumptions0:
                warnings.warn(filldedent("""
                    \tWarning: assumptions about variable '%s' are
                    not handled currently.""" % symbols[0]))
            return reduce_inequalities(f, symbols=symbols)

        # convert Poly to expression
        if isinstance(fi, Poly):
            f[i] = fi.as_expr()

        # rewrite hyperbolics in terms of exp if they have symbols of
        # interest
        f[i] = f[i].replace(lambda w: isinstance(w, HyperbolicFunction) and \
            w.has_free(*symbols), lambda w: w.rewrite(exp))

        # if we have a Matrix, we need to iterate over its elements again
        if f[i].is_Matrix:
            try:
                f[i] = f[i].as_explicit()
            except ValueError:
                raise ValueError(
                    "solve cannot handle matrices with symbolic shape."
                )
            bare_f = False
            f.extend(list(f[i]))
            f[i] = S.Zero

        # if we can split it into real and imaginary parts then do so
        freei = f[i].free_symbols
        if freei and all(s.is_extended_real or s.is_imaginary for s in freei):
            fr, fi = f[i].as_real_imag()
            # accept as long as new re, im, arg or atan2 are not introduced
            had = f[i].atoms(re, im, arg, atan2)
            if fr and fi and fr != fi and not any(
                    i.atoms(re, im, arg, atan2) - had for i in (fr, fi)):
                if bare_f:
                    bare_f = False
                f[i: i + 1] = [fr, fi]

    # real/imag handling -----------------------------
    if any(isinstance(fi, (bool, BooleanAtom)) for fi in f):
        if as_set:
            return [], set()
        return []

    for i, fi in enumerate(f):
        # Abs
        while True:
            was = fi
            fi = fi.replace(Abs, lambda arg:
                separatevars(Abs(arg)).rewrite(Piecewise) if arg.has(*symbols)
                else Abs(arg))
            if was == fi:
                break

        for e in fi.find(Abs):
            if e.has(*symbols):
                raise NotImplementedError('solving %s when the argument '
                    'is not real or imaginary.' % e)

        # arg
        fi = fi.replace(arg, lambda a: arg(a).rewrite(atan2).rewrite(atan))

        # save changes
        f[i] = fi

    # see if re(s) or im(s) appear
    freim = [fi for fi in f if fi.has(re, im)]
    if freim:
        irf = []
        for s in symbols:
            if s.is_real or s.is_imaginary:
                continue  # neither re(x) nor im(x) will appear
            # if re(s) or im(s) appear, the auxiliary equation must be present
            if any(fi.has(re(s), im(s)) for fi in freim):
                irf.append((s, re(s) + S.ImaginaryUnit*im(s)))
        if irf:
            for s, rhs in irf:
                f = [fi.xreplace({s: rhs}) for fi in f] + [s - rhs]
                symbols.extend([re(s), im(s)])
            if bare_f:
                bare_f = False
            flags['dict'] = True
    # end of real/imag handling  -----------------------------

    # we can solve for non-symbol entities by replacing them with Dummy symbols
    f, symbols, swap_sym = recast_to_symbols(f, symbols)
    # this set of symbols (perhaps recast) is needed below
    symset = set(symbols)

    # get rid of equations that have no symbols of interest; we don't
    # try to solve them because the user didn't ask and they might be
    # hard to solve; this means that solutions may be given in terms
    # of the eliminated equations e.g. solve((x-y, y-3), x) -> {x: y}
    newf = []
    for fi in f:
        # let the solver handle equations that..
        # - have no symbols but are expressions
        # - have symbols of interest
        # - have no symbols of interest but are constant
        # but when an expression is not constant and has no symbols of
        # interest, it can't change what we obtain for a solution from
        # the remaining equations so we don't include it; and if it's
        # zero it can be removed and if it's not zero, there is no
        # solution for the equation set as a whole
        #
        # The reason for doing this filtering is to allow an answer
        # to be obtained to queries like solve((x - y, y), x); without
        # this mod the return value is []
        ok = False
        if fi.free_symbols & symset:
            ok = True
        else:
            if fi.is_number:
                if fi.is_Number:
                    if fi.is_zero:
                        continue
                    return []
                ok = True
            else:
                if fi.is_constant():
                    ok = True
        if ok:
            newf.append(fi)
    if not newf:
        if as_set:
            return symbols, set()
        return []
    f = newf
    del newf

    # mask off any Object that we aren't going to invert: Derivative,
    # Integral, etc... so that solving for anything that they contain will
    # give an implicit solution
    seen = set()
    non_inverts = set()
    for fi in f:
        pot = preorder_traversal(fi)
        for p in pot:
            if not isinstance(p, Expr) or isinstance(p, Piecewise):
                pass
            elif (isinstance(p, bool) or
                    not p.args or
                    p in symset or
                    p.is_Add or p.is_Mul or
                    p.is_Pow and not implicit or
                    p.is_Function and not implicit) and p.func not in (re, im):
                continue
            elif p not in seen:
                seen.add(p)
                if p.free_symbols & symset:
                    non_inverts.add(p)
                else:
                    continue
            pot.skip()
    del seen
    non_inverts = dict(list(zip(non_inverts, [Dummy() for _ in non_inverts])))
    f = [fi.subs(non_inverts) for fi in f]

    # Both xreplace and subs are needed below: xreplace to force substitution
    # inside Derivative, subs to handle non-straightforward substitutions
    non_inverts = [(v, k.xreplace(swap_sym).subs(swap_sym)) for k, v in non_inverts.items()]

    # rationalize Floats
    floats = False
    if flags.get('rational', True) is not False:
        for i, fi in enumerate(f):
            if fi.has(Float):
                floats = True
                f[i] = nsimplify(fi, rational=True)

    # capture any denominators before rewriting since
    # they may disappear after the rewrite, e.g. issue 14779
    flags['_denominators'] = _simple_dens(f[0], symbols)

    # Any embedded piecewise functions need to be brought out to the
    # top level so that the appropriate strategy gets selected.
    # However, this is necessary only if one of the piecewise
    # functions depends on one of the symbols we are solving for.
    def _has_piecewise(e):
        if e.is_Piecewise:
            return e.has(*symbols)
        return any(_has_piecewise(a) for a in e.args)
    for i, fi in enumerate(f):
        if _has_piecewise(fi):
            f[i] = piecewise_fold(fi)

    # expand angles of sums; in general, expand_trig will allow
    # more roots to be found but this is not a great solultion
    # to not returning a parametric solution, otherwise
    # many values can be returned that have a simple
    # relationship between values
    targs = {t for fi in f for t in fi.atoms(TrigonometricFunction)}
    if len(targs) > 1:
        add, other = sift(targs, lambda x: x.args[0].is_Add, binary=True)
        add, other = [[i for i in l if i.has_free(*symbols)] for l in (add, other)]
        trep = {}
        for t in add:
            a = t.args[0]
            ind, dep = a.as_independent(*symbols)
            if dep in symbols or -dep in symbols:
                # don't let expansion expand wrt anything in ind
                n = Dummy() if not ind.is_Number else ind
                trep[t] = TR10(t.func(dep + n)).xreplace({n: ind})
        if other and len(other) <= 2:
            base = gcd(*[i.args[0] for i in other]) if len(other) > 1 else other[0].args[0]
            for i in other:
                trep[i] = TR11(i, base)
        f = [fi.xreplace(trep) for fi in f]

    #
    # try to get a solution
    ###########################################################################
    if bare_f:
        solution = None
        if len(symbols) != 1:
            solution = _solve_undetermined(f[0], symbols, flags)
        if not solution:
            solution = _solve(f[0], *symbols, **flags)
    else:
        linear, solution = _solve_system(f, symbols, **flags)
    assert type(solution) is list
    assert not solution or type(solution[0]) is dict, solution
    #
    # postprocessing
    ###########################################################################
    # capture as_dict flag now (as_set already captured)
    as_dict = flags.get('dict', False)

    # define how solution will get unpacked
    tuple_format = lambda s: [tuple([i.get(x, x) for x in symbols]) for i in s]
    if as_dict or as_set:
        unpack = None
    elif bare_f:
        if len(symbols) == 1:
            unpack = lambda s: [i[symbols[0]] for i in s]
        elif len(solution) == 1 and len(solution[0]) == len(symbols):
            # undetermined linear coeffs solution
            unpack = lambda s: s[0]
        elif ordered_symbols:
            unpack = tuple_format
        else:
            unpack = lambda s: s
    else:
        if solution:
            if linear and len(solution) == 1:
                # if you want the tuple solution for the linear
                # case, use `set=True`
                unpack = lambda s: s[0]
            elif ordered_symbols:
                unpack = tuple_format
            else:
                unpack = lambda s: s
        else:
            unpack = None

    # Restore masked-off objects
    if non_inverts and type(solution) is list:
        solution = [{k: v.subs(non_inverts) for k, v in s.items()}
            for s in solution]

    # Restore original "symbols" if a dictionary is returned.
    # This is not necessary for
    #   - the single univariate equation case
    #     since the symbol will have been removed from the solution;
    #   - the nonlinear poly_system since that only supports zero-dimensional
    #     systems and those results come back as a list
    #
    # ** unless there were Derivatives with the symbols, but those were handled
    #    above.
    if swap_sym:
        symbols = [swap_sym.get(k, k) for k in symbols]
        for i, sol in enumerate(solution):
            solution[i] = {swap_sym.get(k, k): v.subs(swap_sym)
                      for k, v in sol.items()}

    # Get assumptions about symbols, to filter solutions.
    # Note that if assumptions about a solution can't be verified, it is still
    # returned.
    check = flags.get('check', True)

    # restore floats
    if floats and solution and flags.get('rational', None) is None:
        solution = nfloat(solution, exponent=False)
        # nfloat might reveal more duplicates
        solution = _remove_duplicate_solutions(solution)

    if check and solution:  # assumption checking
        warn = flags.get('warn', False)
        got_None = []  # solutions for which one or more symbols gave None
        no_False = []  # solutions for which no symbols gave False
        for sol in solution:
            v = fuzzy_and(check_assumptions(val, **symb.assumptions0)
                          for symb, val in sol.items())
            if v is False:
                continue
            no_False.append(sol)
            if v is None:
                got_None.append(sol)

        solution = no_False
        if warn and got_None:
            warnings.warn(filldedent("""
                \tWarning: assumptions concerning following solution(s)
                cannot be checked:""" + '\n\t' +
                ', '.join(str(s) for s in got_None)))

    #
    # done
    ###########################################################################

    if not solution:
        if as_set:
            return symbols, set()
        return []

    # make orderings canonical for list of dictionaries
    if not as_set:  # for set, no point in ordering
        solution = [{k: s[k] for k in ordered(s)} for s in solution]
        solution.sort(key=default_sort_key)

    if not (as_set or as_dict):
        return unpack(solution)

    if as_dict:
        return solution

    # set output: (symbols, {t1, t2, ...}) from list of dictionaries;
    # include all symbols for those that like a verbose solution
    # and to resolve any differences in dictionary keys.
    #
    # The set results can easily be used to make a verbose dict as
    #   k, v = solve(eqs, syms, set=True)
    #   sol = [dict(zip(k,i)) for i in v]
    #
    if ordered_symbols:
        k = symbols  # keep preferred order
    else:
        # just unify the symbols for which solutions were found
        k = list(ordered(set(flatten(tuple(i.keys()) for i in solution))))
    return k, {tuple([s.get(ki, ki) for ki in k]) for s in solution}


def _solve_undetermined(g, symbols, flags):
    """solve helper to return a list with one dict (solution) else None

    A direct call to solve_undetermined_coeffs is more flexible and
    can return both multiple solutions and handle more than one independent
    variable. Here, we have to be more cautious to keep from solving
    something that does not look like an undetermined coeffs system --
    to minimize the surprise factor since singularities that cancel are not
    prohibited in solve_undetermined_coeffs.
    """
    if g.free_symbols - set(symbols):
        sol = solve_undetermined_coeffs(g, symbols, **dict(flags, dict=True, set=None))
        if len(sol) == 1:
            return sol


def _solve(f, *symbols, **flags):
    """Return a checked solution for *f* in terms of one or more of the
    symbols in the form of a list of dictionaries.

    If no method is implemented to solve the equation, a NotImplementedError
    will be raised. In the case that conversion of an expression to a Poly
    gives None a ValueError will be raised.
    """

    not_impl_msg = "No algorithms are implemented to solve equation %s"

    if len(symbols) != 1:
        # look for solutions for desired symbols that are independent
        # of symbols already solved for, e.g. if we solve for x = y
        # then no symbol having x in its solution will be returned.

        # First solve for linear symbols (since that is easier and limits
        # solution size) and then proceed with symbols appearing
        # in a non-linear fashion. Ideally, if one is solving a single
        # expression for several symbols, they would have to be
        # appear in factors of an expression, but we do not here
        # attempt factorization.  XXX perhaps handling a Mul
        # should come first in this routine whether there is
        # one or several symbols.
        nonlin_s = []
        got_s = set()
        rhs_s = set()
        result = []
        for s in symbols:
            xi, v = solve_linear(f, symbols=[s])
            if xi == s:
                # no need to check but we should simplify if desired
                if flags.get('simplify', True):
                    v = simplify(v)
                vfree = v.free_symbols
                if vfree & got_s:
                    # was linear, but has redundant relationship
                    # e.g. x - y = 0 has y == x is redundant for x == y
                    # so ignore
                    continue
                rhs_s |= vfree
                got_s.add(xi)
                result.append({xi: v})
            elif xi:  # there might be a non-linear solution if xi is not 0
                nonlin_s.append(s)
        if not nonlin_s:
            return result
        for s in nonlin_s:
            try:
                soln = _solve(f, s, **flags)
                for sol in soln:
                    if sol[s].free_symbols & got_s:
                        # depends on previously solved symbols: ignore
                        continue
                    got_s.add(s)
                    result.append(sol)
            except NotImplementedError:
                continue
        if got_s:
            return result
        else:
            raise NotImplementedError(not_impl_msg % f)

    # solve f for a single variable

    symbol = symbols[0]

    # expand binomials only if it has the unknown symbol
    f = f.replace(lambda e: isinstance(e, binomial) and e.has(symbol),
        lambda e: expand_func(e))

    # checking will be done unless it is turned off before making a
    # recursive call; the variables `checkdens` and `check` are
    # captured here (for reference below) in case flag value changes
    flags['check'] = checkdens = check = flags.pop('check', True)

    # build up solutions if f is a Mul
    if f.is_Mul:
        result = set()
        for m in f.args:
            if m in {S.NegativeInfinity, S.ComplexInfinity, S.Infinity}:
                result = set()
                break
            soln = _vsolve(m, symbol, **flags)
            result.update(set(soln))
        result = [{symbol: v} for v in result]
        if check:
            # all solutions have been checked but now we must
            # check that the solutions do not set denominators
            # in any factor to zero
            dens = flags.get('_denominators', _simple_dens(f, symbols))
            result = [s for s in result if
                not any(checksol(den, s, **flags) for den in
                        dens)]
        # set flags for quick exit at end; solutions for each
        # factor were already checked and simplified
        check = False
        flags['simplify'] = False

    elif f.is_Piecewise:
        result = set()
        if any(e.is_zero for e, c in f.args):
            f = f.simplify()  # failure imminent w/o help

        cond = neg = True
        for expr, cnd in f.args:
            # the explicit condition for this expr is the current cond
            # and none of the previous conditions
            cond = And(neg, cnd)
            neg = And(neg, ~cond)

            if expr.is_zero and cond.simplify() != False:
                raise NotImplementedError(filldedent('''
                    An expression is already zero when %s.
                    This means that in this *region* the solution
                    is zero but solve can only represent discrete,
                    not interval, solutions. If this is a spurious
                    interval it might be resolved with simplification
                    of the Piecewise conditions.''' % cond))
            candidates = _vsolve(expr, symbol, **flags)

            for candidate in candidates:
                if candidate in result:
                    # an unconditional value was already there
                    continue
                try:
                    v = cond.subs(symbol, candidate)
                    _eval_simplify = getattr(v, '_eval_simplify', None)
                    if _eval_simplify is not None:
                        # unconditionally take the simplification of v
                        v = _eval_simplify(ratio=2, measure=lambda x: 1)
                except TypeError:
                    # incompatible type with condition(s)
                    continue
                if v == False:
                    continue
                if v == True:
                    result.add(candidate)
                else:
                    result.add(Piecewise(
                        (candidate, v),
                        (S.NaN, True)))
        # solutions already checked and simplified
        # ****************************************
        return [{symbol: r} for r in result]
    else:
        # first see if it really depends on symbol and whether there
        # is only a linear solution
        f_num, sol = solve_linear(f, symbols=symbols)
        if f_num.is_zero or sol is S.NaN:
            return []
        elif f_num.is_Symbol:
            # no need to check but simplify if desired
            if flags.get('simplify', True):
                sol = simplify(sol)
            return [{f_num: sol}]

        poly = None
        # check for a single Add generator
        if not f_num.is_Add:
            add_args = [i for i in f_num.atoms(Add)
                if symbol in i.free_symbols]
            if len(add_args) == 1:
                gen = add_args[0]
                spart = gen.as_independent(symbol)[1].as_base_exp()[0]
                if spart == symbol:
                    try:
                        poly = Poly(f_num, spart)
                    except PolynomialError:
                        pass

        result = False  # no solution was obtained
        msg = ''  # there is no failure message

        # Poly is generally robust enough to convert anything to
        # a polynomial and tell us the different generators that it
        # contains, so we will inspect the generators identified by
        # polys to figure out what to do.

        # try to identify a single generator that will allow us to solve this
        # as a polynomial, followed (perhaps) by a change of variables if the
        # generator is not a symbol

        try:
            if poly is None:
                poly = Poly(f_num)
            if poly is None:
                raise ValueError('could not convert %s to Poly' % f_num)
        except GeneratorsNeeded:
            simplified_f = simplify(f_num)
            if simplified_f != f_num:
                return _solve(simplified_f, symbol, **flags)
            raise ValueError('expression appears to be a constant')

        gens = [g for g in poly.gens if g.has(symbol)]

        def _as_base_q(x):
            """Return (b**e, q) for x = b**(p*e/q) where p/q is the leading
            Rational of the exponent of x, e.g. exp(-2*x/3) -> (exp(x), 3)
            """
            b, e = x.as_base_exp()
            if e.is_Rational:
                return b, e.q
            if not e.is_Mul:
                return x, 1
            c, ee = e.as_coeff_Mul()
            if c.is_Rational and c is not S.One:  # c could be a Float
                return b**ee, c.q
            return x, 1

        if len(gens) > 1:
            # If there is more than one generator, it could be that the
            # generators have the same base but different powers, e.g.
            #   >>> Poly(exp(x) + 1/exp(x))
            #   Poly(exp(-x) + exp(x), exp(-x), exp(x), domain='ZZ')
            #
            # If unrad was not disabled then there should be no rational
            # exponents appearing as in
            #   >>> Poly(sqrt(x) + sqrt(sqrt(x)))
            #   Poly(sqrt(x) + x**(1/4), sqrt(x), x**(1/4), domain='ZZ')

            bases, qs = list(zip(*[_as_base_q(g) for g in gens]))
            bases = set(bases)

            if len(bases) > 1 or not all(q == 1 for q in qs):
                funcs = {b for b in bases if b.is_Function}

                trig = {_ for _ in funcs if
                    isinstance(_, TrigonometricFunction)}
                other = funcs - trig
                if not other and len(funcs.intersection(trig)) > 1:
                    newf = None
                    if f_num.is_Add and len(f_num.args) == 2:
                        # check for sin(x)**p = cos(x)**p
                        _args = f_num.args
                        t = a, b = [i.atoms(Function).intersection(
                            trig) for i in _args]
                        if all(len(i) == 1 for i in t):
                            a, b = [i.pop() for i in t]
                            if isinstance(a, cos):
                                a, b = b, a
                                _args = _args[::-1]
                            if isinstance(a, sin) and isinstance(b, cos
                                    ) and a.args[0] == b.args[0]:
                                # sin(x) + cos(x) = 0 -> tan(x) + 1 = 0
                                newf, _d = (TR2i(_args[0]/_args[1]) + 1
                                    ).as_numer_denom()
                                if not _d.is_Number:
                                    newf = None
                    if newf is None:
                        newf = TR1(f_num).rewrite(tan)
                    if newf != f_num:
                        # don't check the rewritten form --check
                        # solutions in the un-rewritten form below
                        flags['check'] = False
                        result = _solve(newf, symbol, **flags)
                        flags['check'] = check

                # just a simple case - see if replacement of single function
                # clears all symbol-dependent functions, e.g.
                # log(x) - log(log(x) - 1) - 3 can be solved even though it has
                # two generators.

                if result is False and funcs:
                    funcs = list(ordered(funcs))  # put shallowest function first
                    f1 = funcs[0]
                    t = Dummy('t')
                    # perform the substitution
                    ftry = f_num.subs(f1, t)

                    # if no Functions left, we can proceed with usual solve
                    if not ftry.has(symbol):
                        cv_sols = _solve(ftry, t, **flags)
                        cv_inv = list(ordered(_vsolve(t - f1, symbol, **flags)))[0]
                        result = [{symbol: cv_inv.subs(sol)} for sol in cv_sols]

                if result is False:
                    msg = 'multiple generators %s' % gens

            else:
                # e.g. case where gens are exp(x), exp(-x)
                u = bases.pop()
                t = Dummy('t')
                inv = _vsolve(u - t, symbol, **flags)
                if isinstance(u, (Pow, exp)):
                    # this will be resolved by factor in _tsolve but we might
                    # as well try a simple expansion here to get things in
                    # order so something like the following will work now without
                    # having to factor:
                    #
                    # >>> eq = (exp(I*(-x-2))+exp(I*(x+2)))
                    # >>> eq.subs(exp(x),y)  # fails
                    # exp(I*(-x - 2)) + exp(I*(x + 2))
                    # >>> eq.expand().subs(exp(x),y)  # works
                    # y**I*exp(2*I) + y**(-I)*exp(-2*I)
                    def _expand(p):
                        b, e = p.as_base_exp()
                        e = expand_mul(e)
                        return expand_power_exp(b**e)
                    ftry = f_num.replace(
                        lambda w: w.is_Pow or isinstance(w, exp),
                        _expand).subs(u, t)
                    if not ftry.has(symbol):
                        soln = _solve(ftry, t, **flags)
                        result = [{symbol: i.subs(s)} for i in inv for s in soln]

        elif len(gens) == 1:

            # There is only one generator that we are interested in, but
            # there may have been more than one generator identified by
            # polys (e.g. for symbols other than the one we are interested
            # in) so recast the poly in terms of our generator of interest.
            # Also use composite=True with f_num since Poly won't update
            # poly as documented in issue 8810.

            poly = Poly(f_num, gens[0], composite=True)

            # if we aren't on the tsolve-pass, use roots
            if not flags.pop('tsolve', False):
                soln = None
                deg = poly.degree()
                flags['tsolve'] = True
                hints = ('cubics', 'quartics', 'quintics')
                solvers = {h: flags.get(h) for h in hints}
                soln = roots(poly, **solvers)
                if sum(soln.values()) < deg:
                    # e.g. roots(32*x**5 + 400*x**4 + 2032*x**3 +
                    #            5000*x**2 + 6250*x + 3189) -> {}
                    # so all_roots is used and RootOf instances are
                    # returned *unless* the system is multivariate
                    # or high-order EX domain.
                    try:
                        soln = poly.all_roots()
                    except NotImplementedError:
                        if not flags.get('incomplete', True):
                                raise NotImplementedError(
                                filldedent('''
    Neither high-order multivariate polynomials
    nor sorting of EX-domain polynomials is supported.
    If you want to see any results, pass keyword incomplete=True to
    solve; to see numerical values of roots
    for univariate expressions, use nroots.
    '''))
                        else:
                            pass
                else:
                    soln = list(soln.keys())

                if soln is not None:
                    u = poly.gen
                    if u != symbol:
                        try:
                            t = Dummy('t')
                            inv = _vsolve(u - t, symbol, **flags)
                            soln = {i.subs(t, s) for i in inv for s in soln}
                        except NotImplementedError:
                            # perhaps _tsolve can handle f_num
                            soln = None
                    else:
                        check = False  # only dens need to be checked
                    if soln is not None:
                        if len(soln) > 2:
                            # if the flag wasn't set then unset it since high-order
                            # results are quite long. Perhaps one could base this
                            # decision on a certain critical length of the
                            # roots. In addition, wester test M2 has an expression
                            # whose roots can be shown to be real with the
                            # unsimplified form of the solution whereas only one of
                            # the simplified forms appears to be real.
                            flags['simplify'] = flags.get('simplify', False)
                if soln is not None:
                    result = [{symbol: v} for v in soln]

    # fallback if above fails
    # -----------------------
    if result is False:
        # try unrad
        if flags.pop('_unrad', True):
            try:
                u = unrad(f_num, symbol)
            except (ValueError, NotImplementedError):
                u = False
            if u:
                eq, cov = u
                if cov:
                    isym, ieq = cov
                    inv = _vsolve(ieq, symbol, **flags)[0]
                    rv = {inv.subs(xi) for xi in _solve(eq, isym, **flags)}
                else:
                    try:
                        rv = set(_vsolve(eq, symbol, **flags))
                    except NotImplementedError:
                        rv = None
                if rv is not None:
                    result = [{symbol: v} for v in rv]
                    # if the flag wasn't set then unset it since unrad results
                    # can be quite long or of very high order
                    flags['simplify'] = flags.get('simplify', False)
            else:
                pass  # for coverage

    # try _tsolve
    if result is False:
        flags.pop('tsolve', None)  # allow tsolve to be used on next pass
        try:
            soln = _tsolve(f_num, symbol, **flags)
            if soln is not None:
                result = [{symbol: v} for v in soln]
        except PolynomialError:
            pass
    # ----------- end of fallback ----------------------------

    if result is False:
        raise NotImplementedError('\n'.join([msg, not_impl_msg % f]))

    result = _remove_duplicate_solutions(result)

    if flags.get('simplify', True):
        result = [{k: d[k].simplify() for k in d} for d in result]
        # Simplification might reveal more duplicates
        result = _remove_duplicate_solutions(result)
        # we just simplified the solution so we now set the flag to
        # False so the simplification doesn't happen again in checksol()
        flags['simplify'] = False

    if checkdens:
        # reject any result that makes any denom. affirmatively 0;
        # if in doubt, keep it
        dens = _simple_dens(f, symbols)
        result = [r for r in result if
                  not any(checksol(d, r, **flags)
                          for d in dens)]
    if check:
        # keep only results if the check is not False
        result = [r for r in result if
                  checksol(f_num, r, **flags) is not False]
    return result


def _remove_duplicate_solutions(solutions: list[dict[Expr, Expr]]
                                ) -> list[dict[Expr, Expr]]:
    """Remove duplicates from a list of dicts"""
    solutions_set = set()
    solutions_new = []

    for sol in solutions:
        solset = frozenset(sol.items())
        if solset not in solutions_set:
            solutions_new.append(sol)
            solutions_set.add(solset)

    return solutions_new


def _solve_system(exprs, symbols, **flags):
    """return ``(linear, solution)`` where ``linear`` is True
    if the system was linear, else False; ``solution``
    is a list of dictionaries giving solutions for the symbols
    """
    if not exprs:
        return False, []

    if flags.pop('_split', True):
        # Split the system into connected components
        V = exprs
        symsset = set(symbols)
        exprsyms = {e: e.free_symbols & symsset for e in exprs}
        E = []
        sym_indices = {sym: i for i, sym in enumerate(symbols)}
        for n, e1 in enumerate(exprs):
            for e2 in exprs[:n]:
                # Equations are connected if they share a symbol
                if exprsyms[e1] & exprsyms[e2]:
                    E.append((e1, e2))
        G = V, E
        subexprs = connected_components(G)
        if len(subexprs) > 1:
            subsols = []
            linear = True
            for subexpr in subexprs:
                subsyms = set()
                for e in subexpr:
                    subsyms |= exprsyms[e]
                subsyms = sorted(subsyms, key = lambda x: sym_indices[x])
                flags['_split'] = False  # skip split step
                _linear, subsol = _solve_system(subexpr, subsyms, **flags)
                if linear:
                    linear = linear and _linear
                if not isinstance(subsol, list):
                    subsol = [subsol]
                subsols.append(subsol)
            # Full solution is cartesian product of subsystems
            sols = []
            for soldicts in product(*subsols):
                sols.append(dict(item for sd in soldicts
                    for item in sd.items()))
            return linear, sols

    polys = []
    dens = set()
    failed = []
    result = []
    solved_syms = []
    linear = True
    manual = flags.get('manual', False)
    checkdens = check = flags.get('check', True)

    for j, g in enumerate(exprs):
        dens.update(_simple_dens(g, symbols))
        i, d = _invert(g, *symbols)
        if d in symbols:
            if linear:
                linear = solve_linear(g, 0, [d])[0] == d
        g = d - i
        g = g.as_numer_denom()[0]
        if manual:
            failed.append(g)
            continue

        poly = g.as_poly(*symbols, extension=True)

        if poly is not None:
            polys.append(poly)
        else:
            failed.append(g)

    if polys:
        if all(p.is_linear for p in polys):
            n, m = len(polys), len(symbols)
            matrix = zeros(n, m + 1)

            for i, poly in enumerate(polys):
                for monom, coeff in poly.terms():
                    try:
                        j = monom.index(1)
                        matrix[i, j] = coeff
                    except ValueError:
                        matrix[i, m] = -coeff

            # returns a dictionary ({symbols: values}) or None
            if flags.pop('particular', False):
                result = minsolve_linear_system(matrix, *symbols, **flags)
            else:
                result = solve_linear_system(matrix, *symbols, **flags)
            result = [result] if result else []
            if failed:
                if result:
                    solved_syms = list(result[0].keys())  # there is only one result dict
                else:
                    solved_syms = []
            # linear doesn't change
        else:
            linear = False
            if len(symbols) > len(polys):

                free = set().union(*[p.free_symbols for p in polys])
                free = list(ordered(free.intersection(symbols)))
                got_s = set()
                result = []
                for syms in subsets(free, min(len(free), len(polys))):
                    try:
                        # returns [], None or list of tuples
                        res = solve_poly_system(polys, *syms)
                        if res:
                            for r in set(res):
                                skip = False
                                for r1 in r:
                                    if got_s and any(ss in r1.free_symbols
                                           for ss in got_s):
                                        # sol depends on previously
                                        # solved symbols: discard it
                                        skip = True
                                if not skip:
                                    got_s.update(syms)
                                    result.append(dict(list(zip(syms, r))))
                    except NotImplementedError:
                        pass
                if got_s:
                    solved_syms = list(got_s)
                else:
                    failed.extend([g.as_expr() for g in polys])
            else:
                try:
                    result = solve_poly_system(polys, *symbols)
                    if result:
                        solved_syms = symbols
                        result = [dict(list(zip(solved_syms, r))) for r in set(result)]
                except NotImplementedError:
                    failed.extend([g.as_expr() for g in polys])
                    solved_syms = []

    # convert None or [] to [{}]
    result = result or [{}]

    if failed:
        linear = False
        # For each failed equation, see if we can solve for one of the
        # remaining symbols from that equation. If so, we update the
        # solution set and continue with the next failed equation,
        # repeating until we are done or we get an equation that can't
        # be solved.
        def _ok_syms(e, sort=False):
            rv = e.free_symbols & legal

            # Solve first for symbols that have lower degree in the equation.
            # Ideally we want to solve firstly for symbols that appear linearly
            # with rational coefficients e.g. if e = x*y + z then we should
            # solve for z first.
            def key(sym):
                ep = e.as_poly(sym)
                if ep is None:
                    complexity = (S.Infinity, S.Infinity, S.Infinity)
                else:
                    coeff_syms = ep.LC().free_symbols
                    complexity = (ep.degree(), len(coeff_syms & rv), len(coeff_syms))
                return complexity + (default_sort_key(sym),)

            if sort:
                rv = sorted(rv, key=key)
            return rv

        legal = set(symbols)  # what we are interested in
        # sort so equation with the fewest potential symbols is first
        u = Dummy()  # used in solution checking
        for eq in ordered(failed, lambda _: len(_ok_syms(_))):
            newresult = []
            bad_results = []
            hit = False
            for r in result:
                got_s = set()
                # update eq with everything that is known so far
                eq2 = eq.subs(r)
                # if check is True then we see if it satisfies this
                # equation, otherwise we just accept it
                if check and r:
                    b = checksol(u, u, eq2, minimal=True)
                    if b is not None:
                        # this solution is sufficient to know whether
                        # it is valid or not so we either accept or
                        # reject it, then continue
                        if b:
                            newresult.append(r)
                        else:
                            bad_results.append(r)
                        continue
                # search for a symbol amongst those available that
                # can be solved for
                ok_syms = _ok_syms(eq2, sort=True)
                if not ok_syms:
                    if r:
                        newresult.append(r)
                    break  # skip as it's independent of desired symbols
                for s in ok_syms:
                    try:
                        soln = _vsolve(eq2, s, **flags)
                    except NotImplementedError:
                        continue
                    # put each solution in r and append the now-expanded
                    # result in the new result list; use copy since the
                    # solution for s is being added in-place
                    for sol in soln:
                        if got_s and any(ss in sol.free_symbols for ss in got_s):
                            # sol depends on previously solved symbols: discard it
                            continue
                        rnew = r.copy()
                        for k, v in r.items():
                            rnew[k] = v.subs(s, sol)
                        # and add this new solution
                        rnew[s] = sol
                        # check that it is independent of previous solutions
                        iset = set(rnew.items())
                        for i in newresult:
                            if len(i) < len(iset):
                                # update i with what is known
                                i_items_updated = {(k, v.xreplace(rnew)) for k, v in i.items()}
                                if not i_items_updated - iset:
                                    # this is a superset of a known solution that
                                    # is smaller
                                    break
                        else:
                            # keep it
                            newresult.append(rnew)
                    hit = True
                    got_s.add(s)
                if not hit:
                    raise NotImplementedError('could not solve %s' % eq2)
            else:
                result = newresult
                for b in bad_results:
                    if b in result:
                        result.remove(b)

    if not result:
        return False, []

    # rely on linear/polynomial system solvers to simplify
    # XXX the following tests show that the expressions
    # returned are not the same as they would be if simplify
    # were applied to this:
    #   sympy/solvers/ode/tests/test_systems/test__classify_linear_system
    #   sympy/solvers/tests/test_solvers/test_issue_4886
    # so the docs should be updated to reflect that or else
    # the following should be `bool(failed) or not linear`
    default_simplify = bool(failed)
    if flags.get('simplify', default_simplify):
        for r in result:
            for k in r:
                r[k] = simplify(r[k])
        flags['simplify'] = False  # don't need to do so in checksol now

    if checkdens:
        result = [r for r in result
            if not any(checksol(d, r, **flags) for d in dens)]

    if check and not linear:
        result = [r for r in result
            if not any(checksol(e, r, **flags) is False for e in exprs)]

    result = [r for r in result if r]
    return linear, result


def solve_linear(lhs, rhs=0, symbols=[], exclude=[]):
    r"""
    Return a tuple derived from ``f = lhs - rhs`` that is one of
    the following: ``(0, 1)``, ``(0, 0)``, ``(symbol, solution)``, ``(n, d)``.

    Explanation
    ===========

    ``(0, 1)`` meaning that ``f`` is independent of the symbols in *symbols*
    that are not in *exclude*.

    ``(0, 0)`` meaning that there is no solution to the equation amongst the
    symbols given. If the first element of the tuple is not zero, then the
    function is guaranteed to be dependent on a symbol in *symbols*.

    ``(symbol, solution)`` where symbol appears linearly in the numerator of
    ``f``, is in *symbols* (if given), and is not in *exclude* (if given). No
    simplification is done to ``f`` other than a ``mul=True`` expansion, so the
    solution will correspond strictly to a unique solution.

    ``(n, d)`` where ``n`` and ``d`` are the numerator and denominator of ``f``
    when the numerator was not linear in any symbol of interest; ``n`` will
    never be a symbol unless a solution for that symbol was found (in which case
    the second element is the solution, not the denominator).

    Examples
    ========

    >>> from sympy import cancel, Pow

    ``f`` is independent of the symbols in *symbols* that are not in
    *exclude*:

    >>> from sympy import cos, sin, solve_linear
    >>> from sympy.abc import x, y, z
    >>> eq = y*cos(x)**2 + y*sin(x)**2 - y  # = y*(1 - 1) = 0
    >>> solve_linear(eq)
    (0, 1)
    >>> eq = cos(x)**2 + sin(x)**2  # = 1
    >>> solve_linear(eq)
    (0, 1)
    >>> solve_linear(x, exclude=[x])
    (0, 1)

    The variable ``x`` appears as a linear variable in each of the
    following:

    >>> solve_linear(x + y**2)
    (x, -y**2)
    >>> solve_linear(1/x - y**2)
    (x, y**(-2))

    When not linear in ``x`` or ``y`` then the numerator and denominator are
    returned:

    >>> solve_linear(x**2/y**2 - 3)
    (x**2 - 3*y**2, y**2)

    If the numerator of the expression is a symbol, then ``(0, 0)`` is
    returned if the solution for that symbol would have set any
    denominator to 0:

    >>> eq = 1/(1/x - 2)
    >>> eq.as_numer_denom()
    (x, 1 - 2*x)
    >>> solve_linear(eq)
    (0, 0)

    But automatic rewriting may cause a symbol in the denominator to
    appear in the numerator so a solution will be returned:

    >>> (1/x)**-1
    x
    >>> solve_linear((1/x)**-1)
    (x, 0)

    Use an unevaluated expression to avoid this:

    >>> solve_linear(Pow(1/x, -1, evaluate=False))
    (0, 0)

    If ``x`` is allowed to cancel in the following expression, then it
    appears to be linear in ``x``, but this sort of cancellation is not
    done by ``solve_linear`` so the solution will always satisfy the
    original expression without causing a division by zero error.

    >>> eq = x**2*(1/x - z**2/x)
    >>> solve_linear(cancel(eq))
    (x, 0)
    >>> solve_linear(eq)
    (x**2*(1 - z**2), x)

    A list of symbols for which a solution is desired may be given:

    >>> solve_linear(x + y + z, symbols=[y])
    (y, -x - z)

    A list of symbols to ignore may also be given:

    >>> solve_linear(x + y + z, exclude=[x])
    (y, -x - z)

    (A solution for ``y`` is obtained because it is the first variable
    from the canonically sorted list of symbols that had a linear
    solution.)

    """
    if isinstance(lhs, Eq):
        if rhs:
            raise ValueError(filldedent('''
            If lhs is an Equality, rhs must be 0 but was %s''' % rhs))
        rhs = lhs.rhs
        lhs = lhs.lhs
    dens = None
    eq = lhs - rhs
    n, d = eq.as_numer_denom()
    if not n:
        return S.Zero, S.One

    free = n.free_symbols
    if not symbols:
        symbols = free
    else:
        bad = [s for s in symbols if not s.is_Symbol]
        if bad:
            if len(bad) == 1:
                bad = bad[0]
            if len(symbols) == 1:
                eg = 'solve(%s, %s)' % (eq, symbols[0])
            else:
                eg = 'solve(%s, *%s)' % (eq, list(symbols))
            raise ValueError(filldedent('''
                solve_linear only handles symbols, not %s. To isolate
                non-symbols use solve, e.g. >>> %s <<<.
                             ''' % (bad, eg)))
        symbols = free.intersection(symbols)
    symbols = symbols.difference(exclude)
    if not symbols:
        return S.Zero, S.One

    # derivatives are easy to do but tricky to analyze to see if they
    # are going to disallow a linear solution, so for simplicity we
    # just evaluate the ones that have the symbols of interest
    derivs = defaultdict(list)
    for der in n.atoms(Derivative):
        csym = der.free_symbols & symbols
        for c in csym:
            derivs[c].append(der)

    all_zero = True
    for xi in sorted(symbols, key=default_sort_key):  # canonical order
        # if there are derivatives in this var, calculate them now
        if isinstance(derivs[xi], list):
            derivs[xi] = {der: der.doit() for der in derivs[xi]}
        newn = n.subs(derivs[xi])
        dnewn_dxi = newn.diff(xi)
        # dnewn_dxi can be nonzero if it survives differentation by any
        # of its free symbols
        free = dnewn_dxi.free_symbols
        if dnewn_dxi and (not free or any(dnewn_dxi.diff(s) for s in free) or free == symbols):
            all_zero = False
            if dnewn_dxi is S.NaN:
                break
            if xi not in dnewn_dxi.free_symbols:
                vi = -1/dnewn_dxi*(newn.subs(xi, 0))
                if dens is None:
                    dens = _simple_dens(eq, symbols)
                if not any(checksol(di, {xi: vi}, minimal=True) is True
                          for di in dens):
                    # simplify any trivial integral
                    irep = [(i, i.doit()) for i in vi.atoms(Integral) if
                            i.function.is_number]
                    # do a slight bit of simplification
                    vi = expand_mul(vi.subs(irep))
                    return xi, vi
    if all_zero:
        return S.Zero, S.One
    if n.is_Symbol: # no solution for this symbol was found
        return S.Zero, S.Zero
    return n, d


def minsolve_linear_system(system, *symbols, **flags):
    r"""
    Find a particular solution to a linear system.

    Explanation
    ===========

    In particular, try to find a solution with the minimal possible number
    of non-zero variables using a naive algorithm with exponential complexity.
    If ``quick=True``, a heuristic is used.

    """
    quick = flags.get('quick', False)
    # Check if there are any non-zero solutions at all
    s0 = solve_linear_system(system, *symbols, **flags)
    if not s0 or all(v == 0 for v in s0.values()):
        return s0
    if quick:
        # We just solve the system and try to heuristically find a nice
        # solution.
        s = solve_linear_system(system, *symbols)
        def update(determined, solution):
            delete = []
            for k, v in solution.items():
                solution[k] = v.subs(determined)
                if not solution[k].free_symbols:
                    delete.append(k)
                    determined[k] = solution[k]
            for k in delete:
                del solution[k]
        determined = {}
        update(determined, s)
        while s:
            # NOTE sort by default_sort_key to get deterministic result
            k = max((k for k in s.values()),
                    key=lambda x: (len(x.free_symbols), default_sort_key(x)))
            kfree = k.free_symbols
            x = next(reversed(list(ordered(kfree))))
            if len(kfree) != 1:
                determined[x] = S.Zero
            else:
                val = _vsolve(k, x, check=False)[0]
                if not val and not any(v.subs(x, val) for v in s.values()):
                    determined[x] = S.One
                else:
                    determined[x] = val
            update(determined, s)
        return determined
    else:
        # We try to select n variables which we want to be non-zero.
        # All others will be assumed zero. We try to solve the modified system.
        # If there is a non-trivial solution, just set the free variables to
        # one. If we do this for increasing n, trying all combinations of
        # variables, we will find an optimal solution.
        # We speed up slightly by starting at one less than the number of
        # variables the quick method manages.
        N = len(symbols)
        bestsol = minsolve_linear_system(system, *symbols, quick=True)
        n0 = len([x for x in bestsol.values() if x != 0])
        for n in range(n0 - 1, 1, -1):
            debugf('minsolve: %s', n)
            thissol = None
            for nonzeros in combinations(range(N), n):
                subm = Matrix([system.col(i).T for i in nonzeros] + [system.col(-1).T]).T
                s = solve_linear_system(subm, *[symbols[i] for i in nonzeros])
                if s and not all(v == 0 for v in s.values()):
                    subs = [(symbols[v], S.One) for v in nonzeros]
                    for k, v in s.items():
                        s[k] = v.subs(subs)
                    for sym in symbols:
                        if sym not in s:
                            if symbols.index(sym) in nonzeros:
                                s[sym] = S.One
                            else:
                                s[sym] = S.Zero
                    thissol = s
                    break
            if thissol is None:
                break
            bestsol = thissol
        return bestsol


def solve_linear_system(system, *symbols, **flags):
    r"""
    Solve system of $N$ linear equations with $M$ variables, which means
    both under- and overdetermined systems are supported.

    Explanation
    ===========

    The possible number of solutions is zero, one, or infinite. Respectively,
    this procedure will return None or a dictionary with solutions. In the
    case of underdetermined systems, all arbitrary parameters are skipped.
    This may cause a situation in which an empty dictionary is returned.
    In that case, all symbols can be assigned arbitrary values.

    Input to this function is a $N\times M + 1$ matrix, which means it has
    to be in augmented form. If you prefer to enter $N$ equations and $M$
    unknowns then use ``solve(Neqs, *Msymbols)`` instead. Note: a local
    copy of the matrix is made by this routine so the matrix that is
    passed will not be modified.

    The algorithm used here is fraction-free Gaussian elimination,
    which results, after elimination, in an upper-triangular matrix.
    Then solutions are found using back-substitution. This approach
    is more efficient and compact than the Gauss-Jordan method.

    Examples
    ========

    >>> from sympy import Matrix, solve_linear_system
    >>> from sympy.abc import x, y

    Solve the following system::

           x + 4 y ==  2
        -2 x +   y == 14

    >>> system = Matrix(( (1, 4, 2), (-2, 1, 14)))
    >>> solve_linear_system(system, x, y)
    {x: -6, y: 2}

    A degenerate system returns an empty dictionary:

    >>> system = Matrix(( (0,0,0), (0,0,0) ))
    >>> solve_linear_system(system, x, y)
    {}

    """
    assert system.shape[1] == len(symbols) + 1

    # This is just a wrapper for solve_lin_sys
    eqs = list(system * Matrix(symbols + (-1,)))
    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is not None:
        sol = {sym:val for sym, val in sol.items() if sym != val}
    return sol


def solve_undetermined_coeffs(equ, coeffs, *syms, **flags):
    r"""
    Solve a system of equations in $k$ parameters that is formed by
    matching coefficients in variables ``coeffs`` that are on
    factors dependent on the remaining variables (or those given
    explicitly by ``syms``.

    Explanation
    ===========

    The result of this function is a dictionary with symbolic values of those
    parameters with respect to coefficients in $q$ -- empty if there
    is no solution or coefficients do not appear in the equation -- else
    None (if the system was not recognized). If there is more than one
    solution, the solutions are passed as a list. The output can be modified using
    the same semantics as for `solve` since the flags that are passed are sent
    directly to `solve` so, for example the flag ``dict=True`` will always return a list
    of solutions as dictionaries.

    This function accepts both Equality and Expr class instances.
    The solving process is most efficient when symbols are specified
    in addition to parameters to be determined,  but an attempt to
    determine them (if absent) will be made. If an expected solution is not
    obtained (and symbols were not specified) try specifying them.

    Examples
    ========

    >>> from sympy import Eq, solve_undetermined_coeffs
    >>> from sympy.abc import a, b, c, h, p, k, x, y

    >>> solve_undetermined_coeffs(Eq(a*x + a + b, x/2), [a, b], x)
    {a: 1/2, b: -1/2}
    >>> solve_undetermined_coeffs(a - 2, [a])
    {a: 2}

    The equation can be nonlinear in the symbols:

    >>> X, Y, Z = y, x**y, y*x**y
    >>> eq = a*X + b*Y + c*Z - X - 2*Y - 3*Z
    >>> coeffs = a, b, c
    >>> syms = x, y
    >>> solve_undetermined_coeffs(eq, coeffs, syms)
    {a: 1, b: 2, c: 3}

    And the system can be nonlinear in coefficients, too, but if
    there is only a single solution, it will be returned as a
    dictionary:

    >>> eq = a*x**2 + b*x + c - ((x - h)**2 + 4*p*k)/4/p
    >>> solve_undetermined_coeffs(eq, (h, p, k), x)
    {h: -b/(2*a), k: (4*a*c - b**2)/(4*a), p: 1/(4*a)}

    Multiple solutions are always returned in a list:

    >>> solve_undetermined_coeffs(a**2*x + b - x, [a, b], x)
    [{a: -1, b: 0}, {a: 1, b: 0}]

    Using flag ``dict=True`` (in keeping with semantics in :func:`~.solve`)
    will force the result to always be a list with any solutions
    as elements in that list.

    >>> solve_undetermined_coeffs(a*x - 2*x, [a], dict=True)
    [{a: 2}]
    """
    if not (coeffs and all(i.is_Symbol for i in coeffs)):
        raise ValueError('must provide symbols for coeffs')

    if isinstance(equ, Eq):
        eq = equ.lhs - equ.rhs
    else:
        eq = equ

    ceq = cancel(eq)
    xeq = _mexpand(ceq.as_numer_denom()[0], recursive=True)

    free = xeq.free_symbols
    coeffs = free & set(coeffs)
    if not coeffs:
        return ([], {}) if flags.get('set', None) else []  # solve(0, x) -> []

    if not syms:
        # e.g. A*exp(x) + B - (exp(x) + y) separated into parts that
        # don't/do depend on coeffs gives
        # -(exp(x) + y), A*exp(x) + B
        # then see what symbols are common to both
        # {x} = {x, A, B} - {x, y}
        ind, dep = xeq.as_independent(*coeffs, as_Add=True)
        dfree = dep.free_symbols
        syms = dfree & ind.free_symbols
        if not syms:
            # but if the system looks like (a + b)*x + b - c
            # then {} = {a, b, x} - c
            # so calculate {x} = {a, b, x} - {a, b}
            syms = dfree - set(coeffs)
        if not syms:
            syms = [Dummy()]
    else:
        if len(syms) == 1 and iterable(syms[0]):
            syms = syms[0]
        e, s, _ = recast_to_symbols([xeq], syms)
        xeq = e[0]
        syms = s

    # find the functional forms in which symbols appear

    gens = set(xeq.as_coefficients_dict(*syms).keys()) - {1}
    cset = set(coeffs)
    if any(g.has_xfree(cset) for g in gens):
        return  # a generator contained a coefficient symbol

    # make sure we are working with symbols for generators

    e, gens, _ = recast_to_symbols([xeq], list(gens))
    xeq = e[0]

    # collect coefficients in front of generators

    system = list(collect(xeq, gens, evaluate=False).values())

    # get a solution

    soln = solve(system, coeffs, **flags)

    # unpack unless told otherwise if length is 1

    settings = flags.get('dict', None) or flags.get('set', None)
    if type(soln) is dict or settings or len(soln) != 1:
        return soln
    return soln[0]


def solve_linear_system_LU(matrix, syms):
    """
    Solves the augmented matrix system using ``LUsolve`` and returns a
    dictionary in which solutions are keyed to the symbols of *syms* as ordered.

    Explanation
    ===========

    The matrix must be invertible.

    Examples
    ========

    >>> from sympy import Matrix, solve_linear_system_LU
    >>> from sympy.abc import x, y, z

    >>> solve_linear_system_LU(Matrix([
    ... [1, 2, 0, 1],
    ... [3, 2, 2, 1],
    ... [2, 0, 0, 1]]), [x, y, z])
    {x: 1/2, y: 1/4, z: -1/2}

    See Also
    ========

    LUsolve

    """
    if matrix.rows != matrix.cols - 1:
        raise ValueError("Rows should be equal to columns - 1")
    A = matrix[:matrix.rows, :matrix.rows]
    b = matrix[:, matrix.cols - 1:]
    soln = A.LUsolve(b)
    solutions = {}
    for i in range(soln.rows):
        solutions[syms[i]] = soln[i, 0]
    return solutions


def det_perm(M):
    """
    Return the determinant of *M* by using permutations to select factors.

    Explanation
    ===========

    For sizes larger than 8 the number of permutations becomes prohibitively
    large, or if there are no symbols in the matrix, it is better to use the
    standard determinant routines (e.g., ``M.det()``.)

    See Also
    ========

    det_minor
    det_quick

    """
    args = []
    s = True
    n = M.rows
    list_ = M.flat()
    for perm in generate_bell(n):
        fac = []
        idx = 0
        for j in perm:
            fac.append(list_[idx + j])
            idx += n
        term = Mul(*fac) # disaster with unevaluated Mul -- takes forever for n=7
        args.append(term if s else -term)
        s = not s
    return Add(*args)


def det_minor(M):
    """
    Return the ``det(M)`` computed from minors without
    introducing new nesting in products.

    See Also
    ========

    det_perm
    det_quick

    """
    n = M.rows
    if n == 2:
        return M[0, 0]*M[1, 1] - M[1, 0]*M[0, 1]
    else:
        return sum((1, -1)[i % 2]*Add(*[M[0, i]*d for d in
            Add.make_args(det_minor(M.minor_submatrix(0, i)))])
            if M[0, i] else S.Zero for i in range(n))


def det_quick(M, method=None):
    """
    Return ``det(M)`` assuming that either
    there are lots of zeros or the size of the matrix
    is small. If this assumption is not met, then the normal
    Matrix.det function will be used with method = ``method``.

    See Also
    ========

    det_minor
    det_perm

    """
    if any(i.has(Symbol) for i in M):
        if M.rows < 8 and all(i.has(Symbol) for i in M):
            return det_perm(M)
        return det_minor(M)
    else:
        return M.det(method=method) if method else M.det()


def inv_quick(M):
    """Return the inverse of ``M``, assuming that either
    there are lots of zeros or the size of the matrix
    is small.
    """
    if not all(i.is_Number for i in M):
        if not any(i.is_Number for i in M):
            det = lambda _: det_perm(_)
        else:
            det = lambda _: det_minor(_)
    else:
        return M.inv()
    n = M.rows
    d = det(M)
    if d == S.Zero:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible")
    ret = zeros(n)
    s1 = -1
    for i in range(n):
        s = s1 = -s1
        for j in range(n):
            di = det(M.minor_submatrix(i, j))
            ret[j, i] = s*di/d
            s = -s
    return ret


# these are functions that have multiple inverse values per period
multi_inverses = {
    sin: lambda x: (asin(x), S.Pi - asin(x)),
    cos: lambda x: (acos(x), 2*S.Pi - acos(x)),
}


def _vsolve(e, s, **flags):
    """return list of scalar values for the solution of e for symbol s"""
    return [i[s] for i in _solve(e, s, **flags)]


def _tsolve(eq, sym, **flags):
    """
    Helper for ``_solve`` that solves a transcendental equation with respect
    to the given symbol. Various equations containing powers and logarithms,
    can be solved.

    There is currently no guarantee that all solutions will be returned or
    that a real solution will be favored over a complex one.

    Either a list of potential solutions will be returned or None will be
    returned (in the case that no method was known to get a solution
    for the equation). All other errors (like the inability to cast an
    expression as a Poly) are unhandled.

    Examples
    ========

    >>> from sympy import log, ordered
    >>> from sympy.solvers.solvers import _tsolve as tsolve
    >>> from sympy.abc import x

    >>> list(ordered(tsolve(3**(2*x + 5) - 4, x)))
    [-5/2 + log(2)/log(3), (-5*log(3)/2 + log(2) + I*pi)/log(3)]

    >>> tsolve(log(x) + 2*x, x)
    [LambertW(2)/2]

    """
    if 'tsolve_saw' not in flags:
        flags['tsolve_saw'] = []
    if eq in flags['tsolve_saw']:
        return None
    else:
        flags['tsolve_saw'].append(eq)

    rhs, lhs = _invert(eq, sym)

    if lhs == sym:
        return [rhs]
    try:
        if lhs.is_Add:
            # it's time to try factoring; powdenest is used
            # to try get powers in standard form for better factoring
            f = factor(powdenest(lhs - rhs))
            if f.is_Mul:
                return _vsolve(f, sym, **flags)
            if rhs:
                f = logcombine(lhs, force=flags.get('force', True))
                if f.count(log) != lhs.count(log):
                    if isinstance(f, log):
                        return _vsolve(f.args[0] - exp(rhs), sym, **flags)
                    return _tsolve(f - rhs, sym, **flags)

        elif lhs.is_Pow:
            if lhs.exp.is_Integer:
                if lhs - rhs != eq:
                    return _vsolve(lhs - rhs, sym, **flags)

            if sym not in lhs.exp.free_symbols:
                return _vsolve(lhs.base - rhs**(1/lhs.exp), sym, **flags)

            # _tsolve calls this with Dummy before passing the actual number in.
            if any(t.is_Dummy for t in rhs.free_symbols):
                raise NotImplementedError # _tsolve will call here again...

            # a ** g(x) == 0
            if not rhs:
                # f(x)**g(x) only has solutions where f(x) == 0 and g(x) != 0 at
                # the same place
                sol_base = _vsolve(lhs.base, sym, **flags)
                return [s for s in sol_base if lhs.exp.subs(sym, s) != 0]  # XXX use checksol here?

            # a ** g(x) == b
            if not lhs.base.has(sym):
                if lhs.base == 0:
                    return _vsolve(lhs.exp, sym, **flags) if rhs != 0 else []

                # Gets most solutions...
                if lhs.base == rhs.as_base_exp()[0]:
                    # handles case when bases are equal
                    sol = _vsolve(lhs.exp - rhs.as_base_exp()[1], sym, **flags)
                else:
                    # handles cases when bases are not equal and exp
                    # may or may not be equal
                    f = exp(log(lhs.base)*lhs.exp) - exp(log(rhs))
                    sol = _vsolve(f, sym, **flags)

                # Check for duplicate solutions
                def equal(expr1, expr2):
                    _ = Dummy()
                    eq = checksol(expr1 - _, _, expr2)
                    if eq is None:
                        if nsimplify(expr1) != nsimplify(expr2):
                            return False
                        # they might be coincidentally the same
                        # so check more rigorously
                        eq = expr1.equals(expr2)  # XXX expensive but necessary?
                    return eq

                # Guess a rational exponent
                e_rat = nsimplify(log(abs(rhs))/log(abs(lhs.base)))
                e_rat = simplify(posify(e_rat)[0])
                n, d = fraction(e_rat)
                if expand(lhs.base**n - rhs**d) == 0:
                    sol = [s for s in sol if not equal(lhs.exp.subs(sym, s), e_rat)]
                    sol.extend(_vsolve(lhs.exp - e_rat, sym, **flags))

                return list(set(sol))

            # f(x) ** g(x) == c
            else:
                sol = []
                logform = lhs.exp*log(lhs.base) - log(rhs)
                if logform != lhs - rhs:
                    try:
                        sol.extend(_vsolve(logform, sym, **flags))
                    except NotImplementedError:
                        pass

                # Collect possible solutions and check with substitution later.
                check = []
                if rhs == 1:
                    # f(x) ** g(x) = 1 -- g(x)=0 or f(x)=+-1
                    check.extend(_vsolve(lhs.exp, sym, **flags))
                    check.extend(_vsolve(lhs.base - 1, sym, **flags))
                    check.extend(_vsolve(lhs.base + 1, sym, **flags))
                elif rhs.is_Rational:
                    for d in (i for i in divisors(abs(rhs.p)) if i != 1):
                        e, t = integer_log(rhs.p, d)
                        if not t:
                            continue  # rhs.p != d**b
                        for s in divisors(abs(rhs.q)):
                            if s**e== rhs.q:
                                r = Rational(d, s)
                                check.extend(_vsolve(lhs.base - r, sym, **flags))
                                check.extend(_vsolve(lhs.base + r, sym, **flags))
                                check.extend(_vsolve(lhs.exp - e, sym, **flags))
                elif rhs.is_irrational:
                    b_l, e_l = lhs.base.as_base_exp()
                    n, d = (e_l*lhs.exp).as_numer_denom()
                    b, e = sqrtdenest(rhs).as_base_exp()
                    check = [sqrtdenest(i) for i in (_vsolve(lhs.base - b, sym, **flags))]
                    check.extend([sqrtdenest(i) for i in (_vsolve(lhs.exp - e, sym, **flags))])
                    if e_l*d != 1:
                        check.extend(_vsolve(b_l**n - rhs**(e_l*d), sym, **flags))
                for s in check:
                    ok = checksol(eq, sym, s)
                    if ok is None:
                        ok = eq.subs(sym, s).equals(0)
                    if ok:
                        sol.append(s)
                return list(set(sol))

        elif lhs.is_Function and len(lhs.args) == 1:
            if lhs.func in multi_inverses:
                # sin(x) = 1/3 -> x - asin(1/3) & x - (pi - asin(1/3))
                soln = []
                for i in multi_inverses[type(lhs)](rhs):
                    soln.extend(_vsolve(lhs.args[0] - i, sym, **flags))
                return list(set(soln))
            elif lhs.func == LambertW:
                return _vsolve(lhs.args[0] - rhs*exp(rhs), sym, **flags)

        rewrite = lhs.rewrite(exp)
        rewrite = rebuild(rewrite) # avoid rewrites involving evaluate=False
        if rewrite != lhs:
            return _vsolve(rewrite - rhs, sym, **flags)
    except NotImplementedError:
        pass

    # maybe it is a lambert pattern
    if flags.pop('bivariate', True):
        # lambert forms may need some help being recognized, e.g. changing
        # 2**(3*x) + x**3*log(2)**3 + 3*x**2*log(2)**2 + 3*x*log(2) + 1
        # to 2**(3*x) + (x*log(2) + 1)**3

        # make generator in log have exponent of 1
        logs = eq.atoms(log)
        spow = min(
            {i.exp for j in logs for i in j.atoms(Pow)
             if i.base == sym} or {1})
        if spow != 1:
            p = sym**spow
            u = Dummy('bivariate-cov')
            ueq = eq.subs(p, u)
            if not ueq.has_free(sym):
                sol = _vsolve(ueq, u, **flags)
                inv = _vsolve(p - u, sym)
                return [i.subs(u, s) for i in inv for s in sol]

        g = _filtered_gens(eq.as_poly(), sym)
        up_or_log = set()
        for gi in g:
            if isinstance(gi, (exp, log)) or (gi.is_Pow and gi.base == S.Exp1):
                up_or_log.add(gi)
            elif gi.is_Pow:
                gisimp = powdenest(expand_power_exp(gi))
                if gisimp.is_Pow and sym in gisimp.exp.free_symbols:
                    up_or_log.add(gi)
        eq_down = expand_log(expand_power_exp(eq)).subs(
            dict(list(zip(up_or_log, [0]*len(up_or_log)))))
        eq = expand_power_exp(factor(eq_down, deep=True) + (eq - eq_down))
        rhs, lhs = _invert(eq, sym)
        if lhs.has(sym):
            try:
                poly = lhs.as_poly()
                g = _filtered_gens(poly, sym)
                _eq = lhs - rhs
                sols = _solve_lambert(_eq, sym, g)
                # use a simplified form if it satisfies eq
                # and has fewer operations
                for n, s in enumerate(sols):
                    ns = nsimplify(s)
                    if ns != s and ns.count_ops() <= s.count_ops():
                        ok = checksol(_eq, sym, ns)
                        if ok is None:
                            ok = _eq.subs(sym, ns).equals(0)
                        if ok:
                            sols[n] = ns
                return sols
            except NotImplementedError:
                # maybe it's a convoluted function
                if len(g) == 2:
                    try:
                        gpu = bivariate_type(lhs - rhs, *g)
                        if gpu is None:
                            raise NotImplementedError
                        g, p, u = gpu
                        flags['bivariate'] = False
                        inversion = _tsolve(g - u, sym, **flags)
                        if inversion:
                            sol = _vsolve(p, u, **flags)
                            return list({i.subs(u, s)
                                for i in inversion for s in sol})
                    except NotImplementedError:
                        pass
                else:
                    pass

    if flags.pop('force', True):
        flags['force'] = False
        pos, reps = posify(lhs - rhs)
        if rhs == S.ComplexInfinity:
            return []
        for u, s in reps.items():
            if s == sym:
                break
        else:
            u = sym
        if pos.has(u):
            try:
                soln = _vsolve(pos, u, **flags)
                return [s.subs(reps) for s in soln]
            except NotImplementedError:
                pass
        else:
            pass  # here for coverage

    return  # here for coverage


# TODO: option for calculating J numerically

@conserve_mpmath_dps
def nsolve(*args, dict=False, **kwargs):
    r"""
    Solve a nonlinear equation system numerically: ``nsolve(f, [args,] x0,
    modules=['mpmath'], **kwargs)``.

    Explanation
    ===========

    ``f`` is a vector function of symbolic expressions representing the system.
    *args* are the variables. If there is only one variable, this argument can
    be omitted. ``x0`` is a starting vector close to a solution.

    Use the modules keyword to specify which modules should be used to
    evaluate the function and the Jacobian matrix. Make sure to use a module
    that supports matrices. For more information on the syntax, please see the
    docstring of ``lambdify``.

    If the keyword arguments contain ``dict=True`` (default is False) ``nsolve``
    will return a list (perhaps empty) of solution mappings. This might be
    especially useful if you want to use ``nsolve`` as a fallback to solve since
    using the dict argument for both methods produces return values of
    consistent type structure. Please note: to keep this consistent with
    ``solve``, the solution will be returned in a list even though ``nsolve``
    (currently at least) only finds one solution at a time.

    Overdetermined systems are supported.

    Examples
    ========

    >>> from sympy import Symbol, nsolve
    >>> import mpmath
    >>> mpmath.mp.dps = 15
    >>> x1 = Symbol('x1')
    >>> x2 = Symbol('x2')
    >>> f1 = 3 * x1**2 - 2 * x2**2 - 1
    >>> f2 = x1**2 - 2 * x1 + x2**2 + 2 * x2 - 8
    >>> print(nsolve((f1, f2), (x1, x2), (-1, 1)))
    Matrix([[-1.19287309935246], [1.27844411169911]])

    For one-dimensional functions the syntax is simplified:

    >>> from sympy import sin, nsolve
    >>> from sympy.abc import x
    >>> nsolve(sin(x), x, 2)
    3.14159265358979
    >>> nsolve(sin(x), 2)
    3.14159265358979

    To solve with higher precision than the default, use the prec argument:

    >>> from sympy import cos
    >>> nsolve(cos(x) - x, 1)
    0.739085133215161
    >>> nsolve(cos(x) - x, 1, prec=50)
    0.73908513321516064165531208767387340401341175890076
    >>> cos(_)
    0.73908513321516064165531208767387340401341175890076

    To solve for complex roots of real functions, a nonreal initial point
    must be specified:

    >>> from sympy import I
    >>> nsolve(x**2 + 2, I)
    1.4142135623731*I

    ``mpmath.findroot`` is used and you can find their more extensive
    documentation, especially concerning keyword parameters and
    available solvers. Note, however, that functions which are very
    steep near the root, the verification of the solution may fail. In
    this case you should use the flag ``verify=False`` and
    independently verify the solution.

    >>> from sympy import cos, cosh
    >>> f = cos(x)*cosh(x) - 1
    >>> nsolve(f, 3.14*100)
    Traceback (most recent call last):
    ...
    ValueError: Could not find root within given tolerance. (1.39267e+230 > 2.1684e-19)
    >>> ans = nsolve(f, 3.14*100, verify=False); ans
    312.588469032184
    >>> f.subs(x, ans).n(2)
    2.1e+121
    >>> (f/f.diff(x)).subs(x, ans).n(2)
    7.4e-15

    One might safely skip the verification if bounds of the root are known
    and a bisection method is used:

    >>> bounds = lambda i: (3.14*i, 3.14*(i + 1))
    >>> nsolve(f, bounds(100), solver='bisect', verify=False)
    315.730061685774

    Alternatively, a function may be better behaved when the
    denominator is ignored. Since this is not always the case, however,
    the decision of what function to use is left to the discretion of
    the user.

    >>> eq = x**2/(1 - x)/(1 - 2*x)**2 - 100
    >>> nsolve(eq, 0.46)
    Traceback (most recent call last):
    ...
    ValueError: Could not find root within given tolerance. (10000 > 2.1684e-19)
    Try another starting point or tweak arguments.
    >>> nsolve(eq.as_numer_denom()[0], 0.46)
    0.46792545969349058

    """
    # there are several other SymPy functions that use method= so
    # guard against that here
    if 'method' in kwargs:
        raise ValueError(filldedent('''
            Keyword "method" should not be used in this context.  When using
            some mpmath solvers directly, the keyword "method" is
            used, but when using nsolve (and findroot) the keyword to use is
            "solver".'''))

    if 'prec' in kwargs:
        import mpmath
        mpmath.mp.dps = kwargs.pop('prec')

    # keyword argument to return result as a dictionary
    as_dict = dict
    from builtins import dict  # to unhide the builtin

    # interpret arguments
    if len(args) == 3:
        f = args[0]
        fargs = args[1]
        x0 = args[2]
        if iterable(fargs) and iterable(x0):
            if len(x0) != len(fargs):
                raise TypeError('nsolve expected exactly %i guess vectors, got %i'
                                % (len(fargs), len(x0)))
    elif len(args) == 2:
        f = args[0]
        fargs = None
        x0 = args[1]
        if iterable(f):
            raise TypeError('nsolve expected 3 arguments, got 2')
    elif len(args) < 2:
        raise TypeError('nsolve expected at least 2 arguments, got %i'
                        % len(args))
    else:
        raise TypeError('nsolve expected at most 3 arguments, got %i'
                        % len(args))
    modules = kwargs.get('modules', ['mpmath'])
    if iterable(f):
        f = list(f)
        for i, fi in enumerate(f):
            if isinstance(fi, Eq):
                f[i] = fi.lhs - fi.rhs
        f = Matrix(f).T
    if iterable(x0):
        x0 = list(x0)
    if not isinstance(f, Matrix):
        # assume it's a SymPy expression
        if isinstance(f, Eq):
            f = f.lhs - f.rhs
        elif f.is_Relational:
            raise TypeError('nsolve cannot accept inequalities')
        syms = f.free_symbols
        if fargs is None:
            fargs = syms.copy().pop()
        if not (len(syms) == 1 and (fargs in syms or fargs[0] in syms)):
            raise ValueError(filldedent('''
                expected a one-dimensional and numerical function'''))

        # the function is much better behaved if there is no denominator
        # but sending the numerator is left to the user since sometimes
        # the function is better behaved when the denominator is present
        # e.g., issue 11768

        f = lambdify(fargs, f, modules)
        x = sympify(findroot(f, x0, **kwargs))
        if as_dict:
            return [{fargs: x}]
        return x

    if len(fargs) > f.cols:
        raise NotImplementedError(filldedent('''
            need at least as many equations as variables'''))
    verbose = kwargs.get('verbose', False)
    if verbose:
        print('f(x):')
        print(f)
    # derive Jacobian
    J = f.jacobian(fargs)
    if verbose:
        print('J(x):')
        print(J)
    # create functions
    f = lambdify(fargs, f.T, modules)
    J = lambdify(fargs, J, modules)
    # solve the system numerically
    x = findroot(f, x0, J=J, **kwargs)
    if as_dict:
        return [dict(zip(fargs, [sympify(xi) for xi in x]))]
    return Matrix(x)


def _invert(eq, *symbols, **kwargs):
    """
    Return tuple (i, d) where ``i`` is independent of *symbols* and ``d``
    contains symbols.

    Explanation
    ===========

    ``i`` and ``d`` are obtained after recursively using algebraic inversion
    until an uninvertible ``d`` remains. If there are no free symbols then
    ``d`` will be zero. Some (but not necessarily all) solutions to the
    expression ``i - d`` will be related to the solutions of the original
    expression.

    Examples
    ========

    >>> from sympy.solvers.solvers import _invert as invert
    >>> from sympy import sqrt, cos
    >>> from sympy.abc import x, y
    >>> invert(x - 3)
    (3, x)
    >>> invert(3)
    (3, 0)
    >>> invert(2*cos(x) - 1)
    (1/2, cos(x))
    >>> invert(sqrt(x) - 3)
    (3, sqrt(x))
    >>> invert(sqrt(x) + y, x)
    (-y, sqrt(x))
    >>> invert(sqrt(x) + y, y)
    (-sqrt(x), y)
    >>> invert(sqrt(x) + y, x, y)
    (0, sqrt(x) + y)

    If there is more than one symbol in a power's base and the exponent
    is not an Integer, then the principal root will be used for the
    inversion:

    >>> invert(sqrt(x + y) - 2)
    (4, x + y)
    >>> invert(sqrt(x + y) + 2)  # note +2 instead of -2
    (4, x + y)

    If the exponent is an Integer, setting ``integer_power`` to True
    will force the principal root to be selected:

    >>> invert(x**2 - 4, integer_power=True)
    (2, x)

    """
    eq = sympify(eq)
    if eq.args:
        # make sure we are working with flat eq
        eq = eq.func(*eq.args)
    free = eq.free_symbols
    if not symbols:
        symbols = free
    if not free & set(symbols):
        return eq, S.Zero

    dointpow = bool(kwargs.get('integer_power', False))

    lhs = eq
    rhs = S.Zero
    while True:
        was = lhs
        while True:
            indep, dep = lhs.as_independent(*symbols)

            # dep + indep == rhs
            if lhs.is_Add:
                # this indicates we have done it all
                if indep.is_zero:
                    break

                lhs = dep
                rhs -= indep

            # dep * indep == rhs
            else:
                # this indicates we have done it all
                if indep is S.One:
                    break

                lhs = dep
                rhs /= indep

        # collect like-terms in symbols
        if lhs.is_Add:
            terms = {}
            for a in lhs.args:
                i, d = a.as_independent(*symbols)
                terms.setdefault(d, []).append(i)
            if any(len(v) > 1 for v in terms.values()):
                args = []
                for d, i in terms.items():
                    if len(i) > 1:
                        args.append(Add(*i)*d)
                    else:
                        args.append(i[0]*d)
                lhs = Add(*args)

        # if it's a two-term Add with rhs = 0 and two powers we can get the
        # dependent terms together, e.g. 3*f(x) + 2*g(x) -> f(x)/g(x) = -2/3
        if lhs.is_Add and not rhs and len(lhs.args) == 2 and \
                not lhs.is_polynomial(*symbols):
            a, b = ordered(lhs.args)
            ai, ad = a.as_independent(*symbols)
            bi, bd = b.as_independent(*symbols)
            if any(_ispow(i) for i in (ad, bd)):
                a_base, a_exp = ad.as_base_exp()
                b_base, b_exp = bd.as_base_exp()
                if a_base == b_base and a_exp.extract_additively(b_exp) is None:
                    # a = -b and exponents do not have canceling terms/factors
                    # e.g. if exponents were 3*x and x then the ratio would have
                    # an exponent of 2*x: one of the roots would be lost
                    rat = powsimp(powdenest(ad/bd))
                    lhs = rat
                    rhs = -bi/ai
                else:
                    rat = ad/bd
                    _lhs = powsimp(ad/bd)
                    if _lhs != rat:
                        lhs = _lhs
                        rhs = -bi/ai
            elif ai == -bi:
                if isinstance(ad, Function) and ad.func == bd.func:
                    if len(ad.args) == len(bd.args) == 1:
                        lhs = ad.args[0] - bd.args[0]
                    elif len(ad.args) == len(bd.args):
                        # should be able to solve
                        # f(x, y) - f(2 - x, 0) == 0 -> x == 1
                        raise NotImplementedError(
                            'equal function with more than 1 argument')
                    else:
                        raise ValueError(
                            'function with different numbers of args')

        elif lhs.is_Mul and any(_ispow(a) for a in lhs.args):
            lhs = powsimp(powdenest(lhs))

        if lhs.is_Function:
            if hasattr(lhs, 'inverse') and lhs.inverse() is not None and len(lhs.args) == 1:
                #                    -1
                # f(x) = g  ->  x = f  (g)
                #
                # /!\ inverse should not be defined if there are multiple values
                # for the function -- these are handled in _tsolve
                #
                rhs = lhs.inverse()(rhs)
                lhs = lhs.args[0]
            elif isinstance(lhs, atan2):
                y, x = lhs.args
                lhs = 2*atan(y/(sqrt(x**2 + y**2) + x))
            elif lhs.func == rhs.func:
                if len(lhs.args) == len(rhs.args) == 1:
                    lhs = lhs.args[0]
                    rhs = rhs.args[0]
                elif len(lhs.args) == len(rhs.args):
                    # should be able to solve
                    # f(x, y) == f(2, 3) -> x == 2
                    # f(x, x + y) == f(2, 3) -> x == 2
                    raise NotImplementedError(
                        'equal function with more than 1 argument')
                else:
                    raise ValueError(
                        'function with different numbers of args')


        if rhs and lhs.is_Pow and lhs.exp.is_Integer and lhs.exp < 0:
            lhs = 1/lhs
            rhs = 1/rhs

        # base**a = b -> base = b**(1/a) if
        #    a is an Integer and dointpow=True (this gives real branch of root)
        #    a is not an Integer and the equation is multivariate and the
        #      base has more than 1 symbol in it
        # The rationale for this is that right now the multi-system solvers
        # doesn't try to resolve generators to see, for example, if the whole
        # system is written in terms of sqrt(x + y) so it will just fail, so we
        # do that step here.
        if lhs.is_Pow and (
            lhs.exp.is_Integer and dointpow or not lhs.exp.is_Integer and
                len(symbols) > 1 and len(lhs.base.free_symbols & set(symbols)) > 1):
            rhs = rhs**(1/lhs.exp)
            lhs = lhs.base

        if lhs == was:
            break
    return rhs, lhs


def unrad(eq, *syms, **flags):
    """
    Remove radicals with symbolic arguments and return (eq, cov),
    None, or raise an error.

    Explanation
    ===========

    None is returned if there are no radicals to remove.

    NotImplementedError is raised if there are radicals and they cannot be
    removed or if the relationship between the original symbols and the
    change of variable needed to rewrite the system as a polynomial cannot
    be solved.

    Otherwise the tuple, ``(eq, cov)``, is returned where:

    *eq*, ``cov``
        *eq* is an equation without radicals (in the symbol(s) of
        interest) whose solutions are a superset of the solutions to the
        original expression. *eq* might be rewritten in terms of a new
        variable; the relationship to the original variables is given by
        ``cov`` which is a list containing ``v`` and ``v**p - b`` where
        ``p`` is the power needed to clear the radical and ``b`` is the
        radical now expressed as a polynomial in the symbols of interest.
        For example, for sqrt(2 - x) the tuple would be
        ``(c, c**2 - 2 + x)``. The solutions of *eq* will contain
        solutions to the original equation (if there are any).

    *syms*
        An iterable of symbols which, if provided, will limit the focus of
        radical removal: only radicals with one or more of the symbols of
        interest will be cleared. All free symbols are used if *syms* is not
        set.

    *flags* are used internally for communication during recursive calls.
    Two options are also recognized:

        ``take``, when defined, is interpreted as a single-argument function
        that returns True if a given Pow should be handled.

    Radicals can be removed from an expression if:

        *   All bases of the radicals are the same; a change of variables is
            done in this case.
        *   If all radicals appear in one term of the expression.
        *   There are only four terms with sqrt() factors or there are less than
            four terms having sqrt() factors.
        *   There are only two terms with radicals.

    Examples
    ========

    >>> from sympy.solvers.solvers import unrad
    >>> from sympy.abc import x
    >>> from sympy import sqrt, Rational, root

    >>> unrad(sqrt(x)*x**Rational(1, 3) + 2)
    (x**5 - 64, [])
    >>> unrad(sqrt(x) + root(x + 1, 3))
    (-x**3 + x**2 + 2*x + 1, [])
    >>> eq = sqrt(x) + root(x, 3) - 2
    >>> unrad(eq)
    (_p**3 + _p**2 - 2, [_p, _p**6 - x])

    """

    uflags = {"check": False, "simplify": False}

    def _cov(p, e):
        if cov:
            # XXX - uncovered
            oldp, olde = cov
            if Poly(e, p).degree(p) in (1, 2):
                cov[:] = [p, olde.subs(oldp, _vsolve(e, p, **uflags)[0])]
            else:
                raise NotImplementedError
        else:
            cov[:] = [p, e]

    def _canonical(eq, cov):
        if cov:
            # change symbol to vanilla so no solutions are eliminated
            p, e = cov
            rep = {p: Dummy(p.name)}
            eq = eq.xreplace(rep)
            cov = [p.xreplace(rep), e.xreplace(rep)]

        # remove constants and powers of factors since these don't change
        # the location of the root; XXX should factor or factor_terms be used?
        eq = factor_terms(_mexpand(eq.as_numer_denom()[0], recursive=True), clear=True)
        if eq.is_Mul:
            args = []
            for f in eq.args:
                if f.is_number:
                    continue
                if f.is_Pow:
                    args.append(f.base)
                else:
                    args.append(f)
            eq = Mul(*args)  # leave as Mul for more efficient solving

        # make the sign canonical
        margs = list(Mul.make_args(eq))
        changed = False
        for i, m in enumerate(margs):
            if m.could_extract_minus_sign():
                margs[i] = -m
                changed = True
        if changed:
            eq = Mul(*margs, evaluate=False)

        return eq, cov

    def _Q(pow):
        # return leading Rational of denominator of Pow's exponent
        c = pow.as_base_exp()[1].as_coeff_Mul()[0]
        if not c.is_Rational:
            return S.One
        return c.q

    # define the _take method that will determine whether a term is of interest
    def _take(d):
        # return True if coefficient of any factor's exponent's den is not 1
        for pow in Mul.make_args(d):
            if not pow.is_Pow:
                continue
            if _Q(pow) == 1:
                continue
            if pow.free_symbols & syms:
                return True
        return False
    _take = flags.setdefault('_take', _take)

    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs  # XXX legacy Eq as Eqn support
    elif not isinstance(eq, Expr):
        return

    cov, nwas, rpt = [flags.setdefault(k, v) for k, v in
        sorted({"cov": [], "n": None, "rpt": 0}.items())]

    # preconditioning
    eq = powdenest(factor_terms(eq, radical=True, clear=True))
    eq = eq.as_numer_denom()[0]
    eq = _mexpand(eq, recursive=True)
    if eq.is_number:
        return

    # see if there are radicals in symbols of interest
    syms = set(syms) or eq.free_symbols  # _take uses this
    poly = eq.as_poly()
    gens = [g for g in poly.gens if _take(g)]
    if not gens:
        return

    # recast poly in terms of eigen-gens
    poly = eq.as_poly(*gens)

    # not a polynomial e.g. 1 + sqrt(x)*exp(sqrt(x)) with gen sqrt(x)
    if poly is None:
        return

    # - an exponent has a symbol of interest (don't handle)
    if any(g.exp.has(*syms) for g in gens):
        return

    def _rads_bases_lcm(poly):
        # if all the bases are the same or all the radicals are in one
        # term, `lcm` will be the lcm of the denominators of the
        # exponents of the radicals
        lcm = 1
        rads = set()
        bases = set()
        for g in poly.gens:
            q = _Q(g)
            if q != 1:
                rads.add(g)
                lcm = ilcm(lcm, q)
                bases.add(g.base)
        return rads, bases, lcm
    rads, bases, lcm = _rads_bases_lcm(poly)

    covsym = Dummy('p', nonnegative=True)

    # only keep in syms symbols that actually appear in radicals;
    # and update gens
    newsyms = set()
    for r in rads:
        newsyms.update(syms & r.free_symbols)
    if newsyms != syms:
        syms = newsyms
    # get terms together that have common generators
    drad = dict(zip(rads, range(len(rads))))
    rterms = {(): []}
    args = Add.make_args(poly.as_expr())
    for t in args:
        if _take(t):
            common = set(t.as_poly().gens).intersection(rads)
            key = tuple(sorted([drad[i] for i in common]))
        else:
            key = ()
        rterms.setdefault(key, []).append(t)
    others = Add(*rterms.pop(()))
    rterms = [Add(*rterms[k]) for k in rterms.keys()]

    # the output will depend on the order terms are processed, so
    # make it canonical quickly
    rterms = list(reversed(list(ordered(rterms))))

    ok = False  # we don't have a solution yet
    depth = sqrt_depth(eq)

    if len(rterms) == 1 and not (rterms[0].is_Add and lcm > 2):
        eq = rterms[0]**lcm - ((-others)**lcm)
        ok = True
    else:
        if len(rterms) == 1 and rterms[0].is_Add:
            rterms = list(rterms[0].args)
        if len(bases) == 1:
            b = bases.pop()
            if len(syms) > 1:
                x = b.free_symbols
            else:
                x = syms
            x = list(ordered(x))[0]
            try:
                inv = _vsolve(covsym**lcm - b, x, **uflags)
                if not inv:
                    raise NotImplementedError
                eq = poly.as_expr().subs(b, covsym**lcm).subs(x, inv[0])
                _cov(covsym, covsym**lcm - b)
                return _canonical(eq, cov)
            except NotImplementedError:
                pass

        if len(rterms) == 2:
            if not others:
                eq = rterms[0]**lcm - (-rterms[1])**lcm
                ok = True
            elif not log(lcm, 2).is_Integer:
                # the lcm-is-power-of-two case is handled below
                r0, r1 = rterms
                if flags.get('_reverse', False):
                    r1, r0 = r0, r1
                i0 = _rads0, _bases0, lcm0 = _rads_bases_lcm(r0.as_poly())
                i1 = _rads1, _bases1, lcm1 = _rads_bases_lcm(r1.as_poly())
                for reverse in range(2):
                    if reverse:
                        i0, i1 = i1, i0
                        r0, r1 = r1, r0
                    _rads1, _, lcm1 = i1
                    _rads1 = Mul(*_rads1)
                    t1 = _rads1**lcm1
                    c = covsym**lcm1 - t1
                    for x in syms:
                        try:
                            sol = _vsolve(c, x, **uflags)
                            if not sol:
                                raise NotImplementedError
                            neweq = r0.subs(x, sol[0]) + covsym*r1/_rads1 + \
                                others
                            tmp = unrad(neweq, covsym)
                            if tmp:
                                eq, newcov = tmp
                                if newcov:
                                    newp, newc = newcov
                                    _cov(newp, c.subs(covsym,
                                        _vsolve(newc, covsym, **uflags)[0]))
                                else:
                                    _cov(covsym, c)
                            else:
                                eq = neweq
                                _cov(covsym, c)
                            ok = True
                            break
                        except NotImplementedError:
                            if reverse:
                                raise NotImplementedError(
                                    'no successful change of variable found')
                            else:
                                pass
                    if ok:
                        break
        elif len(rterms) == 3:
            # two cube roots and another with order less than 5
            # (so an analytical solution can be found) or a base
            # that matches one of the cube root bases
            info = [_rads_bases_lcm(i.as_poly()) for i in rterms]
            RAD = 0
            BASES = 1
            LCM = 2
            if info[0][LCM] != 3:
                info.append(info.pop(0))
                rterms.append(rterms.pop(0))
            elif info[1][LCM] != 3:
                info.append(info.pop(1))
                rterms.append(rterms.pop(1))
            if info[0][LCM] == info[1][LCM] == 3:
                if info[1][BASES] != info[2][BASES]:
                    info[0], info[1] = info[1], info[0]
                    rterms[0], rterms[1] = rterms[1], rterms[0]
                if info[1][BASES] == info[2][BASES]:
                    eq = rterms[0]**3 + (rterms[1] + rterms[2] + others)**3
                    ok = True
                elif info[2][LCM] < 5:
                    # a*root(A, 3) + b*root(B, 3) + others = c
                    a, b, c, d, A, B = [Dummy(i) for i in 'abcdAB']
                    # zz represents the unraded expression into which the
                    # specifics for this case are substituted
                    zz = (c - d)*(A**3*a**9 + 3*A**2*B*a**6*b**3 -
                        3*A**2*a**6*c**3 + 9*A**2*a**6*c**2*d - 9*A**2*a**6*c*d**2 +
                        3*A**2*a**6*d**3 + 3*A*B**2*a**3*b**6 + 21*A*B*a**3*b**3*c**3 -
                        63*A*B*a**3*b**3*c**2*d + 63*A*B*a**3*b**3*c*d**2 -
                        21*A*B*a**3*b**3*d**3 + 3*A*a**3*c**6 - 18*A*a**3*c**5*d +
                        45*A*a**3*c**4*d**2 - 60*A*a**3*c**3*d**3 + 45*A*a**3*c**2*d**4 -
                        18*A*a**3*c*d**5 + 3*A*a**3*d**6 + B**3*b**9 - 3*B**2*b**6*c**3 +
                        9*B**2*b**6*c**2*d - 9*B**2*b**6*c*d**2 + 3*B**2*b**6*d**3 +
                        3*B*b**3*c**6 - 18*B*b**3*c**5*d + 45*B*b**3*c**4*d**2 -
                        60*B*b**3*c**3*d**3 + 45*B*b**3*c**2*d**4 - 18*B*b**3*c*d**5 +
                        3*B*b**3*d**6 - c**9 + 9*c**8*d - 36*c**7*d**2 + 84*c**6*d**3 -
                        126*c**5*d**4 + 126*c**4*d**5 - 84*c**3*d**6 + 36*c**2*d**7 -
                        9*c*d**8 + d**9)
                    def _t(i):
                        b = Mul(*info[i][RAD])
                        return cancel(rterms[i]/b), Mul(*info[i][BASES])
                    aa, AA = _t(0)
                    bb, BB = _t(1)
                    cc = -rterms[2]
                    dd = others
                    eq = zz.xreplace(dict(zip(
                        (a, A, b, B, c, d),
                        (aa, AA, bb, BB, cc, dd))))
                    ok = True
        # handle power-of-2 cases
        if not ok:
            if log(lcm, 2).is_Integer and (not others and
                    len(rterms) == 4 or len(rterms) < 4):
                def _norm2(a, b):
                    return a**2 + b**2 + 2*a*b

                if len(rterms) == 4:
                    # (r0+r1)**2 - (r2+r3)**2
                    r0, r1, r2, r3 = rterms
                    eq = _norm2(r0, r1) - _norm2(r2, r3)
                    ok = True
                elif len(rterms) == 3:
                    # (r1+r2)**2 - (r0+others)**2
                    r0, r1, r2 = rterms
                    eq = _norm2(r1, r2) - _norm2(r0, others)
                    ok = True
                elif len(rterms) == 2:
                    # r0**2 - (r1+others)**2
                    r0, r1 = rterms
                    eq = r0**2 - _norm2(r1, others)
                    ok = True

    new_depth = sqrt_depth(eq) if ok else depth
    rpt += 1  # XXX how many repeats with others unchanging is enough?
    if not ok or (
                nwas is not None and len(rterms) == nwas and
                new_depth is not None and new_depth == depth and
                rpt > 3):
        raise NotImplementedError('Cannot remove all radicals')

    flags.update({"cov": cov, "n": len(rterms), "rpt": rpt})
    neq = unrad(eq, *syms, **flags)
    if neq:
        eq, cov = neq
    eq, cov = _canonical(eq, cov)
    return eq, cov


# delayed imports
from sympy.solvers.bivariate import (
    bivariate_type, _solve_lambert, _filtered_gens)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\latex\_antlr\latexparser.py ===
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated from ../LaTeX.g4, derived from latex2sympy
#     latex2sympy is licensed under the MIT license
#     https://github.com/augustt198/latex2sympy/blob/master/LICENSE.txt
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,91,522,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,
        7,33,2,34,7,34,2,35,7,35,2,36,7,36,2,37,7,37,2,38,7,38,2,39,7,39,
        2,40,7,40,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,5,1,91,8,1,10,1,12,1,94,
        9,1,1,2,1,2,1,2,1,2,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,4,5,4,108,8,4,
        10,4,12,4,111,9,4,1,5,1,5,1,5,1,5,1,5,1,5,5,5,119,8,5,10,5,12,5,
        122,9,5,1,6,1,6,1,6,1,6,1,6,1,6,5,6,130,8,6,10,6,12,6,133,9,6,1,
        7,1,7,1,7,4,7,138,8,7,11,7,12,7,139,3,7,142,8,7,1,8,1,8,1,8,1,8,
        5,8,148,8,8,10,8,12,8,151,9,8,3,8,153,8,8,1,9,1,9,5,9,157,8,9,10,
        9,12,9,160,9,9,1,10,1,10,5,10,164,8,10,10,10,12,10,167,9,10,1,11,
        1,11,3,11,171,8,11,1,12,1,12,1,12,1,12,1,12,1,12,3,12,179,8,12,1,
        13,1,13,1,13,1,13,3,13,185,8,13,1,13,1,13,1,14,1,14,1,14,1,14,3,
        14,193,8,14,1,14,1,14,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,
        15,1,15,3,15,207,8,15,1,15,3,15,210,8,15,5,15,212,8,15,10,15,12,
        15,215,9,15,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,3,
        16,227,8,16,1,16,3,16,230,8,16,5,16,232,8,16,10,16,12,16,235,9,16,
        1,17,1,17,1,17,1,17,1,17,1,17,3,17,243,8,17,1,18,1,18,1,18,1,18,
        1,18,3,18,250,8,18,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,1,19,
        1,19,1,19,1,19,1,19,1,19,1,19,1,19,3,19,268,8,19,1,20,1,20,1,20,
        1,20,1,21,4,21,275,8,21,11,21,12,21,276,1,21,1,21,1,21,1,21,5,21,
        283,8,21,10,21,12,21,286,9,21,1,21,1,21,4,21,290,8,21,11,21,12,21,
        291,3,21,294,8,21,1,22,1,22,3,22,298,8,22,1,22,3,22,301,8,22,1,22,
        3,22,304,8,22,1,22,3,22,307,8,22,3,22,309,8,22,1,22,1,22,1,22,1,
        22,1,22,1,22,1,22,3,22,318,8,22,1,23,1,23,1,23,1,23,1,24,1,24,1,
        24,1,24,1,25,1,25,1,25,1,25,1,25,1,26,5,26,334,8,26,10,26,12,26,
        337,9,26,1,27,1,27,1,27,1,27,1,27,1,27,3,27,345,8,27,1,27,1,27,1,
        27,1,27,1,27,3,27,352,8,27,1,28,1,28,1,28,1,28,1,28,1,28,1,28,1,
        28,1,29,1,29,1,29,1,29,1,30,1,30,1,30,1,30,1,31,1,31,1,32,1,32,3,
        32,374,8,32,1,32,3,32,377,8,32,1,32,3,32,380,8,32,1,32,3,32,383,
        8,32,3,32,385,8,32,1,32,1,32,1,32,1,32,1,32,3,32,392,8,32,1,32,1,
        32,3,32,396,8,32,1,32,3,32,399,8,32,1,32,3,32,402,8,32,1,32,3,32,
        405,8,32,3,32,407,8,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,
        32,1,32,1,32,3,32,420,8,32,1,32,3,32,423,8,32,1,32,1,32,1,32,3,32,
        428,8,32,1,32,1,32,1,32,1,32,1,32,3,32,435,8,32,1,32,1,32,1,32,1,
        32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,3,
        32,453,8,32,1,32,1,32,1,32,1,32,1,32,1,32,3,32,461,8,32,1,33,1,33,
        1,33,1,33,1,33,3,33,468,8,33,1,34,1,34,1,34,1,34,1,34,1,34,1,34,
        1,34,1,34,1,34,1,34,3,34,481,8,34,3,34,483,8,34,1,34,1,34,1,35,1,
        35,1,35,1,35,1,35,3,35,492,8,35,1,36,1,36,1,37,1,37,1,37,1,37,1,
        37,1,37,3,37,502,8,37,1,38,1,38,1,38,1,38,1,38,1,38,3,38,510,8,38,
        1,39,1,39,1,39,1,39,1,39,1,40,1,40,1,40,1,40,1,40,1,40,0,6,2,8,10,
        12,30,32,41,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,
        38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,
        0,9,2,0,79,82,85,86,1,0,15,16,3,0,17,18,65,67,75,75,2,0,77,77,91,
        91,1,0,27,28,2,0,27,27,29,29,1,0,69,71,1,0,37,58,1,0,35,36,563,0,
        82,1,0,0,0,2,84,1,0,0,0,4,95,1,0,0,0,6,99,1,0,0,0,8,101,1,0,0,0,
        10,112,1,0,0,0,12,123,1,0,0,0,14,141,1,0,0,0,16,152,1,0,0,0,18,154,
        1,0,0,0,20,161,1,0,0,0,22,170,1,0,0,0,24,172,1,0,0,0,26,180,1,0,
        0,0,28,188,1,0,0,0,30,196,1,0,0,0,32,216,1,0,0,0,34,242,1,0,0,0,
        36,249,1,0,0,0,38,267,1,0,0,0,40,269,1,0,0,0,42,274,1,0,0,0,44,317,
        1,0,0,0,46,319,1,0,0,0,48,323,1,0,0,0,50,327,1,0,0,0,52,335,1,0,
        0,0,54,338,1,0,0,0,56,353,1,0,0,0,58,361,1,0,0,0,60,365,1,0,0,0,
        62,369,1,0,0,0,64,460,1,0,0,0,66,467,1,0,0,0,68,469,1,0,0,0,70,491,
        1,0,0,0,72,493,1,0,0,0,74,495,1,0,0,0,76,503,1,0,0,0,78,511,1,0,
        0,0,80,516,1,0,0,0,82,83,3,2,1,0,83,1,1,0,0,0,84,85,6,1,-1,0,85,
        86,3,6,3,0,86,92,1,0,0,0,87,88,10,2,0,0,88,89,7,0,0,0,89,91,3,2,
        1,3,90,87,1,0,0,0,91,94,1,0,0,0,92,90,1,0,0,0,92,93,1,0,0,0,93,3,
        1,0,0,0,94,92,1,0,0,0,95,96,3,6,3,0,96,97,5,79,0,0,97,98,3,6,3,0,
        98,5,1,0,0,0,99,100,3,8,4,0,100,7,1,0,0,0,101,102,6,4,-1,0,102,103,
        3,10,5,0,103,109,1,0,0,0,104,105,10,2,0,0,105,106,7,1,0,0,106,108,
        3,8,4,3,107,104,1,0,0,0,108,111,1,0,0,0,109,107,1,0,0,0,109,110,
        1,0,0,0,110,9,1,0,0,0,111,109,1,0,0,0,112,113,6,5,-1,0,113,114,3,
        14,7,0,114,120,1,0,0,0,115,116,10,2,0,0,116,117,7,2,0,0,117,119,
        3,10,5,3,118,115,1,0,0,0,119,122,1,0,0,0,120,118,1,0,0,0,120,121,
        1,0,0,0,121,11,1,0,0,0,122,120,1,0,0,0,123,124,6,6,-1,0,124,125,
        3,16,8,0,125,131,1,0,0,0,126,127,10,2,0,0,127,128,7,2,0,0,128,130,
        3,12,6,3,129,126,1,0,0,0,130,133,1,0,0,0,131,129,1,0,0,0,131,132,
        1,0,0,0,132,13,1,0,0,0,133,131,1,0,0,0,134,135,7,1,0,0,135,142,3,
        14,7,0,136,138,3,18,9,0,137,136,1,0,0,0,138,139,1,0,0,0,139,137,
        1,0,0,0,139,140,1,0,0,0,140,142,1,0,0,0,141,134,1,0,0,0,141,137,
        1,0,0,0,142,15,1,0,0,0,143,144,7,1,0,0,144,153,3,16,8,0,145,149,
        3,18,9,0,146,148,3,20,10,0,147,146,1,0,0,0,148,151,1,0,0,0,149,147,
        1,0,0,0,149,150,1,0,0,0,150,153,1,0,0,0,151,149,1,0,0,0,152,143,
        1,0,0,0,152,145,1,0,0,0,153,17,1,0,0,0,154,158,3,30,15,0,155,157,
        3,22,11,0,156,155,1,0,0,0,157,160,1,0,0,0,158,156,1,0,0,0,158,159,
        1,0,0,0,159,19,1,0,0,0,160,158,1,0,0,0,161,165,3,32,16,0,162,164,
        3,22,11,0,163,162,1,0,0,0,164,167,1,0,0,0,165,163,1,0,0,0,165,166,
        1,0,0,0,166,21,1,0,0,0,167,165,1,0,0,0,168,171,5,89,0,0,169,171,
        3,24,12,0,170,168,1,0,0,0,170,169,1,0,0,0,171,23,1,0,0,0,172,178,
        5,27,0,0,173,179,3,28,14,0,174,179,3,26,13,0,175,176,3,28,14,0,176,
        177,3,26,13,0,177,179,1,0,0,0,178,173,1,0,0,0,178,174,1,0,0,0,178,
        175,1,0,0,0,179,25,1,0,0,0,180,181,5,73,0,0,181,184,5,21,0,0,182,
        185,3,6,3,0,183,185,3,4,2,0,184,182,1,0,0,0,184,183,1,0,0,0,185,
        186,1,0,0,0,186,187,5,22,0,0,187,27,1,0,0,0,188,189,5,74,0,0,189,
        192,5,21,0,0,190,193,3,6,3,0,191,193,3,4,2,0,192,190,1,0,0,0,192,
        191,1,0,0,0,193,194,1,0,0,0,194,195,5,22,0,0,195,29,1,0,0,0,196,
        197,6,15,-1,0,197,198,3,34,17,0,198,213,1,0,0,0,199,200,10,2,0,0,
        200,206,5,74,0,0,201,207,3,44,22,0,202,203,5,21,0,0,203,204,3,6,
        3,0,204,205,5,22,0,0,205,207,1,0,0,0,206,201,1,0,0,0,206,202,1,0,
        0,0,207,209,1,0,0,0,208,210,3,74,37,0,209,208,1,0,0,0,209,210,1,
        0,0,0,210,212,1,0,0,0,211,199,1,0,0,0,212,215,1,0,0,0,213,211,1,
        0,0,0,213,214,1,0,0,0,214,31,1,0,0,0,215,213,1,0,0,0,216,217,6,16,
        -1,0,217,218,3,36,18,0,218,233,1,0,0,0,219,220,10,2,0,0,220,226,
        5,74,0,0,221,227,3,44,22,0,222,223,5,21,0,0,223,224,3,6,3,0,224,
        225,5,22,0,0,225,227,1,0,0,0,226,221,1,0,0,0,226,222,1,0,0,0,227,
        229,1,0,0,0,228,230,3,74,37,0,229,228,1,0,0,0,229,230,1,0,0,0,230,
        232,1,0,0,0,231,219,1,0,0,0,232,235,1,0,0,0,233,231,1,0,0,0,233,
        234,1,0,0,0,234,33,1,0,0,0,235,233,1,0,0,0,236,243,3,38,19,0,237,
        243,3,40,20,0,238,243,3,64,32,0,239,243,3,44,22,0,240,243,3,58,29,
        0,241,243,3,60,30,0,242,236,1,0,0,0,242,237,1,0,0,0,242,238,1,0,
        0,0,242,239,1,0,0,0,242,240,1,0,0,0,242,241,1,0,0,0,243,35,1,0,0,
        0,244,250,3,38,19,0,245,250,3,40,20,0,246,250,3,44,22,0,247,250,
        3,58,29,0,248,250,3,60,30,0,249,244,1,0,0,0,249,245,1,0,0,0,249,
        246,1,0,0,0,249,247,1,0,0,0,249,248,1,0,0,0,250,37,1,0,0,0,251,252,
        5,19,0,0,252,253,3,6,3,0,253,254,5,20,0,0,254,268,1,0,0,0,255,256,
        5,25,0,0,256,257,3,6,3,0,257,258,5,26,0,0,258,268,1,0,0,0,259,260,
        5,21,0,0,260,261,3,6,3,0,261,262,5,22,0,0,262,268,1,0,0,0,263,264,
        5,23,0,0,264,265,3,6,3,0,265,266,5,24,0,0,266,268,1,0,0,0,267,251,
        1,0,0,0,267,255,1,0,0,0,267,259,1,0,0,0,267,263,1,0,0,0,268,39,1,
        0,0,0,269,270,5,27,0,0,270,271,3,6,3,0,271,272,5,27,0,0,272,41,1,
        0,0,0,273,275,5,78,0,0,274,273,1,0,0,0,275,276,1,0,0,0,276,274,1,
        0,0,0,276,277,1,0,0,0,277,284,1,0,0,0,278,279,5,1,0,0,279,280,5,
        78,0,0,280,281,5,78,0,0,281,283,5,78,0,0,282,278,1,0,0,0,283,286,
        1,0,0,0,284,282,1,0,0,0,284,285,1,0,0,0,285,293,1,0,0,0,286,284,
        1,0,0,0,287,289,5,2,0,0,288,290,5,78,0,0,289,288,1,0,0,0,290,291,
        1,0,0,0,291,289,1,0,0,0,291,292,1,0,0,0,292,294,1,0,0,0,293,287,
        1,0,0,0,293,294,1,0,0,0,294,43,1,0,0,0,295,308,7,3,0,0,296,298,3,
        74,37,0,297,296,1,0,0,0,297,298,1,0,0,0,298,300,1,0,0,0,299,301,
        5,90,0,0,300,299,1,0,0,0,300,301,1,0,0,0,301,309,1,0,0,0,302,304,
        5,90,0,0,303,302,1,0,0,0,303,304,1,0,0,0,304,306,1,0,0,0,305,307,
        3,74,37,0,306,305,1,0,0,0,306,307,1,0,0,0,307,309,1,0,0,0,308,297,
        1,0,0,0,308,303,1,0,0,0,309,318,1,0,0,0,310,318,3,42,21,0,311,318,
        5,76,0,0,312,318,3,50,25,0,313,318,3,54,27,0,314,318,3,56,28,0,315,
        318,3,46,23,0,316,318,3,48,24,0,317,295,1,0,0,0,317,310,1,0,0,0,
        317,311,1,0,0,0,317,312,1,0,0,0,317,313,1,0,0,0,317,314,1,0,0,0,
        317,315,1,0,0,0,317,316,1,0,0,0,318,45,1,0,0,0,319,320,5,30,0,0,
        320,321,3,6,3,0,321,322,7,4,0,0,322,47,1,0,0,0,323,324,7,5,0,0,324,
        325,3,6,3,0,325,326,5,31,0,0,326,49,1,0,0,0,327,328,5,72,0,0,328,
        329,5,21,0,0,329,330,3,52,26,0,330,331,5,22,0,0,331,51,1,0,0,0,332,
        334,5,77,0,0,333,332,1,0,0,0,334,337,1,0,0,0,335,333,1,0,0,0,335,
        336,1,0,0,0,336,53,1,0,0,0,337,335,1,0,0,0,338,344,5,68,0,0,339,
        345,5,78,0,0,340,341,5,21,0,0,341,342,3,6,3,0,342,343,5,22,0,0,343,
        345,1,0,0,0,344,339,1,0,0,0,344,340,1,0,0,0,345,351,1,0,0,0,346,
        352,5,78,0,0,347,348,5,21,0,0,348,349,3,6,3,0,349,350,5,22,0,0,350,
        352,1,0,0,0,351,346,1,0,0,0,351,347,1,0,0,0,352,55,1,0,0,0,353,354,
        7,6,0,0,354,355,5,21,0,0,355,356,3,6,3,0,356,357,5,22,0,0,357,358,
        5,21,0,0,358,359,3,6,3,0,359,360,5,22,0,0,360,57,1,0,0,0,361,362,
        5,59,0,0,362,363,3,6,3,0,363,364,5,60,0,0,364,59,1,0,0,0,365,366,
        5,61,0,0,366,367,3,6,3,0,367,368,5,62,0,0,368,61,1,0,0,0,369,370,
        7,7,0,0,370,63,1,0,0,0,371,384,3,62,31,0,372,374,3,74,37,0,373,372,
        1,0,0,0,373,374,1,0,0,0,374,376,1,0,0,0,375,377,3,76,38,0,376,375,
        1,0,0,0,376,377,1,0,0,0,377,385,1,0,0,0,378,380,3,76,38,0,379,378,
        1,0,0,0,379,380,1,0,0,0,380,382,1,0,0,0,381,383,3,74,37,0,382,381,
        1,0,0,0,382,383,1,0,0,0,383,385,1,0,0,0,384,373,1,0,0,0,384,379,
        1,0,0,0,385,391,1,0,0,0,386,387,5,19,0,0,387,388,3,70,35,0,388,389,
        5,20,0,0,389,392,1,0,0,0,390,392,3,72,36,0,391,386,1,0,0,0,391,390,
        1,0,0,0,392,461,1,0,0,0,393,406,7,3,0,0,394,396,3,74,37,0,395,394,
        1,0,0,0,395,396,1,0,0,0,396,398,1,0,0,0,397,399,5,90,0,0,398,397,
        1,0,0,0,398,399,1,0,0,0,399,407,1,0,0,0,400,402,5,90,0,0,401,400,
        1,0,0,0,401,402,1,0,0,0,402,404,1,0,0,0,403,405,3,74,37,0,404,403,
        1,0,0,0,404,405,1,0,0,0,405,407,1,0,0,0,406,395,1,0,0,0,406,401,
        1,0,0,0,407,408,1,0,0,0,408,409,5,19,0,0,409,410,3,66,33,0,410,411,
        5,20,0,0,411,461,1,0,0,0,412,419,5,34,0,0,413,414,3,74,37,0,414,
        415,3,76,38,0,415,420,1,0,0,0,416,417,3,76,38,0,417,418,3,74,37,
        0,418,420,1,0,0,0,419,413,1,0,0,0,419,416,1,0,0,0,419,420,1,0,0,
        0,420,427,1,0,0,0,421,423,3,8,4,0,422,421,1,0,0,0,422,423,1,0,0,
        0,423,424,1,0,0,0,424,428,5,76,0,0,425,428,3,54,27,0,426,428,3,8,
        4,0,427,422,1,0,0,0,427,425,1,0,0,0,427,426,1,0,0,0,428,461,1,0,
        0,0,429,434,5,63,0,0,430,431,5,25,0,0,431,432,3,6,3,0,432,433,5,
        26,0,0,433,435,1,0,0,0,434,430,1,0,0,0,434,435,1,0,0,0,435,436,1,
        0,0,0,436,437,5,21,0,0,437,438,3,6,3,0,438,439,5,22,0,0,439,461,
        1,0,0,0,440,441,5,64,0,0,441,442,5,21,0,0,442,443,3,6,3,0,443,444,
        5,22,0,0,444,461,1,0,0,0,445,452,7,8,0,0,446,447,3,78,39,0,447,448,
        3,76,38,0,448,453,1,0,0,0,449,450,3,76,38,0,450,451,3,78,39,0,451,
        453,1,0,0,0,452,446,1,0,0,0,452,449,1,0,0,0,453,454,1,0,0,0,454,
        455,3,10,5,0,455,461,1,0,0,0,456,457,5,32,0,0,457,458,3,68,34,0,
        458,459,3,10,5,0,459,461,1,0,0,0,460,371,1,0,0,0,460,393,1,0,0,0,
        460,412,1,0,0,0,460,429,1,0,0,0,460,440,1,0,0,0,460,445,1,0,0,0,
        460,456,1,0,0,0,461,65,1,0,0,0,462,463,3,6,3,0,463,464,5,1,0,0,464,
        465,3,66,33,0,465,468,1,0,0,0,466,468,3,6,3,0,467,462,1,0,0,0,467,
        466,1,0,0,0,468,67,1,0,0,0,469,470,5,73,0,0,470,471,5,21,0,0,471,
        472,7,3,0,0,472,473,5,33,0,0,473,482,3,6,3,0,474,480,5,74,0,0,475,
        476,5,21,0,0,476,477,7,1,0,0,477,481,5,22,0,0,478,481,5,15,0,0,479,
        481,5,16,0,0,480,475,1,0,0,0,480,478,1,0,0,0,480,479,1,0,0,0,481,
        483,1,0,0,0,482,474,1,0,0,0,482,483,1,0,0,0,483,484,1,0,0,0,484,
        485,5,22,0,0,485,69,1,0,0,0,486,492,3,6,3,0,487,488,3,6,3,0,488,
        489,5,1,0,0,489,490,3,70,35,0,490,492,1,0,0,0,491,486,1,0,0,0,491,
        487,1,0,0,0,492,71,1,0,0,0,493,494,3,12,6,0,494,73,1,0,0,0,495,501,
        5,73,0,0,496,502,3,44,22,0,497,498,5,21,0,0,498,499,3,6,3,0,499,
        500,5,22,0,0,500,502,1,0,0,0,501,496,1,0,0,0,501,497,1,0,0,0,502,
        75,1,0,0,0,503,509,5,74,0,0,504,510,3,44,22,0,505,506,5,21,0,0,506,
        507,3,6,3,0,507,508,5,22,0,0,508,510,1,0,0,0,509,504,1,0,0,0,509,
        505,1,0,0,0,510,77,1,0,0,0,511,512,5,73,0,0,512,513,5,21,0,0,513,
        514,3,4,2,0,514,515,5,22,0,0,515,79,1,0,0,0,516,517,5,73,0,0,517,
        518,5,21,0,0,518,519,3,4,2,0,519,520,5,22,0,0,520,81,1,0,0,0,59,
        92,109,120,131,139,141,149,152,158,165,170,178,184,192,206,209,213,
        226,229,233,242,249,267,276,284,291,293,297,300,303,306,308,317,
        335,344,351,373,376,379,382,384,391,395,398,401,404,406,419,422,
        427,434,452,460,467,480,482,491,501,509
    ]

class LaTeXParser ( Parser ):

    grammarFileName = "LaTeX.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "','", "'.'", "<INVALID>", "<INVALID>",
                     "<INVALID>", "<INVALID>", "'\\quad'", "'\\qquad'",
                     "<INVALID>", "'\\negmedspace'", "'\\negthickspace'",
                     "'\\left'", "'\\right'", "<INVALID>", "'+'", "'-'",
                     "'*'", "'/'", "'('", "')'", "'{'", "'}'", "'\\{'",
                     "'\\}'", "'['", "']'", "'|'", "'\\right|'", "'\\left|'",
                     "'\\langle'", "'\\rangle'", "'\\lim'", "<INVALID>",
                     "<INVALID>", "'\\sum'", "'\\prod'", "'\\exp'", "'\\log'",
                     "'\\lg'", "'\\ln'", "'\\sin'", "'\\cos'", "'\\tan'",
                     "'\\csc'", "'\\sec'", "'\\cot'", "'\\arcsin'", "'\\arccos'",
                     "'\\arctan'", "'\\arccsc'", "'\\arcsec'", "'\\arccot'",
                     "'\\sinh'", "'\\cosh'", "'\\tanh'", "'\\arsinh'", "'\\arcosh'",
                     "'\\artanh'", "'\\lfloor'", "'\\rfloor'", "'\\lceil'",
                     "'\\rceil'", "'\\sqrt'", "'\\overline'", "'\\times'",
                     "'\\cdot'", "'\\div'", "<INVALID>", "'\\binom'", "'\\dbinom'",
                     "'\\tbinom'", "'\\mathit'", "'_'", "'^'", "':'", "<INVALID>",
                     "<INVALID>", "<INVALID>", "<INVALID>", "'\\neq'", "'<'",
                     "<INVALID>", "'\\leqq'", "'\\leqslant'", "'>'", "<INVALID>",
                     "'\\geqq'", "'\\geqslant'", "'!'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "WS", "THINSPACE",
                      "MEDSPACE", "THICKSPACE", "QUAD", "QQUAD", "NEGTHINSPACE",
                      "NEGMEDSPACE", "NEGTHICKSPACE", "CMD_LEFT", "CMD_RIGHT",
                      "IGNORE", "ADD", "SUB", "MUL", "DIV", "L_PAREN", "R_PAREN",
                      "L_BRACE", "R_BRACE", "L_BRACE_LITERAL", "R_BRACE_LITERAL",
                      "L_BRACKET", "R_BRACKET", "BAR", "R_BAR", "L_BAR",
                      "L_ANGLE", "R_ANGLE", "FUNC_LIM", "LIM_APPROACH_SYM",
                      "FUNC_INT", "FUNC_SUM", "FUNC_PROD", "FUNC_EXP", "FUNC_LOG",
                      "FUNC_LG", "FUNC_LN", "FUNC_SIN", "FUNC_COS", "FUNC_TAN",
                      "FUNC_CSC", "FUNC_SEC", "FUNC_COT", "FUNC_ARCSIN",
                      "FUNC_ARCCOS", "FUNC_ARCTAN", "FUNC_ARCCSC", "FUNC_ARCSEC",
                      "FUNC_ARCCOT", "FUNC_SINH", "FUNC_COSH", "FUNC_TANH",
                      "FUNC_ARSINH", "FUNC_ARCOSH", "FUNC_ARTANH", "L_FLOOR",
                      "R_FLOOR", "L_CEIL", "R_CEIL", "FUNC_SQRT", "FUNC_OVERLINE",
                      "CMD_TIMES", "CMD_CDOT", "CMD_DIV", "CMD_FRAC", "CMD_BINOM",
                      "CMD_DBINOM", "CMD_TBINOM", "CMD_MATHIT", "UNDERSCORE",
                      "CARET", "COLON", "DIFFERENTIAL", "LETTER", "DIGIT",
                      "EQUAL", "NEQ", "LT", "LTE", "LTE_Q", "LTE_S", "GT",
                      "GTE", "GTE_Q", "GTE_S", "BANG", "SINGLE_QUOTES",
                      "SYMBOL" ]

    RULE_math = 0
    RULE_relation = 1
    RULE_equality = 2
    RULE_expr = 3
    RULE_additive = 4
    RULE_mp = 5
    RULE_mp_nofunc = 6
    RULE_unary = 7
    RULE_unary_nofunc = 8
    RULE_postfix = 9
    RULE_postfix_nofunc = 10
    RULE_postfix_op = 11
    RULE_eval_at = 12
    RULE_eval_at_sub = 13
    RULE_eval_at_sup = 14
    RULE_exp = 15
    RULE_exp_nofunc = 16
    RULE_comp = 17
    RULE_comp_nofunc = 18
    RULE_group = 19
    RULE_abs_group = 20
    RULE_number = 21
    RULE_atom = 22
    RULE_bra = 23
    RULE_ket = 24
    RULE_mathit = 25
    RULE_mathit_text = 26
    RULE_frac = 27
    RULE_binom = 28
    RULE_floor = 29
    RULE_ceil = 30
    RULE_func_normal = 31
    RULE_func = 32
    RULE_args = 33
    RULE_limit_sub = 34
    RULE_func_arg = 35
    RULE_func_arg_noparens = 36
    RULE_subexpr = 37
    RULE_supexpr = 38
    RULE_subeq = 39
    RULE_supeq = 40

    ruleNames =  [ "math", "relation", "equality", "expr", "additive", "mp",
                   "mp_nofunc", "unary", "unary_nofunc", "postfix", "postfix_nofunc",
                   "postfix_op", "eval_at", "eval_at_sub", "eval_at_sup",
                   "exp", "exp_nofunc", "comp", "comp_nofunc", "group",
                   "abs_group", "number", "atom", "bra", "ket", "mathit",
                   "mathit_text", "frac", "binom", "floor", "ceil", "func_normal",
                   "func", "args", "limit_sub", "func_arg", "func_arg_noparens",
                   "subexpr", "supexpr", "subeq", "supeq" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    WS=3
    THINSPACE=4
    MEDSPACE=5
    THICKSPACE=6
    QUAD=7
    QQUAD=8
    NEGTHINSPACE=9
    NEGMEDSPACE=10
    NEGTHICKSPACE=11
    CMD_LEFT=12
    CMD_RIGHT=13
    IGNORE=14
    ADD=15
    SUB=16
    MUL=17
    DIV=18
    L_PAREN=19
    R_PAREN=20
    L_BRACE=21
    R_BRACE=22
    L_BRACE_LITERAL=23
    R_BRACE_LITERAL=24
    L_BRACKET=25
    R_BRACKET=26
    BAR=27
    R_BAR=28
    L_BAR=29
    L_ANGLE=30
    R_ANGLE=31
    FUNC_LIM=32
    LIM_APPROACH_SYM=33
    FUNC_INT=34
    FUNC_SUM=35
    FUNC_PROD=36
    FUNC_EXP=37
    FUNC_LOG=38
    FUNC_LG=39
    FUNC_LN=40
    FUNC_SIN=41
    FUNC_COS=42
    FUNC_TAN=43
    FUNC_CSC=44
    FUNC_SEC=45
    FUNC_COT=46
    FUNC_ARCSIN=47
    FUNC_ARCCOS=48
    FUNC_ARCTAN=49
    FUNC_ARCCSC=50
    FUNC_ARCSEC=51
    FUNC_ARCCOT=52
    FUNC_SINH=53
    FUNC_COSH=54
    FUNC_TANH=55
    FUNC_ARSINH=56
    FUNC_ARCOSH=57
    FUNC_ARTANH=58
    L_FLOOR=59
    R_FLOOR=60
    L_CEIL=61
    R_CEIL=62
    FUNC_SQRT=63
    FUNC_OVERLINE=64
    CMD_TIMES=65
    CMD_CDOT=66
    CMD_DIV=67
    CMD_FRAC=68
    CMD_BINOM=69
    CMD_DBINOM=70
    CMD_TBINOM=71
    CMD_MATHIT=72
    UNDERSCORE=73
    CARET=74
    COLON=75
    DIFFERENTIAL=76
    LETTER=77
    DIGIT=78
    EQUAL=79
    NEQ=80
    LT=81
    LTE=82
    LTE_Q=83
    LTE_S=84
    GT=85
    GTE=86
    GTE_Q=87
    GTE_S=88
    BANG=89
    SINGLE_QUOTES=90
    SYMBOL=91

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.11.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class MathContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def relation(self):
            return self.getTypedRuleContext(LaTeXParser.RelationContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_math




    def math(self):

        localctx = LaTeXParser.MathContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_math)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 82
            self.relation(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class RelationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def relation(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.RelationContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.RelationContext,i)


        def EQUAL(self):
            return self.getToken(LaTeXParser.EQUAL, 0)

        def LT(self):
            return self.getToken(LaTeXParser.LT, 0)

        def LTE(self):
            return self.getToken(LaTeXParser.LTE, 0)

        def GT(self):
            return self.getToken(LaTeXParser.GT, 0)

        def GTE(self):
            return self.getToken(LaTeXParser.GTE, 0)

        def NEQ(self):
            return self.getToken(LaTeXParser.NEQ, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_relation



    def relation(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = LaTeXParser.RelationContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 2
        self.enterRecursionRule(localctx, 2, self.RULE_relation, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 85
            self.expr()
            self._ctx.stop = self._input.LT(-1)
            self.state = 92
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,0,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = LaTeXParser.RelationContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_relation)
                    self.state = 87
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 88
                    _la = self._input.LA(1)
                    if not((((_la - 79)) & ~0x3f) == 0 and ((1 << (_la - 79)) & 207) != 0):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 89
                    self.relation(3)
                self.state = 94
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,0,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class EqualityContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.ExprContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.ExprContext,i)


        def EQUAL(self):
            return self.getToken(LaTeXParser.EQUAL, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_equality




    def equality(self):

        localctx = LaTeXParser.EqualityContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_equality)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 95
            self.expr()
            self.state = 96
            self.match(LaTeXParser.EQUAL)
            self.state = 97
            self.expr()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def additive(self):
            return self.getTypedRuleContext(LaTeXParser.AdditiveContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_expr




    def expr(self):

        localctx = LaTeXParser.ExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_expr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 99
            self.additive(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AdditiveContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def mp(self):
            return self.getTypedRuleContext(LaTeXParser.MpContext,0)


        def additive(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.AdditiveContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.AdditiveContext,i)


        def ADD(self):
            return self.getToken(LaTeXParser.ADD, 0)

        def SUB(self):
            return self.getToken(LaTeXParser.SUB, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_additive



    def additive(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = LaTeXParser.AdditiveContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 8
        self.enterRecursionRule(localctx, 8, self.RULE_additive, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 102
            self.mp(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 109
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,1,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = LaTeXParser.AdditiveContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_additive)
                    self.state = 104
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 105
                    _la = self._input.LA(1)
                    if not(_la==15 or _la==16):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 106
                    self.additive(3)
                self.state = 111
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class MpContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def unary(self):
            return self.getTypedRuleContext(LaTeXParser.UnaryContext,0)


        def mp(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.MpContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.MpContext,i)


        def MUL(self):
            return self.getToken(LaTeXParser.MUL, 0)

        def CMD_TIMES(self):
            return self.getToken(LaTeXParser.CMD_TIMES, 0)

        def CMD_CDOT(self):
            return self.getToken(LaTeXParser.CMD_CDOT, 0)

        def DIV(self):
            return self.getToken(LaTeXParser.DIV, 0)

        def CMD_DIV(self):
            return self.getToken(LaTeXParser.CMD_DIV, 0)

        def COLON(self):
            return self.getToken(LaTeXParser.COLON, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_mp



    def mp(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = LaTeXParser.MpContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 10
        self.enterRecursionRule(localctx, 10, self.RULE_mp, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 113
            self.unary()
            self._ctx.stop = self._input.LT(-1)
            self.state = 120
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = LaTeXParser.MpContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mp)
                    self.state = 115
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 116
                    _la = self._input.LA(1)
                    if not((((_la - 17)) & ~0x3f) == 0 and ((1 << (_la - 17)) & 290200700988686339) != 0):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 117
                    self.mp(3)
                self.state = 122
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class Mp_nofuncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def unary_nofunc(self):
            return self.getTypedRuleContext(LaTeXParser.Unary_nofuncContext,0)


        def mp_nofunc(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.Mp_nofuncContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.Mp_nofuncContext,i)


        def MUL(self):
            return self.getToken(LaTeXParser.MUL, 0)

        def CMD_TIMES(self):
            return self.getToken(LaTeXParser.CMD_TIMES, 0)

        def CMD_CDOT(self):
            return self.getToken(LaTeXParser.CMD_CDOT, 0)

        def DIV(self):
            return self.getToken(LaTeXParser.DIV, 0)

        def CMD_DIV(self):
            return self.getToken(LaTeXParser.CMD_DIV, 0)

        def COLON(self):
            return self.getToken(LaTeXParser.COLON, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_mp_nofunc



    def mp_nofunc(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = LaTeXParser.Mp_nofuncContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 12
        self.enterRecursionRule(localctx, 12, self.RULE_mp_nofunc, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 124
            self.unary_nofunc()
            self._ctx.stop = self._input.LT(-1)
            self.state = 131
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,3,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = LaTeXParser.Mp_nofuncContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mp_nofunc)
                    self.state = 126
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 127
                    _la = self._input.LA(1)
                    if not((((_la - 17)) & ~0x3f) == 0 and ((1 << (_la - 17)) & 290200700988686339) != 0):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 128
                    self.mp_nofunc(3)
                self.state = 133
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,3,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class UnaryContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def unary(self):
            return self.getTypedRuleContext(LaTeXParser.UnaryContext,0)


        def ADD(self):
            return self.getToken(LaTeXParser.ADD, 0)

        def SUB(self):
            return self.getToken(LaTeXParser.SUB, 0)

        def postfix(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.PostfixContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.PostfixContext,i)


        def getRuleIndex(self):
            return LaTeXParser.RULE_unary




    def unary(self):

        localctx = LaTeXParser.UnaryContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_unary)
        self._la = 0 # Token type
        try:
            self.state = 141
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [15, 16]:
                self.enterOuterAlt(localctx, 1)
                self.state = 134
                _la = self._input.LA(1)
                if not(_la==15 or _la==16):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 135
                self.unary()
                pass
            elif token in [19, 21, 23, 25, 27, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 63, 64, 68, 69, 70, 71, 72, 76, 77, 78, 91]:
                self.enterOuterAlt(localctx, 2)
                self.state = 137
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 136
                        self.postfix()

                    else:
                        raise NoViableAltException(self)
                    self.state = 139
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,4,self._ctx)

                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Unary_nofuncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def unary_nofunc(self):
            return self.getTypedRuleContext(LaTeXParser.Unary_nofuncContext,0)


        def ADD(self):
            return self.getToken(LaTeXParser.ADD, 0)

        def SUB(self):
            return self.getToken(LaTeXParser.SUB, 0)

        def postfix(self):
            return self.getTypedRuleContext(LaTeXParser.PostfixContext,0)


        def postfix_nofunc(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.Postfix_nofuncContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.Postfix_nofuncContext,i)


        def getRuleIndex(self):
            return LaTeXParser.RULE_unary_nofunc




    def unary_nofunc(self):

        localctx = LaTeXParser.Unary_nofuncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_unary_nofunc)
        self._la = 0 # Token type
        try:
            self.state = 152
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [15, 16]:
                self.enterOuterAlt(localctx, 1)
                self.state = 143
                _la = self._input.LA(1)
                if not(_la==15 or _la==16):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 144
                self.unary_nofunc()
                pass
            elif token in [19, 21, 23, 25, 27, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 63, 64, 68, 69, 70, 71, 72, 76, 77, 78, 91]:
                self.enterOuterAlt(localctx, 2)
                self.state = 145
                self.postfix()
                self.state = 149
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,6,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 146
                        self.postfix_nofunc()
                    self.state = 151
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,6,self._ctx)

                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PostfixContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def exp(self):
            return self.getTypedRuleContext(LaTeXParser.ExpContext,0)


        def postfix_op(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.Postfix_opContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.Postfix_opContext,i)


        def getRuleIndex(self):
            return LaTeXParser.RULE_postfix




    def postfix(self):

        localctx = LaTeXParser.PostfixContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_postfix)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 154
            self.exp(0)
            self.state = 158
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 155
                    self.postfix_op()
                self.state = 160
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,8,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Postfix_nofuncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def exp_nofunc(self):
            return self.getTypedRuleContext(LaTeXParser.Exp_nofuncContext,0)


        def postfix_op(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.Postfix_opContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.Postfix_opContext,i)


        def getRuleIndex(self):
            return LaTeXParser.RULE_postfix_nofunc




    def postfix_nofunc(self):

        localctx = LaTeXParser.Postfix_nofuncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_postfix_nofunc)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 161
            self.exp_nofunc(0)
            self.state = 165
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,9,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 162
                    self.postfix_op()
                self.state = 167
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,9,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Postfix_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BANG(self):
            return self.getToken(LaTeXParser.BANG, 0)

        def eval_at(self):
            return self.getTypedRuleContext(LaTeXParser.Eval_atContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_postfix_op




    def postfix_op(self):

        localctx = LaTeXParser.Postfix_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_postfix_op)
        try:
            self.state = 170
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [89]:
                self.enterOuterAlt(localctx, 1)
                self.state = 168
                self.match(LaTeXParser.BANG)
                pass
            elif token in [27]:
                self.enterOuterAlt(localctx, 2)
                self.state = 169
                self.eval_at()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Eval_atContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BAR(self):
            return self.getToken(LaTeXParser.BAR, 0)

        def eval_at_sup(self):
            return self.getTypedRuleContext(LaTeXParser.Eval_at_supContext,0)


        def eval_at_sub(self):
            return self.getTypedRuleContext(LaTeXParser.Eval_at_subContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_eval_at




    def eval_at(self):

        localctx = LaTeXParser.Eval_atContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_eval_at)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 172
            self.match(LaTeXParser.BAR)
            self.state = 178
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,11,self._ctx)
            if la_ == 1:
                self.state = 173
                self.eval_at_sup()
                pass

            elif la_ == 2:
                self.state = 174
                self.eval_at_sub()
                pass

            elif la_ == 3:
                self.state = 175
                self.eval_at_sup()
                self.state = 176
                self.eval_at_sub()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Eval_at_subContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def UNDERSCORE(self):
            return self.getToken(LaTeXParser.UNDERSCORE, 0)

        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def equality(self):
            return self.getTypedRuleContext(LaTeXParser.EqualityContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_eval_at_sub




    def eval_at_sub(self):

        localctx = LaTeXParser.Eval_at_subContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_eval_at_sub)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 180
            self.match(LaTeXParser.UNDERSCORE)
            self.state = 181
            self.match(LaTeXParser.L_BRACE)
            self.state = 184
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
            if la_ == 1:
                self.state = 182
                self.expr()
                pass

            elif la_ == 2:
                self.state = 183
                self.equality()
                pass


            self.state = 186
            self.match(LaTeXParser.R_BRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Eval_at_supContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CARET(self):
            return self.getToken(LaTeXParser.CARET, 0)

        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def equality(self):
            return self.getTypedRuleContext(LaTeXParser.EqualityContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_eval_at_sup




    def eval_at_sup(self):

        localctx = LaTeXParser.Eval_at_supContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_eval_at_sup)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 188
            self.match(LaTeXParser.CARET)
            self.state = 189
            self.match(LaTeXParser.L_BRACE)
            self.state = 192
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
            if la_ == 1:
                self.state = 190
                self.expr()
                pass

            elif la_ == 2:
                self.state = 191
                self.equality()
                pass


            self.state = 194
            self.match(LaTeXParser.R_BRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExpContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def comp(self):
            return self.getTypedRuleContext(LaTeXParser.CompContext,0)


        def exp(self):
            return self.getTypedRuleContext(LaTeXParser.ExpContext,0)


        def CARET(self):
            return self.getToken(LaTeXParser.CARET, 0)

        def atom(self):
            return self.getTypedRuleContext(LaTeXParser.AtomContext,0)


        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def subexpr(self):
            return self.getTypedRuleContext(LaTeXParser.SubexprContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_exp



    def exp(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = LaTeXParser.ExpContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 30
        self.enterRecursionRule(localctx, 30, self.RULE_exp, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 197
            self.comp()
            self._ctx.stop = self._input.LT(-1)
            self.state = 213
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,16,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = LaTeXParser.ExpContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_exp)
                    self.state = 199
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 200
                    self.match(LaTeXParser.CARET)
                    self.state = 206
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [27, 29, 30, 68, 69, 70, 71, 72, 76, 77, 78, 91]:
                        self.state = 201
                        self.atom()
                        pass
                    elif token in [21]:
                        self.state = 202
                        self.match(LaTeXParser.L_BRACE)
                        self.state = 203
                        self.expr()
                        self.state = 204
                        self.match(LaTeXParser.R_BRACE)
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 209
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,15,self._ctx)
                    if la_ == 1:
                        self.state = 208
                        self.subexpr()


                self.state = 215
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,16,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class Exp_nofuncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def comp_nofunc(self):
            return self.getTypedRuleContext(LaTeXParser.Comp_nofuncContext,0)


        def exp_nofunc(self):
            return self.getTypedRuleContext(LaTeXParser.Exp_nofuncContext,0)


        def CARET(self):
            return self.getToken(LaTeXParser.CARET, 0)

        def atom(self):
            return self.getTypedRuleContext(LaTeXParser.AtomContext,0)


        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def subexpr(self):
            return self.getTypedRuleContext(LaTeXParser.SubexprContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_exp_nofunc



    def exp_nofunc(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = LaTeXParser.Exp_nofuncContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 32
        self.enterRecursionRule(localctx, 32, self.RULE_exp_nofunc, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 217
            self.comp_nofunc()
            self._ctx.stop = self._input.LT(-1)
            self.state = 233
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,19,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = LaTeXParser.Exp_nofuncContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_exp_nofunc)
                    self.state = 219
                    if not self.precpred(self._ctx, 2):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                    self.state = 220
                    self.match(LaTeXParser.CARET)
                    self.state = 226
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [27, 29, 30, 68, 69, 70, 71, 72, 76, 77, 78, 91]:
                        self.state = 221
                        self.atom()
                        pass
                    elif token in [21]:
                        self.state = 222
                        self.match(LaTeXParser.L_BRACE)
                        self.state = 223
                        self.expr()
                        self.state = 224
                        self.match(LaTeXParser.R_BRACE)
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 229
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,18,self._ctx)
                    if la_ == 1:
                        self.state = 228
                        self.subexpr()


                self.state = 235
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,19,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class CompContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def group(self):
            return self.getTypedRuleContext(LaTeXParser.GroupContext,0)


        def abs_group(self):
            return self.getTypedRuleContext(LaTeXParser.Abs_groupContext,0)


        def func(self):
            return self.getTypedRuleContext(LaTeXParser.FuncContext,0)


        def atom(self):
            return self.getTypedRuleContext(LaTeXParser.AtomContext,0)


        def floor(self):
            return self.getTypedRuleContext(LaTeXParser.FloorContext,0)


        def ceil(self):
            return self.getTypedRuleContext(LaTeXParser.CeilContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_comp




    def comp(self):

        localctx = LaTeXParser.CompContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_comp)
        try:
            self.state = 242
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,20,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 236
                self.group()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 237
                self.abs_group()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 238
                self.func()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 239
                self.atom()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 240
                self.floor()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 241
                self.ceil()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Comp_nofuncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def group(self):
            return self.getTypedRuleContext(LaTeXParser.GroupContext,0)


        def abs_group(self):
            return self.getTypedRuleContext(LaTeXParser.Abs_groupContext,0)


        def atom(self):
            return self.getTypedRuleContext(LaTeXParser.AtomContext,0)


        def floor(self):
            return self.getTypedRuleContext(LaTeXParser.FloorContext,0)


        def ceil(self):
            return self.getTypedRuleContext(LaTeXParser.CeilContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_comp_nofunc




    def comp_nofunc(self):

        localctx = LaTeXParser.Comp_nofuncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_comp_nofunc)
        try:
            self.state = 249
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,21,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 244
                self.group()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 245
                self.abs_group()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 246
                self.atom()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 247
                self.floor()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 248
                self.ceil()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class GroupContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def L_PAREN(self):
            return self.getToken(LaTeXParser.L_PAREN, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_PAREN(self):
            return self.getToken(LaTeXParser.R_PAREN, 0)

        def L_BRACKET(self):
            return self.getToken(LaTeXParser.L_BRACKET, 0)

        def R_BRACKET(self):
            return self.getToken(LaTeXParser.R_BRACKET, 0)

        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def L_BRACE_LITERAL(self):
            return self.getToken(LaTeXParser.L_BRACE_LITERAL, 0)

        def R_BRACE_LITERAL(self):
            return self.getToken(LaTeXParser.R_BRACE_LITERAL, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_group




    def group(self):

        localctx = LaTeXParser.GroupContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_group)
        try:
            self.state = 267
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [19]:
                self.enterOuterAlt(localctx, 1)
                self.state = 251
                self.match(LaTeXParser.L_PAREN)
                self.state = 252
                self.expr()
                self.state = 253
                self.match(LaTeXParser.R_PAREN)
                pass
            elif token in [25]:
                self.enterOuterAlt(localctx, 2)
                self.state = 255
                self.match(LaTeXParser.L_BRACKET)
                self.state = 256
                self.expr()
                self.state = 257
                self.match(LaTeXParser.R_BRACKET)
                pass
            elif token in [21]:
                self.enterOuterAlt(localctx, 3)
                self.state = 259
                self.match(LaTeXParser.L_BRACE)
                self.state = 260
                self.expr()
                self.state = 261
                self.match(LaTeXParser.R_BRACE)
                pass
            elif token in [23]:
                self.enterOuterAlt(localctx, 4)
                self.state = 263
                self.match(LaTeXParser.L_BRACE_LITERAL)
                self.state = 264
                self.expr()
                self.state = 265
                self.match(LaTeXParser.R_BRACE_LITERAL)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Abs_groupContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BAR(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.BAR)
            else:
                return self.getToken(LaTeXParser.BAR, i)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_abs_group




    def abs_group(self):

        localctx = LaTeXParser.Abs_groupContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_abs_group)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 269
            self.match(LaTeXParser.BAR)
            self.state = 270
            self.expr()
            self.state = 271
            self.match(LaTeXParser.BAR)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class NumberContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def DIGIT(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.DIGIT)
            else:
                return self.getToken(LaTeXParser.DIGIT, i)

        def getRuleIndex(self):
            return LaTeXParser.RULE_number




    def number(self):

        localctx = LaTeXParser.NumberContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_number)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 274
            self._errHandler.sync(self)
            _alt = 1
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt == 1:
                    self.state = 273
                    self.match(LaTeXParser.DIGIT)

                else:
                    raise NoViableAltException(self)
                self.state = 276
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,23,self._ctx)

            self.state = 284
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,24,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 278
                    self.match(LaTeXParser.T__0)
                    self.state = 279
                    self.match(LaTeXParser.DIGIT)
                    self.state = 280
                    self.match(LaTeXParser.DIGIT)
                    self.state = 281
                    self.match(LaTeXParser.DIGIT)
                self.state = 286
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,24,self._ctx)

            self.state = 293
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,26,self._ctx)
            if la_ == 1:
                self.state = 287
                self.match(LaTeXParser.T__1)
                self.state = 289
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 288
                        self.match(LaTeXParser.DIGIT)

                    else:
                        raise NoViableAltException(self)
                    self.state = 291
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,25,self._ctx)



        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AtomContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LETTER(self):
            return self.getToken(LaTeXParser.LETTER, 0)

        def SYMBOL(self):
            return self.getToken(LaTeXParser.SYMBOL, 0)

        def subexpr(self):
            return self.getTypedRuleContext(LaTeXParser.SubexprContext,0)


        def SINGLE_QUOTES(self):
            return self.getToken(LaTeXParser.SINGLE_QUOTES, 0)

        def number(self):
            return self.getTypedRuleContext(LaTeXParser.NumberContext,0)


        def DIFFERENTIAL(self):
            return self.getToken(LaTeXParser.DIFFERENTIAL, 0)

        def mathit(self):
            return self.getTypedRuleContext(LaTeXParser.MathitContext,0)


        def frac(self):
            return self.getTypedRuleContext(LaTeXParser.FracContext,0)


        def binom(self):
            return self.getTypedRuleContext(LaTeXParser.BinomContext,0)


        def bra(self):
            return self.getTypedRuleContext(LaTeXParser.BraContext,0)


        def ket(self):
            return self.getTypedRuleContext(LaTeXParser.KetContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_atom




    def atom(self):

        localctx = LaTeXParser.AtomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_atom)
        self._la = 0 # Token type
        try:
            self.state = 317
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [77, 91]:
                self.enterOuterAlt(localctx, 1)
                self.state = 295
                _la = self._input.LA(1)
                if not(_la==77 or _la==91):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 308
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,31,self._ctx)
                if la_ == 1:
                    self.state = 297
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,27,self._ctx)
                    if la_ == 1:
                        self.state = 296
                        self.subexpr()


                    self.state = 300
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,28,self._ctx)
                    if la_ == 1:
                        self.state = 299
                        self.match(LaTeXParser.SINGLE_QUOTES)


                    pass

                elif la_ == 2:
                    self.state = 303
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,29,self._ctx)
                    if la_ == 1:
                        self.state = 302
                        self.match(LaTeXParser.SINGLE_QUOTES)


                    self.state = 306
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,30,self._ctx)
                    if la_ == 1:
                        self.state = 305
                        self.subexpr()


                    pass


                pass
            elif token in [78]:
                self.enterOuterAlt(localctx, 2)
                self.state = 310
                self.number()
                pass
            elif token in [76]:
                self.enterOuterAlt(localctx, 3)
                self.state = 311
                self.match(LaTeXParser.DIFFERENTIAL)
                pass
            elif token in [72]:
                self.enterOuterAlt(localctx, 4)
                self.state = 312
                self.mathit()
                pass
            elif token in [68]:
                self.enterOuterAlt(localctx, 5)
                self.state = 313
                self.frac()
                pass
            elif token in [69, 70, 71]:
                self.enterOuterAlt(localctx, 6)
                self.state = 314
                self.binom()
                pass
            elif token in [30]:
                self.enterOuterAlt(localctx, 7)
                self.state = 315
                self.bra()
                pass
            elif token in [27, 29]:
                self.enterOuterAlt(localctx, 8)
                self.state = 316
                self.ket()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BraContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def L_ANGLE(self):
            return self.getToken(LaTeXParser.L_ANGLE, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_BAR(self):
            return self.getToken(LaTeXParser.R_BAR, 0)

        def BAR(self):
            return self.getToken(LaTeXParser.BAR, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_bra




    def bra(self):

        localctx = LaTeXParser.BraContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_bra)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 319
            self.match(LaTeXParser.L_ANGLE)
            self.state = 320
            self.expr()
            self.state = 321
            _la = self._input.LA(1)
            if not(_la==27 or _la==28):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class KetContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_ANGLE(self):
            return self.getToken(LaTeXParser.R_ANGLE, 0)

        def L_BAR(self):
            return self.getToken(LaTeXParser.L_BAR, 0)

        def BAR(self):
            return self.getToken(LaTeXParser.BAR, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_ket




    def ket(self):

        localctx = LaTeXParser.KetContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_ket)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 323
            _la = self._input.LA(1)
            if not(_la==27 or _la==29):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 324
            self.expr()
            self.state = 325
            self.match(LaTeXParser.R_ANGLE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MathitContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CMD_MATHIT(self):
            return self.getToken(LaTeXParser.CMD_MATHIT, 0)

        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def mathit_text(self):
            return self.getTypedRuleContext(LaTeXParser.Mathit_textContext,0)


        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_mathit




    def mathit(self):

        localctx = LaTeXParser.MathitContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_mathit)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 327
            self.match(LaTeXParser.CMD_MATHIT)
            self.state = 328
            self.match(LaTeXParser.L_BRACE)
            self.state = 329
            self.mathit_text()
            self.state = 330
            self.match(LaTeXParser.R_BRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Mathit_textContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LETTER(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.LETTER)
            else:
                return self.getToken(LaTeXParser.LETTER, i)

        def getRuleIndex(self):
            return LaTeXParser.RULE_mathit_text




    def mathit_text(self):

        localctx = LaTeXParser.Mathit_textContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_mathit_text)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 335
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==77:
                self.state = 332
                self.match(LaTeXParser.LETTER)
                self.state = 337
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FracContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.upperd = None # Token
            self.upper = None # ExprContext
            self.lowerd = None # Token
            self.lower = None # ExprContext

        def CMD_FRAC(self):
            return self.getToken(LaTeXParser.CMD_FRAC, 0)

        def L_BRACE(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.L_BRACE)
            else:
                return self.getToken(LaTeXParser.L_BRACE, i)

        def R_BRACE(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.R_BRACE)
            else:
                return self.getToken(LaTeXParser.R_BRACE, i)

        def DIGIT(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.DIGIT)
            else:
                return self.getToken(LaTeXParser.DIGIT, i)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.ExprContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.ExprContext,i)


        def getRuleIndex(self):
            return LaTeXParser.RULE_frac




    def frac(self):

        localctx = LaTeXParser.FracContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_frac)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 338
            self.match(LaTeXParser.CMD_FRAC)
            self.state = 344
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [78]:
                self.state = 339
                localctx.upperd = self.match(LaTeXParser.DIGIT)
                pass
            elif token in [21]:
                self.state = 340
                self.match(LaTeXParser.L_BRACE)
                self.state = 341
                localctx.upper = self.expr()
                self.state = 342
                self.match(LaTeXParser.R_BRACE)
                pass
            else:
                raise NoViableAltException(self)

            self.state = 351
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [78]:
                self.state = 346
                localctx.lowerd = self.match(LaTeXParser.DIGIT)
                pass
            elif token in [21]:
                self.state = 347
                self.match(LaTeXParser.L_BRACE)
                self.state = 348
                localctx.lower = self.expr()
                self.state = 349
                self.match(LaTeXParser.R_BRACE)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BinomContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.n = None # ExprContext
            self.k = None # ExprContext

        def L_BRACE(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.L_BRACE)
            else:
                return self.getToken(LaTeXParser.L_BRACE, i)

        def R_BRACE(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.R_BRACE)
            else:
                return self.getToken(LaTeXParser.R_BRACE, i)

        def CMD_BINOM(self):
            return self.getToken(LaTeXParser.CMD_BINOM, 0)

        def CMD_DBINOM(self):
            return self.getToken(LaTeXParser.CMD_DBINOM, 0)

        def CMD_TBINOM(self):
            return self.getToken(LaTeXParser.CMD_TBINOM, 0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.ExprContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.ExprContext,i)


        def getRuleIndex(self):
            return LaTeXParser.RULE_binom




    def binom(self):

        localctx = LaTeXParser.BinomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_binom)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 353
            _la = self._input.LA(1)
            if not((((_la - 69)) & ~0x3f) == 0 and ((1 << (_la - 69)) & 7) != 0):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 354
            self.match(LaTeXParser.L_BRACE)
            self.state = 355
            localctx.n = self.expr()
            self.state = 356
            self.match(LaTeXParser.R_BRACE)
            self.state = 357
            self.match(LaTeXParser.L_BRACE)
            self.state = 358
            localctx.k = self.expr()
            self.state = 359
            self.match(LaTeXParser.R_BRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FloorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.val = None # ExprContext

        def L_FLOOR(self):
            return self.getToken(LaTeXParser.L_FLOOR, 0)

        def R_FLOOR(self):
            return self.getToken(LaTeXParser.R_FLOOR, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_floor




    def floor(self):

        localctx = LaTeXParser.FloorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_floor)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 361
            self.match(LaTeXParser.L_FLOOR)
            self.state = 362
            localctx.val = self.expr()
            self.state = 363
            self.match(LaTeXParser.R_FLOOR)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CeilContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.val = None # ExprContext

        def L_CEIL(self):
            return self.getToken(LaTeXParser.L_CEIL, 0)

        def R_CEIL(self):
            return self.getToken(LaTeXParser.R_CEIL, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_ceil




    def ceil(self):

        localctx = LaTeXParser.CeilContext(self, self._ctx, self.state)
        self.enterRule(localctx, 60, self.RULE_ceil)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 365
            self.match(LaTeXParser.L_CEIL)
            self.state = 366
            localctx.val = self.expr()
            self.state = 367
            self.match(LaTeXParser.R_CEIL)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Func_normalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def FUNC_EXP(self):
            return self.getToken(LaTeXParser.FUNC_EXP, 0)

        def FUNC_LOG(self):
            return self.getToken(LaTeXParser.FUNC_LOG, 0)

        def FUNC_LG(self):
            return self.getToken(LaTeXParser.FUNC_LG, 0)

        def FUNC_LN(self):
            return self.getToken(LaTeXParser.FUNC_LN, 0)

        def FUNC_SIN(self):
            return self.getToken(LaTeXParser.FUNC_SIN, 0)

        def FUNC_COS(self):
            return self.getToken(LaTeXParser.FUNC_COS, 0)

        def FUNC_TAN(self):
            return self.getToken(LaTeXParser.FUNC_TAN, 0)

        def FUNC_CSC(self):
            return self.getToken(LaTeXParser.FUNC_CSC, 0)

        def FUNC_SEC(self):
            return self.getToken(LaTeXParser.FUNC_SEC, 0)

        def FUNC_COT(self):
            return self.getToken(LaTeXParser.FUNC_COT, 0)

        def FUNC_ARCSIN(self):
            return self.getToken(LaTeXParser.FUNC_ARCSIN, 0)

        def FUNC_ARCCOS(self):
            return self.getToken(LaTeXParser.FUNC_ARCCOS, 0)

        def FUNC_ARCTAN(self):
            return self.getToken(LaTeXParser.FUNC_ARCTAN, 0)

        def FUNC_ARCCSC(self):
            return self.getToken(LaTeXParser.FUNC_ARCCSC, 0)

        def FUNC_ARCSEC(self):
            return self.getToken(LaTeXParser.FUNC_ARCSEC, 0)

        def FUNC_ARCCOT(self):
            return self.getToken(LaTeXParser.FUNC_ARCCOT, 0)

        def FUNC_SINH(self):
            return self.getToken(LaTeXParser.FUNC_SINH, 0)

        def FUNC_COSH(self):
            return self.getToken(LaTeXParser.FUNC_COSH, 0)

        def FUNC_TANH(self):
            return self.getToken(LaTeXParser.FUNC_TANH, 0)

        def FUNC_ARSINH(self):
            return self.getToken(LaTeXParser.FUNC_ARSINH, 0)

        def FUNC_ARCOSH(self):
            return self.getToken(LaTeXParser.FUNC_ARCOSH, 0)

        def FUNC_ARTANH(self):
            return self.getToken(LaTeXParser.FUNC_ARTANH, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_func_normal




    def func_normal(self):

        localctx = LaTeXParser.Func_normalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 62, self.RULE_func_normal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 369
            _la = self._input.LA(1)
            if not(((_la) & ~0x3f) == 0 and ((1 << _la) & 576460614864470016) != 0):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FuncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser
            self.root = None # ExprContext
            self.base = None # ExprContext

        def func_normal(self):
            return self.getTypedRuleContext(LaTeXParser.Func_normalContext,0)


        def L_PAREN(self):
            return self.getToken(LaTeXParser.L_PAREN, 0)

        def func_arg(self):
            return self.getTypedRuleContext(LaTeXParser.Func_argContext,0)


        def R_PAREN(self):
            return self.getToken(LaTeXParser.R_PAREN, 0)

        def func_arg_noparens(self):
            return self.getTypedRuleContext(LaTeXParser.Func_arg_noparensContext,0)


        def subexpr(self):
            return self.getTypedRuleContext(LaTeXParser.SubexprContext,0)


        def supexpr(self):
            return self.getTypedRuleContext(LaTeXParser.SupexprContext,0)


        def args(self):
            return self.getTypedRuleContext(LaTeXParser.ArgsContext,0)


        def LETTER(self):
            return self.getToken(LaTeXParser.LETTER, 0)

        def SYMBOL(self):
            return self.getToken(LaTeXParser.SYMBOL, 0)

        def SINGLE_QUOTES(self):
            return self.getToken(LaTeXParser.SINGLE_QUOTES, 0)

        def FUNC_INT(self):
            return self.getToken(LaTeXParser.FUNC_INT, 0)

        def DIFFERENTIAL(self):
            return self.getToken(LaTeXParser.DIFFERENTIAL, 0)

        def frac(self):
            return self.getTypedRuleContext(LaTeXParser.FracContext,0)


        def additive(self):
            return self.getTypedRuleContext(LaTeXParser.AdditiveContext,0)


        def FUNC_SQRT(self):
            return self.getToken(LaTeXParser.FUNC_SQRT, 0)

        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(LaTeXParser.ExprContext)
            else:
                return self.getTypedRuleContext(LaTeXParser.ExprContext,i)


        def L_BRACKET(self):
            return self.getToken(LaTeXParser.L_BRACKET, 0)

        def R_BRACKET(self):
            return self.getToken(LaTeXParser.R_BRACKET, 0)

        def FUNC_OVERLINE(self):
            return self.getToken(LaTeXParser.FUNC_OVERLINE, 0)

        def mp(self):
            return self.getTypedRuleContext(LaTeXParser.MpContext,0)


        def FUNC_SUM(self):
            return self.getToken(LaTeXParser.FUNC_SUM, 0)

        def FUNC_PROD(self):
            return self.getToken(LaTeXParser.FUNC_PROD, 0)

        def subeq(self):
            return self.getTypedRuleContext(LaTeXParser.SubeqContext,0)


        def FUNC_LIM(self):
            return self.getToken(LaTeXParser.FUNC_LIM, 0)

        def limit_sub(self):
            return self.getTypedRuleContext(LaTeXParser.Limit_subContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_func




    def func(self):

        localctx = LaTeXParser.FuncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 64, self.RULE_func)
        self._la = 0 # Token type
        try:
            self.state = 460
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]:
                self.enterOuterAlt(localctx, 1)
                self.state = 371
                self.func_normal()
                self.state = 384
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,40,self._ctx)
                if la_ == 1:
                    self.state = 373
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==73:
                        self.state = 372
                        self.subexpr()


                    self.state = 376
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==74:
                        self.state = 375
                        self.supexpr()


                    pass

                elif la_ == 2:
                    self.state = 379
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==74:
                        self.state = 378
                        self.supexpr()


                    self.state = 382
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==73:
                        self.state = 381
                        self.subexpr()


                    pass


                self.state = 391
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,41,self._ctx)
                if la_ == 1:
                    self.state = 386
                    self.match(LaTeXParser.L_PAREN)
                    self.state = 387
                    self.func_arg()
                    self.state = 388
                    self.match(LaTeXParser.R_PAREN)
                    pass

                elif la_ == 2:
                    self.state = 390
                    self.func_arg_noparens()
                    pass


                pass
            elif token in [77, 91]:
                self.enterOuterAlt(localctx, 2)
                self.state = 393
                _la = self._input.LA(1)
                if not(_la==77 or _la==91):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 406
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,46,self._ctx)
                if la_ == 1:
                    self.state = 395
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==73:
                        self.state = 394
                        self.subexpr()


                    self.state = 398
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==90:
                        self.state = 397
                        self.match(LaTeXParser.SINGLE_QUOTES)


                    pass

                elif la_ == 2:
                    self.state = 401
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==90:
                        self.state = 400
                        self.match(LaTeXParser.SINGLE_QUOTES)


                    self.state = 404
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==73:
                        self.state = 403
                        self.subexpr()


                    pass


                self.state = 408
                self.match(LaTeXParser.L_PAREN)
                self.state = 409
                self.args()
                self.state = 410
                self.match(LaTeXParser.R_PAREN)
                pass
            elif token in [34]:
                self.enterOuterAlt(localctx, 3)
                self.state = 412
                self.match(LaTeXParser.FUNC_INT)
                self.state = 419
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [73]:
                    self.state = 413
                    self.subexpr()
                    self.state = 414
                    self.supexpr()
                    pass
                elif token in [74]:
                    self.state = 416
                    self.supexpr()
                    self.state = 417
                    self.subexpr()
                    pass
                elif token in [15, 16, 19, 21, 23, 25, 27, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 63, 64, 68, 69, 70, 71, 72, 76, 77, 78, 91]:
                    pass
                else:
                    pass
                self.state = 427
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,49,self._ctx)
                if la_ == 1:
                    self.state = 422
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,48,self._ctx)
                    if la_ == 1:
                        self.state = 421
                        self.additive(0)


                    self.state = 424
                    self.match(LaTeXParser.DIFFERENTIAL)
                    pass

                elif la_ == 2:
                    self.state = 425
                    self.frac()
                    pass

                elif la_ == 3:
                    self.state = 426
                    self.additive(0)
                    pass


                pass
            elif token in [63]:
                self.enterOuterAlt(localctx, 4)
                self.state = 429
                self.match(LaTeXParser.FUNC_SQRT)
                self.state = 434
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==25:
                    self.state = 430
                    self.match(LaTeXParser.L_BRACKET)
                    self.state = 431
                    localctx.root = self.expr()
                    self.state = 432
                    self.match(LaTeXParser.R_BRACKET)


                self.state = 436
                self.match(LaTeXParser.L_BRACE)
                self.state = 437
                localctx.base = self.expr()
                self.state = 438
                self.match(LaTeXParser.R_BRACE)
                pass
            elif token in [64]:
                self.enterOuterAlt(localctx, 5)
                self.state = 440
                self.match(LaTeXParser.FUNC_OVERLINE)
                self.state = 441
                self.match(LaTeXParser.L_BRACE)
                self.state = 442
                localctx.base = self.expr()
                self.state = 443
                self.match(LaTeXParser.R_BRACE)
                pass
            elif token in [35, 36]:
                self.enterOuterAlt(localctx, 6)
                self.state = 445
                _la = self._input.LA(1)
                if not(_la==35 or _la==36):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 452
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [73]:
                    self.state = 446
                    self.subeq()
                    self.state = 447
                    self.supexpr()
                    pass
                elif token in [74]:
                    self.state = 449
                    self.supexpr()
                    self.state = 450
                    self.subeq()
                    pass
                else:
                    raise NoViableAltException(self)

                self.state = 454
                self.mp(0)
                pass
            elif token in [32]:
                self.enterOuterAlt(localctx, 7)
                self.state = 456
                self.match(LaTeXParser.FUNC_LIM)
                self.state = 457
                self.limit_sub()
                self.state = 458
                self.mp(0)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ArgsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def args(self):
            return self.getTypedRuleContext(LaTeXParser.ArgsContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_args




    def args(self):

        localctx = LaTeXParser.ArgsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 66, self.RULE_args)
        try:
            self.state = 467
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,53,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 462
                self.expr()
                self.state = 463
                self.match(LaTeXParser.T__0)
                self.state = 464
                self.args()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 466
                self.expr()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Limit_subContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def UNDERSCORE(self):
            return self.getToken(LaTeXParser.UNDERSCORE, 0)

        def L_BRACE(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.L_BRACE)
            else:
                return self.getToken(LaTeXParser.L_BRACE, i)

        def LIM_APPROACH_SYM(self):
            return self.getToken(LaTeXParser.LIM_APPROACH_SYM, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_BRACE(self, i:int=None):
            if i is None:
                return self.getTokens(LaTeXParser.R_BRACE)
            else:
                return self.getToken(LaTeXParser.R_BRACE, i)

        def LETTER(self):
            return self.getToken(LaTeXParser.LETTER, 0)

        def SYMBOL(self):
            return self.getToken(LaTeXParser.SYMBOL, 0)

        def CARET(self):
            return self.getToken(LaTeXParser.CARET, 0)

        def ADD(self):
            return self.getToken(LaTeXParser.ADD, 0)

        def SUB(self):
            return self.getToken(LaTeXParser.SUB, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_limit_sub




    def limit_sub(self):

        localctx = LaTeXParser.Limit_subContext(self, self._ctx, self.state)
        self.enterRule(localctx, 68, self.RULE_limit_sub)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 469
            self.match(LaTeXParser.UNDERSCORE)
            self.state = 470
            self.match(LaTeXParser.L_BRACE)
            self.state = 471
            _la = self._input.LA(1)
            if not(_la==77 or _la==91):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 472
            self.match(LaTeXParser.LIM_APPROACH_SYM)
            self.state = 473
            self.expr()
            self.state = 482
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==74:
                self.state = 474
                self.match(LaTeXParser.CARET)
                self.state = 480
                self._errHandler.sync(self)
                token = self._input.LA(1)
                if token in [21]:
                    self.state = 475
                    self.match(LaTeXParser.L_BRACE)
                    self.state = 476
                    _la = self._input.LA(1)
                    if not(_la==15 or _la==16):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 477
                    self.match(LaTeXParser.R_BRACE)
                    pass
                elif token in [15]:
                    self.state = 478
                    self.match(LaTeXParser.ADD)
                    pass
                elif token in [16]:
                    self.state = 479
                    self.match(LaTeXParser.SUB)
                    pass
                else:
                    raise NoViableAltException(self)



            self.state = 484
            self.match(LaTeXParser.R_BRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Func_argContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def func_arg(self):
            return self.getTypedRuleContext(LaTeXParser.Func_argContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_func_arg




    def func_arg(self):

        localctx = LaTeXParser.Func_argContext(self, self._ctx, self.state)
        self.enterRule(localctx, 70, self.RULE_func_arg)
        try:
            self.state = 491
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,56,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 486
                self.expr()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 487
                self.expr()
                self.state = 488
                self.match(LaTeXParser.T__0)
                self.state = 489
                self.func_arg()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Func_arg_noparensContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def mp_nofunc(self):
            return self.getTypedRuleContext(LaTeXParser.Mp_nofuncContext,0)


        def getRuleIndex(self):
            return LaTeXParser.RULE_func_arg_noparens




    def func_arg_noparens(self):

        localctx = LaTeXParser.Func_arg_noparensContext(self, self._ctx, self.state)
        self.enterRule(localctx, 72, self.RULE_func_arg_noparens)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 493
            self.mp_nofunc(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SubexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def UNDERSCORE(self):
            return self.getToken(LaTeXParser.UNDERSCORE, 0)

        def atom(self):
            return self.getTypedRuleContext(LaTeXParser.AtomContext,0)


        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_subexpr




    def subexpr(self):

        localctx = LaTeXParser.SubexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 74, self.RULE_subexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 495
            self.match(LaTeXParser.UNDERSCORE)
            self.state = 501
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [27, 29, 30, 68, 69, 70, 71, 72, 76, 77, 78, 91]:
                self.state = 496
                self.atom()
                pass
            elif token in [21]:
                self.state = 497
                self.match(LaTeXParser.L_BRACE)
                self.state = 498
                self.expr()
                self.state = 499
                self.match(LaTeXParser.R_BRACE)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SupexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CARET(self):
            return self.getToken(LaTeXParser.CARET, 0)

        def atom(self):
            return self.getTypedRuleContext(LaTeXParser.AtomContext,0)


        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def expr(self):
            return self.getTypedRuleContext(LaTeXParser.ExprContext,0)


        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_supexpr




    def supexpr(self):

        localctx = LaTeXParser.SupexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 76, self.RULE_supexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 503
            self.match(LaTeXParser.CARET)
            self.state = 509
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [27, 29, 30, 68, 69, 70, 71, 72, 76, 77, 78, 91]:
                self.state = 504
                self.atom()
                pass
            elif token in [21]:
                self.state = 505
                self.match(LaTeXParser.L_BRACE)
                self.state = 506
                self.expr()
                self.state = 507
                self.match(LaTeXParser.R_BRACE)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SubeqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def UNDERSCORE(self):
            return self.getToken(LaTeXParser.UNDERSCORE, 0)

        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def equality(self):
            return self.getTypedRuleContext(LaTeXParser.EqualityContext,0)


        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_subeq




    def subeq(self):

        localctx = LaTeXParser.SubeqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 78, self.RULE_subeq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 511
            self.match(LaTeXParser.UNDERSCORE)
            self.state = 512
            self.match(LaTeXParser.L_BRACE)
            self.state = 513
            self.equality()
            self.state = 514
            self.match(LaTeXParser.R_BRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SupeqContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def UNDERSCORE(self):
            return self.getToken(LaTeXParser.UNDERSCORE, 0)

        def L_BRACE(self):
            return self.getToken(LaTeXParser.L_BRACE, 0)

        def equality(self):
            return self.getTypedRuleContext(LaTeXParser.EqualityContext,0)


        def R_BRACE(self):
            return self.getToken(LaTeXParser.R_BRACE, 0)

        def getRuleIndex(self):
            return LaTeXParser.RULE_supeq




    def supeq(self):

        localctx = LaTeXParser.SupeqContext(self, self._ctx, self.state)
        self.enterRule(localctx, 80, self.RULE_supeq)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 516
            self.match(LaTeXParser.UNDERSCORE)
            self.state = 517
            self.match(LaTeXParser.L_BRACE)
            self.state = 518
            self.equality()
            self.state = 519
            self.match(LaTeXParser.R_BRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[1] = self.relation_sempred
        self._predicates[4] = self.additive_sempred
        self._predicates[5] = self.mp_sempred
        self._predicates[6] = self.mp_nofunc_sempred
        self._predicates[15] = self.exp_sempred
        self._predicates[16] = self.exp_nofunc_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def relation_sempred(self, localctx:RelationContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 2)


    def additive_sempred(self, localctx:AdditiveContext, predIndex:int):
            if predIndex == 1:
                return self.precpred(self._ctx, 2)


    def mp_sempred(self, localctx:MpContext, predIndex:int):
            if predIndex == 2:
                return self.precpred(self._ctx, 2)


    def mp_nofunc_sempred(self, localctx:Mp_nofuncContext, predIndex:int):
            if predIndex == 3:
                return self.precpred(self._ctx, 2)


    def exp_sempred(self, localctx:ExpContext, predIndex:int):
            if predIndex == 4:
                return self.precpred(self._ctx, 2)


    def exp_nofunc_sempred(self, localctx:Exp_nofuncContext, predIndex:int):
            if predIndex == 5:
                return self.precpred(self._ctx, 2)






# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\trigonometric.py ===
from __future__ import annotations
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import DefinedFunction, ArgumentIndexError, PoleError, expand_mul
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.relational import Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.combinatorial.numbers import bernoulli, euler
from sympy.functions.elementary.complexes import arg as arg_f, im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary._trigonometric_special import (
    cos_table, ipartfrac, fermat_coords)
from sympy.logic.boolalg import And
from sympy.ntheory import factorint
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.iterables import numbered_symbols


###############################################################################
########################## UTILITIES ##########################################
###############################################################################


def _imaginary_unit_as_coefficient(arg):
    """ Helper to extract symbolic coefficient for imaginary unit """
    if isinstance(arg, Float):
        return None
    else:
        return arg.as_coefficient(S.ImaginaryUnit)

###############################################################################
########################## TRIGONOMETRIC FUNCTIONS ############################
###############################################################################


class TrigonometricFunction(DefinedFunction):
    """Base class for trigonometric functions. """

    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational and fuzzy_not(s.args[0].is_zero):
                return False
        else:
            return s.is_rational

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if fuzzy_not(self.args[0].is_zero) and self.args[0].is_algebraic:
                return False
            pi_coeff = _pi_coeff(self.args[0])
            if pi_coeff is not None and pi_coeff.is_rational:
                return True
        else:
            return s.is_algebraic

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*S.ImaginaryUnit

    def _as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.args[0].expand(deep, **hints), S.Zero)
            else:
                return (self.args[0], S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (re, im)

    def _period(self, general_period, symbol=None):
        f = expand_mul(self.args[0])
        if symbol is None:
            symbol = tuple(f.free_symbols)[0]

        if not f.has(symbol):
            return S.Zero

        if f == symbol:
            return general_period

        if symbol in f.free_symbols:
            if f.is_Mul:
                g, h = f.as_independent(symbol)
                if h == symbol:
                    return general_period/abs(g)

            if f.is_Add:
                a, h = f.as_independent(symbol)
                g, h = h.as_independent(symbol, as_Add=False)
                if h == symbol:
                    return general_period/abs(g)

        raise NotImplementedError("Use the periodicity function instead.")


@cacheit
def _table2():
    # If nested sqrt's are worse than un-evaluation
    # you can require q to be in (1, 2, 3, 4, 6, 12)
    # q <= 12, q=15, q=20, q=24, q=30, q=40, q=60, q=120 return
    # expressions with 2 or fewer sqrt nestings.
    return {
        12: (3, 4),
        20: (4, 5),
        30: (5, 6),
        15: (6, 10),
        24: (6, 8),
        40: (8, 10),
        60: (20, 30),
        120: (40, 60)
    }


def _peeloff_pi(arg):
    r"""
    Split ARG into two parts, a "rest" and a multiple of $\pi$.
    This assumes ARG to be an Add.
    The multiple of $\pi$ returned in the second position is always a Rational.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _peeloff_pi
    >>> from sympy import pi
    >>> from sympy.abc import x, y
    >>> _peeloff_pi(x + pi/2)
    (x, 1/2)
    >>> _peeloff_pi(x + 2*pi/3 + pi*y)
    (x + pi*y + pi/6, 1/2)

    """
    pi_coeff = S.Zero
    rest_terms = []
    for a in Add.make_args(arg):
        K = a.coeff(pi)
        if K and K.is_rational:
            pi_coeff += K
        else:
            rest_terms.append(a)

    if pi_coeff is S.Zero:
        return arg, S.Zero

    m1 = (pi_coeff % S.Half)
    m2 = pi_coeff - m1
    if m2.is_integer or ((2*m2).is_integer and m2.is_even is False):
        return Add(*(rest_terms + [m1*pi])), m2
    return arg, S.Zero


def _pi_coeff(arg: Expr, cycles: int = 1) -> Expr | None:
    r"""
    When arg is a Number times $\pi$ (e.g. $3\pi/2$) then return the Number
    normalized to be in the range $[0, 2]$, else `None`.

    When an even multiple of $\pi$ is encountered, if it is multiplying
    something with known parity then the multiple is returned as 0 otherwise
    as 2.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _pi_coeff
    >>> from sympy import pi, Dummy
    >>> from sympy.abc import x
    >>> _pi_coeff(3*x*pi)
    3*x
    >>> _pi_coeff(11*pi/7)
    11/7
    >>> _pi_coeff(-11*pi/7)
    3/7
    >>> _pi_coeff(4*pi)
    0
    >>> _pi_coeff(5*pi)
    1
    >>> _pi_coeff(5.0*pi)
    1
    >>> _pi_coeff(5.5*pi)
    3/2
    >>> _pi_coeff(2 + pi)

    >>> _pi_coeff(2*Dummy(integer=True)*pi)
    2
    >>> _pi_coeff(2*Dummy(even=True)*pi)
    0

    """
    if arg is pi:
        return S.One
    elif not arg:
        return S.Zero
    elif arg.is_Mul:
        cx = arg.coeff(pi)
        if cx:
            c, x = cx.as_coeff_Mul()  # pi is not included as coeff
            if c.is_Float:
                # recast exact binary fractions to Rationals
                f = abs(c) % 1
                if f != 0:
                    p = -int(round(log(f, 2).evalf()))
                    m = 2**p
                    cm = c*m
                    i = int(cm)
                    if equal_valued(i, cm):
                        c = Rational(i, m)
                        cx = c*x
                else:
                    c = Rational(int(c))
                    cx = c*x
            if x.is_integer:
                c2 = c % 2
                if c2 == 1:
                    return x
                elif not c2:
                    if x.is_even is not None:  # known parity
                        return S.Zero
                    return Integer(2)
                else:
                    return c2*x
            return cx
    elif arg.is_zero:
        return S.Zero
    return None


class sin(TrigonometricFunction):
    r"""
    The sine function.

    Returns the sine of x (measured in radians).

    Explanation
    ===========

    This function will evaluate automatically in the
    case $x/\pi$ is some rational number [4]_.  For example,
    if $x$ is a multiple of $\pi$, $\pi/2$, $\pi/3$, $\pi/4$, and $\pi/6$.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x
    >>> sin(x**2).diff(x)
    2*x*cos(x**2)
    >>> sin(1).diff(x)
    0
    >>> sin(pi)
    0
    >>> sin(pi/2)
    1
    >>> sin(pi/6)
    1/2
    >>> sin(pi/12)
    -sqrt(2)/4 + sqrt(6)/4


    See Also
    ========

    csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sin
    .. [4] https://mathworld.wolfram.com/TrigonometryAngles.html

    """

    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return cos(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.sets.setexpr import SetExpr
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(-1, 1)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            from sympy.sets.sets import FiniteSet
            min, max = arg.min, arg.max
            d = floor(min/(2*pi))
            if min is not S.NegativeInfinity:
                min = min - d*2*pi
            if max is not S.Infinity:
                max = max - d*2*pi
            if AccumBounds(min, max).intersection(FiniteSet(pi/2, pi*Rational(5, 2))) \
                    is not S.EmptySet and \
                    AccumBounds(min, max).intersection(FiniteSet(pi*Rational(3, 2),
                        pi*Rational(7, 2))) is not S.EmptySet:
                return AccumBounds(-1, 1)
            elif AccumBounds(min, max).intersection(FiniteSet(pi/2, pi*Rational(5, 2))) \
                    is not S.EmptySet:
                return AccumBounds(Min(sin(min), sin(max)), 1)
            elif AccumBounds(min, max).intersection(FiniteSet(pi*Rational(3, 2), pi*Rational(8, 2))) \
                        is not S.EmptySet:
                return AccumBounds(-1, Max(sin(min), sin(max)))
            else:
                return AccumBounds(Min(sin(min), sin(max)),
                                Max(sin(min), sin(max)))
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import sinh
            return S.ImaginaryUnit*sinh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.Zero

            if (2*pi_coeff).is_integer:
                # is_even-case handled above as then pi_coeff.is_integer,
                # so check if known to be not even
                if pi_coeff.is_even is False:
                    return S.NegativeOne**(pi_coeff - S.Half)

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            # https://github.com/sympy/sympy/issues/6048
            # transform a sine to a cosine, to avoid redundant code
            if pi_coeff.is_Rational:
                x = pi_coeff % 2
                if x > 1:
                    return -cls((x % 1)*pi)
                if 2*x > 1:
                    return cls((1 - x)*pi)
                narg = ((pi_coeff + Rational(3, 2)) % 2)*pi
                result = cos(narg)
                if not isinstance(result, cos):
                    return result
                if pi_coeff*pi != arg:
                    return cls(pi_coeff*pi)
                return None

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                m = m*pi
                return sin(m)*cos(x) + cos(m)*sin(x)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, asin):
            return arg.args[0]

        if isinstance(arg, atan):
            x = arg.args[0]
            return x/sqrt(1 + x**2)

        if isinstance(arg, atan2):
            y, x = arg.args
            return y/sqrt(x**2 + y**2)

        if isinstance(arg, acos):
            x = arg.args[0]
            return sqrt(1 - x**2)

        if isinstance(arg, acot):
            x = arg.args[0]
            return 1/(sqrt(1 + 1/x**2)*x)

        if isinstance(arg, acsc):
            x = arg.args[0]
            return 1/x

        if isinstance(arg, asec):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p*x**2/(n*(n - 1))
            else:
                return S.NegativeOne**(n//2)*x**n/factorial(n)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        if logx is not None:
            arg = arg.subs(log(x), logx)
        if arg.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            raise PoleError("Cannot expand %s around 0" % (self))
        return super()._eval_nseries(x, n=n, logx=logx, cdir=cdir)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        I = S.ImaginaryUnit
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp)
        return (exp(arg*I) - exp(-arg*I))/(2*I)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return I*x**-I/2 - I*x**I /2

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(arg - pi/2, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        tan_half = tan(S.Half*arg)
        return 2*tan_half/(1 + tan_half**2)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)*cos(arg)/cos(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(S.Half*arg)
        return Piecewise((0, And(Eq(im(arg), 0), Eq(Mod(arg, pi), 0))),
                         (2*cot_half/(1 + cot_half**2), True))

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        return self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return 1/csc(arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1/sec(arg - pi/2, evaluate=False)

    def _eval_rewrite_as_sinc(self, arg, **kwargs):
        return arg*sinc(arg)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return sqrt(pi*arg/2)*besselj(S.Half, arg)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.hyperbolic import cosh, sinh
        re, im = self._as_real_imag(deep=deep, **hints)
        return (sin(re)*cosh(im), cos(re)*sinh(im))

    def _eval_expand_trig(self, **hints):
        from sympy.functions.special.polynomials import chebyshevt, chebyshevu
        arg = self.args[0]
        x = None
        if arg.is_Add:  # TODO, implement more if deep stuff here
            # TODO: Do this more efficiently for more than two terms
            x, y = arg.as_two_terms()
            sx = sin(x, evaluate=False)._eval_expand_trig()
            sy = sin(y, evaluate=False)._eval_expand_trig()
            cx = cos(x, evaluate=False)._eval_expand_trig()
            cy = cos(y, evaluate=False)._eval_expand_trig()
            return sx*cy + sy*cx
        elif arg.is_Mul:
            n, x = arg.as_coeff_Mul(rational=True)
            if n.is_Integer:  # n will be positive because of .eval
                # canonicalization

                # See https://mathworld.wolfram.com/Multiple-AngleFormulas.html
                if n.is_odd:
                    return S.NegativeOne**((n - 1)/2)*chebyshevt(n, sin(x))
                else:
                    return expand_mul(S.NegativeOne**(n/2 - 1)*cos(x)*
                                      chebyshevu(n - 1, sin(x)), deep=False)
        return sin(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = x0/pi
        if n.is_integer:
            lt = (arg - n*pi).as_leading_term(x)
            return (S.NegativeOne**n)*lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in [S.Infinity, S.NegativeInfinity]:
            return AccumBounds(-1, 1)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_extended_real:
            return True

    def _eval_is_zero(self):
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return pi_mult.is_integer

    def _eval_is_complex(self):
        if self.args[0].is_extended_real \
                or self.args[0].is_complex:
            return True


class cos(TrigonometricFunction):
    """
    The cosine function.

    Returns the cosine of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cos, pi
    >>> from sympy.abc import x
    >>> cos(x**2).diff(x)
    -2*x*sin(x**2)
    >>> cos(1).diff(x)
    0
    >>> cos(pi)
    -1
    >>> cos(pi/2)
    0
    >>> cos(2*pi/3)
    -1/2
    >>> cos(pi/12)
    sqrt(2)/4 + sqrt(6)/4

    See Also
    ========

    sin, csc, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cos

    """

    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -sin(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.functions.special.polynomials import chebyshevt
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.sets.setexpr import SetExpr
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.One
            elif arg in (S.Infinity, S.NegativeInfinity):
                # In this case it is better to return AccumBounds(-1, 1)
                # rather than returning S.NaN, since AccumBounds(-1, 1)
                # preserves the information that sin(oo) is between
                # -1 and 1, where S.NaN does not do that.
                return AccumBounds(-1, 1)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            return sin(arg + pi/2)
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.is_extended_real and arg.is_finite is False:
            return AccumBounds(-1, 1)

        if arg.could_extract_minus_sign():
            return cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import cosh
            return cosh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return (S.NegativeOne)**pi_coeff

            if (2*pi_coeff).is_integer:
                # is_even-case handled above as then pi_coeff.is_integer,
                # so check if known to be not even
                if pi_coeff.is_even is False:
                    return S.Zero

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            # cosine formula #####################
            # https://github.com/sympy/sympy/issues/6048
            # explicit calculations are performed for
            # cos(k pi/n) for n = 8,10,12,15,20,24,30,40,60,120
            # Some other exact values like cos(k pi/240) can be
            # calculated using a partial-fraction decomposition
            # by calling cos( X ).rewrite(sqrt)
            if pi_coeff.is_Rational:
                q = pi_coeff.q
                p = pi_coeff.p % (2*q)
                if p > q:
                    narg = (pi_coeff - 1)*pi
                    return -cls(narg)
                if 2*p > q:
                    narg = (1 - pi_coeff)*pi
                    return -cls(narg)

                # If nested sqrt's are worse than un-evaluation
                # you can require q to be in (1, 2, 3, 4, 6, 12)
                # q <= 12, q=15, q=20, q=24, q=30, q=40, q=60, q=120 return
                # expressions with 2 or fewer sqrt nestings.
                table2 = _table2()
                if q in table2:
                    a, b = table2[q]
                    a, b = p*pi/a, p*pi/b
                    nvala, nvalb = cls(a), cls(b)
                    if None in (nvala, nvalb):
                        return None
                    return nvala*nvalb + cls(pi/2 - a)*cls(pi/2 - b)

                if q > 12:
                    return None

                cst_table_some = {
                    3: S.Half,
                    5: (sqrt(5) + 1) / 4,
                }
                if q in cst_table_some:
                    cts = cst_table_some[pi_coeff.q]
                    return chebyshevt(pi_coeff.p, cts).expand()

                if 0 == q % 2:
                    narg = (pi_coeff*2)*pi
                    nval = cls(narg)
                    if None == nval:
                        return None
                    x = (2*pi_coeff + 1)/2
                    sign_cos = (-1)**((-1 if x < 0 else 1)*int(abs(x)))
                    return sign_cos*sqrt( (1 + nval)/2 )
            return None

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                m = m*pi
                return cos(m)*cos(x) - sin(m)*sin(x)

        if arg.is_zero:
            return S.One

        if isinstance(arg, acos):
            return arg.args[0]

        if isinstance(arg, atan):
            x = arg.args[0]
            return 1/sqrt(1 + x**2)

        if isinstance(arg, atan2):
            y, x = arg.args
            return x/sqrt(x**2 + y**2)

        if isinstance(arg, asin):
            x = arg.args[0]
            return sqrt(1 - x ** 2)

        if isinstance(arg, acot):
            x = arg.args[0]
            return 1/sqrt(1 + 1/x**2)

        if isinstance(arg, acsc):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)

        if isinstance(arg, asec):
            x = arg.args[0]
            return 1/x

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p*x**2/(n*(n - 1))
            else:
                return S.NegativeOne**(n//2)*x**n/factorial(n)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        if logx is not None:
            arg = arg.subs(log(x), logx)
        if arg.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            raise PoleError("Cannot expand %s around 0" % (self))
        return super()._eval_nseries(x, n=n, logx=logx, cdir=cdir)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        I = S.ImaginaryUnit
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp, **kwargs)
        return (exp(arg*I) + exp(-arg*I))/2

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return x**I/2 + x**-I/2

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return sin(arg + pi/2, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        tan_half = tan(S.Half*arg)**2
        return (1 - tan_half)/(1 + tan_half)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)*cos(arg)/sin(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(S.Half*arg)**2
        return Piecewise((1, And(Eq(im(arg), 0), Eq(Mod(arg, 2*pi), 0))),
                         ((cot_half - 1)/(cot_half + 1), True))

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self._eval_rewrite_as_sqrt(arg, **kwargs)

    def _eval_rewrite_as_sqrt(self, arg: Expr, **kwargs):
        from sympy.functions.special.polynomials import chebyshevt

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is None:
            return None

        if isinstance(pi_coeff, Integer):
            return None

        if not isinstance(pi_coeff, Rational):
            return None

        cst_table_some = cos_table()

        if pi_coeff.q in cst_table_some:
            rv = chebyshevt(pi_coeff.p, cst_table_some[pi_coeff.q]())
            if pi_coeff.q < 257:
                rv = rv.expand()
            return rv

        if not pi_coeff.q % 2:  # recursively remove factors of 2
            pico2 = pi_coeff * 2
            nval = cos(pico2 * pi).rewrite(sqrt, **kwargs)
            x = (pico2 + 1) / 2
            sign_cos = -1 if int(x) % 2 else 1
            return sign_cos * sqrt((1 + nval) / 2)

        FC = fermat_coords(pi_coeff.q)
        if FC:
            denoms = FC
        else:
            denoms = [b**e for b, e in factorint(pi_coeff.q).items()]

        apart = ipartfrac(*denoms)
        decomp = (pi_coeff.p * Rational(n, d) for n, d in zip(apart, denoms))
        X = [(x[1], x[0]*pi) for x in zip(decomp, numbered_symbols('z'))]
        pcls = cos(sum(x[0] for x in X))._eval_expand_trig().subs(X)

        if not FC or len(FC) == 1:
            return pcls
        return pcls.rewrite(sqrt, **kwargs)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1/sec(arg)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return 1/sec(arg).rewrite(csc, **kwargs)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return Piecewise(
                (sqrt(pi*arg/2)*besselj(-S.Half, arg), Ne(arg, 0)),
                (1, True)
            )

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.hyperbolic import cosh, sinh
        re, im = self._as_real_imag(deep=deep, **hints)
        return (cos(re)*cosh(im), -sin(re)*sinh(im))

    def _eval_expand_trig(self, **hints):
        from sympy.functions.special.polynomials import chebyshevt
        arg = self.args[0]
        x = None
        if arg.is_Add:  # TODO: Do this more efficiently for more than two terms
            x, y = arg.as_two_terms()
            sx = sin(x, evaluate=False)._eval_expand_trig()
            sy = sin(y, evaluate=False)._eval_expand_trig()
            cx = cos(x, evaluate=False)._eval_expand_trig()
            cy = cos(y, evaluate=False)._eval_expand_trig()
            return cx*cy - sx*sy
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer:
                return chebyshevt(coeff, cos(terms))
        return cos(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = (x0 + pi/2)/pi
        if n.is_integer:
            lt = (arg - n*pi + pi/2).as_leading_term(x)
            return (S.NegativeOne**n)*lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in [S.Infinity, S.NegativeInfinity]:
            return AccumBounds(-1, 1)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_finite(self):
        arg = self.args[0]

        if arg.is_extended_real:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_extended_real \
            or self.args[0].is_complex:
            return True

    def _eval_is_zero(self):
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero and pi_mult:
            return (pi_mult - S.Half).is_integer


class tan(TrigonometricFunction):
    """
    The tangent function.

    Returns the tangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import tan, pi
    >>> from sympy.abc import x
    >>> tan(x**2).diff(x)
    2*x*(tan(x**2)**2 + 1)
    >>> tan(1).diff(x)
    0
    >>> tan(pi/8).expand()
    -1 + sqrt(2)

    See Also
    ========

    sin, csc, cos, sec, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Tan

    """

    def period(self, symbol=None):
        return self._period(pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.One + self**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return atan

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(S.NegativeInfinity, S.Infinity)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            min, max = arg.min, arg.max
            d = floor(min/pi)
            if min is not S.NegativeInfinity:
                min = min - d*pi
            if max is not S.Infinity:
                max = max - d*pi
            from sympy.sets.sets import FiniteSet
            if AccumBounds(min, max).intersection(FiniteSet(pi/2, pi*Rational(3, 2))):
                return AccumBounds(S.NegativeInfinity, S.Infinity)
            else:
                return AccumBounds(tan(min), tan(max))

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import tanh
            return S.ImaginaryUnit*tanh(i_coeff)

        pi_coeff = _pi_coeff(arg, 2)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.Zero

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            if pi_coeff.is_Rational:
                q = pi_coeff.q
                p = pi_coeff.p % q
                # ensure simplified results are returned for n*pi/5, n*pi/10
                table10 = {
                    1: sqrt(1 - 2*sqrt(5)/5),
                    2: sqrt(5 - 2*sqrt(5)),
                    3: sqrt(1 + 2*sqrt(5)/5),
                    4: sqrt(5 + 2*sqrt(5))
                    }
                if q in (5, 10):
                    n = 10*p/q
                    if n > 5:
                        n = 10 - n
                        return -table10[n]
                    else:
                        return table10[n]
                if not pi_coeff.q % 2:
                    narg = pi_coeff*pi*2
                    cresult, sresult = cos(narg), cos(narg - pi/2)
                    if not isinstance(cresult, cos) \
                            and not isinstance(sresult, cos):
                        if sresult == 0:
                            return S.ComplexInfinity
                        return 1/sresult - cresult/sresult

                table2 = _table2()
                if q in table2:
                    a, b = table2[q]
                    nvala, nvalb = cls(p*pi/a), cls(p*pi/b)
                    if None in (nvala, nvalb):
                        return None
                    return (nvala - nvalb)/(1 + nvala*nvalb)
                narg = ((pi_coeff + S.Half) % 1 - S.Half)*pi
                # see cos() to specify which expressions should  be
                # expanded automatically in terms of radicals
                cresult, sresult = cos(narg), cos(narg - pi/2)
                if not isinstance(cresult, cos) \
                        and not isinstance(sresult, cos):
                    if cresult == 0:
                        return S.ComplexInfinity
                    return (sresult/cresult)
                if narg != arg:
                    return cls(narg)

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                tanm = tan(m*pi)
                if tanm is S.ComplexInfinity:
                    return -cot(x)
                else: # tanm == 0
                    return tan(x)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, atan):
            return arg.args[0]

        if isinstance(arg, atan2):
            y, x = arg.args
            return y/x

        if isinstance(arg, asin):
            x = arg.args[0]
            return x/sqrt(1 - x**2)

        if isinstance(arg, acos):
            x = arg.args[0]
            return sqrt(1 - x**2)/x

        if isinstance(arg, acot):
            x = arg.args[0]
            return 1/x

        if isinstance(arg, acsc):
            x = arg.args[0]
            return 1/(sqrt(1 - 1/x**2)*x)

        if isinstance(arg, asec):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)*x

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            a, b = ((n - 1)//2), 2**(n + 1)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return S.NegativeOne**a*b*(b - 1)*B/F*x**n

    def _eval_nseries(self, x, n, logx, cdir=0):
        i = self.args[0].limit(x, 0)*2/pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return super()._eval_nseries(x, n=n, logx=logx)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return I*(x**-I - x**I)/(x**-I + x**I)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        re, im = self._as_real_imag(deep=deep, **hints)
        if im:
            from sympy.functions.elementary.hyperbolic import cosh, sinh
            denom = cos(2*re) + cosh(2*im)
            return (sin(2*re)/denom, sinh(2*im)/denom)
        else:
            return (self.func(re), S.Zero)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        x = None
        if arg.is_Add:
            n = len(arg.args)
            TX = []
            for x in arg.args:
                tx = tan(x, evaluate=False)._eval_expand_trig()
                TX.append(tx)

            Yg = numbered_symbols('Y')
            Y = [ next(Yg) for i in range(n) ]

            p = [0, 0]
            for i in range(n + 1):
                p[1 - i % 2] += symmetric_poly(i, Y)*(-1)**((i % 4)//2)
            return (p[0]/p[1]).subs(list(zip(Y, TX)))

        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                I = S.ImaginaryUnit
                z = Symbol('dummy', real=True)
                P = ((1 + I*z)**coeff).expand()
                return (im(P)/re(P)).subs([(z, tan(terms))])
        return tan(arg)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        I = S.ImaginaryUnit
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp)
        neg_exp, pos_exp = exp(-arg*I), exp(arg*I)
        return I*(neg_exp - pos_exp)/(neg_exp + pos_exp)

    def _eval_rewrite_as_sin(self, x, **kwargs):
        return 2*sin(x)**2/sin(2*x)

    def _eval_rewrite_as_cos(self, x, **kwargs):
        return cos(x - pi/2, evaluate=False)/cos(x)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)/cos(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        return 1/cot(arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        sin_in_sec_form = sin(arg).rewrite(sec, **kwargs)
        cos_in_sec_form = cos(arg).rewrite(sec, **kwargs)
        return sin_in_sec_form/cos_in_sec_form

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        sin_in_csc_form = sin(arg).rewrite(csc, **kwargs)
        cos_in_csc_form = cos(arg).rewrite(csc, **kwargs)
        return sin_in_csc_form/cos_in_csc_form

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return besselj(S.Half, arg)/besselj(-S.Half, arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = 2*x0/pi
        if n.is_integer:
            lt = (arg - n*pi/2).as_leading_term(x)
            return lt if n.is_even else -1/lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        # FIXME: currently tan(pi/2) return zoo
        return self.args[0].is_extended_real

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True

    def _eval_is_finite(self):
        arg = self.args[0]

        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True

        if arg.is_imaginary:
            return True

    def _eval_is_zero(self):
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return pi_mult.is_integer

    def _eval_is_complex(self):
        arg = self.args[0]

        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True


class cot(TrigonometricFunction):
    """
    The cotangent function.

    Returns the cotangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cot, pi
    >>> from sympy.abc import x
    >>> cot(x**2).diff(x)
    2*x*(-cot(x**2)**2 - 1)
    >>> cot(1).diff(x)
    0
    >>> cot(pi/12)
    sqrt(3) + 2

    See Also
    ========

    sin, csc, cos, sec, tan
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cot

    """

    def period(self, symbol=None):
        return self._period(pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.NegativeOne - self**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return acot

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            if arg.is_zero:
                return S.ComplexInfinity
            elif arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(S.NegativeInfinity, S.Infinity)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            return -tan(arg + pi/2)

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import coth
            return -S.ImaginaryUnit*coth(i_coeff)

        pi_coeff = _pi_coeff(arg, 2)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.ComplexInfinity

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            if pi_coeff.is_Rational:
                if pi_coeff.q in (5, 10):
                    return tan(pi/2 - arg)
                if pi_coeff.q > 2 and not pi_coeff.q % 2:
                    narg = pi_coeff*pi*2
                    cresult, sresult = cos(narg), cos(narg - pi/2)
                    if not isinstance(cresult, cos) \
                            and not isinstance(sresult, cos):
                        return 1/sresult + cresult/sresult
                q = pi_coeff.q
                p = pi_coeff.p % q
                table2 = _table2()
                if q in table2:
                    a, b = table2[q]
                    nvala, nvalb = cls(p*pi/a), cls(p*pi/b)
                    if None in (nvala, nvalb):
                        return None
                    return (1 + nvala*nvalb)/(nvalb - nvala)
                narg = (((pi_coeff + S.Half) % 1) - S.Half)*pi
                # see cos() to specify which expressions should be
                # expanded automatically in terms of radicals
                cresult, sresult = cos(narg), cos(narg - pi/2)
                if not isinstance(cresult, cos) \
                        and not isinstance(sresult, cos):
                    if sresult == 0:
                        return S.ComplexInfinity
                    return cresult/sresult
                if narg != arg:
                    return cls(narg)

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                cotm = cot(m*pi)
                if cotm is S.ComplexInfinity:
                    return cot(x)
                else: # cotm == 0
                    return -tan(x)

        if arg.is_zero:
            return S.ComplexInfinity

        if isinstance(arg, acot):
            return arg.args[0]

        if isinstance(arg, atan):
            x = arg.args[0]
            return 1/x

        if isinstance(arg, atan2):
            y, x = arg.args
            return x/y

        if isinstance(arg, asin):
            x = arg.args[0]
            return sqrt(1 - x**2)/x

        if isinstance(arg, acos):
            x = arg.args[0]
            return x/sqrt(1 - x**2)

        if isinstance(arg, acsc):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)*x

        if isinstance(arg, asec):
            x = arg.args[0]
            return 1/(sqrt(1 - 1/x**2)*x)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return S.NegativeOne**((n + 1)//2)*2**(n + 1)*B/F*x**n

    def _eval_nseries(self, x, n, logx, cdir=0):
        i = self.args[0].limit(x, 0)/pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return self.rewrite(tan)._eval_nseries(x, n=n, logx=logx)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        re, im = self._as_real_imag(deep=deep, **hints)
        if im:
            from sympy.functions.elementary.hyperbolic import cosh, sinh
            denom = cos(2*re) - cosh(2*im)
            return (-sin(2*re)/denom, sinh(2*im)/denom)
        else:
            return (self.func(re), S.Zero)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        I = S.ImaginaryUnit
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp, **kwargs)
        neg_exp, pos_exp = exp(-arg*I), exp(arg*I)
        return I*(pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return -I*(x**-I + x**I)/(x**-I - x**I)

    def _eval_rewrite_as_sin(self, x, **kwargs):
        return sin(2*x)/(2*(sin(x)**2))

    def _eval_rewrite_as_cos(self, x, **kwargs):
        return cos(x)/cos(x - pi/2, evaluate=False)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return cos(arg)/sin(arg)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return 1/tan(arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        cos_in_sec_form = cos(arg).rewrite(sec, **kwargs)
        sin_in_sec_form = sin(arg).rewrite(sec, **kwargs)
        return cos_in_sec_form/sin_in_sec_form

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        cos_in_csc_form = cos(arg).rewrite(csc, **kwargs)
        sin_in_csc_form = sin(arg).rewrite(csc, **kwargs)
        return cos_in_csc_form/sin_in_csc_form

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return besselj(-S.Half, arg)/besselj(S.Half, arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = 2*x0/pi
        if n.is_integer:
            lt = (arg - n*pi/2).as_leading_term(x)
            return 1/lt if n.is_even else -lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        x = None
        if arg.is_Add:
            n = len(arg.args)
            CX = []
            for x in arg.args:
                cx = cot(x, evaluate=False)._eval_expand_trig()
                CX.append(cx)

            Yg = numbered_symbols('Y')
            Y = [ next(Yg) for i in range(n) ]

            p = [0, 0]
            for i in range(n, -1, -1):
                p[(n - i) % 2] += symmetric_poly(i, Y)*(-1)**(((n - i) % 4)//2)
            return (p[0]/p[1]).subs(list(zip(Y, CX)))
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                I = S.ImaginaryUnit
                z = Symbol('dummy', real=True)
                P = ((z + I)**coeff).expand()
                return (re(P)/im(P)).subs([(z, cot(terms))])
        return cot(arg)  # XXX sec and csc return 1/cos and 1/sin

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True
        if arg.is_imaginary:
            return True

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    def _eval_is_complex(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    def _eval_is_zero(self):
        rest, pimult = _peeloff_pi(self.args[0])
        if pimult and rest.is_zero:
            return (pimult - S.Half).is_integer

    def _eval_subs(self, old, new):
        arg = self.args[0]
        argnew = arg.subs(old, new)
        if arg != argnew and (argnew/pi).is_integer:
            return S.ComplexInfinity
        return cot(argnew)


class ReciprocalTrigonometricFunction(TrigonometricFunction):
    """Base class for reciprocal functions of trigonometric functions. """

    _reciprocal_of = None       # mandatory, to be defined in subclass
    _singularities = (S.ComplexInfinity,)

    # _is_even and _is_odd are used for correct evaluation of csc(-x), sec(-x)
    # TODO refactor into TrigonometricFunction common parts of
    # trigonometric functions eval() like even/odd, func(x+2*k*pi), etc.

    # optional, to be defined in subclasses:
    _is_even: FuzzyBool = None
    _is_odd: FuzzyBool = None

    @classmethod
    def eval(cls, arg):
        if arg.could_extract_minus_sign():
            if cls._is_even:
                return cls(-arg)
            if cls._is_odd:
                return -cls(-arg)

        pi_coeff = _pi_coeff(arg)
        if (pi_coeff is not None
            and not (2*pi_coeff).is_integer
            and pi_coeff.is_Rational):
                q = pi_coeff.q
                p = pi_coeff.p % (2*q)
                if p > q:
                    narg = (pi_coeff - 1)*pi
                    return -cls(narg)
                if 2*p > q:
                    narg = (1 - pi_coeff)*pi
                    if cls._is_odd:
                        return cls(narg)
                    elif cls._is_even:
                        return -cls(narg)

        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]

        t = cls._reciprocal_of.eval(arg)
        if t is None:
            return t
        elif any(isinstance(i, cos) for i in (t, -t)):
            return (1/t).rewrite(sec)
        elif any(isinstance(i, sin) for i in (t, -t)):
            return (1/t).rewrite(csc)
        else:
            return 1/t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        # Calls method_name on _reciprocal_of
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        # If calling method_name on _reciprocal_of returns a value != None
        # then return the reciprocal of that value
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1/t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        # Special handling for rewrite functions. If reciprocal rewrite returns
        # unmodified expression, then return None
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1/t

    def _period(self, symbol):
        f = expand_mul(self.args[0])
        return self._reciprocal_of(f).period(symbol)

    def fdiff(self, argindex=1):
        return -self._calculate_reciprocal("fdiff", argindex)/self**2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_exp", arg)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_Pow", arg)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_sin", arg)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_cos", arg)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tan", arg)

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_pow", arg)

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_sqrt", arg)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        return (1/self._reciprocal_of(self.args[0])).as_real_imag(deep,
                                                                  **hints)

    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal("_eval_expand_trig", **hints)

    def _eval_is_extended_real(self):
        return self._reciprocal_of(self.args[0])._eval_is_extended_real()

    def _eval_as_leading_term(self, x, logx, cdir):
        return (1/self._reciprocal_of(self.args[0]))._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_finite(self):
        return (1/self._reciprocal_of(self.args[0])).is_finite

    def _eval_nseries(self, x, n, logx, cdir=0):
        return (1/self._reciprocal_of(self.args[0]))._eval_nseries(x, n, logx)


class sec(ReciprocalTrigonometricFunction):
    """
    The secant function.

    Returns the secant of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import sec
    >>> from sympy.abc import x
    >>> sec(x**2).diff(x)
    2*x*tan(x**2)*sec(x**2)
    >>> sec(1).diff(x)
    0

    See Also
    ========

    sin, csc, cos, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sec

    """

    _reciprocal_of = cos
    _is_even = True

    def period(self, symbol=None):
        return self._period(symbol)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half_sq = cot(arg/2)**2
        return (cot_half_sq + 1)/(cot_half_sq - 1)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return (1/cos(arg))

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)/(cos(arg)*sin(arg))

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return (1/cos(arg).rewrite(sin, **kwargs))

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return (1/cos(arg).rewrite(tan, **kwargs))

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return csc(pi/2 - arg, evaluate=False)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return tan(self.args[0])*sec(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return Piecewise(
                (1/(sqrt(pi*arg)/(sqrt(2))*besselj(-S.Half, arg)), Ne(arg, 0)),
                (1, True)
            )

    def _eval_is_complex(self):
        arg = self.args[0]

        if arg.is_complex and (arg/pi - S.Half).is_integer is False:
            return True

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # Reference Formula:
        # https://functions.wolfram.com/ElementaryFunctions/Sec/06/01/02/01/
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            k = n//2
            return S.NegativeOne**k*euler(2*k)/factorial(2*k)*x**(2*k)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = (x0 + pi/2)/pi
        if n.is_integer:
            lt = (arg - n*pi + pi/2).as_leading_term(x)
            return (S.NegativeOne**n)/lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self


class csc(ReciprocalTrigonometricFunction):
    """
    The cosecant function.

    Returns the cosecant of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import csc
    >>> from sympy.abc import x
    >>> csc(x**2).diff(x)
    -2*x*cot(x**2)*csc(x**2)
    >>> csc(1).diff(x)
    0

    See Also
    ========

    sin, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Csc

    """

    _reciprocal_of = sin
    _is_odd = True

    def period(self, symbol=None):
        return self._period(symbol)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return (1/sin(arg))

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return cos(arg)/(sin(arg)*cos(arg))

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(arg/2)
        return (1 + cot_half**2)/(2*cot_half)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1/sin(arg).rewrite(cos, **kwargs)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(pi/2 - arg, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return (1/sin(arg).rewrite(tan, **kwargs))

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return sqrt(2/pi)*(1/(sqrt(arg)*besselj(S.Half, arg)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -cot(self.args[0])*csc(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_complex(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n//2 + 1
            return (S.NegativeOne**(k - 1)*2*(2**(2*k - 1) - 1)*
                    bernoulli(2*k)*x**(2*k - 1)/factorial(2*k))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = x0/pi
        if n.is_integer:
            lt = (arg - n*pi).as_leading_term(x)
            return (S.NegativeOne**n)/lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self


class sinc(DefinedFunction):
    r"""
    Represents an unnormalized sinc function:

    .. math::

        \operatorname{sinc}(x) =
        \begin{cases}
          \frac{\sin x}{x} & \qquad x \neq 0 \\
          1 & \qquad x = 0
        \end{cases}

    Examples
    ========

    >>> from sympy import sinc, oo, jn
    >>> from sympy.abc import x
    >>> sinc(x)
    sinc(x)

    * Automated Evaluation

    >>> sinc(0)
    1
    >>> sinc(oo)
    0

    * Differentiation

    >>> sinc(x).diff()
    cos(x)/x - sin(x)/x**2

    * Series Expansion

    >>> sinc(x).series()
    1 - x**2/6 + x**4/120 + O(x**6)

    * As zero'th order spherical Bessel Function

    >>> sinc(x).rewrite(jn)
    jn(0, x)

    See also
    ========

    sin

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Sinc_function

    """
    _singularities = (S.ComplexInfinity,)

    def fdiff(self, argindex=1):
        x = self.args[0]
        if argindex == 1:
            # We would like to return the Piecewise here, but Piecewise.diff
            # currently can't handle removable singularities, meaning things
            # like sinc(x).diff(x, 2) give the wrong answer at x = 0. See
            # https://github.com/sympy/sympy/issues/11402.
            #
            # return Piecewise(((x*cos(x) - sin(x))/x**2, Ne(x, S.Zero)), (S.Zero, S.true))
            return cos(x)/x - sin(x)/x**2
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.One
        if arg.is_Number:
            if arg in [S.Infinity, S.NegativeInfinity]:
                return S.Zero
            elif arg is S.NaN:
                return S.NaN

        if arg is S.ComplexInfinity:
            return S.NaN

        if arg.could_extract_minus_sign():
            return cls(-arg)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                if fuzzy_not(arg.is_zero):
                    return S.Zero
            elif (2*pi_coeff).is_integer:
                return S.NegativeOne**(pi_coeff - S.Half)/arg

    def _eval_nseries(self, x, n, logx, cdir=0):
        x = self.args[0]
        return (sin(x)/x)._eval_nseries(x, n, logx)

    def _eval_rewrite_as_jn(self, arg, **kwargs):
        from sympy.functions.special.bessel import jn
        return jn(0, arg)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return Piecewise((sin(arg)/arg, Ne(arg, S.Zero)), (S.One, S.true))

    def _eval_is_zero(self):
        if self.args[0].is_infinite:
            return True
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return fuzzy_and([pi_mult.is_integer, pi_mult.is_nonzero])
        if rest.is_Number and pi_mult.is_integer:
            return False

    def _eval_is_real(self):
        if self.args[0].is_extended_real or self.args[0].is_imaginary:
            return True

    _eval_is_finite = _eval_is_real


###############################################################################
########################### TRIGONOMETRIC INVERSES ############################
###############################################################################


class InverseTrigonometricFunction(DefinedFunction):
    """Base class for inverse trigonometric functions."""
    _singularities: tuple[Expr, ...] = (S.One, S.NegativeOne, S.Zero, S.ComplexInfinity)

    @staticmethod
    @cacheit
    def _asin_table():
        # Only keys with could_extract_minus_sign() == False
        # are actually needed.
        return {
            sqrt(3)/2: pi/3,
            sqrt(2)/2: pi/4,
            1/sqrt(2): pi/4,
            sqrt((5 - sqrt(5))/8): pi/5,
            sqrt(2)*sqrt(5 - sqrt(5))/4: pi/5,
            sqrt((5 + sqrt(5))/8): pi*Rational(2, 5),
            sqrt(2)*sqrt(5 + sqrt(5))/4: pi*Rational(2, 5),
            S.Half: pi/6,
            sqrt(2 - sqrt(2))/2: pi/8,
            sqrt(S.Half - sqrt(2)/4): pi/8,
            sqrt(2 + sqrt(2))/2: pi*Rational(3, 8),
            sqrt(S.Half + sqrt(2)/4): pi*Rational(3, 8),
            (sqrt(5) - 1)/4: pi/10,
            (1 - sqrt(5))/4: -pi/10,
            (sqrt(5) + 1)/4: pi*Rational(3, 10),
            sqrt(6)/4 - sqrt(2)/4: pi/12,
            -sqrt(6)/4 + sqrt(2)/4: -pi/12,
            (sqrt(3) - 1)/sqrt(8): pi/12,
            (1 - sqrt(3))/sqrt(8): -pi/12,
            sqrt(6)/4 + sqrt(2)/4: pi*Rational(5, 12),
            (1 + sqrt(3))/sqrt(8): pi*Rational(5, 12)
        }


    @staticmethod
    @cacheit
    def _atan_table():
        # Only keys with could_extract_minus_sign() == False
        # are actually needed.
        return {
            sqrt(3)/3: pi/6,
            1/sqrt(3): pi/6,
            sqrt(3): pi/3,
            sqrt(2) - 1: pi/8,
            1 - sqrt(2): -pi/8,
            1 + sqrt(2): pi*Rational(3, 8),
            sqrt(5 - 2*sqrt(5)): pi/5,
            sqrt(5 + 2*sqrt(5)): pi*Rational(2, 5),
            sqrt(1 - 2*sqrt(5)/5): pi/10,
            sqrt(1 + 2*sqrt(5)/5): pi*Rational(3, 10),
            2 - sqrt(3): pi/12,
            -2 + sqrt(3): -pi/12,
            2 + sqrt(3): pi*Rational(5, 12)
        }

    @staticmethod
    @cacheit
    def _acsc_table():
        # Keys for which could_extract_minus_sign()
        # will obviously return True are omitted.
        return {
            2*sqrt(3)/3: pi/3,
            sqrt(2): pi/4,
            sqrt(2 + 2*sqrt(5)/5): pi/5,
            1/sqrt(Rational(5, 8) - sqrt(5)/8): pi/5,
            sqrt(2 - 2*sqrt(5)/5): pi*Rational(2, 5),
            1/sqrt(Rational(5, 8) + sqrt(5)/8): pi*Rational(2, 5),
            2: pi/6,
            sqrt(4 + 2*sqrt(2)): pi/8,
            2/sqrt(2 - sqrt(2)): pi/8,
            sqrt(4 - 2*sqrt(2)): pi*Rational(3, 8),
            2/sqrt(2 + sqrt(2)): pi*Rational(3, 8),
            1 + sqrt(5): pi/10,
            sqrt(5) - 1: pi*Rational(3, 10),
            -(sqrt(5) - 1): pi*Rational(-3, 10),
            sqrt(6) + sqrt(2): pi/12,
            sqrt(6) - sqrt(2): pi*Rational(5, 12),
            -(sqrt(6) - sqrt(2)): pi*Rational(-5, 12)
        }


class asin(InverseTrigonometricFunction):
    r"""
    The inverse sine function.

    Returns the arcsine of x in radians.

    Explanation
    ===========

    ``asin(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the ``eval`` class method).

    A purely imaginary argument will lead to an asinh expression.

    Examples
    ========

    >>> from sympy import asin, oo
    >>> asin(1)
    pi/2
    >>> asin(-1)
    -pi/2
    >>> asin(-oo)
    oo*I
    >>> asin(oo)
    -oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSin

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/sqrt(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    def _eval_is_positive(self):
        return self._eval_is_extended_real() and self.args[0].is_positive

    def _eval_is_negative(self):
        return self._eval_is_extended_real() and self.args[0].is_negative

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.NegativeInfinity*S.ImaginaryUnit
            elif arg is S.NegativeInfinity:
                return S.Infinity*S.ImaginaryUnit
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return pi/2
            elif arg is S.NegativeOne:
                return -pi/2

        if arg is S.ComplexInfinity:
            return S.ComplexInfinity

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            asin_table = cls._asin_table()
            if arg in asin_table:
                return asin_table[arg]

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import asinh
            return S.ImaginaryUnit*asinh(i_coeff)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, sin):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= 2*pi # restrict to [0,2*pi)
                if ang > pi: # restrict to (-pi,pi]
                    ang = pi - ang

                # restrict to [-pi/2,pi/2]
                if ang > pi/2:
                    ang = pi - ang
                if ang < -pi/2:
                    ang = -pi - ang

                return ang

        if isinstance(arg, cos): # acos(x) + asin(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                return pi/2 - acos(arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p*(n - 2)**2/(n*(n - 1))*x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return R/F*x**n/n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        if x0.is_zero:
            return arg.as_leading_term(x)

        # Handling branch points
        if x0 in (-S.One, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # Handling points lying on branch cuts (-oo, -1) U (1, oo)
        if (1 - x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return -pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return pi - self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asin
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asin(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asin(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else -pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        # Handling points lying on branch cuts (-oo, -1) U (1, oo)
        if (1 - arg0**2).is_negative:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return -pi - res
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return pi - res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return pi/2 - acos(x)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        return 2*atan(x/(1 + sqrt(1 - x**2)))

    def _eval_rewrite_as_log(self, x, **kwargs):
        return -S.ImaginaryUnit*log(S.ImaginaryUnit*x + sqrt(1 - x**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return 2*acot((1 + sqrt(1 - arg**2))/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return pi/2 - asec(1/arg)

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return acsc(1/arg)

    def _eval_is_extended_real(self):
        x = self.args[0]
        return x.is_extended_real and (1 - abs(x)).is_nonnegative

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sin


class acos(InverseTrigonometricFunction):
    r"""
    The inverse cosine function.

    Explanation
    ===========

    Returns the arc cosine of x (measured in radians).

    ``acos(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when
    the result is a rational multiple of $\pi$ (see the eval class method).

    ``acos(zoo)`` evaluates to ``zoo``
    (see note in :class:`sympy.functions.elementary.trigonometric.asec`)

    A purely imaginary argument will be rewritten to asinh.

    Examples
    ========

    >>> from sympy import acos, oo
    >>> acos(1)
    0
    >>> acos(0)
    pi/2
    >>> acos(oo)
    oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCos

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/sqrt(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity*S.ImaginaryUnit
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity*S.ImaginaryUnit
            elif arg.is_zero:
                return pi/2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi

        if arg is S.ComplexInfinity:
            return S.ComplexInfinity

        if arg.is_number:
            asin_table = cls._asin_table()
            if arg in asin_table:
                return pi/2 - asin_table[arg]
            elif -arg in asin_table:
                return pi/2 + asin_table[-arg]

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            return pi/2 - asin(arg)

        if arg.is_Mul and len(arg.args) == 2 and arg.args[0] == -1:
            narg = arg.args[1]
            minus = True
        else:
            narg = arg
            minus = False

        if isinstance(narg, cos):
            # acos(cos(x)) = x or acos(-cos(x)) = pi - x
            ang = narg.args[0]
            if ang.is_comparable:
                if minus:
                    ang = pi - ang
                ang %= 2*pi # restrict to [0,2*pi)
                if ang > pi: # restrict to [0,pi]
                    ang = 2*pi - ang
                return ang

        if isinstance(narg, sin): # acos(x) + asin(x) = pi/2
            ang = narg.args[0]
            if ang.is_comparable:
                if minus:
                    return pi/2 + asin(narg)
                return pi/2 - asin(narg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p*(n - 2)**2/(n*(n - 1))*x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return -R/F*x**n/n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # Handling branch points
        if x0 == 1:
            return sqrt(2)*sqrt((S.One - arg).as_leading_term(x))
        if x0 in (-S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-oo, -1) U (1, oo)
        if (1 - x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return 2*pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return -self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_is_extended_real(self):
        x = self.args[0]
        return x.is_extended_real and (1 - abs(x)).is_nonnegative

    def _eval_is_nonnegative(self):
        return self._eval_is_extended_real()

    def _eval_nseries(self, x, n, logx, cdir=0):  # acos
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = acos(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = acos(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else pi + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        # Handling points lying on branch cuts (-oo, -1) U (1, oo)
        if (1 - arg0**2).is_negative:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return 2*pi - res
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return -res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return pi/2 + S.ImaginaryUnit*\
            log(S.ImaginaryUnit*x + sqrt(1 - x**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return pi/2 - asin(x)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        return atan(sqrt(1 - x**2)/x) + (pi/2)*(1 - x*sqrt(1/x**2))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cos

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return pi/2 - 2*acot((1 + sqrt(1 - arg**2))/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return asec(1/arg)

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return pi/2 - acsc(1/arg)

    def _eval_conjugate(self):
        z = self.args[0]
        r = self.func(self.args[0].conjugate())
        if z.is_extended_real is False:
            return r
        elif z.is_extended_real and (z + 1).is_nonnegative and (z - 1).is_nonpositive:
            return r


class atan(InverseTrigonometricFunction):
    r"""
    The inverse tangent function.

    Returns the arc tangent of x (measured in radians).

    Explanation
    ===========

    ``atan(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the eval class method).

    Examples
    ========

    >>> from sympy import atan, oo
    >>> atan(0)
    0
    >>> atan(1)
    pi/4
    >>> atan(oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan

    """

    args: tuple[Expr]

    _singularities = (S.ImaginaryUnit, -S.ImaginaryUnit)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1 + self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    def _eval_is_positive(self):
        return self.args[0].is_extended_positive

    def _eval_is_nonnegative(self):
        return self.args[0].is_extended_nonnegative

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return pi/2
            elif arg is S.NegativeInfinity:
                return -pi/2
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return pi/4
            elif arg is S.NegativeOne:
                return -pi/4

        if arg is S.ComplexInfinity:
            from sympy.calculus.accumulationbounds import AccumBounds
            return AccumBounds(-pi/2, pi/2)

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            atan_table = cls._atan_table()
            if arg in atan_table:
                return atan_table[arg]

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import atanh
            return S.ImaginaryUnit*atanh(i_coeff)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, tan):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= pi # restrict to [0,pi)
                if ang > pi/2: # restrict to [-pi/2,pi/2]
                    ang -= pi

                return ang

        if isinstance(arg, cot): # atan(x) + acot(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                ang = pi/2 - acot(arg)
                if ang > pi/2: # restrict to [-pi/2,pi/2]
                    ang -= pi
                return ang

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return S.NegativeOne**((n - 1)//2)*x**n/n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        if x0.is_zero:
            return arg.as_leading_term(x)
        # Handling branch points
        if x0 in (-S.ImaginaryUnit, S.ImaginaryUnit, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_negative:
                if im(x0).is_positive:
                    return self.func(x0) - pi
            elif re(ndir).is_positive:
                if im(x0).is_negative:
                    return self.func(x0) + pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # atan
        arg0 = self.args[0].subs(x, 0)

        # Handling branch points
        if arg0 in (S.ImaginaryUnit, S.NegativeOne*S.ImaginaryUnit):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        ndir = self.args[0].dir(x, cdir if cdir else 1)
        if arg0 is S.ComplexInfinity:
            if re(ndir) > 0:
                return res - pi
            return res
        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + arg0**2).is_negative:
            if re(ndir).is_negative:
                if im(arg0).is_positive:
                    return res - pi
            elif re(ndir).is_positive:
                if im(arg0).is_negative:
                    return res + pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return S.ImaginaryUnit/2*(log(S.One - S.ImaginaryUnit*x)
            - log(S.One + S.ImaginaryUnit*x))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] in [S.Infinity, S.NegativeInfinity]:
            return (pi/2 - atan(1/self.args[0]))._eval_nseries(x, n, logx)
        else:
            return super()._eval_aseries(n, args0, x, logx)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return tan

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return sqrt(arg**2)/arg*(pi/2 - asin(1/sqrt(1 + arg**2)))

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return sqrt(arg**2)/arg*acos(1/sqrt(1 + arg**2))

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return acot(1/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return sqrt(arg**2)/arg*asec(sqrt(1 + arg**2))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return sqrt(arg**2)/arg*(pi/2 - acsc(sqrt(1 + arg**2)))


class acot(InverseTrigonometricFunction):
    r"""
    The inverse cotangent function.

    Returns the arc cotangent of x (measured in radians).

    Explanation
    ===========

    ``acot(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, \tilde{\infty}, 0, 1, -1\}$
    and for some instances when the result is a rational multiple of $\pi$
    (see the eval class method).

    A purely imaginary argument will lead to an ``acoth`` expression.

    ``acot(x)`` has a branch cut along $(-i, i)$, hence it is discontinuous
    at 0. Its range for real $x$ is $(-\frac{\pi}{2}, \frac{\pi}{2}]$.

    Examples
    ========

    >>> from sympy import acot, sqrt
    >>> acot(0)
    pi/2
    >>> acot(1)
    pi/4
    >>> acot(sqrt(3) - 2)
    -5*pi/12

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, atan2

    References
    ==========

    .. [1] https://dlmf.nist.gov/4.23
    .. [2] https://functions.wolfram.com/ElementaryFunctions/ArcCot

    """
    _singularities = (S.ImaginaryUnit, -S.ImaginaryUnit)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/(1 + self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    def _eval_is_positive(self):
        return self.args[0].is_nonnegative

    def _eval_is_negative(self):
        return self.args[0].is_negative

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return pi/ 2
            elif arg is S.One:
                return pi/4
            elif arg is S.NegativeOne:
                return -pi/4

        if arg is S.ComplexInfinity:
            return S.Zero

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            atan_table = cls._atan_table()
            if arg in atan_table:
                ang = pi/2 - atan_table[arg]
                if ang > pi/2: # restrict to (-pi/2,pi/2]
                    ang -= pi
                return ang

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import acoth
            return -S.ImaginaryUnit*acoth(i_coeff)

        if arg.is_zero:
            return pi*S.Half

        if isinstance(arg, cot):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= pi # restrict to [0,pi)
                if ang > pi/2: # restrict to (-pi/2,pi/2]
                    ang -= pi
                return ang

        if isinstance(arg, tan): # atan(x) + acot(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                ang = pi/2 - atan(arg)
                if ang > pi/2: # restrict to (-pi/2,pi/2]
                    ang -= pi
                return ang

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return pi/2  # FIX THIS
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return S.NegativeOne**((n + 1)//2)*x**n/n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # Handling branch points
        if x0 in (-S.ImaginaryUnit, S.ImaginaryUnit, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # Handling points lying on branch cuts [-I, I]
        if x0.is_imaginary and (1 + x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_positive:
                    return self.func(x0) + pi
            elif re(ndir).is_negative:
                if im(x0).is_negative:
                    return self.func(x0) - pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acot
        arg0 = self.args[0].subs(x, 0)

        # Handling branch points
        if arg0 in (S.ImaginaryUnit, S.NegativeOne*S.ImaginaryUnit):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        ndir = self.args[0].dir(x, cdir if cdir else 1)
        if arg0.is_zero:
            if re(ndir) < 0:
                return res - pi
            return res
        # Handling points lying on branch cuts [-I, I]
        if arg0.is_imaginary and (1 + arg0**2).is_positive:
            if re(ndir).is_positive:
                if im(arg0).is_positive:
                    return res + pi
            elif re(ndir).is_negative:
                if im(arg0).is_negative:
                    return res - pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] in [S.Infinity, S.NegativeInfinity]:
            return atan(1/self.args[0])._eval_nseries(x, n, logx)
        else:
            return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_log(self, x, **kwargs):
        return S.ImaginaryUnit/2*(log(1 - S.ImaginaryUnit/x)
            - log(1 + S.ImaginaryUnit/x))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cot

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return (arg*sqrt(1/arg**2)*
                (pi/2 - asin(sqrt(-arg**2)/sqrt(-arg**2 - 1))))

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return arg*sqrt(1/arg**2)*acos(sqrt(-arg**2)/sqrt(-arg**2 - 1))

    def _eval_rewrite_as_atan(self, arg, **kwargs):
        return atan(1/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return arg*sqrt(1/arg**2)*asec(sqrt((1 + arg**2)/arg**2))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return arg*sqrt(1/arg**2)*(pi/2 - acsc(sqrt((1 + arg**2)/arg**2)))


class asec(InverseTrigonometricFunction):
    r"""
    The inverse secant function.

    Returns the arc secant of x (measured in radians).

    Explanation
    ===========

    ``asec(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the eval class method).

    ``asec(x)`` has branch cut in the interval $[-1, 1]$. For complex arguments,
    it can be defined [4]_ as

    .. math::
        \operatorname{sec^{-1}}(z) = -i\frac{\log\left(\sqrt{1 - z^2} + 1\right)}{z}

    At ``x = 0``, for positive branch cut, the limit evaluates to ``zoo``. For
    negative branch cut, the limit

    .. math::
        \lim_{z \to 0}-i\frac{\log\left(-\sqrt{1 - z^2} + 1\right)}{z}

    simplifies to :math:`-i\log\left(z/2 + O\left(z^3\right)\right)` which
    ultimately evaluates to ``zoo``.

    As ``acos(x) = asec(1/x)``, a similar argument can be given for
    ``acos(x)``.

    Examples
    ========

    >>> from sympy import asec, oo
    >>> asec(1)
    0
    >>> asec(-1)
    pi
    >>> asec(0)
    zoo
    >>> asec(-oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSec
    .. [4] https://reference.wolfram.com/language/ref/ArcSec.html

    """

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.ComplexInfinity
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi
        if arg in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
            return pi/2

        if arg.is_number:
            acsc_table = cls._acsc_table()
            if arg in acsc_table:
                return pi/2 - acsc_table[arg]
            elif -arg in acsc_table:
                return pi/2 + acsc_table[-arg]

        if arg.is_infinite:
            return pi/2

        if arg.is_Mul and len(arg.args) == 2 and arg.args[0] == -1:
            narg = arg.args[1]
            minus = True
        else:
            narg = arg
            minus = False

        if isinstance(narg, sec):
            # asec(sec(x)) = x or asec(-sec(x)) = pi - x
            ang = narg.args[0]
            if ang.is_comparable:
                if minus:
                    ang = pi - ang
                ang %= 2*pi # restrict to [0,2*pi)
                if ang > pi: # restrict to [0,pi]
                    ang = 2*pi - ang
                return ang

        if isinstance(narg, csc): # asec(x) + acsc(x) = pi/2
            ang = narg.args[0]
            if ang.is_comparable:
                if minus:
                    pi/2 + acsc(narg)
                return pi/2 - acsc(narg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(self.args[0]**2*sqrt(1 - 1/self.args[0]**2))
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sec

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return S.ImaginaryUnit*log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return -S.ImaginaryUnit * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # Handling branch points
        if x0 == 1:
            return sqrt(2)*sqrt((arg - S.One).as_leading_term(x))
        if x0 in (-S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-1, 1)
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return -self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return 2*pi - self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asec
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asec(S.One + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.NegativeOne + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asec(S.NegativeOne - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.NegativeOne - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        # Handling points lying on branch cuts (-1, 1)
        if arg0.is_real and (1 - arg0**2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return -res
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return 2*pi - res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_is_extended_real(self):
        x = self.args[0]
        if x.is_extended_real is False:
            return False
        return fuzzy_or(((x - 1).is_nonnegative, (-x - 1).is_nonnegative))

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return pi/2 + S.ImaginaryUnit*log(S.ImaginaryUnit/arg + sqrt(1 - 1/arg**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return pi/2 - asin(1/arg)

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return acos(1/arg)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        sx2x = sqrt(x**2)/x
        return pi/2*(1 - sx2x) + sx2x*atan(sqrt(x**2 - 1))

    def _eval_rewrite_as_acot(self, x, **kwargs):
        sx2x = sqrt(x**2)/x
        return pi/2*(1 - sx2x) + sx2x*acot(1/sqrt(x**2 - 1))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return pi/2 - acsc(arg)


class acsc(InverseTrigonometricFunction):
    r"""
    The inverse cosecant function.

    Returns the arc cosecant of x (measured in radians).

    Explanation
    ===========

    ``acsc(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$` and for some instances when the
    result is a rational multiple of $\pi$ (see the ``eval`` class method).

    Examples
    ========

    >>> from sympy import acsc, oo
    >>> acsc(1)
    pi/2
    >>> acsc(-1)
    -pi/2
    >>> acsc(oo)
    0
    >>> acsc(-oo) == acsc(oo)
    True
    >>> acsc(0)
    zoo

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsc

    """

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.ComplexInfinity
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.One:
                return pi/2
            elif arg is S.NegativeOne:
                return -pi/2
        if arg in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
            return S.Zero

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_infinite:
            return S.Zero

        if arg.is_number:
            acsc_table = cls._acsc_table()
            if arg in acsc_table:
                return acsc_table[arg]

        if isinstance(arg, csc):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= 2*pi # restrict to [0,2*pi)
                if ang > pi: # restrict to (-pi,pi]
                    ang = pi - ang

                # restrict to [-pi/2,pi/2]
                if ang > pi/2:
                    ang = pi - ang
                if ang < -pi/2:
                    ang = -pi - ang

                return ang

        if isinstance(arg, sec): # asec(x) + acsc(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                return pi/2 - asec(arg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/(self.args[0]**2*sqrt(1 - 1/self.args[0]**2))
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return csc

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return pi/2 - S.ImaginaryUnit*log(2) + S.ImaginaryUnit*log(x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return S.ImaginaryUnit * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # Handling branch points
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # Handling points lying on branch cuts (-1, 1)
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return -pi - self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acsc
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = acsc(S.One + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.NegativeOne + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = acsc(S.NegativeOne - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.NegativeOne - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        # Handling points lying on branch cuts (-1, 1)
        if arg0.is_real and (1 - arg0**2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return pi - res
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return -pi - res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return -S.ImaginaryUnit*log(S.ImaginaryUnit/arg + sqrt(1 - 1/arg**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return asin(1/arg)

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return pi/2 - acos(1/arg)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        return sqrt(x**2)/x*(pi/2 - atan(sqrt(x**2 - 1)))

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return sqrt(arg**2)/arg*(pi/2 - acot(1/sqrt(arg**2 - 1)))

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return pi/2 - asec(arg)


class atan2(InverseTrigonometricFunction):
    r"""
    The function ``atan2(y, x)`` computes `\operatorname{atan}(y/x)` taking
    two arguments `y` and `x`.  Signs of both `y` and `x` are considered to
    determine the appropriate quadrant of `\operatorname{atan}(y/x)`.
    The range is `(-\pi, \pi]`. The complete definition reads as follows:

    .. math::

        \operatorname{atan2}(y, x) =
        \begin{cases}
          \arctan\left(\frac y x\right) & \qquad x > 0 \\
          \arctan\left(\frac y x\right) + \pi& \qquad y \ge 0, x < 0 \\
          \arctan\left(\frac y x\right) - \pi& \qquad y < 0, x < 0 \\
          +\frac{\pi}{2} & \qquad y > 0, x = 0 \\
          -\frac{\pi}{2} & \qquad y < 0, x = 0 \\
          \text{undefined} & \qquad y = 0, x = 0
        \end{cases}

    Attention: Note the role reversal of both arguments. The `y`-coordinate
    is the first argument and the `x`-coordinate the second.

    If either `x` or `y` is complex:

    .. math::

        \operatorname{atan2}(y, x) =
            -i\log\left(\frac{x + iy}{\sqrt{x^2 + y^2}}\right)

    Examples
    ========

    Going counter-clock wise around the origin we find the
    following angles:

    >>> from sympy import atan2
    >>> atan2(0, 1)
    0
    >>> atan2(1, 1)
    pi/4
    >>> atan2(1, 0)
    pi/2
    >>> atan2(1, -1)
    3*pi/4
    >>> atan2(0, -1)
    pi
    >>> atan2(-1, -1)
    -3*pi/4
    >>> atan2(-1, 0)
    -pi/2
    >>> atan2(-1, 1)
    -pi/4

    which are all correct. Compare this to the results of the ordinary
    `\operatorname{atan}` function for the point `(x, y) = (-1, 1)`

    >>> from sympy import atan, S
    >>> atan(S(1)/-1)
    -pi/4
    >>> atan2(1, -1)
    3*pi/4

    where only the `\operatorname{atan2}` function returns what we expect.
    We can differentiate the function with respect to both arguments:

    >>> from sympy import diff
    >>> from sympy.abc import x, y
    >>> diff(atan2(y, x), x)
    -y/(x**2 + y**2)

    >>> diff(atan2(y, x), y)
    x/(x**2 + y**2)

    We can express the `\operatorname{atan2}` function in terms of
    complex logarithms:

    >>> from sympy import log
    >>> atan2(y, x).rewrite(log)
    -I*log((x + I*y)/sqrt(x**2 + y**2))

    and in terms of `\operatorname(atan)`:

    >>> from sympy import atan
    >>> atan2(y, x).rewrite(atan)
    Piecewise((2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)), (pi, re(x) < 0), (0, Ne(x, 0)), (nan, True))

    but note that this form is undefined on the negative real axis.

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://en.wikipedia.org/wiki/Atan2
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan2

    """

    @classmethod
    def eval(cls, y, x):
        from sympy.functions.special.delta_functions import Heaviside
        if x is S.NegativeInfinity:
            if y.is_zero:
                # Special case y = 0 because we define Heaviside(0) = 1/2
                return pi
            return 2*pi*(Heaviside(re(y))) - pi
        elif x is S.Infinity:
            return S.Zero
        elif x.is_imaginary and y.is_imaginary and x.is_number and y.is_number:
            x = im(x)
            y = im(y)

        if x.is_extended_real and y.is_extended_real:
            if x.is_positive:
                return atan(y/x)
            elif x.is_negative:
                if y.is_negative:
                    return atan(y/x) - pi
                elif y.is_nonnegative:
                    return atan(y/x) + pi
            elif x.is_zero:
                if y.is_positive:
                    return pi/2
                elif y.is_negative:
                    return -pi/2
                elif y.is_zero:
                    return S.NaN
        if y.is_zero:
            if x.is_extended_nonzero:
                return pi*(S.One - Heaviside(x))
            if x.is_number:
                return Piecewise((pi, re(x) < 0),
                                 (0, Ne(x, 0)),
                                 (S.NaN, True))
        if x.is_number and y.is_number:
            return -S.ImaginaryUnit*log(
                (x + S.ImaginaryUnit*y)/sqrt(x**2 + y**2))

    def _eval_rewrite_as_log(self, y, x, **kwargs):
        return -S.ImaginaryUnit*log((x + S.ImaginaryUnit*y)/sqrt(x**2 + y**2))

    def _eval_rewrite_as_atan(self, y, x, **kwargs):
        return Piecewise((2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)),
                         (pi, re(x) < 0),
                         (0, Ne(x, 0)),
                         (S.NaN, True))

    def _eval_rewrite_as_arg(self, y, x, **kwargs):
        if x.is_extended_real and y.is_extended_real:
            return arg_f(x + y*S.ImaginaryUnit)
        n = x + S.ImaginaryUnit*y
        d = x**2 + y**2
        return arg_f(n/sqrt(d)) - S.ImaginaryUnit*log(abs(n)/sqrt(abs(d)))

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def fdiff(self, argindex):
        y, x = self.args
        if argindex == 1:
            # Diff wrt y
            return x/(x**2 + y**2)
        elif argindex == 2:
            # Diff wrt x
            return -y/(x**2 + y**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        y, x = self.args
        if x.is_extended_real and y.is_extended_real:
            return super()._eval_evalf(prec)