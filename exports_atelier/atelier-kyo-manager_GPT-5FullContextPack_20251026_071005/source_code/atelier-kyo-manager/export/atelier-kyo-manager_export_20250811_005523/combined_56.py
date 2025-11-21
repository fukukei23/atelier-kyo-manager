
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\fu.py ===
from collections import defaultdict

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
    cosh, sinh, tanh, coth, sech, csch, HyperbolicFunction)
from sympy.functions.elementary.trigonometric import (
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug

from sympy import SYMPY_DEBUG


# ================== Fu-like tools ===========================


def TR0(rv):
    """Simplification of rational polynomials, trying to simplify
    the expression, e.g. combine things like 3*x + 2*x, etc....
    """
    # although it would be nice to use cancel, it doesn't work
    # with noncommutatives
    return rv.normal().factor().expand()


def TR1(rv):
    """Replace sec, csc with 1/cos, 1/sin

    Examples
    ========

    >>> from sympy.simplify.fu import TR1, sec, csc
    >>> from sympy.abc import x
    >>> TR1(2*csc(x) + sec(x))
    1/cos(x) + 2/sin(x)
    """

    def f(rv):
        if isinstance(rv, sec):
            a = rv.args[0]
            return S.One/cos(a)
        elif isinstance(rv, csc):
            a = rv.args[0]
            return S.One/sin(a)
        return rv

    return bottom_up(rv, f)


def TR2(rv):
    """Replace tan and cot with sin/cos and cos/sin

    Examples
    ========

    >>> from sympy.simplify.fu import TR2
    >>> from sympy.abc import x
    >>> from sympy import tan, cot, sin, cos
    >>> TR2(tan(x))
    sin(x)/cos(x)
    >>> TR2(cot(x))
    cos(x)/sin(x)
    >>> TR2(tan(tan(x) - sin(x)/cos(x)))
    0

    """

    def f(rv):
        if isinstance(rv, tan):
            a = rv.args[0]
            return sin(a)/cos(a)
        elif isinstance(rv, cot):
            a = rv.args[0]
            return cos(a)/sin(a)
        return rv

    return bottom_up(rv, f)


def TR2i(rv, half=False):
    """Converts ratios involving sin and cos as follows::
        sin(x)/cos(x) -> tan(x)
        sin(x)/(cos(x) + 1) -> tan(x/2) if half=True

    Examples
    ========

    >>> from sympy.simplify.fu import TR2i
    >>> from sympy.abc import x, a
    >>> from sympy import sin, cos
    >>> TR2i(sin(x)/cos(x))
    tan(x)

    Powers of the numerator and denominator are also recognized

    >>> TR2i(sin(x)**2/(cos(x) + 1)**2, half=True)
    tan(x/2)**2

    The transformation does not take place unless assumptions allow
    (i.e. the base must be positive or the exponent must be an integer
    for both numerator and denominator)

    >>> TR2i(sin(x)**a/(cos(x) + 1)**a)
    sin(x)**a/(cos(x) + 1)**a

    """

    def f(rv):
        if not rv.is_Mul:
            return rv

        n, d = rv.as_numer_denom()
        if n.is_Atom or d.is_Atom:
            return rv

        def ok(k, e):
            # initial filtering of factors
            return (
                (e.is_integer or k.is_positive) and (
                k.func in (sin, cos) or (half and
                k.is_Add and
                len(k.args) >= 2 and
                any(any(isinstance(ai, cos) or ai.is_Pow and ai.base is cos
                for ai in Mul.make_args(a)) for a in k.args))))

        n = n.as_powers_dict()
        ndone = [(k, n.pop(k)) for k in list(n.keys()) if not ok(k, n[k])]
        if not n:
            return rv

        d = d.as_powers_dict()
        ddone = [(k, d.pop(k)) for k in list(d.keys()) if not ok(k, d[k])]
        if not d:
            return rv

        # factoring if necessary

        def factorize(d, ddone):
            newk = []
            for k in d:
                if k.is_Add and len(k.args) > 1:
                    knew = factor(k) if half else factor_terms(k)
                    if knew != k:
                        newk.append((k, knew))
            if newk:
                for i, (k, knew) in enumerate(newk):
                    del d[k]
                    newk[i] = knew
                newk = Mul(*newk).as_powers_dict()
                for k in newk:
                    v = d[k] + newk[k]
                    if ok(k, v):
                        d[k] = v
                    else:
                        ddone.append((k, v))
                del newk
        factorize(n, ndone)
        factorize(d, ddone)

        # joining
        t = []
        for k in n:
            if isinstance(k, sin):
                a = cos(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0])**n[k])
                    n[k] = d[a] = None
                elif half:
                    a1 = 1 + a
                    if a1 in d and d[a1] == n[k]:
                        t.append((tan(k.args[0]/2))**n[k])
                        n[k] = d[a1] = None
            elif isinstance(k, cos):
                a = sin(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0])**-n[k])
                    n[k] = d[a] = None
            elif half and k.is_Add and k.args[0] is S.One and \
                    isinstance(k.args[1], cos):
                a = sin(k.args[1].args[0], evaluate=False)
                if a in d and d[a] == n[k] and (d[a].is_integer or \
                        a.is_positive):
                    t.append(tan(a.args[0]/2)**-n[k])
                    n[k] = d[a] = None

        if t:
            rv = Mul(*(t + [b**e for b, e in n.items() if e]))/\
                Mul(*[b**e for b, e in d.items() if e])
            rv *= Mul(*[b**e for b, e in ndone])/Mul(*[b**e for b, e in ddone])

        return rv

    return bottom_up(rv, f)


def TR3(rv):
    """Induced formula: example sin(-a) = -sin(a)

    Examples
    ========

    >>> from sympy.simplify.fu import TR3
    >>> from sympy.abc import x, y
    >>> from sympy import pi
    >>> from sympy import cos
    >>> TR3(cos(y - x*(y - x)))
    cos(x*(x - y) + y)
    >>> cos(pi/2 + x)
    -sin(x)
    >>> cos(30*pi/2 + x)
    -cos(x)

    """
    from sympy.simplify.simplify import signsimp

    # Negative argument (already automatic for funcs like sin(-x) -> -sin(x)
    # but more complicated expressions can use it, too). Also, trig angles
    # between pi/4 and pi/2 are not reduced to an angle between 0 and pi/4.
    # The following are automatically handled:
    #   Argument of type: pi/2 +/- angle
    #   Argument of type: pi +/- angle
    #   Argument of type : 2k*pi +/- angle

    def f(rv):
        if not isinstance(rv, TrigonometricFunction):
            return rv
        rv = rv.func(signsimp(rv.args[0]))
        if not isinstance(rv, TrigonometricFunction):
            return rv
        if (rv.args[0] - S.Pi/4).is_positive is (S.Pi/2 - rv.args[0]).is_positive is True:
            fmap = {cos: sin, sin: cos, tan: cot, cot: tan, sec: csc, csc: sec}
            rv = fmap[type(rv)](S.Pi/2 - rv.args[0])
        return rv

    # touch numbers iside of trig functions to let them automatically update
    rv = rv.replace(
        lambda x: isinstance(x, TrigonometricFunction),
        lambda x: x.replace(
            lambda n: n.is_number and n.is_Mul,
            lambda n: n.func(*n.args)))

    return bottom_up(rv, f)


def TR4(rv):
    """Identify values of special angles.

        a=  0   pi/6        pi/4        pi/3        pi/2
    ----------------------------------------------------
    sin(a)  0   1/2         sqrt(2)/2   sqrt(3)/2   1
    cos(a)  1   sqrt(3)/2   sqrt(2)/2   1/2         0
    tan(a)  0   sqt(3)/3    1           sqrt(3)     --

    Examples
    ========

    >>> from sympy import pi
    >>> from sympy import cos, sin, tan, cot
    >>> for s in (0, pi/6, pi/4, pi/3, pi/2):
    ...    print('%s %s %s %s' % (cos(s), sin(s), tan(s), cot(s)))
    ...
    1 0 0 zoo
    sqrt(3)/2 1/2 sqrt(3)/3 sqrt(3)
    sqrt(2)/2 sqrt(2)/2 1 1
    1/2 sqrt(3)/2 sqrt(3) sqrt(3)/3
    0 1 zoo 0
    """
    # special values at 0, pi/6, pi/4, pi/3, pi/2 already handled
    return rv.replace(
        lambda x:
            isinstance(x, TrigonometricFunction) and
            (r:=x.args[0]/pi).is_Rational and r.q in (1, 2, 3, 4, 6),
        lambda x:
            x.func(x.args[0].func(*x.args[0].args)))


def _TR56(rv, f, g, h, max, pow):
    """Helper for TR5 and TR6 to replace f**2 with h(g**2)

    Options
    =======

    max :   controls size of exponent that can appear on f
            e.g. if max=4 then f**4 will be changed to h(g**2)**2.
    pow :   controls whether the exponent must be a perfect power of 2
            e.g. if pow=True (and max >= 6) then f**6 will not be changed
            but f**8 will be changed to h(g**2)**4

    >>> from sympy.simplify.fu import _TR56 as T
    >>> from sympy.abc import x
    >>> from sympy import sin, cos
    >>> h = lambda x: 1 - x
    >>> T(sin(x)**3, sin, cos, h, 4, False)
    (1 - cos(x)**2)*sin(x)
    >>> T(sin(x)**6, sin, cos, h, 6, False)
    (1 - cos(x)**2)**3
    >>> T(sin(x)**6, sin, cos, h, 6, True)
    sin(x)**6
    >>> T(sin(x)**8, sin, cos, h, 10, True)
    (1 - cos(x)**2)**4
    """

    def _f(rv):
        # I'm not sure if this transformation should target all even powers
        # or only those expressible as powers of 2. Also, should it only
        # make the changes in powers that appear in sums -- making an isolated
        # change is not going to allow a simplification as far as I can tell.
        if not (rv.is_Pow and rv.base.func == f):
            return rv
        if not rv.exp.is_real:
            return rv

        if (rv.exp < 0) == True:
            return rv
        if (rv.exp > max) == True:
            return rv
        if rv.exp == 1:
            return rv
        if rv.exp == 2:
            return h(g(rv.base.args[0])**2)
        else:
            if rv.exp % 2 == 1:
                e = rv.exp//2
                return f(rv.base.args[0])*h(g(rv.base.args[0])**2)**e
            elif rv.exp == 4:
                e = 2
            elif not pow:
                if rv.exp % 2:
                    return rv
                e = rv.exp//2
            else:
                p = perfect_power(rv.exp)
                if not p:
                    return rv
                e = rv.exp//2
            return h(g(rv.base.args[0])**2)**e

    return bottom_up(rv, _f)


def TR5(rv, max=4, pow=False):
    """Replacement of sin**2 with 1 - cos(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR5
    >>> from sympy.abc import x
    >>> from sympy import sin
    >>> TR5(sin(x)**2)
    1 - cos(x)**2
    >>> TR5(sin(x)**-2)  # unchanged
    sin(x)**(-2)
    >>> TR5(sin(x)**4)
    (1 - cos(x)**2)**2
    """
    return _TR56(rv, sin, cos, lambda x: 1 - x, max=max, pow=pow)


def TR6(rv, max=4, pow=False):
    """Replacement of cos**2 with 1 - sin(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR6
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR6(cos(x)**2)
    1 - sin(x)**2
    >>> TR6(cos(x)**-2)  #unchanged
    cos(x)**(-2)
    >>> TR6(cos(x)**4)
    (1 - sin(x)**2)**2
    """
    return _TR56(rv, cos, sin, lambda x: 1 - x, max=max, pow=pow)


def TR7(rv):
    """Lowering the degree of cos(x)**2.

    Examples
    ========

    >>> from sympy.simplify.fu import TR7
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR7(cos(x)**2)
    cos(2*x)/2 + 1/2
    >>> TR7(cos(x)**2 + 1)
    cos(2*x)/2 + 3/2

    """

    def f(rv):
        if not (rv.is_Pow and rv.base.func == cos and rv.exp == 2):
            return rv
        return (1 + cos(2*rv.base.args[0]))/2

    return bottom_up(rv, f)


def TR8(rv, first=True):
    """Converting products of ``cos`` and/or ``sin`` to a sum or
    difference of ``cos`` and or ``sin`` terms.

    Examples
    ========

    >>> from sympy.simplify.fu import TR8
    >>> from sympy import cos, sin
    >>> TR8(cos(2)*cos(3))
    cos(5)/2 + cos(1)/2
    >>> TR8(cos(2)*sin(3))
    sin(5)/2 + sin(1)/2
    >>> TR8(sin(2)*sin(3))
    -cos(5)/2 + cos(1)/2
    """

    def f(rv):
        if not (
            rv.is_Mul or
            rv.is_Pow and
            rv.base.func in (cos, sin) and
            (rv.exp.is_integer or rv.base.is_positive)):
            return rv

        if first:
            n, d = [expand_mul(i) for i in rv.as_numer_denom()]
            newn = TR8(n, first=False)
            newd = TR8(d, first=False)
            if newn != n or newd != d:
                rv = gcd_terms(newn/newd)
                if rv.is_Mul and rv.args[0].is_Rational and \
                        len(rv.args) == 2 and rv.args[1].is_Add:
                    rv = Mul(*rv.as_coeff_Mul())
            return rv

        args = {cos: [], sin: [], None: []}
        for a in Mul.make_args(rv):
            if a.func in (cos, sin):
                args[type(a)].append(a.args[0])
            elif (a.is_Pow and a.exp.is_Integer and a.exp > 0 and \
                    a.base.func in (cos, sin)):
                # XXX this is ok but pathological expression could be handled
                # more efficiently as in TRmorrie
                args[type(a.base)].extend([a.base.args[0]]*a.exp)
            else:
                args[None].append(a)
        c = args[cos]
        s = args[sin]
        if not (c and s or len(c) > 1 or len(s) > 1):
            return rv

        args = args[None]
        n = min(len(c), len(s))
        for i in range(n):
            a1 = s.pop()
            a2 = c.pop()
            args.append((sin(a1 + a2) + sin(a1 - a2))/2)
        while len(c) > 1:
            a1 = c.pop()
            a2 = c.pop()
            args.append((cos(a1 + a2) + cos(a1 - a2))/2)
        if c:
            args.append(cos(c.pop()))
        while len(s) > 1:
            a1 = s.pop()
            a2 = s.pop()
            args.append((-cos(a1 + a2) + cos(a1 - a2))/2)
        if s:
            args.append(sin(s.pop()))
        return TR8(expand_mul(Mul(*args)))

    return bottom_up(rv, f)


def TR9(rv):
    """Sum of ``cos`` or ``sin`` terms as a product of ``cos`` or ``sin``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR9
    >>> from sympy import cos, sin
    >>> TR9(cos(1) + cos(2))
    2*cos(1/2)*cos(3/2)
    >>> TR9(cos(1) + 2*sin(1) + 2*sin(2))
    cos(1) + 4*sin(3/2)*cos(1/2)

    If no change is made by TR9, no re-arrangement of the
    expression will be made. For example, though factoring
    of common term is attempted, if the factored expression
    was not changed, the original expression will be returned:

    >>> TR9(cos(3) + cos(3)*cos(2))
    cos(3) + cos(2)*cos(3)

    """

    def f(rv):
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            # cos(a)+/-cos(b) can be combined into a product of cosines and
            # sin(a)+/-sin(b) can be combined into a product of cosine and
            # sine.
            #
            # If there are more than two args, the pairs which "work" will
            # have a gcd extractable and the remaining two terms will have
            # the above structure -- all pairs must be checked to find the
            # ones that work. args that don't have a common set of symbols
            # are skipped since this doesn't lead to a simpler formula and
            # also has the arbitrariness of combining, for example, the x
            # and y term instead of the y and z term in something like
            # cos(x) + cos(y) + cos(z).

            if not rv.is_Add:
                return rv

            args = list(ordered(rv.args))
            if len(args) != 2:
                hit = False
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        new = do(was)
                        if new != was:
                            args[i] = new  # update in place
                            args[j] = None
                            hit = True
                            break  # go to next i
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    if rv.is_Add:
                        rv = do(rv)

                return rv

            # two-arg Add
            split = trig_split(*args)
            if not split:
                return rv
            gcd, n1, n2, a, b, iscos = split

            # application of rule if possible
            if iscos:
                if n1 == n2:
                    return gcd*n1*2*cos((a + b)/2)*cos((a - b)/2)
                if n1 < 0:
                    a, b = b, a
                return -2*gcd*sin((a + b)/2)*sin((a - b)/2)
            else:
                if n1 == n2:
                    return gcd*n1*2*sin((a + b)/2)*cos((a - b)/2)
                if n1 < 0:
                    a, b = b, a
                return 2*gcd*cos((a + b)/2)*sin((a - b)/2)

        return process_common_addends(rv, do)  # DON'T sift by free symbols

    return bottom_up(rv, f)


def TR10(rv, first=True):
    """Separate sums in ``cos`` and ``sin``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR10
    >>> from sympy.abc import a, b, c
    >>> from sympy import cos, sin
    >>> TR10(cos(a + b))
    -sin(a)*sin(b) + cos(a)*cos(b)
    >>> TR10(sin(a + b))
    sin(a)*cos(b) + sin(b)*cos(a)
    >>> TR10(sin(a + b + c))
    (-sin(a)*sin(b) + cos(a)*cos(b))*sin(c) + \
    (sin(a)*cos(b) + sin(b)*cos(a))*cos(c)
    """

    def f(rv):
        if rv.func not in (cos, sin):
            return rv

        f = rv.func
        arg = rv.args[0]
        if arg.is_Add:
            if first:
                args = list(ordered(arg.args))
            else:
                args = list(arg.args)
            a = args.pop()
            b = Add._from_args(args)
            if b.is_Add:
                if f == sin:
                    return sin(a)*TR10(cos(b), first=False) + \
                        cos(a)*TR10(sin(b), first=False)
                else:
                    return cos(a)*TR10(cos(b), first=False) - \
                        sin(a)*TR10(sin(b), first=False)
            else:
                if f == sin:
                    return sin(a)*cos(b) + cos(a)*sin(b)
                else:
                    return cos(a)*cos(b) - sin(a)*sin(b)
        return rv

    return bottom_up(rv, f)


def TR10i(rv):
    """Sum of products to function of sum.

    Examples
    ========

    >>> from sympy.simplify.fu import TR10i
    >>> from sympy import cos, sin, sqrt
    >>> from sympy.abc import x

    >>> TR10i(cos(1)*cos(3) + sin(1)*sin(3))
    cos(2)
    >>> TR10i(cos(1)*sin(3) + sin(1)*cos(3) + cos(3))
    cos(3) + sin(4)
    >>> TR10i(sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x)
    2*sqrt(2)*x*sin(x + pi/6)

    """
    def f(rv):
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            # args which can be expressed as A*(cos(a)*cos(b)+/-sin(a)*sin(b))
            # or B*(cos(a)*sin(b)+/-cos(b)*sin(a)) can be combined into
            # A*f(a+/-b) where f is either sin or cos.
            #
            # If there are more than two args, the pairs which "work" will have
            # a gcd extractable and the remaining two terms will have the above
            # structure -- all pairs must be checked to find the ones that
            # work.

            if not rv.is_Add:
                return rv

            args = list(ordered(rv.args))
            if len(args) != 2:
                hit = False
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        new = do(was)
                        if new != was:
                            args[i] = new  # update in place
                            args[j] = None
                            hit = True
                            break  # go to next i
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    if rv.is_Add:
                        rv = do(rv)

                return rv

            # two-arg Add
            split = trig_split(*args, two=True)
            if not split:
                return rv
            gcd, n1, n2, a, b, same = split

            # identify and get c1 to be cos then apply rule if possible
            if same:  # coscos, sinsin
                gcd = n1*gcd
                if n1 == n2:
                    return gcd*cos(a - b)
                return gcd*cos(a + b)
            else:  #cossin, cossin
                gcd = n1*gcd
                if n1 == n2:
                    return gcd*sin(a + b)
                return gcd*sin(b - a)

        rv = process_common_addends(
            rv, do, lambda x: tuple(ordered(x.free_symbols)))

        # need to check for inducible pairs in ratio of sqrt(3):1 that
        # appeared in different lists when sorting by coefficient
        while rv.is_Add:
            byrad = defaultdict(list)
            for a in rv.args:
                hit = 0
                if a.is_Mul:
                    for ai in a.args:
                        if ai.is_Pow and ai.exp is S.Half and \
                                ai.base.is_Integer:
                            byrad[ai].append(a)
                            hit = 1
                            break
                if not hit:
                    byrad[S.One].append(a)

            # no need to check all pairs -- just check for the onees
            # that have the right ratio
            args = []
            for a in byrad:
                for b in [_ROOT3()*a, _invROOT3()]:
                    if b in byrad:
                        for i in range(len(byrad[a])):
                            if byrad[a][i] is None:
                                continue
                            for j in range(len(byrad[b])):
                                if byrad[b][j] is None:
                                    continue
                                was = Add(byrad[a][i] + byrad[b][j])
                                new = do(was)
                                if new != was:
                                    args.append(new)
                                    byrad[a][i] = None
                                    byrad[b][j] = None
                                    break
            if args:
                rv = Add(*(args + [Add(*[_f for _f in v if _f])
                    for v in byrad.values()]))
            else:
                rv = do(rv)  # final pass to resolve any new inducible pairs
                break

        return rv

    return bottom_up(rv, f)


def TR11(rv, base=None):
    """Function of double angle to product. The ``base`` argument can be used
    to indicate what is the un-doubled argument, e.g. if 3*pi/7 is the base
    then cosine and sine functions with argument 6*pi/7 will be replaced.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11
    >>> from sympy import cos, sin, pi
    >>> from sympy.abc import x
    >>> TR11(sin(2*x))
    2*sin(x)*cos(x)
    >>> TR11(cos(2*x))
    -sin(x)**2 + cos(x)**2
    >>> TR11(sin(4*x))
    4*(-sin(x)**2 + cos(x)**2)*sin(x)*cos(x)
    >>> TR11(sin(4*x/3))
    4*(-sin(x/3)**2 + cos(x/3)**2)*sin(x/3)*cos(x/3)

    If the arguments are simply integers, no change is made
    unless a base is provided:

    >>> TR11(cos(2))
    cos(2)
    >>> TR11(cos(4), 2)
    -sin(2)**2 + cos(2)**2

    There is a subtle issue here in that autosimplification will convert
    some higher angles to lower angles

    >>> cos(6*pi/7) + cos(3*pi/7)
    -cos(pi/7) + cos(3*pi/7)

    The 6*pi/7 angle is now pi/7 but can be targeted with TR11 by supplying
    the 3*pi/7 base:

    >>> TR11(_, 3*pi/7)
    -sin(3*pi/7)**2 + cos(3*pi/7)**2 + cos(3*pi/7)

    """

    def f(rv):
        if rv.func not in (cos, sin):
            return rv

        if base:
            f = rv.func
            t = f(base*2)
            co = S.One
            if t.is_Mul:
                co, t = t.as_coeff_Mul()
            if t.func not in (cos, sin):
                return rv
            if rv.args[0] == t.args[0]:
                c = cos(base)
                s = sin(base)
                if f is cos:
                    return (c**2 - s**2)/co
                else:
                    return 2*c*s/co
            return rv

        elif not rv.args[0].is_Number:
            # make a change if the leading coefficient's numerator is
            # divisible by 2
            c, m = rv.args[0].as_coeff_Mul(rational=True)
            if c.p % 2 == 0:
                arg = c.p//2*m/c.q
                c = TR11(cos(arg))
                s = TR11(sin(arg))
                if rv.func == sin:
                    rv = 2*s*c
                else:
                    rv = c**2 - s**2
        return rv

    return bottom_up(rv, f)


def _TR11(rv):
    """
    Helper for TR11 to find half-arguments for sin in factors of
    num/den that appear in cos or sin factors in the den/num.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11, _TR11
    >>> from sympy import cos, sin
    >>> from sympy.abc import x
    >>> TR11(sin(x/3)/(cos(x/6)))
    sin(x/3)/cos(x/6)
    >>> _TR11(sin(x/3)/(cos(x/6)))
    2*sin(x/6)
    >>> TR11(sin(x/6)/(sin(x/3)))
    sin(x/6)/sin(x/3)
    >>> _TR11(sin(x/6)/(sin(x/3)))
    1/(2*cos(x/6))

    """
    def f(rv):
        if not isinstance(rv, Expr):
            return rv

        def sincos_args(flat):
            # find arguments of sin and cos that
            # appears as bases in args of flat
            # and have Integer exponents
            args = defaultdict(set)
            for fi in Mul.make_args(flat):
                b, e = fi.as_base_exp()
                if e.is_Integer and e > 0:
                    if b.func in (cos, sin):
                        args[type(b)].add(b.args[0])
            return args
        num_args, den_args = map(sincos_args, rv.as_numer_denom())
        def handle_match(rv, num_args, den_args):
            # for arg in sin args of num_args, look for arg/2
            # in den_args and pass this half-angle to TR11
            # for handling in rv
            for narg in num_args[sin]:
                half = narg/2
                if half in den_args[cos]:
                    func = cos
                elif half in den_args[sin]:
                    func = sin
                else:
                    continue
                rv = TR11(rv, half)
                den_args[func].remove(half)
            return rv
        # sin in num, sin or cos in den
        rv = handle_match(rv, num_args, den_args)
        # sin in den, sin or cos in num
        rv = handle_match(rv, den_args, num_args)
        return rv

    return bottom_up(rv, f)


def TR12(rv, first=True):
    """Separate sums in ``tan``.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import tan
    >>> from sympy.simplify.fu import TR12
    >>> TR12(tan(x + y))
    (tan(x) + tan(y))/(-tan(x)*tan(y) + 1)
    """

    def f(rv):
        if not rv.func == tan:
            return rv

        arg = rv.args[0]
        if arg.is_Add:
            if first:
                args = list(ordered(arg.args))
            else:
                args = list(arg.args)
            a = args.pop()
            b = Add._from_args(args)
            if b.is_Add:
                tb = TR12(tan(b), first=False)
            else:
                tb = tan(b)
            return (tan(a) + tb)/(1 - tan(a)*tb)
        return rv

    return bottom_up(rv, f)


def TR12i(rv):
    """Combine tan arguments as
    (tan(y) + tan(x))/(tan(x)*tan(y) - 1) -> -tan(x + y).

    Examples
    ========

    >>> from sympy.simplify.fu import TR12i
    >>> from sympy import tan
    >>> from sympy.abc import a, b, c
    >>> ta, tb, tc = [tan(i) for i in (a, b, c)]
    >>> TR12i((ta + tb)/(-ta*tb + 1))
    tan(a + b)
    >>> TR12i((ta + tb)/(ta*tb - 1))
    -tan(a + b)
    >>> TR12i((-ta - tb)/(ta*tb - 1))
    tan(a + b)
    >>> eq = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))
    >>> TR12i(eq.expand())
    -3*tan(a + b)*tan(a + c)/(2*(tan(a) + tan(b) - 1))
    """
    def f(rv):
        if not (rv.is_Add or rv.is_Mul or rv.is_Pow):
            return rv

        n, d = rv.as_numer_denom()
        if not d.args or not n.args:
            return rv

        dok = {}

        def ok(di):
            m = as_f_sign_1(di)
            if m:
                g, f, s = m
                if s is S.NegativeOne and f.is_Mul and len(f.args) == 2 and \
                        all(isinstance(fi, tan) for fi in f.args):
                    return g, f

        d_args = list(Mul.make_args(d))
        for i, di in enumerate(d_args):
            m = ok(di)
            if m:
                g, t = m
                s = Add(*[_.args[0] for _ in t.args])
                dok[s] = S.One
                d_args[i] = g
                continue
            if di.is_Add:
                di = factor(di)
                if di.is_Mul:
                    d_args.extend(di.args)
                    d_args[i] = S.One
            elif di.is_Pow and (di.exp.is_integer or di.base.is_positive):
                m = ok(di.base)
                if m:
                    g, t = m
                    s = Add(*[_.args[0] for _ in t.args])
                    dok[s] = di.exp
                    d_args[i] = g**di.exp
                else:
                    di = factor(di)
                    if di.is_Mul:
                        d_args.extend(di.args)
                        d_args[i] = S.One
        if not dok:
            return rv

        def ok(ni):
            if ni.is_Add and len(ni.args) == 2:
                a, b = ni.args
                if isinstance(a, tan) and isinstance(b, tan):
                    return a, b
        n_args = list(Mul.make_args(factor_terms(n)))
        hit = False
        for i, ni in enumerate(n_args):
            m = ok(ni)
            if not m:
                m = ok(-ni)
                if m:
                    n_args[i] = S.NegativeOne
                else:
                    if ni.is_Add:
                        ni = factor(ni)
                        if ni.is_Mul:
                            n_args.extend(ni.args)
                            n_args[i] = S.One
                        continue
                    elif ni.is_Pow and (
                            ni.exp.is_integer or ni.base.is_positive):
                        m = ok(ni.base)
                        if m:
                            n_args[i] = S.One
                        else:
                            ni = factor(ni)
                            if ni.is_Mul:
                                n_args.extend(ni.args)
                                n_args[i] = S.One
                            continue
                    else:
                        continue
            else:
                n_args[i] = S.One
            hit = True
            s = Add(*[_.args[0] for _ in m])
            ed = dok[s]
            newed = ed.extract_additively(S.One)
            if newed is not None:
                if newed:
                    dok[s] = newed
                else:
                    dok.pop(s)
            n_args[i] *= -tan(s)

        if hit:
            rv = Mul(*n_args)/Mul(*d_args)/Mul(*[(Add(*[
                tan(a) for a in i.args]) - 1)**e for i, e in dok.items()])

        return rv

    return bottom_up(rv, f)


def TR13(rv):
    """Change products of ``tan`` or ``cot``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR13
    >>> from sympy import tan, cot
    >>> TR13(tan(3)*tan(2))
    -tan(2)/tan(5) - tan(3)/tan(5) + 1
    >>> TR13(cot(3)*cot(2))
    cot(2)*cot(5) + 1 + cot(3)*cot(5)
    """

    def f(rv):
        if not rv.is_Mul:
            return rv

        # XXX handle products of powers? or let power-reducing handle it?
        args = {tan: [], cot: [], None: []}
        for a in Mul.make_args(rv):
            if a.func in (tan, cot):
                args[type(a)].append(a.args[0])
            else:
                args[None].append(a)
        t = args[tan]
        c = args[cot]
        if len(t) < 2 and len(c) < 2:
            return rv
        args = args[None]
        while len(t) > 1:
            t1 = t.pop()
            t2 = t.pop()
            args.append(1 - (tan(t1)/tan(t1 + t2) + tan(t2)/tan(t1 + t2)))
        if t:
            args.append(tan(t.pop()))
        while len(c) > 1:
            t1 = c.pop()
            t2 = c.pop()
            args.append(1 + cot(t1)*cot(t1 + t2) + cot(t2)*cot(t1 + t2))
        if c:
            args.append(cot(c.pop()))
        return Mul(*args)

    return bottom_up(rv, f)


def TRmorrie(rv):
    """Returns cos(x)*cos(2*x)*...*cos(2**(k-1)*x) -> sin(2**k*x)/(2**k*sin(x))

    Examples
    ========

    >>> from sympy.simplify.fu import TRmorrie, TR8, TR3
    >>> from sympy.abc import x
    >>> from sympy import Mul, cos, pi
    >>> TRmorrie(cos(x)*cos(2*x))
    sin(4*x)/(4*sin(x))
    >>> TRmorrie(7*Mul(*[cos(x) for x in range(10)]))
    7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))

    Sometimes autosimplification will cause a power to be
    not recognized. e.g. in the following, cos(4*pi/7) automatically
    simplifies to -cos(3*pi/7) so only 2 of the 3 terms are
    recognized:

    >>> TRmorrie(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7))
    -sin(3*pi/7)*cos(3*pi/7)/(4*sin(pi/7))

    A touch by TR8 resolves the expression to a Rational

    >>> TR8(_)
    -1/8

    In this case, if eq is unsimplified, the answer is obtained
    directly:

    >>> eq = cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9)
    >>> TRmorrie(eq)
    1/16

    But if angles are made canonical with TR3 then the answer
    is not simplified without further work:

    >>> TR3(eq)
    sin(pi/18)*cos(pi/9)*cos(2*pi/9)/2
    >>> TRmorrie(_)
    sin(pi/18)*sin(4*pi/9)/(8*sin(pi/9))
    >>> TR8(_)
    cos(7*pi/18)/(16*sin(pi/9))
    >>> TR3(_)
    1/16

    The original expression would have resolve to 1/16 directly with TR8,
    however:

    >>> TR8(eq)
    1/16

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morrie%27s_law

    """

    def f(rv, first=True):
        if not rv.is_Mul:
            return rv
        if first:
            n, d = rv.as_numer_denom()
            return f(n, 0)/f(d, 0)

        args = defaultdict(list)
        coss = {}
        other = []
        for c in rv.args:
            b, e = c.as_base_exp()
            if e.is_Integer and isinstance(b, cos):
                co, a = b.args[0].as_coeff_Mul()
                args[a].append(co)
                coss[b] = e
            else:
                other.append(c)

        new = []
        for a in args:
            c = args[a]
            c.sort()
            while c:
                k = 0
                cc = ci = c[0]
                while cc in c:
                    k += 1
                    cc *= 2
                if k > 1:
                    newarg = sin(2**k*ci*a)/2**k/sin(ci*a)
                    # see how many times this can be taken
                    take = None
                    ccs = []
                    for i in range(k):
                        cc /= 2
                        key = cos(a*cc, evaluate=False)
                        ccs.append(cc)
                        take = min(coss[key], take or coss[key])
                    # update exponent counts
                    for i in range(k):
                        cc = ccs.pop()
                        key = cos(a*cc, evaluate=False)
                        coss[key] -= take
                        if not coss[key]:
                            c.remove(cc)
                    new.append(newarg**take)
                else:
                    b = cos(c.pop(0)*a)
                    other.append(b**coss[b])

        if new:
            rv = Mul(*(new + other + [
                cos(k*a, evaluate=False) for a in args for k in args[a]]))

        return rv

    return bottom_up(rv, f)


def TR14(rv, first=True):
    """Convert factored powers of sin and cos identities into simpler
    expressions.

    Examples
    ========

    >>> from sympy.simplify.fu import TR14
    >>> from sympy.abc import x, y
    >>> from sympy import cos, sin
    >>> TR14((cos(x) - 1)*(cos(x) + 1))
    -sin(x)**2
    >>> TR14((sin(x) - 1)*(sin(x) + 1))
    -cos(x)**2
    >>> p1 = (cos(x) + 1)*(cos(x) - 1)
    >>> p2 = (cos(y) - 1)*2*(cos(y) + 1)
    >>> p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))
    >>> TR14(p1*p2*p3*(x - 1))
    -18*(x - 1)*sin(x)**2*sin(y)**4

    """

    def f(rv):
        if not rv.is_Mul:
            return rv

        if first:
            # sort them by location in numerator and denominator
            # so the code below can just deal with positive exponents
            n, d = rv.as_numer_denom()
            if d is not S.One:
                newn = TR14(n, first=False)
                newd = TR14(d, first=False)
                if newn != n or newd != d:
                    rv = newn/newd
                return rv

        other = []
        process = []
        for a in rv.args:
            if a.is_Pow:
                b, e = a.as_base_exp()
                if not (e.is_integer or b.is_positive):
                    other.append(a)
                    continue
                a = b
            else:
                e = S.One
            m = as_f_sign_1(a)
            if not m or m[1].func not in (cos, sin):
                if e is S.One:
                    other.append(a)
                else:
                    other.append(a**e)
                continue
            g, f, si = m
            process.append((g, e.is_Number, e, f, si, a))

        # sort them to get like terms next to each other
        process = list(ordered(process))

        # keep track of whether there was any change
        nother = len(other)

        # access keys
        keys = (g, t, e, f, si, a) = list(range(6))

        while process:
            A = process.pop(0)
            if process:
                B = process[0]

                if A[e].is_Number and B[e].is_Number:
                    # both exponents are numbers
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = min(A[e], B[e])

                            # reinsert any remainder
                            # the B will likely sort after A so check it first
                            if B[e] != take:
                                rem = [B[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)
                            elif A[e] != take:
                                rem = [A[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)

                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g]*B[g]*t(A[f].args[0])**2)**take)
                            continue

                elif A[e] == B[e]:
                    # both exponents are equal symbols
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = A[e]
                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g]*B[g]*t(A[f].args[0])**2)**take)
                            continue

            # either we are done or neither condition above applied
            other.append(A[a]**A[e])

        if len(other) != nother:
            rv = Mul(*other)

        return rv

    return bottom_up(rv, f)


def TR15(rv, max=4, pow=False):
    """Convert sin(x)**-2 to 1 + cot(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR15
    >>> from sympy.abc import x
    >>> from sympy import sin
    >>> TR15(1 - 1/sin(x)**2)
    -cot(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, sin)):
            return rv

        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base**(e + 1))/rv.base

        ia = 1/rv
        a = _TR56(ia, sin, cot, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv

    return bottom_up(rv, f)


def TR16(rv, max=4, pow=False):
    """Convert cos(x)**-2 to 1 + tan(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR16
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR16(1 - 1/cos(x)**2)
    -tan(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, cos)):
            return rv

        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base**(e + 1))/rv.base

        ia = 1/rv
        a = _TR56(ia, cos, tan, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv

    return bottom_up(rv, f)


def TR111(rv):
    """Convert f(x)**-i to g(x)**i where either ``i`` is an integer
    or the base is positive and f, g are: tan, cot; sin, csc; or cos, sec.

    Examples
    ========

    >>> from sympy.simplify.fu import TR111
    >>> from sympy.abc import x
    >>> from sympy import tan
    >>> TR111(1 - 1/tan(x)**2)
    1 - cot(x)**2

    """

    def f(rv):
        if not (
            isinstance(rv, Pow) and
            (rv.base.is_positive or rv.exp.is_integer and rv.exp.is_negative)):
            return rv

        if isinstance(rv.base, tan):
            return cot(rv.base.args[0])**-rv.exp
        elif isinstance(rv.base, sin):
            return csc(rv.base.args[0])**-rv.exp
        elif isinstance(rv.base, cos):
            return sec(rv.base.args[0])**-rv.exp
        return rv

    return bottom_up(rv, f)


def TR22(rv, max=4, pow=False):
    """Convert tan(x)**2 to sec(x)**2 - 1 and cot(x)**2 to csc(x)**2 - 1.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR22
    >>> from sympy.abc import x
    >>> from sympy import tan, cot
    >>> TR22(1 + tan(x)**2)
    sec(x)**2
    >>> TR22(1 + cot(x)**2)
    csc(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and rv.base.func in (cot, tan)):
            return rv

        rv = _TR56(rv, tan, sec, lambda x: x - 1, max=max, pow=pow)
        rv = _TR56(rv, cot, csc, lambda x: x - 1, max=max, pow=pow)
        return rv

    return bottom_up(rv, f)


def TRpower(rv):
    """Convert sin(x)**n and cos(x)**n with positive n to sums.

    Examples
    ========

    >>> from sympy.simplify.fu import TRpower
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> TRpower(sin(x)**6)
    -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 + 5/16
    >>> TRpower(sin(x)**3*cos(2*x)**4)
    (3*sin(x)/4 - sin(3*x)/4)*(cos(4*x)/2 + cos(8*x)/8 + 3/8)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Power-reduction_formulae

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, (sin, cos))):
            return rv
        b, n = rv.as_base_exp()
        x = b.args[0]
        if n.is_Integer and n.is_positive:
            if n.is_odd and isinstance(b, cos):
                rv = 2**(1-n)*Add(*[binomial(n, k)*cos((n - 2*k)*x)
                    for k in range((n + 1)/2)])
            elif n.is_odd and isinstance(b, sin):
                rv = 2**(1-n)*S.NegativeOne**((n-1)/2)*Add(*[binomial(n, k)*
                    S.NegativeOne**k*sin((n - 2*k)*x) for k in range((n + 1)/2)])
            elif n.is_even and isinstance(b, cos):
                rv = 2**(1-n)*Add(*[binomial(n, k)*cos((n - 2*k)*x)
                    for k in range(n/2)])
            elif n.is_even and isinstance(b, sin):
                rv = 2**(1-n)*S.NegativeOne**(n/2)*Add(*[binomial(n, k)*
                    S.NegativeOne**k*cos((n - 2*k)*x) for k in range(n/2)])
            if n.is_even:
                rv += 2**(-n)*binomial(n, n/2)
        return rv

    return bottom_up(rv, f)


def L(rv):
    """Return count of trigonometric functions in expression.

    Examples
    ========

    >>> from sympy.simplify.fu import L
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> L(cos(x)+sin(x))
    2
    """
    return S(rv.count(TrigonometricFunction))


# ============== end of basic Fu-like tools =====================

if SYMPY_DEBUG:
    (TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13,
    TR2i, TRmorrie, TR14, TR15, TR16, TR12i, TR111, TR22
    )= list(map(debug,
    (TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13,
    TR2i, TRmorrie, TR14, TR15, TR16, TR12i, TR111, TR22)))


# tuples are chains  --  (f, g) -> lambda x: g(f(x))
# lists are choices  --  [f, g] -> lambda x: min(f(x), g(x), key=objective)

CTR1 = [(TR5, TR0), (TR6, TR0), identity]

CTR2 = (TR11, [(TR5, TR0), (TR6, TR0), TR0])

CTR3 = [(TRmorrie, TR8, TR0), (TRmorrie, TR8, TR10i, TR0), identity]

CTR4 = [(TR4, TR10i), identity]

RL1 = (TR4, TR3, TR4, TR12, TR4, TR13, TR4, TR0)


# XXX it's a little unclear how this one is to be implemented
# see Fu paper of reference, page 7. What is the Union symbol referring to?
# The diagram shows all these as one chain of transformations, but the
# text refers to them being applied independently. Also, a break
# if L starts to increase has not been implemented.
RL2 = [
    (TR4, TR3, TR10, TR4, TR3, TR11),
    (TR5, TR7, TR11, TR4),
    (CTR3, CTR1, TR9, CTR2, TR4, TR9, TR9, CTR4),
    identity,
    ]


def fu(rv, measure=lambda x: (L(x), x.count_ops())):
    """Attempt to simplify expression by using transformation rules given
    in the algorithm by Fu et al.

    :func:`fu` will try to minimize the objective function ``measure``.
    By default this first minimizes the number of trig terms and then minimizes
    the number of total operations.

    Examples
    ========

    >>> from sympy.simplify.fu import fu
    >>> from sympy import cos, sin, tan, pi, S, sqrt
    >>> from sympy.abc import x, y, a, b

    >>> fu(sin(50)**2 + cos(50)**2 + sin(pi/6))
    3/2
    >>> fu(sqrt(6)*cos(x) + sqrt(2)*sin(x))
    2*sqrt(2)*sin(x + pi/3)

    CTR1 example

    >>> eq = sin(x)**4 - cos(y)**2 + sin(y)**2 + 2*cos(x)**2
    >>> fu(eq)
    cos(x)**4 - 2*cos(y)**2 + 2

    CTR2 example

    >>> fu(S.Half - cos(2*x)/2)
    sin(x)**2

    CTR3 example

    >>> fu(sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b)))
    sqrt(2)*sin(a + b + pi/4)

    CTR4 example

    >>> fu(sqrt(3)*cos(x)/2 + sin(x)/2)
    sin(x + pi/3)

    Example 1

    >>> fu(1-sin(2*x)**2/4-sin(y)**2-cos(x)**4)
    -cos(x)**2 + cos(y)**2

    Example 2

    >>> fu(cos(4*pi/9))
    sin(pi/18)
    >>> fu(cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9))
    1/16

    Example 3

    >>> fu(tan(7*pi/18)+tan(5*pi/18)-sqrt(3)*tan(5*pi/18)*tan(7*pi/18))
    -sqrt(3)

    Objective function example

    >>> fu(sin(x)/cos(x))  # default objective function
    tan(x)
    >>> fu(sin(x)/cos(x), measure=lambda x: -x.count_ops()) # maximize op count
    sin(x)/cos(x)

    References
    ==========

    .. [1] https://www.sciencedirect.com/science/article/pii/S0895717706001609
    """
    fRL1 = greedy(RL1, measure)
    fRL2 = greedy(RL2, measure)

    was = rv
    rv = sympify(rv)
    if not isinstance(rv, Expr):
        return rv.func(*[fu(a, measure=measure) for a in rv.args])
    rv = TR1(rv)
    if rv.has(tan, cot):
        rv1 = fRL1(rv)
        if (measure(rv1) < measure(rv)):
            rv = rv1
        if rv.has(tan, cot):
            rv = TR2(rv)
    if rv.has(sin, cos):
        rv1 = fRL2(rv)
        rv2 = TR8(TRmorrie(rv1))
        rv = min([was, rv, rv1, rv2], key=measure)
    return min(TR2i(rv), rv, key=measure)


def process_common_addends(rv, do, key2=None, key1=True):
    """Apply ``do`` to addends of ``rv`` that (if ``key1=True``) share at least
    a common absolute value of their coefficient and the value of ``key2`` when
    applied to the argument. If ``key1`` is False ``key2`` must be supplied and
    will be the only key applied.
    """

    # collect by absolute value of coefficient and key2
    absc = defaultdict(list)
    if key1:
        for a in rv.args:
            c, a = a.as_coeff_Mul()
            if c < 0:
                c = -c
                a = -a  # put the sign on `a`
            absc[(c, key2(a) if key2 else 1)].append(a)
    elif key2:
        for a in rv.args:
            absc[(S.One, key2(a))].append(a)
    else:
        raise ValueError('must have at least one key')

    args = []
    hit = False
    for k in absc:
        v = absc[k]
        c, _ = k
        if len(v) > 1:
            e = Add(*v, evaluate=False)
            new = do(e)
            if new != e:
                e = new
                hit = True
            args.append(c*e)
        else:
            args.append(c*v[0])
    if hit:
        rv = Add(*args)

    return rv


fufuncs = '''
    TR0 TR1 TR2 TR3 TR4 TR5 TR6 TR7 TR8 TR9 TR10 TR10i TR11
    TR12 TR13 L TR2i TRmorrie TR12i
    TR14 TR15 TR16 TR111 TR22'''.split()
FU = dict(list(zip(fufuncs, list(map(locals().get, fufuncs)))))


@cacheit
def _ROOT2():
    return sqrt(2)


@cacheit
def _ROOT3():
    return sqrt(3)


@cacheit
def _invROOT3():
    return 1/sqrt(3)


def trig_split(a, b, two=False):
    """Return the gcd, s1, s2, a1, a2, bool where

    If two is False (default) then::
        a + b = gcd*(s1*f(a1) + s2*f(a2)) where f = cos if bool else sin
    else:
        if bool, a + b was +/- cos(a1)*cos(a2) +/- sin(a1)*sin(a2) and equals
            n1*gcd*cos(a - b) if n1 == n2 else
            n1*gcd*cos(a + b)
        else a + b was +/- cos(a1)*sin(a2) +/- sin(a1)*cos(a2) and equals
            n1*gcd*sin(a + b) if n1 = n2 else
            n1*gcd*sin(b - a)

    Examples
    ========

    >>> from sympy.simplify.fu import trig_split
    >>> from sympy.abc import x, y, z
    >>> from sympy import cos, sin, sqrt

    >>> trig_split(cos(x), cos(y))
    (1, 1, 1, x, y, True)
    >>> trig_split(2*cos(x), -2*cos(y))
    (2, 1, -1, x, y, True)
    >>> trig_split(cos(x)*sin(y), cos(y)*sin(y))
    (sin(y), 1, 1, x, y, True)

    >>> trig_split(cos(x), -sqrt(3)*sin(x), two=True)
    (2, 1, -1, x, pi/6, False)
    >>> trig_split(cos(x), sin(x), two=True)
    (sqrt(2), 1, 1, x, pi/4, False)
    >>> trig_split(cos(x), -sin(x), two=True)
    (sqrt(2), 1, -1, x, pi/4, False)
    >>> trig_split(sqrt(2)*cos(x), -sqrt(6)*sin(x), two=True)
    (2*sqrt(2), 1, -1, x, pi/6, False)
    >>> trig_split(-sqrt(6)*cos(x), -sqrt(2)*sin(x), two=True)
    (-2*sqrt(2), 1, 1, x, pi/3, False)
    >>> trig_split(cos(x)/sqrt(6), sin(x)/sqrt(2), two=True)
    (sqrt(6)/3, 1, 1, x, pi/6, False)
    >>> trig_split(-sqrt(6)*cos(x)*sin(y), -sqrt(2)*sin(x)*sin(y), two=True)
    (-2*sqrt(2)*sin(y), 1, 1, x, pi/3, False)

    >>> trig_split(cos(x), sin(x))
    >>> trig_split(cos(x), sin(z))
    >>> trig_split(2*cos(x), -sin(x))
    >>> trig_split(cos(x), -sqrt(3)*sin(x))
    >>> trig_split(cos(x)*cos(y), sin(x)*sin(z))
    >>> trig_split(cos(x)*cos(y), sin(x)*sin(y))
    >>> trig_split(-sqrt(6)*cos(x), sqrt(2)*sin(x)*sin(y), two=True)
    """
    a, b = [Factors(i) for i in (a, b)]
    ua, ub = a.normal(b)
    gcd = a.gcd(b).as_expr()
    n1 = n2 = 1
    if S.NegativeOne in ua.factors:
        ua = ua.quo(S.NegativeOne)
        n1 = -n1
    elif S.NegativeOne in ub.factors:
        ub = ub.quo(S.NegativeOne)
        n2 = -n2
    a, b = [i.as_expr() for i in (ua, ub)]

    def pow_cos_sin(a, two):
        """Return ``a`` as a tuple (r, c, s) such that
        ``a = (r or 1)*(c or 1)*(s or 1)``.

        Three arguments are returned (radical, c-factor, s-factor) as
        long as the conditions set by ``two`` are met; otherwise None is
        returned. If ``two`` is True there will be one or two non-None
        values in the tuple: c and s or c and r or s and r or s or c with c
        being a cosine function (if possible) else a sine, and s being a sine
        function (if possible) else oosine. If ``two`` is False then there
        will only be a c or s term in the tuple.

        ``two`` also require that either two cos and/or sin be present (with
        the condition that if the functions are the same the arguments are
        different or vice versa) or that a single cosine or a single sine
        be present with an optional radical.

        If the above conditions dictated by ``two`` are not met then None
        is returned.
        """
        c = s = None
        co = S.One
        if a.is_Mul:
            co, a = a.as_coeff_Mul()
            if len(a.args) > 2 or not two:
                return None
            if a.is_Mul:
                args = list(a.args)
            else:
                args = [a]
            a = args.pop(0)
            if isinstance(a, cos):
                c = a
            elif isinstance(a, sin):
                s = a
            elif a.is_Pow and a.exp is S.Half:  # autoeval doesn't allow -1/2
                co *= a
            else:
                return None
            if args:
                b = args[0]
                if isinstance(b, cos):
                    if c:
                        s = b
                    else:
                        c = b
                elif isinstance(b, sin):
                    if s:
                        c = b
                    else:
                        s = b
                elif b.is_Pow and b.exp is S.Half:
                    co *= b
                else:
                    return None
            return co if co is not S.One else None, c, s
        elif isinstance(a, cos):
            c = a
        elif isinstance(a, sin):
            s = a
        if c is None and s is None:
            return
        co = co if co is not S.One else None
        return co, c, s

    # get the parts
    m = pow_cos_sin(a, two)
    if m is None:
        return
    coa, ca, sa = m
    m = pow_cos_sin(b, two)
    if m is None:
        return
    cob, cb, sb = m

    # check them
    if (not ca) and cb or ca and isinstance(ca, sin):
        coa, ca, sa, cob, cb, sb = cob, cb, sb, coa, ca, sa
        n1, n2 = n2, n1
    if not two:  # need cos(x) and cos(y) or sin(x) and sin(y)
        c = ca or sa
        s = cb or sb
        if not isinstance(c, s.func):
            return None
        return gcd, n1, n2, c.args[0], s.args[0], isinstance(c, cos)
    else:
        if not coa and not cob:
            if (ca and cb and sa and sb):
                if isinstance(ca, sa.func) is not isinstance(cb, sb.func):
                    return
                args = {j.args for j in (ca, sa)}
                if not all(i.args in args for i in (cb, sb)):
                    return
                return gcd, n1, n2, ca.args[0], sa.args[0], isinstance(ca, sa.func)
        if ca and sa or cb and sb or \
            two and (ca is None and sa is None or cb is None and sb is None):
            return
        c = ca or sa
        s = cb or sb
        if c.args != s.args:
            return
        if not coa:
            coa = S.One
        if not cob:
            cob = S.One
        if coa is cob:
            gcd *= _ROOT2()
            return gcd, n1, n2, c.args[0], pi/4, False
        elif coa/cob == _ROOT3():
            gcd *= 2*cob
            return gcd, n1, n2, c.args[0], pi/3, False
        elif coa/cob == _invROOT3():
            gcd *= 2*coa
            return gcd, n1, n2, c.args[0], pi/6, False


def as_f_sign_1(e):
    """If ``e`` is a sum that can be written as ``g*(a + s)`` where
    ``s`` is ``+/-1``, return ``g``, ``a``, and ``s`` where ``a`` does
    not have a leading negative coefficient.

    Examples
    ========

    >>> from sympy.simplify.fu import as_f_sign_1
    >>> from sympy.abc import x
    >>> as_f_sign_1(x + 1)
    (1, x, 1)
    >>> as_f_sign_1(x - 1)
    (1, x, -1)
    >>> as_f_sign_1(-x + 1)
    (-1, x, -1)
    >>> as_f_sign_1(-x - 1)
    (-1, x, 1)
    >>> as_f_sign_1(2*x + 2)
    (2, x, 1)
    """
    if not e.is_Add or len(e.args) != 2:
        return
    # exact match
    a, b = e.args
    if a in (S.NegativeOne, S.One):
        g = S.One
        if b.is_Mul and b.args[0].is_Number and b.args[0] < 0:
            a, b = -a, -b
            g = -g
        return g, b, a
    # gcd match
    a, b = [Factors(i) for i in e.args]
    ua, ub = a.normal(b)
    gcd = a.gcd(b).as_expr()
    if S.NegativeOne in ua.factors:
        ua = ua.quo(S.NegativeOne)
        n1 = -1
        n2 = 1
    elif S.NegativeOne in ub.factors:
        ub = ub.quo(S.NegativeOne)
        n1 = 1
        n2 = -1
    else:
        n1 = n2 = 1
    a, b = [i.as_expr() for i in (ua, ub)]
    if a is S.One:
        a, b = b, a
        n1, n2 = n2, n1
    if n1 == -1:
        gcd = -gcd
        n2 = -n2

    if b is S.One:
        return gcd, a, n2


def _osborne(e, d):
    """Replace all hyperbolic functions with trig functions using
    the Osborne rule.

    Notes
    =====

    ``d`` is a dummy variable to prevent automatic evaluation
    of trigonometric/hyperbolic functions.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """

    def f(rv):
        if not isinstance(rv, HyperbolicFunction):
            return rv
        a = rv.args[0]
        a = a*d if not a.is_Add else Add._from_args([i*d for i in a.args])
        if isinstance(rv, sinh):
            return I*sin(a)
        elif isinstance(rv, cosh):
            return cos(a)
        elif isinstance(rv, tanh):
            return I*tan(a)
        elif isinstance(rv, coth):
            return cot(a)/I
        elif isinstance(rv, sech):
            return sec(a)
        elif isinstance(rv, csch):
            return csc(a)/I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)

    return bottom_up(e, f)


def _osbornei(e, d):
    """Replace all trig functions with hyperbolic functions using
    the Osborne rule.

    Notes
    =====

    ``d`` is a dummy variable to prevent automatic evaluation
    of trigonometric/hyperbolic functions.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """

    def f(rv):
        if not isinstance(rv, TrigonometricFunction):
            return rv
        const, x = rv.args[0].as_independent(d, as_Add=True)
        a = x.xreplace({d: S.One}) + const*I
        if isinstance(rv, sin):
            return sinh(a)/I
        elif isinstance(rv, cos):
            return cosh(a)
        elif isinstance(rv, tan):
            return tanh(a)/I
        elif isinstance(rv, cot):
            return coth(a)*I
        elif isinstance(rv, sec):
            return sech(a)
        elif isinstance(rv, csc):
            return csch(a)*I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)

    return bottom_up(e, f)


def hyper_as_trig(rv):
    """Return an expression containing hyperbolic functions in terms
    of trigonometric functions. Any trigonometric functions initially
    present are replaced with Dummy symbols and the function to undo
    the masking and the conversion back to hyperbolics is also returned. It
    should always be true that::

        t, f = hyper_as_trig(expr)
        expr == f(t)

    Examples
    ========

    >>> from sympy.simplify.fu import hyper_as_trig, fu
    >>> from sympy.abc import x
    >>> from sympy import cosh, sinh
    >>> eq = sinh(x)**2 + cosh(x)**2
    >>> t, f = hyper_as_trig(eq)
    >>> f(fu(t))
    cosh(2*x)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """
    from sympy.simplify.simplify import signsimp
    from sympy.simplify.radsimp import collect

    # mask off trig functions
    trigs = rv.atoms(TrigonometricFunction)
    reps = [(t, Dummy()) for t in trigs]
    masked = rv.xreplace(dict(reps))

    # get inversion substitutions in place
    reps = [(v, k) for k, v in reps]

    d = Dummy()

    return _osborne(masked, d), lambda x: collect(signsimp(
        _osbornei(x, d).xreplace(dict(reps))), S.ImaginaryUnit)


def sincos_to_sum(expr):
    """Convert products and powers of sin and cos to sums.

    Explanation
    ===========

    Applied power reduction TRpower first, then expands products, and
    converts products to sums with TR8.

    Examples
    ========

    >>> from sympy.simplify.fu import sincos_to_sum
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> sincos_to_sum(16*sin(x)**3*cos(2*x)**2)
    7*sin(x) - 5*sin(3*x) + 3*sin(5*x) - sin(7*x)
    """

    if not expr.has(cos, sin):
        return expr
    else:
        return TR8(expand_mul(TRpower(expr)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\functions.py ===
# sql/functions.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""SQL function API, factories, and built-in functions."""

from __future__ import annotations

import datetime
import decimal
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import annotation
from . import coercions
from . import operators
from . import roles
from . import schema
from . import sqltypes
from . import type_api
from . import util as sqlutil
from ._typing import is_table_value_type
from .base import _entity_namespace
from .base import ColumnCollection
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .elements import _type_from_args
from .elements import BinaryExpression
from .elements import BindParameter
from .elements import Cast
from .elements import ClauseList
from .elements import ColumnElement
from .elements import Extract
from .elements import FunctionFilter
from .elements import Grouping
from .elements import literal_column
from .elements import NamedColumn
from .elements import Over
from .elements import WithinGroup
from .selectable import FromClause
from .selectable import Select
from .selectable import TableValuedAlias
from .sqltypes import TableValueType
from .type_api import TypeEngine
from .visitors import InternalTraversal
from .. import util


if TYPE_CHECKING:
    from ._typing import _ByArgument
    from ._typing import _ColumnExpressionArgument
    from ._typing import _ColumnExpressionOrLiteralArgument
    from ._typing import _ColumnExpressionOrStrLabelArgument
    from ._typing import _StarOrOne
    from ._typing import _TypeEngineArgument
    from .base import _EntityNamespace
    from .elements import ClauseElement
    from .elements import KeyedColumnElement
    from .elements import TableValuedColumn
    from .operators import OperatorType
    from ..engine.base import Connection
    from ..engine.cursor import CursorResult
    from ..engine.interfaces import _CoreMultiExecuteParams
    from ..engine.interfaces import CoreExecuteOptionsParameter
    from ..util.typing import Self

_T = TypeVar("_T", bound=Any)
_S = TypeVar("_S", bound=Any)

_registry: util.defaultdict[str, Dict[str, Type[Function[Any]]]] = (
    util.defaultdict(dict)
)


def register_function(
    identifier: str, fn: Type[Function[Any]], package: str = "_default"
) -> None:
    """Associate a callable with a particular func. name.

    This is normally called by GenericFunction, but is also
    available by itself so that a non-Function construct
    can be associated with the :data:`.func` accessor (i.e.
    CAST, EXTRACT).

    """
    reg = _registry[package]

    identifier = str(identifier).lower()

    # Check if a function with the same identifier is registered.
    if identifier in reg:
        util.warn(
            "The GenericFunction '{}' is already registered and "
            "is going to be overridden.".format(identifier)
        )
    reg[identifier] = fn


class FunctionElement(Executable, ColumnElement[_T], FromClause, Generative):
    """Base for SQL function-oriented constructs.

    This is a `generic type <https://peps.python.org/pep-0484/#generics>`_,
    meaning that type checkers and IDEs can be instructed on the types to
    expect in a :class:`_engine.Result` for this function. See
    :class:`.GenericFunction` for an example of how this is done.

    .. seealso::

        :ref:`tutorial_functions` - in the :ref:`unified_tutorial`

        :class:`.Function` - named SQL function.

        :data:`.func` - namespace which produces registered or ad-hoc
        :class:`.Function` instances.

        :class:`.GenericFunction` - allows creation of registered function
        types.

    """

    _traverse_internals = [
        ("clause_expr", InternalTraversal.dp_clauseelement),
        ("_with_ordinality", InternalTraversal.dp_boolean),
        ("_table_value_type", InternalTraversal.dp_has_cache_key),
    ]

    packagenames: Tuple[str, ...] = ()

    _has_args = False
    _with_ordinality = False
    _table_value_type: Optional[TableValueType] = None

    # some attributes that are defined between both ColumnElement and
    # FromClause are set to Any here to avoid typing errors
    primary_key: Any
    _is_clone_of: Any

    clause_expr: Grouping[Any]

    def __init__(
        self, *clauses: _ColumnExpressionOrLiteralArgument[Any]
    ) -> None:
        r"""Construct a :class:`.FunctionElement`.

        :param \*clauses: list of column expressions that form the arguments
         of the SQL function call.

        :param \**kwargs:  additional kwargs are typically consumed by
         subclasses.

        .. seealso::

            :data:`.func`

            :class:`.Function`

        """
        args: Sequence[_ColumnExpressionArgument[Any]] = [
            coercions.expect(
                roles.ExpressionElementRole,
                c,
                name=getattr(self, "name", None),
                apply_propagate_attrs=self,
            )
            for c in clauses
        ]
        self._has_args = self._has_args or bool(args)
        self.clause_expr = Grouping(
            ClauseList(operator=operators.comma_op, group_contents=True, *args)
        )

    _non_anon_label = None

    @property
    def _proxy_key(self) -> Any:
        return super()._proxy_key or getattr(self, "name", None)

    def _execute_on_connection(
        self,
        connection: Connection,
        distilled_params: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> CursorResult[Any]:
        return connection._execute_function(
            self, distilled_params, execution_options
        )

    def scalar_table_valued(
        self, name: str, type_: Optional[_TypeEngineArgument[_T]] = None
    ) -> ScalarFunctionColumn[_T]:
        """Return a column expression that's against this
        :class:`_functions.FunctionElement` as a scalar
        table-valued expression.

        The returned expression is similar to that returned by a single column
        accessed off of a :meth:`_functions.FunctionElement.table_valued`
        construct, except no FROM clause is generated; the function is rendered
        in the similar way as a scalar subquery.

        E.g.:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import func, select
            >>> fn = func.jsonb_each("{'k', 'v'}").scalar_table_valued("key")
            >>> print(select(fn))
            {printsql}SELECT (jsonb_each(:jsonb_each_1)).key

        .. versionadded:: 1.4.0b2

        .. seealso::

            :meth:`_functions.FunctionElement.table_valued`

            :meth:`_functions.FunctionElement.alias`

            :meth:`_functions.FunctionElement.column_valued`

        """  # noqa: E501

        return ScalarFunctionColumn(self, name, type_)

    def table_valued(
        self, *expr: _ColumnExpressionOrStrLabelArgument[Any], **kw: Any
    ) -> TableValuedAlias:
        r"""Return a :class:`_sql.TableValuedAlias` representation of this
        :class:`_functions.FunctionElement` with table-valued expressions added.

        e.g.:

        .. sourcecode:: pycon+sql

            >>> fn = func.generate_series(1, 5).table_valued(
            ...     "value", "start", "stop", "step"
            ... )

            >>> print(select(fn))
            {printsql}SELECT anon_1.value, anon_1.start, anon_1.stop, anon_1.step
            FROM generate_series(:generate_series_1, :generate_series_2) AS anon_1{stop}

            >>> print(select(fn.c.value, fn.c.stop).where(fn.c.value > 2))
            {printsql}SELECT anon_1.value, anon_1.stop
            FROM generate_series(:generate_series_1, :generate_series_2) AS anon_1
            WHERE anon_1.value > :value_1{stop}

        A WITH ORDINALITY expression may be generated by passing the keyword
        argument "with_ordinality":

        .. sourcecode:: pycon+sql

            >>> fn = func.generate_series(4, 1, -1).table_valued(
            ...     "gen", with_ordinality="ordinality"
            ... )
            >>> print(select(fn))
            {printsql}SELECT anon_1.gen, anon_1.ordinality
            FROM generate_series(:generate_series_1, :generate_series_2, :generate_series_3) WITH ORDINALITY AS anon_1

        :param \*expr: A series of string column names that will be added to the
         ``.c`` collection of the resulting :class:`_sql.TableValuedAlias`
         construct as columns.  :func:`_sql.column` objects with or without
         datatypes may also be used.

        :param name: optional name to assign to the alias name that's generated.
         If omitted, a unique anonymizing name is used.

        :param with_ordinality: string name that when present results in the
         ``WITH ORDINALITY`` clause being added to the alias, and the given
         string name will be added as a column to the .c collection
         of the resulting :class:`_sql.TableValuedAlias`.

        :param joins_implicitly: when True, the table valued function may be
         used in the FROM clause without any explicit JOIN to other tables
         in the SQL query, and no "cartesian product" warning will be generated.
         May be useful for SQL functions such as ``func.json_each()``.

         .. versionadded:: 1.4.33

        .. versionadded:: 1.4.0b2


        .. seealso::

            :ref:`tutorial_functions_table_valued` - in the :ref:`unified_tutorial`

            :ref:`postgresql_table_valued` - in the :ref:`postgresql_toplevel` documentation

            :meth:`_functions.FunctionElement.scalar_table_valued` - variant of
            :meth:`_functions.FunctionElement.table_valued` which delivers the
            complete table valued expression as a scalar column expression

            :meth:`_functions.FunctionElement.column_valued`

            :meth:`_sql.TableValuedAlias.render_derived` - renders the alias
            using a derived column clause, e.g. ``AS name(col1, col2, ...)``

        """  # noqa: 501

        new_func = self._generate()

        with_ordinality = kw.pop("with_ordinality", None)
        joins_implicitly = kw.pop("joins_implicitly", None)
        name = kw.pop("name", None)

        if with_ordinality:
            expr += (with_ordinality,)
            new_func._with_ordinality = True

        new_func.type = new_func._table_value_type = TableValueType(*expr)

        return new_func.alias(name=name, joins_implicitly=joins_implicitly)

    def column_valued(
        self, name: Optional[str] = None, joins_implicitly: bool = False
    ) -> TableValuedColumn[_T]:
        """Return this :class:`_functions.FunctionElement` as a column expression that
        selects from itself as a FROM clause.

        E.g.:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import select, func
            >>> gs = func.generate_series(1, 5, -1).column_valued()
            >>> print(select(gs))
            {printsql}SELECT anon_1
            FROM generate_series(:generate_series_1, :generate_series_2, :generate_series_3) AS anon_1

        This is shorthand for::

            gs = func.generate_series(1, 5, -1).alias().column

        :param name: optional name to assign to the alias name that's generated.
         If omitted, a unique anonymizing name is used.

        :param joins_implicitly: when True, the "table" portion of the column
         valued function may be a member of the FROM clause without any
         explicit JOIN to other tables in the SQL query, and no "cartesian
         product" warning will be generated. May be useful for SQL functions
         such as ``func.json_array_elements()``.

         .. versionadded:: 1.4.46

        .. seealso::

            :ref:`tutorial_functions_column_valued` - in the :ref:`unified_tutorial`

            :ref:`postgresql_column_valued` - in the :ref:`postgresql_toplevel` documentation

            :meth:`_functions.FunctionElement.table_valued`

        """  # noqa: 501

        return self.alias(name=name, joins_implicitly=joins_implicitly).column

    @util.ro_non_memoized_property
    def columns(self) -> ColumnCollection[str, KeyedColumnElement[Any]]:  # type: ignore[override]  # noqa: E501
        r"""The set of columns exported by this :class:`.FunctionElement`.

        This is a placeholder collection that allows the function to be
        placed in the FROM clause of a statement:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import column, select, func
            >>> stmt = select(column("x"), column("y")).select_from(func.myfunction())
            >>> print(stmt)
            {printsql}SELECT x, y FROM myfunction()

        The above form is a legacy feature that is now superseded by the
        fully capable :meth:`_functions.FunctionElement.table_valued`
        method; see that method for details.

        .. seealso::

            :meth:`_functions.FunctionElement.table_valued` - generates table-valued
            SQL function expressions.

        """  # noqa: E501
        return self.c

    @util.ro_memoized_property
    def c(self) -> ColumnCollection[str, KeyedColumnElement[Any]]:  # type: ignore[override]  # noqa: E501
        """synonym for :attr:`.FunctionElement.columns`."""

        return ColumnCollection(
            columns=[(col.key, col) for col in self._all_selected_columns]
        )

    @property
    def _all_selected_columns(self) -> Sequence[KeyedColumnElement[Any]]:
        if is_table_value_type(self.type):
            # TODO: this might not be fully accurate
            cols = cast(
                "Sequence[KeyedColumnElement[Any]]", self.type._elements
            )
        else:
            cols = [self.label(None)]

        return cols

    @property
    def exported_columns(  # type: ignore[override]
        self,
    ) -> ColumnCollection[str, KeyedColumnElement[Any]]:
        return self.columns

    @HasMemoized.memoized_attribute
    def clauses(self) -> ClauseList:
        """Return the underlying :class:`.ClauseList` which contains
        the arguments for this :class:`.FunctionElement`.

        """
        return cast(ClauseList, self.clause_expr.element)

    def over(
        self,
        *,
        partition_by: Optional[_ByArgument] = None,
        order_by: Optional[_ByArgument] = None,
        rows: Optional[Tuple[Optional[int], Optional[int]]] = None,
        range_: Optional[Tuple[Optional[int], Optional[int]]] = None,
        groups: Optional[Tuple[Optional[int], Optional[int]]] = None,
    ) -> Over[_T]:
        """Produce an OVER clause against this function.

        Used against aggregate or so-called "window" functions,
        for database backends that support window functions.

        The expression::

            func.row_number().over(order_by="x")

        is shorthand for::

            from sqlalchemy import over

            over(func.row_number(), order_by="x")

        See :func:`_expression.over` for a full description.

        .. seealso::

            :func:`_expression.over`

            :ref:`tutorial_window_functions` - in the :ref:`unified_tutorial`

        """
        return Over(
            self,
            partition_by=partition_by,
            order_by=order_by,
            rows=rows,
            range_=range_,
            groups=groups,
        )

    def within_group(
        self, *order_by: _ColumnExpressionArgument[Any]
    ) -> WithinGroup[_T]:
        """Produce a WITHIN GROUP (ORDER BY expr) clause against this function.

        Used against so-called "ordered set aggregate" and "hypothetical
        set aggregate" functions, including :class:`.percentile_cont`,
        :class:`.rank`, :class:`.dense_rank`, etc.

        See :func:`_expression.within_group` for a full description.

        .. seealso::

            :ref:`tutorial_functions_within_group` -
            in the :ref:`unified_tutorial`


        """
        return WithinGroup(self, *order_by)

    @overload
    def filter(self) -> Self: ...

    @overload
    def filter(
        self,
        __criterion0: _ColumnExpressionArgument[bool],
        *criterion: _ColumnExpressionArgument[bool],
    ) -> FunctionFilter[_T]: ...

    def filter(
        self, *criterion: _ColumnExpressionArgument[bool]
    ) -> Union[Self, FunctionFilter[_T]]:
        """Produce a FILTER clause against this function.

        Used against aggregate and window functions,
        for database backends that support the "FILTER" clause.

        The expression::

            func.count(1).filter(True)

        is shorthand for::

            from sqlalchemy import funcfilter

            funcfilter(func.count(1), True)

        .. seealso::

            :ref:`tutorial_functions_within_group` -
            in the :ref:`unified_tutorial`

            :class:`.FunctionFilter`

            :func:`.funcfilter`


        """
        if not criterion:
            return self
        return FunctionFilter(self, *criterion)

    def as_comparison(
        self, left_index: int, right_index: int
    ) -> FunctionAsBinary:
        """Interpret this expression as a boolean comparison between two
        values.

        This method is used for an ORM use case described at
        :ref:`relationship_custom_operator_sql_function`.

        A hypothetical SQL function "is_equal()" which compares to values
        for equality would be written in the Core expression language as::

            expr = func.is_equal("a", "b")

        If "is_equal()" above is comparing "a" and "b" for equality, the
        :meth:`.FunctionElement.as_comparison` method would be invoked as::

            expr = func.is_equal("a", "b").as_comparison(1, 2)

        Where above, the integer value "1" refers to the first argument of the
        "is_equal()" function and the integer value "2" refers to the second.

        This would create a :class:`.BinaryExpression` that is equivalent to::

            BinaryExpression("a", "b", operator=op.eq)

        However, at the SQL level it would still render as
        "is_equal('a', 'b')".

        The ORM, when it loads a related object or collection, needs to be able
        to manipulate the "left" and "right" sides of the ON clause of a JOIN
        expression. The purpose of this method is to provide a SQL function
        construct that can also supply this information to the ORM, when used
        with the :paramref:`_orm.relationship.primaryjoin` parameter. The
        return value is a containment object called :class:`.FunctionAsBinary`.

        An ORM example is as follows::

            class Venue(Base):
                __tablename__ = "venue"
                id = Column(Integer, primary_key=True)
                name = Column(String)

                descendants = relationship(
                    "Venue",
                    primaryjoin=func.instr(
                        remote(foreign(name)), name + "/"
                    ).as_comparison(1, 2)
                    == 1,
                    viewonly=True,
                    order_by=name,
                )

        Above, the "Venue" class can load descendant "Venue" objects by
        determining if the name of the parent Venue is contained within the
        start of the hypothetical descendant value's name, e.g. "parent1" would
        match up to "parent1/child1", but not to "parent2/child1".

        Possible use cases include the "materialized path" example given above,
        as well as making use of special SQL functions such as geometric
        functions to create join conditions.

        :param left_index: the integer 1-based index of the function argument
         that serves as the "left" side of the expression.
        :param right_index: the integer 1-based index of the function argument
         that serves as the "right" side of the expression.

        .. versionadded:: 1.3

        .. seealso::

            :ref:`relationship_custom_operator_sql_function` -
            example use within the ORM

        """
        return FunctionAsBinary(self, left_index, right_index)

    @property
    def _from_objects(self) -> Any:
        return self.clauses._from_objects

    def within_group_type(
        self, within_group: WithinGroup[_S]
    ) -> Optional[TypeEngine[_S]]:
        """For types that define their return type as based on the criteria
        within a WITHIN GROUP (ORDER BY) expression, called by the
        :class:`.WithinGroup` construct.

        Returns None by default, in which case the function's normal ``.type``
        is used.

        """

        return None

    def alias(
        self, name: Optional[str] = None, joins_implicitly: bool = False
    ) -> TableValuedAlias:
        r"""Produce a :class:`_expression.Alias` construct against this
        :class:`.FunctionElement`.

        .. tip::

            The :meth:`_functions.FunctionElement.alias` method is part of the
            mechanism by which "table valued" SQL functions are created.
            However, most use cases are covered by higher level methods on
            :class:`_functions.FunctionElement` including
            :meth:`_functions.FunctionElement.table_valued`, and
            :meth:`_functions.FunctionElement.column_valued`.

        This construct wraps the function in a named alias which
        is suitable for the FROM clause, in the style accepted for example
        by PostgreSQL.  A column expression is also provided using the
        special ``.column`` attribute, which may
        be used to refer to the output of the function as a scalar value
        in the columns or where clause, for a backend such as PostgreSQL.

        For a full table-valued expression, use the
        :meth:`_functions.FunctionElement.table_valued` method first to
        establish named columns.

        e.g.:

        .. sourcecode:: pycon+sql

            >>> from sqlalchemy import func, select, column
            >>> data_view = func.unnest([1, 2, 3]).alias("data_view")
            >>> print(select(data_view.column))
            {printsql}SELECT data_view
            FROM unnest(:unnest_1) AS data_view

        The :meth:`_functions.FunctionElement.column_valued` method provides
        a shortcut for the above pattern:

        .. sourcecode:: pycon+sql

            >>> data_view = func.unnest([1, 2, 3]).column_valued("data_view")
            >>> print(select(data_view))
            {printsql}SELECT data_view
            FROM unnest(:unnest_1) AS data_view

        .. versionadded:: 1.4.0b2  Added the ``.column`` accessor

        :param name: alias name, will be rendered as ``AS <name>`` in the
         FROM clause

        :param joins_implicitly: when True, the table valued function may be
         used in the FROM clause without any explicit JOIN to other tables
         in the SQL query, and no "cartesian product" warning will be
         generated.  May be useful for SQL functions such as
         ``func.json_each()``.

         .. versionadded:: 1.4.33

        .. seealso::

            :ref:`tutorial_functions_table_valued` -
            in the :ref:`unified_tutorial`

            :meth:`_functions.FunctionElement.table_valued`

            :meth:`_functions.FunctionElement.scalar_table_valued`

            :meth:`_functions.FunctionElement.column_valued`


        """

        return TableValuedAlias._construct(
            self,
            name=name,
            table_value_type=self.type,
            joins_implicitly=joins_implicitly,
        )

    def select(self) -> Select[Tuple[_T]]:
        """Produce a :func:`_expression.select` construct
        against this :class:`.FunctionElement`.

        This is shorthand for::

            s = select(function_element)

        """
        s: Select[Any] = Select(self)
        if self._execution_options:
            s = s.execution_options(**self._execution_options)
        return s

    def _bind_param(
        self,
        operator: OperatorType,
        obj: Any,
        type_: Optional[TypeEngine[_T]] = None,
        expanding: bool = False,
        **kw: Any,
    ) -> BindParameter[_T]:
        return BindParameter(
            None,
            obj,
            _compared_to_operator=operator,
            _compared_to_type=self.type,
            unique=True,
            type_=type_,
            expanding=expanding,
            **kw,
        )

    def self_group(self, against: Optional[OperatorType] = None) -> ClauseElement:  # type: ignore[override]  # noqa E501
        # for the moment, we are parenthesizing all array-returning
        # expressions against getitem.  This may need to be made
        # more portable if in the future we support other DBs
        # besides postgresql.
        if against is operators.getitem and isinstance(
            self.type, sqltypes.ARRAY
        ):
            return Grouping(self)
        else:
            return super().self_group(against=against)

    @property
    def entity_namespace(self) -> _EntityNamespace:
        """overrides FromClause.entity_namespace as functions are generally
        column expressions and not FromClauses.

        """
        # ideally functions would not be fromclauses but we failed to make
        # this adjustment in 1.4
        return _entity_namespace(self.clause_expr)


class FunctionAsBinary(BinaryExpression[Any]):
    _traverse_internals = [
        ("sql_function", InternalTraversal.dp_clauseelement),
        ("left_index", InternalTraversal.dp_plain_obj),
        ("right_index", InternalTraversal.dp_plain_obj),
        ("modifiers", InternalTraversal.dp_plain_dict),
    ]

    sql_function: FunctionElement[Any]
    left_index: int
    right_index: int

    def _gen_cache_key(self, anon_map: Any, bindparams: Any) -> Any:
        return ColumnElement._gen_cache_key(self, anon_map, bindparams)

    def __init__(
        self, fn: FunctionElement[Any], left_index: int, right_index: int
    ) -> None:
        self.sql_function = fn
        self.left_index = left_index
        self.right_index = right_index

        self.operator = operators.function_as_comparison_op
        self.type = sqltypes.BOOLEANTYPE
        self.negate = None
        self._is_implicitly_boolean = True
        self.modifiers = {}

    @property
    def left_expr(self) -> ColumnElement[Any]:
        return self.sql_function.clauses.clauses[self.left_index - 1]

    @left_expr.setter
    def left_expr(self, value: ColumnElement[Any]) -> None:
        self.sql_function.clauses.clauses[self.left_index - 1] = value

    @property
    def right_expr(self) -> ColumnElement[Any]:
        return self.sql_function.clauses.clauses[self.right_index - 1]

    @right_expr.setter
    def right_expr(self, value: ColumnElement[Any]) -> None:
        self.sql_function.clauses.clauses[self.right_index - 1] = value

    if not TYPE_CHECKING:
        # mypy can't accommodate @property to replace an instance
        # variable

        left = left_expr
        right = right_expr


class ScalarFunctionColumn(NamedColumn[_T]):
    __visit_name__ = "scalar_function_column"

    _traverse_internals = [
        ("name", InternalTraversal.dp_anon_name),
        ("type", InternalTraversal.dp_type),
        ("fn", InternalTraversal.dp_clauseelement),
    ]

    is_literal = False
    table = None

    def __init__(
        self,
        fn: FunctionElement[_T],
        name: str,
        type_: Optional[_TypeEngineArgument[_T]] = None,
    ) -> None:
        self.fn = fn
        self.name = name

        # if type is None, we get NULLTYPE, which is our _T.  But I don't
        # know how to get the overloads to express that correctly
        self.type = type_api.to_instance(type_)  # type: ignore


class _FunctionGenerator:
    """Generate SQL function expressions.

    :data:`.func` is a special object instance which generates SQL
    functions based on name-based attributes, e.g.:

    .. sourcecode:: pycon+sql

        >>> print(func.count(1))
        {printsql}count(:param_1)

    The returned object is an instance of :class:`.Function`, and  is a
    column-oriented SQL element like any other, and is used in that way:

    .. sourcecode:: pycon+sql

        >>> print(select(func.count(table.c.id)))
        {printsql}SELECT count(sometable.id) FROM sometable

    Any name can be given to :data:`.func`. If the function name is unknown to
    SQLAlchemy, it will be rendered exactly as is. For common SQL functions
    which SQLAlchemy is aware of, the name may be interpreted as a *generic
    function* which will be compiled appropriately to the target database:

    .. sourcecode:: pycon+sql

        >>> print(func.current_timestamp())
        {printsql}CURRENT_TIMESTAMP

    To call functions which are present in dot-separated packages,
    specify them in the same manner:

    .. sourcecode:: pycon+sql

        >>> print(func.stats.yield_curve(5, 10))
        {printsql}stats.yield_curve(:yield_curve_1, :yield_curve_2)

    SQLAlchemy can be made aware of the return type of functions to enable
    type-specific lexical and result-based behavior. For example, to ensure
    that a string-based function returns a Unicode value and is similarly
    treated as a string in expressions, specify
    :class:`~sqlalchemy.types.Unicode` as the type:

    .. sourcecode:: pycon+sql

        >>> print(
        ...     func.my_string("hi", type_=Unicode)
        ...     + " "
        ...     + func.my_string("there", type_=Unicode)
        ... )
        {printsql}my_string(:my_string_1) || :my_string_2 || my_string(:my_string_3)

    The object returned by a :data:`.func` call is usually an instance of
    :class:`.Function`.
    This object meets the "column" interface, including comparison and labeling
    functions.  The object can also be passed the :meth:`~.Connectable.execute`
    method of a :class:`_engine.Connection` or :class:`_engine.Engine`,
    where it will be
    wrapped inside of a SELECT statement first::

        print(connection.execute(func.current_timestamp()).scalar())

    In a few exception cases, the :data:`.func` accessor
    will redirect a name to a built-in expression such as :func:`.cast`
    or :func:`.extract`, as these names have well-known meaning
    but are not exactly the same as "functions" from a SQLAlchemy
    perspective.

    Functions which are interpreted as "generic" functions know how to
    calculate their return type automatically. For a listing of known generic
    functions, see :ref:`generic_functions`.

    .. note::

        The :data:`.func` construct has only limited support for calling
        standalone "stored procedures", especially those with special
        parameterization concerns.

        See the section :ref:`stored_procedures` for details on how to use
        the DBAPI-level ``callproc()`` method for fully traditional stored
        procedures.

    .. seealso::

        :ref:`tutorial_functions` - in the :ref:`unified_tutorial`

        :class:`.Function`

    """  # noqa

    def __init__(self, **opts: Any) -> None:
        self.__names: List[str] = []
        self.opts = opts

    def __getattr__(self, name: str) -> _FunctionGenerator:
        # passthru __ attributes; fixes pydoc
        if name.startswith("__"):
            try:
                return self.__dict__[name]  # type: ignore
            except KeyError:
                raise AttributeError(name)

        elif name.endswith("_"):
            name = name[0:-1]
        f = _FunctionGenerator(**self.opts)
        f.__names = list(self.__names) + [name]
        return f

    @overload
    def __call__(
        self, *c: Any, type_: _TypeEngineArgument[_T], **kwargs: Any
    ) -> Function[_T]: ...

    @overload
    def __call__(self, *c: Any, **kwargs: Any) -> Function[Any]: ...

    def __call__(self, *c: Any, **kwargs: Any) -> Function[Any]:
        o = self.opts.copy()
        o.update(kwargs)

        tokens = len(self.__names)

        if tokens == 2:
            package, fname = self.__names
        elif tokens == 1:
            package, fname = "_default", self.__names[0]
        else:
            package = None

        if package is not None:
            func = _registry[package].get(fname.lower())
            if func is not None:
                return func(*c, **o)

        return Function(
            self.__names[-1], packagenames=tuple(self.__names[0:-1]), *c, **o
        )

    if TYPE_CHECKING:
        # START GENERATED FUNCTION ACCESSORS

        # code within this block is **programmatically,
        # statically generated** by tools/generate_sql_functions.py

        @property
        def aggregate_strings(self) -> Type[aggregate_strings]: ...

        @property
        def ansifunction(self) -> Type[AnsiFunction[Any]]: ...

        # set ColumnElement[_T] as a separate overload, to appease
        # mypy which seems to not want to accept _T from
        # _ColumnExpressionArgument. Seems somewhat related to the covariant
        # _HasClauseElement as of mypy 1.15

        @overload
        def array_agg(
            self,
            col: ColumnElement[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> array_agg[_T]: ...

        @overload
        def array_agg(
            self,
            col: _ColumnExpressionArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> array_agg[_T]: ...

        @overload
        def array_agg(
            self,
            col: _T,
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> array_agg[_T]: ...

        def array_agg(
            self,
            col: _ColumnExpressionOrLiteralArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> array_agg[_T]: ...

        @property
        def cast(self) -> Type[Cast[Any]]: ...

        @property
        def char_length(self) -> Type[char_length]: ...

        # set ColumnElement[_T] as a separate overload, to appease
        # mypy which seems to not want to accept _T from
        # _ColumnExpressionArgument. Seems somewhat related to the covariant
        # _HasClauseElement as of mypy 1.15

        @overload
        def coalesce(
            self,
            col: ColumnElement[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> coalesce[_T]: ...

        @overload
        def coalesce(
            self,
            col: _ColumnExpressionArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> coalesce[_T]: ...

        @overload
        def coalesce(
            self,
            col: _T,
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> coalesce[_T]: ...

        def coalesce(
            self,
            col: _ColumnExpressionOrLiteralArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> coalesce[_T]: ...

        @property
        def concat(self) -> Type[concat]: ...

        @property
        def count(self) -> Type[count]: ...

        @property
        def cube(self) -> Type[cube[Any]]: ...

        @property
        def cume_dist(self) -> Type[cume_dist]: ...

        @property
        def current_date(self) -> Type[current_date]: ...

        @property
        def current_time(self) -> Type[current_time]: ...

        @property
        def current_timestamp(self) -> Type[current_timestamp]: ...

        @property
        def current_user(self) -> Type[current_user]: ...

        @property
        def dense_rank(self) -> Type[dense_rank]: ...

        @property
        def extract(self) -> Type[Extract]: ...

        @property
        def grouping_sets(self) -> Type[grouping_sets[Any]]: ...

        @property
        def localtime(self) -> Type[localtime]: ...

        @property
        def localtimestamp(self) -> Type[localtimestamp]: ...

        # set ColumnElement[_T] as a separate overload, to appease
        # mypy which seems to not want to accept _T from
        # _ColumnExpressionArgument. Seems somewhat related to the covariant
        # _HasClauseElement as of mypy 1.15

        @overload
        def max(  # noqa: A001
            self,
            col: ColumnElement[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> max[_T]: ...

        @overload
        def max(  # noqa: A001
            self,
            col: _ColumnExpressionArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> max[_T]: ...

        @overload
        def max(  # noqa: A001
            self,
            col: _T,
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> max[_T]: ...

        def max(  # noqa: A001
            self,
            col: _ColumnExpressionOrLiteralArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> max[_T]: ...

        # set ColumnElement[_T] as a separate overload, to appease
        # mypy which seems to not want to accept _T from
        # _ColumnExpressionArgument. Seems somewhat related to the covariant
        # _HasClauseElement as of mypy 1.15

        @overload
        def min(  # noqa: A001
            self,
            col: ColumnElement[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> min[_T]: ...

        @overload
        def min(  # noqa: A001
            self,
            col: _ColumnExpressionArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> min[_T]: ...

        @overload
        def min(  # noqa: A001
            self,
            col: _T,
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> min[_T]: ...

        def min(  # noqa: A001
            self,
            col: _ColumnExpressionOrLiteralArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> min[_T]: ...

        @property
        def mode(self) -> Type[mode[Any]]: ...

        @property
        def next_value(self) -> Type[next_value]: ...

        @property
        def now(self) -> Type[now]: ...

        @property
        def orderedsetagg(self) -> Type[OrderedSetAgg[Any]]: ...

        @property
        def percent_rank(self) -> Type[percent_rank]: ...

        @property
        def percentile_cont(self) -> Type[percentile_cont[Any]]: ...

        @property
        def percentile_disc(self) -> Type[percentile_disc[Any]]: ...

        @property
        def random(self) -> Type[random]: ...

        @property
        def rank(self) -> Type[rank]: ...

        @property
        def rollup(self) -> Type[rollup[Any]]: ...

        @property
        def session_user(self) -> Type[session_user]: ...

        # set ColumnElement[_T] as a separate overload, to appease
        # mypy which seems to not want to accept _T from
        # _ColumnExpressionArgument. Seems somewhat related to the covariant
        # _HasClauseElement as of mypy 1.15

        @overload
        def sum(  # noqa: A001
            self,
            col: ColumnElement[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> sum[_T]: ...

        @overload
        def sum(  # noqa: A001
            self,
            col: _ColumnExpressionArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> sum[_T]: ...

        @overload
        def sum(  # noqa: A001
            self,
            col: _T,
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> sum[_T]: ...

        def sum(  # noqa: A001
            self,
            col: _ColumnExpressionOrLiteralArgument[_T],
            *args: _ColumnExpressionOrLiteralArgument[Any],
            **kwargs: Any,
        ) -> sum[_T]: ...

        @property
        def sysdate(self) -> Type[sysdate]: ...

        @property
        def user(self) -> Type[user]: ...

        # END GENERATED FUNCTION ACCESSORS


func = _FunctionGenerator()
func.__doc__ = _FunctionGenerator.__doc__

modifier = _FunctionGenerator(group=False)


class Function(FunctionElement[_T]):
    r"""Describe a named SQL function.

    The :class:`.Function` object is typically generated from the
    :data:`.func` generation object.


    :param \*clauses: list of column expressions that form the arguments
     of the SQL function call.

    :param type\_: optional :class:`.TypeEngine` datatype object that will be
     used as the return value of the column expression generated by this
     function call.

    :param packagenames: a string which indicates package prefix names
     to be prepended to the function name when the SQL is generated.
     The :data:`.func` generator creates these when it is called using
     dotted format, e.g.::

        func.mypackage.some_function(col1, col2)

    .. seealso::

        :ref:`tutorial_functions` - in the :ref:`unified_tutorial`

        :data:`.func` - namespace which produces registered or ad-hoc
        :class:`.Function` instances.

        :class:`.GenericFunction` - allows creation of registered function
        types.

    """

    __visit_name__ = "function"

    _traverse_internals = FunctionElement._traverse_internals + [
        ("packagenames", InternalTraversal.dp_plain_obj),
        ("name", InternalTraversal.dp_string),
        ("type", InternalTraversal.dp_type),
    ]

    name: str

    identifier: str

    type: TypeEngine[_T]
    """A :class:`_types.TypeEngine` object which refers to the SQL return
    type represented by this SQL function.

    This datatype may be configured when generating a
    :class:`_functions.Function` object by passing the
    :paramref:`_functions.Function.type_` parameter, e.g.::

        >>> select(func.lower("some VALUE", type_=String))

    The small number of built-in classes of :class:`_functions.Function` come
    with a built-in datatype that's appropriate to the class of function and
    its arguments. For functions that aren't known, the type defaults to the
    "null type".

    """

    @overload
    def __init__(
        self,
        name: str,
        *clauses: _ColumnExpressionOrLiteralArgument[_T],
        type_: None = ...,
        packagenames: Optional[Tuple[str, ...]] = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        *clauses: _ColumnExpressionOrLiteralArgument[Any],
        type_: _TypeEngineArgument[_T] = ...,
        packagenames: Optional[Tuple[str, ...]] = ...,
    ) -> None: ...

    def __init__(
        self,
        name: str,
        *clauses: _ColumnExpressionOrLiteralArgument[Any],
        type_: Optional[_TypeEngineArgument[_T]] = None,
        packagenames: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """Construct a :class:`.Function`.

        The :data:`.func` construct is normally used to construct
        new :class:`.Function` instances.

        """
        self.packagenames = packagenames or ()
        self.name = name

        # if type is None, we get NULLTYPE, which is our _T.  But I don't
        # know how to get the overloads to express that correctly
        self.type = type_api.to_instance(type_)  # type: ignore

        FunctionElement.__init__(self, *clauses)

    def _bind_param(
        self,
        operator: OperatorType,
        obj: Any,
        type_: Optional[TypeEngine[_T]] = None,
        expanding: bool = False,
        **kw: Any,
    ) -> BindParameter[_T]:
        return BindParameter(
            self.name,
            obj,
            _compared_to_operator=operator,
            _compared_to_type=self.type,
            type_=type_,
            unique=True,
            expanding=expanding,
            **kw,
        )


class GenericFunction(Function[_T]):
    """Define a 'generic' function.

    A generic function is a pre-established :class:`.Function`
    class that is instantiated automatically when called
    by name from the :data:`.func` attribute.    Note that
    calling any name from :data:`.func` has the effect that
    a new :class:`.Function` instance is created automatically,
    given that name.  The primary use case for defining
    a :class:`.GenericFunction` class is so that a function
    of a particular name may be given a fixed return type.
    It can also include custom argument parsing schemes as well
    as additional methods.

    Subclasses of :class:`.GenericFunction` are automatically
    registered under the name of the class.  For
    example, a user-defined function ``as_utc()`` would
    be available immediately::

        from sqlalchemy.sql.functions import GenericFunction
        from sqlalchemy.types import DateTime


        class as_utc(GenericFunction):
            type = DateTime()
            inherit_cache = True


        print(select(func.as_utc()))

    User-defined generic functions can be organized into
    packages by specifying the "package" attribute when defining
    :class:`.GenericFunction`.   Third party libraries
    containing many functions may want to use this in order
    to avoid name conflicts with other systems.   For example,
    if our ``as_utc()`` function were part of a package
    "time"::

        class as_utc(GenericFunction):
            type = DateTime()
            package = "time"
            inherit_cache = True

    The above function would be available from :data:`.func`
    using the package name ``time``::

        print(select(func.time.as_utc()))

    A final option is to allow the function to be accessed
    from one name in :data:`.func` but to render as a different name.
    The ``identifier`` attribute will override the name used to
    access the function as loaded from :data:`.func`, but will retain
    the usage of ``name`` as the rendered name::

        class GeoBuffer(GenericFunction):
            type = Geometry()
            package = "geo"
            name = "ST_Buffer"
            identifier = "buffer"
            inherit_cache = True

    The above function will render as follows:

    .. sourcecode:: pycon+sql

        >>> print(func.geo.buffer())
        {printsql}ST_Buffer()

    The name will be rendered as is, however without quoting unless the name
    contains special characters that require quoting.  To force quoting
    on or off for the name, use the :class:`.sqlalchemy.sql.quoted_name`
    construct::

        from sqlalchemy.sql import quoted_name


        class GeoBuffer(GenericFunction):
            type = Geometry()
            package = "geo"
            name = quoted_name("ST_Buffer", True)
            identifier = "buffer"
            inherit_cache = True

    The above function will render as:

    .. sourcecode:: pycon+sql

        >>> print(func.geo.buffer())
        {printsql}"ST_Buffer"()

    Type parameters for this class as a
    `generic type <https://peps.python.org/pep-0484/#generics>`_ can be passed
    and should match the type seen in a :class:`_engine.Result`. For example::

        class as_utc(GenericFunction[datetime.datetime]):
            type = DateTime()
            inherit_cache = True

    The above indicates that the following expression returns a ``datetime``
    object::

        connection.scalar(select(func.as_utc()))

    .. versionadded:: 1.3.13  The :class:`.quoted_name` construct is now
       recognized for quoting when used with the "name" attribute of the
       object, so that quoting can be forced on or off for the function
       name.


    """

    coerce_arguments = True
    inherit_cache = True

    _register: bool

    name = "GenericFunction"

    def __init_subclass__(cls) -> None:
        if annotation.Annotated not in cls.__mro__:
            cls._register_generic_function(cls.__name__, cls.__dict__)
        super().__init_subclass__()

    @classmethod
    def _register_generic_function(
        cls, clsname: str, clsdict: Mapping[str, Any]
    ) -> None:
        cls.name = name = clsdict.get("name", clsname)
        cls.identifier = identifier = clsdict.get("identifier", name)
        package = clsdict.get("package", "_default")
        # legacy
        if "__return_type__" in clsdict:
            cls.type = clsdict["__return_type__"]

        # Check _register attribute status
        cls._register = getattr(cls, "_register", True)

        # Register the function if required
        if cls._register:
            register_function(identifier, cls, package)
        else:
            # Set _register to True to register child classes by default
            cls._register = True

    def __init__(
        self, *args: _ColumnExpressionOrLiteralArgument[Any], **kwargs: Any
    ) -> None:
        parsed_args = kwargs.pop("_parsed_args", None)
        if parsed_args is None:
            parsed_args = [
                coercions.expect(
                    roles.ExpressionElementRole,
                    c,
                    name=self.name,
                    apply_propagate_attrs=self,
                )
                for c in args
            ]
        self._has_args = self._has_args or bool(parsed_args)
        self.packagenames = ()

        self.clause_expr = Grouping(
            ClauseList(
                operator=operators.comma_op, group_contents=True, *parsed_args
            )
        )

        self.type = type_api.to_instance(  # type: ignore
            kwargs.pop("type_", None) or getattr(self, "type", None)
        )


register_function("cast", Cast)  # type: ignore
register_function("extract", Extract)  # type: ignore


class next_value(GenericFunction[int]):
    """Represent the 'next value', given a :class:`.Sequence`
    as its single argument.

    Compiles into the appropriate function on each backend,
    or will raise NotImplementedError if used on a backend
    that does not provide support for sequences.

    """

    type = sqltypes.Integer()
    name = "next_value"

    _traverse_internals = [
        ("sequence", InternalTraversal.dp_named_ddl_element)
    ]

    def __init__(self, seq: schema.Sequence, **kw: Any) -> None:
        assert isinstance(
            seq, schema.Sequence
        ), "next_value() accepts a Sequence object as input."
        self.sequence = seq
        self.type = sqltypes.to_instance(  # type: ignore
            seq.data_type or getattr(self, "type", None)
        )

    def compare(self, other: Any, **kw: Any) -> bool:
        return (
            isinstance(other, next_value)
            and self.sequence.name == other.sequence.name
        )

    @property
    def _from_objects(self) -> Any:
        return []


class AnsiFunction(GenericFunction[_T]):
    """Define a function in "ansi" format, which doesn't render parenthesis."""

    inherit_cache = True

    def __init__(
        self, *args: _ColumnExpressionArgument[Any], **kwargs: Any
    ) -> None:
        GenericFunction.__init__(self, *args, **kwargs)


class ReturnTypeFromArgs(GenericFunction[_T]):
    """Define a function whose return type is bound to the type of its
    arguments.
    """

    inherit_cache = True

    # set ColumnElement[_T] as a separate overload, to appease
    # mypy which seems to not want to accept _T from
    # _ColumnExpressionArgument. Seems somewhat related to the covariant
    # _HasClauseElement as of mypy 1.15

    @overload
    def __init__(
        self,
        col: ColumnElement[_T],
        *args: _ColumnExpressionOrLiteralArgument[Any],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        col: _ColumnExpressionArgument[_T],
        *args: _ColumnExpressionOrLiteralArgument[Any],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        col: _T,
        *args: _ColumnExpressionOrLiteralArgument[Any],
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self, *args: _ColumnExpressionOrLiteralArgument[_T], **kwargs: Any
    ) -> None:
        fn_args: Sequence[ColumnElement[Any]] = [
            coercions.expect(
                roles.ExpressionElementRole,
                c,
                name=self.name,
                apply_propagate_attrs=self,
            )
            for c in args
        ]
        kwargs.setdefault("type_", _type_from_args(fn_args))
        kwargs["_parsed_args"] = fn_args
        super().__init__(*fn_args, **kwargs)


class coalesce(ReturnTypeFromArgs[_T]):
    _has_args = True
    inherit_cache = True


class max(ReturnTypeFromArgs[_T]):  # noqa:  A001
    """The SQL MAX() aggregate function."""

    inherit_cache = True


class min(ReturnTypeFromArgs[_T]):  # noqa: A001
    """The SQL MIN() aggregate function."""

    inherit_cache = True


class sum(ReturnTypeFromArgs[_T]):  # noqa: A001
    """The SQL SUM() aggregate function."""

    inherit_cache = True


class now(GenericFunction[datetime.datetime]):
    """The SQL now() datetime function.

    SQLAlchemy dialects will usually render this particular function
    in a backend-specific way, such as rendering it as ``CURRENT_TIMESTAMP``.

    """

    type = sqltypes.DateTime()
    inherit_cache = True


class concat(GenericFunction[str]):
    """The SQL CONCAT() function, which concatenates strings.

    E.g.:

    .. sourcecode:: pycon+sql

        >>> print(select(func.concat("a", "b")))
        {printsql}SELECT concat(:concat_2, :concat_3) AS concat_1

    String concatenation in SQLAlchemy is more commonly available using the
    Python ``+`` operator with string datatypes, which will render a
    backend-specific concatenation operator, such as :

    .. sourcecode:: pycon+sql

        >>> print(select(literal("a") + "b"))
        {printsql}SELECT :param_1 || :param_2 AS anon_1


    """

    type = sqltypes.String()
    inherit_cache = True


class char_length(GenericFunction[int]):
    """The CHAR_LENGTH() SQL function."""

    type = sqltypes.Integer()
    inherit_cache = True

    def __init__(self, arg: _ColumnExpressionArgument[str], **kw: Any) -> None:
        # slight hack to limit to just one positional argument
        # not sure why this one function has this special treatment
        super().__init__(arg, **kw)


class random(GenericFunction[float]):
    """The RANDOM() SQL function."""

    _has_args = True
    inherit_cache = True


class count(GenericFunction[int]):
    r"""The ANSI COUNT aggregate function.  With no arguments,
    emits COUNT \*.

    E.g.::

        from sqlalchemy import func
        from sqlalchemy import select
        from sqlalchemy import table, column

        my_table = table("some_table", column("id"))

        stmt = select(func.count()).select_from(my_table)

    Executing ``stmt`` would emit:

    .. sourcecode:: sql

        SELECT count(*) AS count_1
        FROM some_table


    """

    type = sqltypes.Integer()
    inherit_cache = True

    def __init__(
        self,
        expression: Union[
            _ColumnExpressionArgument[Any], _StarOrOne, None
        ] = None,
        **kwargs: Any,
    ) -> None:
        if expression is None:
            expression = literal_column("*")
        super().__init__(expression, **kwargs)


class current_date(AnsiFunction[datetime.date]):
    """The CURRENT_DATE() SQL function."""

    type = sqltypes.Date()
    inherit_cache = True


class current_time(AnsiFunction[datetime.time]):
    """The CURRENT_TIME() SQL function."""

    type = sqltypes.Time()
    inherit_cache = True


class current_timestamp(AnsiFunction[datetime.datetime]):
    """The CURRENT_TIMESTAMP() SQL function."""

    type = sqltypes.DateTime()
    inherit_cache = True


class current_user(AnsiFunction[str]):
    """The CURRENT_USER() SQL function."""

    type = sqltypes.String()
    inherit_cache = True


class localtime(AnsiFunction[datetime.datetime]):
    """The localtime() SQL function."""

    type = sqltypes.DateTime()
    inherit_cache = True


class localtimestamp(AnsiFunction[datetime.datetime]):
    """The localtimestamp() SQL function."""

    type = sqltypes.DateTime()
    inherit_cache = True


class session_user(AnsiFunction[str]):
    """The SESSION_USER() SQL function."""

    type = sqltypes.String()
    inherit_cache = True


class sysdate(AnsiFunction[datetime.datetime]):
    """The SYSDATE() SQL function."""

    type = sqltypes.DateTime()
    inherit_cache = True


class user(AnsiFunction[str]):
    """The USER() SQL function."""

    type = sqltypes.String()
    inherit_cache = True


class array_agg(ReturnTypeFromArgs[Sequence[_T]]):
    """Support for the ARRAY_AGG function.

    The ``func.array_agg(expr)`` construct returns an expression of
    type :class:`_types.ARRAY`.

    e.g.::

        stmt = select(func.array_agg(table.c.values)[2:5])

    .. seealso::

        :func:`_postgresql.array_agg` - PostgreSQL-specific version that
        returns :class:`_postgresql.ARRAY`, which has PG-specific operators
        added.

    """

    inherit_cache = True

    def __init__(
        self, *args: _ColumnExpressionArgument[Any], **kwargs: Any
    ) -> None:
        fn_args: Sequence[ColumnElement[Any]] = [
            coercions.expect(
                roles.ExpressionElementRole, c, apply_propagate_attrs=self
            )
            for c in args
        ]

        default_array_type = kwargs.pop("_default_array_type", sqltypes.ARRAY)
        if "type_" not in kwargs:
            type_from_args = _type_from_args(fn_args)
            if isinstance(type_from_args, sqltypes.ARRAY):
                kwargs["type_"] = type_from_args
            else:
                kwargs["type_"] = default_array_type(
                    type_from_args, dimensions=1
                )
        kwargs["_parsed_args"] = fn_args
        super().__init__(*fn_args, **kwargs)


class OrderedSetAgg(GenericFunction[_T]):
    """Define a function where the return type is based on the sort
    expression type as defined by the expression passed to the
    :meth:`.FunctionElement.within_group` method."""

    array_for_multi_clause = False
    inherit_cache = True

    def within_group_type(
        self, within_group: WithinGroup[Any]
    ) -> TypeEngine[Any]:
        func_clauses = cast(ClauseList, self.clause_expr.element)
        order_by: Sequence[ColumnElement[Any]] = sqlutil.unwrap_order_by(
            within_group.order_by
        )
        if self.array_for_multi_clause and len(func_clauses.clauses) > 1:
            return sqltypes.ARRAY(order_by[0].type)
        else:
            return order_by[0].type


class mode(OrderedSetAgg[_T]):
    """Implement the ``mode`` ordered-set aggregate function.

    This function must be used with the :meth:`.FunctionElement.within_group`
    modifier to supply a sort expression to operate upon.

    The return type of this function is the same as the sort expression.

    """

    inherit_cache = True


class percentile_cont(OrderedSetAgg[_T]):
    """Implement the ``percentile_cont`` ordered-set aggregate function.

    This function must be used with the :meth:`.FunctionElement.within_group`
    modifier to supply a sort expression to operate upon.

    The return type of this function is the same as the sort expression,
    or if the arguments are an array, an :class:`_types.ARRAY` of the sort
    expression's type.

    """

    array_for_multi_clause = True
    inherit_cache = True


class percentile_disc(OrderedSetAgg[_T]):
    """Implement the ``percentile_disc`` ordered-set aggregate function.

    This function must be used with the :meth:`.FunctionElement.within_group`
    modifier to supply a sort expression to operate upon.

    The return type of this function is the same as the sort expression,
    or if the arguments are an array, an :class:`_types.ARRAY` of the sort
    expression's type.

    """

    array_for_multi_clause = True
    inherit_cache = True


class rank(GenericFunction[int]):
    """Implement the ``rank`` hypothetical-set aggregate function.

    This function must be used with the :meth:`.FunctionElement.within_group`
    modifier to supply a sort expression to operate upon.

    The return type of this function is :class:`.Integer`.

    """

    type = sqltypes.Integer()
    inherit_cache = True


class dense_rank(GenericFunction[int]):
    """Implement the ``dense_rank`` hypothetical-set aggregate function.

    This function must be used with the :meth:`.FunctionElement.within_group`
    modifier to supply a sort expression to operate upon.

    The return type of this function is :class:`.Integer`.

    """

    type = sqltypes.Integer()
    inherit_cache = True


class percent_rank(GenericFunction[decimal.Decimal]):
    """Implement the ``percent_rank`` hypothetical-set aggregate function.

    This function must be used with the :meth:`.FunctionElement.within_group`
    modifier to supply a sort expression to operate upon.

    The return type of this function is :class:`.Numeric`.

    """

    type: sqltypes.Numeric[decimal.Decimal] = sqltypes.Numeric()
    inherit_cache = True


class cume_dist(GenericFunction[decimal.Decimal]):
    """Implement the ``cume_dist`` hypothetical-set aggregate function.

    This function must be used with the :meth:`.FunctionElement.within_group`
    modifier to supply a sort expression to operate upon.

    The return type of this function is :class:`.Numeric`.

    """

    type: sqltypes.Numeric[decimal.Decimal] = sqltypes.Numeric()
    inherit_cache = True


class cube(GenericFunction[_T]):
    r"""Implement the ``CUBE`` grouping operation.

    This function is used as part of the GROUP BY of a statement,
    e.g. :meth:`_expression.Select.group_by`::

        stmt = select(
            func.sum(table.c.value), table.c.col_1, table.c.col_2
        ).group_by(func.cube(table.c.col_1, table.c.col_2))

    .. versionadded:: 1.2

    """

    _has_args = True
    inherit_cache = True


class rollup(GenericFunction[_T]):
    r"""Implement the ``ROLLUP`` grouping operation.

    This function is used as part of the GROUP BY of a statement,
    e.g. :meth:`_expression.Select.group_by`::

        stmt = select(
            func.sum(table.c.value), table.c.col_1, table.c.col_2
        ).group_by(func.rollup(table.c.col_1, table.c.col_2))

    .. versionadded:: 1.2

    """

    _has_args = True
    inherit_cache = True


class grouping_sets(GenericFunction[_T]):
    r"""Implement the ``GROUPING SETS`` grouping operation.

    This function is used as part of the GROUP BY of a statement,
    e.g. :meth:`_expression.Select.group_by`::

        stmt = select(
            func.sum(table.c.value), table.c.col_1, table.c.col_2
        ).group_by(func.grouping_sets(table.c.col_1, table.c.col_2))

    In order to group by multiple sets, use the :func:`.tuple_` construct::

        from sqlalchemy import tuple_

        stmt = select(
            func.sum(table.c.value), table.c.col_1, table.c.col_2, table.c.col_3
        ).group_by(
            func.grouping_sets(
                tuple_(table.c.col_1, table.c.col_2),
                tuple_(table.c.value, table.c.col_3),
            )
        )

    .. versionadded:: 1.2

    """  # noqa: E501

    _has_args = True
    inherit_cache = True


class aggregate_strings(GenericFunction[str]):
    """Implement a generic string aggregation function.

    This function will concatenate non-null values into a string and
    separate the values by a delimiter.

    This function is compiled on a per-backend basis, into functions
    such as ``group_concat()``, ``string_agg()``, or ``LISTAGG()``.

    e.g. Example usage with delimiter '.'::

        stmt = select(func.aggregate_strings(table.c.str_col, "."))

    The return type of this function is :class:`.String`.

    .. versionadded: 2.0.21

    """

    type = sqltypes.String()
    _has_args = True
    inherit_cache = True

    def __init__(
        self, clause: _ColumnExpressionArgument[Any], separator: str
    ) -> None:
        super().__init__(clause, separator)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\reflection.py ===
# engine/reflection.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Provides an abstraction for obtaining database schema information.

Usage Notes:

Here are some general conventions when accessing the low level inspector
methods such as get_table_names, get_columns, etc.

1. Inspector methods return lists of dicts in most cases for the following
   reasons:

   * They're both standard types that can be serialized.
   * Using a dict instead of a tuple allows easy expansion of attributes.
   * Using a list for the outer structure maintains order and is easy to work
     with (e.g. list comprehension [d['name'] for d in cols]).

2. Records that contain a name, such as the column name in a column record
   use the key 'name'. So for most return values, each record will have a
   'name' attribute..
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import quoted_name
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final

if TYPE_CHECKING:
    from .interfaces import Dialect
    from .interfaces import ReflectedCheckConstraint
    from .interfaces import ReflectedColumn
    from .interfaces import ReflectedForeignKeyConstraint
    from .interfaces import ReflectedIndex
    from .interfaces import ReflectedPrimaryKeyConstraint
    from .interfaces import ReflectedTableComment
    from .interfaces import ReflectedUniqueConstraint
    from .interfaces import TableKey

_R = TypeVar("_R")


@util.decorator
def cache(
    fn: Callable[..., _R],
    self: Dialect,
    con: Connection,
    *args: Any,
    **kw: Any,
) -> _R:
    info_cache = kw.get("info_cache", None)
    if info_cache is None:
        return fn(self, con, *args, **kw)
    exclude = {"info_cache", "unreflectable"}
    key = (
        fn.__name__,
        tuple(
            (str(a), a.quote) if isinstance(a, quoted_name) else a
            for a in args
            if isinstance(a, str)
        ),
        tuple(
            (k, (str(v), v.quote) if isinstance(v, quoted_name) else v)
            for k, v in kw.items()
            if k not in exclude
        ),
    )
    ret: _R = info_cache.get(key)
    if ret is None:
        ret = fn(self, con, *args, **kw)
        info_cache[key] = ret
    return ret


def flexi_cache(
    *traverse_args: Tuple[str, InternalTraversal]
) -> Callable[[Callable[..., _R]], Callable[..., _R]]:
    @util.decorator
    def go(
        fn: Callable[..., _R],
        self: Dialect,
        con: Connection,
        *args: Any,
        **kw: Any,
    ) -> _R:
        info_cache = kw.get("info_cache", None)
        if info_cache is None:
            return fn(self, con, *args, **kw)
        key = _ad_hoc_cache_key_from_args((fn.__name__,), traverse_args, args)
        ret: _R = info_cache.get(key)
        if ret is None:
            ret = fn(self, con, *args, **kw)
            info_cache[key] = ret
        return ret

    return go


@unique
class ObjectKind(Flag):
    """Enumerator that indicates which kind of object to return when calling
    the ``get_multi`` methods.

    This is a Flag enum, so custom combinations can be passed. For example,
    to reflect tables and plain views ``ObjectKind.TABLE | ObjectKind.VIEW``
    may be used.

    .. note::
      Not all dialect may support all kind of object. If a dialect does
      not support a particular object an empty dict is returned.
      In case a dialect supports an object, but the requested method
      is not applicable for the specified kind the default value
      will be returned for each reflected object. For example reflecting
      check constraints of view return a dict with all the views with
      empty lists as values.
    """

    TABLE = auto()
    "Reflect table objects"
    VIEW = auto()
    "Reflect plain view objects"
    MATERIALIZED_VIEW = auto()
    "Reflect materialized view object"

    ANY_VIEW = VIEW | MATERIALIZED_VIEW
    "Reflect any kind of view objects"
    ANY = TABLE | VIEW | MATERIALIZED_VIEW
    "Reflect all type of objects"


@unique
class ObjectScope(Flag):
    """Enumerator that indicates which scope to use when calling
    the ``get_multi`` methods.
    """

    DEFAULT = auto()
    "Include default scope"
    TEMPORARY = auto()
    "Include only temp scope"
    ANY = DEFAULT | TEMPORARY
    "Include both default and temp scope"


@inspection._self_inspects
class Inspector(inspection.Inspectable["Inspector"]):
    """Performs database schema inspection.

    The Inspector acts as a proxy to the reflection methods of the
    :class:`~sqlalchemy.engine.interfaces.Dialect`, providing a
    consistent interface as well as caching support for previously
    fetched metadata.

    A :class:`_reflection.Inspector` object is usually created via the
    :func:`_sa.inspect` function, which may be passed an
    :class:`_engine.Engine`
    or a :class:`_engine.Connection`::

        from sqlalchemy import inspect, create_engine

        engine = create_engine("...")
        insp = inspect(engine)

    Where above, the :class:`~sqlalchemy.engine.interfaces.Dialect` associated
    with the engine may opt to return an :class:`_reflection.Inspector`
    subclass that
    provides additional methods specific to the dialect's target database.

    """

    bind: Union[Engine, Connection]
    engine: Engine
    _op_context_requires_connect: bool
    dialect: Dialect
    info_cache: Dict[Any, Any]

    @util.deprecated(
        "1.4",
        "The __init__() method on :class:`_reflection.Inspector` "
        "is deprecated and "
        "will be removed in a future release.  Please use the "
        ":func:`.sqlalchemy.inspect` "
        "function on an :class:`_engine.Engine` or "
        ":class:`_engine.Connection` "
        "in order to "
        "acquire an :class:`_reflection.Inspector`.",
    )
    def __init__(self, bind: Union[Engine, Connection]):
        """Initialize a new :class:`_reflection.Inspector`.

        :param bind: a :class:`~sqlalchemy.engine.Connection`,
          which is typically an instance of
          :class:`~sqlalchemy.engine.Engine` or
          :class:`~sqlalchemy.engine.Connection`.

        For a dialect-specific instance of :class:`_reflection.Inspector`, see
        :meth:`_reflection.Inspector.from_engine`

        """
        self._init_legacy(bind)

    @classmethod
    def _construct(
        cls, init: Callable[..., Any], bind: Union[Engine, Connection]
    ) -> Inspector:
        if hasattr(bind.dialect, "inspector"):
            cls = bind.dialect.inspector

        self = cls.__new__(cls)
        init(self, bind)
        return self

    def _init_legacy(self, bind: Union[Engine, Connection]) -> None:
        if hasattr(bind, "exec_driver_sql"):
            self._init_connection(bind)  # type: ignore[arg-type]
        else:
            self._init_engine(bind)

    def _init_engine(self, engine: Engine) -> None:
        self.bind = self.engine = engine
        engine.connect().close()
        self._op_context_requires_connect = True
        self.dialect = self.engine.dialect
        self.info_cache = {}

    def _init_connection(self, connection: Connection) -> None:
        self.bind = connection
        self.engine = connection.engine
        self._op_context_requires_connect = False
        self.dialect = self.engine.dialect
        self.info_cache = {}

    def clear_cache(self) -> None:
        """reset the cache for this :class:`.Inspector`.

        Inspection methods that have data cached will emit SQL queries
        when next called to get new data.

        .. versionadded:: 2.0

        """
        self.info_cache.clear()

    @classmethod
    @util.deprecated(
        "1.4",
        "The from_engine() method on :class:`_reflection.Inspector` "
        "is deprecated and "
        "will be removed in a future release.  Please use the "
        ":func:`.sqlalchemy.inspect` "
        "function on an :class:`_engine.Engine` or "
        ":class:`_engine.Connection` "
        "in order to "
        "acquire an :class:`_reflection.Inspector`.",
    )
    def from_engine(cls, bind: Engine) -> Inspector:
        """Construct a new dialect-specific Inspector object from the given
        engine or connection.

        :param bind: a :class:`~sqlalchemy.engine.Connection`
         or :class:`~sqlalchemy.engine.Engine`.

        This method differs from direct a direct constructor call of
        :class:`_reflection.Inspector` in that the
        :class:`~sqlalchemy.engine.interfaces.Dialect` is given a chance to
        provide a dialect-specific :class:`_reflection.Inspector` instance,
        which may
        provide additional methods.

        See the example at :class:`_reflection.Inspector`.

        """
        return cls._construct(cls._init_legacy, bind)

    @inspection._inspects(Engine)
    def _engine_insp(bind: Engine) -> Inspector:  # type: ignore[misc]
        return Inspector._construct(Inspector._init_engine, bind)

    @inspection._inspects(Connection)
    def _connection_insp(bind: Connection) -> Inspector:  # type: ignore[misc]
        return Inspector._construct(Inspector._init_connection, bind)

    @contextlib.contextmanager
    def _operation_context(self) -> Generator[Connection, None, None]:
        """Return a context that optimizes for multiple operations on a single
        transaction.

        This essentially allows connect()/close() to be called if we detected
        that we're against an :class:`_engine.Engine` and not a
        :class:`_engine.Connection`.

        """
        conn: Connection
        if self._op_context_requires_connect:
            conn = self.bind.connect()  # type: ignore[union-attr]
        else:
            conn = self.bind  # type: ignore[assignment]
        try:
            yield conn
        finally:
            if self._op_context_requires_connect:
                conn.close()

    @contextlib.contextmanager
    def _inspection_context(self) -> Generator[Inspector, None, None]:
        """Return an :class:`_reflection.Inspector`
        from this one that will run all
        operations on a single connection.

        """

        with self._operation_context() as conn:
            sub_insp = self._construct(self.__class__._init_connection, conn)
            sub_insp.info_cache = self.info_cache
            yield sub_insp

    @property
    def default_schema_name(self) -> Optional[str]:
        """Return the default schema name presented by the dialect
        for the current engine's database user.

        E.g. this is typically ``public`` for PostgreSQL and ``dbo``
        for SQL Server.

        """
        return self.dialect.default_schema_name

    def get_schema_names(self, **kw: Any) -> List[str]:
        r"""Return all schema names.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.
        """

        with self._operation_context() as conn:
            return self.dialect.get_schema_names(
                conn, info_cache=self.info_cache, **kw
            )

    def get_table_names(
        self, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        r"""Return all table names within a particular schema.

        The names are expected to be real tables only, not views.
        Views are instead returned using the
        :meth:`_reflection.Inspector.get_view_names` and/or
        :meth:`_reflection.Inspector.get_materialized_view_names`
        methods.

        :param schema: Schema name. If ``schema`` is left at ``None``, the
         database's default schema is
         used, else the named schema is searched.  If the database does not
         support named schemas, behavior is undefined if ``schema`` is not
         passed as ``None``.  For special quoting, use :class:`.quoted_name`.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        .. seealso::

            :meth:`_reflection.Inspector.get_sorted_table_and_fkc_names`

            :attr:`_schema.MetaData.sorted_tables`

        """

        with self._operation_context() as conn:
            return self.dialect.get_table_names(
                conn, schema, info_cache=self.info_cache, **kw
            )

    def has_table(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> bool:
        r"""Return True if the backend has a table, view, or temporary
        table of the given name.

        :param table_name: name of the table to check
        :param schema: schema name to query, if not the default schema.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        .. versionadded:: 1.4 - the :meth:`.Inspector.has_table` method
           replaces the :meth:`_engine.Engine.has_table` method.

        .. versionchanged:: 2.0:: :meth:`.Inspector.has_table` now formally
           supports checking for additional table-like objects:

           * any type of views (plain or materialized)
           * temporary tables of any kind

           Previously, these two checks were not formally specified and
           different dialects would vary in their behavior.   The dialect
           testing suite now includes tests for all of these object types
           and should be supported by all SQLAlchemy-included dialects.
           Support among third party dialects may be lagging, however.

        """
        with self._operation_context() as conn:
            return self.dialect.has_table(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )

    def has_sequence(
        self, sequence_name: str, schema: Optional[str] = None, **kw: Any
    ) -> bool:
        r"""Return True if the backend has a sequence with the given name.

        :param sequence_name: name of the sequence to check
        :param schema: schema name to query, if not the default schema.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        .. versionadded:: 1.4

        """
        with self._operation_context() as conn:
            return self.dialect.has_sequence(
                conn, sequence_name, schema, info_cache=self.info_cache, **kw
            )

    def has_index(
        self,
        table_name: str,
        index_name: str,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> bool:
        r"""Check the existence of a particular index name in the database.

        :param table_name: the name of the table the index belongs to
        :param index_name: the name of the index to check
        :param schema: schema name to query, if not the default schema.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        .. versionadded:: 2.0

        """
        with self._operation_context() as conn:
            return self.dialect.has_index(
                conn,
                table_name,
                index_name,
                schema,
                info_cache=self.info_cache,
                **kw,
            )

    def has_schema(self, schema_name: str, **kw: Any) -> bool:
        r"""Return True if the backend has a schema with the given name.

        :param schema_name: name of the schema to check
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        .. versionadded:: 2.0

        """
        with self._operation_context() as conn:
            return self.dialect.has_schema(
                conn, schema_name, info_cache=self.info_cache, **kw
            )

    def get_sorted_table_and_fkc_names(
        self,
        schema: Optional[str] = None,
        **kw: Any,
    ) -> List[Tuple[Optional[str], List[Tuple[str, Optional[str]]]]]:
        r"""Return dependency-sorted table and foreign key constraint names in
        referred to within a particular schema.

        This will yield 2-tuples of
        ``(tablename, [(tname, fkname), (tname, fkname), ...])``
        consisting of table names in CREATE order grouped with the foreign key
        constraint names that are not detected as belonging to a cycle.
        The final element
        will be ``(None, [(tname, fkname), (tname, fkname), ..])``
        which will consist of remaining
        foreign key constraint names that would require a separate CREATE
        step after-the-fact, based on dependencies between tables.

        :param schema: schema name to query, if not the default schema.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        .. seealso::

            :meth:`_reflection.Inspector.get_table_names`

            :func:`.sort_tables_and_constraints` - similar method which works
            with an already-given :class:`_schema.MetaData`.

        """

        return [
            (
                table_key[1] if table_key else None,
                [(tname, fks) for (_, tname), fks in fk_collection],
            )
            for (
                table_key,
                fk_collection,
            ) in self.sort_tables_on_foreign_key_dependency(
                consider_schemas=(schema,)
            )
        ]

    def sort_tables_on_foreign_key_dependency(
        self,
        consider_schemas: Collection[Optional[str]] = (None,),
        **kw: Any,
    ) -> List[
        Tuple[
            Optional[Tuple[Optional[str], str]],
            List[Tuple[Tuple[Optional[str], str], Optional[str]]],
        ]
    ]:
        r"""Return dependency-sorted table and foreign key constraint names
        referred to within multiple schemas.

        This method may be compared to
        :meth:`.Inspector.get_sorted_table_and_fkc_names`, which
        works on one schema at a time; here, the method is a generalization
        that will consider multiple schemas at once including that it will
        resolve for cross-schema foreign keys.

        .. versionadded:: 2.0

        """
        SchemaTab = Tuple[Optional[str], str]

        tuples: Set[Tuple[SchemaTab, SchemaTab]] = set()
        remaining_fkcs: Set[Tuple[SchemaTab, Optional[str]]] = set()
        fknames_for_table: Dict[SchemaTab, Set[Optional[str]]] = {}
        tnames: List[SchemaTab] = []

        for schname in consider_schemas:
            schema_fkeys = self.get_multi_foreign_keys(schname, **kw)
            tnames.extend(schema_fkeys)
            for (_, tname), fkeys in schema_fkeys.items():
                fknames_for_table[(schname, tname)] = {
                    fk["name"] for fk in fkeys
                }
                for fkey in fkeys:
                    if (
                        tname != fkey["referred_table"]
                        or schname != fkey["referred_schema"]
                    ):
                        tuples.add(
                            (
                                (
                                    fkey["referred_schema"],
                                    fkey["referred_table"],
                                ),
                                (schname, tname),
                            )
                        )
        try:
            candidate_sort = list(topological.sort(tuples, tnames))
        except exc.CircularDependencyError as err:
            edge: Tuple[SchemaTab, SchemaTab]
            for edge in err.edges:
                tuples.remove(edge)
                remaining_fkcs.update(
                    (edge[1], fkc) for fkc in fknames_for_table[edge[1]]
                )

            candidate_sort = list(topological.sort(tuples, tnames))
        ret: List[
            Tuple[Optional[SchemaTab], List[Tuple[SchemaTab, Optional[str]]]]
        ]
        ret = [
            (
                (schname, tname),
                [
                    ((schname, tname), fk)
                    for fk in fknames_for_table[(schname, tname)].difference(
                        name for _, name in remaining_fkcs
                    )
                ],
            )
            for (schname, tname) in candidate_sort
        ]
        return ret + [(None, list(remaining_fkcs))]

    def get_temp_table_names(self, **kw: Any) -> List[str]:
        r"""Return a list of temporary table names for the current bind.

        This method is unsupported by most dialects; currently
        only Oracle Database, PostgreSQL and SQLite implements it.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        """

        with self._operation_context() as conn:
            return self.dialect.get_temp_table_names(
                conn, info_cache=self.info_cache, **kw
            )

    def get_temp_view_names(self, **kw: Any) -> List[str]:
        r"""Return a list of temporary view names for the current bind.

        This method is unsupported by most dialects; currently
        only PostgreSQL and SQLite implements it.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        """
        with self._operation_context() as conn:
            return self.dialect.get_temp_view_names(
                conn, info_cache=self.info_cache, **kw
            )

    def get_table_options(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> Dict[str, Any]:
        r"""Return a dictionary of options specified when the table of the
        given name was created.

        This currently includes some options that apply to MySQL and Oracle
        Database tables.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dict with the table options. The returned keys depend on the
         dialect in use. Each one is prefixed with the dialect name.

        .. seealso:: :meth:`Inspector.get_multi_table_options`

        """
        with self._operation_context() as conn:
            return self.dialect.get_table_options(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )

    def get_multi_table_options(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Sequence[str]] = None,
        kind: ObjectKind = ObjectKind.TABLE,
        scope: ObjectScope = ObjectScope.DEFAULT,
        **kw: Any,
    ) -> Dict[TableKey, Dict[str, Any]]:
        r"""Return a dictionary of options specified when the tables in the
        given schema were created.

        The tables can be filtered by passing the names to use to
        ``filter_names``.

        This currently includes some options that apply to MySQL and Oracle
        tables.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param filter_names: optionally return information only for the
         objects listed here.

        :param kind: a :class:`.ObjectKind` that specifies the type of objects
         to reflect. Defaults to ``ObjectKind.TABLE``.

        :param scope: a :class:`.ObjectScope` that specifies if options of
         default, temporary or any tables should be reflected.
         Defaults to ``ObjectScope.DEFAULT``.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary where the keys are two-tuple schema,table-name
         and the values are dictionaries with the table options.
         The returned keys in each dict depend on the
         dialect in use. Each one is prefixed with the dialect name.
         The schema is ``None`` if no schema is provided.

        .. versionadded:: 2.0

        .. seealso:: :meth:`Inspector.get_table_options`
        """
        with self._operation_context() as conn:
            res = self.dialect.get_multi_table_options(
                conn,
                schema=schema,
                filter_names=filter_names,
                kind=kind,
                scope=scope,
                info_cache=self.info_cache,
                **kw,
            )
            return dict(res)

    def get_view_names(
        self, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        r"""Return all non-materialized view names in `schema`.

        :param schema: Optional, retrieve names from a non-default schema.
         For special quoting, use :class:`.quoted_name`.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.


        .. versionchanged:: 2.0  For those dialects that previously included
           the names of materialized views in this list (currently PostgreSQL),
           this method no longer returns the names of materialized views.
           the :meth:`.Inspector.get_materialized_view_names` method should
           be used instead.

        .. seealso::

            :meth:`.Inspector.get_materialized_view_names`

        """

        with self._operation_context() as conn:
            return self.dialect.get_view_names(
                conn, schema, info_cache=self.info_cache, **kw
            )

    def get_materialized_view_names(
        self, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        r"""Return all materialized view names in `schema`.

        :param schema: Optional, retrieve names from a non-default schema.
         For special quoting, use :class:`.quoted_name`.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        .. versionadded:: 2.0

        .. seealso::

            :meth:`.Inspector.get_view_names`

        """

        with self._operation_context() as conn:
            return self.dialect.get_materialized_view_names(
                conn, schema, info_cache=self.info_cache, **kw
            )

    def get_sequence_names(
        self, schema: Optional[str] = None, **kw: Any
    ) -> List[str]:
        r"""Return all sequence names in `schema`.

        :param schema: Optional, retrieve names from a non-default schema.
         For special quoting, use :class:`.quoted_name`.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        """

        with self._operation_context() as conn:
            return self.dialect.get_sequence_names(
                conn, schema, info_cache=self.info_cache, **kw
            )

    def get_view_definition(
        self, view_name: str, schema: Optional[str] = None, **kw: Any
    ) -> str:
        r"""Return definition for the plain or materialized view called
        ``view_name``.

        :param view_name: Name of the view.
        :param schema: Optional, retrieve names from a non-default schema.
         For special quoting, use :class:`.quoted_name`.
        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        """

        with self._operation_context() as conn:
            return self.dialect.get_view_definition(
                conn, view_name, schema, info_cache=self.info_cache, **kw
            )

    def get_columns(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> List[ReflectedColumn]:
        r"""Return information about columns in ``table_name``.

        Given a string ``table_name`` and an optional string ``schema``,
        return column information as a list of :class:`.ReflectedColumn`.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: list of dictionaries, each representing the definition of
         a database column.

        .. seealso:: :meth:`Inspector.get_multi_columns`.

        """

        with self._operation_context() as conn:
            col_defs = self.dialect.get_columns(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )
        if col_defs:
            self._instantiate_types([col_defs])
        return col_defs

    def _instantiate_types(
        self, data: Iterable[List[ReflectedColumn]]
    ) -> None:
        # make this easy and only return instances for coltype
        for col_defs in data:
            for col_def in col_defs:
                coltype = col_def["type"]
                if not isinstance(coltype, TypeEngine):
                    col_def["type"] = coltype()

    def get_multi_columns(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Sequence[str]] = None,
        kind: ObjectKind = ObjectKind.TABLE,
        scope: ObjectScope = ObjectScope.DEFAULT,
        **kw: Any,
    ) -> Dict[TableKey, List[ReflectedColumn]]:
        r"""Return information about columns in all objects in the given
        schema.

        The objects can be filtered by passing the names to use to
        ``filter_names``.

        For each table the value is a list of :class:`.ReflectedColumn`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param filter_names: optionally return information only for the
         objects listed here.

        :param kind: a :class:`.ObjectKind` that specifies the type of objects
         to reflect. Defaults to ``ObjectKind.TABLE``.

        :param scope: a :class:`.ObjectScope` that specifies if columns of
         default, temporary or any tables should be reflected.
         Defaults to ``ObjectScope.DEFAULT``.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary where the keys are two-tuple schema,table-name
         and the values are list of dictionaries, each representing the
         definition of a database column.
         The schema is ``None`` if no schema is provided.

        .. versionadded:: 2.0

        .. seealso:: :meth:`Inspector.get_columns`
        """

        with self._operation_context() as conn:
            table_col_defs = dict(
                self.dialect.get_multi_columns(
                    conn,
                    schema=schema,
                    filter_names=filter_names,
                    kind=kind,
                    scope=scope,
                    info_cache=self.info_cache,
                    **kw,
                )
            )
        self._instantiate_types(table_col_defs.values())
        return table_col_defs

    def get_pk_constraint(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> ReflectedPrimaryKeyConstraint:
        r"""Return information about primary key constraint in ``table_name``.

        Given a string ``table_name``, and an optional string `schema`, return
        primary key information as a :class:`.ReflectedPrimaryKeyConstraint`.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary representing the definition of
         a primary key constraint.

        .. seealso:: :meth:`Inspector.get_multi_pk_constraint`
        """
        with self._operation_context() as conn:
            return self.dialect.get_pk_constraint(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )

    def get_multi_pk_constraint(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Sequence[str]] = None,
        kind: ObjectKind = ObjectKind.TABLE,
        scope: ObjectScope = ObjectScope.DEFAULT,
        **kw: Any,
    ) -> Dict[TableKey, ReflectedPrimaryKeyConstraint]:
        r"""Return information about primary key constraints in
        all tables in the given schema.

        The tables can be filtered by passing the names to use to
        ``filter_names``.

        For each table the value is a :class:`.ReflectedPrimaryKeyConstraint`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param filter_names: optionally return information only for the
         objects listed here.

        :param kind: a :class:`.ObjectKind` that specifies the type of objects
         to reflect. Defaults to ``ObjectKind.TABLE``.

        :param scope: a :class:`.ObjectScope` that specifies if primary keys of
         default, temporary or any tables should be reflected.
         Defaults to ``ObjectScope.DEFAULT``.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary where the keys are two-tuple schema,table-name
         and the values are dictionaries, each representing the
         definition of a primary key constraint.
         The schema is ``None`` if no schema is provided.

        .. versionadded:: 2.0

        .. seealso:: :meth:`Inspector.get_pk_constraint`
        """
        with self._operation_context() as conn:
            return dict(
                self.dialect.get_multi_pk_constraint(
                    conn,
                    schema=schema,
                    filter_names=filter_names,
                    kind=kind,
                    scope=scope,
                    info_cache=self.info_cache,
                    **kw,
                )
            )

    def get_foreign_keys(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> List[ReflectedForeignKeyConstraint]:
        r"""Return information about foreign_keys in ``table_name``.

        Given a string ``table_name``, and an optional string `schema`, return
        foreign key information as a list of
        :class:`.ReflectedForeignKeyConstraint`.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a list of dictionaries, each representing the
         a foreign key definition.

        .. seealso:: :meth:`Inspector.get_multi_foreign_keys`
        """

        with self._operation_context() as conn:
            return self.dialect.get_foreign_keys(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )

    def get_multi_foreign_keys(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Sequence[str]] = None,
        kind: ObjectKind = ObjectKind.TABLE,
        scope: ObjectScope = ObjectScope.DEFAULT,
        **kw: Any,
    ) -> Dict[TableKey, List[ReflectedForeignKeyConstraint]]:
        r"""Return information about foreign_keys in all tables
        in the given schema.

        The tables can be filtered by passing the names to use to
        ``filter_names``.

        For each table the value is a list of
        :class:`.ReflectedForeignKeyConstraint`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param filter_names: optionally return information only for the
         objects listed here.

        :param kind: a :class:`.ObjectKind` that specifies the type of objects
         to reflect. Defaults to ``ObjectKind.TABLE``.

        :param scope: a :class:`.ObjectScope` that specifies if foreign keys of
         default, temporary or any tables should be reflected.
         Defaults to ``ObjectScope.DEFAULT``.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary where the keys are two-tuple schema,table-name
         and the values are list of dictionaries, each representing
         a foreign key definition.
         The schema is ``None`` if no schema is provided.

        .. versionadded:: 2.0

        .. seealso:: :meth:`Inspector.get_foreign_keys`
        """

        with self._operation_context() as conn:
            return dict(
                self.dialect.get_multi_foreign_keys(
                    conn,
                    schema=schema,
                    filter_names=filter_names,
                    kind=kind,
                    scope=scope,
                    info_cache=self.info_cache,
                    **kw,
                )
            )

    def get_indexes(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> List[ReflectedIndex]:
        r"""Return information about indexes in ``table_name``.

        Given a string ``table_name`` and an optional string `schema`, return
        index information as a list of :class:`.ReflectedIndex`.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a list of dictionaries, each representing the
         definition of an index.

        .. seealso:: :meth:`Inspector.get_multi_indexes`
        """

        with self._operation_context() as conn:
            return self.dialect.get_indexes(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )

    def get_multi_indexes(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Sequence[str]] = None,
        kind: ObjectKind = ObjectKind.TABLE,
        scope: ObjectScope = ObjectScope.DEFAULT,
        **kw: Any,
    ) -> Dict[TableKey, List[ReflectedIndex]]:
        r"""Return information about indexes in in all objects
        in the given schema.

        The objects can be filtered by passing the names to use to
        ``filter_names``.

        For each table the value is a list of :class:`.ReflectedIndex`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param filter_names: optionally return information only for the
         objects listed here.

        :param kind: a :class:`.ObjectKind` that specifies the type of objects
         to reflect. Defaults to ``ObjectKind.TABLE``.

        :param scope: a :class:`.ObjectScope` that specifies if indexes of
         default, temporary or any tables should be reflected.
         Defaults to ``ObjectScope.DEFAULT``.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary where the keys are two-tuple schema,table-name
         and the values are list of dictionaries, each representing the
         definition of an index.
         The schema is ``None`` if no schema is provided.

        .. versionadded:: 2.0

        .. seealso:: :meth:`Inspector.get_indexes`
        """

        with self._operation_context() as conn:
            return dict(
                self.dialect.get_multi_indexes(
                    conn,
                    schema=schema,
                    filter_names=filter_names,
                    kind=kind,
                    scope=scope,
                    info_cache=self.info_cache,
                    **kw,
                )
            )

    def get_unique_constraints(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> List[ReflectedUniqueConstraint]:
        r"""Return information about unique constraints in ``table_name``.

        Given a string ``table_name`` and an optional string `schema`, return
        unique constraint information as a list of
        :class:`.ReflectedUniqueConstraint`.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a list of dictionaries, each representing the
         definition of an unique constraint.

        .. seealso:: :meth:`Inspector.get_multi_unique_constraints`
        """

        with self._operation_context() as conn:
            return self.dialect.get_unique_constraints(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )

    def get_multi_unique_constraints(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Sequence[str]] = None,
        kind: ObjectKind = ObjectKind.TABLE,
        scope: ObjectScope = ObjectScope.DEFAULT,
        **kw: Any,
    ) -> Dict[TableKey, List[ReflectedUniqueConstraint]]:
        r"""Return information about unique constraints in all tables
        in the given schema.

        The tables can be filtered by passing the names to use to
        ``filter_names``.

        For each table the value is a list of
        :class:`.ReflectedUniqueConstraint`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param filter_names: optionally return information only for the
         objects listed here.

        :param kind: a :class:`.ObjectKind` that specifies the type of objects
         to reflect. Defaults to ``ObjectKind.TABLE``.

        :param scope: a :class:`.ObjectScope` that specifies if constraints of
         default, temporary or any tables should be reflected.
         Defaults to ``ObjectScope.DEFAULT``.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary where the keys are two-tuple schema,table-name
         and the values are list of dictionaries, each representing the
         definition of an unique constraint.
         The schema is ``None`` if no schema is provided.

        .. versionadded:: 2.0

        .. seealso:: :meth:`Inspector.get_unique_constraints`
        """

        with self._operation_context() as conn:
            return dict(
                self.dialect.get_multi_unique_constraints(
                    conn,
                    schema=schema,
                    filter_names=filter_names,
                    kind=kind,
                    scope=scope,
                    info_cache=self.info_cache,
                    **kw,
                )
            )

    def get_table_comment(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> ReflectedTableComment:
        r"""Return information about the table comment for ``table_name``.

        Given a string ``table_name`` and an optional string ``schema``,
        return table comment information as a :class:`.ReflectedTableComment`.

        Raises ``NotImplementedError`` for a dialect that does not support
        comments.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary, with the table comment.

        .. versionadded:: 1.2

        .. seealso:: :meth:`Inspector.get_multi_table_comment`
        """

        with self._operation_context() as conn:
            return self.dialect.get_table_comment(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )

    def get_multi_table_comment(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Sequence[str]] = None,
        kind: ObjectKind = ObjectKind.TABLE,
        scope: ObjectScope = ObjectScope.DEFAULT,
        **kw: Any,
    ) -> Dict[TableKey, ReflectedTableComment]:
        r"""Return information about the table comment in all objects
        in the given schema.

        The objects can be filtered by passing the names to use to
        ``filter_names``.

        For each table the value is a :class:`.ReflectedTableComment`.

        Raises ``NotImplementedError`` for a dialect that does not support
        comments.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param filter_names: optionally return information only for the
         objects listed here.

        :param kind: a :class:`.ObjectKind` that specifies the type of objects
         to reflect. Defaults to ``ObjectKind.TABLE``.

        :param scope: a :class:`.ObjectScope` that specifies if comments of
         default, temporary or any tables should be reflected.
         Defaults to ``ObjectScope.DEFAULT``.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary where the keys are two-tuple schema,table-name
         and the values are dictionaries, representing the
         table comments.
         The schema is ``None`` if no schema is provided.

        .. versionadded:: 2.0

        .. seealso:: :meth:`Inspector.get_table_comment`
        """

        with self._operation_context() as conn:
            return dict(
                self.dialect.get_multi_table_comment(
                    conn,
                    schema=schema,
                    filter_names=filter_names,
                    kind=kind,
                    scope=scope,
                    info_cache=self.info_cache,
                    **kw,
                )
            )

    def get_check_constraints(
        self, table_name: str, schema: Optional[str] = None, **kw: Any
    ) -> List[ReflectedCheckConstraint]:
        r"""Return information about check constraints in ``table_name``.

        Given a string ``table_name`` and an optional string `schema`, return
        check constraint information as a list of
        :class:`.ReflectedCheckConstraint`.

        :param table_name: string name of the table.  For special quoting,
         use :class:`.quoted_name`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a list of dictionaries, each representing the
         definition of a check constraints.

        .. seealso:: :meth:`Inspector.get_multi_check_constraints`
        """

        with self._operation_context() as conn:
            return self.dialect.get_check_constraints(
                conn, table_name, schema, info_cache=self.info_cache, **kw
            )

    def get_multi_check_constraints(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Sequence[str]] = None,
        kind: ObjectKind = ObjectKind.TABLE,
        scope: ObjectScope = ObjectScope.DEFAULT,
        **kw: Any,
    ) -> Dict[TableKey, List[ReflectedCheckConstraint]]:
        r"""Return information about check constraints in all tables
        in the given schema.

        The tables can be filtered by passing the names to use to
        ``filter_names``.

        For each table the value is a list of
        :class:`.ReflectedCheckConstraint`.

        :param schema: string schema name; if omitted, uses the default schema
         of the database connection.  For special quoting,
         use :class:`.quoted_name`.

        :param filter_names: optionally return information only for the
         objects listed here.

        :param kind: a :class:`.ObjectKind` that specifies the type of objects
         to reflect. Defaults to ``ObjectKind.TABLE``.

        :param scope: a :class:`.ObjectScope` that specifies if constraints of
         default, temporary or any tables should be reflected.
         Defaults to ``ObjectScope.DEFAULT``.

        :param \**kw: Additional keyword argument to pass to the dialect
         specific implementation. See the documentation of the dialect
         in use for more information.

        :return: a dictionary where the keys are two-tuple schema,table-name
         and the values are list of dictionaries, each representing the
         definition of a check constraints.
         The schema is ``None`` if no schema is provided.

        .. versionadded:: 2.0

        .. seealso:: :meth:`Inspector.get_check_constraints`
        """

        with self._operation_context() as conn:
            return dict(
                self.dialect.get_multi_check_constraints(
                    conn,
                    schema=schema,
                    filter_names=filter_names,
                    kind=kind,
                    scope=scope,
                    info_cache=self.info_cache,
                    **kw,
                )
            )

    def reflect_table(
        self,
        table: sa_schema.Table,
        include_columns: Optional[Collection[str]],
        exclude_columns: Collection[str] = (),
        resolve_fks: bool = True,
        _extend_on: Optional[Set[sa_schema.Table]] = None,
        _reflect_info: Optional[_ReflectionInfo] = None,
    ) -> None:
        """Given a :class:`_schema.Table` object, load its internal
        constructs based on introspection.

        This is the underlying method used by most dialects to produce
        table reflection.  Direct usage is like::

            from sqlalchemy import create_engine, MetaData, Table
            from sqlalchemy import inspect

            engine = create_engine("...")
            meta = MetaData()
            user_table = Table("user", meta)
            insp = inspect(engine)
            insp.reflect_table(user_table, None)

        .. versionchanged:: 1.4 Renamed from ``reflecttable`` to
           ``reflect_table``

        :param table: a :class:`~sqlalchemy.schema.Table` instance.
        :param include_columns: a list of string column names to include
          in the reflection process.  If ``None``, all columns are reflected.

        """

        if _extend_on is not None:
            if table in _extend_on:
                return
            else:
                _extend_on.add(table)

        dialect = self.bind.dialect

        with self._operation_context() as conn:
            schema = conn.schema_for_object(table)

        table_name = table.name

        # get table-level arguments that are specifically
        # intended for reflection, e.g. oracle_resolve_synonyms.
        # these are unconditionally passed to related Table
        # objects
        reflection_options = {
            k: table.dialect_kwargs.get(k)
            for k in dialect.reflection_options
            if k in table.dialect_kwargs
        }

        table_key = (schema, table_name)
        if _reflect_info is None or table_key not in _reflect_info.columns:
            _reflect_info = self._get_reflection_info(
                schema,
                filter_names=[table_name],
                kind=ObjectKind.ANY,
                scope=ObjectScope.ANY,
                _reflect_info=_reflect_info,
                **table.dialect_kwargs,
            )
        if table_key in _reflect_info.unreflectable:
            raise _reflect_info.unreflectable[table_key]

        if table_key not in _reflect_info.columns:
            raise exc.NoSuchTableError(table_name)

        # reflect table options, like mysql_engine
        if _reflect_info.table_options:
            tbl_opts = _reflect_info.table_options.get(table_key)
            if tbl_opts:
                # add additional kwargs to the Table if the dialect
                # returned them
                table._validate_dialect_kwargs(tbl_opts)

        found_table = False
        cols_by_orig_name: Dict[str, sa_schema.Column[Any]] = {}

        for col_d in _reflect_info.columns[table_key]:
            found_table = True

            self._reflect_column(
                table,
                col_d,
                include_columns,
                exclude_columns,
                cols_by_orig_name,
            )

        # NOTE: support tables/views with no columns
        if not found_table and not self.has_table(table_name, schema):
            raise exc.NoSuchTableError(table_name)

        self._reflect_pk(
            _reflect_info, table_key, table, cols_by_orig_name, exclude_columns
        )

        self._reflect_fk(
            _reflect_info,
            table_key,
            table,
            cols_by_orig_name,
            include_columns,
            exclude_columns,
            resolve_fks,
            _extend_on,
            reflection_options,
        )

        self._reflect_indexes(
            _reflect_info,
            table_key,
            table,
            cols_by_orig_name,
            include_columns,
            exclude_columns,
            reflection_options,
        )

        self._reflect_unique_constraints(
            _reflect_info,
            table_key,
            table,
            cols_by_orig_name,
            include_columns,
            exclude_columns,
            reflection_options,
        )

        self._reflect_check_constraints(
            _reflect_info,
            table_key,
            table,
            cols_by_orig_name,
            include_columns,
            exclude_columns,
            reflection_options,
        )

        self._reflect_table_comment(
            _reflect_info,
            table_key,
            table,
            reflection_options,
        )

    def _reflect_column(
        self,
        table: sa_schema.Table,
        col_d: ReflectedColumn,
        include_columns: Optional[Collection[str]],
        exclude_columns: Collection[str],
        cols_by_orig_name: Dict[str, sa_schema.Column[Any]],
    ) -> None:
        orig_name = col_d["name"]

        table.metadata.dispatch.column_reflect(self, table, col_d)
        table.dispatch.column_reflect(self, table, col_d)

        # fetch name again as column_reflect is allowed to
        # change it
        name = col_d["name"]
        if (include_columns and name not in include_columns) or (
            exclude_columns and name in exclude_columns
        ):
            return

        coltype = col_d["type"]

        col_kw = {
            k: col_d[k]  # type: ignore[literal-required]
            for k in [
                "nullable",
                "autoincrement",
                "quote",
                "info",
                "key",
                "comment",
            ]
            if k in col_d
        }

        if "dialect_options" in col_d:
            col_kw.update(col_d["dialect_options"])

        colargs = []
        default: Any
        if col_d.get("default") is not None:
            default_text = col_d["default"]
            assert default_text is not None
            if isinstance(default_text, TextClause):
                default = sa_schema.DefaultClause(
                    default_text, _reflected=True
                )
            elif not isinstance(default_text, sa_schema.FetchedValue):
                default = sa_schema.DefaultClause(
                    sql.text(default_text), _reflected=True
                )
            else:
                default = default_text
            colargs.append(default)

        if "computed" in col_d:
            computed = sa_schema.Computed(**col_d["computed"])
            colargs.append(computed)

        if "identity" in col_d:
            identity = sa_schema.Identity(**col_d["identity"])
            colargs.append(identity)

        cols_by_orig_name[orig_name] = col = sa_schema.Column(
            name, coltype, *colargs, **col_kw
        )

        if col.key in table.primary_key:
            col.primary_key = True
        table.append_column(col, replace_existing=True)

    def _reflect_pk(
        self,
        _reflect_info: _ReflectionInfo,
        table_key: TableKey,
        table: sa_schema.Table,
        cols_by_orig_name: Dict[str, sa_schema.Column[Any]],
        exclude_columns: Collection[str],
    ) -> None:
        pk_cons = _reflect_info.pk_constraint.get(table_key)
        if pk_cons:
            pk_cols = [
                cols_by_orig_name[pk]
                for pk in pk_cons["constrained_columns"]
                if pk in cols_by_orig_name and pk not in exclude_columns
            ]

            # update pk constraint name, comment and dialect_kwargs
            table.primary_key.name = pk_cons.get("name")
            table.primary_key.comment = pk_cons.get("comment", None)
            dialect_options = pk_cons.get("dialect_options")
            if dialect_options:
                table.primary_key.dialect_kwargs.update(dialect_options)

            # tell the PKConstraint to re-initialize
            # its column collection
            table.primary_key._reload(pk_cols)

    def _reflect_fk(
        self,
        _reflect_info: _ReflectionInfo,
        table_key: TableKey,
        table: sa_schema.Table,
        cols_by_orig_name: Dict[str, sa_schema.Column[Any]],
        include_columns: Optional[Collection[str]],
        exclude_columns: Collection[str],
        resolve_fks: bool,
        _extend_on: Optional[Set[sa_schema.Table]],
        reflection_options: Dict[str, Any],
    ) -> None:
        fkeys = _reflect_info.foreign_keys.get(table_key, [])
        for fkey_d in fkeys:
            conname = fkey_d["name"]
            # look for columns by orig name in cols_by_orig_name,
            # but support columns that are in-Python only as fallback
            constrained_columns = [
                cols_by_orig_name[c].key if c in cols_by_orig_name else c
                for c in fkey_d["constrained_columns"]
            ]

            if (
                exclude_columns
                and set(constrained_columns).intersection(exclude_columns)
                or (
                    include_columns
                    and set(constrained_columns).difference(include_columns)
                )
            ):
                continue

            referred_schema = fkey_d["referred_schema"]
            referred_table = fkey_d["referred_table"]
            referred_columns = fkey_d["referred_columns"]
            refspec = []
            if referred_schema is not None:
                if resolve_fks:
                    sa_schema.Table(
                        referred_table,
                        table.metadata,
                        schema=referred_schema,
                        autoload_with=self.bind,
                        _extend_on=_extend_on,
                        _reflect_info=_reflect_info,
                        **reflection_options,
                    )
                for column in referred_columns:
                    refspec.append(
                        ".".join([referred_schema, referred_table, column])
                    )
            else:
                if resolve_fks:
                    sa_schema.Table(
                        referred_table,
                        table.metadata,
                        autoload_with=self.bind,
                        schema=sa_schema.BLANK_SCHEMA,
                        _extend_on=_extend_on,
                        _reflect_info=_reflect_info,
                        **reflection_options,
                    )
                for column in referred_columns:
                    refspec.append(".".join([referred_table, column]))
            if "options" in fkey_d:
                options = fkey_d["options"]
            else:
                options = {}

            try:
                table.append_constraint(
                    sa_schema.ForeignKeyConstraint(
                        constrained_columns,
                        refspec,
                        conname,
                        link_to_name=True,
                        comment=fkey_d.get("comment"),
                        **options,
                    )
                )
            except exc.ConstraintColumnNotFoundError:
                util.warn(
                    f"On reflected table {table.name}, skipping reflection of "
                    "foreign key constraint "
                    f"{conname}; one or more subject columns within "
                    f"name(s) {', '.join(constrained_columns)} are not "
                    "present in the table"
                )

    _index_sort_exprs = {
        "asc": operators.asc_op,
        "desc": operators.desc_op,
        "nulls_first": operators.nulls_first_op,
        "nulls_last": operators.nulls_last_op,
    }

    def _reflect_indexes(
        self,
        _reflect_info: _ReflectionInfo,
        table_key: TableKey,
        table: sa_schema.Table,
        cols_by_orig_name: Dict[str, sa_schema.Column[Any]],
        include_columns: Optional[Collection[str]],
        exclude_columns: Collection[str],
        reflection_options: Dict[str, Any],
    ) -> None:
        # Indexes
        indexes = _reflect_info.indexes.get(table_key, [])
        for index_d in indexes:
            name = index_d["name"]
            columns = index_d["column_names"]
            expressions = index_d.get("expressions")
            column_sorting = index_d.get("column_sorting", {})
            unique = index_d["unique"]
            flavor = index_d.get("type", "index")
            dialect_options = index_d.get("dialect_options", {})

            duplicates = index_d.get("duplicates_constraint")
            if include_columns and not set(columns).issubset(include_columns):
                continue
            if duplicates:
                continue
            # look for columns by orig name in cols_by_orig_name,
            # but support columns that are in-Python only as fallback
            idx_element: Any
            idx_elements = []
            for index, c in enumerate(columns):
                if c is None:
                    if not expressions:
                        util.warn(
                            f"Skipping {flavor} {name!r} because key "
                            f"{index + 1} reflected as None but no "
                            "'expressions' were returned"
                        )
                        break
                    idx_element = sql.text(expressions[index])
                else:
                    try:
                        if c in cols_by_orig_name:
                            idx_element = cols_by_orig_name[c]
                        else:
                            idx_element = table.c[c]
                    except KeyError:
                        util.warn(
                            f"{flavor} key {c!r} was not located in "
                            f"columns for table {table.name!r}"
                        )
                        continue
                    for option in column_sorting.get(c, ()):
                        if option in self._index_sort_exprs:
                            op = self._index_sort_exprs[option]
                            idx_element = op(idx_element)
                idx_elements.append(idx_element)
            else:
                sa_schema.Index(
                    name,
                    *idx_elements,
                    _table=table,
                    unique=unique,
                    **dialect_options,
                )

    def _reflect_unique_constraints(
        self,
        _reflect_info: _ReflectionInfo,
        table_key: TableKey,
        table: sa_schema.Table,
        cols_by_orig_name: Dict[str, sa_schema.Column[Any]],
        include_columns: Optional[Collection[str]],
        exclude_columns: Collection[str],
        reflection_options: Dict[str, Any],
    ) -> None:
        constraints = _reflect_info.unique_constraints.get(table_key, [])
        # Unique Constraints
        for const_d in constraints:
            conname = const_d["name"]
            columns = const_d["column_names"]
            comment = const_d.get("comment")
            duplicates = const_d.get("duplicates_index")
            dialect_options = const_d.get("dialect_options", {})
            if include_columns and not set(columns).issubset(include_columns):
                continue
            if duplicates:
                continue
            # look for columns by orig name in cols_by_orig_name,
            # but support columns that are in-Python only as fallback
            constrained_cols = []
            for c in columns:
                try:
                    constrained_col = (
                        cols_by_orig_name[c]
                        if c in cols_by_orig_name
                        else table.c[c]
                    )
                except KeyError:
                    util.warn(
                        "unique constraint key '%s' was not located in "
                        "columns for table '%s'" % (c, table.name)
                    )
                else:
                    constrained_cols.append(constrained_col)
            table.append_constraint(
                sa_schema.UniqueConstraint(
                    *constrained_cols,
                    name=conname,
                    comment=comment,
                    **dialect_options,
                )
            )

    def _reflect_check_constraints(
        self,
        _reflect_info: _ReflectionInfo,
        table_key: TableKey,
        table: sa_schema.Table,
        cols_by_orig_name: Dict[str, sa_schema.Column[Any]],
        include_columns: Optional[Collection[str]],
        exclude_columns: Collection[str],
        reflection_options: Dict[str, Any],
    ) -> None:
        constraints = _reflect_info.check_constraints.get(table_key, [])
        for const_d in constraints:
            table.append_constraint(sa_schema.CheckConstraint(**const_d))

    def _reflect_table_comment(
        self,
        _reflect_info: _ReflectionInfo,
        table_key: TableKey,
        table: sa_schema.Table,
        reflection_options: Dict[str, Any],
    ) -> None:
        comment_dict = _reflect_info.table_comment.get(table_key)
        if comment_dict:
            table.comment = comment_dict["text"]

    def _get_reflection_info(
        self,
        schema: Optional[str] = None,
        filter_names: Optional[Collection[str]] = None,
        available: Optional[Collection[str]] = None,
        _reflect_info: Optional[_ReflectionInfo] = None,
        **kw: Any,
    ) -> _ReflectionInfo:
        kw["schema"] = schema

        if filter_names and available and len(filter_names) > 100:
            fraction = len(filter_names) / len(available)
        else:
            fraction = None

        unreflectable: Dict[TableKey, exc.UnreflectableTableError]
        kw["unreflectable"] = unreflectable = {}

        has_result: bool = True

        def run(
            meth: Any,
            *,
            optional: bool = False,
            check_filter_names_from_meth: bool = False,
        ) -> Any:
            nonlocal has_result
            # simple heuristic to improve reflection performance if a
            # dialect implements multi_reflection:
            # if more than 50% of the tables in the db are in filter_names
            # load all the tables, since it's most likely faster to avoid
            # a filter on that many tables.
            if (
                fraction is None
                or fraction <= 0.5
                or not self.dialect._overrides_default(meth.__name__)
            ):
                _fn = filter_names
            else:
                _fn = None
            try:
                if has_result:
                    res = meth(filter_names=_fn, **kw)
                    if check_filter_names_from_meth and not res:
                        # method returned no result data.
                        # skip any future call methods
                        has_result = False
                else:
                    res = {}
            except NotImplementedError:
                if not optional:
                    raise
                res = {}
            return res

        info = _ReflectionInfo(
            columns=run(
                self.get_multi_columns, check_filter_names_from_meth=True
            ),
            pk_constraint=run(self.get_multi_pk_constraint),
            foreign_keys=run(self.get_multi_foreign_keys),
            indexes=run(self.get_multi_indexes),
            unique_constraints=run(
                self.get_multi_unique_constraints, optional=True
            ),
            table_comment=run(self.get_multi_table_comment, optional=True),
            check_constraints=run(
                self.get_multi_check_constraints, optional=True
            ),
            table_options=run(self.get_multi_table_options, optional=True),
            unreflectable=unreflectable,
        )
        if _reflect_info:
            _reflect_info.update(info)
            return _reflect_info
        else:
            return info


@final
class ReflectionDefaults:
    """provides blank default values for reflection methods."""

    @classmethod
    def columns(cls) -> List[ReflectedColumn]:
        return []

    @classmethod
    def pk_constraint(cls) -> ReflectedPrimaryKeyConstraint:
        return {
            "name": None,
            "constrained_columns": [],
        }

    @classmethod
    def foreign_keys(cls) -> List[ReflectedForeignKeyConstraint]:
        return []

    @classmethod
    def indexes(cls) -> List[ReflectedIndex]:
        return []

    @classmethod
    def unique_constraints(cls) -> List[ReflectedUniqueConstraint]:
        return []

    @classmethod
    def check_constraints(cls) -> List[ReflectedCheckConstraint]:
        return []

    @classmethod
    def table_options(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def table_comment(cls) -> ReflectedTableComment:
        return {"text": None}


@dataclass
class _ReflectionInfo:
    columns: Dict[TableKey, List[ReflectedColumn]]
    pk_constraint: Dict[TableKey, Optional[ReflectedPrimaryKeyConstraint]]
    foreign_keys: Dict[TableKey, List[ReflectedForeignKeyConstraint]]
    indexes: Dict[TableKey, List[ReflectedIndex]]
    # optionals
    unique_constraints: Dict[TableKey, List[ReflectedUniqueConstraint]]
    table_comment: Dict[TableKey, Optional[ReflectedTableComment]]
    check_constraints: Dict[TableKey, List[ReflectedCheckConstraint]]
    table_options: Dict[TableKey, Dict[str, Any]]
    unreflectable: Dict[TableKey, exc.UnreflectableTableError]

    def update(self, other: _ReflectionInfo) -> None:
        for k, v in self.__dict__.items():
            ov = getattr(other, k)
            if ov is not None:
                if v is None:
                    setattr(self, k, ov)
                else:
                    v.update(ov)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\autolev\_listener_autolev_antlr.py ===
import collections
import warnings

from sympy.external import import_module

autolevparser = import_module('sympy.parsing.autolev._antlr.autolevparser',
                              import_kwargs={'fromlist': ['AutolevParser']})
autolevlexer = import_module('sympy.parsing.autolev._antlr.autolevlexer',
                             import_kwargs={'fromlist': ['AutolevLexer']})
autolevlistener = import_module('sympy.parsing.autolev._antlr.autolevlistener',
                                import_kwargs={'fromlist': ['AutolevListener']})

AutolevParser = getattr(autolevparser, 'AutolevParser', None)
AutolevLexer = getattr(autolevlexer, 'AutolevLexer', None)
AutolevListener = getattr(autolevlistener, 'AutolevListener', None)


def strfunc(z):
    if z == 0:
        return ""
    elif z == 1:
        return "_d"
    else:
        return "_" + "d" * z

def declare_phy_entities(self, ctx, phy_type, i, j=None):
    if phy_type in ("frame", "newtonian"):
        declare_frames(self, ctx, i, j)
    elif phy_type == "particle":
        declare_particles(self, ctx, i, j)
    elif phy_type == "point":
        declare_points(self, ctx, i, j)
    elif phy_type == "bodies":
        declare_bodies(self, ctx, i, j)

def declare_frames(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()
    name2 = "frame_" + name1
    if self.getValue(ctx.parentCtx.varType()) == "newtonian":
        self.newtonian = name2

    self.symbol_table2.update({name1: name2})

    self.symbol_table.update({name1 + "1>": name2 + ".x"})
    self.symbol_table.update({name1 + "2>": name2 + ".y"})
    self.symbol_table.update({name1 + "3>": name2 + ".z"})

    self.type2.update({name1: "frame"})
    self.write(name2 + " = " + "_me.ReferenceFrame('" + name1 + "')\n")

def declare_points(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "point_" + name1

    self.symbol_table2.update({name1: name2})
    self.type2.update({name1: "point"})
    self.write(name2 + " = " + "_me.Point('" + name1 + "')\n")

def declare_particles(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "particle_" + name1

    self.symbol_table2.update({name1: name2})
    self.type2.update({name1: "particle"})
    self.bodies.update({name1: name2})
    self.write(name2 + " = " + "_me.Particle('" + name1 + "', " + "_me.Point('" +
                name1 + "_pt" + "'), " + "_sm.Symbol('m'))\n")

def declare_bodies(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "body_" + name1
    self.bodies.update({name1: name2})
    masscenter = name2 + "_cm"
    refFrame = name2 + "_f"

    self.symbol_table2.update({name1: name2})
    self.symbol_table2.update({name1 + "o": masscenter})
    self.symbol_table.update({name1 + "1>": refFrame+".x"})
    self.symbol_table.update({name1 + "2>": refFrame+".y"})
    self.symbol_table.update({name1 + "3>": refFrame+".z"})

    self.type2.update({name1: "bodies"})
    self.type2.update({name1+"o": "point"})

    self.write(masscenter + " = " + "_me.Point('" + name1 + "_cm" + "')\n")
    if self.newtonian:
        self.write(masscenter + ".set_vel(" + self.newtonian + ", " + "0)\n")
    self.write(refFrame + " = " + "_me.ReferenceFrame('" + name1 + "_f" + "')\n")
    # We set a dummy mass and inertia here.
    # They will be reset using the setters later in the code anyway.
    self.write(name2 + " = " + "_me.RigidBody('" + name1 + "', " + masscenter + ", " +
                refFrame + ", " + "_sm.symbols('m'), (_me.outer(" + refFrame +
                ".x," + refFrame + ".x)," + masscenter + "))\n")

def inertia_func(self, v1, v2, l, frame):

    if self.type2[v1] == "particle":
        l.append("_me.inertia_of_point_mass(" + self.bodies[v1] + ".mass, " + self.bodies[v1] +
                 ".point.pos_from(" + self.symbol_table2[v2] + "), " + frame + ")")

    elif self.type2[v1] == "bodies":
        # Inertia has been defined about center of mass.
        if self.inertia_point[v1] == v1 + "o":
            # Asking point is cm as well
            if v2 == self.inertia_point[v1]:
                l.append(self.symbol_table2[v1] + ".inertia[0]")

            # Asking point is not cm
            else:
                l.append(self.bodies[v1] + ".inertia[0]" + " + " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[v2] +
                         "), " + frame + ")")

        # Inertia has been defined about another point
        else:
            # Asking point is the defined point
            if v2 == self.inertia_point[v1]:
                l.append(self.symbol_table2[v1] + ".inertia[0]")
            # Asking point is cm
            elif v2 == v1 + "o":
                l.append(self.bodies[v1] + ".inertia[0]" + " - " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[self.inertia_point[v1]] +
                         "), " + frame + ")")
            # Asking point is some other point
            else:
                l.append(self.bodies[v1] + ".inertia[0]" + " - " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[self.inertia_point[v1]] +
                         "), " + frame + ")" + " + " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[v2] +
                         "), " + frame + ")")


def processConstants(self, ctx):
    # Process constant declarations of the type: Constants F = 3, g = 9.81
    name = ctx.ID().getText().lower()
    if "=" in ctx.getText():
        self.symbol_table.update({name: name})
        # self.inputs.update({self.symbol_table[name]: self.getValue(ctx.getChild(2))})
        self.write(self.symbol_table[name] + " = " + "_sm.S(" + self.getValue(ctx.getChild(2)) + ")\n")
        self.type.update({name: "constants"})
        return

    # Constants declarations of the type: Constants A, B
    else:
        if "{" not in ctx.getText():
            self.symbol_table[name] = name
            self.type[name] = "constants"

    # Process constant declarations of the type: Constants C+, D-
    if ctx.getChildCount() == 2:
        # This is set for declaring nonpositive=True and nonnegative=True
        if ctx.getChild(1).getText() == "+":
            self.sign[name] = "+"
        elif ctx.getChild(1).getText() == "-":
            self.sign[name] = "-"
    else:
        if "{" not in ctx.getText():
            self.sign[name] = "o"

    # Process constant declarations of the type: Constants K{4}, a{1:2, 1:2}, b{1:2}
    if "{" in ctx.getText():
        if ":" in ctx.getText():
            num1 = int(ctx.INT(0).getText())
            num2 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1

        if ":" in ctx.getText():
            if "," in ctx.getText():
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
                for i in range(num1, num2):
                    for j in range(num3, num4):
                        self.symbol_table[name + str(i) + str(j)] = name + str(i) + str(j)
                        self.type[name + str(i) + str(j)] = "constants"
                        self.var_list.append(name + str(i) + str(j))
                        self.sign[name + str(i) + str(j)] = "o"
            else:
                for i in range(num1, num2):
                    self.symbol_table[name + str(i)] = name + str(i)
                    self.type[name + str(i)] = "constants"
                    self.var_list.append(name + str(i))
                    self.sign[name + str(i)] = "o"

        elif "," in ctx.getText():
            for i in range(1, int(ctx.INT(0).getText()) + 1):
                for j in range(1, int(ctx.INT(1).getText()) + 1):
                    self.symbol_table[name] = name + str(i) + str(j)
                    self.type[name + str(i) + str(j)] = "constants"
                    self.var_list.append(name + str(i) + str(j))
                    self.sign[name + str(i) + str(j)] = "o"

        else:
            for i in range(num1, num2):
                self.symbol_table[name + str(i)] = name + str(i)
                self.type[name + str(i)] = "constants"
                self.var_list.append(name + str(i))
                self.sign[name + str(i)] = "o"

    if "{" not in ctx.getText():
        self.var_list.append(name)


def writeConstants(self, ctx):
    l1 = list(filter(lambda x: self.sign[x] == "o", self.var_list))
    l2 = list(filter(lambda x: self.sign[x] == "+", self.var_list))
    l3 = list(filter(lambda x: self.sign[x] == "-", self.var_list))
    try:
        if self.settings["complex"] == "on":
            real = ", real=True"
        elif self.settings["complex"] == "off":
            real = ""
    except Exception:
        real = ", real=True"

    if l1:
        a = ", ".join(l1) + " = " + "_sm.symbols(" + "'" +\
            " ".join(l1) + "'" + real + ")\n"
        self.write(a)
    if l2:
        a = ", ".join(l2) + " = " + "_sm.symbols(" + "'" +\
            " ".join(l2) + "'" + real + ", nonnegative=True)\n"
        self.write(a)
    if l3:
        a = ", ".join(l3) + " = " + "_sm.symbols(" + "'" + \
            " ".join(l3) + "'" + real + ", nonpositive=True)\n"
        self.write(a)
    self.var_list = []


def processVariables(self, ctx):
    # Specified F = x*N1> + y*N2>
    name = ctx.ID().getText().lower()
    if "=" in ctx.getText():
        text = name + "'"*(ctx.getChildCount()-3)
        self.write(text + " = " + self.getValue(ctx.expr()) + "\n")
        return

    # Process variables of the type: Variables qA, qB
    if ctx.getChildCount() == 1:
        self.symbol_table[name] = name
        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
            self.type.update({name: self.getValue(ctx.parentCtx.getChild(0))})

        self.var_list.append(name)
        self.sign[name] = 0

    # Process variables of the type: Variables x', y''
    elif "'" in ctx.getText() and "{" not in ctx.getText():
        if ctx.getText().count("'") > self.maxDegree:
            self.maxDegree = ctx.getText().count("'")
        for i in range(ctx.getChildCount()):
            self.sign[name + strfunc(i)] = i
            self.symbol_table[name + "'"*i] = name + strfunc(i)
            if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                self.type.update({name + "'"*i: self.getValue(ctx.parentCtx.getChild(0))})
            self.var_list.append(name + strfunc(i))

    elif "{" in ctx.getText():
        # Process variables of the type: Variables x{3}, y{2}

        if "'" in ctx.getText():
            dash_count = ctx.getText().count("'")
            if dash_count > self.maxDegree:
                self.maxDegree = dash_count

        if ":" in ctx.getText():
            # Variables C{1:2, 1:2}
            if "," in ctx.getText():
                num1 = int(ctx.INT(0).getText())
                num2 = int(ctx.INT(1).getText()) + 1
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
            # Variables C{1:2}
            else:
                num1 = int(ctx.INT(0).getText())
                num2 = int(ctx.INT(1).getText()) + 1

        # Variables C{1,3}
        elif "," in ctx.getText():
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1
            num3 = 1
            num4 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1

        for i in range(num1, num2):
            try:
                for j in range(num3, num4):
                    try:
                        for z in range(dash_count+1):
                            self.symbol_table.update({name + str(i) + str(j) + "'"*z: name + str(i) + str(j) + strfunc(z)})
                            if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                                self.type.update({name + str(i) + str(j) +  "'"*z: self.getValue(ctx.parentCtx.getChild(0))})
                            self.var_list.append(name + str(i) + str(j) + strfunc(z))
                            self.sign.update({name + str(i) + str(j) + strfunc(z): z})
                            if dash_count > self.maxDegree:
                                self.maxDegree = dash_count
                    except Exception:
                        self.symbol_table.update({name + str(i) + str(j): name + str(i) + str(j)})
                        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                            self.type.update({name + str(i) + str(j): self.getValue(ctx.parentCtx.getChild(0))})
                        self.var_list.append(name + str(i) + str(j))
                        self.sign.update({name + str(i) + str(j): 0})
            except Exception:
                try:
                    for z in range(dash_count+1):
                        self.symbol_table.update({name + str(i) + "'"*z: name + str(i) + strfunc(z)})
                        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                            self.type.update({name + str(i) +  "'"*z: self.getValue(ctx.parentCtx.getChild(0))})
                        self.var_list.append(name + str(i) + strfunc(z))
                        self.sign.update({name + str(i) + strfunc(z): z})
                        if dash_count > self.maxDegree:
                            self.maxDegree = dash_count
                except Exception:
                    self.symbol_table.update({name + str(i): name + str(i)})
                    if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                        self.type.update({name + str(i): self.getValue(ctx.parentCtx.getChild(0))})
                    self.var_list.append(name + str(i))
                    self.sign.update({name + str(i): 0})

def writeVariables(self, ctx):
    #print(self.sign)
    #print(self.symbol_table)
    if self.var_list:
        for i in range(self.maxDegree+1):
            if i == 0:
                j = ""
                t = ""
            else:
                j = str(i)
                t = ", "
            l = []
            for k in list(filter(lambda x: self.sign[x] == i, self.var_list)):
                if i == 0:
                    l.append(k)
                if i == 1:
                    l.append(k[:-1])
                if i > 1:
                    l.append(k[:-2])
            a = ", ".join(list(filter(lambda x: self.sign[x] == i, self.var_list))) + " = " +\
                "_me.dynamicsymbols(" + "'" + " ".join(l) + "'" + t + j + ")\n"
            l = []
            self.write(a)
        self.maxDegree = 0
    self.var_list = []

def processImaginary(self, ctx):
    name = ctx.ID().getText().lower()
    self.symbol_table[name] = name
    self.type[name] = "imaginary"
    self.var_list.append(name)


def writeImaginary(self, ctx):
    a = ", ".join(self.var_list) + " = " + "_sm.symbols(" + "'" + \
        " ".join(self.var_list) + "')\n"
    b = ", ".join(self.var_list) + " = " + "_sm.I\n"
    self.write(a)
    self.write(b)
    self.var_list = []

if AutolevListener:
    class MyListener(AutolevListener):  # type: ignore
        def __init__(self, include_numeric=False):
            # Stores data in tree nodes(tree annotation). Especially useful for expr reconstruction.
            self.tree_property = {}

            # Stores the declared variables, constants etc as they are declared in Autolev and SymPy
            # {"<Autolev symbol>": "<SymPy symbol>"}.
            self.symbol_table = collections.OrderedDict()

            # Similar to symbol_table. Used for storing Physical entities like Frames, Points,
            # Particles, Bodies etc
            self.symbol_table2 = collections.OrderedDict()

            # Used to store nonpositive, nonnegative etc for constants and number of "'"s (order of diff)
            # in variables.
            self.sign = {}

            # Simple list used as a store to pass around variables between the 'process' and 'write'
            # methods.
            self.var_list = []

            # Stores the type of a declared variable (constants, variables, specifieds etc)
            self.type = collections.OrderedDict()

            # Similar to self.type. Used for storing the type of Physical entities like Frames, Points,
            # Particles, Bodies etc
            self.type2 = collections.OrderedDict()

            # These lists are used to distinguish matrix, numeric and vector expressions.
            self.matrix_expr = []
            self.numeric_expr = []
            self.vector_expr = []
            self.fr_expr = []

            self.output_code = []

            # Stores the variables and their rhs for substituting upon the Autolev command EXPLICIT.
            self.explicit = collections.OrderedDict()

            # Write code to import common dependencies.
            self.output_code.append("import sympy.physics.mechanics as _me\n")
            self.output_code.append("import sympy as _sm\n")
            self.output_code.append("import math as m\n")
            self.output_code.append("import numpy as _np\n")
            self.output_code.append("\n")

            # Just a store for the max degree variable in a line.
            self.maxDegree = 0

            # Stores the input parameters which are then used for codegen and numerical analysis.
            self.inputs = collections.OrderedDict()
            # Stores the variables which appear in Output Autolev commands.
            self.outputs = []
            # Stores the settings specified by the user. Ex: Complex on/off, Degrees on/off
            self.settings = {}
            # Boolean which changes the behaviour of some expression reconstruction
            # when parsing Input Autolev commands.
            self.in_inputs = False
            self.in_outputs = False

            # Stores for the physical entities.
            self.newtonian = None
            self.bodies = collections.OrderedDict()
            self.constants = []
            self.forces = collections.OrderedDict()
            self.q_ind = []
            self.q_dep = []
            self.u_ind = []
            self.u_dep = []
            self.kd_eqs = []
            self.dependent_variables = []
            self.kd_equivalents = collections.OrderedDict()
            self.kd_equivalents2 = collections.OrderedDict()
            self.kd_eqs_supplied = None
            self.kane_type = "no_args"
            self.inertia_point = collections.OrderedDict()
            self.kane_parsed = False
            self.t = False

            # PyDy ode code will be included only if this flag is set to True.
            self.include_numeric = include_numeric

        def write(self, string):
            self.output_code.append(string)

        def getValue(self, node):
            return self.tree_property[node]

        def setValue(self, node, value):
            self.tree_property[node] = value

        def getSymbolTable(self):
            return self.symbol_table

        def getType(self):
            return self.type

        def exitVarDecl(self, ctx):
            # This event method handles variable declarations. The parse tree node varDecl contains
            # one or more varDecl2 nodes. Eg varDecl for 'Constants a{1:2, 1:2}, b{1:2}' has two varDecl2
            # nodes(one for a{1:2, 1:2} and one for b{1:2}).

            # Variable declarations are processed and stored in the event method exitVarDecl2.
            # This stored information is used to write the final SymPy output code in the exitVarDecl event method.

            # determine the type of declaration
            if self.getValue(ctx.varType()) == "constant":
                writeConstants(self, ctx)
            elif self.getValue(ctx.varType()) in\
            ("variable", "motionvariable", "motionvariable'", "specified"):
                writeVariables(self, ctx)
            elif self.getValue(ctx.varType()) == "imaginary":
                writeImaginary(self, ctx)

        def exitVarType(self, ctx):
            # Annotate the varType tree node with the type of the variable declaration.
            name = ctx.getChild(0).getText().lower()
            if name[-1] == "s" and name != "bodies":
                self.setValue(ctx, name[:-1])
            else:
                self.setValue(ctx, name)

        def exitVarDecl2(self, ctx):
            # Variable declarations are processed and stored in the event method exitVarDecl2.
            # This stored information is used to write the final SymPy output code in the exitVarDecl event method.
            # This is the case for constants, variables, specifieds etc.

            # This isn't the case for all types of declarations though. For instance
            # Frames A, B, C, N cannot be defined on one line in SymPy. So we do not append A, B, C, N
            # to a var_list or use exitVarDecl. exitVarDecl2 directly writes out to the file.

            # determine the type of declaration
            if self.getValue(ctx.parentCtx.varType()) == "constant":
                processConstants(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) in \
            ("variable", "motionvariable", "motionvariable'", "specified"):
                processVariables(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) == "imaginary":
                processImaginary(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) in ("frame", "newtonian", "point", "particle", "bodies"):
                if "{" in ctx.getText():
                    if ":" in ctx.getText() and "," not in ctx.getText():
                        num1 = int(ctx.INT(0).getText())
                        num2 = int(ctx.INT(1).getText()) + 1
                    elif ":" not in ctx.getText() and "," in ctx.getText():
                        num1 = 1
                        num2 = int(ctx.INT(0).getText()) + 1
                        num3 = 1
                        num4 = int(ctx.INT(1).getText()) + 1
                    elif ":" in ctx.getText() and "," in ctx.getText():
                        num1 = int(ctx.INT(0).getText())
                        num2 = int(ctx.INT(1).getText()) + 1
                        num3 = int(ctx.INT(2).getText())
                        num4 = int(ctx.INT(3).getText()) + 1
                    else:
                        num1 = 1
                        num2 = int(ctx.INT(0).getText()) + 1
                else:
                    num1 = 1
                    num2 = 2
                for i in range(num1, num2):
                    try:
                        for j in range(num3, num4):
                            declare_phy_entities(self, ctx, self.getValue(ctx.parentCtx.varType()), i, j)
                    except Exception:
                        declare_phy_entities(self, ctx, self.getValue(ctx.parentCtx.varType()), i)
        # ================== Subrules of parser rule expr (Start) ====================== #

        def exitId(self, ctx):
            # Tree annotation for ID which is a labeled subrule of the parser rule expr.
            # A_C
            python_keywords = ["and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except",\
            "exec", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "not", "or", "pass", "print",\
            "raise", "return", "try", "while", "with", "yield"]

            if ctx.ID().getText().lower() in python_keywords:
                warnings.warn("Python keywords must not be used as identifiers. Please refer to the list of keywords at https://docs.python.org/2.5/ref/keywords.html",
                SyntaxWarning)

            if "_" in ctx.ID().getText() and ctx.ID().getText().count('_') == 1:
                e1, e2 = ctx.ID().getText().lower().split('_')
                try:
                    if self.type2[e1] == "frame":
                        e1 = self.symbol_table2[e1]
                    elif self.type2[e1] == "bodies":
                        e1 = self.symbol_table2[e1] + "_f"
                    if self.type2[e2] == "frame":
                        e2 = self.symbol_table2[e2]
                    elif self.type2[e2] == "bodies":
                        e2 = self.symbol_table2[e2] + "_f"

                    self.setValue(ctx, e1 + ".dcm(" + e2 + ")")
                except Exception:
                    self.setValue(ctx, ctx.ID().getText().lower())
            else:
                # Reserved constant Pi
                if ctx.ID().getText().lower() == "pi":
                    self.setValue(ctx, "_sm.pi")
                    self.numeric_expr.append(ctx)

                # Reserved variable T (for time)
                elif ctx.ID().getText().lower() == "t":
                    self.setValue(ctx, "_me.dynamicsymbols._t")
                    if not self.in_inputs and not self.in_outputs:
                        self.t = True

                else:
                    idText = ctx.ID().getText().lower() + "'"*(ctx.getChildCount() - 1)
                    if idText in self.type.keys() and self.type[idText] == "matrix":
                        self.matrix_expr.append(ctx)
                    if self.in_inputs:
                        try:
                            self.setValue(ctx, self.symbol_table[idText])
                        except Exception:
                            self.setValue(ctx, idText.lower())
                    else:
                        try:
                            self.setValue(ctx, self.symbol_table[idText])
                        except Exception:
                            pass

        def exitInt(self, ctx):
            # Tree annotation for int which is a labeled subrule of the parser rule expr.
            int_text = ctx.INT().getText()
            self.setValue(ctx, int_text)
            self.numeric_expr.append(ctx)

        def exitFloat(self, ctx):
            # Tree annotation for float which is a labeled subrule of the parser rule expr.
            floatText = ctx.FLOAT().getText()
            self.setValue(ctx, floatText)
            self.numeric_expr.append(ctx)

        def exitAddSub(self, ctx):
            # Tree annotation for AddSub which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr (+|-) expr
            if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, self.getValue(ctx.expr(0)) + ctx.getChild(1).getText() +
                          self.getValue(ctx.expr(1)))

        def exitMulDiv(self, ctx):
            # Tree annotation for MulDiv which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr (*|/) expr
            try:
                if ctx.expr(0) in self.vector_expr and ctx.expr(1) in self.vector_expr:
                    self.setValue(ctx, "_me.outer(" + self.getValue(ctx.expr(0)) + ", " +
                                  self.getValue(ctx.expr(1)) + ")")
                else:
                    if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                        self.matrix_expr.append(ctx)
                    if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                        self.vector_expr.append(ctx)
                    if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                        self.numeric_expr.append(ctx)
                    self.setValue(ctx, self.getValue(ctx.expr(0)) + ctx.getChild(1).getText() +
                                  self.getValue(ctx.expr(1)))
            except Exception:
                pass

        def exitNegativeOne(self, ctx):
            # Tree annotation for negativeOne which is a labeled subrule of the parser rule expr.
            self.setValue(ctx, "-1*" + self.getValue(ctx.getChild(1)))
            if ctx.getChild(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.getChild(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)

        def exitParens(self, ctx):
            # Tree annotation for parens which is a labeled subrule of the parser rule expr.
            # The subrule is expr = '(' expr ')'
            if ctx.expr() in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr() in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr() in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, "(" + self.getValue(ctx.expr()) + ")")

        def exitExponent(self, ctx):
            # Tree annotation for Exponent which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr ^ expr
            if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, self.getValue(ctx.expr(0)) + "**" + self.getValue(ctx.expr(1)))

        def exitExp(self, ctx):
            s = ctx.EXP().getText()[ctx.EXP().getText().index('E')+1:]
            if "-" in s:
                s = s[0] + s[1:].lstrip("0")
            else:
                s = s.lstrip("0")
            self.setValue(ctx, ctx.EXP().getText()[:ctx.EXP().getText().index('E')] +
                          "*10**(" + s + ")")

        def exitFunction(self, ctx):
            # Tree annotation for function which is a labeled subrule of the parser rule expr.

            # The difference between this and FunctionCall is that this is used for non standalone functions
            # appearing in expressions and assignments.
            # Eg:
            # When we come across a standalone function say Expand(E, n:m) then it is categorized as FunctionCall
            # which is a parser rule in itself under rule stat. exitFunctionCall() takes care of it and writes to the file.
            #
            # On the other hand, while we come across E_diff = D(E, y), we annotate the tree node
            # of the function D(E, y) with the SymPy equivalent in exitFunction().
            # In this case it is the method exitAssignment() that writes the code to the file and not exitFunction().

            ch = ctx.getChild(0)
            func_name = ch.getChild(0).getText().lower()

            # Expand(y, n:m) *
            if func_name == "expand":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    # _sm.Matrix([i.expand() for i in z]).reshape(z.shape[0], z.shape[1])
                    self.setValue(ctx, "_sm.Matrix([i.expand() for i in " + expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr + ")" + "." + "expand()")

            # Factor(y, x) *
            elif func_name == "factor":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([_sm.factor(i, " + self.getValue(ch.expr(1)) + ") for i in " +
                                  expr + "])" + ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "_sm.factor(" + "(" + expr + ")" +
                                  ", " + self.getValue(ch.expr(1)) + ")")

            # D(y, x)
            elif func_name == "d":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.diff(" + self.getValue(ch.expr(1)) + ") for i in " +
                                  expr + "])" + ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if ch.getChildCount() == 8:
                        frame = self.symbol_table2[ch.expr(2).getText().lower()]
                        self.setValue(ctx, "(" + expr + ")" + "." + "diff(" + self.getValue(ch.expr(1)) +
                                      ", " + frame + ")")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + "." + "diff(" +
                                      self.getValue(ch.expr(1)) + ")")

            # Dt(y)
            elif func_name == "dt":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.vector_expr:
                    text = "dt("
                else:
                    text = "diff(_sm.Symbol('t')"
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i." + text +
                                  ") for i in " + expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if ch.getChildCount() == 6:
                        frame = self.symbol_table2[ch.expr(1).getText().lower()]
                        self.setValue(ctx, "(" + expr + ")" + "." + "dt(" +
                                      frame + ")")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + "." + text + ")")

            # Explicit(EXPRESS(IMPLICIT>,C))
            elif func_name == "explicit":
                if ch.expr(0) in self.vector_expr:
                    self.vector_expr.append(ctx)
                expr = self.getValue(ch.expr(0))
                if self.explicit.keys():
                    explicit_list = []
                    for i in self.explicit.keys():
                        explicit_list.append(i + ":" + self.explicit[i])
                    self.setValue(ctx, "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})")
                else:
                    self.setValue(ctx, expr)

            # Taylor(y, 0:2, w=a, x=0)
            # TODO: Currently only works with symbols. Make it work for dynamicsymbols.
            elif func_name == "taylor":
                exp = self.getValue(ch.expr(0))
                order = self.getValue(ch.expr(1).expr(1))
                x = (ch.getChildCount()-6)//2
                l = []
                for i in range(x):
                    index = 2 + i
                    child = ch.expr(index)
                    l.append(".series(" + self.getValue(child.getChild(0)) +
                             ", " + self.getValue(child.getChild(2)) +
                             ", " + order + ").removeO()")
                self.setValue(ctx, "(" + exp + ")" + "".join(l))

            # Evaluate(y, a=x, b=2)
            elif func_name == "evaluate":
                expr = self.getValue(ch.expr(0))
                l = []
                x = (ch.getChildCount()-4)//2
                for i in range(x):
                    index = 1 + i
                    child = ch.expr(index)
                    l.append(self.getValue(child.getChild(0)) + ":" +
                             self.getValue(child.getChild(2)))

                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.subs({" + ",".join(l) + "}) for i in " +
                                  expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if self.explicit:
                        explicit_list = []
                        for i in self.explicit.keys():
                            explicit_list.append(i + ":" + self.explicit[i])
                        self.setValue(ctx, "(" + expr + ")" + ".subs({" + ",".join(explicit_list) +
                                      "}).subs({" + ",".join(l) + "})")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + ".subs({" + ",".join(l) + "})")

            # Polynomial([a, b, c], x)
            elif func_name == "polynomial":
                self.setValue(ctx, "_sm.Poly(" + self.getValue(ch.expr(0)) + ", " +
                              self.getValue(ch.expr(1)) + ")")

            # Roots(Poly, x, 2)
            # Roots([1; 2; 3; 4])
            elif func_name == "roots":
                self.matrix_expr.append(ctx)
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.setValue(ctx, "[i.evalf() for i in " + "_sm.solve(" +
                                  "_sm.Poly(" + expr + ", " + "x),x)]")
                else:
                    self.setValue(ctx, "[i.evalf() for i in " + "_sm.solve(" +
                                  expr + ", " + self.getValue(ch.expr(1)) + ")]")

            # Transpose(A), Inv(A)
            elif func_name in ("transpose", "inv", "inverse"):
                self.matrix_expr.append(ctx)
                if func_name == "transpose":
                    e = ".T"
                elif func_name in ("inv", "inverse"):
                    e = "**(-1)"
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + e)

            # Eig(A)
            elif func_name == "eig":
                # "_sm.Matrix([i.evalf() for i in " +
                self.setValue(ctx, "_sm.Matrix([i.evalf() for i in (" +
                              self.getValue(ch.expr(0)) + ").eigenvals().keys()])")

            # Diagmat(n, m, x)
            # Diagmat(3, 1)
            elif func_name == "diagmat":
                self.matrix_expr.append(ctx)
                if ch.getChildCount() == 6:
                    l = []
                    for i in range(int(self.getValue(ch.expr(0)))):
                        l.append(self.getValue(ch.expr(1)) + ",")

                    self.setValue(ctx, "_sm.diag(" + ("".join(l))[:-1] + ")")

                elif ch.getChildCount() == 8:
                    # _sm.Matrix([x if i==j else 0 for i in range(n) for j in range(m)]).reshape(n, m)
                    n = self.getValue(ch.expr(0))
                    m = self.getValue(ch.expr(1))
                    x = self.getValue(ch.expr(2))
                    self.setValue(ctx, "_sm.Matrix([" + x + " if i==j else 0 for i in range(" +
                                  n + ") for j in range(" + m + ")]).reshape(" + n + ", " + m + ")")

            # Cols(A)
            # Cols(A, 1)
            # Cols(A, 1, 2:4, 3)
            elif func_name in ("cols", "rows"):
                self.matrix_expr.append(ctx)
                if func_name == "cols":
                    e1 = ".cols"
                    e2 = ".T."
                else:
                    e1 = ".rows"
                    e2 = "."
                if ch.getChildCount() == 4:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + e1)
                elif ch.getChildCount() == 6:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" +
                                  e1[:-1] + "(" + str(int(self.getValue(ch.expr(1))) - 1) + ")")
                else:
                    l = []
                    for i in range(4, ch.getChildCount()):
                        try:
                            if ch.getChild(i).getChildCount() > 1 and ch.getChild(i).getChild(1).getText() == ":":
                                for j in range(int(ch.getChild(i).getChild(0).getText()),
                                int(ch.getChild(i).getChild(2).getText())+1):
                                    l.append("(" + self.getValue(ch.getChild(2)) + ")" + e2 +
                                             "row(" + str(j-1) + ")")
                            else:
                                l.append("(" + self.getValue(ch.getChild(2)) + ")" + e2 +
                                         "row(" + str(int(ch.getChild(i).getText())-1) + ")")
                        except Exception:
                            pass
                    self.setValue(ctx, "_sm.Matrix([" + ",".join(l) + "])")

            # Det(A) Trace(A)
            elif func_name in ["det", "trace"]:
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + "." +
                              func_name + "()")

            # Element(A, 2, 3)
            elif func_name == "element":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + "[" +
                              str(int(self.getValue(ch.expr(1)))-1) + "," +
                              str(int(self.getValue(ch.expr(2)))-1) + "]")

            elif func_name in \
            ["cos", "sin", "tan", "cosh", "sinh", "tanh", "acos", "asin", "atan",
            "log", "exp", "sqrt", "factorial", "floor", "sign"]:
                self.setValue(ctx, "_sm." + func_name + "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "ceil":
                self.setValue(ctx, "_sm.ceiling" + "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "sqr":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) +
                              ")" + "**2")

            elif func_name == "log10":
                self.setValue(ctx, "_sm.log" +
                              "(" + self.getValue(ch.expr(0)) + ", 10)")

            elif func_name == "atan2":
                self.setValue(ctx, "_sm.atan2" + "(" + self.getValue(ch.expr(0)) + ", " +
                              self.getValue(ch.expr(1)) + ")")

            elif func_name in ["int", "round"]:
                self.setValue(ctx, func_name +
                              "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "abs":
                self.setValue(ctx, "_sm.Abs(" + self.getValue(ch.expr(0)) + ")")

            elif func_name in ["max", "min"]:
                # max(x, y, z)
                l = []
                for i in range(1, ch.getChildCount()):
                    if ch.getChild(i) in self.tree_property.keys():
                        l.append(self.getValue(ch.getChild(i)))
                    elif ch.getChild(i).getText() in [",", "(", ")"]:
                        l.append(ch.getChild(i).getText())
                self.setValue(ctx, "_sm." + ch.getChild(0).getText().capitalize() + "".join(l))

            # Coef(y, x)
            elif func_name == "coef":
                #A41_A53=COEF([RHS(U4);RHS(U5)],[U1,U2,U3])
                if ch.expr(0) in self.matrix_expr and ch.expr(1) in self.matrix_expr:
                    icount = jcount = 0
                    for i in range(ch.expr(0).getChild(0).getChildCount()):
                        try:
                            ch.expr(0).getChild(0).getChild(i).getRuleIndex()
                            icount+=1
                        except Exception:
                            pass
                    for j in range(ch.expr(1).getChild(0).getChildCount()):
                        try:
                            ch.expr(1).getChild(0).getChild(j).getRuleIndex()
                            jcount+=1
                        except Exception:
                            pass
                    l = []
                    for i in range(icount):
                        for j in range(jcount):
                            # a41_a53[i,j] = u4.expand().coeff(u1)
                            l.append(self.getValue(ch.expr(0).getChild(0).expr(i)) + ".expand().coeff("
                                     + self.getValue(ch.expr(1).getChild(0).expr(j)) + ")")
                    self.setValue(ctx, "_sm.Matrix([" + ", ".join(l) + "]).reshape(" + str(icount) + ", " + str(jcount) + ")")
                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) +
                                  ")" + ".expand().coeff(" + self.getValue(ch.expr(1)) + ")")

            # Exclude(y, x) Include(y, x)
            elif func_name in ("exclude", "include"):
                if func_name == "exclude":
                    e = "0"
                else:
                    e = "1"
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.collect(" + self.getValue(ch.expr(1)) + "])" +
                                  ".coeff(" + self.getValue(ch.expr(1)) + "," + e + ")" + "for i in " + expr + ")" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr +
                                  ")" + ".collect(" + self.getValue(ch.expr(1)) + ")" +
                                  ".coeff(" + self.getValue(ch.expr(1)) + "," + e + ")")

            # RHS(y)
            elif func_name == "rhs":
                self.setValue(ctx, self.explicit[self.getValue(ch.expr(0))])

            # Arrange(y, n, x) *
            elif func_name == "arrange":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.collect(" + self.getValue(ch.expr(2)) +
                                  ")" + "for i in " + expr + "])"+
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr +
                                  ")" + ".collect(" + self.getValue(ch.expr(2)) + ")")

            # Replace(y, sin(x)=3)
            elif func_name == "replace":
                l = []
                for i in range(1, ch.getChildCount()):
                    try:
                        if ch.getChild(i).getChild(1).getText() == "=":
                            l.append(self.getValue(ch.getChild(i).getChild(0)) +
                                     ":" + self.getValue(ch.getChild(i).getChild(2)))
                    except Exception:
                        pass
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.subs({" + ",".join(l) + "}) for i in " +
                                  expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" +
                                  ".subs({" + ",".join(l) + "})")

            # Dot(Loop>, N1>)
            elif func_name == "dot":
                l = []
                num = (ch.expr(1).getChild(0).getChildCount()-1)//2
                if ch.expr(1) in self.matrix_expr:
                    for i in range(num):
                        l.append("_me.dot(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1).getChild(0).expr(i)) + ")")
                    self.setValue(ctx, "_sm.Matrix([" + ",".join(l) + "]).reshape(" + str(num) + ", " + "1)")
                else:
                    self.setValue(ctx, "_me.dot(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1)) + ")")
            # Cross(w_A_N>, P_NA_AB>)
            elif func_name == "cross":
                self.vector_expr.append(ctx)
                self.setValue(ctx, "_me.cross(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1)) + ")")

            # Mag(P_O_Q>)
            elif func_name == "mag":
                self.setValue(ctx, self.getValue(ch.expr(0)) + "." + "magnitude()")

            # MATRIX(A, I_R>>)
            elif func_name == "matrix":
                if self.type2[ch.expr(0).getText().lower()] == "frame":
                    text = ""
                elif self.type2[ch.expr(0).getText().lower()] == "bodies":
                    text = "_f"
                self.setValue(ctx, "(" + self.getValue(ch.expr(1)) + ")" + ".to_matrix(" +
                              self.symbol_table2[ch.expr(0).getText().lower()] + text + ")")

            # VECTOR(A, ROWS(EIGVECS,1))
            elif func_name == "vector":
                if self.type2[ch.expr(0).getText().lower()] == "frame":
                    text = ""
                elif self.type2[ch.expr(0).getText().lower()] == "bodies":
                    text = "_f"
                v = self.getValue(ch.expr(1))
                f = self.symbol_table2[ch.expr(0).getText().lower()] + text
                self.setValue(ctx, v + "[0]*" + f + ".x +" + v + "[1]*" + f + ".y +" +
                              v + "[2]*" + f + ".z")

            # Express(A2>, B)
            # Here I am dealing with all the Inertia commands as I expect the users to use Inertia
            # commands only with Express because SymPy needs the Reference frame to be specified unlike Autolev.
            elif func_name == "express":
                self.vector_expr.append(ctx)
                if self.type2[ch.expr(1).getText().lower()] == "frame":
                    frame = self.symbol_table2[ch.expr(1).getText().lower()]
                else:
                    frame = self.symbol_table2[ch.expr(1).getText().lower()] + "_f"
                if ch.expr(0).getText().lower() == "1>>":
                    self.setValue(ctx, "_me.inertia(" + frame + ", 1, 1, 1)")

                elif '_' in ch.expr(0).getText().lower() and ch.expr(0).getText().lower().count('_') == 2\
                and ch.expr(0).getText().lower()[0] == "i" and ch.expr(0).getText().lower()[-2:] == ">>":
                    v1 = ch.expr(0).getText().lower()[:-2].split('_')[1]
                    v2 = ch.expr(0).getText().lower()[:-2].split('_')[2]
                    l = []
                    inertia_func(self, v1, v2, l, frame)
                    self.setValue(ctx, " + ".join(l))

                elif ch.expr(0).getChild(0).getChild(0).getText().lower() == "inertia":
                    if ch.expr(0).getChild(0).getChildCount() == 4:
                        l = []
                        v2 = ch.expr(0).getChild(0).ID(0).getText().lower()
                        for v1 in self.bodies:
                            inertia_func(self, v1, v2, l, frame)
                        self.setValue(ctx, " + ".join(l))

                    else:
                        l = []
                        l2 = []
                        v2 = ch.expr(0).getChild(0).ID(0).getText().lower()
                        for i in range(1, (ch.expr(0).getChild(0).getChildCount()-2)//2):
                            l2.append(ch.expr(0).getChild(0).ID(i).getText().lower())
                        for v1 in l2:
                            inertia_func(self, v1, v2, l, frame)
                        self.setValue(ctx, " + ".join(l))

                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + ".express(" +
                                  self.symbol_table2[ch.expr(1).getText().lower()] + ")")
            # CM(P)
            elif func_name == "cm":
                if self.type2[ch.expr(0).getText().lower()] == "point":
                    text = ""
                else:
                    text = ".point"
                if ch.getChildCount() == 4:
                    self.setValue(ctx, "_me.functions.center_of_mass(" + self.symbol_table2[ch.expr(0).getText().lower()] +
                                  text + "," + ", ".join(self.bodies.values()) + ")")
                else:
                    bodies = []
                    for i in range(1, (ch.getChildCount()-1)//2):
                        bodies.append(self.symbol_table2[ch.expr(i).getText().lower()])
                    self.setValue(ctx, "_me.functions.center_of_mass(" + self.symbol_table2[ch.expr(0).getText().lower()] +
                                  text + "," + ", ".join(bodies) + ")")

            # PARTIALS(V_P1_E>,U1)
            elif func_name == "partials":
                speeds = []
                for i in range(1, (ch.getChildCount()-1)//2):
                    if self.kd_equivalents2:
                        speeds.append(self.kd_equivalents2[self.symbol_table[ch.expr(i).getText().lower()]])
                    else:
                        speeds.append(self.symbol_table[ch.expr(i).getText().lower()])
                v1, v2, v3 = ch.expr(0).getText().lower().replace(">","").split('_')
                if self.type2[v2] == "point":
                    point = self.symbol_table2[v2]
                elif self.type2[v2] == "particle":
                    point = self.symbol_table2[v2] + ".point"
                frame = self.symbol_table2[v3]
                self.setValue(ctx, point + ".partial_velocity(" + frame + ", " + ",".join(speeds) + ")")

            # UnitVec(A1>+A2>+A3>)
            elif func_name == "unitvec":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + ".normalize()")

            # Units(deg, rad)
            elif func_name == "units":
                if ch.expr(0).getText().lower() == "deg" and ch.expr(1).getText().lower() == "rad":
                    factor = 0.0174533
                elif ch.expr(0).getText().lower() == "rad" and ch.expr(1).getText().lower() == "deg":
                    factor = 57.2958
                self.setValue(ctx, str(factor))
            # Mass(A)
            elif func_name == "mass":
                l = []
                try:
                    ch.ID(0).getText().lower()
                    for i in range((ch.getChildCount()-1)//2):
                        l.append(self.symbol_table2[ch.ID(i).getText().lower()] + ".mass")
                    self.setValue(ctx, "+".join(l))
                except Exception:
                    for i in self.bodies.keys():
                        l.append(self.bodies[i] + ".mass")
                    self.setValue(ctx, "+".join(l))

            # Fr() FrStar()
            # _me.KanesMethod(n, q_ind, u_ind, kd, velocity_constraints).kanes_equations(pl, fl)[0]
            elif func_name in ["fr", "frstar"]:
                if not self.kane_parsed:
                    if self.kd_eqs:
                        for i in self.kd_eqs:
                            self.q_ind.append(self.symbol_table[i.strip().split('-')[0].replace("'","")])
                            self.u_ind.append(self.symbol_table[i.strip().split('-')[1].replace("'","")])

                    for i in range(len(self.kd_eqs)):
                        self.kd_eqs[i] = self.symbol_table[self.kd_eqs[i].strip().split('-')[0]] + " - " +\
                                         self.symbol_table[self.kd_eqs[i].strip().split('-')[1]]

                    # Do all of this if kd_eqs are not specified
                    if not self.kd_eqs:
                        self.kd_eqs_supplied = False
                        self.matrix_expr.append(ctx)
                        for i in self.type.keys():
                            if self.type[i] == "motionvariable":
                                if self.sign[self.symbol_table[i.lower()]] == 0:
                                    self.q_ind.append(self.symbol_table[i.lower()])
                                elif self.sign[self.symbol_table[i.lower()]] == 1:
                                    name = "u_" + self.symbol_table[i.lower()]
                                    self.symbol_table.update({name: name})
                                    self.write(name + " = " + "_me.dynamicsymbols('" + name + "')\n")
                                    if self.symbol_table[i.lower()] not in self.dependent_variables:
                                        self.u_ind.append(name)
                                        self.kd_equivalents.update({name: self.symbol_table[i.lower()]})
                                    else:
                                        self.u_dep.append(name)
                                        self.kd_equivalents.update({name: self.symbol_table[i.lower()]})

                        for i in self.kd_equivalents.keys():
                            self.kd_eqs.append(self.kd_equivalents[i] + "-" + i)

                    if not self.u_ind and not self.kd_eqs:
                        self.u_ind = self.q_ind.copy()
                        self.q_ind = []

                # deal with velocity constraints
                if self.dependent_variables:
                    for i in self.dependent_variables:
                        self.u_dep.append(i)
                        if i in self.u_ind:
                            self.u_ind.remove(i)


                self.u_dep[:] = [i for i in self.u_dep if i not in self.kd_equivalents.values()]

                force_list = []
                for i in self.forces.keys():
                    force_list.append("(" + i + "," + self.forces[i] + ")")
                if self.u_dep:
                    u_dep_text = ", u_dependent=[" + ", ".join(self.u_dep) + "]"
                else:
                    u_dep_text = ""
                if self.dependent_variables:
                    velocity_constraints_text = ", velocity_constraints = velocity_constraints"
                else:
                    velocity_constraints_text = ""
                if ctx.parentCtx not in self.fr_expr:
                    self.write("kd_eqs = [" + ", ".join(self.kd_eqs) + "]\n")
                    self.write("forceList = " + "[" + ", ".join(force_list) + "]\n")
                    self.write("kane = _me.KanesMethod(" + self.newtonian + ", " + "q_ind=[" +
                            ",".join(self.q_ind) + "], " + "u_ind=[" +
                            ", ".join(self.u_ind) + "]" + u_dep_text + ", " +
                            "kd_eqs = kd_eqs" + velocity_constraints_text + ")\n")
                    self.write("fr, frstar = kane." + "kanes_equations([" +
                                ", ".join(self.bodies.values()) + "], forceList)\n")
                    self.fr_expr.append(ctx.parentCtx)
                self.kane_parsed = True
                self.setValue(ctx, func_name)

        def exitMatrices(self, ctx):
            # Tree annotation for Matrices which is a labeled subrule of the parser rule expr.

            # MO = [a, b; c, d]
            # we generate _sm.Matrix([a, b, c, d]).reshape(2, 2)
            # The reshape values are determined by counting the "," and ";" in the Autolev matrix

            # Eg:
            # [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]
            # semicolon_count = 3 and rows = 3+1 = 4
            # comma_count = 8 and cols = 8/rows + 1 = 8/4 + 1 = 3

            # TODO** Parse block matrices
            self.matrix_expr.append(ctx)
            l = []
            semicolon_count = 0
            comma_count = 0
            for i in range(ctx.matrix().getChildCount()):
                child = ctx.matrix().getChild(i)
                if child == AutolevParser.ExprContext:
                    l.append(self.getValue(child))
                elif child.getText() == ";":
                    semicolon_count += 1
                    l.append(",")
                elif child.getText() == ",":
                    comma_count += 1
                    l.append(",")
                else:
                    try:
                        try:
                            l.append(self.getValue(child))
                        except Exception:
                            l.append(self.symbol_table[child.getText().lower()])
                    except Exception:
                        l.append(child.getText().lower())
            num_of_rows = semicolon_count + 1
            num_of_cols = (comma_count//num_of_rows) + 1

            self.setValue(ctx, "_sm.Matrix(" + "".join(l) + ")" + ".reshape(" +
                          str(num_of_rows) + ", " + str(num_of_cols) + ")")

        def exitVectorOrDyadic(self, ctx):
            self.vector_expr.append(ctx)
            ch = ctx.vec()

            if ch.getChild(0).getText() == "0>":
                self.setValue(ctx, "0")

            elif ch.getChild(0).getText() == "1>>":
                self.setValue(ctx, "1>>")

            elif "_" in ch.ID().getText() and ch.ID().getText().count('_') == 2:
                vec_text = ch.getText().lower()
                v1, v2, v3 = ch.ID().getText().lower().split('_')

                if v1 == "p":
                    if self.type2[v2] == "point":
                        e2 = self.symbol_table2[v2]
                    elif self.type2[v2] == "particle":
                        e2 = self.symbol_table2[v2] + ".point"
                    if self.type2[v3] == "point":
                        e3 = self.symbol_table2[v3]
                    elif self.type2[v3] == "particle":
                        e3 = self.symbol_table2[v3] + ".point"
                    get_vec = e3 + ".pos_from(" + e2 + ")"
                    self.setValue(ctx, get_vec)

                elif v1 in ("w", "alf"):
                    if v1 == "w":
                        text = ".ang_vel_in("
                    elif v1 == "alf":
                        text = ".ang_acc_in("
                    if self.type2[v2] == "bodies":
                        e2 = self.symbol_table2[v2] + "_f"
                    elif self.type2[v2] == "frame":
                        e2 = self.symbol_table2[v2]
                    if self.type2[v3] == "bodies":
                        e3 = self.symbol_table2[v3] + "_f"
                    elif self.type2[v3] == "frame":
                        e3 = self.symbol_table2[v3]
                    get_vec = e2 + text + e3 + ")"
                    self.setValue(ctx, get_vec)

                elif v1 in ("v", "a"):
                    if v1 == "v":
                        text = ".vel("
                    elif v1 == "a":
                        text = ".acc("
                    if self.type2[v2] == "point":
                        e2 = self.symbol_table2[v2]
                    elif self.type2[v2] == "particle":
                        e2 = self.symbol_table2[v2] + ".point"
                    get_vec = e2 + text + self.symbol_table2[v3] + ")"
                    self.setValue(ctx, get_vec)

                else:
                    self.setValue(ctx, vec_text.replace(">", ""))

            else:
                vec_text = ch.getText().lower()
                name = self.symbol_table[vec_text]
                self.setValue(ctx, name)

        def exitIndexing(self, ctx):
            if ctx.getChildCount() == 4:
                try:
                    int_text = str(int(self.getValue(ctx.getChild(2))) - 1)
                except Exception:
                    int_text = self.getValue(ctx.getChild(2)) + " - 1"
                self.setValue(ctx, ctx.ID().getText().lower() + "[" + int_text + "]")
            elif ctx.getChildCount() == 6:
                try:
                    int_text1 = str(int(self.getValue(ctx.getChild(2))) - 1)
                except Exception:
                    int_text1 = self.getValue(ctx.getChild(2)) + " - 1"
                try:
                    int_text2 = str(int(self.getValue(ctx.getChild(4))) - 1)
                except Exception:
                    int_text2 = self.getValue(ctx.getChild(2)) + " - 1"
                self.setValue(ctx, ctx.ID().getText().lower() + "[" + int_text1 + ", " + int_text2 + "]")


        # ================== Subrules of parser rule expr (End) ====================== #

        def exitRegularAssign(self, ctx):
            # Handle assignments of type ID = expr
            if ctx.equals().getText() in ["=", "+=", "-=", "*=", "/="]:
                equals = ctx.equals().getText()
            elif ctx.equals().getText() == ":=":
                equals = " = "
            elif ctx.equals().getText() == "^=":
                equals = "**="

            try:
                a = ctx.ID().getText().lower() + "'"*ctx.diff().getText().count("'")
            except Exception:
                a = ctx.ID().getText().lower()

            if a in self.type.keys() and self.type[a] in ("motionvariable", "motionvariable'") and\
            self.type[ctx.expr().getText().lower()] in ("motionvariable", "motionvariable'"):
                b = ctx.expr().getText().lower()
                if "'" in b and "'" not in a:
                    a, b = b, a
                if not self.kane_parsed:
                    self.kd_eqs.append(a + "-" + b)
                    self.kd_equivalents.update({self.symbol_table[a]:
                                                self.symbol_table[b]})
                    self.kd_equivalents2.update({self.symbol_table[b]:
                                                    self.symbol_table[a]})

            if a in self.symbol_table.keys() and a in self.type.keys() and self.type[a] in ("variable", "motionvariable"):
                self.explicit.update({self.symbol_table[a]: self.getValue(ctx.expr())})

            else:
                if ctx.expr() in self.matrix_expr:
                    self.type.update({a: "matrix"})

                try:
                    b = self.symbol_table[a]
                except KeyError:
                    self.symbol_table[a] = a

                if "_" in a and a.count("_") == 1:
                    e1, e2 = a.split('_')
                    if e1 in self.type2.keys() and self.type2[e1] in ("frame", "bodies")\
                    and e2 in self.type2.keys() and self.type2[e2] in ("frame", "bodies"):
                        if self.type2[e1] == "bodies":
                            t1 = "_f"
                        else:
                            t1 = ""
                        if self.type2[e2] == "bodies":
                            t2 = "_f"
                        else:
                            t2 = ""

                        self.write(self.symbol_table2[e2] + t2 + ".orient(" + self.symbol_table2[e1] +
                                   t1 + ", 'DCM', " + self.getValue(ctx.expr()) + ")\n")
                    else:
                        self.write(self.symbol_table[a] + " " + equals + " " +
                                    self.getValue(ctx.expr()) + "\n")
                else:
                    self.write(self.symbol_table[a] + " " + equals + " " +
                                self.getValue(ctx.expr()) + "\n")

        def exitIndexAssign(self, ctx):
            # Handle assignments of type ID[index] = expr
                if ctx.equals().getText() in ["=", "+=", "-=", "*=", "/="]:
                    equals = ctx.equals().getText()
                elif ctx.equals().getText() == ":=":
                    equals = " = "
                elif ctx.equals().getText() == "^=":
                    equals = "**="

                text = ctx.ID().getText().lower()
                self.type.update({text: "matrix"})
                # Handle assignments of type ID[2] = expr
                if ctx.index().getChildCount() == 1:
                    if ctx.index().getChild(0).getText() == "1":
                        self.type.update({text: "matrix"})
                        self.symbol_table.update({text: text})
                        self.write(text + " = " + "_sm.Matrix([[0]])\n")
                        self.write(text + "[0] = " + self.getValue(ctx.expr()) + "\n")
                    else:
                        # m = m.row_insert(m.shape[0], _sm.Matrix([[0]]))
                        self.write(text + " = " + text +
                                   ".row_insert(" + text + ".shape[0]" + ", " + "_sm.Matrix([[0]])" + ")\n")
                        self.write(text + "[" + text + ".shape[0]-1" + "] = " + self.getValue(ctx.expr()) + "\n")

                # Handle assignments of type ID[2, 2] = expr
                elif ctx.index().getChildCount() == 3:
                    l = []
                    try:
                        l.append(str(int(self.getValue(ctx.index().getChild(0)))-1))
                    except Exception:
                        l.append(self.getValue(ctx.index().getChild(0)) + "-1")
                    l.append(",")
                    try:
                        l.append(str(int(self.getValue(ctx.index().getChild(2)))-1))
                    except Exception:
                        l.append(self.getValue(ctx.index().getChild(2)) + "-1")
                    self.write(self.symbol_table[ctx.ID().getText().lower()] +
                               "[" + "".join(l) + "]" + " " + equals + " " + self.getValue(ctx.expr()) + "\n")

        def exitVecAssign(self, ctx):
            # Handle assignments of the type vec = expr
            ch = ctx.vec()
            vec_text = ch.getText().lower()

            if "_" in ch.ID().getText():
                num = ch.ID().getText().count('_')

                if num == 2:
                    v1, v2, v3 = ch.ID().getText().lower().split('_')

                    if v1 == "p":
                        if self.type2[v2] == "point":
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        if self.type2[v3] == "point":
                            e3 = self.symbol_table2[v3]
                        elif self.type2[v3] == "particle":
                            e3 = self.symbol_table2[v3] + ".point"
                        # ab.set_pos(na, la*a.x)
                        self.write(e3 + ".set_pos(" + e2 + ", " + self.getValue(ctx.expr()) + ")\n")

                    elif v1 in ("w", "alf"):
                        if v1 == "w":
                            text = ".set_ang_vel("
                        elif v1 == "alf":
                            text = ".set_ang_acc("
                        # a.set_ang_vel(n, qad*a.z)
                        if self.type2[v2] == "bodies":
                            e2 = self.symbol_table2[v2] + "_f"
                        else:
                            e2 = self.symbol_table2[v2]
                        if self.type2[v3] == "bodies":
                            e3 = self.symbol_table2[v3] + "_f"
                        else:
                            e3 = self.symbol_table2[v3]
                        self.write(e2 + text + e3 + ", " + self.getValue(ctx.expr()) + ")\n")

                    elif v1 in ("v", "a"):
                        if v1 == "v":
                            text = ".set_vel("
                        elif v1 == "a":
                            text = ".set_acc("
                        if self.type2[v2] == "point":
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        self.write(e2 + text + self.symbol_table2[v3] +
                                   ", " + self.getValue(ctx.expr()) + ")\n")
                    elif v1 == "i":
                        if v2 in self.type2.keys() and self.type2[v2] == "bodies":
                            self.write(self.symbol_table2[v2] + ".inertia = (" + self.getValue(ctx.expr()) +
                            ", " + self.symbol_table2[v3] + ")\n")
                            self.inertia_point.update({v2: v3})
                        elif v2 in self.type2.keys() and self.type2[v2] == "particle":
                            self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                        else:
                            self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                    else:
                        self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")

                elif num == 1:
                    v1, v2 = ch.ID().getText().lower().split('_')

                    if v1 in ("force", "torque"):
                        if self.type2[v2] in ("point", "frame"):
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        self.symbol_table.update({vec_text: ch.ID().getText().lower()})

                        if e2 in self.forces.keys():
                            self.forces[e2] = self.forces[e2] + " + " + self.getValue(ctx.expr())
                        else:
                            self.forces.update({e2: self.getValue(ctx.expr())})
                        self.write(ch.ID().getText().lower() + " = " + self.forces[e2] + "\n")

                    else:
                        name = ch.ID().getText().lower()
                        self.symbol_table.update({vec_text: name})
                        self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                else:
                    name = ch.ID().getText().lower()
                    self.symbol_table.update({vec_text: name})
                    self.write(name + " " + ctx.getChild(1).getText() + " " + self.getValue(ctx.expr()) + "\n")
            else:
                name = ch.ID().getText().lower()
                self.symbol_table.update({vec_text: name})
                self.write(name + " " + ctx.getChild(1).getText() + " " + self.getValue(ctx.expr()) + "\n")

        def enterInputs2(self, ctx):
            self.in_inputs = True

        # Inputs
        def exitInputs2(self, ctx):
            # Stores numerical values given by the input command which
            # are used for codegen and numerical analysis.
            if ctx.getChildCount() == 3:
                try:
                    self.inputs.update({self.symbol_table[ctx.id_diff().getText().lower()]: self.getValue(ctx.expr(0))})
                except Exception:
                    self.inputs.update({ctx.id_diff().getText().lower(): self.getValue(ctx.expr(0))})
            elif ctx.getChildCount() == 4:
                try:
                    self.inputs.update({self.symbol_table[ctx.id_diff().getText().lower()]:
                    (self.getValue(ctx.expr(0)), self.getValue(ctx.expr(1)))})
                except Exception:
                    self.inputs.update({ctx.id_diff().getText().lower():
                    (self.getValue(ctx.expr(0)), self.getValue(ctx.expr(1)))})

            self.in_inputs = False

        def enterOutputs(self, ctx):
            self.in_outputs = True
        def exitOutputs(self, ctx):
            self.in_outputs = False

        def exitOutputs2(self, ctx):
            try:
                if "[" in ctx.expr(1).getText():
                    self.outputs.append(self.symbol_table[ctx.expr(0).getText().lower()] +
                                        ctx.expr(1).getText().lower())
                else:
                    self.outputs.append(self.symbol_table[ctx.expr(0).getText().lower()])

            except Exception:
                pass

        # Code commands
        def exitCodegen(self, ctx):
            # Handles the CODE() command ie the solvers and the codgen part.
            # Uses linsolve for the algebraic solvers and nsolve for non linear solvers.

            if ctx.functionCall().getChild(0).getText().lower() == "algebraic":
                matrix_name = self.getValue(ctx.functionCall().expr(0))
                e = []
                d = []
                for i in range(1, (ctx.functionCall().getChildCount()-2)//2):
                    a = self.getValue(ctx.functionCall().expr(i))
                    e.append(a)

                for i in self.inputs.keys():
                    d.append(i + ":" + self.inputs[i])
                self.write(matrix_name + "_list" + " = " + "[]\n")
                self.write("for i in " + matrix_name + ":  " + matrix_name +
                           "_list" + ".append(i.subs({" + ", ".join(d) + "}))\n")
                self.write("print(_sm.linsolve(" + matrix_name + "_list" + ", " + ",".join(e) + "))\n")

            elif ctx.functionCall().getChild(0).getText().lower() == "nonlinear":
                e = []
                d = []
                guess = []
                for i in range(1, (ctx.functionCall().getChildCount()-2)//2):
                    a = self.getValue(ctx.functionCall().expr(i))
                    e.append(a)
                #print(self.inputs)
                for i in self.inputs.keys():
                    if i in self.symbol_table.keys():
                        if type(self.inputs[i]) is tuple:
                            j, z = self.inputs[i]
                        else:
                            j = self.inputs[i]
                            z = ""
                        if i not in e:
                            if z == "deg":
                                d.append(i + ":" + "_np.deg2rad(" + j + ")")
                            else:
                                d.append(i + ":" + j)
                        else:
                            if z == "deg":
                                guess.append("_np.deg2rad(" + j + ")")
                            else:
                                guess.append(j)

                self.write("matrix_list" + " = " + "[]\n")
                self.write("for i in " + self.getValue(ctx.functionCall().expr(0)) + ":")
                self.write("matrix_list" + ".append(i.subs({" + ", ".join(d) + "}))\n")
                self.write("print(_sm.nsolve(matrix_list," + "(" + ",".join(e) + ")" +
                           ",(" + ",".join(guess) + ")" + "))\n")

            elif ctx.functionCall().getChild(0).getText().lower() in ["ode", "dynamics"] and self.include_numeric:
                if self.kane_type == "no_args":
                    for i in self.symbol_table.keys():
                        try:
                            if self.type[i] == "constants" or self.type[self.symbol_table[i]] == "constants":
                                self.constants.append(self.symbol_table[i])
                        except Exception:
                            pass
                    q_add_u = self.q_ind + self.q_dep + self.u_ind + self.u_dep
                    x0 = []
                    for i in q_add_u:
                        try:
                            if i in self.inputs.keys():
                                if type(self.inputs[i]) is tuple:
                                    if self.inputs[i][1] == "deg":
                                        x0.append(i + ":" + "_np.deg2rad(" + self.inputs[i][0] + ")")
                                    else:
                                        x0.append(i + ":" + self.inputs[i][0])
                                else:
                                    x0.append(i + ":" + self.inputs[i])
                            elif self.kd_equivalents[i] in self.inputs.keys():
                                if type(self.inputs[self.kd_equivalents[i]]) is tuple:
                                    x0.append(i + ":" + self.inputs[self.kd_equivalents[i]][0])
                                else:
                                    x0.append(i + ":" + self.inputs[self.kd_equivalents[i]])
                        except Exception:
                            pass

                    # numerical constants
                    numerical_constants = []
                    for i in self.constants:
                        if i in self.inputs.keys():
                            if type(self.inputs[i]) is tuple:
                                numerical_constants.append(self.inputs[i][0])
                            else:
                                numerical_constants.append(self.inputs[i])

                    # t = linspace
                    t_final = self.inputs["tfinal"]
                    integ_stp = self.inputs["integstp"]

                    self.write("from pydy.system import System\n")
                    const_list = []
                    if numerical_constants:
                        for i in range(len(self.constants)):
                            const_list.append(self.constants[i] + ":" + numerical_constants[i])
                    specifieds = []
                    if self.t:
                        specifieds.append("_me.dynamicsymbols('t')" + ":" + "lambda x, t: t")

                    for i in self.inputs:
                        if i in self.symbol_table.keys() and self.symbol_table[i] not in\
                        self.constants + self.q_ind + self.q_dep + self.u_ind + self.u_dep:
                            specifieds.append(self.symbol_table[i] + ":" + self.inputs[i])

                    self.write("sys = System(kane, constants = {" + ", ".join(const_list) + "},\n" +
                               "specifieds={" + ", ".join(specifieds) + "},\n" +
                               "initial_conditions={" + ", ".join(x0) + "},\n" +
                               "times = _np.linspace(0.0, " + str(t_final) + ", " + str(t_final) +
                               "/" + str(integ_stp) + "))\n\ny=sys.integrate()\n")

                    # For outputs other than qs and us.
                    other_outputs = []
                    for i in self.outputs:
                        if i not in q_add_u:
                            if "[" in i:
                                other_outputs.append((i[:-3] + i[-2], i[:-3] + "[" + str(int(i[-2])-1) + "]"))
                            else:
                                other_outputs.append((i, i))

                    for i in other_outputs:
                        self.write(i[0] + "_out" + " = " + "[]\n")
                    if other_outputs:
                        self.write("for i in y:\n")
                        self.write("    q_u_dict = dict(zip(sys.coordinates+sys.speeds, i))\n")
                        for i in other_outputs:
                            self.write(" "*4 + i[0] + "_out" + ".append(" + i[1] + ".subs(q_u_dict)" +
                                    ".subs(sys.constants).evalf())\n")

        # Standalone function calls (used for dual functions)
        def exitFunctionCall(self, ctx):
            # Basically deals with standalone function calls ie functions which are not a part of
            # expressions and assignments. Autolev Dual functions can both appear in standalone
            # function calls and also on the right hand side as part of expr or assignment.

            # Dual functions are indicated by a * in the comments below

            # Checks if the function is a statement on its own
            if ctx.parentCtx.getRuleIndex() == AutolevParser.RULE_stat:
                func_name = ctx.getChild(0).getText().lower()
                # Expand(E, n:m) *
                if func_name == "expand":
                    # If the first argument is a pre declared variable.
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]
                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([i.expand() for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(symbol + " = " + symbol + "." + "expand()\n")

                # Factor(E, x) *
                elif func_name == "factor":
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]
                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([_sm.factor(i," + self.getValue(ctx.expr(1)) +
                                   ") for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(expr + " = " + "_sm.factor(" + expr + ", " +
                                   self.getValue(ctx.expr(1)) + ")\n")

                # Solve(Zero, x, y)
                elif func_name == "solve":
                    l = []
                    l2 = []
                    num = 0
                    for i in range(1, ctx.getChildCount()):
                        if ctx.getChild(i).getText() == ",":
                            num+=1
                        try:
                            l.append(self.getValue(ctx.getChild(i)))
                        except Exception:
                            l.append(ctx.getChild(i).getText())

                        if i != 2:
                            try:
                                l2.append(self.getValue(ctx.getChild(i)))
                            except Exception:
                                pass

                    for i in l2:
                        self.explicit.update({i: "_sm.solve" + "".join(l) + "[" + i + "]"})

                    self.write("print(_sm.solve" + "".join(l) + ")\n")

                # Arrange(y, n, x) *
                elif func_name == "arrange":
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]

                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([i.collect(" + self.getValue(ctx.expr(2)) +
                                   ")" + "for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(self.getValue(ctx.expr(0)) + ".collect(" +
                                   self.getValue(ctx.expr(2)) + ")\n")

                # Eig(M, EigenValue, EigenVec)
                elif func_name == "eig":
                    self.symbol_table.update({ctx.expr(1).getText().lower(): ctx.expr(1).getText().lower()})
                    self.symbol_table.update({ctx.expr(2).getText().lower(): ctx.expr(2).getText().lower()})
                    # _sm.Matrix([i.evalf() for i in (i_s_so).eigenvals().keys()])
                    self.write(ctx.expr(1).getText().lower() + " = " +
                               "_sm.Matrix([i.evalf() for i in " +
                               "(" + self.getValue(ctx.expr(0)) + ")" + ".eigenvals().keys()])\n")
                    # _sm.Matrix([i[2][0].evalf() for i in (i_s_o).eigenvects()]).reshape(i_s_o.shape[0], i_s_o.shape[1])
                    self.write(ctx.expr(2).getText().lower() + " = " +
                               "_sm.Matrix([i[2][0].evalf() for i in " + "(" + self.getValue(ctx.expr(0)) + ")" +
                               ".eigenvects()]).reshape(" + self.getValue(ctx.expr(0)) + ".shape[0], " +
                               self.getValue(ctx.expr(0)) + ".shape[1])\n")

                # Simprot(N, A, 3, qA)
                elif func_name == "simprot":
                    # A.orient(N, 'Axis', qA, N.z)
                    if self.type2[ctx.expr(0).getText().lower()] == "frame":
                        frame1 = self.symbol_table2[ctx.expr(0).getText().lower()]
                    elif self.type2[ctx.expr(0).getText().lower()] == "bodies":
                        frame1 = self.symbol_table2[ctx.expr(0).getText().lower()] + "_f"
                    if self.type2[ctx.expr(1).getText().lower()] == "frame":
                        frame2 = self.symbol_table2[ctx.expr(1).getText().lower()]
                    elif self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        frame2 = self.symbol_table2[ctx.expr(1).getText().lower()] + "_f"
                    e2 = ""
                    if ctx.expr(2).getText()[0] == "-":
                        e2 = "-1*"
                    if ctx.expr(2).getText() in ("1", "-1"):
                        e = frame1 + ".x"
                    elif ctx.expr(2).getText() in ("2", "-2"):
                        e = frame1 + ".y"
                    elif ctx.expr(2).getText() in ("3", "-3"):
                        e = frame1 + ".z"
                    else:
                        e = self.getValue(ctx.expr(2))
                        e2 = ""

                    if "degrees" in self.settings.keys() and self.settings["degrees"] == "off":
                        value = self.getValue(ctx.expr(3))
                    else:
                        if ctx.expr(3) in self.numeric_expr:
                            value = "_np.deg2rad(" + self.getValue(ctx.expr(3)) + ")"
                        else:
                            value = self.getValue(ctx.expr(3))
                    self.write(frame2 + ".orient(" + frame1 +
                               ", " + "'Axis'" + ", " + "[" + value +
                               ", " + e2 + e + "]" + ")\n")

                # Express(A2>, B) *
                elif func_name == "express":
                    if self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        f = "_f"
                    else:
                        f = ""

                    if '_' in ctx.expr(0).getText().lower() and ctx.expr(0).getText().count('_') == 2:
                        vec = ctx.expr(0).getText().lower().replace(">", "").split('_')
                        v1 = self.symbol_table2[vec[1]]
                        v2 = self.symbol_table2[vec[2]]
                        if vec[0] == "p":
                            self.write(v2 + ".set_pos(" + v1 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        elif vec[0] == "v":
                            self.write(v1 + ".set_vel(" + v2 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        elif vec[0] == "a":
                            self.write(v1 + ".set_acc(" + v2 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        else:
                            self.write(self.getValue(ctx.expr(0)) + " = " + "(" + self.getValue(ctx.expr(0)) + ")" + ".express(" +
                                        self.symbol_table2[ctx.expr(1).getText().lower()] + f + ")\n")
                    else:
                        self.write(self.getValue(ctx.expr(0)) + " = " + "(" + self.getValue(ctx.expr(0)) + ")" + ".express(" +
                                    self.symbol_table2[ctx.expr(1).getText().lower()] + f + ")\n")

                # Angvel(A, B)
                elif func_name == "angvel":
                    self.write("print(" + self.symbol_table2[ctx.expr(1).getText().lower()] +
                               ".ang_vel_in(" + self.symbol_table2[ctx.expr(0).getText().lower()] + "))\n")

                # v2pts(N, A, O, P)
                elif func_name in ("v2pts", "a2pts", "v2pt", "a1pt"):
                    if func_name == "v2pts":
                        text = ".v2pt_theory("
                    elif func_name == "a2pts":
                        text = ".a2pt_theory("
                    elif func_name == "v1pt":
                        text = ".v1pt_theory("
                    elif func_name == "a1pt":
                        text = ".a1pt_theory("
                    if self.type2[ctx.expr(1).getText().lower()] == "frame":
                        frame = self.symbol_table2[ctx.expr(1).getText().lower()]
                    elif self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        frame = self.symbol_table2[ctx.expr(1).getText().lower()] + "_f"
                    expr_list = []
                    for i in range(2, 4):
                        if self.type2[ctx.expr(i).getText().lower()] == "point":
                            expr_list.append(self.symbol_table2[ctx.expr(i).getText().lower()])
                        elif self.type2[ctx.expr(i).getText().lower()] == "particle":
                            expr_list.append(self.symbol_table2[ctx.expr(i).getText().lower()] + ".point")

                    self.write(expr_list[1] + text + expr_list[0] +
                               "," + self.symbol_table2[ctx.expr(0).getText().lower()] + "," +
                               frame + ")\n")

                # Gravity(g*N1>)
                elif func_name == "gravity":
                    for i in self.bodies.keys():
                        if self.type2[i] == "bodies":
                            e = self.symbol_table2[i] + ".masscenter"
                        elif self.type2[i] == "particle":
                            e = self.symbol_table2[i] + ".point"
                        if e in self.forces.keys():
                            self.forces[e] = self.forces[e] + self.symbol_table2[i] +\
                                             ".mass*(" + self.getValue(ctx.expr(0)) + ")"
                        else:
                            self.forces.update({e: self.symbol_table2[i] +
                                               ".mass*(" + self.getValue(ctx.expr(0)) + ")"})
                        self.write("force_" + i + " = " + self.forces[e] + "\n")

                # Explicit(EXPRESS(IMPLICIT>,C))
                elif func_name == "explicit":
                    if ctx.expr(0) in self.vector_expr:
                        self.vector_expr.append(ctx)
                    expr = self.getValue(ctx.expr(0))
                    if self.explicit.keys():
                        explicit_list = []
                        for i in self.explicit.keys():
                            explicit_list.append(i + ":" + self.explicit[i])
                        if '_' in ctx.expr(0).getText().lower() and ctx.expr(0).getText().count('_') == 2:
                            vec = ctx.expr(0).getText().lower().replace(">", "").split('_')
                            v1 = self.symbol_table2[vec[1]]
                            v2 = self.symbol_table2[vec[2]]
                            if vec[0] == "p":
                                self.write(v2 + ".set_pos(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            elif vec[0] == "v":
                                self.write(v2 + ".set_vel(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            elif vec[0] == "a":
                                self.write(v2 + ".set_acc(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            else:
                                self.write(expr + " = " + "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})\n")
                        else:
                            self.write(expr + " = " + "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})\n")

                # Force(O/Q, -k*Stretch*Uvec>)
                elif func_name in ("force", "torque"):

                    if "/" in ctx.expr(0).getText().lower():
                        p1 = ctx.expr(0).getText().lower().split('/')[0]
                        p2 = ctx.expr(0).getText().lower().split('/')[1]
                        if self.type2[p1] in ("point", "frame"):
                            pt1 = self.symbol_table2[p1]
                        elif self.type2[p1] == "particle":
                            pt1 = self.symbol_table2[p1] + ".point"
                        if self.type2[p2] in ("point", "frame"):
                            pt2 = self.symbol_table2[p2]
                        elif self.type2[p2] == "particle":
                            pt2 = self.symbol_table2[p2] + ".point"
                        if pt1 in self.forces.keys():
                            self.forces[pt1] = self.forces[pt1] + " + -1*("+self.getValue(ctx.expr(1)) + ")"
                            self.write("force_" + p1 + " = " + self.forces[pt1] + "\n")
                        else:
                            self.forces.update({pt1: "-1*("+self.getValue(ctx.expr(1)) + ")"})
                            self.write("force_" + p1 + " = " + self.forces[pt1] + "\n")
                        if pt2 in self.forces.keys():
                            self.forces[pt2] = self.forces[pt2] + "+ " + self.getValue(ctx.expr(1))
                            self.write("force_" + p2 + " = " + self.forces[pt2] + "\n")
                        else:
                            self.forces.update({pt2: self.getValue(ctx.expr(1))})
                            self.write("force_" + p2 + " = " + self.forces[pt2] + "\n")

                    elif ctx.expr(0).getChildCount() == 1:
                        p1 = ctx.expr(0).getText().lower()
                        if self.type2[p1] in ("point", "frame"):
                            pt1 = self.symbol_table2[p1]
                        elif self.type2[p1] == "particle":
                            pt1 = self.symbol_table2[p1] + ".point"
                        if pt1 in self.forces.keys():
                            self.forces[pt1] = self.forces[pt1] + "+ -1*(" + self.getValue(ctx.expr(1)) + ")"
                        else:
                            self.forces.update({pt1: "-1*(" + self.getValue(ctx.expr(1)) + ")"})

                # Constrain(Dependent[qB])
                elif func_name == "constrain":
                    if ctx.getChild(2).getChild(0).getText().lower() == "dependent":
                        self.write("velocity_constraints = [i for i in dependent]\n")
                    x = (ctx.expr(0).getChildCount()-2)//2
                    for i in range(x):
                        self.dependent_variables.append(self.getValue(ctx.expr(0).expr(i)))

                # Kane()
                elif func_name == "kane":
                    if ctx.getChildCount() == 3:
                        self.kane_type = "no_args"

        # Settings
        def exitSettings(self, ctx):
            # Stores settings like Complex on/off, Degrees on/off etc in self.settings.
            try:
                self.settings.update({ctx.getChild(0).getText().lower():
                                     ctx.getChild(1).getText().lower()})
            except Exception:
                pass

        def exitMassDecl2(self, ctx):
            # Used for declaring the masses of particles and rigidbodies.
            particle = self.symbol_table2[ctx.getChild(0).getText().lower()]
            if ctx.getText().count("=") == 2:
                if ctx.expr().expr(1) in self.numeric_expr:
                    e = "_sm.S(" + self.getValue(ctx.expr().expr(1)) + ")"
                else:
                    e = self.getValue(ctx.expr().expr(1))
                self.symbol_table.update({ctx.expr().expr(0).getText().lower(): ctx.expr().expr(0).getText().lower()})
                self.write(ctx.expr().expr(0).getText().lower() + " = " + e + "\n")
                mass = ctx.expr().expr(0).getText().lower()
            else:
                try:
                    if ctx.expr() in self.numeric_expr:
                        mass = "_sm.S(" + self.getValue(ctx.expr()) + ")"
                    else:
                        mass = self.getValue(ctx.expr())
                except Exception:
                    a_text = ctx.expr().getText().lower()
                    self.symbol_table.update({a_text: a_text})
                    self.type.update({a_text: "constants"})
                    self.write(a_text + " = " + "_sm.symbols('" + a_text + "')\n")
                    mass = a_text

            self.write(particle + ".mass = " + mass + "\n")

        def exitInertiaDecl(self, ctx):
            inertia_list = []
            try:
                ctx.ID(1).getText()
                num = 5
            except Exception:
                num = 2
            for i in range((ctx.getChildCount()-num)//2):
                try:
                    if ctx.expr(i) in self.numeric_expr:
                        inertia_list.append("_sm.S(" + self.getValue(ctx.expr(i)) + ")")
                    else:
                        inertia_list.append(self.getValue(ctx.expr(i)))
                except Exception:
                    a_text = ctx.expr(i).getText().lower()
                    self.symbol_table.update({a_text: a_text})
                    self.type.update({a_text: "constants"})
                    self.write(a_text + " = " + "_sm.symbols('" + a_text + "')\n")
                    inertia_list.append(a_text)

            if len(inertia_list) < 6:
                for i in range(6-len(inertia_list)):
                    inertia_list.append("0")
            # body_a.inertia = (_me.inertia(body_a, I1, I2, I3, 0, 0, 0), body_a_cm)
            try:
                frame = self.symbol_table2[ctx.ID(1).getText().lower()]
                point = self.symbol_table2[ctx.ID(0).getText().lower().split('_')[1]]
                body = self.symbol_table2[ctx.ID(0).getText().lower().split('_')[0]]
                self.inertia_point.update({ctx.ID(0).getText().lower().split('_')[0]
                                          : ctx.ID(0).getText().lower().split('_')[1]})
                self.write(body + ".inertia" + " = " + "(_me.inertia(" + frame + ", " +
                           ", ".join(inertia_list) + "), " + point + ")\n")

            except Exception:
                body_name = self.symbol_table2[ctx.ID(0).getText().lower()]
                body_name_cm = body_name + "_cm"
                self.inertia_point.update({ctx.ID(0).getText().lower(): ctx.ID(0).getText().lower() + "o"})
                self.write(body_name + ".inertia" + " = " + "(_me.inertia(" + body_name + "_f" + ", " +
                           ", ".join(inertia_list) + "), " + body_name_cm + ")\n")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\tests\test__version.py ===
"""Tests for the NumpyVersion class.

"""
from numpy.lib import NumpyVersion
from numpy.testing import assert_, assert_raises


def test_main_versions():
    assert_(NumpyVersion('1.8.0') == '1.8.0')
    for ver in ['1.9.0', '2.0.0', '1.8.1', '10.0.1']:
        assert_(NumpyVersion('1.8.0') < ver)

    for ver in ['1.7.0', '1.7.1', '0.9.9']:
        assert_(NumpyVersion('1.8.0') > ver)


def test_version_1_point_10():
    # regression test for gh-2998.
    assert_(NumpyVersion('1.9.0') < '1.10.0')
    assert_(NumpyVersion('1.11.0') < '1.11.1')
    assert_(NumpyVersion('1.11.0') == '1.11.0')
    assert_(NumpyVersion('1.99.11') < '1.99.12')


def test_alpha_beta_rc():
    assert_(NumpyVersion('1.8.0rc1') == '1.8.0rc1')
    for ver in ['1.8.0', '1.8.0rc2']:
        assert_(NumpyVersion('1.8.0rc1') < ver)

    for ver in ['1.8.0a2', '1.8.0b3', '1.7.2rc4']:
        assert_(NumpyVersion('1.8.0rc1') > ver)

    assert_(NumpyVersion('1.8.0b1') > '1.8.0a2')


def test_dev_version():
    assert_(NumpyVersion('1.9.0.dev-Unknown') < '1.9.0')
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev-ffffffff']:
        assert_(NumpyVersion('1.9.0.dev-f16acvda') < ver)

    assert_(NumpyVersion('1.9.0.dev-f16acvda') == '1.9.0.dev-11111111')


def test_dev_a_b_rc_mixed():
    assert_(NumpyVersion('1.9.0a2.dev-f16acvda') == '1.9.0a2.dev-11111111')
    assert_(NumpyVersion('1.9.0a2.dev-6acvda54') < '1.9.0a2')


def test_dev0_version():
    assert_(NumpyVersion('1.9.0.dev0+Unknown') < '1.9.0')
    for ver in ['1.9.0', '1.9.0a1', '1.9.0b2', '1.9.0b2.dev0+ffffffff']:
        assert_(NumpyVersion('1.9.0.dev0+f16acvda') < ver)

    assert_(NumpyVersion('1.9.0.dev0+f16acvda') == '1.9.0.dev0+11111111')


def test_dev0_a_b_rc_mixed():
    assert_(NumpyVersion('1.9.0a2.dev0+f16acvda') == '1.9.0a2.dev0+11111111')
    assert_(NumpyVersion('1.9.0a2.dev0+6acvda54') < '1.9.0a2')


def test_raises():
    for ver in ['1.9', '1,9.0', '1.7.x']:
        assert_raises(ValueError, NumpyVersion, ver)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\conftest.py ===
"""
This file is very long and growing, but it was decided to not split it yet, as
it's still manageable (2020-03-17, ~1.1k LoC). See gh-31989

Instead of splitting it was decided to define sections here:
- Configuration / Settings
- Autouse fixtures
- Common arguments
- Missing values & co.
- Classes
- Indices
- Series'
- DataFrames
- Operators & Operations
- Data sets/files
- Time zones
- Dtypes
- Misc
"""
from __future__ import annotations

from collections import abc
from datetime import (
    date,
    datetime,
    time,
    timedelta,
    timezone,
)
from decimal import Decimal
import operator
import os
from typing import (
    TYPE_CHECKING,
    Callable,
)

from dateutil.tz import (
    tzlocal,
    tzutc,
)
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import (
    FixedOffset,
    utc,
)

from pandas._config.config import _get_option

import pandas.util._test_decorators as td

from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    IntervalDtype,
)

import pandas as pd
from pandas import (
    CategoricalIndex,
    DataFrame,
    Interval,
    IntervalIndex,
    Period,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
)
from pandas.util.version import Version

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

try:
    import pyarrow as pa
except ImportError:
    has_pyarrow = False
else:
    del pa
    has_pyarrow = True

import zoneinfo

try:
    zoneinfo.ZoneInfo("UTC")
except zoneinfo.ZoneInfoNotFoundError:
    zoneinfo = None  # type: ignore[assignment]


# ----------------------------------------------------------------
# Configuration / Settings
# ----------------------------------------------------------------
# pytest


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--no-strict-data-files",
        action="store_false",
        help="Don't fail if a test is skipped for missing data file.",
    )


def ignore_doctest_warning(item: pytest.Item, path: str, message: str) -> None:
    """Ignore doctest warning.

    Parameters
    ----------
    item : pytest.Item
        pytest test item.
    path : str
        Module path to Python object, e.g. "pandas.core.frame.DataFrame.append". A
        warning will be filtered when item.name ends with in given path. So it is
        sufficient to specify e.g. "DataFrame.append".
    message : str
        Message to be filtered.
    """
    if item.name.endswith(path):
        item.add_marker(pytest.mark.filterwarnings(f"ignore:{message}"))


def pytest_collection_modifyitems(items, config) -> None:
    is_doctest = config.getoption("--doctest-modules") or config.getoption(
        "--doctest-cython", default=False
    )

    # Warnings from doctests that can be ignored; place reason in comment above.
    # Each entry specifies (path, message) - see the ignore_doctest_warning function
    ignored_doctest_warnings = [
        ("is_int64_dtype", "is_int64_dtype is deprecated"),
        ("is_interval_dtype", "is_interval_dtype is deprecated"),
        ("is_period_dtype", "is_period_dtype is deprecated"),
        ("is_datetime64tz_dtype", "is_datetime64tz_dtype is deprecated"),
        ("is_categorical_dtype", "is_categorical_dtype is deprecated"),
        ("is_sparse", "is_sparse is deprecated"),
        ("DataFrameGroupBy.fillna", "DataFrameGroupBy.fillna is deprecated"),
        ("NDFrame.replace", "The 'method' keyword"),
        ("NDFrame.replace", "Series.replace without 'value'"),
        ("NDFrame.clip", "Downcasting behavior in Series and DataFrame methods"),
        ("Series.idxmin", "The behavior of Series.idxmin"),
        ("Series.idxmax", "The behavior of Series.idxmax"),
        ("SeriesGroupBy.fillna", "SeriesGroupBy.fillna is deprecated"),
        ("SeriesGroupBy.idxmin", "The behavior of Series.idxmin"),
        ("SeriesGroupBy.idxmax", "The behavior of Series.idxmax"),
        # Docstring divides by zero to show behavior difference
        ("missing.mask_zero_div_zero", "divide by zero encountered"),
        (
            "to_pydatetime",
            "The behavior of DatetimeProperties.to_pydatetime is deprecated",
        ),
        (
            "pandas.core.generic.NDFrame.bool",
            "(Series|DataFrame).bool is now deprecated and will be removed "
            "in future version of pandas",
        ),
        (
            "pandas.core.generic.NDFrame.first",
            "first is deprecated and will be removed in a future version. "
            "Please create a mask and filter using `.loc` instead",
        ),
        (
            "Resampler.fillna",
            "DatetimeIndexResampler.fillna is deprecated",
        ),
        (
            "DataFrameGroupBy.fillna",
            "DataFrameGroupBy.fillna with 'method' is deprecated",
        ),
        (
            "DataFrameGroupBy.fillna",
            "DataFrame.fillna with 'method' is deprecated",
        ),
        ("read_parquet", "Passing a BlockManager to DataFrame is deprecated"),
    ]

    if is_doctest:
        for item in items:
            for path, message in ignored_doctest_warnings:
                ignore_doctest_warning(item, path, message)


hypothesis_health_checks = [hypothesis.HealthCheck.too_slow]
if Version(hypothesis.__version__) >= Version("6.83.2"):
    hypothesis_health_checks.append(hypothesis.HealthCheck.differing_executors)

# Hypothesis
hypothesis.settings.register_profile(
    "ci",
    # Hypothesis timing checks are tuned for scalars by default, so we bump
    # them from 200ms to 500ms per test case as the global default.  If this
    # is too short for a specific test, (a) try to make it faster, and (b)
    # if it really is slow add `@settings(deadline=...)` with a working value,
    # or `deadline=None` to entirely disable timeouts for that test.
    # 2022-02-09: Changed deadline from 500 -> None. Deadline leads to
    # non-actionable, flaky CI failures (# GH 24641, 44969, 45118, 44969)
    deadline=None,
    suppress_health_check=tuple(hypothesis_health_checks),
)
hypothesis.settings.load_profile("ci")

# Registering these strategies makes them globally available via st.from_type,
# which is use for offsets in tests/tseries/offsets/test_offsets_properties.py
for name in "MonthBegin MonthEnd BMonthBegin BMonthEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls, st.builds(cls, n=st.integers(-99, 99), normalize=st.booleans())
    )

for name in "YearBegin YearEnd BYearBegin BYearEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls,
        st.builds(
            cls,
            n=st.integers(-5, 5),
            normalize=st.booleans(),
            month=st.integers(min_value=1, max_value=12),
        ),
    )

for name in "QuarterBegin QuarterEnd BQuarterBegin BQuarterEnd".split():
    cls = getattr(pd.tseries.offsets, name)
    st.register_type_strategy(
        cls,
        st.builds(
            cls,
            n=st.integers(-24, 24),
            normalize=st.booleans(),
            startingMonth=st.integers(min_value=1, max_value=12),
        ),
    )


# ----------------------------------------------------------------
# Autouse fixtures
# ----------------------------------------------------------------


# https://github.com/pytest-dev/pytest/issues/11873
# Would like to avoid autouse=True, but cannot as of pytest 8.0.0
@pytest.fixture(autouse=True)
def add_doctest_imports(doctest_namespace) -> None:
    """
    Make `np` and `pd` names available for doctests.
    """
    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd


@pytest.fixture(autouse=True)
def configure_tests() -> None:
    """
    Configure settings for all tests and test modules.
    """
    pd.set_option("chained_assignment", "raise")


# ----------------------------------------------------------------
# Common arguments
# ----------------------------------------------------------------
@pytest.fixture(params=[0, 1, "index", "columns"], ids=lambda x: f"axis={repr(x)}")
def axis(request):
    """
    Fixture for returning the axis numbers of a DataFrame.
    """
    return request.param


axis_frame = axis


@pytest.fixture(params=[1, "columns"], ids=lambda x: f"axis={repr(x)}")
def axis_1(request):
    """
    Fixture for returning aliases of axis 1 of a DataFrame.
    """
    return request.param


@pytest.fixture(params=[True, False, None])
def observed(request):
    """
    Pass in the observed keyword to groupby for [True, False]
    This indicates whether categoricals should return values for
    values which are not in the grouper [False / None], or only values which
    appear in the grouper [True]. [None] is supported for future compatibility
    if we decide to change the default (and would need to warn if this
    parameter is not passed).
    """
    return request.param


@pytest.fixture(params=[True, False, None])
def ordered(request):
    """
    Boolean 'ordered' parameter for Categorical.
    """
    return request.param


@pytest.fixture(params=[True, False])
def skipna(request):
    """
    Boolean 'skipna' parameter.
    """
    return request.param


@pytest.fixture(params=["first", "last", False])
def keep(request):
    """
    Valid values for the 'keep' parameter used in
    .duplicated or .drop_duplicates
    """
    return request.param


@pytest.fixture(params=["both", "neither", "left", "right"])
def inclusive_endpoints_fixture(request):
    """
    Fixture for trying all interval 'inclusive' parameters.
    """
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def closed(request):
    """
    Fixture for trying all interval closed parameters.
    """
    return request.param


@pytest.fixture(params=["left", "right", "both", "neither"])
def other_closed(request):
    """
    Secondary closed fixture to allow parametrizing over all pairs of closed.
    """
    return request.param


@pytest.fixture(
    params=[
        None,
        "gzip",
        "bz2",
        "zip",
        "xz",
        "tar",
        pytest.param("zstd", marks=td.skip_if_no("zstandard")),
    ]
)
def compression(request):
    """
    Fixture for trying common compression types in compression tests.
    """
    return request.param


@pytest.fixture(
    params=[
        "gzip",
        "bz2",
        "zip",
        "xz",
        "tar",
        pytest.param("zstd", marks=td.skip_if_no("zstandard")),
    ]
)
def compression_only(request):
    """
    Fixture for trying common compression types in compression tests excluding
    uncompressed case.
    """
    return request.param


@pytest.fixture(params=[True, False])
def writable(request):
    """
    Fixture that an array is writable.
    """
    return request.param


@pytest.fixture(params=["inner", "outer", "left", "right"])
def join_type(request):
    """
    Fixture for trying all types of join operations.
    """
    return request.param


@pytest.fixture(params=["nlargest", "nsmallest"])
def nselect_method(request):
    """
    Fixture for trying all nselect methods.
    """
    return request.param


# ----------------------------------------------------------------
# Missing values & co.
# ----------------------------------------------------------------
@pytest.fixture(params=tm.NULL_OBJECTS, ids=lambda x: type(x).__name__)
def nulls_fixture(request):
    """
    Fixture for each null type in pandas.
    """
    return request.param


nulls_fixture2 = nulls_fixture  # Generate cartesian product of nulls_fixture


@pytest.fixture(params=[None, np.nan, pd.NaT])
def unique_nulls_fixture(request):
    """
    Fixture for each null type in pandas, each null type exactly once.
    """
    return request.param


# Generate cartesian product of unique_nulls_fixture:
unique_nulls_fixture2 = unique_nulls_fixture


@pytest.fixture(params=tm.NP_NAT_OBJECTS, ids=lambda x: type(x).__name__)
def np_nat_fixture(request):
    """
    Fixture for each NaT type in numpy.
    """
    return request.param


# Generate cartesian product of np_nat_fixture:
np_nat_fixture2 = np_nat_fixture


# ----------------------------------------------------------------
# Classes
# ----------------------------------------------------------------


@pytest.fixture(params=[DataFrame, Series])
def frame_or_series(request):
    """
    Fixture to parametrize over DataFrame and Series.
    """
    return request.param


@pytest.fixture(params=[Index, Series], ids=["index", "series"])
def index_or_series(request):
    """
    Fixture to parametrize over Index and Series, made necessary by a mypy
    bug, giving an error:

    List item 0 has incompatible type "Type[Series]"; expected "Type[PandasObject]"

    See GH#29725
    """
    return request.param


# Generate cartesian product of index_or_series fixture:
index_or_series2 = index_or_series


@pytest.fixture(params=[Index, Series, pd.array], ids=["index", "series", "array"])
def index_or_series_or_array(request):
    """
    Fixture to parametrize over Index, Series, and ExtensionArray
    """
    return request.param


@pytest.fixture(params=[Index, Series, DataFrame, pd.array], ids=lambda x: x.__name__)
def box_with_array(request):
    """
    Fixture to test behavior for Index, Series, DataFrame, and pandas Array
    classes
    """
    return request.param


box_with_array2 = box_with_array


@pytest.fixture
def dict_subclass() -> type[dict]:
    """
    Fixture for a dictionary subclass.
    """

    class TestSubDict(dict):
        def __init__(self, *args, **kwargs) -> None:
            dict.__init__(self, *args, **kwargs)

    return TestSubDict


@pytest.fixture
def non_dict_mapping_subclass() -> type[abc.Mapping]:
    """
    Fixture for a non-mapping dictionary subclass.
    """

    class TestNonDictMapping(abc.Mapping):
        def __init__(self, underlying_dict) -> None:
            self._data = underlying_dict

        def __getitem__(self, key):
            return self._data.__getitem__(key)

        def __iter__(self) -> Iterator:
            return self._data.__iter__()

        def __len__(self) -> int:
            return self._data.__len__()

    return TestNonDictMapping


# ----------------------------------------------------------------
# Indices
# ----------------------------------------------------------------
@pytest.fixture
def multiindex_year_month_day_dataframe_random_data():
    """
    DataFrame with 3 level MultiIndex (year, month, day) covering
    first 100 business days from 2000-01-01 with random data
    """
    tdf = DataFrame(
        np.random.default_rng(2).standard_normal((100, 4)),
        columns=Index(list("ABCD")),
        index=date_range("2000-01-01", periods=100, freq="B"),
    )
    ymd = tdf.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day]).sum()
    # use int64 Index, to make sure things work
    ymd.index = ymd.index.set_levels([lev.astype("i8") for lev in ymd.index.levels])
    ymd.index.set_names(["year", "month", "day"], inplace=True)
    return ymd


@pytest.fixture
def lexsorted_two_level_string_multiindex() -> MultiIndex:
    """
    2-level MultiIndex, lexsorted, with string names.
    """
    return MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )


@pytest.fixture
def multiindex_dataframe_random_data(
    lexsorted_two_level_string_multiindex,
) -> DataFrame:
    """DataFrame with 2 level MultiIndex with random data"""
    index = lexsorted_two_level_string_multiindex
    return DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        index=index,
        columns=Index(["A", "B", "C"], name="exp"),
    )


def _create_multiindex():
    """
    MultiIndex used to test the general functionality of this object
    """

    # See Also: tests.multi.conftest.idx
    major_axis = Index(["foo", "bar", "baz", "qux"])
    minor_axis = Index(["one", "two"])

    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    index_names = ["first", "second"]
    return MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )


def _create_mi_with_dt64tz_level():
    """
    MultiIndex with a level that is a tzaware DatetimeIndex.
    """
    # GH#8367 round trip with pickle
    return MultiIndex.from_product(
        [[1, 2], ["a", "b"], date_range("20130101", periods=3, tz="US/Eastern")],
        names=["one", "two", "three"],
    )


indices_dict = {
    "object": Index([f"pandas_{i}" for i in range(100)], dtype=object),
    "string": Index([f"pandas_{i}" for i in range(100)], dtype="str"),
    "datetime": date_range("2020-01-01", periods=100),
    "datetime-tz": date_range("2020-01-01", periods=100, tz="US/Pacific"),
    "period": period_range("2020-01-01", periods=100, freq="D"),
    "timedelta": timedelta_range(start="1 day", periods=100, freq="D"),
    "range": RangeIndex(100),
    "int8": Index(np.arange(100), dtype="int8"),
    "int16": Index(np.arange(100), dtype="int16"),
    "int32": Index(np.arange(100), dtype="int32"),
    "int64": Index(np.arange(100), dtype="int64"),
    "uint8": Index(np.arange(100), dtype="uint8"),
    "uint16": Index(np.arange(100), dtype="uint16"),
    "uint32": Index(np.arange(100), dtype="uint32"),
    "uint64": Index(np.arange(100), dtype="uint64"),
    "float32": Index(np.arange(100), dtype="float32"),
    "float64": Index(np.arange(100), dtype="float64"),
    "bool-object": Index([True, False] * 5, dtype=object),
    "bool-dtype": Index([True, False] * 5, dtype=bool),
    "complex64": Index(
        np.arange(100, dtype="complex64") + 1.0j * np.arange(100, dtype="complex64")
    ),
    "complex128": Index(
        np.arange(100, dtype="complex128") + 1.0j * np.arange(100, dtype="complex128")
    ),
    "categorical": CategoricalIndex(list("abcd") * 25),
    "interval": IntervalIndex.from_breaks(np.linspace(0, 100, num=101)),
    "empty": Index([]),
    "tuples": MultiIndex.from_tuples(zip(["foo", "bar", "baz"], [1, 2, 3])),
    "mi-with-dt64tz-level": _create_mi_with_dt64tz_level(),
    "multi": _create_multiindex(),
    "repeats": Index([0, 0, 1, 1, 2, 2]),
    "nullable_int": Index(np.arange(100), dtype="Int64"),
    "nullable_uint": Index(np.arange(100), dtype="UInt16"),
    "nullable_float": Index(np.arange(100), dtype="Float32"),
    "nullable_bool": Index(np.arange(100).astype(bool), dtype="boolean"),
    "string-python": Index(
        pd.array([f"pandas_{i}" for i in range(100)], dtype="string[python]")
    ),
}
if has_pyarrow:
    idx = Index(pd.array([f"pandas_{i}" for i in range(100)], dtype="string[pyarrow]"))
    indices_dict["string-pyarrow"] = idx


@pytest.fixture(params=indices_dict.keys())
def index(request):
    """
    Fixture for many "simple" kinds of indices.

    These indices are unlikely to cover corner cases, e.g.
        - no names
        - no NaTs/NaNs
        - no values near implementation bounds
        - ...
    """
    # copy to avoid mutation, e.g. setting .name
    return indices_dict[request.param].copy()


# Needed to generate cartesian product of indices
index_fixture2 = index


@pytest.fixture(
    params=[
        key for key, value in indices_dict.items() if not isinstance(value, MultiIndex)
    ]
)
def index_flat(request):
    """
    index fixture, but excluding MultiIndex cases.
    """
    key = request.param
    return indices_dict[key].copy()


# Alias so we can test with cartesian product of index_flat
index_flat2 = index_flat


@pytest.fixture(
    params=[
        key
        for key, value in indices_dict.items()
        if not (
            key.startswith(("int", "uint", "float"))
            or key in ["range", "empty", "repeats", "bool-dtype"]
        )
        and not isinstance(value, MultiIndex)
    ]
)
def index_with_missing(request):
    """
    Fixture for indices with missing values.

    Integer-dtype and empty cases are excluded because they cannot hold missing
    values.

    MultiIndex is excluded because isna() is not defined for MultiIndex.
    """

    # GH 35538. Use deep copy to avoid illusive bug on np-dev
    # GHA pipeline that writes into indices_dict despite copy
    ind = indices_dict[request.param].copy(deep=True)
    vals = ind.values.copy()
    if request.param in ["tuples", "mi-with-dt64tz-level", "multi"]:
        # For setting missing values in the top level of MultiIndex
        vals = ind.tolist()
        vals[0] = (None,) + vals[0][1:]
        vals[-1] = (None,) + vals[-1][1:]
        return MultiIndex.from_tuples(vals)
    else:
        vals[0] = None
        vals[-1] = None
        return type(ind)(vals)


# ----------------------------------------------------------------
# Series'
# ----------------------------------------------------------------
@pytest.fixture
def string_series() -> Series:
    """
    Fixture for Series of floats with Index of unique strings
    """
    return Series(
        np.arange(30, dtype=np.float64) * 1.1,
        index=Index([f"i_{i}" for i in range(30)]),
        name="series",
    )


@pytest.fixture
def object_series() -> Series:
    """
    Fixture for Series of dtype object with Index of unique strings
    """
    data = [f"foo_{i}" for i in range(30)]
    index = Index([f"bar_{i}" for i in range(30)])
    return Series(data, index=index, name="objects", dtype=object)


@pytest.fixture
def datetime_series() -> Series:
    """
    Fixture for Series of floats with DatetimeIndex
    """
    return Series(
        np.random.default_rng(2).standard_normal(30),
        index=date_range("2000-01-01", periods=30, freq="B"),
        name="ts",
    )


def _create_series(index):
    """Helper for the _series dict"""
    size = len(index)
    data = np.random.default_rng(2).standard_normal(size)
    return Series(data, index=index, name="a", copy=False)


_series = {
    f"series-with-{index_id}-index": _create_series(index)
    for index_id, index in indices_dict.items()
}


@pytest.fixture
def series_with_simple_index(index) -> Series:
    """
    Fixture for tests on series with changing types of indices.
    """
    return _create_series(index)


_narrow_series = {
    f"{dtype.__name__}-series": Series(
        range(30), index=[f"i-{i}" for i in range(30)], name="a", dtype=dtype
    )
    for dtype in tm.NARROW_NP_DTYPES
}


_index_or_series_objs = {**indices_dict, **_series, **_narrow_series}


@pytest.fixture(params=_index_or_series_objs.keys())
def index_or_series_obj(request):
    """
    Fixture for tests on indexes, series and series with a narrow dtype
    copy to avoid mutation, e.g. setting .name
    """
    return _index_or_series_objs[request.param].copy(deep=True)


_typ_objects_series = {
    f"{dtype.__name__}-series": Series(dtype) for dtype in tm.PYTHON_DATA_TYPES
}


_index_or_series_memory_objs = {
    **indices_dict,
    **_series,
    **_narrow_series,
    **_typ_objects_series,
}


@pytest.fixture(params=_index_or_series_memory_objs.keys())
def index_or_series_memory_obj(request):
    """
    Fixture for tests on indexes, series, series with a narrow dtype and
    series with empty objects type
    copy to avoid mutation, e.g. setting .name
    """
    return _index_or_series_memory_objs[request.param].copy(deep=True)


# ----------------------------------------------------------------
# DataFrames
# ----------------------------------------------------------------
@pytest.fixture
def int_frame() -> DataFrame:
    """
    Fixture for DataFrame of ints with index of unique strings

    Columns are ['A', 'B', 'C', 'D']
    """
    return DataFrame(
        np.ones((30, 4), dtype=np.int64),
        index=Index([f"foo_{i}" for i in range(30)]),
        columns=Index(list("ABCD")),
    )


@pytest.fixture
def float_frame() -> DataFrame:
    """
    Fixture for DataFrame of floats with index of unique strings

    Columns are ['A', 'B', 'C', 'D'].
    """
    return DataFrame(
        np.random.default_rng(2).standard_normal((30, 4)),
        index=Index([f"foo_{i}" for i in range(30)]),
        columns=Index(list("ABCD")),
    )


@pytest.fixture
def rand_series_with_duplicate_datetimeindex() -> Series:
    """
    Fixture for Series with a DatetimeIndex that has duplicates.
    """
    dates = [
        datetime(2000, 1, 2),
        datetime(2000, 1, 2),
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
        datetime(2000, 1, 3),
        datetime(2000, 1, 3),
        datetime(2000, 1, 4),
        datetime(2000, 1, 4),
        datetime(2000, 1, 4),
        datetime(2000, 1, 5),
    ]

    return Series(np.random.default_rng(2).standard_normal(len(dates)), index=dates)


# ----------------------------------------------------------------
# Scalars
# ----------------------------------------------------------------
@pytest.fixture(
    params=[
        (Interval(left=0, right=5), IntervalDtype("int64", "right")),
        (Interval(left=0.1, right=0.5), IntervalDtype("float64", "right")),
        (Period("2012-01", freq="M"), "period[M]"),
        (Period("2012-02-01", freq="D"), "period[D]"),
        (
            Timestamp("2011-01-01", tz="US/Eastern"),
            DatetimeTZDtype(unit="s", tz="US/Eastern"),
        ),
        (Timedelta(seconds=500), "timedelta64[ns]"),
    ]
)
def ea_scalar_and_dtype(request):
    return request.param


# ----------------------------------------------------------------
# Operators & Operations
# ----------------------------------------------------------------


@pytest.fixture(params=tm.arithmetic_dunder_methods)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        ops.radd,
        operator.sub,
        ops.rsub,
        operator.mul,
        ops.rmul,
        operator.truediv,
        ops.rtruediv,
        operator.floordiv,
        ops.rfloordiv,
        operator.mod,
        ops.rmod,
        operator.pow,
        ops.rpow,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
        operator.and_,
        ops.rand_,
        operator.xor,
        ops.rxor,
        operator.or_,
        ops.ror_,
    ]
)
def all_binary_operators(request):
    """
    Fixture for operator and roperator arithmetic, comparison, and logical ops.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.add,
        ops.radd,
        operator.sub,
        ops.rsub,
        operator.mul,
        ops.rmul,
        operator.truediv,
        ops.rtruediv,
        operator.floordiv,
        ops.rfloordiv,
        operator.mod,
        ops.rmod,
        operator.pow,
        ops.rpow,
    ]
)
def all_arithmetic_functions(request):
    """
    Fixture for operator and roperator arithmetic functions.

    Notes
    -----
    This includes divmod and rdivmod, whereas all_arithmetic_operators
    does not.
    """
    return request.param


_all_numeric_reductions = [
    "count",
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
    "sem",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    return request.param


_all_reductions = _all_numeric_reductions + _all_boolean_reductions


@pytest.fixture(params=_all_reductions)
def all_reductions(request):
    """
    Fixture for all (boolean + numeric) reduction names.
    """
    return request.param


@pytest.fixture(
    params=[
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        operator.lt,
        operator.le,
    ]
)
def comparison_op(request):
    """
    Fixture for operator module comparison functions.
    """
    return request.param


@pytest.fixture(params=["__le__", "__lt__", "__ge__", "__gt__"])
def compare_operators_no_eq_ne(request):
    """
    Fixture for dunder names for compare operations except == and !=

    * >=
    * >
    * <
    * <=
    """
    return request.param


@pytest.fixture(
    params=["__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__"]
)
def all_logical_operators(request):
    """
    Fixture for dunder names for common logical operations

    * |
    * &
    * ^
    """
    return request.param


_all_numeric_accumulations = ["cumsum", "cumprod", "cummin", "cummax"]


@pytest.fixture(params=_all_numeric_accumulations)
def all_numeric_accumulations(request):
    """
    Fixture for numeric accumulation names
    """
    return request.param


# ----------------------------------------------------------------
# Data sets/files
# ----------------------------------------------------------------
@pytest.fixture
def strict_data_files(pytestconfig):
    """
    Returns the configuration for the test setting `--no-strict-data-files`.
    """
    return pytestconfig.getoption("--no-strict-data-files")


@pytest.fixture
def datapath(strict_data_files: str) -> Callable[..., str]:
    """
    Get the path to a data file.

    Parameters
    ----------
    path : str
        Path to the file, relative to ``pandas/tests/``

    Returns
    -------
    path including ``pandas/tests``.

    Raises
    ------
    ValueError
        If the path doesn't exist and the --no-strict-data-files option is not set.
    """
    BASE_PATH = os.path.join(os.path.dirname(__file__), "tests")

    def deco(*args):
        path = os.path.join(BASE_PATH, *args)
        if not os.path.exists(path):
            if strict_data_files:
                raise ValueError(
                    f"Could not find file {path} and --no-strict-data-files is not set."
                )
            pytest.skip(f"Could not find {path}.")
        return path

    return deco


# ----------------------------------------------------------------
# Time zones
# ----------------------------------------------------------------
TIMEZONES = [
    None,
    "UTC",
    "US/Eastern",
    "Asia/Tokyo",
    "dateutil/US/Pacific",
    "dateutil/Asia/Singapore",
    "+01:15",
    "-02:15",
    "UTC+01:15",
    "UTC-02:15",
    tzutc(),
    tzlocal(),
    FixedOffset(300),
    FixedOffset(0),
    FixedOffset(-300),
    timezone.utc,
    timezone(timedelta(hours=1)),
    timezone(timedelta(hours=-1), name="foo"),
]
if zoneinfo is not None:
    TIMEZONES.extend(
        [
            zoneinfo.ZoneInfo("US/Pacific"),  # type: ignore[list-item]
            zoneinfo.ZoneInfo("UTC"),  # type: ignore[list-item]
        ]
    )
TIMEZONE_IDS = [repr(i) for i in TIMEZONES]


@td.parametrize_fixture_doc(str(TIMEZONE_IDS))
@pytest.fixture(params=TIMEZONES, ids=TIMEZONE_IDS)
def tz_naive_fixture(request):
    """
    Fixture for trying timezones including default (None): {0}
    """
    return request.param


@td.parametrize_fixture_doc(str(TIMEZONE_IDS[1:]))
@pytest.fixture(params=TIMEZONES[1:], ids=TIMEZONE_IDS[1:])
def tz_aware_fixture(request):
    """
    Fixture for trying explicit timezones: {0}
    """
    return request.param


# Generate cartesian product of tz_aware_fixture:
tz_aware_fixture2 = tz_aware_fixture


_UTCS = ["utc", "dateutil/UTC", utc, tzutc(), timezone.utc]
if zoneinfo is not None:
    _UTCS.append(zoneinfo.ZoneInfo("UTC"))


@pytest.fixture(params=_UTCS)
def utc_fixture(request):
    """
    Fixture to provide variants of UTC timezone strings and tzinfo objects.
    """
    return request.param


utc_fixture2 = utc_fixture


@pytest.fixture(params=["s", "ms", "us", "ns"])
def unit(request):
    """
    datetime64 units we support.
    """
    return request.param


unit2 = unit


# ----------------------------------------------------------------
# Dtypes
# ----------------------------------------------------------------
@pytest.fixture(params=tm.STRING_DTYPES)
def string_dtype(request):
    """
    Parametrized fixture for string dtypes.

    * str
    * 'str'
    * 'U'
    """
    return request.param


@pytest.fixture(
    params=[
        ("python", pd.NA),
        pytest.param(("pyarrow", pd.NA), marks=td.skip_if_no("pyarrow")),
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
        ("python", np.nan),
    ],
    ids=[
        "string=string[python]",
        "string=string[pyarrow]",
        "string=str[pyarrow]",
        "string=str[python]",
    ],
)
def string_dtype_no_object(request):
    """
    Parametrized fixture for string dtypes.
    * 'string[python]' (NA variant)
    * 'string[pyarrow]' (NA variant)
    * 'str' (NaN variant, with pyarrow)
    * 'str' (NaN variant, without pyarrow)
    """
    # need to instantiate the StringDtype here instead of in the params
    # to avoid importing pyarrow during test collection
    storage, na_value = request.param
    return pd.StringDtype(storage, na_value)


@pytest.fixture(
    params=[
        "string[python]",
        pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
    ]
)
def nullable_string_dtype(request):
    """
    Parametrized fixture for string dtypes.

    * 'string[python]'
    * 'string[pyarrow]'
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
        pytest.param(("pyarrow", pd.NA), marks=td.skip_if_no("pyarrow")),
    ]
)
def pyarrow_string_dtype(request):
    """
    Parametrized fixture for string dtypes backed by Pyarrow.

    * 'str[pyarrow]'
    * 'string[pyarrow]'
    """
    return pd.StringDtype(*request.param)


@pytest.fixture(
    params=[
        "python",
        pytest.param("pyarrow", marks=td.skip_if_no("pyarrow")),
    ]
)
def string_storage(request):
    """
    Parametrized fixture for pd.options.mode.string_storage.

    * 'python'
    * 'pyarrow'
    """
    return request.param


@pytest.fixture(
    params=[
        ("python", pd.NA),
        pytest.param(("pyarrow", pd.NA), marks=td.skip_if_no("pyarrow")),
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
        ("python", np.nan),
    ],
    ids=[
        "string=string[python]",
        "string=string[pyarrow]",
        "string=str[pyarrow]",
        "string=str[python]",
    ],
)
def string_dtype_arguments(request):
    """
    Parametrized fixture for StringDtype storage and na_value.

    * 'python' + pd.NA
    * 'pyarrow' + pd.NA
    * 'pyarrow' + np.nan
    """
    return request.param


@pytest.fixture(
    params=[
        "numpy_nullable",
        pytest.param("pyarrow", marks=td.skip_if_no("pyarrow")),
    ]
)
def dtype_backend(request):
    """
    Parametrized fixture for pd.options.mode.string_storage.

    * 'python'
    * 'pyarrow'
    """
    return request.param


# Alias so we can test with cartesian product of string_storage
string_storage2 = string_storage
string_dtype_arguments2 = string_dtype_arguments


@pytest.fixture(params=tm.BYTES_DTYPES)
def bytes_dtype(request):
    """
    Parametrized fixture for bytes dtypes.

    * bytes
    * 'bytes'
    """
    return request.param


@pytest.fixture(params=tm.OBJECT_DTYPES)
def object_dtype(request):
    """
    Parametrized fixture for object dtypes.

    * object
    * 'object'
    """
    return request.param


@pytest.fixture(
    params=[
        np.dtype("object"),
        ("python", pd.NA),
        pytest.param(("pyarrow", pd.NA), marks=td.skip_if_no("pyarrow")),
        pytest.param(("pyarrow", np.nan), marks=td.skip_if_no("pyarrow")),
        ("python", np.nan),
    ],
    ids=[
        "string=object",
        "string=string[python]",
        "string=string[pyarrow]",
        "string=str[pyarrow]",
        "string=str[python]",
    ],
)
def any_string_dtype(request):
    """
    Parametrized fixture for string dtypes.
    * 'object'
    * 'string[python]' (NA variant)
    * 'string[pyarrow]' (NA variant)
    * 'str' (NaN variant, with pyarrow)
    * 'str' (NaN variant, without pyarrow)
    """
    if isinstance(request.param, np.dtype):
        return request.param
    else:
        # need to instantiate the StringDtype here instead of in the params
        # to avoid importing pyarrow during test collection
        storage, na_value = request.param
        return pd.StringDtype(storage, na_value)


@pytest.fixture(params=tm.DATETIME64_DTYPES)
def datetime64_dtype(request):
    """
    Parametrized fixture for datetime64 dtypes.

    * 'datetime64[ns]'
    * 'M8[ns]'
    """
    return request.param


@pytest.fixture(params=tm.TIMEDELTA64_DTYPES)
def timedelta64_dtype(request):
    """
    Parametrized fixture for timedelta64 dtypes.

    * 'timedelta64[ns]'
    * 'm8[ns]'
    """
    return request.param


@pytest.fixture
def fixed_now_ts() -> Timestamp:
    """
    Fixture emits fixed Timestamp.now()
    """
    return Timestamp(  # pyright: ignore[reportGeneralTypeIssues]
        year=2021, month=1, day=1, hour=12, minute=4, second=13, microsecond=22
    )


@pytest.fixture(params=tm.FLOAT_NUMPY_DTYPES)
def float_numpy_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    """
    return request.param


@pytest.fixture(params=tm.FLOAT_EA_DTYPES)
def float_ea_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * 'Float32'
    * 'Float64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_FLOAT_DTYPES)
def any_float_dtype(request):
    """
    Parameterized fixture for float dtypes.

    * float
    * 'float32'
    * 'float64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


@pytest.fixture(params=tm.COMPLEX_DTYPES)
def complex_dtype(request):
    """
    Parameterized fixture for complex dtypes.

    * complex
    * 'complex64'
    * 'complex128'
    """
    return request.param


@pytest.fixture(params=tm.COMPLEX_FLOAT_DTYPES)
def complex_or_float_dtype(request):
    """
    Parameterized fixture for complex and numpy float dtypes.

    * complex
    * 'complex64'
    * 'complex128'
    * float
    * 'float32'
    * 'float64'
    """
    return request.param


@pytest.fixture(params=tm.SIGNED_INT_NUMPY_DTYPES)
def any_signed_int_numpy_dtype(request):
    """
    Parameterized fixture for signed integer dtypes.

    * int
    * 'int8'
    * 'int16'
    * 'int32'
    * 'int64'
    """
    return request.param


@pytest.fixture(params=tm.UNSIGNED_INT_NUMPY_DTYPES)
def any_unsigned_int_numpy_dtype(request):
    """
    Parameterized fixture for unsigned integer dtypes.

    * 'uint8'
    * 'uint16'
    * 'uint32'
    * 'uint64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_NUMPY_DTYPES)
def any_int_numpy_dtype(request):
    """
    Parameterized fixture for any integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_EA_DTYPES)
def any_int_ea_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_DTYPES)
def any_int_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_INT_EA_DTYPES + tm.FLOAT_EA_DTYPES)
def any_numeric_ea_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype and
    any float ea dtypes.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


#  Unsupported operand types for + ("List[Union[str, ExtensionDtype, dtype[Any],
#  Type[object]]]" and "List[str]")
@pytest.fixture(
    params=tm.ALL_INT_EA_DTYPES
    + tm.FLOAT_EA_DTYPES
    + tm.ALL_INT_PYARROW_DTYPES_STR_REPR
    + tm.FLOAT_PYARROW_DTYPES_STR_REPR  # type: ignore[operator]
)
def any_numeric_ea_and_arrow_dtype(request):
    """
    Parameterized fixture for any nullable integer dtype and
    any float ea dtypes.

    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    * 'uint8[pyarrow]'
    * 'int8[pyarrow]'
    * 'uint16[pyarrow]'
    * 'int16[pyarrow]'
    * 'uint32[pyarrow]'
    * 'int32[pyarrow]'
    * 'uint64[pyarrow]'
    * 'int64[pyarrow]'
    * 'float32[pyarrow]'
    * 'float64[pyarrow]'
    """
    return request.param


@pytest.fixture(params=tm.SIGNED_INT_EA_DTYPES)
def any_signed_int_ea_dtype(request):
    """
    Parameterized fixture for any signed nullable integer dtype.

    * 'Int8'
    * 'Int16'
    * 'Int32'
    * 'Int64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_NUMPY_DTYPES)
def any_real_numpy_dtype(request):
    """
    Parameterized fixture for any (purely) real numeric dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_DTYPES)
def any_real_numeric_dtype(request):
    """
    Parameterized fixture for any (purely) real numeric dtype.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'

    and associated ea dtypes.
    """
    return request.param


@pytest.fixture(params=tm.ALL_NUMPY_DTYPES)
def any_numpy_dtype(request):
    """
    Parameterized fixture for all numpy dtypes.

    * bool
    * 'bool'
    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    * complex
    * 'complex64'
    * 'complex128'
    * str
    * 'str'
    * 'U'
    * bytes
    * 'bytes'
    * 'datetime64[ns]'
    * 'M8[ns]'
    * 'timedelta64[ns]'
    * 'm8[ns]'
    * object
    * 'object'
    """
    return request.param


@pytest.fixture(params=tm.ALL_REAL_NULLABLE_DTYPES)
def any_real_nullable_dtype(request):
    """
    Parameterized fixture for all real dtypes that can hold NA.

    * float
    * 'float32'
    * 'float64'
    * 'Float32'
    * 'Float64'
    * 'UInt8'
    * 'UInt16'
    * 'UInt32'
    * 'UInt64'
    * 'Int8'
    * 'Int16'
    * 'Int32'
    * 'Int64'
    * 'uint8[pyarrow]'
    * 'uint16[pyarrow]'
    * 'uint32[pyarrow]'
    * 'uint64[pyarrow]'
    * 'int8[pyarrow]'
    * 'int16[pyarrow]'
    * 'int32[pyarrow]'
    * 'int64[pyarrow]'
    * 'float[pyarrow]'
    * 'double[pyarrow]'
    """
    return request.param


@pytest.fixture(params=tm.ALL_NUMERIC_DTYPES)
def any_numeric_dtype(request):
    """
    Parameterized fixture for all numeric dtypes.

    * int
    * 'int8'
    * 'uint8'
    * 'int16'
    * 'uint16'
    * 'int32'
    * 'uint32'
    * 'int64'
    * 'uint64'
    * float
    * 'float32'
    * 'float64'
    * complex
    * 'complex64'
    * 'complex128'
    * 'UInt8'
    * 'Int8'
    * 'UInt16'
    * 'Int16'
    * 'UInt32'
    * 'Int32'
    * 'UInt64'
    * 'Int64'
    * 'Float32'
    * 'Float64'
    """
    return request.param


# categoricals are handled separately
_any_skipna_inferred_dtype = [
    ("string", ["a", np.nan, "c"]),
    ("string", ["a", pd.NA, "c"]),
    ("mixed", ["a", pd.NaT, "c"]),  # pd.NaT not considered valid by is_string_array
    ("bytes", [b"a", np.nan, b"c"]),
    ("empty", [np.nan, np.nan, np.nan]),
    ("empty", []),
    ("mixed-integer", ["a", np.nan, 2]),
    ("mixed", ["a", np.nan, 2.0]),
    ("floating", [1.0, np.nan, 2.0]),
    ("integer", [1, np.nan, 2]),
    ("mixed-integer-float", [1, np.nan, 2.0]),
    ("decimal", [Decimal(1), np.nan, Decimal(2)]),
    ("boolean", [True, np.nan, False]),
    ("boolean", [True, pd.NA, False]),
    ("datetime64", [np.datetime64("2013-01-01"), np.nan, np.datetime64("2018-01-01")]),
    ("datetime", [Timestamp("20130101"), np.nan, Timestamp("20180101")]),
    ("date", [date(2013, 1, 1), np.nan, date(2018, 1, 1)]),
    ("complex", [1 + 1j, np.nan, 2 + 2j]),
    # The following dtype is commented out due to GH 23554
    # ('timedelta64', [np.timedelta64(1, 'D'),
    #                  np.nan, np.timedelta64(2, 'D')]),
    ("timedelta", [timedelta(1), np.nan, timedelta(2)]),
    ("time", [time(1), np.nan, time(2)]),
    ("period", [Period(2013), pd.NaT, Period(2018)]),
    ("interval", [Interval(0, 1), np.nan, Interval(0, 2)]),
]
ids, _ = zip(*_any_skipna_inferred_dtype)  # use inferred type as fixture-id


@pytest.fixture(params=_any_skipna_inferred_dtype, ids=ids)
def any_skipna_inferred_dtype(request):
    """
    Fixture for all inferred dtypes from _libs.lib.infer_dtype

    The covered (inferred) types are:
    * 'string'
    * 'empty'
    * 'bytes'
    * 'mixed'
    * 'mixed-integer'
    * 'mixed-integer-float'
    * 'floating'
    * 'integer'
    * 'decimal'
    * 'boolean'
    * 'datetime64'
    * 'datetime'
    * 'date'
    * 'timedelta'
    * 'time'
    * 'period'
    * 'interval'

    Returns
    -------
    inferred_dtype : str
        The string for the inferred dtype from _libs.lib.infer_dtype
    values : np.ndarray
        An array of object dtype that will be inferred to have
        `inferred_dtype`

    Examples
    --------
    >>> from pandas._libs import lib
    >>>
    >>> def test_something(any_skipna_inferred_dtype):
    ...     inferred_dtype, values = any_skipna_inferred_dtype
    ...     # will pass
    ...     assert lib.infer_dtype(values, skipna=True) == inferred_dtype
    """
    inferred_dtype, values = request.param
    values = np.array(values, dtype=object)  # object dtype to avoid casting

    # correctness of inference tested in tests/dtypes/test_inference.py
    return inferred_dtype, values


# ----------------------------------------------------------------
# Misc
# ----------------------------------------------------------------
@pytest.fixture
def ip():
    """
    Get an instance of IPython.InteractiveShell.

    Will raise a skip if IPython is not installed.
    """
    pytest.importorskip("IPython", minversion="6.0.0")
    from IPython.core.interactiveshell import InteractiveShell

    # GH#35711 make sure sqlite history file handle is not leaked
    from traitlets.config import Config  # isort:skip

    c = Config()
    c.HistoryManager.hist_file = ":memory:"

    return InteractiveShell(config=c)


@pytest.fixture(params=["bsr", "coo", "csc", "csr", "dia", "dok", "lil"])
def spmatrix(request):
    """
    Yields scipy sparse matrix classes.
    """
    sparse = pytest.importorskip("scipy.sparse")

    return getattr(sparse, request.param + "_matrix")


@pytest.fixture(
    params=[
        getattr(pd.offsets, o)
        for o in pd.offsets.__all__
        if issubclass(getattr(pd.offsets, o), pd.offsets.Tick) and o != "Tick"
    ]
)
def tick_classes(request):
    """
    Fixture for Tick based datetime offsets available for a time series.
    """
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


@pytest.fixture(
    params=[
        ("foo", None, None),
        ("Egon", "Venkman", None),
        ("NCC1701D", "NCC1701D", "NCC1701D"),
        # possibly-matching NAs
        (np.nan, np.nan, np.nan),
        (np.nan, pd.NaT, None),
        (np.nan, pd.NA, None),
        (pd.NA, pd.NA, pd.NA),
    ]
)
def names(request) -> tuple[Hashable, Hashable, Hashable]:
    """
    A 3-tuple of names, the first two for operands, the last for a result.
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.loc, tm.iloc])
def indexer_sli(request):
    """
    Parametrize over __setitem__, loc.__setitem__, iloc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.loc, tm.iloc])
def indexer_li(request):
    """
    Parametrize over loc.__getitem__, iloc.__getitem__
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.iloc])
def indexer_si(request):
    """
    Parametrize over __setitem__, iloc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.setitem, tm.loc])
def indexer_sl(request):
    """
    Parametrize over __setitem__, loc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.at, tm.loc])
def indexer_al(request):
    """
    Parametrize over at.__setitem__, loc.__setitem__
    """
    return request.param


@pytest.fixture(params=[tm.iat, tm.iloc])
def indexer_ial(request):
    """
    Parametrize over iat.__setitem__, iloc.__setitem__
    """
    return request.param


@pytest.fixture
def using_array_manager() -> bool:
    """
    Fixture to check if the array manager is being used.
    """
    return _get_option("mode.data_manager", silent=True) == "array"


@pytest.fixture
def using_copy_on_write() -> bool:
    """
    Fixture to check if Copy-on-Write is enabled.
    """
    return (
        pd.options.mode.copy_on_write is True
        and _get_option("mode.data_manager", silent=True) == "block"
    )


@pytest.fixture
def warn_copy_on_write() -> bool:
    """
    Fixture to check if Copy-on-Write is in warning mode.
    """
    return (
        pd.options.mode.copy_on_write == "warn"
        and _get_option("mode.data_manager", silent=True) == "block"
    )


@pytest.fixture
def using_infer_string() -> bool:
    """
    Fixture to check if infer string option is enabled.
    """
    return pd.options.future.infer_string is True


warsaws = ["Europe/Warsaw", "dateutil/Europe/Warsaw"]
if zoneinfo is not None:
    warsaws.append(zoneinfo.ZoneInfo("Europe/Warsaw"))  # type: ignore[arg-type]


@pytest.fixture(params=warsaws)
def warsaw(request) -> str:
    """
    tzinfo for Europe/Warsaw using pytz, dateutil, or zoneinfo.
    """
    return request.param


@pytest.fixture()
def arrow_string_storage():
    return ("pyarrow", "pyarrow_numpy")