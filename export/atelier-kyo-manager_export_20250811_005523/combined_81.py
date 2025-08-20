
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\add.py ===
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from collections import defaultdict
from functools import reduce
from operator import attrgetter
from .basic import _args_sortkey
from .parameters import global_parameters
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .intfunc import ilcm, igcd
from .expr import Expr
from .kind import UndefinedKind
from sympy.utilities.iterables import is_sequence, sift


if TYPE_CHECKING:
    from sympy.core.numbers import Number
    from sympy.series.order import Order


def _could_extract_minus_sign(expr):
    # assume expr is Add-like
    # We choose the one with less arguments with minus signs
    negative_args = sum(1 for i in expr.args
        if i.could_extract_minus_sign())
    positive_args = len(expr.args) - negative_args
    if positive_args > negative_args:
        return False
    elif positive_args < negative_args:
        return True
    # choose based on .sort_key() to prefer
    # x - 1 instead of 1 - x and
    # 3 - sqrt(2) instead of -3 + sqrt(2)
    return bool(expr.sort_key() < (-expr).sort_key())


def _addsort(args):
    # in-place sorting of args
    args.sort(key=_args_sortkey)


def _unevaluated_Add(*args):
    """Return a well-formed unevaluated Add: Numbers are collected and
    put in slot 0 and args are sorted. Use this when args have changed
    but you still want to return an unevaluated Add.

    Examples
    ========

    >>> from sympy.core.add import _unevaluated_Add as uAdd
    >>> from sympy import S, Add
    >>> from sympy.abc import x, y
    >>> a = uAdd(*[S(1.0), x, S(2)])
    >>> a.args[0]
    3.00000000000000
    >>> a.args[1]
    x

    Beyond the Number being in slot 0, there is no other assurance of
    order for the arguments since they are hash sorted. So, for testing
    purposes, output produced by this in some other function can only
    be tested against the output of this function or as one of several
    options:

    >>> opts = (Add(x, y, evaluate=False), Add(y, x, evaluate=False))
    >>> a = uAdd(x, y)
    >>> assert a in opts and a == uAdd(x, y)
    >>> uAdd(x + 1, x + 2)
    x + x + 3
    """
    args = list(args)
    newargs = []
    co = S.Zero
    while args:
        a = args.pop()
        if a.is_Add:
            # this will keep nesting from building up
            # so that x + (x + 1) -> x + x + 1 (3 args)
            args.extend(a.args)
        elif a.is_Number:
            co += a
        else:
            newargs.append(a)
    _addsort(newargs)
    if co:
        newargs.insert(0, co)
    return Add._from_args(newargs)


class Add(Expr, AssocOp):
    """
    Expression representing addition operation for algebraic group.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Add()`` must be ``Expr``. Infix operator ``+``
    on most scalar objects in SymPy calls this class.

    Another use of ``Add()`` is to represent the structure of abstract
    addition so that its arguments can be substituted to return different
    class. Refer to examples section for this.

    ``Add()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Add(x, Add(y, z))`` -> ``Add(x, y, z)``

    2. Identity removing
        ``Add(x, 0, y)`` -> ``Add(x, y)``

    3. Coefficient collecting by ``.as_coeff_Mul()``
        ``Add(x, 2*x)`` -> ``Mul(3, x)``

    4. Term sorting
        ``Add(y, x, 2)`` -> ``Add(2, x, y)``

    If no argument is passed, identity element 0 is returned. If single
    element is passed, that element is returned.

    Note that ``Add(*args)`` is more efficient than ``sum(args)`` because
    it flattens the arguments. ``sum(a, b, c, ...)`` recursively adds the
    arguments as ``a + (b + (c + ...))``, which has quadratic complexity.
    On the other hand, ``Add(a, b, c, d)`` does not assume nested
    structure, making the complexity linear.

    Since addition is group operation, every argument should have the
    same :obj:`sympy.core.kind.Kind()`.

    Examples
    ========

    >>> from sympy import Add, I
    >>> from sympy.abc import x, y
    >>> Add(x, 1)
    x + 1
    >>> Add(x, x)
    2*x
    >>> 2*x**2 + 3*x + I*y + 2*y + 2*x/5 + 1.0*y + 1
    2*x**2 + 17*x/5 + 3.0*y + I*y + 1

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Add(1, 2, evaluate=False)
    1 + 2
    >>> Add(x, x, evaluate=False)
    x + x

    ``Add()`` also represents the general structure of addition operation.

    >>> from sympy import MatrixSymbol
    >>> A,B = MatrixSymbol('A', 2,2), MatrixSymbol('B', 2,2)
    >>> expr = Add(x,y).subs({x:A, y:B})
    >>> expr
    A + B
    >>> type(expr)
    <class 'sympy.matrices.expressions.matadd.MatAdd'>

    Note that the printers do not display in args order.

    >>> Add(x, 1)
    x + 1
    >>> Add(x, 1).args
    (1, x)

    See Also
    ========

    MatAdd

    """

    __slots__ = ()

    is_Add = True

    _args_type = Expr

    identity: ClassVar[Expr]

    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr: # type: ignore
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    @classmethod
    def flatten(cls, seq: list[Expr]) -> tuple[list[Expr], list[Expr], None]:
        """
        Takes the sequence "seq" of nested Adds and returns a flatten list.

        Returns: (commutative_part, noncommutative_part, order_symbols)

        Applies associativity, all terms are commutable with respect to
        addition.

        NB: the removal of 0 is already handled by AssocOp.__new__

        See Also
        ========

        sympy.core.mul.Mul.flatten

        """
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.matrices.expressions import MatrixExpr
        from sympy.tensor.tensor import TensExpr, TensAdd
        rv = None
        if len(seq) == 2:
            a, b = seq
            if b.is_Rational:
                a, b = b, a
            if a.is_Rational:
                if b.is_Mul:
                    rv = [a, b], [], None
            if rv:
                if all(s.is_commutative for s in rv[0]):
                    return rv
                return [], rv[0], None

        # term -> coeff
        # e.g. x**2 -> 5   for ... + 5*x**2 + ...
        terms: dict[Expr, Number] = {}

        # coefficient (Number or zoo) to always be in slot 0
        # e.g. 3 + ...
        coeff: Expr = S.Zero

        order_factors: list[Order] = []

        extra: list[MatrixExpr] = []

        for o in seq:

            # O(x)
            if o.is_Order:
                if o.expr.is_zero: # type: ignore
                    continue
                if any(o1.contains(o) for o1 in order_factors):
                    continue
                order_factors = [o1 for o1 in order_factors if not o.contains(o1)] # type: ignore
                order_factors = [o] + order_factors # type: ignore
                continue

            # 3 or NaN
            elif o.is_Number:
                if (o is S.NaN or coeff is S.ComplexInfinity and
                        o.is_finite is False) and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                if coeff.is_Number or isinstance(coeff, AccumBounds):
                    coeff += o
                    if coeff is S.NaN and not extra:
                        # we know for sure the result will be nan
                        return [S.NaN], [], None
                continue

            elif isinstance(o, AccumBounds):
                coeff = o.__add__(coeff)
                continue

            elif isinstance(o, MatrixExpr):
                # can't add 0 to Matrix so make sure coeff is not 0
                extra.append(o)
                continue

            elif isinstance(o, TensExpr):
                coeff = TensAdd(o, coeff).doit(deep=False)
                continue

            elif o is S.ComplexInfinity:
                if coeff.is_finite is False and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                coeff = S.ComplexInfinity
                continue

            # Add([...])
            elif o.is_Add:
                # NB: here we assume Add is always commutative
                o_args: tuple[Expr, ...] = o.args # type: ignore
                seq.extend(o_args)  # TODO zerocopy?
                continue

            # Mul([...])
            elif o.is_Mul:
                c, s = o.as_coeff_Mul()

            # check for unevaluated Pow, e.g. 2**3 or 2**(-1/2)
            elif o.is_Pow:
                b, e = o.as_base_exp()
                if b.is_Number and (e.is_Integer or
                                   (e.is_Rational and e.is_negative)):
                    seq.append(b**e)
                    continue
                c, s = S.One, o

            else:
                # everything else
                c = S.One
                s = o

            # now we have:
            # o = c*s, where
            #
            # c is a Number
            # s is an expression with number factor extracted
            # let's collect terms with the same s, so e.g.
            # 2*x**2 + 3*x**2  ->  5*x**2
            if s in terms:
                terms[s] += c
                if terms[s] is S.NaN and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
            else:
                terms[s] = c

        # now let's construct new args:
        # [2*x**2, x**3, 7*x**4, pi, ...]
        newseq = []
        noncommutative = False
        for s, c in terms.items():
            # 0*s
            if c.is_zero:
                continue
            # 1*s
            elif c is S.One:
                newseq.append(s)
            # c*s
            else:
                if s.is_Mul:
                    # Mul, already keeps its arguments in perfect order.
                    # so we can simply put c in slot0 and go the fast way.
                    #
                    # XXX: This breaks VectorMul unless it overrides
                    # _new_rawargs
                    cs = s._new_rawargs(*((c,) + s.args)) # type: ignore
                    newseq.append(cs)
                elif s.is_Add:
                    # we just re-create the unevaluated Mul
                    newseq.append(Mul(c, s, evaluate=False))
                else:
                    # alternatively we have to call all Mul's machinery (slow)
                    newseq.append(Mul(c, s))

            noncommutative = noncommutative or not s.is_commutative

        # oo, -oo
        if coeff is S.Infinity:
            newseq = [f for f in newseq if not (f.is_extended_nonnegative or f.is_real)]

        elif coeff is S.NegativeInfinity:
            newseq = [f for f in newseq if not (f.is_extended_nonpositive or f.is_real)]

        if coeff is S.ComplexInfinity:
            # zoo might be
            #   infinite_real + finite_im
            #   finite_real + infinite_im
            #   infinite_real + infinite_im
            # addition of a finite real or imaginary number won't be able to
            # change the zoo nature; adding an infinite qualtity would result
            # in a NaN condition if it had sign opposite of the infinite
            # portion of zoo, e.g., infinite_real - infinite_real.
            newseq = [c for c in newseq if not (c.is_finite and
                                                c.is_extended_real is not None)]

        # process O(x)
        if order_factors:
            newseq2 = []
            for t in newseq:
                # x + O(x) -> O(x)
                if not any(o.contains(t) for o in order_factors):
                    newseq2.append(t)
            newseq = newseq2 + order_factors # type: ignore
            # 1 + O(1) -> O(1)
            for o in order_factors:
                if o.contains(coeff):
                    coeff = S.Zero
                    break

        # order args canonically
        _addsort(newseq)

        # current code expects coeff to be first
        if coeff is not S.Zero:
            newseq.insert(0, coeff)

        if extra:
            newseq += extra
            noncommutative = True

        # we are done
        if noncommutative:
            return [], newseq, None
        else:
            return newseq, [], None

    @classmethod
    def class_key(cls):
        return 3, 1, cls.__name__

    @property
    def kind(self):
        k = attrgetter('kind')
        kinds = map(k, self.args)
        kinds = frozenset(kinds)
        if len(kinds) != 1:
            # Since addition is group operator, kind must be same.
            # We know that this is unexpected signature, so return this.
            result = UndefinedKind
        else:
            result, = kinds
        return result

    def could_extract_minus_sign(self):
        return _could_extract_minus_sign(self)

    @cacheit
    def as_coeff_add(self, *deps):
        """
        Returns a tuple (coeff, args) where self is treated as an Add and coeff
        is the Number term and args is a tuple of all other terms.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (7 + 3*x).as_coeff_add()
        (7, (3*x,))
        >>> (7*x).as_coeff_add()
        (0, (7*x,))
        """
        if deps:
            l1, l2 = sift(self.args, lambda x: x.has_free(*deps), binary=True)
            return self._new_rawargs(*l2), tuple(l1)
        coeff, notrat = self.args[0].as_coeff_add()
        if coeff is not S.Zero:
            return coeff, notrat + self.args[1:]
        return S.Zero, self.args

    def as_coeff_Add(self, rational=False, deps=None) -> tuple[Number, Expr]:
        """
        Efficiently extract the coefficient of a summation.
        """
        coeff, args = self.args[0], self.args[1:]

        if coeff.is_Number and not rational or coeff.is_Rational:
            return coeff, self._new_rawargs(*args) # type: ignore
        return S.Zero, self

    # Note, we intentionally do not implement Add.as_coeff_mul().  Rather, we
    # let Expr.as_coeff_mul() just always return (S.One, self) for an Add.  See
    # issue 5524.

    def _eval_power(self, expt):
        from .evalf import pure_complex
        from .relational import is_eq
        if len(self.args) == 2 and any(_.is_infinite for _ in self.args):
            if expt.is_zero is False and is_eq(expt, S.One) is False:
                # looking for literal a + I*b
                a, b = self.args
                if a.coeff(S.ImaginaryUnit):
                    a, b = b, a
                ico = b.coeff(S.ImaginaryUnit)
                if ico and ico.is_extended_real and a.is_extended_real:
                    if expt.is_extended_negative:
                        return S.Zero
                    if expt.is_extended_positive:
                        return S.ComplexInfinity
            return
        if expt.is_Rational and self.is_number:
            ri = pure_complex(self)
            if ri:
                r, i = ri
                if expt.q == 2:
                    from sympy.functions.elementary.miscellaneous import sqrt
                    D = sqrt(r**2 + i**2)
                    if D.is_Rational:
                        from .exprtools import factor_terms
                        from sympy.functions.elementary.complexes import sign
                        from .function import expand_multinomial
                        # (r, i, D) is a Pythagorean triple
                        root = sqrt(factor_terms((D - r)/2))**expt.p
                        return root*expand_multinomial((
                            # principle value
                            (D + r)/abs(i) + sign(i)*S.ImaginaryUnit)**expt.p)
                elif expt == -1:
                    return _unevaluated_Mul(
                        r - i*S.ImaginaryUnit,
                        1/(r**2 + i**2))

    @cacheit
    def _eval_derivative(self, s):
        return self.func(*[a.diff(s) for a in self.args])

    def _eval_nseries(self, x, n, logx, cdir=0):
        terms = [t.nseries(x, n=n, logx=logx, cdir=cdir) for t in self.args]
        return self.func(*terms)

    def _matches_simple(self, expr, repl_dict):
        # handle (w+3).matches('x+5') -> {w: x+2}
        coeff, terms = self.as_coeff_add()
        if len(terms) == 1:
            return terms[0].matches(expr - coeff, repl_dict)
        return

    def matches(self, expr, repl_dict=None, old=False):
        return self._matches_commutative(expr, repl_dict, old)

    @staticmethod
    def _combine_inverse(lhs, rhs):
        """
        Returns lhs - rhs, but treats oo like a symbol so oo - oo
        returns 0, instead of a nan.
        """
        from sympy.simplify.simplify import signsimp
        inf = (S.Infinity, S.NegativeInfinity)
        if lhs.has(*inf) or rhs.has(*inf):
            from .symbol import Dummy
            oo = Dummy('oo')
            reps = {
                S.Infinity: oo,
                S.NegativeInfinity: -oo}
            ireps = {v: k for k, v in reps.items()}
            eq = lhs.xreplace(reps) - rhs.xreplace(reps)
            if eq.has(oo):
                eq = eq.replace(
                    lambda x: x.is_Pow and x.base is oo,
                    lambda x: x.base)
            rv = eq.xreplace(ireps)
        else:
            rv = lhs - rhs
        srv = signsimp(rv)
        return srv if srv.is_Number else rv

    @cacheit
    def as_two_terms(self):
        """Return head and tail of self.

        This is the most efficient way to get the head and tail of an
        expression.

        - if you want only the head, use self.args[0];
        - if you want to process the arguments of the tail then use
          self.as_coef_add() which gives the head and a tuple containing
          the arguments of the tail when treated as an Add.
        - if you want the coefficient when self is treated as a Mul
          then use self.as_coeff_mul()[0]

        >>> from sympy.abc import x, y
        >>> (3*x - 2*y + 5).as_two_terms()
        (5, 3*x - 2*y)
        """
        return self.args[0], self._new_rawargs(*self.args[1:])

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        """
        Decomposes an expression to its numerator part and its
        denominator part.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> (x*y/z).as_numer_denom()
        (x*y, z)
        >>> (x*(y + 1)/y**7).as_numer_denom()
        (x*(y + 1), y**7)

        See Also
        ========

        sympy.core.expr.Expr.as_numer_denom
        """
        # clear rational denominator
        content, expr = self.primitive()
        if not isinstance(expr, Add):
            return Mul(content, expr, evaluate=False).as_numer_denom()
        ncon, dcon = content.as_numer_denom()

        # collect numerators and denominators of the terms
        nd = defaultdict(list)
        for f in expr.args:
            ni, di = f.as_numer_denom()
            nd[di].append(ni)

        # check for quick exit
        if len(nd) == 1:
            d, n = nd.popitem()
            return self.func(
                *[_keep_coeff(ncon, ni) for ni in n]), _keep_coeff(dcon, d)

        # sum up the terms having a common denominator
        nd2 = {d: self.func(*n) if len(n) > 1 else n[0] for d, n in nd.items()}

        # assemble single numerator and denominator
        denoms, numers = [list(i) for i in zip(*iter(nd2.items()))]
        n, d = self.func(*[Mul(*(denoms[:i] + [numers[i]] + denoms[i + 1:]))
                   for i in range(len(numers))]), Mul(*denoms)

        return _keep_coeff(ncon, n), _keep_coeff(dcon, d)

    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    # assumption methods
    _eval_is_real = lambda self: _fuzzy_group(
        (a.is_real for a in self.args), quick_exit=True)
    _eval_is_extended_real = lambda self: _fuzzy_group(
        (a.is_extended_real for a in self.args), quick_exit=True)
    _eval_is_complex = lambda self: _fuzzy_group(
        (a.is_complex for a in self.args), quick_exit=True)
    _eval_is_antihermitian = lambda self: _fuzzy_group(
        (a.is_antihermitian for a in self.args), quick_exit=True)
    _eval_is_finite = lambda self: _fuzzy_group(
        (a.is_finite for a in self.args), quick_exit=True)
    _eval_is_hermitian = lambda self: _fuzzy_group(
        (a.is_hermitian for a in self.args), quick_exit=True)
    _eval_is_integer = lambda self: _fuzzy_group(
        (a.is_integer for a in self.args), quick_exit=True)
    _eval_is_rational = lambda self: _fuzzy_group(
        (a.is_rational for a in self.args), quick_exit=True)
    _eval_is_algebraic = lambda self: _fuzzy_group(
        (a.is_algebraic for a in self.args), quick_exit=True)
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    def _eval_is_infinite(self):
        sawinf = False
        for a in self.args:
            ainf = a.is_infinite
            if ainf is None:
                return None
            elif ainf is True:
                # infinite+infinite might not be infinite
                if sawinf is True:
                    return None
                sawinf = True
        return sawinf

    def _eval_is_imaginary(self):
        nz = []
        im_I = []
        for a in self.args:
            if a.is_extended_real:
                if a.is_zero:
                    pass
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im_I.append(a*S.ImaginaryUnit)
            elif a.is_Mul and S.ImaginaryUnit in a.args:
                coeff, ai = a.as_coeff_mul(S.ImaginaryUnit)
                if ai == (S.ImaginaryUnit,) and coeff.is_extended_real:
                    im_I.append(-coeff)
                else:
                    return
            else:
                return
        b = self.func(*nz)
        if b != self:
            if b.is_zero:
                return fuzzy_not(self.func(*im_I).is_zero)
            elif b.is_zero is False:
                return False

    def _eval_is_zero(self):
        if self.is_commutative is False:
            # issue 10528: there is no way to know if a nc symbol
            # is zero or not
            return
        nz = []
        z = 0
        im_or_z = False
        im = 0
        for a in self.args:
            if a.is_extended_real:
                if a.is_zero:
                    z += 1
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im += 1
            elif a.is_Mul and S.ImaginaryUnit in a.args:
                coeff, ai = a.as_coeff_mul(S.ImaginaryUnit)
                if ai == (S.ImaginaryUnit,) and coeff.is_extended_real:
                    im_or_z = True
                else:
                    return
            else:
                return
        if z == len(self.args):
            return True
        if len(nz) in [0, len(self.args)]:
            return None
        b = self.func(*nz)
        if b.is_zero:
            if not im_or_z:
                if im == 0:
                    return True
                elif im == 1:
                    return False
        if b.is_zero is False:
            return False

    def _eval_is_odd(self):
        l = [f for f in self.args if not (f.is_even is True)]
        if not l:
            return False
        if l[0].is_odd:
            return self._new_rawargs(*l[1:]).is_even

    def _eval_is_irrational(self):
        for t in self.args:
            a = t.is_irrational
            if a:
                others = list(self.args)
                others.remove(t)
                if all(x.is_rational is True for x in others):
                    return True
                return None
            if a is None:
                return
        return False

    def _all_nonneg_or_nonppos(self):
        nn = np = 0
        for a in self.args:
            if a.is_nonnegative:
                if np:
                    return False
                nn = 1
            elif a.is_nonpositive:
                if nn:
                    return False
                np = 1
            else:
                break
        else:
            return True

    def _eval_is_extended_positive(self):
        if self.is_number:
            return super()._eval_is_extended_positive()
        c, a = self.as_coeff_Add()
        if not c.is_zero:
            from .exprtools import _monotonic_sign
            v = _monotonic_sign(a)
            if v is not None:
                s = v + c
                if s != self and s.is_extended_positive and a.is_extended_nonnegative:
                    return True
                if len(self.free_symbols) == 1:
                    v = _monotonic_sign(self)
                    if v is not None and v != self and v.is_extended_positive:
                        return True
        pos = nonneg = nonpos = unknown_sign = False
        saw_INF = set()
        args = [a for a in self.args if not a.is_zero]
        if not args:
            return False
        for a in args:
            ispos = a.is_extended_positive
            infinite = a.is_infinite
            if infinite:
                saw_INF.add(fuzzy_or((ispos, a.is_extended_nonnegative)))
                if True in saw_INF and False in saw_INF:
                    return
            if ispos:
                pos = True
                continue
            elif a.is_extended_nonnegative:
                nonneg = True
                continue
            elif a.is_extended_nonpositive:
                nonpos = True
                continue

            if infinite is None:
                return
            unknown_sign = True

        if saw_INF:
            if len(saw_INF) > 1:
                return
            return saw_INF.pop()
        elif unknown_sign:
            return
        elif not nonpos and not nonneg and pos:
            return True
        elif not nonpos and pos:
            return True
        elif not pos and not nonneg:
            return False

    def _eval_is_extended_nonnegative(self):
        if not self.is_number:
            c, a = self.as_coeff_Add()
            if not c.is_zero and a.is_extended_nonnegative:
                from .exprtools import _monotonic_sign
                v = _monotonic_sign(a)
                if v is not None:
                    s = v + c
                    if s != self and s.is_extended_nonnegative:
                        return True
                    if len(self.free_symbols) == 1:
                        v = _monotonic_sign(self)
                        if v is not None and v != self and v.is_extended_nonnegative:
                            return True

    def _eval_is_extended_nonpositive(self):
        if not self.is_number:
            c, a = self.as_coeff_Add()
            if not c.is_zero and a.is_extended_nonpositive:
                from .exprtools import _monotonic_sign
                v = _monotonic_sign(a)
                if v is not None:
                    s = v + c
                    if s != self and s.is_extended_nonpositive:
                        return True
                    if len(self.free_symbols) == 1:
                        v = _monotonic_sign(self)
                        if v is not None and v != self and v.is_extended_nonpositive:
                            return True

    def _eval_is_extended_negative(self):
        if self.is_number:
            return super()._eval_is_extended_negative()
        c, a = self.as_coeff_Add()
        if not c.is_zero:
            from .exprtools import _monotonic_sign
            v = _monotonic_sign(a)
            if v is not None:
                s = v + c
                if s != self and s.is_extended_negative and a.is_extended_nonpositive:
                    return True
                if len(self.free_symbols) == 1:
                    v = _monotonic_sign(self)
                    if v is not None and v != self and v.is_extended_negative:
                        return True
        neg = nonpos = nonneg = unknown_sign = False
        saw_INF = set()
        args = [a for a in self.args if not a.is_zero]
        if not args:
            return False
        for a in args:
            isneg = a.is_extended_negative
            infinite = a.is_infinite
            if infinite:
                saw_INF.add(fuzzy_or((isneg, a.is_extended_nonpositive)))
                if True in saw_INF and False in saw_INF:
                    return
            if isneg:
                neg = True
                continue
            elif a.is_extended_nonpositive:
                nonpos = True
                continue
            elif a.is_extended_nonnegative:
                nonneg = True
                continue

            if infinite is None:
                return
            unknown_sign = True

        if saw_INF:
            if len(saw_INF) > 1:
                return
            return saw_INF.pop()
        elif unknown_sign:
            return
        elif not nonneg and not nonpos and neg:
            return True
        elif not nonneg and neg:
            return True
        elif not neg and not nonpos:
            return False

    def _eval_subs(self, old, new):
        if not old.is_Add:
            if old is S.Infinity and -old in self.args:
                # foo - oo is foo + (-oo) internally
                return self.xreplace({-old: -new})
            return None

        coeff_self, terms_self = self.as_coeff_Add()
        coeff_old, terms_old = old.as_coeff_Add()

        if coeff_self.is_Rational and coeff_old.is_Rational:
            if terms_self == terms_old:   # (2 + a).subs( 3 + a, y) -> -1 + y
                return self.func(new, coeff_self, -coeff_old)
            if terms_self == -terms_old:  # (2 + a).subs(-3 - a, y) -> -1 - y
                return self.func(-new, coeff_self, coeff_old)

        if coeff_self.is_Rational and coeff_old.is_Rational \
                or coeff_self == coeff_old:
            args_old, args_self = self.func.make_args(
                terms_old), self.func.make_args(terms_self)
            if len(args_old) < len(args_self):  # (a+b+c).subs(b+c,x) -> a+x
                self_set = set(args_self)
                old_set = set(args_old)

                if old_set < self_set:
                    ret_set = self_set - old_set
                    return self.func(new, coeff_self, -coeff_old,
                               *[s._subs(old, new) for s in ret_set])

                args_old = self.func.make_args(
                    -terms_old)     # (a+b+c+d).subs(-b-c,x) -> a-x+d
                old_set = set(args_old)
                if old_set < self_set:
                    ret_set = self_set - old_set
                    return self.func(-new, coeff_self, coeff_old,
                               *[s._subs(old, new) for s in ret_set])

    def removeO(self):
        args = [a for a in self.args if not a.is_Order]
        return self._new_rawargs(*args)

    def getO(self):
        args = [a for a in self.args if a.is_Order]
        if args:
            return self._new_rawargs(*args)

    @cacheit
    def extract_leading_order(self, symbols, point=None):
        """
        Returns the leading term and its order.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (x + 1 + 1/x**5).extract_leading_order(x)
        ((x**(-5), O(x**(-5))),)
        >>> (1 + x).extract_leading_order(x)
        ((1, O(1)),)
        >>> (x + x**2).extract_leading_order(x)
        ((x, O(x)),)

        """
        from sympy.series.order import Order
        lst = []
        symbols = list(symbols if is_sequence(symbols) else [symbols])
        if not point:
            point = [0]*len(symbols)
        seq = [(f, Order(f, *zip(symbols, point))) for f in self.args]
        for ef, of in seq:
            for e, o in lst:
                if o.contains(of) and o != of:
                    of = None
                    break
            if of is None:
                continue
            new_lst = [(ef, of)]
            for e, o in lst:
                if of.contains(o) and o != of:
                    continue
                new_lst.append((e, o))
            lst = new_lst
        return tuple(lst)

    def as_real_imag(self, deep=True, **hints):
        """
        Return a tuple representing a complex number.

        Examples
        ========

        >>> from sympy import I
        >>> (7 + 9*I).as_real_imag()
        (7, 9)
        >>> ((1 + I)/(1 - I)).as_real_imag()
        (0, 1)
        >>> ((1 + 2*I)*(1 + 3*I)).as_real_imag()
        (-5, 5)
        """
        sargs = self.args
        re_part, im_part = [], []
        for term in sargs:
            re, im = term.as_real_imag(deep=deep)
            re_part.append(re)
            im_part.append(im)
        return (self.func(*re_part), self.func(*im_part))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.core.symbol import Dummy, Symbol
        from sympy.series.order import Order
        from sympy.functions.elementary.exponential import log
        from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
        from .function import expand_mul

        o = self.getO()
        if o is None:
            o = Order(0)
        old = self.removeO()

        if old.has(Piecewise):
            old = piecewise_fold(old)

        # This expansion is the last part of expand_log. expand_log also calls
        # expand_mul with factor=True, which would be more expensive
        if any(isinstance(a, log) for a in self.args):
            logflags = {"deep": True, "log": True, "mul": False, "power_exp": False,
                "power_base": False, "multinomial": False, "basic": False, "force": False,
                "factor": False}
            old = old.expand(**logflags)
        expr = expand_mul(old)

        if not expr.is_Add:
            return expr.as_leading_term(x, logx=logx, cdir=cdir)

        infinite = [t for t in expr.args if t.is_infinite]

        _logx = Dummy('logx') if logx is None else logx
        leading_terms = [t.as_leading_term(x, logx=_logx, cdir=cdir) for t in expr.args]

        min, new_expr = Order(0), S.Zero

        try:
            for term in leading_terms:
                order = Order(term, x)
                if not min or order not in min:
                    min = order
                    new_expr = term
                elif min in order:
                    new_expr += term

        except TypeError:
            return expr

        if logx is None:
            new_expr = new_expr.subs(_logx, log(x))

        is_zero = new_expr.is_zero
        if is_zero is None:
            new_expr = new_expr.trigsimp().cancel()
            is_zero = new_expr.is_zero
        if is_zero is True:
            # simple leading term analysis gave us cancelled terms but we have to send
            # back a term, so compute the leading term (via series)
            try:
                n0 = min.getn()
            except NotImplementedError:
                n0 = S.One
            if n0.has(Symbol):
                n0 = S.One
            res = Order(1)
            incr = S.One
            while res.is_Order:
                res = old._eval_nseries(x, n=n0+incr, logx=logx, cdir=cdir).cancel().powsimp().trigsimp()
                incr *= 2
            return res.as_leading_term(x, logx=logx, cdir=cdir)

        elif new_expr is S.NaN:
            return old.func._from_args(infinite) + o

        else:
            return new_expr

    def _eval_adjoint(self):
        return self.func(*[t.adjoint() for t in self.args])

    def _eval_conjugate(self):
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        return self.func(*[t.transpose() for t in self.args])

    def primitive(self):
        """
        Return ``(R, self/R)`` where ``R``` is the Rational GCD of ``self```.

        ``R`` is collected only from the leading coefficient of each term.

        Examples
        ========

        >>> from sympy.abc import x, y

        >>> (2*x + 4*y).primitive()
        (2, x + 2*y)

        >>> (2*x/3 + 4*y/9).primitive()
        (2/9, 3*x + 2*y)

        >>> (2*x/3 + 4.2*y).primitive()
        (1/3, 2*x + 12.6*y)

        No subprocessing of term factors is performed:

        >>> ((2 + 2*x)*x + 2).primitive()
        (1, x*(2*x + 2) + 2)

        Recursive processing can be done with the ``as_content_primitive()``
        method:

        >>> ((2 + 2*x)*x + 2).as_content_primitive()
        (2, x*(x + 1) + 1)

        See also: primitive() function in polytools.py

        """

        terms = []
        inf = False
        for a in self.args:
            c, m = a.as_coeff_Mul()
            if not c.is_Rational:
                c = S.One
                m = a
            inf = inf or m is S.ComplexInfinity
            terms.append((c.p, c.q, m))

        if not inf:
            ngcd = reduce(igcd, [t[0] for t in terms], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms], 1)
        else:
            ngcd = reduce(igcd, [t[0] for t in terms if t[1]], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms if t[1]], 1)

        if ngcd == dlcm == 1:
            return S.One, self
        if not inf:
            for i, (p, q, term) in enumerate(terms):
                terms[i] = _keep_coeff(Rational((p//ngcd)*(dlcm//q)), term)
        else:
            for i, (p, q, term) in enumerate(terms):
                if q:
                    terms[i] = _keep_coeff(Rational((p//ngcd)*(dlcm//q)), term)
                else:
                    terms[i] = _keep_coeff(Rational(p, q), term)

        # we don't need a complete re-flattening since no new terms will join
        # so we just use the same sort as is used in Add.flatten. When the
        # coefficient changes, the ordering of terms may change, e.g.
        #     (3*x, 6*y) -> (2*y, x)
        #
        # We do need to make sure that term[0] stays in position 0, however.
        #
        if terms[0].is_Number or terms[0] is S.ComplexInfinity:
            c = terms.pop(0)
        else:
            c = None
        _addsort(terms)
        if c:
            terms.insert(0, c)
        return Rational(ngcd, dlcm), self._new_rawargs(*terms)

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self. If radical is True (default is False) then
        common radicals will be removed and included as a factor of the
        primitive expression.

        Examples
        ========

        >>> from sympy import sqrt
        >>> (3 + 3*sqrt(2)).as_content_primitive()
        (3, 1 + sqrt(2))

        Radical content can also be factored out of the primitive:

        >>> (2*sqrt(2) + 4*sqrt(10)).as_content_primitive(radical=True)
        (2, sqrt(2)*(1 + 2*sqrt(5)))

        See docstring of Expr.as_content_primitive for more examples.
        """
        con, prim = self.func(*[_keep_coeff(*a.as_content_primitive(
            radical=radical, clear=clear)) for a in self.args]).primitive()
        if not clear and not con.is_Integer and prim.is_Add:
            con, d = con.as_numer_denom()
            _p = prim/d
            if any(a.as_coeff_Mul()[0].is_Integer for a in _p.args):
                prim = _p
            else:
                con /= d
        if radical and prim.is_Add:
            # look for common radicals that can be removed
            args = prim.args
            rads = []
            common_q = None
            for m in args:
                term_rads = defaultdict(list)
                for ai in Mul.make_args(m):
                    if ai.is_Pow:
                        b, e = ai.as_base_exp()
                        if e.is_Rational and b.is_Integer:
                            term_rads[e.q].append(abs(int(b))**e.p)
                if not term_rads:
                    break
                if common_q is None:
                    common_q = set(term_rads.keys())
                else:
                    common_q = common_q & set(term_rads.keys())
                    if not common_q:
                        break
                rads.append(term_rads)
            else:
                # process rads
                # keep only those in common_q
                for r in rads:
                    for q in list(r.keys()):
                        if q not in common_q:
                            r.pop(q)
                    for q in r:
                        r[q] = Mul(*r[q])
                # find the gcd of bases for each q
                G = []
                for q in common_q:
                    g = reduce(igcd, [r[q] for r in rads], 0)
                    if g != 1:
                        G.append(g**Rational(1, q))
                if G:
                    G = Mul(*G)
                    args = [ai/G for ai in args]
                    prim = G*prim.func(*args)

        return con, prim

    @property
    def _sorted_args(self):
        from .sorting import default_sort_key
        return tuple(sorted(self.args, key=default_sort_key))

    def _eval_difference_delta(self, n, step):
        from sympy.series.limitseq import difference_delta as dd
        return self.func(*[dd(a, n, step) for a in self.args])

    @property
    def _mpc_(self):
        """
        Convert self to an mpmath mpc if possible
        """
        from .numbers import Float
        re_part, rest = self.as_coeff_Add()
        im_part, imag_unit = rest.as_coeff_Mul()
        if not imag_unit == S.ImaginaryUnit:
            # ValueError may seem more reasonable but since it's a @property,
            # we need to use AttributeError to keep from confusing things like
            # hasattr.
            raise AttributeError("Cannot convert Add to mpc. Must be of the form Number + Number*I")

        return (Float(re_part)._mpf_, Float(im_part)._mpf_)

    def __neg__(self):
        if not global_parameters.distribute:
            return super().__neg__()
        return Mul(S.NegativeOne, self)

add = AssocOpDispatcher('add')

from .mul import Mul, _keep_coeff, _unevaluated_Mul
from .numbers import Rational

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\modes\vi.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Michael Graz. <mgraz@plan10.com>
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import os

import pyreadline3.lineeditor.lineobj as lineobj
from pyreadline3.logger import log

from . import basemode


class ViMode(basemode.BaseMode):
    mode = "vi"

    def __init__(self, rlobj):
        super().__init__(rlobj)

        self.__vi_insert_mode = None

    def __repr__(self):
        return "<ViMode>"

    def process_keyevent(self, keyinfo):
        def nop(e):
            pass

        keytuple = keyinfo.tuple()

        # Process exit keys. Only exit on empty line
        if keytuple in self.exit_dispatch:
            if lineobj.EndOfLine(self.l_buffer) == 0:
                raise EOFError

        dispatch_func = self.key_dispatch.get(keytuple, self.vi_key)
        log("readline from keyboard:%s->%s" % (keytuple, dispatch_func))
        r = None
        if dispatch_func:
            r = dispatch_func(keyinfo)
            self.l_buffer.push_undo()

        self.previous_func = dispatch_func
        if r:
            self._update_line()
            return True
        return False

    # Methods below here are bindable emacs functions

    def init_editing_mode(self, e):  # (M-C-j)
        """Initialize vi editingmode"""
        self.show_all_if_ambiguous = "on"
        self.key_dispatch = {}
        self.__vi_insert_mode = None
        self._vi_command = None
        self._vi_command_edit = None
        self._vi_key_find_char = None
        self._vi_key_find_direction = True
        self._vi_yank_buffer = None
        self._vi_multiplier1 = ""
        self._vi_multiplier2 = ""
        self._vi_undo_stack = []
        self._vi_undo_cursor = -1
        self._vi_current = None
        self._vi_search_text = ""
        self._vi_search_position = 0
        self.vi_save_line()
        self.vi_set_insert_mode(True)
        # make ' ' to ~ self insert
        for c in range(ord(" "), 127):
            self._bind_key("%s" % chr(c), self.vi_key)
        self._bind_key("BackSpace", self.vi_backspace)
        self._bind_key("Escape", self.vi_escape)
        self._bind_key("Return", self.vi_accept_line)

        self._bind_key("Left", self.backward_char)
        self._bind_key("Right", self.forward_char)
        self._bind_key("Home", self.beginning_of_line)
        self._bind_key("End", self.end_of_line)
        self._bind_key("Delete", self.delete_char)

        self._bind_key("Control-d", self.vi_eof)
        self._bind_key("Control-z", self.vi_eof)
        self._bind_key("Control-r", self.vi_redo)
        self._bind_key("Up", self.vi_arrow_up)
        self._bind_key("Control-p", self.vi_up)
        self._bind_key("Down", self.vi_arrow_down)
        self._bind_key("Control-n", self.vi_down)
        self._bind_key("Tab", self.vi_complete)

    #        self._bind_key('Control-e', self.emacs)

    def vi_key(self, e):
        if not self._vi_command:
            self._vi_command = ViCommand(self)
        elif self._vi_command.is_end:
            if self._vi_command.is_edit:
                self._vi_command_edit = self._vi_command
            self._vi_command = ViCommand(self)
        self._vi_command.add_char(e.char)

    def vi_error(self):
        self._bell()

    def vi_get_is_insert_mode(self):
        return self.__vi_insert_mode

    vi_is_insert_mode = property(vi_get_is_insert_mode)

    def vi_escape(self, e):
        if self.vi_is_insert_mode:
            if self._vi_command:
                self._vi_command.add_char(e.char)
            else:
                self._vi_command = ViCommand(self)
            self.vi_set_insert_mode(False)
            self.l_buffer.point = lineobj.PrevChar
        elif self._vi_command and self._vi_command.is_replace_one:
            self._vi_command.add_char(e.char)
        else:
            self.vi_error()

    def vi_backspace(self, e):
        if self._vi_command:
            self._vi_command.add_char(e.char)
        else:
            self._vi_do_backspace(self._vi_command)

    def _vi_do_backspace(self, vi_cmd):
        if self.vi_is_insert_mode or (self._vi_command and self._vi_command.is_search):
            if self.l_buffer.point > 0:
                self.l_buffer.point -= 1
                if self.l_buffer.overwrite:
                    try:
                        prev = self._vi_undo_stack[self._vi_undo_cursor][1][
                            self.l_buffer.point
                        ]
                        self.l_buffer.line_buffer[self.l_buffer.point] = prev
                    except IndexError:
                        del self.l_buffer.line_buffer[self.l_buffer.point]
                else:
                    self.vi_save_line()
                    del self.l_buffer.line_buffer[self.l_buffer.point]

    def vi_accept_line(self, e):
        if self._vi_command and self._vi_command.is_search:
            self._vi_command.do_search()
            return False
        self._vi_command = None
        self.vi_set_insert_mode(True)
        self._vi_undo_stack = []
        self._vi_undo_cursor = -1
        self._vi_current = None
        if self.l_buffer.line_buffer:
            self.add_history(self.l_buffer.copy())
        return self.accept_line(e)

    def vi_eof(self, e):
        raise EOFError

    def vi_set_insert_mode(self, value):
        if self.__vi_insert_mode == value:
            return
        self.__vi_insert_mode = value
        if value:
            self.vi_save_line()
            self.cursor_size = 25
        else:
            self.cursor_size = 100

    def vi_undo_restart(self):
        tpl_undo = (
            self.l_buffer.point,
            self.l_buffer.line_buffer[:],
        )
        self._vi_undo_stack = [tpl_undo]
        self._vi_undo_cursor = 0

    def vi_save_line(self):
        if self._vi_undo_stack and self._vi_undo_cursor >= 0:
            del self._vi_undo_stack[self._vi_undo_cursor + 1 :]
        # tpl_undo = (self.l_buffer.point, self.l_buffer[:], )
        tpl_undo = (
            self.l_buffer.point,
            self.l_buffer.line_buffer[:],
        )
        if (
            not self._vi_undo_stack
            or self._vi_undo_stack[self._vi_undo_cursor][1] != tpl_undo[1]
        ):
            self._vi_undo_stack.append(tpl_undo)
            self._vi_undo_cursor += 1

    def vi_undo_prepare(self):
        if self._vi_undo_cursor == len(self._vi_undo_stack) - 1:
            self.vi_save_line()

    def vi_undo(self, do_pop=True):
        self.vi_undo_prepare()
        if not self._vi_undo_stack or self._vi_undo_cursor <= 0:
            self.vi_error()
            return
        self._vi_undo_cursor -= 1
        self.vi_undo_assign()

    def vi_undo_all(self):
        self.vi_undo_prepare()
        if self._vi_undo_cursor > 0:
            self._vi_undo_cursor = 0
            self.vi_undo_assign()
        else:
            self.vi_error()

    def vi_undo_assign(self):
        tpl_undo = self._vi_undo_stack[self._vi_undo_cursor]
        self.l_buffer.line_buffer = tpl_undo[1][:]
        self.l_buffer.point = tpl_undo[0]

    def vi_redo(self, e):
        if self._vi_undo_cursor >= len(self._vi_undo_stack) - 1:
            self.vi_error()
            return
        self._vi_undo_cursor += 1
        self.vi_undo_assign()

    def vi_search(self, rng):
        for i in rng:
            line_history = self._history.history[i]
            pos = line_history.get_line_text().find(self._vi_search_text)
            if pos >= 0:
                self._vi_search_position = i
                self._history.history_cursor = i
                self.l_buffer.line_buffer = list(line_history.line_buffer)
                self.l_buffer.point = pos
                self.vi_undo_restart()
                return True
        self._bell()
        return False

    def vi_search_first(self):
        text = "".join(self.l_buffer.line_buffer[1:])
        if text:
            self._vi_search_text = text
            self._vi_search_position = len(self._history.history) - 1
        elif self._vi_search_text:
            self._vi_search_position -= 1
        else:
            self.vi_error()
            self.vi_undo()
            return
        if not self.vi_search(list(range(self._vi_search_position, -1, -1))):
            # Here: search text not found
            self.vi_undo()

    def vi_search_again_backward(self):
        self.vi_search(list(range(self._vi_search_position - 1, -1, -1)))

    def vi_search_again_forward(self):
        self.vi_search(
            list(range(self._vi_search_position + 1, len(self._history.history)))
        )

    def vi_up(self, e):
        if self._history.history_cursor == len(self._history.history):
            self._vi_current = self.l_buffer.line_buffer[:]
        # self._history.previous_history (e)
        self._history.previous_history(self.l_buffer)
        if self.vi_is_insert_mode:
            self.end_of_line(e)
        else:
            self.beginning_of_line(e)
        self.vi_undo_restart()

    def vi_down(self, e):
        if self._history.history_cursor >= len(self._history.history):
            self.vi_error()
            return
        if self._history.history_cursor < len(self._history.history) - 1:
            # self._history.next_history (e)
            self._history.next_history(self.l_buffer)
            if self.vi_is_insert_mode:
                self.end_of_line(e)
            else:
                self.beginning_of_line(e)
            self.vi_undo_restart()
        elif self._vi_current is not None:
            self._history.history_cursor = len(self._history.history)
            self.l_buffer.line_buffer = self._vi_current
            self.end_of_line(e)
            if not self.vi_is_insert_mode and self.l_buffer.point > 0:
                self.l_buffer.point -= 1
            self._vi_current = None
        else:
            self.vi_error()
            return

    def vi_arrow_up(self, e):
        self.vi_set_insert_mode(True)
        self.vi_up(e)
        self.vi_save_line()

    def vi_arrow_down(self, e):
        self.vi_set_insert_mode(True)
        self.vi_down(e)
        self.vi_save_line()

    def vi_complete(self, e):
        text = self.l_buffer.get_line_text()
        if text and not text.isspace():
            return self.complete(e)
        else:
            return self.vi_key(e)


# vi input states
# sequence of possible states are in the order below
_VI_BEGIN = "vi_begin"
_VI_MULTI1 = "vi_multi1"
_VI_ACTION = "vi_action"
_VI_MULTI2 = "vi_multi2"
_VI_MOTION = "vi_motion"
_VI_MOTION_ARGUMENT = "vi_motion_argument"
_VI_REPLACE_ONE = "vi_replace_one"
_VI_TEXT = "vi_text"
_VI_SEARCH = "vi_search"
_VI_END = "vi_end"

# vi helper class


class ViCommand:
    def __init__(self, readline):
        self.readline = readline
        self.lst_char = []
        self.state = _VI_BEGIN
        self.action = self.movement
        self.motion = None
        self.motion_argument = None
        self.text = None
        self.pos_motion = None
        self.is_edit = False
        self.is_overwrite = False
        self.is_error = False
        self.is_star = False
        self.delete_left = 0
        self.delete_right = 0
        self.readline._vi_multiplier1 = ""
        self.readline._vi_multiplier2 = ""
        self.set_override_multiplier(0)
        self.skip_multipler = False
        self.tabstop = 4
        self.dct_fcn = {
            ord("$"): self.key_dollar,
            ord("^"): self.key_hat,
            ord(";"): self.key_semicolon,
            ord(","): self.key_comma,
            ord("%"): self.key_percent,
            ord("."): self.key_dot,
            ord("/"): self.key_slash,
            ord("*"): self.key_star,
            ord("|"): self.key_bar,
            ord("~"): self.key_tilde,
            8: self.key_backspace,
        }

    def add_char(self, char):
        self.lst_char.append(char)
        if self.state == _VI_BEGIN and self.readline.vi_is_insert_mode:
            self.readline.vi_save_line()
            self.state = _VI_TEXT
        if self.state == _VI_SEARCH:
            if char == "\x08":  # backspace
                self.key_backspace(char)
            else:
                self.set_text(char)
            return
        if self.state == _VI_TEXT:
            if char == "\x1b":  # escape
                self.escape(char)
            elif char == "\x09":  # tab
                ts = self.tabstop
                ws = " " * (ts - (self.readline.l_buffer.point % ts))
                self.set_text(ws)
            elif char == "\x08":  # backspace
                self.key_backspace(char)
            else:
                self.set_text(char)
            return
        if self.state == _VI_MOTION_ARGUMENT:
            self.set_motion_argument(char)
            return
        if self.state == _VI_REPLACE_ONE:
            self.replace_one(char)
            return
        try:
            fcn_instance = self.dct_fcn[ord(char)]
        except BaseException:
            fcn_instance = getattr(self, "key_%s" % char, None)
        if fcn_instance:
            fcn_instance(char)
            return
        if char.isdigit():
            self.key_digit(char)
            return
        # Here: could not process key
        self.error()

    def set_text(self, text):
        if self.text is None:
            self.text = text
        else:
            self.text += text
        self.set_buffer(text)

    def set_buffer(self, text):
        for char in text:
            if not self.char_isprint(char):
                continue
            #             self.readline.l_buffer.insert_text(char)
            #             continue
            #             #overwrite in l_buffer obj
            if self.is_overwrite:
                if self.readline.l_buffer.point < len(
                    self.readline.l_buffer.line_buffer
                ):
                    # self.readline.l_buffer[self.l_buffer.point]=char
                    self.readline.l_buffer.line_buffer[self.readline.l_buffer.point] = (
                        char
                    )
                else:
                    # self.readline.l_buffer.insert_text(char)
                    self.readline.l_buffer.line_buffer.append(char)
            else:
                # self.readline.l_buffer.insert_text(char)
                self.readline.l_buffer.line_buffer.insert(
                    self.readline.l_buffer.point, char
                )
            self.readline.l_buffer.point += 1

    def replace_one(self, char):
        if char == "\x1b":  # escape
            self.end()
            return
        self.is_edit = True
        self.readline.vi_save_line()
        times = self.get_multiplier()
        cursor = self.readline.l_buffer.point
        self.readline.l_buffer.line_buffer[cursor : cursor + times] = char * times
        if times > 1:
            self.readline.l_buffer.point += times - 1
        self.end()

    def char_isprint(self, char):
        return ord(char) >= ord(" ") and ord(char) <= ord("~")

    def key_dollar(self, char):
        self.motion = self.motion_end_in_line
        self.delete_right = 1
        self.state = _VI_MOTION
        self.apply()

    def key_hat(self, char):
        self.motion = self.motion_beginning_of_line
        self.state = _VI_MOTION
        self.apply()

    def key_0(self, char):
        if self.state in [_VI_BEGIN, _VI_ACTION]:
            self.key_hat(char)
        else:
            self.key_digit(char)

    def key_digit(self, char):
        if self.state in [_VI_BEGIN, _VI_MULTI1]:
            self.readline._vi_multiplier1 += char
            self.readline._vi_multiplier2 = ""
            self.state = _VI_MULTI1
        elif self.state in [_VI_ACTION, _VI_MULTI2]:
            self.readline._vi_multiplier2 += char
            self.state = _VI_MULTI2

    def key_w(self, char):
        if self.action == self.change:
            self.key_e(char)
            return
        self.motion = self.motion_word_short
        self.state = _VI_MOTION
        self.apply()

    def key_W(self, char):
        if self.action == self.change:
            self.key_E(char)
            return
        self.motion = self.motion_word_long
        self.state = _VI_MOTION
        self.apply()

    def key_e(self, char):
        self.motion = self.motion_end_short
        self.state = _VI_MOTION
        self.delete_right = 1
        self.apply()

    def key_E(self, char):
        self.motion = self.motion_end_long
        self.state = _VI_MOTION
        self.delete_right = 1
        self.apply()

    def key_b(self, char):
        self.motion = self.motion_back_short
        self.state = _VI_MOTION
        self.apply()

    def key_B(self, char):
        self.motion = self.motion_back_long
        self.state = _VI_MOTION
        self.apply()

    def key_f(self, char):
        self.readline._vi_key_find_direction = True
        self.motion = self.motion_find_char_forward
        self.delete_right = 1
        self.state = _VI_MOTION_ARGUMENT

    def key_F(self, char):
        self.readline._vi_key_find_direction = False
        self.motion = self.motion_find_char_backward
        self.delete_left = 1
        self.state = _VI_MOTION_ARGUMENT

    def key_t(self, char):
        self.motion = self.motion_to_char_forward
        self.delete_right = 1
        self.state = _VI_MOTION_ARGUMENT

    def key_T(self, char):
        self.motion = self.motion_to_char_backward
        self.state = _VI_MOTION_ARGUMENT

    def key_j(self, char):
        self.readline.vi_down(ViEvent(char))
        self.state = _VI_END

    def key_k(self, char):
        self.readline.vi_up(ViEvent(char))
        self.state = _VI_END

    def key_semicolon(self, char):
        if self.readline._vi_key_find_char is None:
            self.error()
            return
        if self.readline._vi_key_find_direction:
            self.motion = self.motion_find_char_forward
        else:
            self.motion = self.motion_find_char_backward
        self.set_motion_argument(self.readline._vi_key_find_char)

    def key_comma(self, char):
        if self.readline._vi_key_find_char is None:
            self.error()
            return
        if self.readline._vi_key_find_direction:
            self.motion = self.motion_find_char_backward
        else:
            self.motion = self.motion_find_char_forward
        self.set_motion_argument(self.readline._vi_key_find_char)

    def key_percent(self, char):
        """find matching <([{}])>"""
        self.motion = self.motion_matching
        self.delete_right = 1
        self.state = _VI_MOTION
        self.apply()

    def key_dot(self, char):
        vi_cmd_edit = self.readline._vi_command_edit
        if not vi_cmd_edit:
            return
        if vi_cmd_edit.is_star:
            self.key_star(char)
            return
        if self.has_multiplier():
            count = self.get_multiplier()
        else:
            count = 0
        # Create the ViCommand object after getting multiplier from self
        # Side effect of the ViCommand creation is resetting of global
        # multipliers
        vi_cmd = ViCommand(self.readline)
        if count >= 1:
            vi_cmd.set_override_multiplier(count)
            vi_cmd_edit.set_override_multiplier(count)
        elif vi_cmd_edit.override_multiplier:
            vi_cmd.set_override_multiplier(vi_cmd_edit.override_multiplier)
        for char in vi_cmd_edit.lst_char:
            vi_cmd.add_char(char)
        if vi_cmd_edit.is_overwrite and self.readline.l_buffer.point > 0:
            self.readline.l_buffer.point -= 1
        self.readline.vi_set_insert_mode(False)
        self.end()

    def key_slash(self, char):
        self.readline.vi_save_line()
        self.readline.l_buffer.line_buffer = ["/"]
        self.readline.l_buffer.point = 1
        self.state = _VI_SEARCH

    def key_star(self, char):
        self.is_star = True
        self.is_edit = True
        self.readline.vi_save_line()
        completions = self.readline._get_completions()
        if completions:
            text = " ".join(completions) + " "
            self.readline.l_buffer.line_buffer[
                self.readline.begidx : self.readline.endidx + 1
            ] = list(text)
            prefix_len = self.readline.endidx - self.readline.begidx
            self.readline.l_buffer.point += len(text) - prefix_len
            self.readline.vi_set_insert_mode(True)
        else:
            self.error()
        self.state = _VI_TEXT

    def key_bar(self, char):
        self.motion = self.motion_column
        self.state = _VI_MOTION
        self.apply()

    def key_tilde(self, char):
        self.is_edit = True
        self.readline.vi_save_line()
        for i in range(self.get_multiplier()):
            try:
                c = self.readline.l_buffer.line_buffer[self.readline.l_buffer.point]
                if c.isupper():
                    self.readline.l_buffer.line_buffer[self.readline.l_buffer.point] = (
                        c.lower()
                    )
                elif c.islower():
                    self.readline.l_buffer.line_buffer[self.readline.l_buffer.point] = (
                        c.upper()
                    )
                self.readline.l_buffer.point += 1
            except IndexError:
                break
        self.end()

    def key_h(self, char):
        self.motion = self.motion_left
        self.state = _VI_MOTION
        self.apply()

    def key_backspace(self, char):
        if self.state in [_VI_TEXT, _VI_SEARCH]:
            if self.text and len(self.text):
                self.text = self.text[:-1]
                try:
                    # Remove backspaces for potential dot command
                    self.lst_char.pop()
                    self.lst_char.pop()
                except IndexError:
                    pass
        else:
            self.key_h(char)
        self.readline._vi_do_backspace(self)
        if self.state == _VI_SEARCH and not (self.readline.l_buffer.line_buffer):
            self.state = _VI_BEGIN

    def key_l(self, char):
        self.motion = self.motion_right
        self.state = _VI_MOTION
        self.apply()

    def key_i(self, char):
        self.is_edit = True
        self.state = _VI_TEXT
        self.readline.vi_set_insert_mode(True)

    def key_I(self, char):
        self.is_edit = True
        self.state = _VI_TEXT
        self.readline.vi_set_insert_mode(True)
        self.readline.l_buffer.point = 0

    def key_a(self, char):
        self.is_edit = True
        self.state = _VI_TEXT
        self.readline.vi_set_insert_mode(True)
        if len(self.readline.l_buffer.line_buffer):
            self.readline.l_buffer.point += 1

    def key_A(self, char):
        self.is_edit = True
        self.state = _VI_TEXT
        self.readline.vi_set_insert_mode(True)
        self.readline.l_buffer.point = len(self.readline.l_buffer.line_buffer)

    def key_d(self, char):
        self.is_edit = True
        self.state = _VI_ACTION
        self.action = self.delete

    def key_D(self, char):
        self.is_edit = True
        self.state = _VI_ACTION
        self.action = self.delete_end_of_line
        self.apply()

    def key_x(self, char):
        self.is_edit = True
        self.state = _VI_ACTION
        self.action = self.delete_char
        self.apply()

    def key_X(self, char):
        self.is_edit = True
        self.state = _VI_ACTION
        self.action = self.delete_prev_char
        self.apply()

    def key_s(self, char):
        self.is_edit = True
        i1 = self.readline.l_buffer.point
        i2 = self.readline.l_buffer.point + self.get_multiplier()
        self.skip_multipler = True
        self.readline.vi_set_insert_mode(True)
        del self.readline.l_buffer.line_buffer[i1:i2]
        self.state = _VI_TEXT

    def key_S(self, char):
        self.is_edit = True
        self.readline.vi_set_insert_mode(True)
        self.readline.l_buffer.line_buffer = []
        self.readline.l_buffer.point = 0
        self.state = _VI_TEXT

    def key_c(self, char):
        self.is_edit = True
        self.state = _VI_ACTION
        self.action = self.change

    def key_C(self, char):
        self.is_edit = True
        self.readline.vi_set_insert_mode(True)
        del self.readline.l_buffer.line_buffer[self.readline.l_buffer.point :]
        self.state = _VI_TEXT

    def key_r(self, char):
        self.state = _VI_REPLACE_ONE

    def key_R(self, char):
        self.is_edit = True
        self.is_overwrite = True
        self.readline.l_buffer.overwrite = True
        self.readline.vi_set_insert_mode(True)
        self.state = _VI_TEXT

    def key_y(self, char):
        self._state = _VI_ACTION
        self.action = self.yank

    def key_Y(self, char):
        self.readline._vi_yank_buffer = self.readline.l_buffer.get_line_text()
        self.end()

    def key_p(self, char):
        if not self.readline._vi_yank_buffer:
            return
        self.is_edit = True
        self.readline.vi_save_line()
        self.readline.l_buffer.point += 1
        self.readline.l_buffer.insert_text(
            self.readline._vi_yank_buffer * self.get_multiplier()
        )
        self.readline.l_buffer.point -= 1
        self.state = _VI_END

    def key_P(self, char):
        if not self.readline._vi_yank_buffer:
            return
        self.is_edit = True
        self.readline.vi_save_line()
        self.readline.l_buffer.insert_text(
            self.readline._vi_yank_buffer * self.get_multiplier()
        )
        self.readline.l_buffer.point -= 1
        self.state = _VI_END

    def key_u(self, char):
        self.readline.vi_undo()
        self.state = _VI_END

    def key_U(self, char):
        self.readline.vi_undo_all()
        self.state = _VI_END

    def key_v(self, char):
        editor = ViExternalEditor(self.readline.l_buffer.line_buffer)
        self.readline.l_buffer.line_buffer = list(editor.result)
        self.readline.l_buffer.point = 0
        self.is_edit = True
        self.state = _VI_END

    def error(self):
        self.readline._bell()
        self.is_error = True

    def state_is_end(self):
        return self.state == _VI_END

    is_end = property(state_is_end)

    def state_is_search(self):
        return self.state == _VI_SEARCH

    is_search = property(state_is_search)

    def state_is_replace_one(self):
        return self.state == _VI_REPLACE_ONE

    is_replace_one = property(state_is_replace_one)

    def do_search(self):
        self.readline.vi_search_first()
        self.state = _VI_END

    def key_n(self, char):
        self.readline.vi_search_again_backward()
        self.state = _VI_END

    def key_N(self, char):
        self.readline.vi_search_again_forward()
        self.state = _VI_END

    def motion_beginning_of_line(self, line, index=0, count=1, **kw):
        return 0

    def motion_end_in_line(self, line, index=0, count=1, **kw):
        return max(0, len(self.readline.l_buffer.line_buffer) - 1)

    def motion_word_short(self, line, index=0, count=1, **kw):
        return vi_pos_word_short(line, index, count)

    def motion_word_long(self, line, index=0, count=1, **kw):
        return vi_pos_word_long(line, index, count)

    def motion_end_short(self, line, index=0, count=1, **kw):
        return vi_pos_end_short(line, index, count)

    def motion_end_long(self, line, index=0, count=1, **kw):
        return vi_pos_end_long(line, index, count)

    def motion_back_short(self, line, index=0, count=1, **kw):
        return vi_pos_back_short(line, index, count)

    def motion_back_long(self, line, index=0, count=1, **kw):
        return vi_pos_back_long(line, index, count)

    def motion_find_char_forward(self, line, index=0, count=1, char=None):
        self.readline._vi_key_find_char = char
        return vi_pos_find_char_forward(line, char, index, count)

    def motion_find_char_backward(self, line, index=0, count=1, char=None):
        self.readline._vi_key_find_char = char
        return vi_pos_find_char_backward(line, char, index, count)

    def motion_to_char_forward(self, line, index=0, count=1, char=None):
        return vi_pos_to_char_forward(line, char, index, count)

    def motion_to_char_backward(self, line, index=0, count=1, char=None):
        return vi_pos_to_char_backward(line, char, index, count)

    def motion_left(self, line, index=0, count=1, char=None):
        return max(0, index - count)

    def motion_right(self, line, index=0, count=1, char=None):
        return min(len(line), index + count)

    def motion_matching(self, line, index=0, count=1, char=None):
        return vi_pos_matching(line, index)

    def motion_column(self, line, index=0, count=1, char=None):
        return max(0, count - 1)

    def has_multiplier(self):
        return (
            self.override_multiplier
            or self.readline._vi_multiplier1
            or self.readline._vi_multiplier2
        )

    def get_multiplier(self):
        if self.override_multiplier:
            return int(self.override_multiplier)
        if self.readline._vi_multiplier1 == "":
            m1 = 1
        else:
            m1 = int(self.readline._vi_multiplier1)
        if self.readline._vi_multiplier2 == "":
            m2 = 1
        else:
            m2 = int(self.readline._vi_multiplier2)
        return m1 * m2

    def set_override_multiplier(self, count):
        self.override_multiplier = count

    def apply(self):
        if self.motion:
            self.pos_motion = self.motion(
                self.readline.l_buffer.line_buffer,
                self.readline.l_buffer.point,
                self.get_multiplier(),
                char=self.motion_argument,
            )
            if self.pos_motion < 0:
                self.error()
                return
        self.action()
        if self.state != _VI_TEXT:
            self.end()

    def movement(self):
        if self.pos_motion <= len(self.readline.l_buffer.line_buffer):
            self.readline.l_buffer.point = self.pos_motion
        else:
            self.readline.l_buffer.point = len(self.readline.l_buffer.line_buffer) - 1

    def yank(self):
        if self.pos_motion > self.readline.l_buffer.point:
            s = self.readline.l_buffer.line_buffer[
                self.readline.l_buffer.point : self.pos_motion + self.delete_right
            ]
        else:
            index = max(0, self.pos_motion - self.delete_left)
            s = self.readline.l_buffer.line_buffer[
                index : self.readline.l_buffer.point + self.delete_right
            ]
        self.readline._vi_yank_buffer = s

    def delete(self):
        self.readline.vi_save_line()
        self.yank()
        #         point=lineobj.Point(self.readline.l_buffer)
        #         pm=self.pos_motion
        #         del self.readline.l_buffer[point:pm]
        #         return
        if self.pos_motion > self.readline.l_buffer.point:
            del self.readline.l_buffer.line_buffer[
                self.readline.l_buffer.point : self.pos_motion + self.delete_right
            ]
            if self.readline.l_buffer.point > len(self.readline.l_buffer.line_buffer):
                self.readline.l_buffer.point = len(self.readline.l_buffer.line_buffer)
        else:
            index = max(0, self.pos_motion - self.delete_left)
            del self.readline.l_buffer.line_buffer[
                index : self.readline.l_buffer.point + self.delete_right
            ]
            self.readline.l_buffer.point = index

    def delete_end_of_line(self):
        self.readline.vi_save_line()
        # del self.readline.l_buffer [self.readline.l_buffer.point : ]
        line_text = self.readline.l_buffer.get_line_text()
        line_text = line_text[: self.readline.l_buffer.point]
        self.readline.l_buffer.set_line(line_text)
        if self.readline.l_buffer.point > 0:
            self.readline.l_buffer.point -= 1

    def delete_char(self):
        #         point=lineobj.Point(self.readline.l_buffer)
        #         del self.readline.l_buffer[point:point+self.get_multiplier ()]
        #         return
        self.pos_motion = self.readline.l_buffer.point + self.get_multiplier()
        self.delete()
        end = max(0, len(self.readline.l_buffer) - 1)
        if self.readline.l_buffer.point > end:
            self.readline.l_buffer.point = end

    def delete_prev_char(self):
        self.pos_motion = self.readline.l_buffer.point - self.get_multiplier()
        self.delete()

    def change(self):
        self.readline.vi_set_insert_mode(True)
        self.delete()
        self.skip_multipler = True
        self.state = _VI_TEXT

    def escape(self, char):
        if self.state == _VI_TEXT:
            if not self.skip_multipler:
                times = self.get_multiplier()
                if times > 1 and self.text:
                    extra = self.text * (times - 1)
                    self.set_buffer(extra)
        self.state = _VI_END

    def set_motion_argument(self, char):
        self.motion_argument = char
        self.apply()

    def end(self):
        self.state = _VI_END
        if self.readline.l_buffer.point >= len(self.readline.l_buffer.line_buffer):
            self.readline.l_buffer.point = max(
                0, len(self.readline.l_buffer.line_buffer) - 1
            )


class ViExternalEditor:
    def __init__(self, line):
        if isinstance(line, type([])):
            line = "".join(line)
        file_tmp = self.get_tempfile()
        fp_tmp = self.file_open(file_tmp, "w")
        fp_tmp.write(line)
        fp_tmp.close()
        self.run_editor(file_tmp)
        fp_tmp = self.file_open(file_tmp, "r")
        self.result = fp_tmp.read()
        fp_tmp.close()
        self.file_remove(file_tmp)

    def get_tempfile(self):
        import tempfile

        return tempfile.mktemp(prefix="readline-", suffix=".py")

    def file_open(self, filename, mode):
        return open(filename, mode)

    def file_remove(self, filename):
        os.remove(filename)

    def get_editor(self):
        try:
            return os.environ["EDITOR"]
        except KeyError:
            return "notepad"  # ouch

    def run_editor(self, filename):
        cmd = "%s %s" % (
            self.get_editor(),
            filename,
        )
        self.run_command(cmd)

    def run_command(self, command):
        os.system(command)


class ViEvent:
    def __init__(self, char):
        self.char = char


# vi standalone functions


def vi_is_word(char):
    log(
        "xx vi_is_word: type(%s), %s"
        % (
            type(char),
            char,
        )
    )
    return char.isalpha() or char.isdigit() or char == "_"


def vi_is_space(char):
    return char.isspace()


def vi_is_word_or_space(char):
    return vi_is_word(char) or vi_is_space(char)


def vi_pos_word_short(line, index=0, count=1):
    try:
        for i in range(count):
            in_word = vi_is_word(line[index])
            if not in_word:
                while not vi_is_word(line[index]):
                    index += 1
            else:
                while vi_is_word(line[index]):
                    index += 1
            while vi_is_space(line[index]):
                index += 1
        return index
    except IndexError:
        return len(line)


def vi_pos_word_long(line, index=0, count=1):
    try:
        for i in range(count):
            in_space = vi_is_space(line[index])
            if not in_space:
                while not vi_is_space(line[index]):
                    index += 1
            while vi_is_space(line[index]):
                index += 1
        return index
    except IndexError:
        return len(line)


def vi_pos_end_short(line, index=0, count=1):
    try:
        for i in range(count):
            index += 1
            while vi_is_space(line[index]):
                index += 1
            in_word = vi_is_word(line[index])
            if not in_word:
                while not vi_is_word_or_space(line[index]):
                    index += 1
            else:
                while vi_is_word(line[index]):
                    index += 1
        return index - 1
    except IndexError:
        return max(0, len(line) - 1)


def vi_pos_end_long(line, index=0, count=1):
    try:
        for i in range(count):
            index += 1
            while vi_is_space(line[index]):
                index += 1
            while not vi_is_space(line[index]):
                index += 1
        return index - 1
    except IndexError:
        return max(0, len(line) - 1)


class vi_list(list):
    """This is a list that cannot have a negative index"""

    def __getitem__(self, key):
        try:
            if int(key) < 0:
                raise IndexError
        except ValueError:
            pass
        return list.__getitem__(self, key)


def vi_pos_back_short(line, index=0, count=1):
    line = vi_list(line)
    try:
        for i in range(count):
            index -= 1
            while vi_is_space(line[index]):
                index -= 1
            in_word = vi_is_word(line[index])
            if in_word:
                while vi_is_word(line[index]):
                    index -= 1
            else:
                while not vi_is_word_or_space(line[index]):
                    index -= 1
        return index + 1
    except IndexError:
        return 0


def vi_pos_back_long(line, index=0, count=1):
    line = vi_list(line)
    try:
        for i in range(count):
            index -= 1
            while vi_is_space(line[index]):
                index -= 1
            while not vi_is_space(line[index]):
                index -= 1
        return index + 1
    except IndexError:
        return 0


def vi_pos_find_char_forward(line, char, index=0, count=1):
    try:
        for i in range(count):
            index += 1
            while line[index] != char:
                index += 1
        return index
    except IndexError:
        return -1


def vi_pos_find_char_backward(line, char, index=0, count=1):
    try:
        for i in range(count):
            index -= 1
            while True:
                if index < 0:
                    return -1
                if line[index] == char:
                    break
                index -= 1
        return index
    except IndexError:
        return -1


def vi_pos_to_char_forward(line, char, index=0, count=1):
    index = vi_pos_find_char_forward(line, char, index, count)
    if index > 0:
        return index - 1
    return index


def vi_pos_to_char_backward(line, char, index=0, count=1):
    index = vi_pos_find_char_backward(line, char, index, count)
    if index >= 0:
        return index + 1
    return index


_vi_dct_matching = {
    "<": (">", +1),
    ">": ("<", -1),
    "(": (")", +1),
    ")": ("(", -1),
    "[": ("]", +1),
    "]": ("[", -1),
    "{": ("}", +1),
    "}": ("{", -1),
}


def vi_pos_matching(line, index=0):
    """find matching <([{}])>"""
    anchor = None
    target = None
    delta = 1
    count = 0
    try:
        while True:
            if anchor is None:
                # first find anchor
                try:
                    target, delta = _vi_dct_matching[line[index]]
                    anchor = line[index]
                    count = 1
                except KeyError:
                    index += 1
                    continue
            else:
                # Here the anchor has been found
                # Need to get corresponding target
                if index < 0:
                    return -1
                if line[index] == anchor:
                    count += 1
                elif line[index] == target:
                    count -= 1
                    if count == 0:
                        return index
            index += delta
    except IndexError:
        return -1

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\response.py ===
from __future__ import annotations

import collections
import io
import json as _json
import logging
import re
import socket
import sys
import typing
import warnings
import zlib
from contextlib import contextmanager
from http.client import HTTPMessage as _HttplibHTTPMessage
from http.client import HTTPResponse as _HttplibHTTPResponse
from socket import timeout as SocketTimeout

if typing.TYPE_CHECKING:
    from ._base_connection import BaseHTTPConnection

try:
    try:
        import brotlicffi as brotli  # type: ignore[import-not-found]
    except ImportError:
        import brotli  # type: ignore[import-not-found]
except ImportError:
    brotli = None

try:
    import zstandard as zstd
except (AttributeError, ImportError, ValueError):  # Defensive:
    HAS_ZSTD = False
else:
    # The package 'zstandard' added the 'eof' property starting
    # in v0.18.0 which we require to ensure a complete and
    # valid zstd stream was fed into the ZstdDecoder.
    # See: https://github.com/urllib3/urllib3/pull/2624
    _zstd_version = tuple(
        map(int, re.search(r"^([0-9]+)\.([0-9]+)", zstd.__version__).groups())  # type: ignore[union-attr]
    )
    if _zstd_version < (0, 18):  # Defensive:
        HAS_ZSTD = False
    else:
        HAS_ZSTD = True

from . import util
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from .connection import BaseSSLError, HTTPConnection, HTTPException
from .exceptions import (
    BodyNotHttplibCompatible,
    DecodeError,
    HTTPError,
    IncompleteRead,
    InvalidChunkLength,
    InvalidHeader,
    ProtocolError,
    ReadTimeoutError,
    ResponseNotChunked,
    SSLError,
)
from .util.response import is_fp_closed, is_response_to_head
from .util.retry import Retry

if typing.TYPE_CHECKING:
    from .connectionpool import HTTPConnectionPool

log = logging.getLogger(__name__)


class ContentDecoder:
    def decompress(self, data: bytes) -> bytes:
        raise NotImplementedError()

    def flush(self) -> bytes:
        raise NotImplementedError()


class DeflateDecoder(ContentDecoder):
    def __init__(self) -> None:
        self._first_try = True
        self._data = b""
        self._obj = zlib.decompressobj()

    def decompress(self, data: bytes) -> bytes:
        if not data:
            return data

        if not self._first_try:
            return self._obj.decompress(data)

        self._data += data
        try:
            decompressed = self._obj.decompress(data)
            if decompressed:
                self._first_try = False
                self._data = None  # type: ignore[assignment]
            return decompressed
        except zlib.error:
            self._first_try = False
            self._obj = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                return self.decompress(self._data)
            finally:
                self._data = None  # type: ignore[assignment]

    def flush(self) -> bytes:
        return self._obj.flush()


class GzipDecoderState:
    FIRST_MEMBER = 0
    OTHER_MEMBERS = 1
    SWALLOW_DATA = 2


class GzipDecoder(ContentDecoder):
    def __init__(self) -> None:
        self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._state = GzipDecoderState.FIRST_MEMBER

    def decompress(self, data: bytes) -> bytes:
        ret = bytearray()
        if self._state == GzipDecoderState.SWALLOW_DATA or not data:
            return bytes(ret)
        while True:
            try:
                ret += self._obj.decompress(data)
            except zlib.error:
                previous_state = self._state
                # Ignore data after the first error
                self._state = GzipDecoderState.SWALLOW_DATA
                if previous_state == GzipDecoderState.OTHER_MEMBERS:
                    # Allow trailing garbage acceptable in other gzip clients
                    return bytes(ret)
                raise
            data = self._obj.unused_data
            if not data:
                return bytes(ret)
            self._state = GzipDecoderState.OTHER_MEMBERS
            self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def flush(self) -> bytes:
        return self._obj.flush()


if brotli is not None:

    class BrotliDecoder(ContentDecoder):
        # Supports both 'brotlipy' and 'Brotli' packages
        # since they share an import name. The top branches
        # are for 'brotlipy' and bottom branches for 'Brotli'
        def __init__(self) -> None:
            self._obj = brotli.Decompressor()
            if hasattr(self._obj, "decompress"):
                setattr(self, "decompress", self._obj.decompress)
            else:
                setattr(self, "decompress", self._obj.process)

        def flush(self) -> bytes:
            if hasattr(self._obj, "flush"):
                return self._obj.flush()  # type: ignore[no-any-return]
            return b""


if HAS_ZSTD:

    class ZstdDecoder(ContentDecoder):
        def __init__(self) -> None:
            self._obj = zstd.ZstdDecompressor().decompressobj()

        def decompress(self, data: bytes) -> bytes:
            if not data:
                return b""
            data_parts = [self._obj.decompress(data)]
            while self._obj.eof and self._obj.unused_data:
                unused_data = self._obj.unused_data
                self._obj = zstd.ZstdDecompressor().decompressobj()
                data_parts.append(self._obj.decompress(unused_data))
            return b"".join(data_parts)

        def flush(self) -> bytes:
            ret = self._obj.flush()  # note: this is a no-op
            if not self._obj.eof:
                raise DecodeError("Zstandard data is incomplete")
            return ret


class MultiDecoder(ContentDecoder):
    """
    From RFC7231:
        If one or more encodings have been applied to a representation, the
        sender that applied the encodings MUST generate a Content-Encoding
        header field that lists the content codings in the order in which
        they were applied.
    """

    def __init__(self, modes: str) -> None:
        self._decoders = [_get_decoder(m.strip()) for m in modes.split(",")]

    def flush(self) -> bytes:
        return self._decoders[0].flush()

    def decompress(self, data: bytes) -> bytes:
        for d in reversed(self._decoders):
            data = d.decompress(data)
        return data


def _get_decoder(mode: str) -> ContentDecoder:
    if "," in mode:
        return MultiDecoder(mode)

    # According to RFC 9110 section 8.4.1.3, recipients should
    # consider x-gzip equivalent to gzip
    if mode in ("gzip", "x-gzip"):
        return GzipDecoder()

    if brotli is not None and mode == "br":
        return BrotliDecoder()

    if HAS_ZSTD and mode == "zstd":
        return ZstdDecoder()

    return DeflateDecoder()


class BytesQueueBuffer:
    """Memory-efficient bytes buffer

    To return decoded data in read() and still follow the BufferedIOBase API, we need a
    buffer to always return the correct amount of bytes.

    This buffer should be filled using calls to put()

    Our maximum memory usage is determined by the sum of the size of:

     * self.buffer, which contains the full data
     * the largest chunk that we will copy in get()

    The worst case scenario is a single chunk, in which case we'll make a full copy of
    the data inside get().
    """

    def __init__(self) -> None:
        self.buffer: typing.Deque[bytes] = collections.deque()
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def put(self, data: bytes) -> None:
        self.buffer.append(data)
        self._size += len(data)

    def get(self, n: int) -> bytes:
        if n == 0:
            return b""
        elif not self.buffer:
            raise RuntimeError("buffer is empty")
        elif n < 0:
            raise ValueError("n should be > 0")

        fetched = 0
        ret = io.BytesIO()
        while fetched < n:
            remaining = n - fetched
            chunk = self.buffer.popleft()
            chunk_length = len(chunk)
            if remaining < chunk_length:
                left_chunk, right_chunk = chunk[:remaining], chunk[remaining:]
                ret.write(left_chunk)
                self.buffer.appendleft(right_chunk)
                self._size -= remaining
                break
            else:
                ret.write(chunk)
                self._size -= chunk_length
            fetched += chunk_length

            if not self.buffer:
                break

        return ret.getvalue()

    def get_all(self) -> bytes:
        buffer = self.buffer
        if not buffer:
            assert self._size == 0
            return b""
        if len(buffer) == 1:
            result = buffer.pop()
        else:
            ret = io.BytesIO()
            ret.writelines(buffer.popleft() for _ in range(len(buffer)))
            result = ret.getvalue()
        self._size = 0
        return result


class BaseHTTPResponse(io.IOBase):
    CONTENT_DECODERS = ["gzip", "x-gzip", "deflate"]
    if brotli is not None:
        CONTENT_DECODERS += ["br"]
    if HAS_ZSTD:
        CONTENT_DECODERS += ["zstd"]
    REDIRECT_STATUSES = [301, 302, 303, 307, 308]

    DECODER_ERROR_CLASSES: tuple[type[Exception], ...] = (IOError, zlib.error)
    if brotli is not None:
        DECODER_ERROR_CLASSES += (brotli.error,)

    if HAS_ZSTD:
        DECODER_ERROR_CLASSES += (zstd.ZstdError,)

    def __init__(
        self,
        *,
        headers: typing.Mapping[str, str] | typing.Mapping[bytes, bytes] | None = None,
        status: int,
        version: int,
        version_string: str,
        reason: str | None,
        decode_content: bool,
        request_url: str | None,
        retries: Retry | None = None,
    ) -> None:
        if isinstance(headers, HTTPHeaderDict):
            self.headers = headers
        else:
            self.headers = HTTPHeaderDict(headers)  # type: ignore[arg-type]
        self.status = status
        self.version = version
        self.version_string = version_string
        self.reason = reason
        self.decode_content = decode_content
        self._has_decoded_content = False
        self._request_url: str | None = request_url
        self.retries = retries

        self.chunked = False
        tr_enc = self.headers.get("transfer-encoding", "").lower()
        # Don't incur the penalty of creating a list and then discarding it
        encodings = (enc.strip() for enc in tr_enc.split(","))
        if "chunked" in encodings:
            self.chunked = True

        self._decoder: ContentDecoder | None = None
        self.length_remaining: int | None

    def get_redirect_location(self) -> str | None | typing.Literal[False]:
        """
        Should we redirect and where to?

        :returns: Truthy redirect location string if we got a redirect status
            code and valid location. ``None`` if redirect status and no
            location. ``False`` if not a redirect status code.
        """
        if self.status in self.REDIRECT_STATUSES:
            return self.headers.get("location")
        return False

    @property
    def data(self) -> bytes:
        raise NotImplementedError()

    def json(self) -> typing.Any:
        """
        Deserializes the body of the HTTP response as a Python object.

        The body of the HTTP response must be encoded using UTF-8, as per
        `RFC 8529 Section 8.1 <https://www.rfc-editor.org/rfc/rfc8259#section-8.1>`_.

        To use a custom JSON decoder pass the result of :attr:`HTTPResponse.data` to
        your custom decoder instead.

        If the body of the HTTP response is not decodable to UTF-8, a
        `UnicodeDecodeError` will be raised. If the body of the HTTP response is not a
        valid JSON document, a `json.JSONDecodeError` will be raised.

        Read more :ref:`here <json_content>`.

        :returns: The body of the HTTP response as a Python object.
        """
        data = self.data.decode("utf-8")
        return _json.loads(data)

    @property
    def url(self) -> str | None:
        raise NotImplementedError()

    @url.setter
    def url(self, url: str | None) -> None:
        raise NotImplementedError()

    @property
    def connection(self) -> BaseHTTPConnection | None:
        raise NotImplementedError()

    @property
    def retries(self) -> Retry | None:
        return self._retries

    @retries.setter
    def retries(self, retries: Retry | None) -> None:
        # Override the request_url if retries has a redirect location.
        if retries is not None and retries.history:
            self.url = retries.history[-1].redirect_location
        self._retries = retries

    def stream(
        self, amt: int | None = 2**16, decode_content: bool | None = None
    ) -> typing.Iterator[bytes]:
        raise NotImplementedError()

    def read(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,
        cache_content: bool = False,
    ) -> bytes:
        raise NotImplementedError()

    def read1(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,
    ) -> bytes:
        raise NotImplementedError()

    def read_chunked(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,
    ) -> typing.Iterator[bytes]:
        raise NotImplementedError()

    def release_conn(self) -> None:
        raise NotImplementedError()

    def drain_conn(self) -> None:
        raise NotImplementedError()

    def shutdown(self) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()

    def _init_decoder(self) -> None:
        """
        Set-up the _decoder attribute if necessary.
        """
        # Note: content-encoding value should be case-insensitive, per RFC 7230
        # Section 3.2
        content_encoding = self.headers.get("content-encoding", "").lower()
        if self._decoder is None:
            if content_encoding in self.CONTENT_DECODERS:
                self._decoder = _get_decoder(content_encoding)
            elif "," in content_encoding:
                encodings = [
                    e.strip()
                    for e in content_encoding.split(",")
                    if e.strip() in self.CONTENT_DECODERS
                ]
                if encodings:
                    self._decoder = _get_decoder(content_encoding)

    def _decode(
        self, data: bytes, decode_content: bool | None, flush_decoder: bool
    ) -> bytes:
        """
        Decode the data passed in and potentially flush the decoder.
        """
        if not decode_content:
            if self._has_decoded_content:
                raise RuntimeError(
                    "Calling read(decode_content=False) is not supported after "
                    "read(decode_content=True) was called."
                )
            return data

        try:
            if self._decoder:
                data = self._decoder.decompress(data)
                self._has_decoded_content = True
        except self.DECODER_ERROR_CLASSES as e:
            content_encoding = self.headers.get("content-encoding", "").lower()
            raise DecodeError(
                "Received response with content-encoding: %s, but "
                "failed to decode it." % content_encoding,
                e,
            ) from e
        if flush_decoder:
            data += self._flush_decoder()

        return data

    def _flush_decoder(self) -> bytes:
        """
        Flushes the decoder. Should only be called if the decoder is actually
        being used.
        """
        if self._decoder:
            return self._decoder.decompress(b"") + self._decoder.flush()
        return b""

    # Compatibility methods for `io` module
    def readinto(self, b: bytearray) -> int:
        temp = self.read(len(b))
        if len(temp) == 0:
            return 0
        else:
            b[: len(temp)] = temp
            return len(temp)

    # Compatibility methods for http.client.HTTPResponse
    def getheaders(self) -> HTTPHeaderDict:
        warnings.warn(
            "HTTPResponse.getheaders() is deprecated and will be removed "
            "in urllib3 v2.1.0. Instead access HTTPResponse.headers directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.headers

    def getheader(self, name: str, default: str | None = None) -> str | None:
        warnings.warn(
            "HTTPResponse.getheader() is deprecated and will be removed "
            "in urllib3 v2.1.0. Instead use HTTPResponse.headers.get(name, default).",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.headers.get(name, default)

    # Compatibility method for http.cookiejar
    def info(self) -> HTTPHeaderDict:
        return self.headers

    def geturl(self) -> str | None:
        return self.url


class HTTPResponse(BaseHTTPResponse):
    """
    HTTP Response container.

    Backwards-compatible with :class:`http.client.HTTPResponse` but the response ``body`` is
    loaded and decoded on-demand when the ``data`` property is accessed.  This
    class is also compatible with the Python standard library's :mod:`io`
    module, and can hence be treated as a readable object in the context of that
    framework.

    Extra parameters for behaviour not present in :class:`http.client.HTTPResponse`:

    :param preload_content:
        If True, the response's body will be preloaded during construction.

    :param decode_content:
        If True, will attempt to decode the body based on the
        'content-encoding' header.

    :param original_response:
        When this HTTPResponse wrapper is generated from an :class:`http.client.HTTPResponse`
        object, it's convenient to include the original for debug purposes. It's
        otherwise unused.

    :param retries:
        The retries contains the last :class:`~urllib3.util.retry.Retry` that
        was used during the request.

    :param enforce_content_length:
        Enforce content length checking. Body returned by server must match
        value of Content-Length header, if present. Otherwise, raise error.
    """

    def __init__(
        self,
        body: _TYPE_BODY = "",
        headers: typing.Mapping[str, str] | typing.Mapping[bytes, bytes] | None = None,
        status: int = 0,
        version: int = 0,
        version_string: str = "HTTP/?",
        reason: str | None = None,
        preload_content: bool = True,
        decode_content: bool = True,
        original_response: _HttplibHTTPResponse | None = None,
        pool: HTTPConnectionPool | None = None,
        connection: HTTPConnection | None = None,
        msg: _HttplibHTTPMessage | None = None,
        retries: Retry | None = None,
        enforce_content_length: bool = True,
        request_method: str | None = None,
        request_url: str | None = None,
        auto_close: bool = True,
        sock_shutdown: typing.Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(
            headers=headers,
            status=status,
            version=version,
            version_string=version_string,
            reason=reason,
            decode_content=decode_content,
            request_url=request_url,
            retries=retries,
        )

        self.enforce_content_length = enforce_content_length
        self.auto_close = auto_close

        self._body = None
        self._fp: _HttplibHTTPResponse | None = None
        self._original_response = original_response
        self._fp_bytes_read = 0
        self.msg = msg

        if body and isinstance(body, (str, bytes)):
            self._body = body

        self._pool = pool
        self._connection = connection

        if hasattr(body, "read"):
            self._fp = body  # type: ignore[assignment]
        self._sock_shutdown = sock_shutdown

        # Are we using the chunked-style of transfer encoding?
        self.chunk_left: int | None = None

        # Determine length of response
        self.length_remaining = self._init_length(request_method)

        # Used to return the correct amount of bytes for partial read()s
        self._decoded_buffer = BytesQueueBuffer()

        # If requested, preload the body.
        if preload_content and not self._body:
            self._body = self.read(decode_content=decode_content)

    def release_conn(self) -> None:
        if not self._pool or not self._connection:
            return None

        self._pool._put_conn(self._connection)
        self._connection = None

    def drain_conn(self) -> None:
        """
        Read and discard any remaining HTTP response data in the response connection.

        Unread data in the HTTPResponse connection blocks the connection from being released back to the pool.
        """
        try:
            self.read()
        except (HTTPError, OSError, BaseSSLError, HTTPException):
            pass

    @property
    def data(self) -> bytes:
        # For backwards-compat with earlier urllib3 0.4 and earlier.
        if self._body:
            return self._body  # type: ignore[return-value]

        if self._fp:
            return self.read(cache_content=True)

        return None  # type: ignore[return-value]

    @property
    def connection(self) -> HTTPConnection | None:
        return self._connection

    def isclosed(self) -> bool:
        return is_fp_closed(self._fp)

    def tell(self) -> int:
        """
        Obtain the number of bytes pulled over the wire so far. May differ from
        the amount of content returned by :meth:``urllib3.response.HTTPResponse.read``
        if bytes are encoded on the wire (e.g, compressed).
        """
        return self._fp_bytes_read

    def _init_length(self, request_method: str | None) -> int | None:
        """
        Set initial length value for Response content if available.
        """
        length: int | None
        content_length: str | None = self.headers.get("content-length")

        if content_length is not None:
            if self.chunked:
                # This Response will fail with an IncompleteRead if it can't be
                # received as chunked. This method falls back to attempt reading
                # the response before raising an exception.
                log.warning(
                    "Received response with both Content-Length and "
                    "Transfer-Encoding set. This is expressly forbidden "
                    "by RFC 7230 sec 3.3.2. Ignoring Content-Length and "
                    "attempting to process response as Transfer-Encoding: "
                    "chunked."
                )
                return None

            try:
                # RFC 7230 section 3.3.2 specifies multiple content lengths can
                # be sent in a single Content-Length header
                # (e.g. Content-Length: 42, 42). This line ensures the values
                # are all valid ints and that as long as the `set` length is 1,
                # all values are the same. Otherwise, the header is invalid.
                lengths = {int(val) for val in content_length.split(",")}
                if len(lengths) > 1:
                    raise InvalidHeader(
                        "Content-Length contained multiple "
                        "unmatching values (%s)" % content_length
                    )
                length = lengths.pop()
            except ValueError:
                length = None
            else:
                if length < 0:
                    length = None

        else:  # if content_length is None
            length = None

        # Convert status to int for comparison
        # In some cases, httplib returns a status of "_UNKNOWN"
        try:
            status = int(self.status)
        except ValueError:
            status = 0

        # Check for responses that shouldn't include a body
        if status in (204, 304) or 100 <= status < 200 or request_method == "HEAD":
            length = 0

        return length

    @contextmanager
    def _error_catcher(self) -> typing.Generator[None]:
        """
        Catch low-level python exceptions, instead re-raising urllib3
        variants, so that low-level exceptions are not leaked in the
        high-level api.

        On exit, release the connection back to the pool.
        """
        clean_exit = False

        try:
            try:
                yield

            except SocketTimeout as e:
                # FIXME: Ideally we'd like to include the url in the ReadTimeoutError but
                # there is yet no clean way to get at it from this context.
                raise ReadTimeoutError(self._pool, None, "Read timed out.") from e  # type: ignore[arg-type]

            except BaseSSLError as e:
                # FIXME: Is there a better way to differentiate between SSLErrors?
                if "read operation timed out" not in str(e):
                    # SSL errors related to framing/MAC get wrapped and reraised here
                    raise SSLError(e) from e

                raise ReadTimeoutError(self._pool, None, "Read timed out.") from e  # type: ignore[arg-type]

            except IncompleteRead as e:
                if (
                    e.expected is not None
                    and e.partial is not None
                    and e.expected == -e.partial
                ):
                    arg = "Response may not contain content."
                else:
                    arg = f"Connection broken: {e!r}"
                raise ProtocolError(arg, e) from e

            except (HTTPException, OSError) as e:
                raise ProtocolError(f"Connection broken: {e!r}", e) from e

            # If no exception is thrown, we should avoid cleaning up
            # unnecessarily.
            clean_exit = True
        finally:
            # If we didn't terminate cleanly, we need to throw away our
            # connection.
            if not clean_exit:
                # The response may not be closed but we're not going to use it
                # anymore so close it now to ensure that the connection is
                # released back to the pool.
                if self._original_response:
                    self._original_response.close()

                # Closing the response may not actually be sufficient to close
                # everything, so if we have a hold of the connection close that
                # too.
                if self._connection:
                    self._connection.close()

            # If we hold the original response but it's closed now, we should
            # return the connection back to the pool.
            if self._original_response and self._original_response.isclosed():
                self.release_conn()

    def _fp_read(
        self,
        amt: int | None = None,
        *,
        read1: bool = False,
    ) -> bytes:
        """
        Read a response with the thought that reading the number of bytes
        larger than can fit in a 32-bit int at a time via SSL in some
        known cases leads to an overflow error that has to be prevented
        if `amt` or `self.length_remaining` indicate that a problem may
        happen.

        The known cases:
          * CPython < 3.9.7 because of a bug
            https://github.com/urllib3/urllib3/issues/2513#issuecomment-1152559900.
          * urllib3 injected with pyOpenSSL-backed SSL-support.
          * CPython < 3.10 only when `amt` does not fit 32-bit int.
        """
        assert self._fp
        c_int_max = 2**31 - 1
        if (
            (amt and amt > c_int_max)
            or (
                amt is None
                and self.length_remaining
                and self.length_remaining > c_int_max
            )
        ) and (util.IS_PYOPENSSL or sys.version_info < (3, 10)):
            if read1:
                return self._fp.read1(c_int_max)
            buffer = io.BytesIO()
            # Besides `max_chunk_amt` being a maximum chunk size, it
            # affects memory overhead of reading a response by this
            # method in CPython.
            # `c_int_max` equal to 2 GiB - 1 byte is the actual maximum
            # chunk size that does not lead to an overflow error, but
            # 256 MiB is a compromise.
            max_chunk_amt = 2**28
            while amt is None or amt != 0:
                if amt is not None:
                    chunk_amt = min(amt, max_chunk_amt)
                    amt -= chunk_amt
                else:
                    chunk_amt = max_chunk_amt
                data = self._fp.read(chunk_amt)
                if not data:
                    break
                buffer.write(data)
                del data  # to reduce peak memory usage by `max_chunk_amt`.
            return buffer.getvalue()
        elif read1:
            return self._fp.read1(amt) if amt is not None else self._fp.read1()
        else:
            # StringIO doesn't like amt=None
            return self._fp.read(amt) if amt is not None else self._fp.read()

    def _raw_read(
        self,
        amt: int | None = None,
        *,
        read1: bool = False,
    ) -> bytes:
        """
        Reads `amt` of bytes from the socket.
        """
        if self._fp is None:
            return None  # type: ignore[return-value]

        fp_closed = getattr(self._fp, "closed", False)

        with self._error_catcher():
            data = self._fp_read(amt, read1=read1) if not fp_closed else b""
            if amt is not None and amt != 0 and not data:
                # Platform-specific: Buggy versions of Python.
                # Close the connection when no data is returned
                #
                # This is redundant to what httplib/http.client _should_
                # already do.  However, versions of python released before
                # December 15, 2012 (http://bugs.python.org/issue16298) do
                # not properly close the connection in all cases. There is
                # no harm in redundantly calling close.
                self._fp.close()
                if (
                    self.enforce_content_length
                    and self.length_remaining is not None
                    and self.length_remaining != 0
                ):
                    # This is an edge case that httplib failed to cover due
                    # to concerns of backward compatibility. We're
                    # addressing it here to make sure IncompleteRead is
                    # raised during streaming, so all calls with incorrect
                    # Content-Length are caught.
                    raise IncompleteRead(self._fp_bytes_read, self.length_remaining)
            elif read1 and (
                (amt != 0 and not data) or self.length_remaining == len(data)
            ):
                # All data has been read, but `self._fp.read1` in
                # CPython 3.12 and older doesn't always close
                # `http.client.HTTPResponse`, so we close it here.
                # See https://github.com/python/cpython/issues/113199
                self._fp.close()

        if data:
            self._fp_bytes_read += len(data)
            if self.length_remaining is not None:
                self.length_remaining -= len(data)
        return data

    def read(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,
        cache_content: bool = False,
    ) -> bytes:
        """
        Similar to :meth:`http.client.HTTPResponse.read`, but with two additional
        parameters: ``decode_content`` and ``cache_content``.

        :param amt:
            How much of the content to read. If specified, caching is skipped
            because it doesn't make sense to cache partial content as the full
            response.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.

        :param cache_content:
            If True, will save the returned data such that the same result is
            returned despite of the state of the underlying file object. This
            is useful if you want the ``.data`` property to continue working
            after having ``.read()`` the file object. (Overridden if ``amt`` is
            set.)
        """
        self._init_decoder()
        if decode_content is None:
            decode_content = self.decode_content

        if amt and amt < 0:
            # Negative numbers and `None` should be treated the same.
            amt = None
        elif amt is not None:
            cache_content = False

            if len(self._decoded_buffer) >= amt:
                return self._decoded_buffer.get(amt)

        data = self._raw_read(amt)

        flush_decoder = amt is None or (amt != 0 and not data)

        if not data and len(self._decoded_buffer) == 0:
            return data

        if amt is None:
            data = self._decode(data, decode_content, flush_decoder)
            if cache_content:
                self._body = data
        else:
            # do not waste memory on buffer when not decoding
            if not decode_content:
                if self._has_decoded_content:
                    raise RuntimeError(
                        "Calling read(decode_content=False) is not supported after "
                        "read(decode_content=True) was called."
                    )
                return data

            decoded_data = self._decode(data, decode_content, flush_decoder)
            self._decoded_buffer.put(decoded_data)

            while len(self._decoded_buffer) < amt and data:
                # TODO make sure to initially read enough data to get past the headers
                # For example, the GZ file header takes 10 bytes, we don't want to read
                # it one byte at a time
                data = self._raw_read(amt)
                decoded_data = self._decode(data, decode_content, flush_decoder)
                self._decoded_buffer.put(decoded_data)
            data = self._decoded_buffer.get(amt)

        return data

    def read1(
        self,
        amt: int | None = None,
        decode_content: bool | None = None,
    ) -> bytes:
        """
        Similar to ``http.client.HTTPResponse.read1`` and documented
        in :meth:`io.BufferedReader.read1`, but with an additional parameter:
        ``decode_content``.

        :param amt:
            How much of the content to read.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        if decode_content is None:
            decode_content = self.decode_content
        if amt and amt < 0:
            # Negative numbers and `None` should be treated the same.
            amt = None
        # try and respond without going to the network
        if self._has_decoded_content:
            if not decode_content:
                raise RuntimeError(
                    "Calling read1(decode_content=False) is not supported after "
                    "read1(decode_content=True) was called."
                )
            if len(self._decoded_buffer) > 0:
                if amt is None:
                    return self._decoded_buffer.get_all()
                return self._decoded_buffer.get(amt)
        if amt == 0:
            return b""

        # FIXME, this method's type doesn't say returning None is possible
        data = self._raw_read(amt, read1=True)
        if not decode_content or data is None:
            return data

        self._init_decoder()
        while True:
            flush_decoder = not data
            decoded_data = self._decode(data, decode_content, flush_decoder)
            self._decoded_buffer.put(decoded_data)
            if decoded_data or flush_decoder:
                break
            data = self._raw_read(8192, read1=True)

        if amt is None:
            return self._decoded_buffer.get_all()
        return self._decoded_buffer.get(amt)

    def stream(
        self, amt: int | None = 2**16, decode_content: bool | None = None
    ) -> typing.Generator[bytes]:
        """
        A generator wrapper for the read() method. A call will block until
        ``amt`` bytes have been read from the connection or until the
        connection is closed.

        :param amt:
            How much of the content to read. The generator will return up to
            much data per iteration, but may return less. This is particularly
            likely when using compressed data. However, the empty string will
            never be returned.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        if self.chunked and self.supports_chunked_reads():
            yield from self.read_chunked(amt, decode_content=decode_content)
        else:
            while not is_fp_closed(self._fp) or len(self._decoded_buffer) > 0:
                data = self.read(amt=amt, decode_content=decode_content)

                if data:
                    yield data

    # Overrides from io.IOBase
    def readable(self) -> bool:
        return True

    def shutdown(self) -> None:
        if not self._sock_shutdown:
            raise ValueError("Cannot shutdown socket as self._sock_shutdown is not set")
        self._sock_shutdown(socket.SHUT_RD)

    def close(self) -> None:
        self._sock_shutdown = None

        if not self.closed and self._fp:
            self._fp.close()

        if self._connection:
            self._connection.close()

        if not self.auto_close:
            io.IOBase.close(self)

    @property
    def closed(self) -> bool:
        if not self.auto_close:
            return io.IOBase.closed.__get__(self)  # type: ignore[no-any-return]
        elif self._fp is None:
            return True
        elif hasattr(self._fp, "isclosed"):
            return self._fp.isclosed()
        elif hasattr(self._fp, "closed"):
            return self._fp.closed
        else:
            return True

    def fileno(self) -> int:
        if self._fp is None:
            raise OSError("HTTPResponse has no file to get a fileno from")
        elif hasattr(self._fp, "fileno"):
            return self._fp.fileno()
        else:
            raise OSError(
                "The file-like object this HTTPResponse is wrapped "
                "around has no file descriptor"
            )

    def flush(self) -> None:
        if (
            self._fp is not None
            and hasattr(self._fp, "flush")
            and not getattr(self._fp, "closed", False)
        ):
            return self._fp.flush()

    def supports_chunked_reads(self) -> bool:
        """
        Checks if the underlying file-like object looks like a
        :class:`http.client.HTTPResponse` object. We do this by testing for
        the fp attribute. If it is present we assume it returns raw chunks as
        processed by read_chunked().
        """
        return hasattr(self._fp, "fp")

    def _update_chunk_length(self) -> None:
        # First, we'll figure out length of a chunk and then
        # we'll try to read it from socket.
        if self.chunk_left is not None:
            return None
        line = self._fp.fp.readline()  # type: ignore[union-attr]
        line = line.split(b";", 1)[0]
        try:
            self.chunk_left = int(line, 16)
        except ValueError:
            self.close()
            if line:
                # Invalid chunked protocol response, abort.
                raise InvalidChunkLength(self, line) from None
            else:
                # Truncated at start of next chunk
                raise ProtocolError("Response ended prematurely") from None

    def _handle_chunk(self, amt: int | None) -> bytes:
        returned_chunk = None
        if amt is None:
            chunk = self._fp._safe_read(self.chunk_left)  # type: ignore[union-attr]
            returned_chunk = chunk
            self._fp._safe_read(2)  # type: ignore[union-attr] # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
        elif self.chunk_left is not None and amt < self.chunk_left:
            value = self._fp._safe_read(amt)  # type: ignore[union-attr]
            self.chunk_left = self.chunk_left - amt
            returned_chunk = value
        elif amt == self.chunk_left:
            value = self._fp._safe_read(amt)  # type: ignore[union-attr]
            self._fp._safe_read(2)  # type: ignore[union-attr] # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
            returned_chunk = value
        else:  # amt > self.chunk_left
            returned_chunk = self._fp._safe_read(self.chunk_left)  # type: ignore[union-attr]
            self._fp._safe_read(2)  # type: ignore[union-attr] # Toss the CRLF at the end of the chunk.
            self.chunk_left = None
        return returned_chunk  # type: ignore[no-any-return]

    def read_chunked(
        self, amt: int | None = None, decode_content: bool | None = None
    ) -> typing.Generator[bytes]:
        """
        Similar to :meth:`HTTPResponse.read`, but with an additional
        parameter: ``decode_content``.

        :param amt:
            How much of the content to read. If specified, caching is skipped
            because it doesn't make sense to cache partial content as the full
            response.

        :param decode_content:
            If True, will attempt to decode the body based on the
            'content-encoding' header.
        """
        self._init_decoder()
        # FIXME: Rewrite this method and make it a class with a better structured logic.
        if not self.chunked:
            raise ResponseNotChunked(
                "Response is not chunked. "
                "Header 'transfer-encoding: chunked' is missing."
            )
        if not self.supports_chunked_reads():
            raise BodyNotHttplibCompatible(
                "Body should be http.client.HTTPResponse like. "
                "It should have have an fp attribute which returns raw chunks."
            )

        with self._error_catcher():
            # Don't bother reading the body of a HEAD request.
            if self._original_response and is_response_to_head(self._original_response):
                self._original_response.close()
                return None

            # If a response is already read and closed
            # then return immediately.
            if self._fp.fp is None:  # type: ignore[union-attr]
                return None

            if amt and amt < 0:
                # Negative numbers and `None` should be treated the same,
                # but httplib handles only `None` correctly.
                amt = None

            while True:
                self._update_chunk_length()
                if self.chunk_left == 0:
                    break
                chunk = self._handle_chunk(amt)
                decoded = self._decode(
                    chunk, decode_content=decode_content, flush_decoder=False
                )
                if decoded:
                    yield decoded

            if decode_content:
                # On CPython and PyPy, we should never need to flush the
                # decoder. However, on Jython we *might* need to, so
                # lets defensively do it anyway.
                decoded = self._flush_decoder()
                if decoded:  # Platform-specific: Jython.
                    yield decoded

            # Chunk content ends with \r\n: discard it.
            while self._fp is not None:
                line = self._fp.fp.readline()
                if not line:
                    # Some sites may not end with '\r\n'.
                    break
                if line == b"\r\n":
                    break

            # We read everything; close the "file".
            if self._original_response:
                self._original_response.close()

    @property
    def url(self) -> str | None:
        """
        Returns the URL that was the source of this response.
        If the request that generated this response redirected, this method
        will return the final redirect location.
        """
        return self._request_url

    @url.setter
    def url(self, url: str) -> None:
        self._request_url = url

    def __iter__(self) -> typing.Iterator[bytes]:
        buffer: list[bytes] = []
        for chunk in self.stream(decode_content=True):
            if b"\n" in chunk:
                chunks = chunk.split(b"\n")
                yield b"".join(buffer) + chunks[0] + b"\n"
                for x in chunks[1:-1]:
                    yield x + b"\n"
                if chunks[-1]:
                    buffer = [chunks[-1]]
                else:
                    buffer = []
            else:
                buffer.append(chunk)
        if buffer:
            yield b"".join(buffer)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\sympy_parser.py ===
"""Transform a string with Python-like source code into SymPy expression. """
from __future__ import annotations
from tokenize import (generate_tokens, untokenize, TokenError,
    NUMBER, STRING, NAME, OP, ENDMARKER, ERRORTOKEN, NEWLINE)

from keyword import iskeyword

import ast
import unicodedata
from io import StringIO
import builtins
import types
from typing import Any, Callable
from functools import reduce
from sympy.assumptions.ask import AssumptionKeys
from sympy.core.basic import Basic
from sympy.core import Symbol
from sympy.core.function import Function
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Max, Min


null = ''

TOKEN = tuple[int, str]
DICT = dict[str, Any]
TRANS = Callable[[list[TOKEN], DICT, DICT], list[TOKEN]]

def _token_splittable(token_name: str) -> bool:
    """
    Predicate for whether a token name can be split into multiple tokens.

    A token is splittable if it does not contain an underscore character and
    it is not the name of a Greek letter. This is used to implicitly convert
    expressions like 'xyz' into 'x*y*z'.
    """
    if '_' in token_name:
        return False
    try:
        return not unicodedata.lookup('GREEK SMALL LETTER ' + token_name)
    except KeyError:
        return len(token_name) > 1


def _token_callable(token: TOKEN, local_dict: DICT, global_dict: DICT, nextToken=None):
    """
    Predicate for whether a token name represents a callable function.

    Essentially wraps ``callable``, but looks up the token name in the
    locals and globals.
    """
    func = local_dict.get(token[1])
    if not func:
        func = global_dict.get(token[1])
    return callable(func) and not isinstance(func, Symbol)


def _add_factorial_tokens(name: str, result: list[TOKEN]) -> list[TOKEN]:
    if result == [] or result[-1][1] == '(':
        raise TokenError()

    beginning = [(NAME, name), (OP, '(')]
    end = [(OP, ')')]

    diff = 0
    length = len(result)

    for index, token in enumerate(result[::-1]):
        toknum, tokval = token
        i = length - index - 1

        if tokval == ')':
            diff += 1
        elif tokval == '(':
            diff -= 1

        if diff == 0:
            if i - 1 >= 0 and result[i - 1][0] == NAME:
                return result[:i - 1] + beginning + result[i - 1:] + end
            else:
                return result[:i] + beginning + result[i:] + end

    return result


class ParenthesisGroup(list[TOKEN]):
    """List of tokens representing an expression in parentheses."""
    pass


class AppliedFunction:
    """
    A group of tokens representing a function and its arguments.

    `exponent` is for handling the shorthand sin^2, ln^2, etc.
    """
    def __init__(self, function: TOKEN, args: ParenthesisGroup, exponent=None):
        if exponent is None:
            exponent = []
        self.function = function
        self.args = args
        self.exponent = exponent
        self.items = ['function', 'args', 'exponent']

    def expand(self) -> list[TOKEN]:
        """Return a list of tokens representing the function"""
        return [self.function, *self.args]

    def __getitem__(self, index):
        return getattr(self, self.items[index])

    def __repr__(self):
        return "AppliedFunction(%s, %s, %s)" % (self.function, self.args,
                                                self.exponent)


def _flatten(result: list[TOKEN | AppliedFunction]):
    result2: list[TOKEN] = []
    for tok in result:
        if isinstance(tok, AppliedFunction):
            result2.extend(tok.expand())
        else:
            result2.append(tok)
    return result2


def _group_parentheses(recursor: TRANS):
    def _inner(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
        """Group tokens between parentheses with ParenthesisGroup.

        Also processes those tokens recursively.

        """
        result: list[TOKEN | ParenthesisGroup] = []
        stacks: list[ParenthesisGroup] = []
        stacklevel = 0
        for token in tokens:
            if token[0] == OP:
                if token[1] == '(':
                    stacks.append(ParenthesisGroup([]))
                    stacklevel += 1
                elif token[1] == ')':
                    stacks[-1].append(token)
                    stack = stacks.pop()

                    if len(stacks) > 0:
                        # We don't recurse here since the upper-level stack
                        # would reprocess these tokens
                        stacks[-1].extend(stack)
                    else:
                        # Recurse here to handle nested parentheses
                        # Strip off the outer parentheses to avoid an infinite loop
                        inner = stack[1:-1]
                        inner = recursor(inner,
                                         local_dict,
                                         global_dict)
                        parenGroup = [stack[0]] + inner + [stack[-1]]
                        result.append(ParenthesisGroup(parenGroup))
                    stacklevel -= 1
                    continue
            if stacklevel:
                stacks[-1].append(token)
            else:
                result.append(token)
        if stacklevel:
            raise TokenError("Mismatched parentheses")
        return result
    return _inner


def _apply_functions(tokens: list[TOKEN | ParenthesisGroup], local_dict: DICT, global_dict: DICT):
    """Convert a NAME token + ParenthesisGroup into an AppliedFunction.

    Note that ParenthesisGroups, if not applied to any function, are
    converted back into lists of tokens.

    """
    result: list[TOKEN | AppliedFunction] = []
    symbol = None
    for tok in tokens:
        if isinstance(tok, ParenthesisGroup):
            if symbol and _token_callable(symbol, local_dict, global_dict):
                result[-1] = AppliedFunction(symbol, tok)
                symbol = None
            else:
                result.extend(tok)
        elif tok[0] == NAME:
            symbol = tok
            result.append(tok)
        else:
            symbol = None
            result.append(tok)
    return result


def _implicit_multiplication(tokens: list[TOKEN | AppliedFunction], local_dict: DICT, global_dict: DICT):
    """Implicitly adds '*' tokens.

    Cases:

    - Two AppliedFunctions next to each other ("sin(x)cos(x)")

    - AppliedFunction next to an open parenthesis ("sin x (cos x + 1)")

    - A close parenthesis next to an AppliedFunction ("(x+2)sin x")\

    - A close parenthesis next to an open parenthesis ("(x+2)(x+3)")

    - AppliedFunction next to an implicitly applied function ("sin(x)cos x")

    """
    result: list[TOKEN | AppliedFunction] = []
    skip = False
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if skip:
            skip = False
            continue
        if tok[0] == OP and tok[1] == '.' and nextTok[0] == NAME:
            # Dotted name. Do not do implicit multiplication
            skip = True
            continue
        if isinstance(tok, AppliedFunction):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '*'))
            elif nextTok == (OP, '('):
                # Applied function followed by an open parenthesis
                if tok.function[1] == "Function":
                    tok.function = (tok.function[0], 'Symbol')
                result.append((OP, '*'))
            elif nextTok[0] == NAME:
                # Applied function followed by implicitly applied function
                result.append((OP, '*'))
        else:
            if tok == (OP, ')'):
                if isinstance(nextTok, AppliedFunction):
                    # Close parenthesis followed by an applied function
                    result.append((OP, '*'))
                elif nextTok[0] == NAME:
                    # Close parenthesis followed by an implicitly applied function
                    result.append((OP, '*'))
                elif nextTok == (OP, '('):
                    # Close parenthesis followed by an open parenthesis
                    result.append((OP, '*'))
            elif tok[0] == NAME and not _token_callable(tok, local_dict, global_dict):
                if isinstance(nextTok, AppliedFunction) or \
                    (nextTok[0] == NAME and _token_callable(nextTok, local_dict, global_dict)):
                    # Constant followed by (implicitly applied) function
                    result.append((OP, '*'))
                elif nextTok == (OP, '('):
                    # Constant followed by parenthesis
                    result.append((OP, '*'))
                elif nextTok[0] == NAME:
                    # Constant followed by constant
                    result.append((OP, '*'))
    if tokens:
        result.append(tokens[-1])
    return result


def _implicit_application(tokens: list[TOKEN | AppliedFunction], local_dict: DICT, global_dict: DICT):
    """Adds parentheses as needed after functions."""
    result: list[TOKEN | AppliedFunction] = []
    appendParen = 0  # number of closing parentheses to add
    skip = 0  # number of tokens to delay before adding a ')' (to
              # capture **, ^, etc.)
    exponentSkip = False  # skipping tokens before inserting parentheses to
                          # work with function exponentiation
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if (tok[0] == NAME and nextTok[0] not in [OP, ENDMARKER, NEWLINE]):
            if _token_callable(tok, local_dict, global_dict, nextTok):  # type: ignore
                result.append((OP, '('))
                appendParen += 1
        # name followed by exponent - function exponentiation
        elif (tok[0] == NAME and nextTok[0] == OP and nextTok[1] == '**'):
            if _token_callable(tok, local_dict, global_dict):  # type: ignore
                exponentSkip = True
        elif exponentSkip:
            # if the last token added was an applied function (i.e. the
            # power of the function exponent) OR a multiplication (as
            # implicit multiplication would have added an extraneous
            # multiplication)
            if (isinstance(tok, AppliedFunction)
                or (tok[0] == OP and tok[1] == '*')):
                # don't add anything if the next token is a multiplication
                # or if there's already a parenthesis (if parenthesis, still
                # stop skipping tokens)
                if not (nextTok[0] == OP and nextTok[1] == '*'):
                    if not(nextTok[0] == OP and nextTok[1] == '('):
                        result.append((OP, '('))
                        appendParen += 1
                    exponentSkip = False
        elif appendParen:
            if nextTok[0] == OP and nextTok[1] in ('^', '**', '*'):
                skip = 1
                continue
            if skip:
                skip -= 1
                continue
            result.append((OP, ')'))
            appendParen -= 1

    if tokens:
        result.append(tokens[-1])

    if appendParen:
        result.extend([(OP, ')')] * appendParen)
    return result


def function_exponentiation(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows functions to be exponentiated, e.g. ``cos**2(x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, function_exponentiation)
    >>> transformations = standard_transformations + (function_exponentiation,)
    >>> parse_expr('sin**4(x)', transformations=transformations)
    sin(x)**4
    """
    result: list[TOKEN] = []
    exponent: list[TOKEN] = []
    consuming_exponent = False
    level = 0
    for tok, nextTok in zip(tokens, tokens[1:]):
        if tok[0] == NAME and nextTok[0] == OP and nextTok[1] == '**':
            if _token_callable(tok, local_dict, global_dict):
                consuming_exponent = True
        elif consuming_exponent:
            if tok[0] == NAME and tok[1] == 'Function':
                tok = (NAME, 'Symbol')
            exponent.append(tok)

            # only want to stop after hitting )
            if tok[0] == nextTok[0] == OP and tok[1] == ')' and nextTok[1] == '(':
                consuming_exponent = False
            # if implicit multiplication was used, we may have )*( instead
            if tok[0] == nextTok[0] == OP and tok[1] == '*' and nextTok[1] == '(':
                consuming_exponent = False
                del exponent[-1]
            continue
        elif exponent and not consuming_exponent:
            if tok[0] == OP:
                if tok[1] == '(':
                    level += 1
                elif tok[1] == ')':
                    level -= 1
            if level == 0:
                result.append(tok)
                result.extend(exponent)
                exponent = []
                continue
        result.append(tok)
    if tokens:
        result.append(tokens[-1])
    if exponent:
        result.extend(exponent)
    return result


def split_symbols_custom(predicate: Callable[[str], bool]):
    """Creates a transformation that splits symbol names.

    ``predicate`` should return True if the symbol name is to be split.

    For instance, to retain the default behavior but avoid splitting certain
    symbol names, a predicate like this would work:


    >>> from sympy.parsing.sympy_parser import (parse_expr, _token_splittable,
    ... standard_transformations, implicit_multiplication,
    ... split_symbols_custom)
    >>> def can_split(symbol):
    ...     if symbol not in ('list', 'of', 'unsplittable', 'names'):
    ...             return _token_splittable(symbol)
    ...     return False
    ...
    >>> transformation = split_symbols_custom(can_split)
    >>> parse_expr('unsplittable', transformations=standard_transformations +
    ... (transformation, implicit_multiplication))
    unsplittable
    """
    def _split_symbols(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
        result: list[TOKEN] = []
        split = False
        split_previous=False

        for tok in tokens:
            if split_previous:
                # throw out closing parenthesis of Symbol that was split
                split_previous=False
                continue
            split_previous=False

            if tok[0] == NAME and tok[1] in ['Symbol', 'Function']:
                split = True

            elif split and tok[0] == NAME:
                symbol = tok[1][1:-1]

                if predicate(symbol):
                    tok_type = result[-2][1]  # Symbol or Function
                    del result[-2:]  # Get rid of the call to Symbol

                    i = 0
                    while i < len(symbol):
                        char = symbol[i]
                        if char in local_dict or char in global_dict:
                            result.append((NAME, "%s" % char))
                        elif char.isdigit():
                            chars = [char]
                            for i in range(i + 1, len(symbol)):
                                if not symbol[i].isdigit():
                                    i -= 1
                                    break
                                chars.append(symbol[i])
                            char = ''.join(chars)
                            result.extend([(NAME, 'Number'), (OP, '('),
                                           (NAME, "'%s'" % char), (OP, ')')])
                        else:
                            use = tok_type if i == len(symbol) else 'Symbol'
                            result.extend([(NAME, use), (OP, '('),
                                           (NAME, "'%s'" % char), (OP, ')')])
                        i += 1

                    # Set split_previous=True so will skip
                    # the closing parenthesis of the original Symbol
                    split = False
                    split_previous = True
                    continue

                else:
                    split = False

            result.append(tok)

        return result

    return _split_symbols


#: Splits symbol names for implicit multiplication.
#:
#: Intended to let expressions like ``xyz`` be parsed as ``x*y*z``. Does not
#: split Greek character names, so ``theta`` will *not* become
#: ``t*h*e*t*a``. Generally this should be used with
#: ``implicit_multiplication``.
split_symbols = split_symbols_custom(_token_splittable)


def implicit_multiplication(tokens: list[TOKEN], local_dict: DICT,
                            global_dict: DICT) -> list[TOKEN]:
    """Makes the multiplication operator optional in most cases.

    Use this before :func:`implicit_application`, otherwise expressions like
    ``sin 2x`` will be parsed as ``x * sin(2)`` rather than ``sin(2*x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication)
    >>> transformations = standard_transformations + (implicit_multiplication,)
    >>> parse_expr('3 x y', transformations=transformations)
    3*x*y
    """
    # These are interdependent steps, so we don't expose them separately
    res1 = _group_parentheses(implicit_multiplication)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_multiplication(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result


def implicit_application(tokens: list[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> list[TOKEN]:
    """Makes parentheses optional in some cases for function calls.

    Use this after :func:`implicit_multiplication`, otherwise expressions
    like ``sin 2x`` will be parsed as ``x * sin(2)`` rather than
    ``sin(2*x)``.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_application)
    >>> transformations = standard_transformations + (implicit_application,)
    >>> parse_expr('cot z + csc z', transformations=transformations)
    cot(z) + csc(z)
    """
    res1 = _group_parentheses(implicit_application)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_application(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result


def implicit_multiplication_application(result: list[TOKEN], local_dict: DICT,
                                        global_dict: DICT) -> list[TOKEN]:
    """Allows a slightly relaxed syntax.

    - Parentheses for single-argument method calls are optional.

    - Multiplication is implicit.

    - Symbol names can be split (i.e. spaces are not needed between
      symbols).

    - Functions can be exponentiated.

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, implicit_multiplication_application)
    >>> parse_expr("10sin**2 x**2 + 3xyz + tan theta",
    ... transformations=(standard_transformations +
    ... (implicit_multiplication_application,)))
    3*x*y*z + 10*sin(x**2)**2 + tan(theta)

    """
    for step in (split_symbols, implicit_multiplication,
                 implicit_application, function_exponentiation):
        result = step(result, local_dict, global_dict)

    return result


def auto_symbol(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Inserts calls to ``Symbol``/``Function`` for undefined variables."""
    result: list[TOKEN] = []
    prevTok = (-1, '')

    tokens.append((-1, ''))  # so zip traverses all tokens
    for tok, nextTok in zip(tokens, tokens[1:]):
        tokNum, tokVal = tok
        nextTokNum, nextTokVal = nextTok
        if tokNum == NAME:
            name = tokVal

            if (name in ['True', 'False', 'None']
                    or iskeyword(name)
                    # Don't convert attribute access
                    or (prevTok[0] == OP and prevTok[1] == '.')
                    # Don't convert keyword arguments
                    or (prevTok[0] == OP and prevTok[1] in ('(', ',')
                        and nextTokNum == OP and nextTokVal == '=')
                    # the name has already been defined
                    or name in local_dict and local_dict[name] is not null):
                result.append((NAME, name))
                continue
            elif name in local_dict:
                local_dict.setdefault(null, set()).add(name)
                if nextTokVal == '(':
                    local_dict[name] = Function(name)
                else:
                    local_dict[name] = Symbol(name)
                result.append((NAME, name))
                continue
            elif name in global_dict:
                obj = global_dict[name]
                if isinstance(obj, (AssumptionKeys, Basic, type)) or callable(obj):
                    result.append((NAME, name))
                    continue

            result.extend([
                (NAME, 'Symbol' if nextTokVal != '(' else 'Function'),
                (OP, '('),
                (NAME, repr(str(name))),
                (OP, ')'),
            ])
        else:
            result.append((tokNum, tokVal))

        prevTok = (tokNum, tokVal)

    return result


def lambda_notation(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Substitutes "lambda" with its SymPy equivalent Lambda().
    However, the conversion does not take place if only "lambda"
    is passed because that is a syntax error.

    """
    result: list[TOKEN] = []
    flag = False
    toknum, tokval = tokens[0]
    tokLen = len(tokens)

    if toknum == NAME and tokval == 'lambda':
        if tokLen == 2 or tokLen == 3 and tokens[1][0] == NEWLINE:
            # In Python 3.6.7+, inputs without a newline get NEWLINE added to
            # the tokens
            result.extend(tokens)
        elif tokLen > 2:
            result.extend([
                (NAME, 'Lambda'),
                (OP, '('),
                (OP, '('),
                (OP, ')'),
                (OP, ')'),
            ])
            for tokNum, tokVal in tokens[1:]:
                if tokNum == OP and tokVal == ':':
                    tokVal = ','
                    flag = True
                if not flag and tokNum == OP and tokVal in ('*', '**'):
                    raise TokenError("Starred arguments in lambda not supported")
                if flag:
                    result.insert(-1, (tokNum, tokVal))
                else:
                    result.insert(-2, (tokNum, tokVal))
    else:
        result.extend(tokens)

    return result


def factorial_notation(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Allows standard notation for factorial."""
    result: list[TOKEN] = []
    nfactorial = 0
    for toknum, tokval in tokens:
        if toknum == OP and tokval == "!":
            # In Python 3.12 "!" are OP instead of ERRORTOKEN
            nfactorial += 1
        elif toknum == ERRORTOKEN:
            op = tokval
            if op == '!':
                nfactorial += 1
            else:
                nfactorial = 0
                result.append((OP, op))
        else:
            if nfactorial == 1:
                result = _add_factorial_tokens('factorial', result)
            elif nfactorial == 2:
                result = _add_factorial_tokens('factorial2', result)
            elif nfactorial > 2:
                raise TokenError
            nfactorial = 0
            result.append((toknum, tokval))
    return result


def convert_xor(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Treats XOR, ``^``, as exponentiation, ``**``."""
    result: list[TOKEN] = []
    for toknum, tokval in tokens:
        if toknum == OP:
            if tokval == '^':
                result.append((OP, '**'))
            else:
                result.append((toknum, tokval))
        else:
            result.append((toknum, tokval))

    return result


def repeated_decimals(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Allows 0.2[1] notation to represent the repeated decimal 0.2111... (19/90)

    Run this before auto_number.

    """
    result: list[TOKEN] = []

    def is_digit(s):
        return all(i in '0123456789_' for i in s)

    # num will running match any DECIMAL [ INTEGER ]
    num: list[TOKEN] = []
    for toknum, tokval in tokens:
        if toknum == NUMBER:
            if (not num and '.' in tokval and 'e' not in tokval.lower() and
                'j' not in tokval.lower()):
                num.append((toknum, tokval))
            elif is_digit(tokval) and (len(num) == 2 or
                    len(num) == 3 and is_digit(num[-1][1])):
                num.append((toknum, tokval))
            else:
                num = []
        elif toknum == OP:
            if tokval == '[' and len(num) == 1:
                num.append((OP, tokval))
            elif tokval == ']' and len(num) >= 3:
                num.append((OP, tokval))
            elif tokval == '.' and not num:
                # handle .[1]
                num.append((NUMBER, '0.'))
            else:
                num = []
        else:
            num = []

        result.append((toknum, tokval))

        if num and num[-1][1] == ']':
            # pre.post[repetend] = a + b/c + d/e where a = pre, b/c = post,
            # and d/e = repetend
            result = result[:-len(num)]
            pre, post = num[0][1].split('.')
            repetend = num[2][1]
            if len(num) == 5:
                repetend += num[3][1]

            pre = pre.replace('_', '')
            post = post.replace('_', '')
            repetend = repetend.replace('_', '')

            zeros = '0'*len(post)
            post, repetends = [w.lstrip('0') for w in [post, repetend]]
                                        # or else interpreted as octal

            a = pre or '0'
            b, c = post or '0', '1' + zeros
            d, e = repetends, ('9'*len(repetend)) + zeros

            seq = [
                (OP, '('),
                    (NAME, 'Integer'),
                    (OP, '('),
                        (NUMBER, a),
                    (OP, ')'),
                    (OP, '+'),
                    (NAME, 'Rational'),
                    (OP, '('),
                        (NUMBER, b),
                        (OP, ','),
                        (NUMBER, c),
                    (OP, ')'),
                    (OP, '+'),
                    (NAME, 'Rational'),
                    (OP, '('),
                        (NUMBER, d),
                        (OP, ','),
                        (NUMBER, e),
                    (OP, ')'),
                (OP, ')'),
            ]
            result.extend(seq)
            num = []

    return result


def auto_number(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """
    Converts numeric literals to use SymPy equivalents.

    Complex numbers use ``I``, integer literals use ``Integer``, and float
    literals use ``Float``.

    """
    result: list[TOKEN] = []

    for toknum, tokval in tokens:
        if toknum == NUMBER:
            number = tokval
            postfix = []

            if number.endswith(('j', 'J')):
                number = number[:-1]
                postfix = [(OP, '*'), (NAME, 'I')]

            if '.' in number or (('e' in number or 'E' in number) and
                    not (number.startswith(('0x', '0X')))):
                seq = [(NAME, 'Float'), (OP, '('),
                    (NUMBER, repr(str(number))), (OP, ')')]
            else:
                seq = [(NAME, 'Integer'), (OP, '('), (
                    NUMBER, number), (OP, ')')]

            result.extend(seq + postfix)
        else:
            result.append((toknum, tokval))

    return result


def rationalize(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Converts floats into ``Rational``. Run AFTER ``auto_number``."""
    result: list[TOKEN] = []
    passed_float = False
    for toknum, tokval in tokens:
        if toknum == NAME:
            if tokval == 'Float':
                passed_float = True
                tokval = 'Rational'
            result.append((toknum, tokval))
        elif passed_float == True and toknum == NUMBER:
            passed_float = False
            result.append((STRING, tokval))
        else:
            result.append((toknum, tokval))

    return result


def _transform_equals_sign(tokens: list[TOKEN], local_dict: DICT, global_dict: DICT):
    """Transforms the equals sign ``=`` to instances of Eq.

    This is a helper function for ``convert_equals_signs``.
    Works with expressions containing one equals sign and no
    nesting. Expressions like ``(1=2)=False`` will not work with this
    and should be used with ``convert_equals_signs``.

    Examples: 1=2     to Eq(1,2)
              1*2=x   to Eq(1*2, x)

    This does not deal with function arguments yet.

    """
    result: list[TOKEN] = []
    if (OP, "=") in tokens:
        result.append((NAME, "Eq"))
        result.append((OP, "("))
        for token in tokens:
            if token == (OP, "="):
                result.append((OP, ","))
                continue
            result.append(token)
        result.append((OP, ")"))
    else:
        result = tokens
    return result


def convert_equals_signs(tokens: list[TOKEN], local_dict: DICT,
                         global_dict: DICT) -> list[TOKEN]:
    """ Transforms all the equals signs ``=`` to instances of Eq.

    Parses the equals signs in the expression and replaces them with
    appropriate Eq instances. Also works with nested equals signs.

    Does not yet play well with function arguments.
    For example, the expression ``(x=y)`` is ambiguous and can be interpreted
    as x being an argument to a function and ``convert_equals_signs`` will not
    work for this.

    See also
    ========
    convert_equality_operators

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import (parse_expr,
    ... standard_transformations, convert_equals_signs)
    >>> parse_expr("1*2=x", transformations=(
    ... standard_transformations + (convert_equals_signs,)))
    Eq(2, x)
    >>> parse_expr("(1*2=x)=False", transformations=(
    ... standard_transformations + (convert_equals_signs,)))
    Eq(Eq(2, x), False)

    """
    res1 = _group_parentheses(convert_equals_signs)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _transform_equals_sign(res2, local_dict, global_dict)
    result = _flatten(res3)
    return result


#: Standard transformations for :func:`parse_expr`.
#: Inserts calls to :class:`~.Symbol`, :class:`~.Integer`, and other SymPy
#: datatypes and allows the use of standard factorial notation (e.g. ``x!``).
standard_transformations: tuple[TRANS, ...] \
    = (lambda_notation, auto_symbol, repeated_decimals, auto_number,
       factorial_notation)


def stringify_expr(s: str, local_dict: DICT, global_dict: DICT,
        transformations: tuple[TRANS, ...]) -> str:
    """
    Converts the string ``s`` to Python code, in ``local_dict``

    Generally, ``parse_expr`` should be used.
    """

    tokens = []
    input_code = StringIO(s.strip())
    for toknum, tokval, _, _, _ in generate_tokens(input_code.readline):
        tokens.append((toknum, tokval))

    for transform in transformations:
        tokens = transform(tokens, local_dict, global_dict)

    return untokenize(tokens)


def eval_expr(code, local_dict: DICT, global_dict: DICT):
    """
    Evaluate Python code generated by ``stringify_expr``.

    Generally, ``parse_expr`` should be used.
    """
    expr = eval(
        code, global_dict, local_dict)  # take local objects in preference
    return expr


def parse_expr(s: str, local_dict: DICT | None = None,
               transformations: tuple[TRANS, ...] | str \
                   = standard_transformations,
               global_dict: DICT | None = None, evaluate=True):
    """Converts the string ``s`` to a SymPy expression, in ``local_dict``.

    .. warning::
        Note that this function uses ``eval``, and thus shouldn't be used on
        unsanitized input.

    Parameters
    ==========

    s : str
        The string to parse.

    local_dict : dict, optional
        A dictionary of local variables to use when parsing.

    global_dict : dict, optional
        A dictionary of global variables. By default, this is initialized
        with ``from sympy import *``; provide this parameter to override
        this behavior (for instance, to parse ``"Q & S"``).

    transformations : tuple or str
        A tuple of transformation functions used to modify the tokens of the
        parsed expression before evaluation. The default transformations
        convert numeric literals into their SymPy equivalents, convert
        undefined variables into SymPy symbols, and allow the use of standard
        mathematical factorial notation (e.g. ``x!``). Selection via
        string is available (see below).

    evaluate : bool, optional
        When False, the order of the arguments will remain as they were in the
        string and automatic simplification that would normally occur is
        suppressed. (see examples)

    Examples
    ========

    >>> from sympy.parsing.sympy_parser import parse_expr
    >>> parse_expr("1/2")
    1/2
    >>> type(_)
    <class 'sympy.core.numbers.Half'>
    >>> from sympy.parsing.sympy_parser import standard_transformations,\\
    ... implicit_multiplication_application
    >>> transformations = (standard_transformations +
    ...     (implicit_multiplication_application,))
    >>> parse_expr("2x", transformations=transformations)
    2*x

    When evaluate=False, some automatic simplifications will not occur:

    >>> parse_expr("2**3"), parse_expr("2**3", evaluate=False)
    (8, 2**3)

    In addition the order of the arguments will not be made canonical.
    This feature allows one to tell exactly how the expression was entered:

    >>> a = parse_expr('1 + x', evaluate=False)
    >>> b = parse_expr('x + 1', evaluate=False)
    >>> a == b
    False
    >>> a.args
    (1, x)
    >>> b.args
    (x, 1)

    Note, however, that when these expressions are printed they will
    appear the same:

    >>> assert str(a) == str(b)

    As a convenience, transformations can be seen by printing ``transformations``:

    >>> from sympy.parsing.sympy_parser import transformations

    >>> print(transformations)
    0: lambda_notation
    1: auto_symbol
    2: repeated_decimals
    3: auto_number
    4: factorial_notation
    5: implicit_multiplication_application
    6: convert_xor
    7: implicit_application
    8: implicit_multiplication
    9: convert_equals_signs
    10: function_exponentiation
    11: rationalize

    The ``T`` object provides a way to select these transformations:

    >>> from sympy.parsing.sympy_parser import T

    If you print it, you will see the same list as shown above.

    >>> str(T) == str(transformations)
    True

    Standard slicing will return a tuple of transformations:

    >>> T[:5] == standard_transformations
    True

    So ``T`` can be used to specify the parsing transformations:

    >>> parse_expr("2x", transformations=T[:5])
    Traceback (most recent call last):
    ...
    SyntaxError: invalid syntax
    >>> parse_expr("2x", transformations=T[:6])
    2*x
    >>> parse_expr('.3', transformations=T[3, 11])
    3/10
    >>> parse_expr('.3x', transformations=T[:])
    3*x/10

    As a further convenience, strings 'implicit' and 'all' can be used
    to select 0-5 and all the transformations, respectively.

    >>> parse_expr('.3x', transformations='all')
    3*x/10

    See Also
    ========

    stringify_expr, eval_expr, standard_transformations,
    implicit_multiplication_application

    """

    if local_dict is None:
        local_dict = {}
    elif not isinstance(local_dict, dict):
        raise TypeError('expecting local_dict to be a dict')
    elif null in local_dict:
        raise ValueError('cannot use "" in local_dict')

    if global_dict is None:
        global_dict = {}
        exec('from sympy import *', global_dict)

        builtins_dict = vars(builtins)
        for name, obj in builtins_dict.items():
            if isinstance(obj, types.BuiltinFunctionType):
                global_dict[name] = obj
        global_dict['max'] = Max
        global_dict['min'] = Min

    elif not isinstance(global_dict, dict):
        raise TypeError('expecting global_dict to be a dict')

    transformations = transformations or ()
    if isinstance(transformations, str):
        if transformations == 'all':
            _transformations = T[:]
        elif transformations == 'implicit':
            _transformations = T[:6]
        else:
            raise ValueError('unknown transformation group name')
    else:
        _transformations = transformations

    code = stringify_expr(s, local_dict, global_dict, _transformations)

    if not evaluate:
        code = compile(evaluateFalse(code), '<string>', 'eval') # type: ignore

    try:
        rv = eval_expr(code, local_dict, global_dict)
        # restore neutral definitions for names
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        return rv
    except Exception as e:
        # restore neutral definitions for names
        for i in local_dict.pop(null, ()):
            local_dict[i] = null
        raise e from ValueError(f"Error from parse_expr with transformed code: {code!r}")


def evaluateFalse(s: str):
    """
    Replaces operators with the SymPy equivalent and sets evaluate=False.
    """
    node = ast.parse(s)
    transformed_node = EvaluateFalseTransformer().visit(node)
    # node is a Module, we want an Expression
    transformed_node = ast.Expression(transformed_node.body[0].value)

    return ast.fix_missing_locations(transformed_node)


class EvaluateFalseTransformer(ast.NodeTransformer):
    operators = {
        ast.Add: 'Add',
        ast.Mult: 'Mul',
        ast.Pow: 'Pow',
        ast.Sub: 'Add',
        ast.Div: 'Mul',
        ast.BitOr: 'Or',
        ast.BitAnd: 'And',
        ast.BitXor: 'Not',
    }
    functions = (
        'Abs', 'im', 're', 'sign', 'arg', 'conjugate',
        'acos', 'acot', 'acsc', 'asec', 'asin', 'atan',
        'acosh', 'acoth', 'acsch', 'asech', 'asinh', 'atanh',
        'cos', 'cot', 'csc', 'sec', 'sin', 'tan',
        'cosh', 'coth', 'csch', 'sech', 'sinh', 'tanh',
        'exp', 'ln', 'log', 'sqrt', 'cbrt',
    )

    relational_operators = {
        ast.NotEq: 'Ne',
        ast.Lt: 'Lt',
        ast.LtE: 'Le',
        ast.Gt: 'Gt',
        ast.GtE: 'Ge',
        ast.Eq: 'Eq'
    }
    def visit_Compare(self, node):
        def reducer(acc, op_right):
            result, left = acc
            op, right = op_right
            if op.__class__ not in self.relational_operators:
                raise ValueError("Only equation or inequality operators are supported")
            new = ast.Call(
                func=ast.Name(
                    id=self.relational_operators[op.__class__], ctx=ast.Load()
                ),
                args=[self.visit(left), self.visit(right)],
                keywords=[ast.keyword(arg="evaluate", value=ast.Constant(value=False))],
            )
            return result + [new], right

        args, _ = reduce(
            reducer, zip(node.ops, node.comparators), ([], node.left)
        )
        if len(args) == 1:
            return args[0]
        return ast.Call(
            func=ast.Name(id=self.operators[ast.BitAnd], ctx=ast.Load()),
            args=args,
            keywords=[ast.keyword(arg="evaluate", value=ast.Constant(value=False))],
        )

    def flatten(self, args, func):
        result = []
        for arg in args:
            if isinstance(arg, ast.Call):
                arg_func = arg.func
                if isinstance(arg_func, ast.Call):
                    arg_func = arg_func.func
                if arg_func.id == func:
                    result.extend(self.flatten(arg.args, func))
                else:
                    result.append(arg)
            else:
                result.append(arg)
        return result

    def visit_BinOp(self, node):
        if node.op.__class__ in self.operators:
            sympy_class = self.operators[node.op.__class__]
            right = self.visit(node.right)
            left = self.visit(node.left)

            rev = False
            if isinstance(node.op, ast.Sub):
                right = ast.Call(
                    func=ast.Name(id='Mul', ctx=ast.Load()),
                    args=[ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1)), right],
                    keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                )
            elif isinstance(node.op, ast.Div):
                if isinstance(node.left, ast.UnaryOp):
                    left, right = right, left
                    rev = True
                    left = ast.Call(
                    func=ast.Name(id='Pow', ctx=ast.Load()),
                    args=[left, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))],
                    keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                )
                else:
                    right = ast.Call(
                    func=ast.Name(id='Pow', ctx=ast.Load()),
                    args=[right, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1))],
                    keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
                )

            if rev:  # undo reversal
                left, right = right, left
            new_node = ast.Call(
                func=ast.Name(id=sympy_class, ctx=ast.Load()),
                args=[left, right],
                keywords=[ast.keyword(arg='evaluate', value=ast.Constant(value=False))]
            )

            if sympy_class in ('Add', 'Mul'):
                # Denest Add or Mul as appropriate
                new_node.args = self.flatten(new_node.args, sympy_class)

            return new_node
        return node

    def visit_Call(self, node):
        new_node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            new_node.keywords.append(ast.keyword(arg='evaluate', value=ast.Constant(value=False)))
        return new_node


_transformation = {  # items can be added but never re-ordered
0: lambda_notation,
1: auto_symbol,
2: repeated_decimals,
3: auto_number,
4: factorial_notation,
5: implicit_multiplication_application,
6: convert_xor,
7: implicit_application,
8: implicit_multiplication,
9: convert_equals_signs,
10: function_exponentiation,
11: rationalize}

transformations = '\n'.join('%s: %s' % (i, func_name(f)) for i, f in _transformation.items())


class _T():
    """class to retrieve transformations from a given slice

    EXAMPLES
    ========

    >>> from sympy.parsing.sympy_parser import T, standard_transformations
    >>> assert T[:5] == standard_transformations
    """
    def __init__(self):
        self.N = len(_transformation)

    def __str__(self):
        return transformations

    def __getitem__(self, t):
        if not type(t) is tuple:
            t = (t,)
        i = []
        for ti in t:
            if type(ti) is int:
                i.append(range(self.N)[ti])
            elif type(ti) is slice:
                i.extend(range(*ti.indices(self.N)))
            else:
                raise TypeError('unexpected slice arg')
        return tuple([_transformation[_] for _ in i])

T = _T()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\common.py ===
"""Common IO api utilities"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import codecs
from collections import defaultdict
from collections.abc import (
    Hashable,
    Mapping,
    Sequence,
)
import dataclasses
import functools
import gzip
from io import (
    BufferedIOBase,
    BytesIO,
    RawIOBase,
    StringIO,
    TextIOBase,
    TextIOWrapper,
)
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    DefaultDict,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)
from urllib.parse import (
    urljoin,
    urlparse as parse_url,
    uses_netloc,
    uses_params,
    uses_relative,
)
import warnings
import zipfile

from pandas._typing import (
    BaseBuffer,
    ReadCsvBuffer,
)
from pandas.compat import (
    get_bz2_file,
    get_lzma_file,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool,
    is_file_like,
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import ABCMultiIndex

from pandas.core.shared_docs import _shared_docs

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")
_RFC_3986_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+\-+.]*://")

BaseBufferT = TypeVar("BaseBufferT", bound=BaseBuffer)


if TYPE_CHECKING:
    from types import TracebackType

    from pandas._typing import (
        CompressionDict,
        CompressionOptions,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import MultiIndex


@dataclasses.dataclass
class IOArgs:
    """
    Return value of io/common.py:_get_filepath_or_buffer.
    """

    filepath_or_buffer: str | BaseBuffer
    encoding: str
    mode: str
    compression: CompressionDict
    should_close: bool = False


@dataclasses.dataclass
class IOHandles(Generic[AnyStr]):
    """
    Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    handle: The file handle to be used.
    created_handles: All file handles that are created by get_handle
    is_wrapped: Whether a TextIOWrapper needs to be detached.
    """

    # handle might not implement the IO-interface
    handle: IO[AnyStr]
    compression: CompressionDict
    created_handles: list[IO[bytes] | IO[str]] = dataclasses.field(default_factory=list)
    is_wrapped: bool = False

    def close(self) -> None:
        """
        Close all created buffers.

        Note: If a TextIOWrapper was inserted, it is flushed and detached to
        avoid closing the potentially user-created buffer.
        """
        if self.is_wrapped:
            assert isinstance(self.handle, TextIOWrapper)
            self.handle.flush()
            self.handle.detach()
            self.created_handles.remove(self.handle)
        for handle in self.created_handles:
            handle.close()
        self.created_handles = []
        self.is_wrapped = False

    def __enter__(self) -> IOHandles[AnyStr]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def is_url(url: object) -> bool:
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
    if not isinstance(url, str):
        return False
    return parse_url(url).scheme in _VALID_URLS


@overload
def _expand_user(filepath_or_buffer: str) -> str:
    ...


@overload
def _expand_user(filepath_or_buffer: BaseBufferT) -> BaseBufferT:
    ...


def _expand_user(filepath_or_buffer: str | BaseBufferT) -> str | BaseBufferT:
    """
    Return the argument with an initial component of ~ or ~user
    replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : object to be converted if possible

    Returns
    -------
    expanded_filepath_or_buffer : an expanded filepath or the
                                  input if not expandable
    """
    if isinstance(filepath_or_buffer, str):
        return os.path.expanduser(filepath_or_buffer)
    return filepath_or_buffer


def validate_header_arg(header: object) -> None:
    if header is None:
        return
    if is_integer(header):
        header = cast(int, header)
        if header < 0:
            # GH 27779
            raise ValueError(
                "Passing negative integer to header is invalid. "
                "For no header, use header=None instead"
            )
        return
    if is_list_like(header, allow_sets=False):
        header = cast(Sequence, header)
        if not all(map(is_integer, header)):
            raise ValueError("header must be integer or list of integers")
        if any(i < 0 for i in header):
            raise ValueError("cannot specify multi-index header with negative integers")
        return
    if is_bool(header):
        raise TypeError(
            "Passing a bool to header is invalid. Use header=None for no header or "
            "header=int or list-like of ints to specify "
            "the row(s) making up the column names"
        )
    # GH 16338
    raise ValueError("header must be integer or list of integers")


@overload
def stringify_path(filepath_or_buffer: FilePath, convert_file_like: bool = ...) -> str:
    ...


@overload
def stringify_path(
    filepath_or_buffer: BaseBufferT, convert_file_like: bool = ...
) -> BaseBufferT:
    ...


def stringify_path(
    filepath_or_buffer: FilePath | BaseBufferT,
    convert_file_like: bool = False,
) -> str | BaseBufferT:
    """
    Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object to be converted

    Returns
    -------
    str_filepath_or_buffer : maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    if not convert_file_like and is_file_like(filepath_or_buffer):
        # GH 38125: some fsspec objects implement os.PathLike but have already opened a
        # file. This prevents opening the file a second time. infer_compression calls
        # this function with convert_file_like=True to infer the compression.
        return cast(BaseBufferT, filepath_or_buffer)

    if isinstance(filepath_or_buffer, os.PathLike):
        filepath_or_buffer = filepath_or_buffer.__fspath__()
    return _expand_user(filepath_or_buffer)


def urlopen(*args, **kwargs):
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
    import urllib.request

    return urllib.request.urlopen(*args, **kwargs)


def is_fsspec_url(url: FilePath | BaseBuffer) -> bool:
    """
    Returns true if the given URL looks like
    something fsspec can handle
    """
    return (
        isinstance(url, str)
        and bool(_RFC_3986_PATTERN.match(url))
        and not url.startswith(("http://", "https://"))
    )


@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "filepath_or_buffer",
)
def _get_filepath_or_buffer(
    filepath_or_buffer: FilePath | BaseBuffer,
    encoding: str = "utf-8",
    compression: CompressionOptions | None = None,
    mode: str = "r",
    storage_options: StorageOptions | None = None,
) -> IOArgs:
    """
    If the filepath_or_buffer is a url, translate and return the buffer.
    Otherwise passthrough.

    Parameters
    ----------
    filepath_or_buffer : a url, filepath (str, py.path.local or pathlib.Path),
                         or buffer
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    encoding : the encoding to use to decode bytes, default is 'utf-8'
    mode : str, optional

    {storage_options}


    Returns the dataclass IOArgs.
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)

    # handle compression dict
    compression_method, compression = get_compression_method(compression)
    compression_method = infer_compression(filepath_or_buffer, compression_method)

    # GH21227 internal compression is not used for non-binary handles.
    if compression_method and hasattr(filepath_or_buffer, "write") and "b" not in mode:
        warnings.warn(
            "compression has no effect when passing a non-binary object as input.",
            RuntimeWarning,
            stacklevel=find_stack_level(),
        )
        compression_method = None

    compression = dict(compression, method=compression_method)

    # bz2 and xz do not write the byte order mark for utf-16 and utf-32
    # print a warning when writing such files
    if (
        "w" in mode
        and compression_method in ["bz2", "xz"]
        and encoding in ["utf-16", "utf-32"]
    ):
        warnings.warn(
            f"{compression} will not write the byte order mark for {encoding}",
            UnicodeWarning,
            stacklevel=find_stack_level(),
        )

    # Use binary mode when converting path-like objects to file-like objects (fsspec)
    # except when text mode is explicitly requested. The original mode is returned if
    # fsspec is not used.
    fsspec_mode = mode
    if "t" not in fsspec_mode and "b" not in fsspec_mode:
        fsspec_mode += "b"

    if isinstance(filepath_or_buffer, str) and is_url(filepath_or_buffer):
        # TODO: fsspec can also handle HTTP via requests, but leaving this
        # unchanged. using fsspec appears to break the ability to infer if the
        # server responded with gzipped data
        storage_options = storage_options or {}

        # waiting until now for importing to match intended lazy logic of
        # urlopen function defined elsewhere in this module
        import urllib.request

        # assuming storage_options is to be interpreted as headers
        req_info = urllib.request.Request(filepath_or_buffer, headers=storage_options)
        with urlopen(req_info) as req:
            content_encoding = req.headers.get("Content-Encoding", None)
            if content_encoding == "gzip":
                # Override compression based on Content-Encoding header
                compression = {"method": "gzip"}
            reader = BytesIO(req.read())
        return IOArgs(
            filepath_or_buffer=reader,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )

    if is_fsspec_url(filepath_or_buffer):
        assert isinstance(
            filepath_or_buffer, str
        )  # just to appease mypy for this branch
        # two special-case s3-like protocols; these have special meaning in Hadoop,
        # but are equivalent to just "s3" from fsspec's point of view
        # cc #11071
        if filepath_or_buffer.startswith("s3a://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3a://", "s3://")
        if filepath_or_buffer.startswith("s3n://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3n://", "s3://")
        fsspec = import_optional_dependency("fsspec")

        # If botocore is installed we fallback to reading with anon=True
        # to allow reads from public buckets
        err_types_to_retry_with_anon: list[Any] = []
        try:
            import_optional_dependency("botocore")
            from botocore.exceptions import (
                ClientError,
                NoCredentialsError,
            )

            err_types_to_retry_with_anon = [
                ClientError,
                NoCredentialsError,
                PermissionError,
            ]
        except ImportError:
            pass

        try:
            file_obj = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            ).open()
        # GH 34626 Reads from Public Buckets without Credentials needs anon=True
        except tuple(err_types_to_retry_with_anon):
            if storage_options is None:
                storage_options = {"anon": True}
            else:
                # don't mutate user input.
                storage_options = dict(storage_options)
                storage_options["anon"] = True
            file_obj = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            ).open()

        return IOArgs(
            filepath_or_buffer=file_obj,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )
    elif storage_options:
        raise ValueError(
            "storage_options passed with file object or non-fsspec file path"
        )

    if isinstance(filepath_or_buffer, (str, bytes, mmap.mmap)):
        return IOArgs(
            filepath_or_buffer=_expand_user(filepath_or_buffer),
            encoding=encoding,
            compression=compression,
            should_close=False,
            mode=mode,
        )

    # is_file_like requires (read | write) & __iter__ but __iter__ is only
    # needed for read_csv(engine=python)
    if not (
        hasattr(filepath_or_buffer, "read") or hasattr(filepath_or_buffer, "write")
    ):
        msg = f"Invalid file path or buffer object type: {type(filepath_or_buffer)}"
        raise ValueError(msg)

    return IOArgs(
        filepath_or_buffer=filepath_or_buffer,
        encoding=encoding,
        compression=compression,
        should_close=False,
        mode=mode,
    )


def file_path_to_url(path: str) -> str:
    """
    converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """
    # lazify expensive import (~30ms)
    from urllib.request import pathname2url

    return urljoin("file:", pathname2url(path))


extension_to_compression = {
    ".tar": "tar",
    ".tar.gz": "tar",
    ".tar.bz2": "tar",
    ".tar.xz": "tar",
    ".gz": "gzip",
    ".bz2": "bz2",
    ".zip": "zip",
    ".xz": "xz",
    ".zst": "zstd",
}
_supported_compressions = set(extension_to_compression.values())


def get_compression_method(
    compression: CompressionOptions,
) -> tuple[str | None, CompressionDict]:
    """
    Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of ({compression method}, Optional[str]
              {compression arguments}, Dict[str, Any])

    Raises
    ------
    ValueError on mapping missing 'method' key
    """
    compression_method: str | None
    if isinstance(compression, Mapping):
        compression_args = dict(compression)
        try:
            compression_method = compression_args.pop("method")
        except KeyError as err:
            raise ValueError("If mapping, compression must have key 'method'") from err
    else:
        compression_args = {}
        compression_method = compression
    return compression_method, compression_args


@doc(compression_options=_shared_docs["compression_options"] % "filepath_or_buffer")
def infer_compression(
    filepath_or_buffer: FilePath | BaseBuffer, compression: str | None
) -> str | None:
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : str or file handle
        File path or object.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    Returns
    -------
    string or None

    Raises
    ------
    ValueError on invalid compression specified.
    """
    if compression is None:
        return None

    # Infer compression
    if compression == "infer":
        # Convert all path types (e.g. pathlib.Path) to strings
        filepath_or_buffer = stringify_path(filepath_or_buffer, convert_file_like=True)
        if not isinstance(filepath_or_buffer, str):
            # Cannot infer compression of a buffer, assume no compression
            return None

        # Infer compression from the filename/URL extension
        for extension, compression in extension_to_compression.items():
            if filepath_or_buffer.lower().endswith(extension):
                return compression
        return None

    # Compression has been specified. Check that it's valid
    if compression in _supported_compressions:
        return compression

    valid = ["infer", None] + sorted(_supported_compressions)
    msg = (
        f"Unrecognized compression type: {compression}\n"
        f"Valid compression types are {valid}"
    )
    raise ValueError(msg)


def check_parent_directory(path: Path | str) -> None:
    """
    Check if parent directory of a file exists, raise OSError if it does not

    Parameters
    ----------
    path: Path or str
        Path to check parent directory of
    """
    parent = Path(path).parent
    if not parent.is_dir():
        raise OSError(rf"Cannot save file into a non-existent directory: '{parent}'")


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[False],
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[bytes]:
    ...


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[True] = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str]:
    ...


@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: bool = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str] | IOHandles[bytes]:
    ...


@doc(compression_options=_shared_docs["compression_options"] % "path_or_buf")
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = None,
    compression: CompressionOptions | None = None,
    memory_map: bool = False,
    is_text: bool = True,
    errors: str | None = None,
    storage_options: StorageOptions | None = None,
) -> IOHandles[str] | IOHandles[bytes]:
    """
    Get file handle for given path/buffer and mode.

    Parameters
    ----------
    path_or_buf : str or file handle
        File path or object.
    mode : str
        Mode to open path_or_buf with.
    encoding : str or None
        Encoding to use.
    {compression_options}

           May be a dict with key 'method' as compression mode
           and other keys as compression options if compression
           mode is 'zip'.

           Passing compression options as keys in dict is
           supported for compression modes 'gzip', 'bz2', 'zstd' and 'zip'.

        .. versionchanged:: 1.4.0 Zstandard support.

    memory_map : bool, default False
        See parsers._parser_params for more information. Only used by read_csv.
    is_text : bool, default True
        Whether the type of the content passed to the file/buffer is string or
        bytes. This is not the same as `"b" not in mode`. If a string content is
        passed to a binary file/buffer, a wrapper is inserted.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    storage_options: StorageOptions = None
        Passed to _get_filepath_or_buffer

    Returns the dataclass IOHandles
    """
    # Windows does not default to utf-8. Set to utf-8 for a consistent behavior
    encoding = encoding or "utf-8"

    errors = errors or "strict"

    # read_csv does not know whether the buffer is opened in binary/text mode
    if _is_binary_mode(path_or_buf, mode) and "b" not in mode:
        mode += "b"

    # validate encoding and errors
    codecs.lookup(encoding)
    if isinstance(errors, str):
        codecs.lookup_error(errors)

    # open URLs
    ioargs = _get_filepath_or_buffer(
        path_or_buf,
        encoding=encoding,
        compression=compression,
        mode=mode,
        storage_options=storage_options,
    )

    handle = ioargs.filepath_or_buffer
    handles: list[BaseBuffer]

    # memory mapping needs to be the first step
    # only used for read_csv
    handle, memory_map, handles = _maybe_memory_map(handle, memory_map)

    is_path = isinstance(handle, str)
    compression_args = dict(ioargs.compression)
    compression = compression_args.pop("method")

    # Only for write methods
    if "r" not in mode and is_path:
        check_parent_directory(str(handle))

    if compression:
        if compression != "zstd":
            # compression libraries do not like an explicit text-mode
            ioargs.mode = ioargs.mode.replace("t", "")
        elif compression == "zstd" and "b" not in ioargs.mode:
            # python-zstandard defaults to text mode, but we always expect
            # compression libraries to use binary mode.
            ioargs.mode += "b"

        # GZ Compression
        if compression == "gzip":
            if isinstance(handle, str):
                # error: Incompatible types in assignment (expression has type
                # "GzipFile", variable has type "Union[str, BaseBuffer]")
                handle = gzip.GzipFile(  # type: ignore[assignment]
                    filename=handle,
                    mode=ioargs.mode,
                    **compression_args,
                )
            else:
                handle = gzip.GzipFile(
                    # No overload variant of "GzipFile" matches argument types
                    # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
                    fileobj=handle,  # type: ignore[call-overload]
                    mode=ioargs.mode,
                    **compression_args,
                )

        # BZ Compression
        elif compression == "bz2":
            # Overload of "BZ2File" to handle pickle protocol 5
            # "Union[str, BaseBuffer]", "str", "Dict[str, Any]"
            handle = get_bz2_file()(  # type: ignore[call-overload]
                handle,
                mode=ioargs.mode,
                **compression_args,
            )

        # ZIP Compression
        elif compression == "zip":
            # error: Argument 1 to "_BytesZipFile" has incompatible type
            # "Union[str, BaseBuffer]"; expected "Union[Union[str, PathLike[str]],
            # ReadBuffer[bytes], WriteBuffer[bytes]]"
            handle = _BytesZipFile(
                handle, ioargs.mode, **compression_args  # type: ignore[arg-type]
            )
            if handle.buffer.mode == "r":
                handles.append(handle)
                zip_names = handle.buffer.namelist()
                if len(zip_names) == 1:
                    handle = handle.buffer.open(zip_names.pop())
                elif not zip_names:
                    raise ValueError(f"Zero files found in ZIP file {path_or_buf}")
                else:
                    raise ValueError(
                        "Multiple files found in ZIP file. "
                        f"Only one file per ZIP: {zip_names}"
                    )

        # TAR Encoding
        elif compression == "tar":
            compression_args.setdefault("mode", ioargs.mode)
            if isinstance(handle, str):
                handle = _BytesTarFile(name=handle, **compression_args)
            else:
                # error: Argument "fileobj" to "_BytesTarFile" has incompatible
                # type "BaseBuffer"; expected "Union[ReadBuffer[bytes],
                # WriteBuffer[bytes], None]"
                handle = _BytesTarFile(
                    fileobj=handle, **compression_args  # type: ignore[arg-type]
                )
            assert isinstance(handle, _BytesTarFile)
            if "r" in handle.buffer.mode:
                handles.append(handle)
                files = handle.buffer.getnames()
                if len(files) == 1:
                    file = handle.buffer.extractfile(files[0])
                    assert file is not None
                    handle = file
                elif not files:
                    raise ValueError(f"Zero files found in TAR archive {path_or_buf}")
                else:
                    raise ValueError(
                        "Multiple files found in TAR archive. "
                        f"Only one file per TAR archive: {files}"
                    )

        # XZ Compression
        elif compression == "xz":
            # error: Argument 1 to "LZMAFile" has incompatible type "Union[str,
            # BaseBuffer]"; expected "Optional[Union[Union[str, bytes, PathLike[str],
            # PathLike[bytes]], IO[bytes]], None]"
            handle = get_lzma_file()(
                handle, ioargs.mode, **compression_args  # type: ignore[arg-type]
            )

        # Zstd Compression
        elif compression == "zstd":
            zstd = import_optional_dependency("zstandard")
            if "r" in ioargs.mode:
                open_args = {"dctx": zstd.ZstdDecompressor(**compression_args)}
            else:
                open_args = {"cctx": zstd.ZstdCompressor(**compression_args)}
            handle = zstd.open(
                handle,
                mode=ioargs.mode,
                **open_args,
            )

        # Unrecognized Compression
        else:
            msg = f"Unrecognized compression type: {compression}"
            raise ValueError(msg)

        assert not isinstance(handle, str)
        handles.append(handle)

    elif isinstance(handle, str):
        # Check whether the filename is to be opened in binary mode.
        # Binary mode does not support 'encoding' and 'newline'.
        if ioargs.encoding and "b" not in ioargs.mode:
            # Encoding
            handle = open(
                handle,
                ioargs.mode,
                encoding=ioargs.encoding,
                errors=errors,
                newline="",
            )
        else:
            # Binary mode
            handle = open(handle, ioargs.mode)
        handles.append(handle)

    # Convert BytesIO or file objects passed with an encoding
    is_wrapped = False
    if not is_text and ioargs.mode == "rb" and isinstance(handle, TextIOBase):
        # not added to handles as it does not open/buffer resources
        handle = _BytesIOWrapper(
            handle,
            encoding=ioargs.encoding,
        )
    elif is_text and (
        compression or memory_map or _is_binary_mode(handle, ioargs.mode)
    ):
        if (
            not hasattr(handle, "readable")
            or not hasattr(handle, "writable")
            or not hasattr(handle, "seekable")
        ):
            handle = _IOWrapper(handle)
        # error: Argument 1 to "TextIOWrapper" has incompatible type
        # "_IOWrapper"; expected "IO[bytes]"
        handle = TextIOWrapper(
            handle,  # type: ignore[arg-type]
            encoding=ioargs.encoding,
            errors=errors,
            newline="",
        )
        handles.append(handle)
        # only marked as wrapped when the caller provided a handle
        is_wrapped = not (
            isinstance(ioargs.filepath_or_buffer, str) or ioargs.should_close
        )

    if "r" in ioargs.mode and not hasattr(handle, "read"):
        raise TypeError(
            "Expected file path name or file-like object, "
            f"got {type(ioargs.filepath_or_buffer)} type"
        )

    handles.reverse()  # close the most recently added buffer first
    if ioargs.should_close:
        assert not isinstance(ioargs.filepath_or_buffer, str)
        handles.append(ioargs.filepath_or_buffer)

    return IOHandles(
        # error: Argument "handle" to "IOHandles" has incompatible type
        # "Union[TextIOWrapper, GzipFile, BaseBuffer, typing.IO[bytes],
        # typing.IO[Any]]"; expected "pandas._typing.IO[Any]"
        handle=handle,  # type: ignore[arg-type]
        # error: Argument "created_handles" to "IOHandles" has incompatible type
        # "List[BaseBuffer]"; expected "List[Union[IO[bytes], IO[str]]]"
        created_handles=handles,  # type: ignore[arg-type]
        is_wrapped=is_wrapped,
        compression=ioargs.compression,
    )


# error: Definition of "__enter__" in base class "IOBase" is incompatible
# with definition in base class "BinaryIO"
class _BufferedWriter(BytesIO, ABC):  # type: ignore[misc]
    """
    Some objects do not support multiple .write() calls (TarFile and ZipFile).
    This wrapper writes to the underlying buffer on close.
    """

    buffer = BytesIO()

    @abstractmethod
    def write_to_buffer(self) -> None:
        ...

    def close(self) -> None:
        if self.closed:
            # already closed
            return
        if self.getbuffer().nbytes:
            # write to buffer
            self.seek(0)
            with self.buffer:
                self.write_to_buffer()
        else:
            self.buffer.close()
        super().close()


class _BytesTarFile(_BufferedWriter):
    def __init__(
        self,
        name: str | None = None,
        mode: Literal["r", "a", "w", "x"] = "r",
        fileobj: ReadBuffer[bytes] | WriteBuffer[bytes] | None = None,
        archive_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.archive_name = archive_name
        self.name = name
        # error: Incompatible types in assignment (expression has type "TarFile",
        # base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: tarfile.TarFile = tarfile.TarFile.open(  # type: ignore[assignment]
            name=name,
            mode=self.extend_mode(mode),
            fileobj=fileobj,
            **kwargs,
        )

    def extend_mode(self, mode: str) -> str:
        mode = mode.replace("b", "")
        if mode != "w":
            return mode
        if self.name is not None:
            suffix = Path(self.name).suffix
            if suffix in (".gz", ".xz", ".bz2"):
                mode = f"{mode}:{suffix[1:]}"
        return mode

    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.tar, because that causes confusion (GH39465).
        """
        if self.name is None:
            return None

        filename = Path(self.name)
        if filename.suffix == ".tar":
            return filename.with_suffix("").name
        elif filename.suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
            return filename.with_suffix("").with_suffix("").name
        return filename.name

    def write_to_buffer(self) -> None:
        # TarFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "tar"
        tarinfo = tarfile.TarInfo(name=archive_name)
        tarinfo.size = len(self.getvalue())
        self.buffer.addfile(tarinfo, self)


class _BytesZipFile(_BufferedWriter):
    def __init__(
        self,
        file: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
        mode: str,
        archive_name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        mode = mode.replace("b", "")
        self.archive_name = archive_name

        kwargs.setdefault("compression", zipfile.ZIP_DEFLATED)
        # error: Incompatible types in assignment (expression has type "ZipFile",
        # base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[assignment]
            file, mode, **kwargs
        )

    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.zip, because that causes confusion (GH39465).
        """
        if isinstance(self.buffer.filename, (os.PathLike, str)):
            filename = Path(self.buffer.filename)
            if filename.suffix == ".zip":
                return filename.with_suffix("").name
            return filename.name
        return None

    def write_to_buffer(self) -> None:
        # ZipFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "zip"
        self.buffer.writestr(archive_name, self.getvalue())


class _IOWrapper:
    # TextIOWrapper is overly strict: it request that the buffer has seekable, readable,
    # and writable. If we have a read-only buffer, we shouldn't need writable and vice
    # versa. Some buffers, are seek/read/writ-able but they do not have the "-able"
    # methods, e.g., tempfile.SpooledTemporaryFile.
    # If a buffer does not have the above "-able" methods, we simple assume they are
    # seek/read/writ-able.
    def __init__(self, buffer: BaseBuffer) -> None:
        self.buffer = buffer

    def __getattr__(self, name: str):
        return getattr(self.buffer, name)

    def readable(self) -> bool:
        if hasattr(self.buffer, "readable"):
            return self.buffer.readable()
        return True

    def seekable(self) -> bool:
        if hasattr(self.buffer, "seekable"):
            return self.buffer.seekable()
        return True

    def writable(self) -> bool:
        if hasattr(self.buffer, "writable"):
            return self.buffer.writable()
        return True


class _BytesIOWrapper:
    # Wrapper that wraps a StringIO buffer and reads bytes from it
    # Created for compat with pyarrow read_csv
    def __init__(self, buffer: StringIO | TextIOBase, encoding: str = "utf-8") -> None:
        self.buffer = buffer
        self.encoding = encoding
        # Because a character can be represented by more than 1 byte,
        # it is possible that reading will produce more bytes than n
        # We store the extra bytes in this overflow variable, and append the
        # overflow to the front of the bytestring the next time reading is performed
        self.overflow = b""

    def __getattr__(self, attr: str):
        return getattr(self.buffer, attr)

    def read(self, n: int | None = -1) -> bytes:
        assert self.buffer is not None
        bytestring = self.buffer.read(n).encode(self.encoding)
        # When n=-1/n greater than remaining bytes: Read entire file/rest of file
        combined_bytestring = self.overflow + bytestring
        if n is None or n < 0 or n >= len(combined_bytestring):
            self.overflow = b""
            return combined_bytestring
        else:
            to_return = combined_bytestring[:n]
            self.overflow = combined_bytestring[n:]
            return to_return


def _maybe_memory_map(
    handle: str | BaseBuffer, memory_map: bool
) -> tuple[str | BaseBuffer, bool, list[BaseBuffer]]:
    """Try to memory map file/buffer."""
    handles: list[BaseBuffer] = []
    memory_map &= hasattr(handle, "fileno") or isinstance(handle, str)
    if not memory_map:
        return handle, memory_map, handles

    # mmap used by only read_csv
    handle = cast(ReadCsvBuffer, handle)

    # need to open the file first
    if isinstance(handle, str):
        handle = open(handle, "rb")
        handles.append(handle)

    try:
        # open mmap and adds *-able
        # error: Argument 1 to "_IOWrapper" has incompatible type "mmap";
        # expected "BaseBuffer"
        wrapped = _IOWrapper(
            mmap.mmap(
                handle.fileno(), 0, access=mmap.ACCESS_READ  # type: ignore[arg-type]
            )
        )
    finally:
        for handle in reversed(handles):
            # error: "BaseBuffer" has no attribute "close"
            handle.close()  # type: ignore[attr-defined]

    return wrapped, memory_map, [wrapped]


def file_exists(filepath_or_buffer: FilePath | BaseBuffer) -> bool:
    """Test whether file exists."""
    exists = False
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    if not isinstance(filepath_or_buffer, str):
        return exists
    try:
        exists = os.path.exists(filepath_or_buffer)
        # gh-5874: if the filepath is too long will raise here
    except (TypeError, ValueError):
        pass
    return exists


def _is_binary_mode(handle: FilePath | BaseBuffer, mode: str) -> bool:
    """Whether the handle is opened in binary mode"""
    # specified by user
    if "t" in mode or "b" in mode:
        return "b" in mode

    # exceptions
    text_classes = (
        # classes that expect string but have 'b' in mode
        codecs.StreamWriter,
        codecs.StreamReader,
        codecs.StreamReaderWriter,
    )
    if issubclass(type(handle), text_classes):
        return False

    return isinstance(handle, _get_binary_io_classes()) or "b" in getattr(
        handle, "mode", mode
    )


@functools.lru_cache
def _get_binary_io_classes() -> tuple[type, ...]:
    """IO classes that that expect bytes"""
    binary_classes: tuple[type, ...] = (BufferedIOBase, RawIOBase)

    # python-zstandard doesn't use any of the builtin base classes; instead we
    # have to use the `zstd.ZstdDecompressionReader` class for isinstance checks.
    # Unfortunately `zstd.ZstdDecompressionReader` isn't exposed by python-zstandard
    # so we have to get it from a `zstd.ZstdDecompressor` instance.
    # See also https://github.com/indygreg/python-zstandard/pull/165.
    zstd = import_optional_dependency("zstandard", errors="ignore")
    if zstd is not None:
        with zstd.ZstdDecompressor().stream_reader(b"") as reader:
            binary_classes += (type(reader),)

    return binary_classes


def is_potential_multi_index(
    columns: Sequence[Hashable] | MultiIndex,
    index_col: bool | Sequence[int] | None = None,
) -> bool:
    """
    Check whether or not the `columns` parameter
    could be converted into a MultiIndex.

    Parameters
    ----------
    columns : array-like
        Object which may or may not be convertible into a MultiIndex
    index_col : None, bool or list, optional
        Column or columns to use as the (possibly hierarchical) index

    Returns
    -------
    bool : Whether or not columns could become a MultiIndex
    """
    if index_col is None or isinstance(index_col, bool):
        index_col = []

    return bool(
        len(columns)
        and not isinstance(columns, ABCMultiIndex)
        and all(isinstance(c, tuple) for c in columns if c not in list(index_col))
    )


def dedup_names(
    names: Sequence[Hashable], is_potential_multiindex: bool
) -> Sequence[Hashable]:
    """
    Rename column names if duplicates exist.

    Currently the renaming is done by appending a period and an autonumeric,
    but a custom pattern may be supported in the future.

    Examples
    --------
    >>> dedup_names(["x", "y", "x", "x"], is_potential_multiindex=False)
    ['x', 'y', 'x.1', 'x.2']
    """
    names = list(names)  # so we can index
    counts: DefaultDict[Hashable, int] = defaultdict(int)

    for i, col in enumerate(names):
        cur_count = counts[col]

        while cur_count > 0:
            counts[col] = cur_count + 1

            if is_potential_multiindex:
                # for mypy
                assert isinstance(col, tuple)
                col = col[:-1] + (f"{col[-1]}.{cur_count}",)
            else:
                col = f"{col}.{cur_count}"
            cur_count = counts[col]

        names[i] = col
        counts[col] = cur_count + 1

    return names

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\pivot\table.py ===
# Copyright (c) 2010-2024 openpyxl


from collections import defaultdict
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Integer,
    NoneSet,
    Set,
    Bool,
    String,
    Bool,
    Sequence,
)

from openpyxl.descriptors.excel import ExtensionList, Relation
from openpyxl.descriptors.sequence import NestedSequence
from openpyxl.xml.constants import SHEET_MAIN_NS
from openpyxl.xml.functions import tostring
from openpyxl.packaging.relationship import (
    RelationshipList,
    Relationship,
    get_rels_path
)
from .fields import Index

from openpyxl.worksheet.filters import (
    AutoFilter,
)


class HierarchyUsage(Serialisable):

    tagname = "hierarchyUsage"

    hierarchyUsage = Integer()

    def __init__(self,
                 hierarchyUsage=None,
                ):
        self.hierarchyUsage = hierarchyUsage


class ColHierarchiesUsage(Serialisable):

    tagname = "colHierarchiesUsage"

    colHierarchyUsage = Sequence(expected_type=HierarchyUsage, )

    __elements__ = ('colHierarchyUsage',)
    __attrs__ = ('count', )

    def __init__(self,
                 count=None,
                 colHierarchyUsage=(),
                ):
        self.colHierarchyUsage = colHierarchyUsage


    @property
    def count(self):
        return len(self.colHierarchyUsage)


class RowHierarchiesUsage(Serialisable):

    tagname = "rowHierarchiesUsage"

    rowHierarchyUsage = Sequence(expected_type=HierarchyUsage, )

    __elements__ = ('rowHierarchyUsage',)
    __attrs__ = ('count', )

    def __init__(self,
                 count=None,
                 rowHierarchyUsage=(),
                ):
        self.rowHierarchyUsage = rowHierarchyUsage

    @property
    def count(self):
        return len(self.rowHierarchyUsage)


class PivotFilter(Serialisable):

    tagname = "filter"

    fld = Integer()
    mpFld = Integer(allow_none=True)
    type = Set(values=(['unknown', 'count', 'percent', 'sum', 'captionEqual',
                        'captionNotEqual', 'captionBeginsWith', 'captionNotBeginsWith',
                        'captionEndsWith', 'captionNotEndsWith', 'captionContains',
                        'captionNotContains', 'captionGreaterThan', 'captionGreaterThanOrEqual',
                        'captionLessThan', 'captionLessThanOrEqual', 'captionBetween',
                        'captionNotBetween', 'valueEqual', 'valueNotEqual', 'valueGreaterThan',
                        'valueGreaterThanOrEqual', 'valueLessThan', 'valueLessThanOrEqual',
                        'valueBetween', 'valueNotBetween', 'dateEqual', 'dateNotEqual',
                        'dateOlderThan', 'dateOlderThanOrEqual', 'dateNewerThan',
                        'dateNewerThanOrEqual', 'dateBetween', 'dateNotBetween', 'tomorrow',
                        'today', 'yesterday', 'nextWeek', 'thisWeek', 'lastWeek', 'nextMonth',
                        'thisMonth', 'lastMonth', 'nextQuarter', 'thisQuarter', 'lastQuarter',
                        'nextYear', 'thisYear', 'lastYear', 'yearToDate', 'Q1', 'Q2', 'Q3', 'Q4',
                        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11',
                        'M12']))
    evalOrder = Integer(allow_none=True)
    id = Integer()
    iMeasureHier = Integer(allow_none=True)
    iMeasureFld = Integer(allow_none=True)
    name = String(allow_none=True)
    description = String(allow_none=True)
    stringValue1 = String(allow_none=True)
    stringValue2 = String(allow_none=True)
    autoFilter = Typed(expected_type=AutoFilter, )
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('autoFilter',)

    def __init__(self,
                 fld=None,
                 mpFld=None,
                 type=None,
                 evalOrder=None,
                 id=None,
                 iMeasureHier=None,
                 iMeasureFld=None,
                 name=None,
                 description=None,
                 stringValue1=None,
                 stringValue2=None,
                 autoFilter=None,
                 extLst=None,
                ):
        self.fld = fld
        self.mpFld = mpFld
        self.type = type
        self.evalOrder = evalOrder
        self.id = id
        self.iMeasureHier = iMeasureHier
        self.iMeasureFld = iMeasureFld
        self.name = name
        self.description = description
        self.stringValue1 = stringValue1
        self.stringValue2 = stringValue2
        self.autoFilter = autoFilter


class PivotFilters(Serialisable):

    count = Integer()
    filter = Typed(expected_type=PivotFilter, allow_none=True)

    __elements__ = ('filter',)

    def __init__(self,
                 count=None,
                 filter=None,
                ):
        self.filter = filter


class PivotTableStyle(Serialisable):

    tagname = "pivotTableStyleInfo"

    name = String(allow_none=True)
    showRowHeaders = Bool()
    showColHeaders = Bool()
    showRowStripes = Bool()
    showColStripes = Bool()
    showLastColumn = Bool()

    def __init__(self,
                 name=None,
                 showRowHeaders=None,
                 showColHeaders=None,
                 showRowStripes=None,
                 showColStripes=None,
                 showLastColumn=None,
                ):
        self.name = name
        self.showRowHeaders = showRowHeaders
        self.showColHeaders = showColHeaders
        self.showRowStripes = showRowStripes
        self.showColStripes = showColStripes
        self.showLastColumn = showLastColumn


class MemberList(Serialisable):

    tagname = "members"

    level = Integer(allow_none=True)
    member = NestedSequence(expected_type=String, attribute="name")

    __elements__ = ('member',)

    def __init__(self,
                 count=None,
                 level=None,
                 member=(),
                ):
        self.level = level
        self.member = member

    @property
    def count(self):
        return len(self.member)


class MemberProperty(Serialisable):

    tagname = "mps"

    name = String(allow_none=True)
    showCell = Bool(allow_none=True)
    showTip = Bool(allow_none=True)
    showAsCaption = Bool(allow_none=True)
    nameLen = Integer(allow_none=True)
    pPos = Integer(allow_none=True)
    pLen = Integer(allow_none=True)
    level = Integer(allow_none=True)
    field = Integer()

    def __init__(self,
                 name=None,
                 showCell=None,
                 showTip=None,
                 showAsCaption=None,
                 nameLen=None,
                 pPos=None,
                 pLen=None,
                 level=None,
                 field=None,
                ):
        self.name = name
        self.showCell = showCell
        self.showTip = showTip
        self.showAsCaption = showAsCaption
        self.nameLen = nameLen
        self.pPos = pPos
        self.pLen = pLen
        self.level = level
        self.field = field


class PivotHierarchy(Serialisable):

    tagname = "pivotHierarchy"

    outline = Bool()
    multipleItemSelectionAllowed = Bool()
    subtotalTop = Bool()
    showInFieldList = Bool()
    dragToRow = Bool()
    dragToCol = Bool()
    dragToPage = Bool()
    dragToData = Bool()
    dragOff = Bool()
    includeNewItemsInFilter = Bool()
    caption = String(allow_none=True)
    mps = NestedSequence(expected_type=MemberProperty, count=True)
    members = Typed(expected_type=MemberList, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('mps', 'members',)

    def __init__(self,
                 outline=None,
                 multipleItemSelectionAllowed=None,
                 subtotalTop=None,
                 showInFieldList=None,
                 dragToRow=None,
                 dragToCol=None,
                 dragToPage=None,
                 dragToData=None,
                 dragOff=None,
                 includeNewItemsInFilter=None,
                 caption=None,
                 mps=(),
                 members=None,
                 extLst=None,
                ):
        self.outline = outline
        self.multipleItemSelectionAllowed = multipleItemSelectionAllowed
        self.subtotalTop = subtotalTop
        self.showInFieldList = showInFieldList
        self.dragToRow = dragToRow
        self.dragToCol = dragToCol
        self.dragToPage = dragToPage
        self.dragToData = dragToData
        self.dragOff = dragOff
        self.includeNewItemsInFilter = includeNewItemsInFilter
        self.caption = caption
        self.mps = mps
        self.members = members
        self.extLst = extLst


class Reference(Serialisable):

    tagname = "reference"

    field = Integer(allow_none=True)
    selected = Bool(allow_none=True)
    byPosition = Bool(allow_none=True)
    relative = Bool(allow_none=True)
    defaultSubtotal = Bool(allow_none=True)
    sumSubtotal = Bool(allow_none=True)
    countASubtotal = Bool(allow_none=True)
    avgSubtotal = Bool(allow_none=True)
    maxSubtotal = Bool(allow_none=True)
    minSubtotal = Bool(allow_none=True)
    productSubtotal = Bool(allow_none=True)
    countSubtotal = Bool(allow_none=True)
    stdDevSubtotal = Bool(allow_none=True)
    stdDevPSubtotal = Bool(allow_none=True)
    varSubtotal = Bool(allow_none=True)
    varPSubtotal = Bool(allow_none=True)
    x = Sequence(expected_type=Index)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('x',)

    def __init__(self,
                 field=None,
                 count=None,
                 selected=None,
                 byPosition=None,
                 relative=None,
                 defaultSubtotal=None,
                 sumSubtotal=None,
                 countASubtotal=None,
                 avgSubtotal=None,
                 maxSubtotal=None,
                 minSubtotal=None,
                 productSubtotal=None,
                 countSubtotal=None,
                 stdDevSubtotal=None,
                 stdDevPSubtotal=None,
                 varSubtotal=None,
                 varPSubtotal=None,
                 x=(),
                 extLst=None,
                ):
        self.field = field
        self.selected = selected
        self.byPosition = byPosition
        self.relative = relative
        self.defaultSubtotal = defaultSubtotal
        self.sumSubtotal = sumSubtotal
        self.countASubtotal = countASubtotal
        self.avgSubtotal = avgSubtotal
        self.maxSubtotal = maxSubtotal
        self.minSubtotal = minSubtotal
        self.productSubtotal = productSubtotal
        self.countSubtotal = countSubtotal
        self.stdDevSubtotal = stdDevSubtotal
        self.stdDevPSubtotal = stdDevPSubtotal
        self.varSubtotal = varSubtotal
        self.varPSubtotal = varPSubtotal
        self.x = x


    @property
    def count(self):
        return len(self.field)


class PivotArea(Serialisable):

    tagname = "pivotArea"

    references = NestedSequence(expected_type=Reference, count=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)
    field = Integer(allow_none=True)
    type = NoneSet(values=(['normal', 'data', 'all', 'origin', 'button',
                            'topEnd', 'topRight']))
    dataOnly = Bool(allow_none=True)
    labelOnly = Bool(allow_none=True)
    grandRow = Bool(allow_none=True)
    grandCol = Bool(allow_none=True)
    cacheIndex = Bool(allow_none=True)
    outline = Bool(allow_none=True)
    offset = String(allow_none=True)
    collapsedLevelsAreSubtotals = Bool(allow_none=True)
    axis = NoneSet(values=(['axisRow', 'axisCol', 'axisPage', 'axisValues']))
    fieldPosition = Integer(allow_none=True)

    __elements__ = ('references',)

    def __init__(self,
                 references=(),
                 extLst=None,
                 field=None,
                 type="normal",
                 dataOnly=True,
                 labelOnly=None,
                 grandRow=None,
                 grandCol=None,
                 cacheIndex=None,
                 outline=True,
                 offset=None,
                 collapsedLevelsAreSubtotals=None,
                 axis=None,
                 fieldPosition=None,
                ):
        self.references = references
        self.extLst = extLst
        self.field = field
        self.type = type
        self.dataOnly = dataOnly
        self.labelOnly = labelOnly
        self.grandRow = grandRow
        self.grandCol = grandCol
        self.cacheIndex = cacheIndex
        self.outline = outline
        self.offset = offset
        self.collapsedLevelsAreSubtotals = collapsedLevelsAreSubtotals
        self.axis = axis
        self.fieldPosition = fieldPosition


class ChartFormat(Serialisable):

    tagname = "chartFormat"

    chart = Integer()
    format = Integer()
    series = Bool()
    pivotArea = Typed(expected_type=PivotArea, )

    __elements__ = ('pivotArea',)

    def __init__(self,
                 chart=None,
                 format=None,
                 series=None,
                 pivotArea=None,
                ):
        self.chart = chart
        self.format = format
        self.series = series
        self.pivotArea = pivotArea


class ConditionalFormat(Serialisable):

    tagname = "conditionalFormat"

    scope = Set(values=(['selection', 'data', 'field']))
    type = NoneSet(values=(['all', 'row', 'column']))
    priority = Integer()
    pivotAreas = NestedSequence(expected_type=PivotArea)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('pivotAreas',)

    def __init__(self,
                 scope="selection",
                 type=None,
                 priority=None,
                 pivotAreas=(),
                 extLst=None,
                ):
        self.scope = scope
        self.type = type
        self.priority = priority
        self.pivotAreas = pivotAreas
        self.extLst = extLst


class ConditionalFormatList(Serialisable):

    tagname = "conditionalFormats"

    conditionalFormat = Sequence(expected_type=ConditionalFormat)

    __attrs__ = ("count",)

    def __init__(self, conditionalFormat=(), count=None):
        self.conditionalFormat = conditionalFormat


    def by_priority(self):
        """
        Return a dictionary of format objects keyed by (field id and format property).
        This can be used to map the formats to field but also to dedupe to match
        worksheet definitions which are grouped by cell range
        """

        fmts = {}
        for fmt in self.conditionalFormat:
            for area in fmt.pivotAreas:
                for ref in area.references:
                    for field in ref.x:
                        key = (field.v, fmt.priority)
                        fmts[key] = fmt

        return fmts


    def _dedupe(self):
        """
        Group formats by field index and priority.
        Sorted to match sorting and grouping for corresponding worksheet formats

        The implemtenters notes contain significant deviance from the OOXML
        specification, in particular how conditional formats in tables relate to
        those defined in corresponding worksheets and how to determine which
        format applies to which fields.

        There are some magical interdependencies:

        * Every pivot table fmt must have a worksheet cxf with the same priority.

        * In the reference part the field 4294967294 refers to a data field, the
        spec says -2

        * Data fields are referenced by the 0-index reference.x.v value

        Things are made more complicated by the fact that field items behave
        diffently if the parent is a reference or shared item: "In Office if the
        parent is the reference element, then restrictions of this value are
        defined by reference@field. If the parent is the tables element, then
        this value specifies the index into the table tag position in @url."
        Yeah, right!
        """
        fmts = self.by_priority()
        # sort by priority in order, keeping the highest numerical priority, least when
        # actually applied
        # this is not documented but it's what Excel is happy with
        fmts = {field:fmt for (field, priority), fmt in sorted(fmts.items(), reverse=True)}
        #fmts = {field:fmt for (field, priority), fmt in fmts.items()}
        if fmts:
            self.conditionalFormat = list(fmts.values())


    @property
    def count(self):
        return len(self.conditionalFormat)


    def to_tree(self, tagname=None):
        self._dedupe()
        return super().to_tree(tagname)


class Format(Serialisable):

    tagname = "format"

    action = NoneSet(values=(['blank', 'formatting', 'drill', 'formula']))
    dxfId = Integer(allow_none=True)
    pivotArea = Typed(expected_type=PivotArea, )
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('pivotArea',)

    def __init__(self,
                 action="formatting",
                 dxfId=None,
                 pivotArea=None,
                 extLst=None,
                ):
        self.action = action
        self.dxfId = dxfId
        self.pivotArea = pivotArea
        self.extLst = extLst


class DataField(Serialisable):

    tagname = "dataField"

    name = String(allow_none=True)
    fld = Integer()
    subtotal = Set(values=(['average', 'count', 'countNums', 'max', 'min',
                            'product', 'stdDev', 'stdDevp', 'sum', 'var', 'varp']))
    showDataAs = Set(values=(['normal', 'difference', 'percent',
                              'percentDiff', 'runTotal', 'percentOfRow', 'percentOfCol',
                              'percentOfTotal', 'index']))
    baseField = Integer()
    baseItem = Integer()
    numFmtId = Integer(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()


    def __init__(self,
                 name=None,
                 fld=None,
                 subtotal="sum",
                 showDataAs="normal",
                 baseField=-1,
                 baseItem=1048832,
                 numFmtId=None,
                 extLst=None,
                ):
        self.name = name
        self.fld = fld
        self.subtotal = subtotal
        self.showDataAs = showDataAs
        self.baseField = baseField
        self.baseItem = baseItem
        self.numFmtId = numFmtId
        self.extLst = extLst


class PageField(Serialisable):

    tagname = "pageField"

    fld = Integer()
    item = Integer(allow_none=True)
    hier = Integer(allow_none=True)
    name = String(allow_none=True)
    cap = String(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()

    def __init__(self,
                 fld=None,
                 item=None,
                 hier=None,
                 name=None,
                 cap=None,
                 extLst=None,
                ):
        self.fld = fld
        self.item = item
        self.hier = hier
        self.name = name
        self.cap = cap
        self.extLst = extLst


class RowColItem(Serialisable):

    tagname = "i"

    t = Set(values=(['data', 'default', 'sum', 'countA', 'avg', 'max', 'min',
                     'product', 'count', 'stdDev', 'stdDevP', 'var', 'varP', 'grand',
                     'blank']))
    r = Integer()
    i = Integer()
    x = Sequence(expected_type=Index, attribute="v")

    __elements__ = ('x',)

    def __init__(self,
                 t="data",
                 r=0,
                 i=0,
                 x=(),
                ):
        self.t = t
        self.r = r
        self.i = i
        self.x = x


class RowColField(Serialisable):

    tagname = "field"

    x = Integer()

    def __init__(self,
                 x=None,
                ):
        self.x = x


class AutoSortScope(Serialisable):

    pivotArea = Typed(expected_type=PivotArea, )

    __elements__ = ('pivotArea',)

    def __init__(self,
                 pivotArea=None,
                ):
        self.pivotArea = pivotArea


class FieldItem(Serialisable):

    tagname = "item"

    n = String(allow_none=True)
    t = Set(values=(['data', 'default', 'sum', 'countA', 'avg', 'max', 'min',
                     'product', 'count', 'stdDev', 'stdDevP', 'var', 'varP', 'grand',
                     'blank']))
    h = Bool(allow_none=True)
    s = Bool(allow_none=True)
    sd = Bool(allow_none=True)
    f = Bool(allow_none=True)
    m = Bool(allow_none=True)
    c = Bool(allow_none=True)
    x = Integer(allow_none=True)
    d = Bool(allow_none=True)
    e = Bool(allow_none=True)

    def __init__(self,
                 n=None,
                 t="data",
                 h=None,
                 s=None,
                 sd=True,
                 f=None,
                 m=None,
                 c=None,
                 x=None,
                 d=None,
                 e=None,
                ):
        self.n = n
        self.t = t
        self.h = h
        self.s = s
        self.sd = sd
        self.f = f
        self.m = m
        self.c = c
        self.x = x
        self.d = d
        self.e = e


class PivotField(Serialisable):

    tagname = "pivotField"

    items = NestedSequence(expected_type=FieldItem, count=True)
    autoSortScope = Typed(expected_type=AutoSortScope, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)
    name = String(allow_none=True)
    axis = NoneSet(values=(['axisRow', 'axisCol', 'axisPage', 'axisValues']))
    dataField = Bool(allow_none=True)
    subtotalCaption = String(allow_none=True)
    showDropDowns = Bool(allow_none=True)
    hiddenLevel = Bool(allow_none=True)
    uniqueMemberProperty = String(allow_none=True)
    compact = Bool(allow_none=True)
    allDrilled = Bool(allow_none=True)
    numFmtId = Integer(allow_none=True)
    outline = Bool(allow_none=True)
    subtotalTop = Bool(allow_none=True)
    dragToRow = Bool(allow_none=True)
    dragToCol = Bool(allow_none=True)
    multipleItemSelectionAllowed = Bool(allow_none=True)
    dragToPage = Bool(allow_none=True)
    dragToData = Bool(allow_none=True)
    dragOff = Bool(allow_none=True)
    showAll = Bool(allow_none=True)
    insertBlankRow = Bool(allow_none=True)
    serverField = Bool(allow_none=True)
    insertPageBreak = Bool(allow_none=True)
    autoShow = Bool(allow_none=True)
    topAutoShow = Bool(allow_none=True)
    hideNewItems = Bool(allow_none=True)
    measureFilter = Bool(allow_none=True)
    includeNewItemsInFilter = Bool(allow_none=True)
    itemPageCount = Integer(allow_none=True)
    sortType = Set(values=(['manual', 'ascending', 'descending']))
    dataSourceSort = Bool(allow_none=True)
    nonAutoSortDefault = Bool(allow_none=True)
    rankBy = Integer(allow_none=True)
    defaultSubtotal = Bool(allow_none=True)
    sumSubtotal = Bool(allow_none=True)
    countASubtotal = Bool(allow_none=True)
    avgSubtotal = Bool(allow_none=True)
    maxSubtotal = Bool(allow_none=True)
    minSubtotal = Bool(allow_none=True)
    productSubtotal = Bool(allow_none=True)
    countSubtotal = Bool(allow_none=True)
    stdDevSubtotal = Bool(allow_none=True)
    stdDevPSubtotal = Bool(allow_none=True)
    varSubtotal = Bool(allow_none=True)
    varPSubtotal = Bool(allow_none=True)
    showPropCell = Bool(allow_none=True)
    showPropTip = Bool(allow_none=True)
    showPropAsCaption = Bool(allow_none=True)
    defaultAttributeDrillState = Bool(allow_none=True)

    __elements__ = ('items', 'autoSortScope',)

    def __init__(self,
                 items=(),
                 autoSortScope=None,
                 name=None,
                 axis=None,
                 dataField=None,
                 subtotalCaption=None,
                 showDropDowns=True,
                 hiddenLevel=None,
                 uniqueMemberProperty=None,
                 compact=True,
                 allDrilled=None,
                 numFmtId=None,
                 outline=True,
                 subtotalTop=True,
                 dragToRow=True,
                 dragToCol=True,
                 multipleItemSelectionAllowed=None,
                 dragToPage=True,
                 dragToData=True,
                 dragOff=True,
                 showAll=True,
                 insertBlankRow=None,
                 serverField=None,
                 insertPageBreak=None,
                 autoShow=None,
                 topAutoShow=True,
                 hideNewItems=None,
                 measureFilter=None,
                 includeNewItemsInFilter=None,
                 itemPageCount=10,
                 sortType="manual",
                 dataSourceSort=None,
                 nonAutoSortDefault=None,
                 rankBy=None,
                 defaultSubtotal=True,
                 sumSubtotal=None,
                 countASubtotal=None,
                 avgSubtotal=None,
                 maxSubtotal=None,
                 minSubtotal=None,
                 productSubtotal=None,
                 countSubtotal=None,
                 stdDevSubtotal=None,
                 stdDevPSubtotal=None,
                 varSubtotal=None,
                 varPSubtotal=None,
                 showPropCell=None,
                 showPropTip=None,
                 showPropAsCaption=None,
                 defaultAttributeDrillState=None,
                 extLst=None,
                ):
        self.items = items
        self.autoSortScope = autoSortScope
        self.name = name
        self.axis = axis
        self.dataField = dataField
        self.subtotalCaption = subtotalCaption
        self.showDropDowns = showDropDowns
        self.hiddenLevel = hiddenLevel
        self.uniqueMemberProperty = uniqueMemberProperty
        self.compact = compact
        self.allDrilled = allDrilled
        self.numFmtId = numFmtId
        self.outline = outline
        self.subtotalTop = subtotalTop
        self.dragToRow = dragToRow
        self.dragToCol = dragToCol
        self.multipleItemSelectionAllowed = multipleItemSelectionAllowed
        self.dragToPage = dragToPage
        self.dragToData = dragToData
        self.dragOff = dragOff
        self.showAll = showAll
        self.insertBlankRow = insertBlankRow
        self.serverField = serverField
        self.insertPageBreak = insertPageBreak
        self.autoShow = autoShow
        self.topAutoShow = topAutoShow
        self.hideNewItems = hideNewItems
        self.measureFilter = measureFilter
        self.includeNewItemsInFilter = includeNewItemsInFilter
        self.itemPageCount = itemPageCount
        self.sortType = sortType
        self.dataSourceSort = dataSourceSort
        self.nonAutoSortDefault = nonAutoSortDefault
        self.rankBy = rankBy
        self.defaultSubtotal = defaultSubtotal
        self.sumSubtotal = sumSubtotal
        self.countASubtotal = countASubtotal
        self.avgSubtotal = avgSubtotal
        self.maxSubtotal = maxSubtotal
        self.minSubtotal = minSubtotal
        self.productSubtotal = productSubtotal
        self.countSubtotal = countSubtotal
        self.stdDevSubtotal = stdDevSubtotal
        self.stdDevPSubtotal = stdDevPSubtotal
        self.varSubtotal = varSubtotal
        self.varPSubtotal = varPSubtotal
        self.showPropCell = showPropCell
        self.showPropTip = showPropTip
        self.showPropAsCaption = showPropAsCaption
        self.defaultAttributeDrillState = defaultAttributeDrillState


class Location(Serialisable):

    tagname = "location"

    ref = String()
    firstHeaderRow = Integer()
    firstDataRow = Integer()
    firstDataCol = Integer()
    rowPageCount = Integer(allow_none=True)
    colPageCount = Integer(allow_none=True)

    def __init__(self,
                 ref=None,
                 firstHeaderRow=None,
                 firstDataRow=None,
                 firstDataCol=None,
                 rowPageCount=None,
                 colPageCount=None,
                ):
        self.ref = ref
        self.firstHeaderRow = firstHeaderRow
        self.firstDataRow = firstDataRow
        self.firstDataCol = firstDataCol
        self.rowPageCount = rowPageCount
        self.colPageCount = colPageCount


class TableDefinition(Serialisable):

    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.pivotTable+xml"
    rel_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/pivotTable"
    _id = 1
    _path = "/xl/pivotTables/pivotTable{0}.xml"

    tagname = "pivotTableDefinition"
    cache = None

    name = String()
    cacheId = Integer()
    dataOnRows = Bool()
    dataPosition = Integer(allow_none=True)
    dataCaption = String()
    grandTotalCaption = String(allow_none=True)
    errorCaption = String(allow_none=True)
    showError = Bool()
    missingCaption = String(allow_none=True)
    showMissing = Bool()
    pageStyle = String(allow_none=True)
    pivotTableStyle = String(allow_none=True)
    vacatedStyle = String(allow_none=True)
    tag = String(allow_none=True)
    updatedVersion = Integer()
    minRefreshableVersion = Integer()
    asteriskTotals = Bool()
    showItems = Bool()
    editData = Bool()
    disableFieldList = Bool()
    showCalcMbrs = Bool()
    visualTotals = Bool()
    showMultipleLabel = Bool()
    showDataDropDown = Bool()
    showDrill = Bool()
    printDrill = Bool()
    showMemberPropertyTips = Bool()
    showDataTips = Bool()
    enableWizard = Bool()
    enableDrill = Bool()
    enableFieldProperties = Bool()
    preserveFormatting = Bool()
    useAutoFormatting = Bool()
    pageWrap = Integer()
    pageOverThenDown = Bool()
    subtotalHiddenItems = Bool()
    rowGrandTotals = Bool()
    colGrandTotals = Bool()
    fieldPrintTitles = Bool()
    itemPrintTitles = Bool()
    mergeItem = Bool()
    showDropZones = Bool()
    createdVersion = Integer()
    indent = Integer()
    showEmptyRow = Bool()
    showEmptyCol = Bool()
    showHeaders = Bool()
    compact = Bool()
    outline = Bool()
    outlineData = Bool()
    compactData = Bool()
    published = Bool()
    gridDropZones = Bool()
    immersive = Bool()
    multipleFieldFilters = Bool()
    chartFormat = Integer()
    rowHeaderCaption = String(allow_none=True)
    colHeaderCaption = String(allow_none=True)
    fieldListSortAscending = Bool()
    mdxSubqueries = Bool()
    customListSort = Bool(allow_none=True)
    autoFormatId = Integer(allow_none=True)
    applyNumberFormats = Bool()
    applyBorderFormats = Bool()
    applyFontFormats = Bool()
    applyPatternFormats = Bool()
    applyAlignmentFormats = Bool()
    applyWidthHeightFormats = Bool()
    location = Typed(expected_type=Location, )
    pivotFields = NestedSequence(expected_type=PivotField, count=True)
    rowFields = NestedSequence(expected_type=RowColField, count=True)
    rowItems = NestedSequence(expected_type=RowColItem, count=True)
    colFields = NestedSequence(expected_type=RowColField, count=True)
    colItems = NestedSequence(expected_type=RowColItem, count=True)
    pageFields = NestedSequence(expected_type=PageField, count=True)
    dataFields = NestedSequence(expected_type=DataField, count=True)
    formats = NestedSequence(expected_type=Format, count=True)
    conditionalFormats = Typed(expected_type=ConditionalFormatList, allow_none=True)
    chartFormats = NestedSequence(expected_type=ChartFormat, count=True)
    pivotHierarchies = NestedSequence(expected_type=PivotHierarchy, count=True)
    pivotTableStyleInfo = Typed(expected_type=PivotTableStyle, allow_none=True)
    filters = NestedSequence(expected_type=PivotFilter, count=True)
    rowHierarchiesUsage = Typed(expected_type=RowHierarchiesUsage, allow_none=True)
    colHierarchiesUsage = Typed(expected_type=ColHierarchiesUsage, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)
    id = Relation()

    __elements__ = ('location', 'pivotFields', 'rowFields', 'rowItems',
                    'colFields', 'colItems', 'pageFields', 'dataFields', 'formats',
                    'conditionalFormats', 'chartFormats', 'pivotHierarchies',
                    'pivotTableStyleInfo', 'filters', 'rowHierarchiesUsage',
                    'colHierarchiesUsage',)

    def __init__(self,
                 name=None,
                 cacheId=None,
                 dataOnRows=False,
                 dataPosition=None,
                 dataCaption=None,
                 grandTotalCaption=None,
                 errorCaption=None,
                 showError=False,
                 missingCaption=None,
                 showMissing=True,
                 pageStyle=None,
                 pivotTableStyle=None,
                 vacatedStyle=None,
                 tag=None,
                 updatedVersion=0,
                 minRefreshableVersion=0,
                 asteriskTotals=False,
                 showItems=True,
                 editData=False,
                 disableFieldList=False,
                 showCalcMbrs=True,
                 visualTotals=True,
                 showMultipleLabel=True,
                 showDataDropDown=True,
                 showDrill=True,
                 printDrill=False,
                 showMemberPropertyTips=True,
                 showDataTips=True,
                 enableWizard=True,
                 enableDrill=True,
                 enableFieldProperties=True,
                 preserveFormatting=True,
                 useAutoFormatting=False,
                 pageWrap=0,
                 pageOverThenDown=False,
                 subtotalHiddenItems=False,
                 rowGrandTotals=True,
                 colGrandTotals=True,
                 fieldPrintTitles=False,
                 itemPrintTitles=False,
                 mergeItem=False,
                 showDropZones=True,
                 createdVersion=0,
                 indent=1,
                 showEmptyRow=False,
                 showEmptyCol=False,
                 showHeaders=True,
                 compact=True,
                 outline=False,
                 outlineData=False,
                 compactData=True,
                 published=False,
                 gridDropZones=False,
                 immersive=True,
                 multipleFieldFilters=None,
                 chartFormat=0,
                 rowHeaderCaption=None,
                 colHeaderCaption=None,
                 fieldListSortAscending=None,
                 mdxSubqueries=None,
                 customListSort=None,
                 autoFormatId=None,
                 applyNumberFormats=False,
                 applyBorderFormats=False,
                 applyFontFormats=False,
                 applyPatternFormats=False,
                 applyAlignmentFormats=False,
                 applyWidthHeightFormats=False,
                 location=None,
                 pivotFields=(),
                 rowFields=(),
                 rowItems=(),
                 colFields=(),
                 colItems=(),
                 pageFields=(),
                 dataFields=(),
                 formats=(),
                 conditionalFormats=None,
                 chartFormats=(),
                 pivotHierarchies=(),
                 pivotTableStyleInfo=None,
                 filters=(),
                 rowHierarchiesUsage=None,
                 colHierarchiesUsage=None,
                 extLst=None,
                 id=None,
                ):
        self.name = name
        self.cacheId = cacheId
        self.dataOnRows = dataOnRows
        self.dataPosition = dataPosition
        self.dataCaption = dataCaption
        self.grandTotalCaption = grandTotalCaption
        self.errorCaption = errorCaption
        self.showError = showError
        self.missingCaption = missingCaption
        self.showMissing = showMissing
        self.pageStyle = pageStyle
        self.pivotTableStyle = pivotTableStyle
        self.vacatedStyle = vacatedStyle
        self.tag = tag
        self.updatedVersion = updatedVersion
        self.minRefreshableVersion = minRefreshableVersion
        self.asteriskTotals = asteriskTotals
        self.showItems = showItems
        self.editData = editData
        self.disableFieldList = disableFieldList
        self.showCalcMbrs = showCalcMbrs
        self.visualTotals = visualTotals
        self.showMultipleLabel = showMultipleLabel
        self.showDataDropDown = showDataDropDown
        self.showDrill = showDrill
        self.printDrill = printDrill
        self.showMemberPropertyTips = showMemberPropertyTips
        self.showDataTips = showDataTips
        self.enableWizard = enableWizard
        self.enableDrill = enableDrill
        self.enableFieldProperties = enableFieldProperties
        self.preserveFormatting = preserveFormatting
        self.useAutoFormatting = useAutoFormatting
        self.pageWrap = pageWrap
        self.pageOverThenDown = pageOverThenDown
        self.subtotalHiddenItems = subtotalHiddenItems
        self.rowGrandTotals = rowGrandTotals
        self.colGrandTotals = colGrandTotals
        self.fieldPrintTitles = fieldPrintTitles
        self.itemPrintTitles = itemPrintTitles
        self.mergeItem = mergeItem
        self.showDropZones = showDropZones
        self.createdVersion = createdVersion
        self.indent = indent
        self.showEmptyRow = showEmptyRow
        self.showEmptyCol = showEmptyCol
        self.showHeaders = showHeaders
        self.compact = compact
        self.outline = outline
        self.outlineData = outlineData
        self.compactData = compactData
        self.published = published
        self.gridDropZones = gridDropZones
        self.immersive = immersive
        self.multipleFieldFilters = multipleFieldFilters
        self.chartFormat = chartFormat
        self.rowHeaderCaption = rowHeaderCaption
        self.colHeaderCaption = colHeaderCaption
        self.fieldListSortAscending = fieldListSortAscending
        self.mdxSubqueries = mdxSubqueries
        self.customListSort = customListSort
        self.autoFormatId = autoFormatId
        self.applyNumberFormats = applyNumberFormats
        self.applyBorderFormats = applyBorderFormats
        self.applyFontFormats = applyFontFormats
        self.applyPatternFormats = applyPatternFormats
        self.applyAlignmentFormats = applyAlignmentFormats
        self.applyWidthHeightFormats = applyWidthHeightFormats
        self.location = location
        self.pivotFields = pivotFields
        self.rowFields = rowFields
        self.rowItems = rowItems
        self.colFields = colFields
        self.colItems = colItems
        self.pageFields = pageFields
        self.dataFields = dataFields
        self.formats = formats
        self.conditionalFormats = conditionalFormats
        self.conditionalFormats = None
        self.chartFormats = chartFormats
        self.pivotHierarchies = pivotHierarchies
        self.pivotTableStyleInfo = pivotTableStyleInfo
        self.filters = filters
        self.rowHierarchiesUsage = rowHierarchiesUsage
        self.colHierarchiesUsage = colHierarchiesUsage
        self.extLst = extLst
        self.id = id


    def to_tree(self):
        tree = super().to_tree()
        tree.set("xmlns", SHEET_MAIN_NS)
        return tree


    @property
    def path(self):
        return self._path.format(self._id)


    def _write(self, archive, manifest):
        """
        Add to zipfile and update manifest
        """
        self._write_rels(archive, manifest)
        xml = tostring(self.to_tree())
        archive.writestr(self.path[1:], xml)
        manifest.append(self)


    def _write_rels(self, archive, manifest):
        """
        Write the relevant child objects and add links
        """
        if self.cache is None:
            return

        rels = RelationshipList()
        r = Relationship(Type=self.cache.rel_type, Target=self.cache.path)
        rels.append(r)
        self.id = r.id
        if self.cache.path[1:] not in archive.namelist():
            self.cache._write(archive, manifest)

        path = get_rels_path(self.path)
        xml = tostring(rels.to_tree())
        archive.writestr(path[1:], xml)


    def formatted_fields(self):
        """Map fields to associated conditional formats by priority"""
        if not self.conditionalFormats:
            return {}
        fields = defaultdict(list)
        for idx, prio in self.conditionalFormats.by_priority():
            name = self.dataFields[idx].name
            fields[name].append(prio)
        return fields


    @property
    def summary(self):
        """
        Provide a simplified summary of the table
        """

        return f"{self.name} {dict(self.location)}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\html.py ===
"""
:mod:`pandas.io.html` is a module containing functionality for dealing with
HTML IO.

"""

from __future__ import annotations

from collections import abc
import numbers
import re
from re import Pattern
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)
import warnings

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    AbstractMethodError,
    EmptyDataError,
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import is_list_like

from pandas import isna
from pandas.core.indexes.base import Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs

from pandas.io.common import (
    file_exists,
    get_handle,
    is_file_like,
    is_fsspec_url,
    is_url,
    stringify_path,
    validate_header_arg,
)
from pandas.io.formats.printing import pprint_thing
from pandas.io.parsers import TextParser

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )

    from pandas._typing import (
        BaseBuffer,
        DtypeBackend,
        FilePath,
        HTMLFlavors,
        ReadBuffer,
        StorageOptions,
    )

    from pandas import DataFrame

#############
# READ HTML #
#############
_RE_WHITESPACE = re.compile(r"[\r\n]+|\s{2,}")


def _remove_whitespace(s: str, regex: Pattern = _RE_WHITESPACE) -> str:
    """
    Replace extra whitespace inside of a string with a single space.

    Parameters
    ----------
    s : str or unicode
        The string from which to remove extra whitespace.
    regex : re.Pattern
        The regular expression to use to remove extra whitespace.

    Returns
    -------
    subd : str or unicode
        `s` with all extra whitespace replaced with a single space.
    """
    return regex.sub(" ", s.strip())


def _get_skiprows(skiprows: int | Sequence[int] | slice | None) -> int | Sequence[int]:
    """
    Get an iterator given an integer, slice or container.

    Parameters
    ----------
    skiprows : int, slice, container
        The iterator to use to skip rows; can also be a slice.

    Raises
    ------
    TypeError
        * If `skiprows` is not a slice, integer, or Container

    Returns
    -------
    it : iterable
        A proper iterator to use to skip rows of a DataFrame.
    """
    if isinstance(skiprows, slice):
        start, step = skiprows.start or 0, skiprows.step or 1
        return list(range(start, skiprows.stop, step))
    elif isinstance(skiprows, numbers.Integral) or is_list_like(skiprows):
        return cast("int | Sequence[int]", skiprows)
    elif skiprows is None:
        return 0
    raise TypeError(f"{type(skiprows).__name__} is not a valid type for skipping rows")


def _read(
    obj: FilePath | BaseBuffer,
    encoding: str | None,
    storage_options: StorageOptions | None,
) -> str | bytes:
    """
    Try to read from a url, file or string.

    Parameters
    ----------
    obj : str, unicode, path object, or file-like object

    Returns
    -------
    raw_text : str
    """
    text: str | bytes
    if (
        is_url(obj)
        or hasattr(obj, "read")
        or (isinstance(obj, str) and file_exists(obj))
    ):
        with get_handle(
            obj, "r", encoding=encoding, storage_options=storage_options
        ) as handles:
            text = handles.handle.read()
    elif isinstance(obj, (str, bytes)):
        text = obj
    else:
        raise TypeError(f"Cannot read object of type '{type(obj).__name__}'")
    return text


class _HtmlFrameParser:
    """
    Base class for parsers that parse HTML into DataFrames.

    Parameters
    ----------
    io : str or file-like
        This can be either a string of raw HTML, a valid URL using the HTTP,
        FTP, or FILE protocols or a file-like object.

    match : str or regex
        The text to match in the document.

    attrs : dict
        List of HTML <table> element attributes to match.

    encoding : str
        Encoding to be used by parser

    displayed_only : bool
        Whether or not items with "display:none" should be ignored

    extract_links : {None, "all", "header", "body", "footer"}
        Table elements in the specified section(s) with <a> tags will have their
        href extracted.

        .. versionadded:: 1.5.0

    Attributes
    ----------
    io : str or file-like
        raw HTML, URL, or file-like object

    match : regex
        The text to match in the raw HTML

    attrs : dict-like
        A dictionary of valid table attributes to use to search for table
        elements.

    encoding : str
        Encoding to be used by parser

    displayed_only : bool
        Whether or not items with "display:none" should be ignored

    extract_links : {None, "all", "header", "body", "footer"}
        Table elements in the specified section(s) with <a> tags will have their
        href extracted.

        .. versionadded:: 1.5.0

    Notes
    -----
    To subclass this class effectively you must override the following methods:
        * :func:`_build_doc`
        * :func:`_attr_getter`
        * :func:`_href_getter`
        * :func:`_text_getter`
        * :func:`_parse_td`
        * :func:`_parse_thead_tr`
        * :func:`_parse_tbody_tr`
        * :func:`_parse_tfoot_tr`
        * :func:`_parse_tables`
        * :func:`_equals_tag`
    See each method's respective documentation for details on their
    functionality.
    """

    def __init__(
        self,
        io: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
        match: str | Pattern,
        attrs: dict[str, str] | None,
        encoding: str,
        displayed_only: bool,
        extract_links: Literal[None, "header", "footer", "body", "all"],
        storage_options: StorageOptions = None,
    ) -> None:
        self.io = io
        self.match = match
        self.attrs = attrs
        self.encoding = encoding
        self.displayed_only = displayed_only
        self.extract_links = extract_links
        self.storage_options = storage_options

    def parse_tables(self):
        """
        Parse and return all tables from the DOM.

        Returns
        -------
        list of parsed (header, body, footer) tuples from tables.
        """
        tables = self._parse_tables(self._build_doc(), self.match, self.attrs)
        return (self._parse_thead_tbody_tfoot(table) for table in tables)

    def _attr_getter(self, obj, attr):
        """
        Return the attribute value of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        attr : str or unicode
            The attribute, such as "colspan"

        Returns
        -------
        str or unicode
            The attribute value.
        """
        # Both lxml and BeautifulSoup have the same implementation:
        return obj.get(attr)

    def _href_getter(self, obj) -> str | None:
        """
        Return a href if the DOM node contains a child <a> or None.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        href : str or unicode
            The href from the <a> child of the DOM node.
        """
        raise AbstractMethodError(self)

    def _text_getter(self, obj):
        """
        Return the text of an individual DOM node.

        Parameters
        ----------
        obj : node-like
            A DOM node.

        Returns
        -------
        text : str or unicode
            The text from an individual DOM node.
        """
        raise AbstractMethodError(self)

    def _parse_td(self, obj):
        """
        Return the td elements from a row element.

        Parameters
        ----------
        obj : node-like
            A DOM <tr> node.

        Returns
        -------
        list of node-like
            These are the elements of each row, i.e., the columns.
        """
        raise AbstractMethodError(self)

    def _parse_thead_tr(self, table):
        """
        Return the list of thead row elements from the parsed table element.

        Parameters
        ----------
        table : a table element that contains zero or more thead elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        raise AbstractMethodError(self)

    def _parse_tbody_tr(self, table):
        """
        Return the list of tbody row elements from the parsed table element.

        HTML5 table bodies consist of either 0 or more <tbody> elements (which
        only contain <tr> elements) or 0 or more <tr> elements. This method
        checks for both structures.

        Parameters
        ----------
        table : a table element that contains row elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        raise AbstractMethodError(self)

    def _parse_tfoot_tr(self, table):
        """
        Return the list of tfoot row elements from the parsed table element.

        Parameters
        ----------
        table : a table element that contains row elements.

        Returns
        -------
        list of node-like
            These are the <tr> row elements of a table.
        """
        raise AbstractMethodError(self)

    def _parse_tables(self, document, match, attrs):
        """
        Return all tables from the parsed DOM.

        Parameters
        ----------
        document : the DOM from which to parse the table element.

        match : str or regular expression
            The text to search for in the DOM tree.

        attrs : dict
            A dictionary of table attributes that can be used to disambiguate
            multiple tables on a page.

        Raises
        ------
        ValueError : `match` does not match any text in the document.

        Returns
        -------
        list of node-like
            HTML <table> elements to be parsed into raw data.
        """
        raise AbstractMethodError(self)

    def _equals_tag(self, obj, tag) -> bool:
        """
        Return whether an individual DOM node matches a tag

        Parameters
        ----------
        obj : node-like
            A DOM node.

        tag : str
            Tag name to be checked for equality.

        Returns
        -------
        boolean
            Whether `obj`'s tag name is `tag`
        """
        raise AbstractMethodError(self)

    def _build_doc(self):
        """
        Return a tree-like object that can be used to iterate over the DOM.

        Returns
        -------
        node-like
            The DOM from which to parse the table element.
        """
        raise AbstractMethodError(self)

    def _parse_thead_tbody_tfoot(self, table_html):
        """
        Given a table, return parsed header, body, and foot.

        Parameters
        ----------
        table_html : node-like

        Returns
        -------
        tuple of (header, body, footer), each a list of list-of-text rows.

        Notes
        -----
        Header and body are lists-of-lists. Top level list is a list of
        rows. Each row is a list of str text.

        Logic: Use <thead>, <tbody>, <tfoot> elements to identify
               header, body, and footer, otherwise:
               - Put all rows into body
               - Move rows from top of body to header only if
                 all elements inside row are <th>
               - Move rows from bottom of body to footer only if
                 all elements inside row are <th>
        """
        header_rows = self._parse_thead_tr(table_html)
        body_rows = self._parse_tbody_tr(table_html)
        footer_rows = self._parse_tfoot_tr(table_html)

        def row_is_all_th(row):
            return all(self._equals_tag(t, "th") for t in self._parse_td(row))

        if not header_rows:
            # The table has no <thead>. Move the top all-<th> rows from
            # body_rows to header_rows. (This is a common case because many
            # tables in the wild have no <thead> or <tfoot>
            while body_rows and row_is_all_th(body_rows[0]):
                header_rows.append(body_rows.pop(0))

        header = self._expand_colspan_rowspan(header_rows, section="header")
        body = self._expand_colspan_rowspan(body_rows, section="body")
        footer = self._expand_colspan_rowspan(footer_rows, section="footer")

        return header, body, footer

    def _expand_colspan_rowspan(
        self, rows, section: Literal["header", "footer", "body"]
    ):
        """
        Given a list of <tr>s, return a list of text rows.

        Parameters
        ----------
        rows : list of node-like
            List of <tr>s
        section : the section that the rows belong to (header, body or footer).

        Returns
        -------
        list of list
            Each returned row is a list of str text, or tuple (text, link)
            if extract_links is not None.

        Notes
        -----
        Any cell with ``rowspan`` or ``colspan`` will have its contents copied
        to subsequent cells.
        """
        all_texts = []  # list of rows, each a list of str
        text: str | tuple
        remainder: list[
            tuple[int, str | tuple, int]
        ] = []  # list of (index, text, nrows)

        for tr in rows:
            texts = []  # the output for this row
            next_remainder = []

            index = 0
            tds = self._parse_td(tr)
            for td in tds:
                # Append texts from previous rows with rowspan>1 that come
                # before this <td>
                while remainder and remainder[0][0] <= index:
                    prev_i, prev_text, prev_rowspan = remainder.pop(0)
                    texts.append(prev_text)
                    if prev_rowspan > 1:
                        next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
                    index += 1

                # Append the text from this <td>, colspan times
                text = _remove_whitespace(self._text_getter(td))
                if self.extract_links in ("all", section):
                    href = self._href_getter(td)
                    text = (text, href)
                rowspan = int(self._attr_getter(td, "rowspan") or 1)
                colspan = int(self._attr_getter(td, "colspan") or 1)

                for _ in range(colspan):
                    texts.append(text)
                    if rowspan > 1:
                        next_remainder.append((index, text, rowspan - 1))
                    index += 1

            # Append texts from previous rows at the final position
            for prev_i, prev_text, prev_rowspan in remainder:
                texts.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))

            all_texts.append(texts)
            remainder = next_remainder

        # Append rows that only appear because the previous row had non-1
        # rowspan
        while remainder:
            next_remainder = []
            texts = []
            for prev_i, prev_text, prev_rowspan in remainder:
                texts.append(prev_text)
                if prev_rowspan > 1:
                    next_remainder.append((prev_i, prev_text, prev_rowspan - 1))
            all_texts.append(texts)
            remainder = next_remainder

        return all_texts

    def _handle_hidden_tables(self, tbl_list, attr_name: str):
        """
        Return list of tables, potentially removing hidden elements

        Parameters
        ----------
        tbl_list : list of node-like
            Type of list elements will vary depending upon parser used
        attr_name : str
            Name of the accessor for retrieving HTML attributes

        Returns
        -------
        list of node-like
            Return type matches `tbl_list`
        """
        if not self.displayed_only:
            return tbl_list

        return [
            x
            for x in tbl_list
            if "display:none"
            not in getattr(x, attr_name).get("style", "").replace(" ", "")
        ]


class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses BeautifulSoup under the hood.

    See Also
    --------
    pandas.io.html._HtmlFrameParser
    pandas.io.html._LxmlFrameParser

    Notes
    -----
    Documentation strings for this class are in the base class
    :class:`pandas.io.html._HtmlFrameParser`.
    """

    def _parse_tables(self, document, match, attrs):
        element_name = "table"
        tables = document.find_all(element_name, attrs=attrs)
        if not tables:
            raise ValueError("No tables found")

        result = []
        unique_tables = set()
        tables = self._handle_hidden_tables(tables, "attrs")

        for table in tables:
            if self.displayed_only:
                for elem in table.find_all("style"):
                    elem.decompose()

                for elem in table.find_all(style=re.compile(r"display:\s*none")):
                    elem.decompose()

            if table not in unique_tables and table.find(string=match) is not None:
                result.append(table)
            unique_tables.add(table)
        if not result:
            raise ValueError(f"No tables found matching pattern {repr(match.pattern)}")
        return result

    def _href_getter(self, obj) -> str | None:
        a = obj.find("a", href=True)
        return None if not a else a["href"]

    def _text_getter(self, obj):
        return obj.text

    def _equals_tag(self, obj, tag) -> bool:
        return obj.name == tag

    def _parse_td(self, row):
        return row.find_all(("td", "th"), recursive=False)

    def _parse_thead_tr(self, table):
        return table.select("thead tr")

    def _parse_tbody_tr(self, table):
        from_tbody = table.select("tbody tr")
        from_root = table.find_all("tr", recursive=False)
        # HTML spec: at most one of these lists has content
        return from_tbody + from_root

    def _parse_tfoot_tr(self, table):
        return table.select("tfoot tr")

    def _setup_build_doc(self):
        raw_text = _read(self.io, self.encoding, self.storage_options)
        if not raw_text:
            raise ValueError(f"No text parsed from document: {self.io}")
        return raw_text

    def _build_doc(self):
        from bs4 import BeautifulSoup

        bdoc = self._setup_build_doc()
        if isinstance(bdoc, bytes) and self.encoding is not None:
            udoc = bdoc.decode(self.encoding)
            from_encoding = None
        else:
            udoc = bdoc
            from_encoding = self.encoding

        soup = BeautifulSoup(udoc, features="html5lib", from_encoding=from_encoding)

        for br in soup.find_all("br"):
            br.replace_with("\n" + br.text)

        return soup


def _build_xpath_expr(attrs) -> str:
    """
    Build an xpath expression to simulate bs4's ability to pass in kwargs to
    search for attributes when using the lxml parser.

    Parameters
    ----------
    attrs : dict
        A dict of HTML attributes. These are NOT checked for validity.

    Returns
    -------
    expr : unicode
        An XPath expression that checks for the given HTML attributes.
    """
    # give class attribute as class_ because class is a python keyword
    if "class_" in attrs:
        attrs["class"] = attrs.pop("class_")

    s = " and ".join([f"@{k}={repr(v)}" for k, v in attrs.items()])
    return f"[{s}]"


_re_namespace = {"re": "http://exslt.org/regular-expressions"}


class _LxmlFrameParser(_HtmlFrameParser):
    """
    HTML to DataFrame parser that uses lxml under the hood.

    Warning
    -------
    This parser can only handle HTTP, FTP, and FILE urls.

    See Also
    --------
    _HtmlFrameParser
    _BeautifulSoupLxmlFrameParser

    Notes
    -----
    Documentation strings for this class are in the base class
    :class:`_HtmlFrameParser`.
    """

    def _href_getter(self, obj) -> str | None:
        href = obj.xpath(".//a/@href")
        return None if not href else href[0]

    def _text_getter(self, obj):
        return obj.text_content()

    def _parse_td(self, row):
        # Look for direct children only: the "row" element here may be a
        # <thead> or <tfoot> (see _parse_thead_tr).
        return row.xpath("./td|./th")

    def _parse_tables(self, document, match, kwargs):
        pattern = match.pattern

        # 1. check all descendants for the given pattern and only search tables
        # GH 49929
        xpath_expr = f"//table[.//text()[re:test(., {repr(pattern)})]]"

        # if any table attributes were given build an xpath expression to
        # search for them
        if kwargs:
            xpath_expr += _build_xpath_expr(kwargs)

        tables = document.xpath(xpath_expr, namespaces=_re_namespace)

        tables = self._handle_hidden_tables(tables, "attrib")
        if self.displayed_only:
            for table in tables:
                # lxml utilizes XPATH 1.0 which does not have regex
                # support. As a result, we find all elements with a style
                # attribute and iterate them to check for display:none
                for elem in table.xpath(".//style"):
                    elem.drop_tree()
                for elem in table.xpath(".//*[@style]"):
                    if "display:none" in elem.attrib.get("style", "").replace(" ", ""):
                        elem.drop_tree()
        if not tables:
            raise ValueError(f"No tables found matching regex {repr(pattern)}")
        return tables

    def _equals_tag(self, obj, tag) -> bool:
        return obj.tag == tag

    def _build_doc(self):
        """
        Raises
        ------
        ValueError
            * If a URL that lxml cannot parse is passed.

        Exception
            * Any other ``Exception`` thrown. For example, trying to parse a
              URL that is syntactically correct on a machine with no internet
              connection will fail.

        See Also
        --------
        pandas.io.html._HtmlFrameParser._build_doc
        """
        from lxml.etree import XMLSyntaxError
        from lxml.html import (
            HTMLParser,
            fromstring,
            parse,
        )

        parser = HTMLParser(recover=True, encoding=self.encoding)

        try:
            if is_url(self.io):
                with get_handle(
                    self.io, "r", storage_options=self.storage_options
                ) as f:
                    r = parse(f.handle, parser=parser)
            else:
                # try to parse the input in the simplest way
                r = parse(self.io, parser=parser)
            try:
                r = r.getroot()
            except AttributeError:
                pass
        except (UnicodeDecodeError, OSError) as e:
            # if the input is a blob of html goop
            if not is_url(self.io):
                r = fromstring(self.io, parser=parser)

                try:
                    r = r.getroot()
                except AttributeError:
                    pass
            else:
                raise e
        else:
            if not hasattr(r, "text_content"):
                raise XMLSyntaxError("no text parsed from document", 0, 0, 0)

        for br in r.xpath("*//br"):
            br.tail = "\n" + (br.tail or "")

        return r

    def _parse_thead_tr(self, table):
        rows = []

        for thead in table.xpath(".//thead"):
            rows.extend(thead.xpath("./tr"))

            # HACK: lxml does not clean up the clearly-erroneous
            # <thead><th>foo</th><th>bar</th></thead>. (Missing <tr>). Add
            # the <thead> and _pretend_ it's a <tr>; _parse_td() will find its
            # children as though it's a <tr>.
            #
            # Better solution would be to use html5lib.
            elements_at_root = thead.xpath("./td|./th")
            if elements_at_root:
                rows.append(thead)

        return rows

    def _parse_tbody_tr(self, table):
        from_tbody = table.xpath(".//tbody//tr")
        from_root = table.xpath("./tr")
        # HTML spec: at most one of these lists has content
        return from_tbody + from_root

    def _parse_tfoot_tr(self, table):
        return table.xpath(".//tfoot//tr")


def _expand_elements(body) -> None:
    data = [len(elem) for elem in body]
    lens = Series(data)
    lens_max = lens.max()
    not_max = lens[lens != lens_max]

    empty = [""]
    for ind, length in not_max.items():
        body[ind] += empty * (lens_max - length)


def _data_to_frame(**kwargs):
    head, body, foot = kwargs.pop("data")
    header = kwargs.pop("header")
    kwargs["skiprows"] = _get_skiprows(kwargs["skiprows"])
    if head:
        body = head + body

        # Infer header when there is a <thead> or top <th>-only rows
        if header is None:
            if len(head) == 1:
                header = 0
            else:
                # ignore all-empty-text rows
                header = [i for i, row in enumerate(head) if any(text for text in row)]

    if foot:
        body += foot

    # fill out elements of body that are "ragged"
    _expand_elements(body)
    with TextParser(body, header=header, **kwargs) as tp:
        return tp.read()


_valid_parsers = {
    "lxml": _LxmlFrameParser,
    None: _LxmlFrameParser,
    "html5lib": _BeautifulSoupHtml5LibFrameParser,
    "bs4": _BeautifulSoupHtml5LibFrameParser,
}


def _parser_dispatch(flavor: HTMLFlavors | None) -> type[_HtmlFrameParser]:
    """
    Choose the parser based on the input flavor.

    Parameters
    ----------
    flavor : {{"lxml", "html5lib", "bs4"}} or None
        The type of parser to use. This must be a valid backend.

    Returns
    -------
    cls : _HtmlFrameParser subclass
        The parser class based on the requested input flavor.

    Raises
    ------
    ValueError
        * If `flavor` is not a valid backend.
    ImportError
        * If you do not have the requested `flavor`
    """
    valid_parsers = list(_valid_parsers.keys())
    if flavor not in valid_parsers:
        raise ValueError(
            f"{repr(flavor)} is not a valid flavor, valid flavors are {valid_parsers}"
        )

    if flavor in ("bs4", "html5lib"):
        import_optional_dependency("html5lib")
        import_optional_dependency("bs4")
    else:
        import_optional_dependency("lxml.etree")
    return _valid_parsers[flavor]


def _print_as_set(s) -> str:
    arg = ", ".join([pprint_thing(el) for el in s])
    return f"{{{arg}}}"


def _validate_flavor(flavor):
    if flavor is None:
        flavor = "lxml", "bs4"
    elif isinstance(flavor, str):
        flavor = (flavor,)
    elif isinstance(flavor, abc.Iterable):
        if not all(isinstance(flav, str) for flav in flavor):
            raise TypeError(
                f"Object of type {repr(type(flavor).__name__)} "
                f"is not an iterable of strings"
            )
    else:
        msg = repr(flavor) if isinstance(flavor, str) else str(flavor)
        msg += " is not a valid flavor"
        raise ValueError(msg)

    flavor = tuple(flavor)
    valid_flavors = set(_valid_parsers)
    flavor_set = set(flavor)

    if not flavor_set & valid_flavors:
        raise ValueError(
            f"{_print_as_set(flavor_set)} is not a valid set of flavors, valid "
            f"flavors are {_print_as_set(valid_flavors)}"
        )
    return flavor


def _parse(
    flavor,
    io,
    match,
    attrs,
    encoding,
    displayed_only,
    extract_links,
    storage_options,
    **kwargs,
):
    flavor = _validate_flavor(flavor)
    compiled_match = re.compile(match)  # you can pass a compiled regex here

    retained = None
    for flav in flavor:
        parser = _parser_dispatch(flav)
        p = parser(
            io,
            compiled_match,
            attrs,
            encoding,
            displayed_only,
            extract_links,
            storage_options,
        )

        try:
            tables = p.parse_tables()
        except ValueError as caught:
            # if `io` is an io-like object, check if it's seekable
            # and try to rewind it before trying the next parser
            if hasattr(io, "seekable") and io.seekable():
                io.seek(0)
            elif hasattr(io, "seekable") and not io.seekable():
                # if we couldn't rewind it, let the user know
                raise ValueError(
                    f"The flavor {flav} failed to parse your input. "
                    "Since you passed a non-rewindable file "
                    "object, we can't rewind it to try "
                    "another parser. Try read_html() with a different flavor."
                ) from caught

            retained = caught
        else:
            break
    else:
        assert retained is not None  # for mypy
        raise retained

    ret = []
    for table in tables:
        try:
            df = _data_to_frame(data=table, **kwargs)
            # Cast MultiIndex header to an Index of tuples when extracting header
            # links and replace nan with None (therefore can't use mi.to_flat_index()).
            # This maintains consistency of selection (e.g. df.columns.str[1])
            if extract_links in ("all", "header") and isinstance(
                df.columns, MultiIndex
            ):
                df.columns = Index(
                    ((col[0], None if isna(col[1]) else col[1]) for col in df.columns),
                    tupleize_cols=False,
                )

            ret.append(df)
        except EmptyDataError:  # empty table
            continue
    return ret


@doc(storage_options=_shared_docs["storage_options"])
def read_html(
    io: FilePath | ReadBuffer[str],
    *,
    match: str | Pattern = ".+",
    flavor: HTMLFlavors | Sequence[HTMLFlavors] | None = None,
    header: int | Sequence[int] | None = None,
    index_col: int | Sequence[int] | None = None,
    skiprows: int | Sequence[int] | slice | None = None,
    attrs: dict[str, str] | None = None,
    parse_dates: bool = False,
    thousands: str | None = ",",
    encoding: str | None = None,
    decimal: str = ".",
    converters: dict | None = None,
    na_values: Iterable[object] | None = None,
    keep_default_na: bool = True,
    displayed_only: bool = True,
    extract_links: Literal[None, "header", "footer", "body", "all"] = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    storage_options: StorageOptions = None,
) -> list[DataFrame]:
    r"""
    Read HTML tables into a ``list`` of ``DataFrame`` objects.

    Parameters
    ----------
    io : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a string ``read()`` function.
        The string can represent a URL or the HTML itself. Note that
        lxml only accepts the http, ftp and file url protocols. If you have a
        URL that starts with ``'https'`` you might try removing the ``'s'``.

        .. deprecated:: 2.1.0
            Passing html literal strings is deprecated.
            Wrap literal string/bytes input in ``io.StringIO``/``io.BytesIO`` instead.

    match : str or compiled regular expression, optional
        The set of tables containing text matching this regex or string will be
        returned. Unless the HTML is extremely simple you will probably need to
        pass a non-empty string here. Defaults to '.+' (match any non-empty
        string). The default value will return all tables contained on a page.
        This value is converted to a regular expression so that there is
        consistent behavior between Beautiful Soup and lxml.

    flavor : {{"lxml", "html5lib", "bs4"}} or list-like, optional
        The parsing engine (or list of parsing engines) to use. 'bs4' and
        'html5lib' are synonymous with each other, they are both there for
        backwards compatibility. The default of ``None`` tries to use ``lxml``
        to parse and if that fails it falls back on ``bs4`` + ``html5lib``.

    header : int or list-like, optional
        The row (or list of rows for a :class:`~pandas.MultiIndex`) to use to
        make the columns headers.

    index_col : int or list-like, optional
        The column (or list of columns) to use to create the index.

    skiprows : int, list-like or slice, optional
        Number of rows to skip after parsing the column integer. 0-based. If a
        sequence of integers or a slice is given, will skip the rows indexed by
        that sequence.  Note that a single element sequence means 'skip the nth
        row' whereas an integer means 'skip n rows'.

    attrs : dict, optional
        This is a dictionary of attributes that you can pass to use to identify
        the table in the HTML. These are not checked for validity before being
        passed to lxml or Beautiful Soup. However, these attributes must be
        valid HTML table attributes to work correctly. For example, ::

            attrs = {{'id': 'table'}}

        is a valid attribute dictionary because the 'id' HTML tag attribute is
        a valid HTML attribute for *any* HTML tag as per `this document
        <https://html.spec.whatwg.org/multipage/dom.html#global-attributes>`__. ::

            attrs = {{'asdf': 'table'}}

        is *not* a valid attribute dictionary because 'asdf' is not a valid
        HTML attribute even if it is a valid XML attribute.  Valid HTML 4.01
        table attributes can be found `here
        <http://www.w3.org/TR/REC-html40/struct/tables.html#h-11.2>`__. A
        working draft of the HTML 5 spec can be found `here
        <https://html.spec.whatwg.org/multipage/tables.html>`__. It contains the
        latest information on table attributes for the modern web.

    parse_dates : bool, optional
        See :func:`~read_csv` for more details.

    thousands : str, optional
        Separator to use to parse thousands. Defaults to ``','``.

    encoding : str, optional
        The encoding used to decode the web page. Defaults to ``None``.``None``
        preserves the previous encoding behavior, which depends on the
        underlying parser library (e.g., the parser library will try to use
        the encoding provided by the document).

    decimal : str, default '.'
        Character to recognize as decimal point (e.g. use ',' for European
        data).

    converters : dict, default None
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the cell (not column) content, and return the
        transformed content.

    na_values : iterable, default None
        Custom NA values.

    keep_default_na : bool, default True
        If na_values are specified and keep_default_na is False the default NaN
        values are overridden, otherwise they're appended to.

    displayed_only : bool, default True
        Whether elements with "display: none" should be parsed.

    extract_links : {{None, "all", "header", "body", "footer"}}
        Table elements in the specified section(s) with <a> tags will have their
        href extracted.

        .. versionadded:: 1.5.0

    dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    {storage_options}

        .. versionadded:: 2.1.0

    Returns
    -------
    dfs
        A list of DataFrames.

    See Also
    --------
    read_csv : Read a comma-separated values (csv) file into DataFrame.

    Notes
    -----
    Before using this function you should read the :ref:`gotchas about the
    HTML parsing libraries <io.html.gotchas>`.

    Expect to do some cleanup after you call this function. For example, you
    might need to manually assign column names if the column names are
    converted to NaN when you pass the `header=0` argument. We try to assume as
    little as possible about the structure of the table and push the
    idiosyncrasies of the HTML contained in the table to the user.

    This function searches for ``<table>`` elements and only for ``<tr>``
    and ``<th>`` rows and ``<td>`` elements within each ``<tr>`` or ``<th>``
    element in the table. ``<td>`` stands for "table data". This function
    attempts to properly handle ``colspan`` and ``rowspan`` attributes.
    If the function has a ``<thead>`` argument, it is used to construct
    the header, otherwise the function attempts to find the header within
    the body (by putting rows with only ``<th>`` elements into the header).

    Similar to :func:`~read_csv` the `header` argument is applied
    **after** `skiprows` is applied.

    This function will *always* return a list of :class:`DataFrame` *or*
    it will fail, e.g., it will *not* return an empty list.

    Examples
    --------
    See the :ref:`read_html documentation in the IO section of the docs
    <io.read_html>` for some examples of reading in HTML tables.
    """
    # Type check here. We don't want to parse only to fail because of an
    # invalid value of an integer skiprows.
    if isinstance(skiprows, numbers.Integral) and skiprows < 0:
        raise ValueError(
            "cannot skip rows starting from the end of the "
            "data (you passed a negative value)"
        )
    if extract_links not in [None, "header", "footer", "body", "all"]:
        raise ValueError(
            "`extract_links` must be one of "
            '{None, "header", "footer", "body", "all"}, got '
            f'"{extract_links}"'
        )

    validate_header_arg(header)
    check_dtype_backend(dtype_backend)

    io = stringify_path(io)

    if isinstance(io, str) and not any(
        [
            is_file_like(io),
            file_exists(io),
            is_url(io),
            is_fsspec_url(io),
        ]
    ):
        warnings.warn(
            "Passing literal html to 'read_html' is deprecated and "
            "will be removed in a future version. To read from a "
            "literal string, wrap it in a 'StringIO' object.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    return _parse(
        flavor=flavor,
        io=io,
        match=match,
        header=header,
        index_col=index_col,
        skiprows=skiprows,
        parse_dates=parse_dates,
        thousands=thousands,
        attrs=attrs,
        encoding=encoding,
        decimal=decimal,
        converters=converters,
        na_values=na_values,
        keep_default_na=keep_default_na,
        displayed_only=displayed_only,
        extract_links=extract_links,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\coset_table.py ===
from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting

from itertools import chain, product
from bisect import bisect_left


###############################################################################
#                           COSET TABLE                                       #
###############################################################################

class CosetTable(DefaultPrinting):
    # coset_table: Mathematically a coset table
    #               represented using a list of lists
    # alpha: Mathematically a coset (precisely, a live coset)
    #       represented by an integer between i with 1 <= i <= n
    #       alpha in c
    # x: Mathematically an element of "A" (set of generators and
    #   their inverses), represented using "FpGroupElement"
    # fp_grp: Finitely Presented Group with < X|R > as presentation.
    # H: subgroup of fp_grp.
    # NOTE: We start with H as being only a list of words in generators
    #       of "fp_grp". Since `.subgroup` method has not been implemented.

    r"""

    Properties
    ==========

    [1] `0 \in \Omega` and `\tau(1) = \epsilon`
    [2] `\alpha^x = \beta \Leftrightarrow \beta^{x^{-1}} = \alpha`
    [3] If `\alpha^x = \beta`, then `H \tau(\alpha)x = H \tau(\beta)`
    [4] `\forall \alpha \in \Omega, 1^{\tau(\alpha)} = \alpha`

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of Computational Group Theory"

    .. [2] John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
           Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490.
           "Implementation and Analysis of the Todd-Coxeter Algorithm"

    """
    # default limit for the number of cosets allowed in a
    # coset enumeration.
    coset_table_max_limit = 4096000
    # limit for the current instance
    coset_table_limit = None
    # maximum size of deduction stack above or equal to
    # which it is emptied
    max_stack_size = 100

    def __init__(self, fp_grp, subgroup, max_cosets=None):
        if not max_cosets:
            max_cosets = CosetTable.coset_table_max_limit
        self.fp_group = fp_grp
        self.subgroup = subgroup
        self.coset_table_limit = max_cosets
        # "p" is setup independent of Omega and n
        self.p = [0]
        # a list of the form `[gen_1, gen_1^{-1}, ... , gen_k, gen_k^{-1}]`
        self.A = list(chain.from_iterable((gen, gen**-1) \
                for gen in self.fp_group.generators))
        #P[alpha, x] Only defined when alpha^x is defined.
        self.P = [[None]*len(self.A)]
        # the mathematical coset table which is a list of lists
        self.table = [[None]*len(self.A)]
        self.A_dict = {x: self.A.index(x) for x in self.A}
        self.A_dict_inv = {}
        for x, index in self.A_dict.items():
            if index % 2 == 0:
                self.A_dict_inv[x] = self.A_dict[x] + 1
            else:
                self.A_dict_inv[x] = self.A_dict[x] - 1
        # used in the coset-table based method of coset enumeration. Each of
        # the element is called a "deduction" which is the form (alpha, x) whenever
        # a value is assigned to alpha^x during a definition or "deduction process"
        self.deduction_stack = []
        # Attributes for modified methods.
        H = self.subgroup
        self._grp = free_group(', ' .join(["a_%d" % i for i in range(len(H))]))[0]
        self.P = [[None]*len(self.A)]
        self.p_p = {}

    @property
    def omega(self):
        """Set of live cosets. """
        return [coset for coset in range(len(self.p)) if self.p[coset] == coset]

    def copy(self):
        """
        Return a shallow copy of Coset Table instance ``self``.

        """
        self_copy = self.__class__(self.fp_group, self.subgroup)
        self_copy.table = [list(perm_rep) for perm_rep in self.table]
        self_copy.p = list(self.p)
        self_copy.deduction_stack = list(self.deduction_stack)
        return self_copy

    def __str__(self):
        return "Coset Table on %s with %s as subgroup generators" \
                % (self.fp_group, self.subgroup)

    __repr__ = __str__

    @property
    def n(self):
        """The number `n` represents the length of the sublist containing the
        live cosets.

        """
        if not self.table:
            return 0
        return max(self.omega) + 1

    # Pg. 152 [1]
    def is_complete(self):
        r"""
        The coset table is called complete if it has no undefined entries
        on the live cosets; that is, `\alpha^x` is defined for all
        `\alpha \in \Omega` and `x \in A`.

        """
        return not any(None in self.table[coset] for coset in self.omega)

    # Pg. 153 [1]
    def define(self, alpha, x, modified=False):
        r"""
        This routine is used in the relator-based strategy of Todd-Coxeter
        algorithm if some `\alpha^x` is undefined. We check whether there is
        space available for defining a new coset. If there is enough space
        then we remedy this by adjoining a new coset `\beta` to `\Omega`
        (i.e to set of live cosets) and put that equal to `\alpha^x`, then
        make an assignment satisfying Property[1]. If there is not enough space
        then we halt the Coset Table creation. The maximum amount of space that
        can be used by Coset Table can be manipulated using the class variable
        ``CosetTable.coset_table_max_limit``.

        See Also
        ========

        define_c

        """
        A = self.A
        table = self.table
        len_table = len(table)
        if len_table >= self.coset_table_limit:
            # abort the further generation of cosets
            raise ValueError("the coset enumeration has defined more than "
                    "%s cosets. Try with a greater value max number of cosets "
                    % self.coset_table_limit)
        table.append([None]*len(A))
        self.P.append([None]*len(self.A))
        # beta is the new coset generated
        beta = len_table
        self.p.append(beta)
        table[alpha][self.A_dict[x]] = beta
        table[beta][self.A_dict_inv[x]] = alpha
        # P[alpha][x] = epsilon, P[beta][x**-1] = epsilon
        if modified:
            self.P[alpha][self.A_dict[x]] = self._grp.identity
            self.P[beta][self.A_dict_inv[x]] = self._grp.identity
            self.p_p[beta] = self._grp.identity

    def define_c(self, alpha, x):
        r"""
        A variation of ``define`` routine, described on Pg. 165 [1], used in
        the coset table-based strategy of Todd-Coxeter algorithm. It differs
        from ``define`` routine in that for each definition it also adds the
        tuple `(\alpha, x)` to the deduction stack.

        See Also
        ========

        define

        """
        A = self.A
        table = self.table
        len_table = len(table)
        if len_table >= self.coset_table_limit:
            # abort the further generation of cosets
            raise ValueError("the coset enumeration has defined more than "
                    "%s cosets. Try with a greater value max number of cosets "
                    % self.coset_table_limit)
        table.append([None]*len(A))
        # beta is the new coset generated
        beta = len_table
        self.p.append(beta)
        table[alpha][self.A_dict[x]] = beta
        table[beta][self.A_dict_inv[x]] = alpha
        # append to deduction stack
        self.deduction_stack.append((alpha, x))

    def scan_c(self, alpha, word):
        """
        A variation of ``scan`` routine, described on pg. 165 of [1], which
        puts at tuple, whenever a deduction occurs, to deduction stack.

        See Also
        ========

        scan, scan_check, scan_and_fill, scan_and_fill_c

        """
        # alpha is an integer representing a "coset"
        # since scanning can be in two cases
        # 1. for alpha=0 and w in Y (i.e generating set of H)
        # 2. alpha in Omega (set of live cosets), w in R (relators)
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        # list of union of generators and their inverses
        while i <= j and table[f][A_dict[word[i]]] is not None:
            f = table[f][A_dict[word[i]]]
            i += 1
        if i > j:
            if f != b:
                self.coincidence_c(f, b)
            return
        while j >= i and table[b][A_dict_inv[word[j]]] is not None:
            b = table[b][A_dict_inv[word[j]]]
            j -= 1
        if j < i:
            # we have an incorrect completed scan with coincidence f ~ b
            # run the "coincidence" routine
            self.coincidence_c(f, b)
        elif j == i:
            # deduction process
            table[f][A_dict[word[i]]] = b
            table[b][A_dict_inv[word[i]]] = f
            self.deduction_stack.append((f, word[i]))
        # otherwise scan is incomplete and yields no information

    # alpha, beta coincide, i.e. alpha, beta represent the pair of cosets where
    # coincidence occurs
    def coincidence_c(self, alpha, beta):
        """
        A variation of ``coincidence`` routine used in the coset-table based
        method of coset enumeration. The only difference being on addition of
        a new coset in coset table(i.e new coset introduction), then it is
        appended to ``deduction_stack``.

        See Also
        ========

        coincidence

        """
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        # behaves as a queue
        q = []
        self.merge(alpha, beta, q)
        while len(q) > 0:
            gamma = q.pop(0)
            for x in A_dict:
                delta = table[gamma][A_dict[x]]
                if delta is not None:
                    table[delta][A_dict_inv[x]] = None
                    # only line of difference from ``coincidence`` routine
                    self.deduction_stack.append((delta, x**-1))
                    mu = self.rep(gamma)
                    nu = self.rep(delta)
                    if table[mu][A_dict[x]] is not None:
                        self.merge(nu, table[mu][A_dict[x]], q)
                    elif table[nu][A_dict_inv[x]] is not None:
                        self.merge(mu, table[nu][A_dict_inv[x]], q)
                    else:
                        table[mu][A_dict[x]] = nu
                        table[nu][A_dict_inv[x]] = mu

    def scan(self, alpha, word, y=None, fill=False, modified=False):
        r"""
        ``scan`` performs a scanning process on the input ``word``.
        It first locates the largest prefix ``s`` of ``word`` for which
        `\alpha^s` is defined (i.e is not ``None``), ``s`` may be empty. Let
        ``word=sv``, let ``t`` be the longest suffix of ``v`` for which
        `\alpha^{t^{-1}}` is defined, and let ``v=ut``. Then three
        possibilities are there:

        1. If ``t=v``, then we say that the scan completes, and if, in addition
        `\alpha^s = \alpha^{t^{-1}}`, then we say that the scan completes
        correctly.

        2. It can also happen that scan does not complete, but `|u|=1`; that
        is, the word ``u`` consists of a single generator `x \in A`. In that
        case, if `\alpha^s = \beta` and `\alpha^{t^{-1}} = \gamma`, then we can
        set `\beta^x = \gamma` and `\gamma^{x^{-1}} = \beta`. These assignments
        are known as deductions and enable the scan to complete correctly.

        3. See ``coicidence`` routine for explanation of third condition.

        Notes
        =====

        The code for the procedure of scanning `\alpha \in \Omega`
        under `w \in A*` is defined on pg. 155 [1]

        See Also
        ========

        scan_c, scan_check, scan_and_fill, scan_and_fill_c

        Scan and Fill
        =============

        Performed when the default argument fill=True.

        Modified Scan
        =============

        Performed when the default argument modified=True

        """
        # alpha is an integer representing a "coset"
        # since scanning can be in two cases
        # 1. for alpha=0 and w in Y (i.e generating set of H)
        # 2. alpha in Omega (set of live cosets), w in R (relators)
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        b_p = y
        if modified:
            f_p = self._grp.identity
        flag = 0
        while fill or flag == 0:
            flag = 1
            while i <= j and table[f][A_dict[word[i]]] is not None:
                if modified:
                    f_p = f_p*self.P[f][A_dict[word[i]]]
                f = table[f][A_dict[word[i]]]
                i += 1
            if i > j:
                if f != b:
                    if modified:
                        self.modified_coincidence(f, b, f_p**-1*y)
                    else:
                        self.coincidence(f, b)
                return
            while j >= i and table[b][A_dict_inv[word[j]]] is not None:
                if modified:
                    b_p = b_p*self.P[b][self.A_dict_inv[word[j]]]
                b = table[b][A_dict_inv[word[j]]]
                j -= 1
            if j < i:
                # we have an incorrect completed scan with coincidence f ~ b
                # run the "coincidence" routine
                if modified:
                    self.modified_coincidence(f, b, f_p**-1*b_p)
                else:
                    self.coincidence(f, b)
            elif j == i:
                # deduction process
                table[f][A_dict[word[i]]] = b
                table[b][A_dict_inv[word[i]]] = f
                if modified:
                    self.P[f][self.A_dict[word[i]]] = f_p**-1*b_p
                    self.P[b][self.A_dict_inv[word[i]]] = b_p**-1*f_p
                return
            elif fill:
                self.define(f, word[i], modified=modified)
            # otherwise scan is incomplete and yields no information

    # used in the low-index subgroups algorithm
    def scan_check(self, alpha, word):
        r"""
        Another version of ``scan`` routine, described on, it checks whether
        `\alpha` scans correctly under `word`, it is a straightforward
        modification of ``scan``. ``scan_check`` returns ``False`` (rather than
        calling ``coincidence``) if the scan completes incorrectly; otherwise
        it returns ``True``.

        See Also
        ========

        scan, scan_c, scan_and_fill, scan_and_fill_c

        """
        # alpha is an integer representing a "coset"
        # since scanning can be in two cases
        # 1. for alpha=0 and w in Y (i.e generating set of H)
        # 2. alpha in Omega (set of live cosets), w in R (relators)
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        f = alpha
        i = 0
        r = len(word)
        b = alpha
        j = r - 1
        while i <= j and table[f][A_dict[word[i]]] is not None:
            f = table[f][A_dict[word[i]]]
            i += 1
        if i > j:
            return f == b
        while j >= i and table[b][A_dict_inv[word[j]]] is not None:
            b = table[b][A_dict_inv[word[j]]]
            j -= 1
        if j < i:
            # we have an incorrect completed scan with coincidence f ~ b
            # return False, instead of calling coincidence routine
            return False
        elif j == i:
            # deduction process
            table[f][A_dict[word[i]]] = b
            table[b][A_dict_inv[word[i]]] = f
        return True

    def merge(self, k, lamda, q, w=None, modified=False):
        """
        Merge two classes with representatives ``k`` and ``lamda``, described
        on Pg. 157 [1] (for pseudocode), start by putting ``p[k] = lamda``.
        It is more efficient to choose the new representative from the larger
        of the two classes being merged, i.e larger among ``k`` and ``lamda``.
        procedure ``merge`` performs the merging operation, adds the deleted
        class representative to the queue ``q``.

        Parameters
        ==========

        'k', 'lamda' being the two class representatives to be merged.

        Notes
        =====

        Pg. 86-87 [1] contains a description of this method.

        See Also
        ========

        coincidence, rep

        """
        p = self.p
        rep = self.rep
        phi = rep(k, modified=modified)
        psi = rep(lamda, modified=modified)
        if phi != psi:
            mu = min(phi, psi)
            v = max(phi, psi)
            p[v] = mu
            if modified:
                if v == phi:
                    self.p_p[phi] = self.p_p[k]**-1*w*self.p_p[lamda]
                else:
                    self.p_p[psi] = self.p_p[lamda]**-1*w**-1*self.p_p[k]
            q.append(v)

    def rep(self, k, modified=False):
        r"""
        Parameters
        ==========

        `k \in [0 \ldots n-1]`, as for ``self`` only array ``p`` is used

        Returns
        =======

        Representative of the class containing ``k``.

        Returns the representative of `\sim` class containing ``k``, it also
        makes some modification to array ``p`` of ``self`` to ease further
        computations, described on Pg. 157 [1].

        The information on classes under `\sim` is stored in array `p` of
        ``self`` argument, which will always satisfy the property:

        `p[\alpha] \sim \alpha` and `p[\alpha]=\alpha \iff \alpha=rep(\alpha)`
        `\forall \in [0 \ldots n-1]`.

        So, for `\alpha \in [0 \ldots n-1]`, we find `rep(self, \alpha)` by
        continually replacing `\alpha` by `p[\alpha]` until it becomes
        constant (i.e satisfies `p[\alpha] = \alpha`):w

        To increase the efficiency of later ``rep`` calculations, whenever we
        find `rep(self, \alpha)=\beta`, we set
        `p[\gamma] = \beta \forall \gamma \in p-chain` from `\alpha` to `\beta`

        Notes
        =====

        ``rep`` routine is also described on Pg. 85-87 [1] in Atkinson's
        algorithm, this results from the fact that ``coincidence`` routine
        introduces functionality similar to that introduced by the
        ``minimal_block`` routine on Pg. 85-87 [1].

        See Also
        ========

        coincidence, merge

        """
        p = self.p
        lamda = k
        rho = p[lamda]
        if modified:
            s = p[:]
        while rho != lamda:
            if modified:
                s[rho] = lamda
            lamda = rho
            rho = p[lamda]
        if modified:
            rho = s[lamda]
            while rho != k:
                mu = rho
                rho = s[mu]
                p[rho] = lamda
                self.p_p[rho] = self.p_p[rho]*self.p_p[mu]
        else:
            mu = k
            rho = p[mu]
            while rho != lamda:
                p[mu] = lamda
                mu = rho
                rho = p[mu]
        return lamda

    # alpha, beta coincide, i.e. alpha, beta represent the pair of cosets
    # where coincidence occurs
    def coincidence(self, alpha, beta, w=None, modified=False):
        r"""
        The third situation described in ``scan`` routine is handled by this
        routine, described on Pg. 156-161 [1].

        The unfortunate situation when the scan completes but not correctly,
        then ``coincidence`` routine is run. i.e when for some `i` with
        `1 \le i \le r+1`, we have `w=st` with `s = x_1 x_2 \dots x_{i-1}`,
        `t = x_i x_{i+1} \dots x_r`, and `\beta = \alpha^s` and
        `\gamma = \alpha^{t-1}` are defined but unequal. This means that
        `\beta` and `\gamma` represent the same coset of `H` in `G`. Described
        on Pg. 156 [1]. ``rep``

        See Also
        ========

        scan

        """
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        # behaves as a queue
        q = []
        if modified:
            self.modified_merge(alpha, beta, w, q)
        else:
            self.merge(alpha, beta, q)
        while len(q) > 0:
            gamma = q.pop(0)
            for x in A_dict:
                delta = table[gamma][A_dict[x]]
                if delta is not None:
                    table[delta][A_dict_inv[x]] = None
                    mu = self.rep(gamma, modified=modified)
                    nu = self.rep(delta, modified=modified)
                    if table[mu][A_dict[x]] is not None:
                        if modified:
                            v = self.p_p[delta]**-1*self.P[gamma][self.A_dict[x]]**-1
                            v = v*self.p_p[gamma]*self.P[mu][self.A_dict[x]]
                            self.modified_merge(nu, table[mu][self.A_dict[x]], v, q)
                        else:
                            self.merge(nu, table[mu][A_dict[x]], q)
                    elif table[nu][A_dict_inv[x]] is not None:
                        if modified:
                            v = self.p_p[gamma]**-1*self.P[gamma][self.A_dict[x]]
                            v = v*self.p_p[delta]*self.P[mu][self.A_dict_inv[x]]
                            self.modified_merge(mu, table[nu][self.A_dict_inv[x]], v, q)
                        else:
                            self.merge(mu, table[nu][A_dict_inv[x]], q)
                    else:
                        table[mu][A_dict[x]] = nu
                        table[nu][A_dict_inv[x]] = mu
                        if modified:
                            v = self.p_p[gamma]**-1*self.P[gamma][self.A_dict[x]]*self.p_p[delta]
                            self.P[mu][self.A_dict[x]] = v
                            self.P[nu][self.A_dict_inv[x]] = v**-1

    # method used in the HLT strategy
    def scan_and_fill(self, alpha, word):
        """
        A modified version of ``scan`` routine used in the relator-based
        method of coset enumeration, described on pg. 162-163 [1], which
        follows the idea that whenever the procedure is called and the scan
        is incomplete then it makes new definitions to enable the scan to
        complete; i.e it fills in the gaps in the scan of the relator or
        subgroup generator.

        """
        self.scan(alpha, word, fill=True)

    def scan_and_fill_c(self, alpha, word):
        """
        A modified version of ``scan`` routine, described on Pg. 165 second
        para. [1], with modification similar to that of ``scan_anf_fill`` the
        only difference being it calls the coincidence procedure used in the
        coset-table based method i.e. the routine ``coincidence_c`` is used.

        See Also
        ========

        scan, scan_and_fill

        """
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        r = len(word)
        f = alpha
        i = 0
        b = alpha
        j = r - 1
        # loop until it has filled the alpha row in the table.
        while True:
            # do the forward scanning
            while i <= j and table[f][A_dict[word[i]]] is not None:
                f = table[f][A_dict[word[i]]]
                i += 1
            if i > j:
                if f != b:
                    self.coincidence_c(f, b)
                return
            # forward scan was incomplete, scan backwards
            while j >= i and table[b][A_dict_inv[word[j]]] is not None:
                b = table[b][A_dict_inv[word[j]]]
                j -= 1
            if j < i:
                self.coincidence_c(f, b)
            elif j == i:
                table[f][A_dict[word[i]]] = b
                table[b][A_dict_inv[word[i]]] = f
                self.deduction_stack.append((f, word[i]))
            else:
                self.define_c(f, word[i])

    # method used in the HLT strategy
    def look_ahead(self):
        """
        When combined with the HLT method this is known as HLT+Lookahead
        method of coset enumeration, described on pg. 164 [1]. Whenever
        ``define`` aborts due to lack of space available this procedure is
        executed. This routine helps in recovering space resulting from
        "coincidence" of cosets.

        """
        R = self.fp_group.relators
        p = self.p
        # complete scan all relators under all cosets(obviously live)
        # without making new definitions
        for beta in self.omega:
            for w in R:
                self.scan(beta, w)
                if p[beta] < beta:
                    break

    # Pg. 166
    def process_deductions(self, R_c_x, R_c_x_inv):
        """
        Processes the deductions that have been pushed onto ``deduction_stack``,
        described on Pg. 166 [1] and is used in coset-table based enumeration.

        See Also
        ========

        deduction_stack

        """
        p = self.p
        table = self.table
        while len(self.deduction_stack) > 0:
            if len(self.deduction_stack) >= CosetTable.max_stack_size:
                self.look_ahead()
                del self.deduction_stack[:]
                continue
            else:
                alpha, x = self.deduction_stack.pop()
                if p[alpha] == alpha:
                    for w in R_c_x:
                        self.scan_c(alpha, w)
                        if p[alpha] < alpha:
                            break
            beta = table[alpha][self.A_dict[x]]
            if beta is not None and p[beta] == beta:
                for w in R_c_x_inv:
                    self.scan_c(beta, w)
                    if p[beta] < beta:
                        break

    def process_deductions_check(self, R_c_x, R_c_x_inv):
        """
        A variation of ``process_deductions``, this calls ``scan_check``
        wherever ``process_deductions`` calls ``scan``, described on Pg. [1].

        See Also
        ========

        process_deductions

        """
        table = self.table
        while len(self.deduction_stack) > 0:
            alpha, x = self.deduction_stack.pop()
            if not all(self.scan_check(alpha, w) for w in R_c_x):
                return False
            beta = table[alpha][self.A_dict[x]]
            if beta is not None:
                if not all(self.scan_check(beta, w) for w in R_c_x_inv):
                    return False
        return True

    def switch(self, beta, gamma):
        r"""Switch the elements `\beta, \gamma \in \Omega` of ``self``, used
        by the ``standardize`` procedure, described on Pg. 167 [1].

        See Also
        ========

        standardize

        """
        A = self.A
        A_dict = self.A_dict
        table = self.table
        for x in A:
            z = table[gamma][A_dict[x]]
            table[gamma][A_dict[x]] = table[beta][A_dict[x]]
            table[beta][A_dict[x]] = z
            for alpha in range(len(self.p)):
                if self.p[alpha] == alpha:
                    if table[alpha][A_dict[x]] == beta:
                        table[alpha][A_dict[x]] = gamma
                    elif table[alpha][A_dict[x]] == gamma:
                        table[alpha][A_dict[x]] = beta

    def standardize(self):
        r"""
        A coset table is standardized if when running through the cosets and
        within each coset through the generator images (ignoring generator
        inverses), the cosets appear in order of the integers
        `0, 1, \dots, n`. "Standardize" reorders the elements of `\Omega`
        such that, if we scan the coset table first by elements of `\Omega`
        and then by elements of A, then the cosets occur in ascending order.
        ``standardize()`` is used at the end of an enumeration to permute the
        cosets so that they occur in some sort of standard order.

        Notes
        =====

        procedure is described on pg. 167-168 [1], it also makes use of the
        ``switch`` routine to replace by smaller integer value.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
        >>> F, x, y = free_group("x, y")

        # Example 5.3 from [1]
        >>> f = FpGroup(F, [x**2*y**2, x**3*y**5])
        >>> C = coset_enumeration_r(f, [])
        >>> C.compress()
        >>> C.table
        [[1, 3, 1, 3], [2, 0, 2, 0], [3, 1, 3, 1], [0, 2, 0, 2]]
        >>> C.standardize()
        >>> C.table
        [[1, 2, 1, 2], [3, 0, 3, 0], [0, 3, 0, 3], [2, 1, 2, 1]]

        """
        A = self.A
        A_dict = self.A_dict
        gamma = 1
        for alpha, x in product(range(self.n), A):
            beta = self.table[alpha][A_dict[x]]
            if beta >= gamma:
                if beta > gamma:
                    self.switch(gamma, beta)
                gamma += 1
                if gamma == self.n:
                    return

    # Compression of a Coset Table
    def compress(self):
        """Removes the non-live cosets from the coset table, described on
        pg. 167 [1].

        """
        gamma = -1
        A = self.A
        A_dict = self.A_dict
        A_dict_inv = self.A_dict_inv
        table = self.table
        chi = tuple([i for i in range(len(self.p)) if self.p[i] != i])
        for alpha in self.omega:
            gamma += 1
            if gamma != alpha:
                # replace alpha by gamma in coset table
                for x in A:
                    beta = table[alpha][A_dict[x]]
                    table[gamma][A_dict[x]] = beta
                    # XXX: The line below uses == rather than = which means
                    # that it has no effect. It is not clear though if it is
                    # correct simply to delete the line or to change it to
                    # use =. Changing it causes some tests to fail.
                    #
                    # https://github.com/sympy/sympy/issues/27633
                    table[beta][A_dict_inv[x]] == gamma # noqa: B015
        # all the cosets in the table are live cosets
        self.p = list(range(gamma + 1))
        # delete the useless columns
        del table[len(self.p):]
        # re-define values
        for row in table:
            for j in range(len(self.A)):
                row[j] -= bisect_left(chi, row[j])

    def conjugates(self, R):
        R_c = list(chain.from_iterable((rel.cyclic_conjugates(), \
                (rel**-1).cyclic_conjugates()) for rel in R))
        R_set = set()
        for conjugate in R_c:
            R_set = R_set.union(conjugate)
        R_c_list = []
        for x in self.A:
            r = {word for word in R_set if word[0] == x}
            R_c_list.append(r)
            R_set.difference_update(r)
        return R_c_list

    def coset_representative(self, coset):
        '''
        Compute the coset representative of a given coset.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
        >>> F, x, y = free_group("x, y")
        >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
        >>> C = coset_enumeration_r(f, [x])
        >>> C.compress()
        >>> C.table
        [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]
        >>> C.coset_representative(0)
        <identity>
        >>> C.coset_representative(1)
        y
        >>> C.coset_representative(2)
        y**-1

        '''
        for x in self.A:
            gamma = self.table[coset][self.A_dict[x]]
            if coset == 0:
                return self.fp_group.identity
            if gamma < coset:
                return self.coset_representative(gamma)*x**-1

    ##############################
    #      Modified Methods      #
    ##############################

    def modified_define(self, alpha, x):
        r"""
        Define a function p_p from from [1..n] to A* as
        an additional component of the modified coset table.

        Parameters
        ==========

        \alpha \in \Omega
        x \in A*

        See Also
        ========

        define

        """
        self.define(alpha, x, modified=True)

    def modified_scan(self, alpha, w, y, fill=False):
        r"""
        Parameters
        ==========
        \alpha \in \Omega
        w \in A*
        y \in (YUY^-1)
        fill -- `modified_scan_and_fill` when set to True.

        See Also
        ========

        scan
        """
        self.scan(alpha, w, y=y, fill=fill, modified=True)

    def modified_scan_and_fill(self, alpha, w, y):
        self.modified_scan(alpha, w, y, fill=True)

    def modified_merge(self, k, lamda, w, q):
        r"""
        Parameters
        ==========

        'k', 'lamda' -- the two class representatives to be merged.
        q -- queue of length l of elements to be deleted from `\Omega` *.
        w -- Word in (YUY^-1)

        See Also
        ========

        merge
        """
        self.merge(k, lamda, q, w=w, modified=True)

    def modified_rep(self, k):
        r"""
        Parameters
        ==========

        `k \in [0 \ldots n-1]`

        See Also
        ========

        rep
        """
        self.rep(k, modified=True)

    def modified_coincidence(self, alpha, beta, w):
        r"""
        Parameters
        ==========

        A coincident pair `\alpha, \beta \in \Omega, w \in Y \cup Y^{-1}`

        See Also
        ========

        coincidence

        """
        self.coincidence(alpha, beta, w=w, modified=True)

###############################################################################
#                           COSET ENUMERATION                                 #
###############################################################################

# relator-based method
def coset_enumeration_r(fp_grp, Y, max_cosets=None, draft=None,
                                    incomplete=False, modified=False):
    """
    This is easier of the two implemented methods of coset enumeration.
    and is often called the HLT method, after Hazelgrove, Leech, Trotter
    The idea is that we make use of ``scan_and_fill`` makes new definitions
    whenever the scan is incomplete to enable the scan to complete; this way
    we fill in the gaps in the scan of the relator or subgroup generator,
    that's why the name relator-based method.

    An instance of `CosetTable` for `fp_grp` can be passed as the keyword
    argument `draft` in which case the coset enumeration will start with
    that instance and attempt to complete it.

    When `incomplete` is `True` and the function is unable to complete for
    some reason, the partially complete table will be returned.

    # TODO: complete the docstring

    See Also
    ========

    scan_and_fill,

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_r
    >>> F, x, y = free_group("x, y")

    # Example 5.1 from [1]
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = coset_enumeration_r(f, [x])
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [0, 0, 1, 2]
    [1, 1, 2, 0]
    [2, 2, 0, 1]
    >>> C.p
    [0, 1, 2, 1, 1]

    # Example from exercises Q2 [1]
    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    >>> C = coset_enumeration_r(f, [])
    >>> C.compress(); C.standardize()
    >>> C.table
    [[1, 2, 3, 4],
    [5, 0, 6, 7],
    [0, 5, 7, 6],
    [7, 6, 5, 0],
    [6, 7, 0, 5],
    [2, 1, 4, 3],
    [3, 4, 2, 1],
    [4, 3, 1, 2]]

    # Example 5.2
    >>> f = FpGroup(F, [x**2, y**3, (x*y)**3])
    >>> Y = [x*y]
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [1, 1, 2, 1]
    [0, 0, 0, 2]
    [3, 3, 1, 0]
    [2, 2, 3, 3]

    # Example 5.3
    >>> f = FpGroup(F, [x**2*y**2, x**3*y**5])
    >>> Y = []
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [1, 3, 1, 3]
    [2, 0, 2, 0]
    [3, 1, 3, 1]
    [0, 2, 0, 2]

    # Example 5.4
    >>> F, a, b, c, d, e = free_group("a, b, c, d, e")
    >>> f = FpGroup(F, [a*b*c**-1, b*c*d**-1, c*d*e**-1, d*e*a**-1, e*a*b**-1])
    >>> Y = [a]
    >>> C = coset_enumeration_r(f, Y)
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # example of "compress" method
    >>> C.compress()
    >>> C.table
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # Exercises Pg. 161, Q2.
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**2*y**2, y**-1*x*y*x**-3])
    >>> Y = []
    >>> C = coset_enumeration_r(f, Y)
    >>> C.compress()
    >>> C.standardize()
    >>> C.table
    [[1, 2, 3, 4],
    [5, 0, 6, 7],
    [0, 5, 7, 6],
    [7, 6, 5, 0],
    [6, 7, 0, 5],
    [2, 1, 4, 3],
    [3, 4, 2, 1],
    [4, 3, 1, 2]]

    # John J. Cannon; Lucien A. Dimino; George Havas; Jane M. Watson
    # Mathematics of Computation, Vol. 27, No. 123. (Jul., 1973), pp. 463-490
    # from 1973chwd.pdf
    # Table 1. Ex. 1
    >>> F, r, s, t = free_group("r, s, t")
    >>> E1 = FpGroup(F, [t**-1*r*t*r**-2, r**-1*s*r*s**-2, s**-1*t*s*t**-2])
    >>> C = coset_enumeration_r(E1, [r])
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         print(C.table[i])
    [0, 0, 0, 0, 0, 0]

    Ex. 2
    >>> F, a, b = free_group("a, b")
    >>> Cox = FpGroup(F, [a**6, b**6, (a*b)**2, (a**2*b**2)**2, (a**3*b**3)**5])
    >>> C = coset_enumeration_r(Cox, [a])
    >>> index = 0
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         index += 1
    >>> index
    500

    # Ex. 3
    >>> F, a, b = free_group("a, b")
    >>> B_2_4 = FpGroup(F, [a**4, b**4, (a*b)**4, (a**-1*b)**4, (a**2*b)**4, \
            (a*b**2)**4, (a**2*b**2)**4, (a**-1*b*a*b)**4, (a*b**-1*a*b)**4])
    >>> C = coset_enumeration_r(B_2_4, [a])
    >>> index = 0
    >>> for i in range(len(C.p)):
    ...     if C.p[i] == i:
    ...         index += 1
    >>> index
    1024

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    # 1. Initialize a coset table C for < X|R >
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    # Define coset table methods.
    if modified:
        _scan_and_fill = C.modified_scan_and_fill
        _define = C.modified_define
    else:
        _scan_and_fill = C.scan_and_fill
        _define = C.define
    if draft:
        C.table = draft.table[:]
        C.p = draft.p[:]
    R = fp_grp.relators
    A_dict = C.A_dict
    p = C.p
    for i in range(len(Y)):
        if modified:
            _scan_and_fill(0, Y[i], C._grp.generators[i])
        else:
            _scan_and_fill(0, Y[i])
    alpha = 0
    while alpha < C.n:
        if p[alpha] == alpha:
            try:
                for w in R:
                    if modified:
                        _scan_and_fill(alpha, w, C._grp.identity)
                    else:
                        _scan_and_fill(alpha, w)
                    # if alpha was eliminated during the scan then break
                    if p[alpha] < alpha:
                        break
                if p[alpha] == alpha:
                    for x in A_dict:
                        if C.table[alpha][A_dict[x]] is None:
                            _define(alpha, x)
            except ValueError as e:
                if incomplete:
                    return C
                raise e
        alpha += 1
    return C

def modified_coset_enumeration_r(fp_grp, Y, max_cosets=None, draft=None,
                                    incomplete=False):
    r"""
    Introduce a new set of symbols y \in Y that correspond to the
    generators of the subgroup. Store the elements of Y as a
    word P[\alpha, x] and compute the coset table similar to that of
    the regular coset enumeration methods.

    Examples
    ========

    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup
    >>> from sympy.combinatorics.coset_table import modified_coset_enumeration_r
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = modified_coset_enumeration_r(f, [x])
    >>> C.table
    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1], [None, 1, None, None], [1, 3, None, None]]

    See Also
    ========

    coset_enumertation_r

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.,
           "Handbook of Computational Group Theory",
           Section 5.3.2
    """
    return coset_enumeration_r(fp_grp, Y, max_cosets=max_cosets, draft=draft,
                             incomplete=incomplete, modified=True)

# Pg. 166
# coset-table based method
def coset_enumeration_c(fp_grp, Y, max_cosets=None, draft=None,
                                                incomplete=False):
    """
    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_c
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = coset_enumeration_c(f, [x])
    >>> C.table
    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]

    """
    # Initialize a coset table C for < X|R >
    X = fp_grp.generators
    R = fp_grp.relators
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    if draft:
        C.table = draft.table[:]
        C.p = draft.p[:]
        C.deduction_stack = draft.deduction_stack
        for alpha, x in product(range(len(C.table)), X):
            if C.table[alpha][C.A_dict[x]] is not None:
                C.deduction_stack.append((alpha, x))
    A = C.A
    # replace all the elements by cyclic reductions
    R_cyc_red = [rel.identity_cyclic_reduction() for rel in R]
    R_c = list(chain.from_iterable((rel.cyclic_conjugates(), (rel**-1).cyclic_conjugates()) \
            for rel in R_cyc_red))
    R_set = set()
    for conjugate in R_c:
        R_set = R_set.union(conjugate)
    # a list of subsets of R_c whose words start with "x".
    R_c_list = []
    for x in C.A:
        r = {word for word in R_set if word[0] == x}
        R_c_list.append(r)
        R_set.difference_update(r)
    for w in Y:
        C.scan_and_fill_c(0, w)
    for x in A:
        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
    alpha = 0
    while alpha < len(C.table):
        if C.p[alpha] == alpha:
            try:
                for x in C.A:
                    if C.p[alpha] != alpha:
                        break
                    if C.table[alpha][C.A_dict[x]] is None:
                        C.define_c(alpha, x)
                        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
            except ValueError as e:
                if incomplete:
                    return C
                raise e
        alpha += 1
    return C