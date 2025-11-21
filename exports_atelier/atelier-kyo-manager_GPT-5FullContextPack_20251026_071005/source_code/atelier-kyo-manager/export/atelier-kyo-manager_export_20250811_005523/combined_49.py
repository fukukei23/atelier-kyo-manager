
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\basic.py ===
"""Base class for all the objects in SymPy"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Iterable
from itertools import zip_longest
from functools import cmp_to_key
from typing import TYPE_CHECKING, overload

from .assumptions import _prepare_class_assumptions
from .cache import cacheit
from .sympify import _sympify, sympify, SympifyError, _external_converter
from .sorting import ordered
from .kind import Kind, UndefinedKind
from ._print_helpers import Printable

from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, numbered_symbols
from sympy.utilities.misc import filldedent, func_name


if TYPE_CHECKING:
    from typing import ClassVar, TypeVar, Any
    from typing_extensions import Self
    from .assumptions import StdFactKB
    from .symbol import Symbol

    Tbasic = TypeVar("Tbasic", bound='Basic')


def as_Basic(expr):
    """Return expr as a Basic instance using strict sympify
    or raise a TypeError; this is just a wrapper to _sympify,
    raising a TypeError instead of a SympifyError."""
    try:
        return _sympify(expr)
    except SympifyError:
        raise TypeError(
            'Argument must be a Basic object, not `%s`' % func_name(
            expr))


# Key for sorting commutative args in canonical order
# by name. This is used for canonical ordering of the
# args for Add and Mul *if* the names of both classes
# being compared appear here. Some things in this list
# are not spelled the same as their name so they do not,
# in effect, appear here. See Basic.compare.
ordering_of_classes = [
    # singleton numbers
    'Zero', 'One', 'Half', 'Infinity', 'NaN', 'NegativeOne', 'NegativeInfinity',
    # numbers
    'Integer', 'Rational', 'Float',
    # singleton symbols
    'Exp1', 'Pi', 'ImaginaryUnit',
    # symbols
    'Symbol', 'Wild',
    # arithmetic operations
    'Pow', 'Mul', 'Add',
    # function values
    'Derivative', 'Integral',
    # defined singleton functions
    'Abs', 'Sign', 'Sqrt',
    'Floor', 'Ceiling',
    'Re', 'Im', 'Arg',
    'Conjugate',
    'Exp', 'Log',
    'Sin', 'Cos', 'Tan', 'Cot', 'ASin', 'ACos', 'ATan', 'ACot',
    'Sinh', 'Cosh', 'Tanh', 'Coth', 'ASinh', 'ACosh', 'ATanh', 'ACoth',
    'RisingFactorial', 'FallingFactorial',
    'factorial', 'binomial',
    'Gamma', 'LowerGamma', 'UpperGamma', 'PolyGamma',
    'Erf',
    # special polynomials
    'Chebyshev', 'Chebyshev2',
    # undefined functions
    'Function', 'WildFunction',
    # anonymous functions
    'Lambda',
    # Landau O symbol
    'Order',
    # relational operations
    'Equality', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',
    'GreaterThan', 'LessThan',
]

def _cmp_name(x: type, y: type) -> int:
    """return -1, 0, 1 if the name of x is before that of y.
    A string comparison is done if either name does not appear
    in `ordering_of_classes`. This is the helper for
    ``Basic.compare``

    Examples
    ========

    >>> from sympy import cos, tan, sin
    >>> from sympy.core import basic
    >>> save = basic.ordering_of_classes
    >>> basic.ordering_of_classes = ()
    >>> basic._cmp_name(cos, tan)
    -1
    >>> basic.ordering_of_classes = ["tan", "sin", "cos"]
    >>> basic._cmp_name(cos, tan)
    1
    >>> basic._cmp_name(sin, cos)
    -1
    >>> basic.ordering_of_classes = save

    """
    n1 = x.__name__
    n2 = y.__name__
    if n1 == n2:
        return 0

    # If the other object is not a Basic subclass, then we are not equal to it.
    if not issubclass(y, Basic):
        return -1

    UNKNOWN = len(ordering_of_classes) + 1
    try:
        i1 = ordering_of_classes.index(n1)
    except ValueError:
        i1 = UNKNOWN
    try:
        i2 = ordering_of_classes.index(n2)
    except ValueError:
        i2 = UNKNOWN
    if i1 == UNKNOWN and i2 == UNKNOWN:
        return (n1 > n2) - (n1 < n2)
    return (i1 > i2) - (i1 < i2)



@cacheit
def _get_postprocessors(clsname, arg_type):
    # Since only Add, Mul, Pow can be clsname, this cache
    # is not quadratic.
    postprocessors = set()
    mappings = _get_postprocessors_for_type(arg_type)
    for mapping in mappings:
        f = mapping.get(clsname, None)
        if f is not None:
            postprocessors.update(f)
    return postprocessors

@cacheit
def _get_postprocessors_for_type(arg_type):
    return tuple(
        Basic._constructor_postprocessor_mapping[cls]
        for cls in arg_type.mro()
        if cls in Basic._constructor_postprocessor_mapping
    )


class Basic(Printable):
    """
    Base class for all SymPy objects.

    Notes and conventions
    =====================

    1) Always use ``.args``, when accessing parameters of some instance:

    >>> from sympy import cot
    >>> from sympy.abc import x, y

    >>> cot(x).args
    (x,)

    >>> cot(x).args[0]
    x

    >>> (x*y).args
    (x, y)

    >>> (x*y).args[1]
    y


    2) Never use internal methods or variables (the ones prefixed with ``_``):

    >>> cot(x)._args    # do not use this, use cot(x).args instead
    (x,)


    3)  By "SymPy object" we mean something that can be returned by
        ``sympify``.  But not all objects one encounters using SymPy are
        subclasses of Basic.  For example, mutable objects are not:

        >>> from sympy import Basic, Matrix, sympify
        >>> A = Matrix([[1, 2], [3, 4]]).as_mutable()
        >>> isinstance(A, Basic)
        False

        >>> B = sympify(A)
        >>> isinstance(B, Basic)
        True
    """
    __slots__ = ('_mhash',              # hash value
                 '_args',               # arguments
                 '_assumptions'
                )

    _args: tuple[Basic, ...]
    _mhash: int | None

    @property
    def __sympy__(self):
        return True

    def __init_subclass__(cls):
        # Initialize the default_assumptions FactKB and also any assumptions
        # property methods. This method will only be called for subclasses of
        # Basic but not for Basic itself so we call
        # _prepare_class_assumptions(Basic) below the class definition.
        super().__init_subclass__()
        _prepare_class_assumptions(cls)

    # To be overridden with True in the appropriate subclasses
    is_number = False
    is_Atom = False
    is_Symbol = False
    is_symbol = False
    is_Indexed = False
    is_Dummy = False
    is_Wild = False
    is_Function = False
    is_Add = False
    is_Mul = False
    is_Pow = False
    is_Number = False
    is_Float = False
    is_Rational = False
    is_Integer = False
    is_NumberSymbol = False
    is_Order = False
    is_Derivative = False
    is_Piecewise = False
    is_Poly = False
    is_AlgebraicNumber = False
    is_Relational = False
    is_Equality = False
    is_Boolean = False
    is_Not = False
    is_Matrix = False
    is_Vector = False
    is_Point = False
    is_MatAdd = False
    is_MatMul = False

    default_assumptions: ClassVar[StdFactKB]

    is_composite: bool | None
    is_noninteger: bool | None
    is_extended_positive: bool | None
    is_negative: bool | None
    is_complex: bool | None
    is_extended_nonpositive: bool | None
    is_integer: bool | None
    is_positive: bool | None
    is_rational: bool | None
    is_extended_nonnegative: bool | None
    is_infinite: bool | None
    is_antihermitian: bool | None
    is_extended_negative: bool | None
    is_extended_real: bool | None
    is_finite: bool | None
    is_polar: bool | None
    is_imaginary: bool | None
    is_transcendental: bool | None
    is_extended_nonzero: bool | None
    is_nonzero: bool | None
    is_odd: bool | None
    is_algebraic: bool | None
    is_prime: bool | None
    is_commutative: bool | None
    is_nonnegative: bool | None
    is_nonpositive: bool | None
    is_hermitian: bool | None
    is_irrational: bool | None
    is_real: bool | None
    is_zero: bool | None
    is_even: bool | None

    kind: Kind = UndefinedKind

    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._assumptions = cls.default_assumptions
        obj._mhash = None  # will be set by __hash__ method.

        obj._args = args  # all items in args must be Basic objects
        return obj

    def copy(self):
        return self.func(*self.args)

    def __getnewargs__(self):
        return self.args

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __reduce_ex__(self, protocol):
        if protocol < 2:
            msg = "Only pickle protocol 2 or higher is supported by SymPy"
            raise NotImplementedError(msg)
        return super().__reduce_ex__(protocol)

    def __hash__(self) -> int:
        # hash cannot be cached using cache_it because infinite recurrence
        # occurs as hash is needed for setting cache dictionary keys
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h

    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.

        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args

    @property
    def assumptions0(self):
        """
        Return object `type` assumptions.

        For example:

          Symbol('x', real=True)
          Symbol('x', integer=True)

        are different objects. In other words, besides Python type (Symbol in
        this case), the initial assumptions are also forming their typeinfo.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.abc import x
        >>> x.assumptions0
        {'commutative': True}
        >>> x = Symbol("x", positive=True)
        >>> x.assumptions0
        {'commutative': True, 'complex': True, 'extended_negative': False,
         'extended_nonnegative': True, 'extended_nonpositive': False,
         'extended_nonzero': True, 'extended_positive': True, 'extended_real':
         True, 'finite': True, 'hermitian': True, 'imaginary': False,
         'infinite': False, 'negative': False, 'nonnegative': True,
         'nonpositive': False, 'nonzero': True, 'positive': True, 'real':
         True, 'zero': False}
        """
        return {}

    def compare(self, other):
        """
        Return -1, 0, 1 if the object is less than, equal,
        or greater than other in a canonical sense.
        Non-Basic are always greater than Basic.
        If both names of the classes being compared appear
        in the `ordering_of_classes` then the ordering will
        depend on the appearance of the names there.
        If either does not appear in that list, then the
        comparison is based on the class name.
        If the names are the same then a comparison is made
        on the length of the hashable content.
        Items of the equal-lengthed contents are then
        successively compared using the same rules. If there
        is never a difference then 0 is returned.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x.compare(y)
        -1
        >>> x.compare(x)
        0
        >>> y.compare(x)
        1

        """
        # all redefinitions of __cmp__ method should start with the
        # following lines:
        if self is other:
            return 0
        n1 = self.__class__
        n2 = other.__class__
        c = _cmp_name(n1, n2)
        if c:
            return c
        #
        st = self._hashable_content()
        ot = other._hashable_content()
        len_st = len(st)
        len_ot = len(ot)
        c = (len_st > len_ot) - (len_st < len_ot)
        if c:
            return c
        for l, r in zip(st, ot):
            if isinstance(l, Basic):
                c = l.compare(r)
            elif isinstance(l, frozenset):
                l = Basic(*l) if isinstance(l, frozenset) else l
                r = Basic(*r) if isinstance(r, frozenset) else r
                c = l.compare(r)
            else:
                c = (l > r) - (l < r)
            if c:
                return c
        return 0

    @classmethod
    def fromiter(cls, args, **assumptions):
        """
        Create a new object from an iterable.

        This is a convenience function that allows one to create objects from
        any iterable, without having to convert to a list or tuple first.

        Examples
        ========

        >>> from sympy import Tuple
        >>> Tuple.fromiter(i for i in range(5))
        (0, 1, 2, 3, 4)

        """
        return cls(*tuple(args), **assumptions)

    @classmethod
    def class_key(cls) -> tuple[int, int, str]:
        """Nice order of classes."""
        return 5, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        """
        Return a sort key.

        Examples
        ========

        >>> from sympy import S, I

        >>> sorted([S(1)/2, I, -I], key=lambda x: x.sort_key())
        [1/2, -I, I]

        >>> S("[x, 1/x, 1/x**2, x**2, x**(1/2), x**(1/4), x**(3/2)]")
        [x, 1/x, x**(-2), x**2, sqrt(x), x**(1/4), x**(3/2)]
        >>> sorted(_, key=lambda x: x.sort_key())
        [x**(-2), 1/x, x**(1/4), sqrt(x), x, x**(3/2), x**2]

        """

        # XXX: remove this when issue 5169 is fixed
        def inner_key(arg):
            if isinstance(arg, Basic):
                return arg.sort_key(order)
            else:
                return arg

        args = self._sorted_args
        args = len(args), tuple([inner_key(arg) for arg in args])
        return self.class_key(), args, S.One.sort_key(), S.One

    def _do_eq_sympify(self, other):
        """Returns a boolean indicating whether a == b when either a
        or b is not a Basic. This is only done for types that were either
        added to `converter` by a 3rd party or when the object has `_sympy_`
        defined. This essentially reuses the code in `_sympify` that is
        specific for this use case. Non-user defined types that are meant
        to work with SymPy should be handled directly in the __eq__ methods
        of the `Basic` classes it could equate to and not be converted. Note
        that after conversion, `==`  is used again since it is not
        necessarily clear whether `self` or `other`'s __eq__ method needs
        to be used."""
        for superclass in type(other).__mro__:
            conv = _external_converter.get(superclass)
            if conv is not None:
                return self == conv(other)
        if hasattr(other, '_sympy_'):
            return self == other._sympy_()
        return NotImplemented

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        their symbolic trees.

        This is the same as a.compare(b) == 0 but faster.

        Notes
        =====

        If a class that overrides __eq__() needs to retain the
        implementation of __hash__() from a parent class, the
        interpreter must be told this explicitly by setting
        __hash__ : Callable[[object], int] = <ParentClass>.__hash__.
        Otherwise the inheritance of __hash__() will be blocked,
        just as if __hash__ had been explicitly set to None.

        References
        ==========

        from https://docs.python.org/dev/reference/datamodel.html#object.__hash__
        """
        if self is other:
            return True

        if not isinstance(other, Basic):
            return self._do_eq_sympify(other)

        # check for pure number expr
        if  not (self.is_Number and other.is_Number) and (
                type(self) != type(other)):
            return False
        a, b = self._hashable_content(), other._hashable_content()
        if a != b:
            return False
        # check number *in* an expression
        for a, b in zip(a, b):
            if not isinstance(a, Basic):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True

    def __ne__(self, other):
        """``a != b``  -> Compare two symbolic trees and see whether they are different

        this is the same as:

        ``a.compare(b) != 0``

        but faster
        """
        return not self == other

    def dummy_eq(self, other, symbol=None):
        """
        Compare two expressions and handle dummy symbols.

        Examples
        ========

        >>> from sympy import Dummy
        >>> from sympy.abc import x, y

        >>> u = Dummy('u')

        >>> (u**2 + 1).dummy_eq(x**2 + 1)
        True
        >>> (u**2 + 1) == (x**2 + 1)
        False

        >>> (u**2 + y).dummy_eq(x**2 + y, x)
        True
        >>> (u**2 + y).dummy_eq(x**2 + y, y)
        False

        """
        s = self.as_dummy()
        o = _sympify(other)
        o = o.as_dummy()

        dummy_symbols = [i for i in s.free_symbols if i.is_Dummy]

        if len(dummy_symbols) == 1:
            dummy = dummy_symbols.pop()
        else:
            return s == o

        if symbol is None:
            symbols = o.free_symbols

            if len(symbols) == 1:
                symbol = symbols.pop()
            else:
                return s == o

        tmp = dummy.__class__()

        return s.xreplace({dummy: tmp}) == o.xreplace({symbol: tmp})

    @overload
    def atoms(self) -> set[Basic]: ...
    @overload
    def atoms(self, *types: Tbasic | type[Tbasic]) -> set[Tbasic]: ...

    def atoms(self, *types: Tbasic | type[Tbasic]) -> set[Basic] | set[Tbasic]:
        """Returns the atoms that form the current object.

        By default, only objects that are truly atomic and cannot
        be divided into smaller pieces are returned: symbols, numbers,
        and number symbols like I and pi. It is possible to request
        atoms of any type, however, as demonstrated below.

        Examples
        ========

        >>> from sympy import I, pi, sin
        >>> from sympy.abc import x, y
        >>> (1 + x + 2*sin(y + I*pi)).atoms()
        {1, 2, I, pi, x, y}

        If one or more types are given, the results will contain only
        those types of atoms.

        >>> from sympy import Number, NumberSymbol, Symbol
        >>> (1 + x + 2*sin(y + I*pi)).atoms(Symbol)
        {x, y}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number)
        {1, 2}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol)
        {1, 2, pi}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol, I)
        {1, 2, I, pi}

        Note that I (imaginary unit) and zoo (complex infinity) are special
        types of number symbols and are not part of the NumberSymbol class.

        The type can be given implicitly, too:

        >>> (1 + x + 2*sin(y + I*pi)).atoms(x) # x is a Symbol
        {x, y}

        Be careful to check your assumptions when using the implicit option
        since ``S(1).is_Integer = True`` but ``type(S(1))`` is ``One``, a special type
        of SymPy atom, while ``type(S(2))`` is type ``Integer`` and will find all
        integers in an expression:

        >>> from sympy import S
        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(1))
        {1}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(2))
        {1, 2}

        Finally, arguments to atoms() can select more than atomic atoms: any
        SymPy type (loaded in core/__init__.py) can be listed as an argument
        and those types of "atoms" as found in scanning the arguments of the
        expression recursively:

        >>> from sympy import Function, Mul
        >>> from sympy.core.function import AppliedUndef
        >>> f = Function('f')
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(Function)
        {f(x), sin(y + I*pi)}
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(AppliedUndef)
        {f(x)}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Mul)
        {I*pi, 2*sin(y + I*pi)}

        """
        nodes = _preorder_traversal(self)
        if types:
            types2 = tuple([t if isinstance(t, type) else type(t) for t in types])
            return {node for node in nodes if isinstance(node, types2)}
        else:
            return {node for node in nodes if not node.args}

    @property
    def free_symbols(self) -> set[Basic]:
        """Return from the atoms of self those which are free symbols.

        Not all free symbols are ``Symbol`` (see examples)

        For most expressions, all symbols are free symbols. For some classes
        this is not true. e.g. Integrals use Symbols for the dummy variables
        which are bound variables, so Integral has a method to return all
        symbols except those. Derivative keeps track of symbols with respect
        to which it will perform a derivative; those are
        bound variables, too, so it has its own free_symbols method.

        Any other method that uses bound variables should implement a
        free_symbols method.

        Examples
        ========

        >>> from sympy import Derivative, Integral, IndexedBase
        >>> from sympy.abc import x, y, n
        >>> (x + 1).free_symbols
        {x}
        >>> Integral(x, y).free_symbols
        {x, y}

        Not all free symbols are actually symbols:

        >>> IndexedBase('F')[0].free_symbols
        {F, F[0]}

        The symbols of differentiation are not included unless they
        appear in the expression being differentiated.

        >>> Derivative(x + y, y).free_symbols
        {x, y}
        >>> Derivative(x, y).free_symbols
        {x}
        >>> Derivative(x, (y, n)).free_symbols
        {n, x}

        If you want to know if a symbol is in the variables of the
        Derivative you can do so as follows:

        >>> Derivative(x, y).has_free(y)
        True
        """
        empty: set[Basic] = set()
        return empty.union(*(a.free_symbols for a in self.args))

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        return set()

    def as_dummy(self) -> "Self":
        """Return the expression with any objects having structurally
        bound symbols replaced with unique, canonical symbols within
        the object in which they appear and having only the default
        assumption for commutativity being True. When applied to a
        symbol a new symbol having only the same commutativity will be
        returned.

        Examples
        ========

        >>> from sympy import Integral, Symbol
        >>> from sympy.abc import x
        >>> r = Symbol('r', real=True)
        >>> Integral(r, (r, x)).as_dummy()
        Integral(_0, (_0, x))
        >>> _.variables[0].is_real is None
        True
        >>> r.as_dummy()
        _r

        Notes
        =====

        Any object that has structurally bound variables should have
        a property, ``bound_symbols`` that returns those symbols
        appearing in the object.
        """
        from .symbol import Dummy, Symbol
        def can(x):
            # mask free that shadow bound
            free = x.free_symbols
            bound = set(x.bound_symbols)
            d = {i: Dummy() for i in bound & free}
            x = x.subs(d)
            # replace bound with canonical names
            x = x.xreplace(x.canonical_variables)
            # return after undoing masking
            return x.xreplace({v: k for k, v in d.items()})
        if not self.has(Symbol):
            return self
        return self.replace(
            lambda x: hasattr(x, 'bound_symbols'),
            can,
            simultaneous=False) # type:ignore

    @property
    def canonical_variables(self) -> dict[Basic, Symbol]:
        """Return a dictionary mapping any variable defined in
        ``self.bound_symbols`` to Symbols that do not clash
        with any free symbols in the expression.

        Examples
        ========

        >>> from sympy import Lambda
        >>> from sympy.abc import x
        >>> Lambda(x, 2*x).canonical_variables
        {x: _0}
        """
        bound: list[Basic] | None = getattr(self, 'bound_symbols', None)
        if bound is None:
            return {}
        dums = numbered_symbols('_')
        reps = {}
        # watch out for free symbol that are not in bound symbols;
        # those that are in bound symbols are about to get changed

        # XXX: free_symbols only returns particular kinds of expressions that
        # generally have a .name attribute. There is not a proper class/type
        # that represents this.
        names = {i.name for i in self.free_symbols - set(bound)} # type: ignore
        for b in bound:
            d = next(dums)
            if b.is_Symbol:
                while d.name in names:
                    d = next(dums)
            reps[b] = d
        return reps

    def rcall(self, *args):
        """Apply on the argument recursively through the expression tree.

        This method is used to simulate a common abuse of notation for
        operators. For instance, in SymPy the following will not work:

        ``(x+Lambda(y, 2*y))(z) == x+2*z``,

        however, you can use:

        >>> from sympy import Lambda
        >>> from sympy.abc import x, y, z
        >>> (x + Lambda(y, 2*y)).rcall(z)
        x + 2*z
        """
        if callable(self):
            return self(*args)
        elif self.args:
            newargs = [sub.rcall(*args) for sub in self.args]
            return self.func(*newargs)
        else:
            return self

    def is_hypergeometric(self, k):
        from sympy.simplify.simplify import hypersimp
        from sympy.functions.elementary.piecewise import Piecewise
        if self.has(Piecewise):
            return None
        return hypersimp(self, k) is not None

    @property
    def is_comparable(self):
        """Return True if self can be computed to a real number
        (or already is a real number) with precision, else False.

        Examples
        ========

        >>> from sympy import exp_polar, pi, I
        >>> (I*exp_polar(I*pi/2)).is_comparable
        True
        >>> (I*exp_polar(I*pi*2)).is_comparable
        False

        A False result does not mean that `self` cannot be rewritten
        into a form that would be comparable. For example, the
        difference computed below is zero but without simplification
        it does not evaluate to a zero with precision:

        >>> e = 2**pi*(1 + 2**pi)
        >>> dif = e - e.expand()
        >>> dif.is_comparable
        False
        >>> dif.n(2)._prec
        1

        """
        return self._eval_is_comparable()

    def _eval_is_comparable(self) -> bool:
        # Expr.is_comparable overrides this
        return False

    @property
    def func(self):
        """
        The top-level function in an expression.

        The following should hold for all objects::

            >> x == x.func(*x.args)

        Examples
        ========

        >>> from sympy.abc import x
        >>> a = 2*x
        >>> a.func
        <class 'sympy.core.mul.Mul'>
        >>> a.args
        (2, x)
        >>> a.func(*a.args)
        2*x
        >>> a == a.func(*a.args)
        True

        """
        return self.__class__

    @property
    def args(self) -> tuple[Basic, ...]:
        """Returns a tuple of arguments of 'self'.

        Examples
        ========

        >>> from sympy import cot
        >>> from sympy.abc import x, y

        >>> cot(x).args
        (x,)

        >>> cot(x).args[0]
        x

        >>> (x*y).args
        (x, y)

        >>> (x*y).args[1]
        y

        Notes
        =====

        Never use self._args, always use self.args.
        Only use _args in __new__ when creating a new function.
        Do not override .args() from Basic (so that it is easy to
        change the interface in the future if needed).
        """
        return self._args

    @property
    def _sorted_args(self):
        """
        The same as ``args``.  Derived classes which do not fix an
        order on their arguments should override this method to
        produce the sorted representation.
        """
        return self.args

    def as_content_primitive(self, radical=False, clear=True):
        """A stub to allow Basic args (like Tuple) to be skipped when computing
        the content and primitive components of an expression.

        See Also
        ========

        sympy.core.expr.Expr.as_content_primitive
        """
        return S.One, self

    @overload
    def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
    @overload
    def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
    @overload
    def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

    def subs(self, arg1: Mapping[Basic | complex, Basic | complex]
            | Iterable[tuple[Basic | complex, Basic | complex]] | Basic | complex,
             arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
        """
        Substitutes old for new in an expression after sympifying args.

        `args` is either:
          - two arguments, e.g. foo.subs(old, new)
          - one iterable argument, e.g. foo.subs(iterable). The iterable may be
             o an iterable container with (old, new) pairs. In this case the
               replacements are processed in the order given with successive
               patterns possibly affecting replacements already made.
             o a dict or set whose key/value items correspond to old/new pairs.
               In this case the old/new pairs will be sorted by op count and in
               case of a tie, by number of args and the default_sort_key. The
               resulting sorted list is then processed as an iterable container
               (see previous).

        If the keyword ``simultaneous`` is True, the subexpressions will not be
        evaluated until all the substitutions have been made.

        Examples
        ========

        >>> from sympy import pi, exp, limit, oo
        >>> from sympy.abc import x, y
        >>> (1 + x*y).subs(x, pi)
        pi*y + 1
        >>> (1 + x*y).subs({x:pi, y:2})
        1 + 2*pi
        >>> (1 + x*y).subs([(x, pi), (y, 2)])
        1 + 2*pi
        >>> reps = [(y, x**2), (x, 2)]
        >>> (x + y).subs(reps)
        6
        >>> (x + y).subs(reversed(reps))
        x**2 + 2

        >>> (x**2 + x**4).subs(x**2, y)
        y**2 + y

        To replace only the x**2 but not the x**4, use xreplace:

        >>> (x**2 + x**4).xreplace({x**2: y})
        x**4 + y

        To delay evaluation until all substitutions have been made,
        set the keyword ``simultaneous`` to True:

        >>> (x/y).subs([(x, 0), (y, 0)])
        0
        >>> (x/y).subs([(x, 0), (y, 0)], simultaneous=True)
        nan

        This has the added feature of not allowing subsequent substitutions
        to affect those already made:

        >>> ((x + y)/y).subs({x + y: y, y: x + y})
        1
        >>> ((x + y)/y).subs({x + y: y, y: x + y}, simultaneous=True)
        y/(x + y)

        In order to obtain a canonical result, unordered iterables are
        sorted by count_op length, number of arguments and by the
        default_sort_key to break any ties. All other iterables are left
        unsorted.

        >>> from sympy import sqrt, sin, cos
        >>> from sympy.abc import a, b, c, d, e

        >>> A = (sqrt(sin(2*x)), a)
        >>> B = (sin(2*x), b)
        >>> C = (cos(2*x), c)
        >>> D = (x, d)
        >>> E = (exp(x), e)

        >>> expr = sqrt(sin(2*x))*sin(exp(x)*x)*cos(2*x) + sin(2*x)

        >>> expr.subs(dict([A, B, C, D, E]))
        a*c*sin(d*e) + b

        The resulting expression represents a literal replacement of the
        old arguments with the new arguments. This may not reflect the
        limiting behavior of the expression:

        >>> (x**3 - 3*x).subs({x: oo})
        nan

        >>> limit(x**3 - 3*x, x, oo)
        oo

        If the substitution will be followed by numerical
        evaluation, it is better to pass the substitution to
        evalf as

        >>> (1/x).evalf(subs={x: 3.0}, n=21)
        0.333333333333333333333

        rather than

        >>> (1/x).subs({x: 3.0}).evalf(21)
        0.333333333333333314830

        as the former will ensure that the desired level of precision is
        obtained.

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules
        sympy.core.evalf.EvalfMixin.evalf: calculates the given formula to a desired level of precision

        """
        from .containers import Dict
        from .symbol import Dummy, Symbol
        from .numbers import _illegal

        items: Iterable[tuple[Basic | complex, Basic | complex]]

        unordered = False
        if arg2 is None:

            if isinstance(arg1, set):
                items = arg1
                unordered = True
            elif isinstance(arg1, (Dict, Mapping)):
                unordered = True
                items = arg1.items() # type: ignore
            elif not iterable(arg1):
                raise ValueError(filldedent("""
                   When a single argument is passed to subs
                   it should be a dictionary of old: new pairs or an iterable
                   of (old, new) tuples."""))
            else:
                items = arg1 # type: ignore
        else:
            items = [(arg1, arg2)] # type: ignore

        def sympify_old(old) -> Basic:
            if isinstance(old, str):
                # Use Symbol rather than parse_expr for old
                return Symbol(old)
            elif isinstance(old, type):
                # Allow a type e.g. Function('f') or sin
                return sympify(old, strict=False)
            else:
                return sympify(old, strict=True)

        def sympify_new(new) -> Basic:
            if isinstance(new, (str, type)):
                # Allow a type or parse a string input
                return sympify(new, strict=False)
            else:
                return sympify(new, strict=True)

        sequence = [(sympify_old(s1), sympify_new(s2)) for s1, s2 in items]

        # skip if there is no change
        sequence = [(s1, s2) for s1, s2 in sequence if not _aresame(s1, s2)]

        simultaneous = kwargs.pop('simultaneous', False)

        if unordered:
            from .sorting import _nodes, default_sort_key
            sequence_dict = dict(sequence)
            # order so more complex items are first and items
            # of identical complexity are ordered so
            # f(x) < f(y) < x < y
            # \___ 2 __/    \_1_/  <- number of nodes
            #
            # For more complex ordering use an unordered sequence.
            k = list(ordered(sequence_dict, default=False, keys=(
                lambda x: -_nodes(x),
                default_sort_key,
                )))
            sequence = [(k, sequence_dict[k]) for k in k]
            # do infinities first
            if not simultaneous:
                redo = [i for i, seq in enumerate(sequence) if seq[1] in _illegal]
                for i in reversed(redo):
                    sequence.insert(0, sequence.pop(i))

        if simultaneous:  # XXX should this be the default for dict subs?
            reps = {}
            rv = self
            kwargs['hack2'] = True
            m = Dummy('subs_m')
            for old, new in sequence:
                com = new.is_commutative
                if com is None:
                    com = True
                d = Dummy('subs_d', commutative=com)
                # using d*m so Subs will be used on dummy variables
                # in things like Derivative(f(x, y), x) in which x
                # is both free and bound
                rv = rv._subs(old, d*m, **kwargs)
                if not isinstance(rv, Basic):
                    break
                reps[d] = new
            reps[m] = S.One  # get rid of m
            return rv.xreplace(reps)
        else:
            rv = self
            for old, new in sequence:
                rv = rv._subs(old, new, **kwargs)
                if not isinstance(rv, Basic):
                    break
            return rv

    @cacheit
    def _subs(self, old, new, **hints):
        """Substitutes an expression old -> new.

        If self is not equal to old then _eval_subs is called.
        If _eval_subs does not want to make any special replacement
        then a None is received which indicates that the fallback
        should be applied wherein a search for replacements is made
        amongst the arguments of self.

        >>> from sympy import Add
        >>> from sympy.abc import x, y, z

        Examples
        ========

        Add's _eval_subs knows how to target x + y in the following
        so it makes the change:

        >>> (x + y + z).subs(x + y, 1)
        z + 1

        Add's _eval_subs does not need to know how to find x + y in
        the following:

        >>> Add._eval_subs(z*(x + y) + 3, x + y, 1) is None
        True

        The returned None will cause the fallback routine to traverse the args and
        pass the z*(x + y) arg to Mul where the change will take place and the
        substitution will succeed:

        >>> (z*(x + y) + 3).subs(x + y, 1)
        z + 3

        ** Developers Notes **

        An _eval_subs routine for a class should be written if:

            1) any arguments are not instances of Basic (e.g. bool, tuple);

            2) some arguments should not be targeted (as in integration
               variables);

            3) if there is something other than a literal replacement
               that should be attempted (as in Piecewise where the condition
               may be updated without doing a replacement).

        If it is overridden, here are some special cases that might arise:

            1) If it turns out that no special change was made and all
               the original sub-arguments should be checked for
               replacements then None should be returned.

            2) If it is necessary to do substitutions on a portion of
               the expression then _subs should be called. _subs will
               handle the case of any sub-expression being equal to old
               (which usually would not be the case) while its fallback
               will handle the recursion into the sub-arguments. For
               example, after Add's _eval_subs removes some matching terms
               it must process the remaining terms so it calls _subs
               on each of the un-matched terms and then adds them
               onto the terms previously obtained.

           3) If the initial expression should remain unchanged then
              the original expression should be returned. (Whenever an
              expression is returned, modified or not, no further
              substitution of old -> new is attempted.) Sum's _eval_subs
              routine uses this strategy when a substitution is attempted
              on any of its summation variables.
        """

        def fallback(self, old, new):
            """
            Try to replace old with new in any of self's arguments.
            """
            hit = False
            args = list(self.args)
            for i, arg in enumerate(args):
                if not hasattr(arg, '_eval_subs'):
                    continue
                arg = arg._subs(old, new, **hints)
                if not _aresame(arg, args[i]):
                    hit = True
                    args[i] = arg
            if hit:
                rv = self.func(*args)
                hack2 = hints.get('hack2', False)
                if hack2 and self.is_Mul and not rv.is_Mul:  # 2-arg hack
                    coeff = S.One
                    nonnumber = []
                    for i in args:
                        if i.is_Number:
                            coeff *= i
                        else:
                            nonnumber.append(i)
                    nonnumber = self.func(*nonnumber)
                    if coeff is S.One:
                        return nonnumber
                    else:
                        return self.func(coeff, nonnumber, evaluate=False)
                return rv
            return self

        if _aresame(self, old):
            return new

        rv = self._eval_subs(old, new)
        if rv is None:
            rv = fallback(self, old, new)
        return rv

    def _eval_subs(self, old, new) -> Basic | None:
        """Override this stub if you want to do anything more than
        attempt a replacement of old with new in the arguments of self.

        See also
        ========

        _subs
        """
        return None

    def xreplace(self, rule):
        """
        Replace occurrences of objects within the expression.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule

        Returns
        =======

        xreplace : the result of the replacement

        Examples
        ========

        >>> from sympy import symbols, pi, exp
        >>> x, y, z = symbols('x y z')
        >>> (1 + x*y).xreplace({x: pi})
        pi*y + 1
        >>> (1 + x*y).xreplace({x: pi, y: 2})
        1 + 2*pi

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> (x*y + z).xreplace({x*y: pi})
        z + pi
        >>> (x*y*z).xreplace({x*y: pi})
        x*y*z
        >>> (2*x).xreplace({2*x: y, x: z})
        y
        >>> (2*2*x).xreplace({2*x: y, x: z})
        4*z
        >>> (x + y + 2).xreplace({x + y: 2})
        x + y + 2
        >>> (x + 2 + exp(x + 2)).xreplace({x + 2: y})
        x + exp(y) + 2

        xreplace does not differentiate between free and bound symbols. In the
        following, subs(x, y) would not change x since it is a bound symbol,
        but xreplace does:

        >>> from sympy import Integral
        >>> Integral(x, (x, 1, 2*x)).xreplace({x: y})
        Integral(y, (y, 1, 2*y))

        Trying to replace x with an expression raises an error:

        >>> Integral(x, (x, 1, 2*x)).xreplace({x: 2*y}) # doctest: +SKIP
        ValueError: Invalid limits given: ((2*y, 1, 4*y),)

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        subs: substitution of subexpressions as defined by the objects
              themselves.

        """
        value, _ = self._xreplace(rule)
        return value

    def _xreplace(self, rule):
        """
        Helper for xreplace. Tracks whether a replacement actually occurred.
        """
        if self in rule:
            return rule[self], True
        elif rule:
            args = []
            changed = False
            for a in self.args:
                _xreplace = getattr(a, '_xreplace', None)
                if _xreplace is not None:
                    a_xr = _xreplace(rule)
                    args.append(a_xr[0])
                    changed |= a_xr[1]
                else:
                    args.append(a)
            args = tuple(args)
            if changed:
                return self.func(*args), True
        return self, False

    @cacheit
    def has(self, *patterns):
        """
        Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import sin
        >>> from sympy.abc import x, y, z
        >>> (x**2 + sin(x*y)).has(z)
        False
        >>> (x**2 + sin(x*y)).has(x, y, z)
        True
        >>> x.has(x)
        True

        Note ``has`` is a structural algorithm with no knowledge of
        mathematics. Consider the following half-open interval:

        >>> from sympy import Interval
        >>> i = Interval.Lopen(0, 5); i
        Interval.Lopen(0, 5)
        >>> i.args
        (0, 5, True, False)
        >>> i.has(4)  # there is no "4" in the arguments
        False
        >>> i.has(0)  # there *is* a "0" in the arguments
        True

        Instead, use ``contains`` to determine whether a number is in the
        interval or not:

        >>> i.contains(4)
        True
        >>> i.contains(0)
        False


        Note that ``expr.has(*patterns)`` is exactly equivalent to
        ``any(expr.has(p) for p in patterns)``. In particular, ``False`` is
        returned when the list of patterns is empty.

        >>> x.has()
        False

        """
        return self._has(iterargs, *patterns)

    def has_xfree(self, s: set[Basic]):
        """Return True if self has any of the patterns in s as a
        free argument, else False. This is like `Basic.has_free`
        but this will only report exact argument matches.

        Examples
        ========

        >>> from sympy import Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> f(x).has_xfree({f})
        False
        >>> f(x).has_xfree({f(x)})
        True
        >>> f(x + 1).has_xfree({x})
        True
        >>> f(x + 1).has_xfree({x + 1})
        True
        >>> f(x + y + 1).has_xfree({x + 1})
        False
        """
        # protect O(1) containment check by requiring:
        if type(s) is not set:
            raise TypeError('expecting set argument')
        return any(a in s for a in iterfreeargs(self))

    @cacheit
    def has_free(self, *patterns):
        """Return True if self has object(s) ``x`` as a free expression
        else False.

        Examples
        ========

        >>> from sympy import Integral, Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> g = Function('g')
        >>> expr = Integral(f(x), (f(x), 1, g(y)))
        >>> expr.free_symbols
        {y}
        >>> expr.has_free(g(y))
        True
        >>> expr.has_free(*(x, f(x)))
        False

        This works for subexpressions and types, too:

        >>> expr.has_free(g)
        True
        >>> (x + y + 1).has_free(y + 1)
        True
        """
        if not patterns:
            return False
        p0 = patterns[0]
        if len(patterns) == 1 and iterable(p0) and not isinstance(p0, Basic):
            # Basic can contain iterables (though not non-Basic, ideally)
            # but don't encourage mixed passing patterns
            raise TypeError(filldedent('''
                Expecting 1 or more Basic args, not a single
                non-Basic iterable. Don't forget to unpack
                iterables: `eq.has_free(*patterns)`'''))
        # try quick test first
        s = set(patterns)
        rv = self.has_xfree(s)
        if rv:
            return rv
        # now try matching through slower _has
        return self._has(iterfreeargs, *patterns)

    def _has(self, iterargs, *patterns):
        # separate out types and unhashable objects
        type_set = set()  # only types
        p_set = set()  # hashable non-types
        for p in patterns:
            if isinstance(p, type) and issubclass(p, Basic):
                type_set.add(p)
                continue
            if not isinstance(p, Basic):
                try:
                    p = _sympify(p)
                except SympifyError:
                    continue  # Basic won't have this in it
            p_set.add(p)  # fails if object defines __eq__ but
                          # doesn't define __hash__
        types = tuple(type_set)   #
        for i in iterargs(self):  #
            if i in p_set:        # <--- here, too
                return True
            if isinstance(i, types):
                return True

        # use matcher if defined, e.g. operations defines
        # matcher that checks for exact subset containment,
        # (x + y + 1).has(x + 1) -> True
        for i in p_set - type_set:  # types don't have matchers
            if not hasattr(i, '_has_matcher'):
                continue
            match = i._has_matcher()
            if any(match(arg) for arg in iterargs(self)):
                return True

        # no success
        return False

    def replace(self, query, value, map=False, simultaneous=True, exact=None) -> Basic:
        """
        Replace matching subexpressions of ``self`` with ``value``.

        If ``map = True`` then also return the mapping {old: new} where ``old``
        was a sub-expression found with query and ``new`` is the replacement
        value for it. If the expression itself does not match the query, then
        the returned value will be ``self.xreplace(map)`` otherwise it should
        be ``self.subs(ordered(map.items()))``.

        Traverses an expression tree and performs replacement of matching
        subexpressions from the bottom to the top of the tree. The default
        approach is to do the replacement in a simultaneous fashion so
        changes made are targeted only once. If this is not desired or causes
        problems, ``simultaneous`` can be set to False.

        In addition, if an expression containing more than one Wild symbol
        is being used to match subexpressions and the ``exact`` flag is None
        it will be set to True so the match will only succeed if all non-zero
        values are received for each Wild that appears in the match pattern.
        Setting this to False accepts a match of 0; while setting it True
        accepts all matches that have a 0 in them. See example below for
        cautions.

        The list of possible combinations of queries and replacement values
        is listed below:

        Examples
        ========

        Initial setup

        >>> from sympy import log, sin, cos, tan, Wild, Mul, Add
        >>> from sympy.abc import x, y
        >>> f = log(sin(x)) + tan(sin(x**2))

        1.1. type -> type
            obj.replace(type, newtype)

            When object of type ``type`` is found, replace it with the
            result of passing its argument(s) to ``newtype``.

            >>> f.replace(sin, cos)
            log(cos(x)) + tan(cos(x**2))
            >>> sin(x).replace(sin, cos, map=True)
            (cos(x), {sin(x): cos(x)})
            >>> (x*y).replace(Mul, Add)
            x + y

        1.2. type -> func
            obj.replace(type, func)

            When object of type ``type`` is found, apply ``func`` to its
            argument(s). ``func`` must be written to handle the number
            of arguments of ``type``.

            >>> f.replace(sin, lambda arg: sin(2*arg))
            log(sin(2*x)) + tan(sin(2*x**2))
            >>> (x*y).replace(Mul, lambda *args: sin(2*Mul(*args)))
            sin(2*x*y)

        2.1. pattern -> expr
            obj.replace(pattern(wild), expr(wild))

            Replace subexpressions matching ``pattern`` with the expression
            written in terms of the Wild symbols in ``pattern``.

            >>> a, b = map(Wild, 'ab')
            >>> f.replace(sin(a), tan(a))
            log(tan(x)) + tan(tan(x**2))
            >>> f.replace(sin(a), tan(a/2))
            log(tan(x/2)) + tan(tan(x**2/2))
            >>> f.replace(sin(a), a)
            log(x) + tan(x**2)
            >>> (x*y).replace(a*x, a)
            y

            Matching is exact by default when more than one Wild symbol
            is used: matching fails unless the match gives non-zero
            values for all Wild symbols:

            >>> (2*x + y).replace(a*x + b, b - a)
            y - 2
            >>> (2*x).replace(a*x + b, b - a)
            2*x

            When set to False, the results may be non-intuitive:

            >>> (2*x).replace(a*x + b, b - a, exact=False)
            2/x

        2.2. pattern -> func
            obj.replace(pattern(wild), lambda wild: expr(wild))

            All behavior is the same as in 2.1 but now a function in terms of
            pattern variables is used rather than an expression:

            >>> f.replace(sin(a), lambda a: sin(2*a))
            log(sin(2*x)) + tan(sin(2*x**2))

        3.1. func -> func
            obj.replace(filter, func)

            Replace subexpression ``e`` with ``func(e)`` if ``filter(e)``
            is True.

            >>> g = 2*sin(x**3)
            >>> g.replace(lambda expr: expr.is_Number, lambda expr: expr**2)
            4*sin(x**9)

        The expression itself is also targeted by the query but is done in
        such a fashion that changes are not made twice.

            >>> e = x*(x*y + 1)
            >>> e.replace(lambda x: x.is_Mul, lambda x: 2*x)
            2*x*(2*x*y + 1)

        When matching a single symbol, `exact` will default to True, but
        this may or may not be the behavior that is desired:

        Here, we want `exact=False`:

        >>> from sympy import Function
        >>> f = Function('f')
        >>> e = f(1) + f(0)
        >>> q = f(a), lambda a: f(a + 1)
        >>> e.replace(*q, exact=False)
        f(1) + f(2)
        >>> e.replace(*q, exact=True)
        f(0) + f(2)

        But here, the nature of matching makes selecting
        the right setting tricky:

        >>> e = x**(1 + y)
        >>> (x**(1 + y)).replace(x**(1 + a), lambda a: x**-a, exact=False)
        x
        >>> (x**(1 + y)).replace(x**(1 + a), lambda a: x**-a, exact=True)
        x**(-x - y + 1)
        >>> (x**y).replace(x**(1 + a), lambda a: x**-a, exact=False)
        x
        >>> (x**y).replace(x**(1 + a), lambda a: x**-a, exact=True)
        x**(1 - y)

        It is probably better to use a different form of the query
        that describes the target expression more precisely:

        >>> (1 + x**(1 + y)).replace(
        ... lambda x: x.is_Pow and x.exp.is_Add and x.exp.args[0] == 1,
        ... lambda x: x.base**(1 - (x.exp - 1)))
        ...
        x**(1 - y) + 1

        See Also
        ========

        subs: substitution of subexpressions as defined by the objects
              themselves.
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules

        """

        try:
            query = _sympify(query)
        except SympifyError:
            pass
        try:
            value = _sympify(value)
        except SympifyError:
            pass
        if isinstance(query, type):
            _query = lambda expr: isinstance(expr, query)

            if isinstance(value, type):
                _value = lambda expr, result: value(*expr.args)
            elif callable(value):
                _value = lambda expr, result: value(*expr.args)
            else:
                raise TypeError(
                    "given a type, replace() expects another "
                    "type or a callable")
        elif isinstance(query, Basic):
            _query = lambda expr: expr.match(query)
            if exact is None:
                from .symbol import Wild
                exact = (len(query.atoms(Wild)) > 1)

            if isinstance(value, Basic):
                if exact:
                    _value = lambda expr, result: (value.subs(result)
                        if all(result.values()) else expr)
                else:
                    _value = lambda expr, result: value.subs(result)
            elif callable(value):
                # match dictionary keys get the trailing underscore stripped
                # from them and are then passed as keywords to the callable;
                # if ``exact`` is True, only accept match if there are no null
                # values amongst those matched.
                if exact:
                    _value = lambda expr, result: (value(**
                        {str(k)[:-1]: v for k, v in result.items()})
                        if all(val for val in result.values()) else expr)
                else:
                    _value = lambda expr, result: value(**
                        {str(k)[:-1]: v for k, v in result.items()})
            else:
                raise TypeError(
                    "given an expression, replace() expects "
                    "another expression or a callable")
        elif callable(query):
            _query = query

            if callable(value):
                _value = lambda expr, result: value(expr)
            else:
                raise TypeError(
                    "given a callable, replace() expects "
                    "another callable")
        else:
            raise TypeError(
                "first argument to replace() must be a "
                "type, an expression or a callable")

        def walk(rv, F):
            """Apply ``F`` to args and then to result.
            """
            args = getattr(rv, 'args', None)
            if args is not None:
                if args:
                    newargs = tuple([walk(a, F) for a in args])
                    if args != newargs:
                        rv = rv.func(*newargs)
                        if simultaneous:
                            # if rv is something that was already
                            # matched (that was changed) then skip
                            # applying F again
                            for i, e in enumerate(args):
                                if rv == e and e != newargs[i]:
                                    return rv
                rv = F(rv)
            return rv

        mapping = {}  # changes that took place

        def rec_replace(expr):
            result = _query(expr)
            if result or result == {}:
                v = _value(expr, result)
                if v is not None and v != expr:
                    if map:
                        mapping[expr] = v
                    expr = v
            return expr

        rv = walk(self, rec_replace)
        return (rv, mapping) if map else rv # type: ignore

    def find(self, query, group=False):
        """Find all subexpressions matching a query."""
        query = _make_find_query(query)
        results = list(filter(query, _preorder_traversal(self)))

        if not group:
            return set(results)
        return dict(Counter(results))

    def count(self, query):
        """Count the number of matching subexpressions."""
        query = _make_find_query(query)
        return sum(bool(query(sub)) for sub in _preorder_traversal(self))

    def matches(self, expr, repl_dict=None, old=False):
        """
        Helper method for match() that looks for a match between Wild symbols
        in self and expressions in expr.

        Examples
        ========

        >>> from sympy import symbols, Wild, Basic
        >>> a, b, c = symbols('a b c')
        >>> x = Wild('x')
        >>> Basic(a + x, x).matches(Basic(a + b, c)) is None
        True
        >>> Basic(a + x, x).matches(Basic(a + b + c, b + c))
        {x_: b + c}
        """
        expr = sympify(expr)
        if not isinstance(expr, self.__class__):
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        if self == expr:
            return repl_dict

        if len(self.args) != len(expr.args):
            return None

        d = repl_dict  # already a copy
        for arg, other_arg in zip(self.args, expr.args):
            if arg == other_arg:
                continue
            if arg.is_Relational:
                try:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
                except TypeError: # Should be InvalidComparisonError when introduced
                    d = None
            else:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
            if d is None:
                return None
        return d

    def match(self, pattern, old=False):
        """
        Pattern matching.

        Wild symbols match all.

        Return ``None`` when expression (self) does not match with pattern.
        Otherwise return a dictionary such that::

          pattern.xreplace(self.match(pattern)) == self

        Examples
        ========

        >>> from sympy import Wild, Sum
        >>> from sympy.abc import x, y
        >>> p = Wild("p")
        >>> q = Wild("q")
        >>> r = Wild("r")
        >>> e = (x+y)**(x+y)
        >>> e.match(p**p)
        {p_: x + y}
        >>> e.match(p**q)
        {p_: x + y, q_: x + y}
        >>> e = (2*x)**2
        >>> e.match(p*q**r)
        {p_: 4, q_: x, r_: 2}
        >>> (p*q**r).xreplace(e.match(p*q**r))
        4*x**2

        Since match is purely structural expressions that are equivalent up to
        bound symbols will not match:

        >>> print(Sum(x, (x, 1, 2)).match(Sum(y, (y, 1, p))))
        None

        An expression with bound symbols can be matched if the pattern uses
        a distinct ``Wild`` for each bound symbol:

        >>> Sum(x, (x, 1, 2)).match(Sum(q, (q, 1, p)))
        {p_: 2, q_: x}

        The ``old`` flag will give the old-style pattern matching where
        expressions and patterns are essentially solved to give the match. Both
        of the following give None unless ``old=True``:

        >>> (x - 2).match(p - x, old=True)
        {p_: 2*x - 2}
        >>> (2/x).match(p*x, old=True)
        {p_: 2/x**2}

        See Also
        ========

        matches: pattern.matches(expr) is the same as expr.match(pattern)
        xreplace: exact structural replacement
        replace: structural replacement with pattern matching
        Wild: symbolic placeholders for expressions in pattern matching
        """
        pattern = sympify(pattern)
        return pattern.matches(self, old=old)

    def count_ops(self, visual=False):
        """Wrapper for count_ops that returns the operation count."""
        from .function import count_ops
        return count_ops(self, visual)

    def doit(self, **hints):
        """Evaluate objects that are not evaluated by default like limits,
        integrals, sums and products. All objects of this kind will be
        evaluated recursively, unless some species were excluded via 'hints'
        or unless the 'deep' hint was set to 'False'.

        >>> from sympy import Integral
        >>> from sympy.abc import x

        >>> 2*Integral(x, x)
        2*Integral(x, x)

        >>> (2*Integral(x, x)).doit()
        x**2

        >>> (2*Integral(x, x)).doit(deep=False)
        2*Integral(x, x)

        """
        if hints.get('deep', True):
            terms = [term.doit(**hints) if isinstance(term, Basic) else term
                                         for term in self.args]
            return self.func(*terms)
        else:
            return self

    def simplify(self, **kwargs) -> Basic:
        """See the simplify function in sympy.simplify"""
        from sympy.simplify.simplify import simplify
        return simplify(self, **kwargs)

    def refine(self, assumption=True):
        """See the refine function in sympy.assumptions"""
        from sympy.assumptions.refine import refine
        return refine(self, assumption)

    def _eval_derivative_n_times(self, s, n):
        # This is the default evaluator for derivatives (as called by `diff`
        # and `Derivative`), it will attempt a loop to derive the expression
        # `n` times by calling the corresponding `_eval_derivative` method,
        # while leaving the derivative unevaluated if `n` is symbolic.  This
        # method should be overridden if the object has a closed form for its
        # symbolic n-th derivative.
        from .numbers import Integer
        if isinstance(n, (int, Integer)):
            obj = self
            for i in range(n):
                prev = obj
                obj = obj._eval_derivative(s)
                if obj is None:
                    return None
                elif obj == prev:
                    break
            return obj
        else:
            return None

    def rewrite(self, *args, deep=True, **hints):
        """
        Rewrite *self* using a defined rule.

        Rewriting transforms an expression to another, which is mathematically
        equivalent but structurally different. For example you can rewrite
        trigonometric functions as complex exponentials or combinatorial
        functions as gamma function.

        This method takes a *pattern* and a *rule* as positional arguments.
        *pattern* is optional parameter which defines the types of expressions
        that will be transformed. If it is not passed, all possible expressions
        will be rewritten. *rule* defines how the expression will be rewritten.

        Parameters
        ==========

        args : Expr
            A *rule*, or *pattern* and *rule*.
            - *pattern* is a type or an iterable of types.
            - *rule* can be any object.

        deep : bool, optional
            If ``True``, subexpressions are recursively transformed. Default is
            ``True``.

        Examples
        ========

        If *pattern* is unspecified, all possible expressions are transformed.

        >>> from sympy import cos, sin, exp, I
        >>> from sympy.abc import x
        >>> expr = cos(x) + I*sin(x)
        >>> expr.rewrite(exp)
        exp(I*x)

        Pattern can be a type or an iterable of types.

        >>> expr.rewrite(sin, exp)
        exp(I*x)/2 + cos(x) - exp(-I*x)/2
        >>> expr.rewrite([cos,], exp)
        exp(I*x)/2 + I*sin(x) + exp(-I*x)/2
        >>> expr.rewrite([cos, sin], exp)
        exp(I*x)

        Rewriting behavior can be implemented by defining ``_eval_rewrite()``
        method.

        >>> from sympy import Expr, sqrt, pi
        >>> class MySin(Expr):
        ...     def _eval_rewrite(self, rule, args, **hints):
        ...         x, = args
        ...         if rule == cos:
        ...             return cos(pi/2 - x, evaluate=False)
        ...         if rule == sqrt:
        ...             return sqrt(1 - cos(x)**2)
        >>> MySin(MySin(x)).rewrite(cos)
        cos(-cos(-x + pi/2) + pi/2)
        >>> MySin(x).rewrite(sqrt)
        sqrt(1 - cos(x)**2)

        Defining ``_eval_rewrite_as_[...]()`` method is supported for backwards
        compatibility reason. This may be removed in the future and using it is
        discouraged.

        >>> class MySin(Expr):
        ...     def _eval_rewrite_as_cos(self, *args, **hints):
        ...         x, = args
        ...         return cos(pi/2 - x, evaluate=False)
        >>> MySin(x).rewrite(cos)
        cos(-x + pi/2)

        """
        if not args:
            return self

        hints.update(deep=deep)

        pattern = args[:-1]
        rule = args[-1]

        # Special case: map `abs` to `Abs`
        if rule is abs:
            from sympy.functions.elementary.complexes import Abs
            rule = Abs

        # support old design by _eval_rewrite_as_[...] method
        if isinstance(rule, str):
            method = "_eval_rewrite_as_%s" % rule
        elif hasattr(rule, "__name__"):
            # rule is class or function
            clsname = rule.__name__
            method = "_eval_rewrite_as_%s" % clsname
        else:
            # rule is instance
            clsname = rule.__class__.__name__
            method = "_eval_rewrite_as_%s" % clsname

        if pattern:
            if iterable(pattern[0]):
                pattern = pattern[0]
            pattern = tuple(p for p in pattern if self.has(p))
            if not pattern:
                return self
        # hereafter, empty pattern is interpreted as all pattern.

        return self._rewrite(pattern, rule, method, **hints)

    def _rewrite(self, pattern, rule, method, **hints):
        deep = hints.pop('deep', True)
        if deep:
            args = [a._rewrite(pattern, rule, method, **hints)
                    for a in self.args]
        else:
            args = self.args
        if not pattern or any(isinstance(self, p) for p in pattern):
            meth = getattr(self, method, None)
            if meth is not None:
                rewritten = meth(*args, **hints)
            else:
                rewritten = self._eval_rewrite(rule, args, **hints)
            if rewritten is not None:
                return rewritten
        if not args:
            return self
        return self.func(*args)

    def _eval_rewrite(self, rule, args, **hints):
        return None

    _constructor_postprocessor_mapping = {}  # type: ignore

    @classmethod
    def _exec_constructor_postprocessors(cls, obj):
        # WARNING: This API is experimental.

        # This is an experimental API that introduces constructor
        # postprosessors for SymPy Core elements. If an argument of a SymPy
        # expression has a `_constructor_postprocessor_mapping` attribute, it will
        # be interpreted as a dictionary containing lists of postprocessing
        # functions for matching expression node names.

        clsname = obj.__class__.__name__
        postprocessors = {f for i in obj.args
                            for f in _get_postprocessors(clsname, type(i))}
        for f in postprocessors:
            obj = f(obj)

        return obj

    def _sage_(self):
        """
        Convert *self* to a symbolic expression of SageMath.

        This version of the method is merely a placeholder.
        """
        old_method = self._sage_
        from sage.interfaces.sympy import sympy_init # type: ignore
        sympy_init()  # may monkey-patch _sage_ method into self's class or superclasses
        if old_method == self._sage_:
            raise NotImplementedError('conversion to SageMath is not implemented')
        else:
            # call the freshly monkey-patched method
            return self._sage_()

    def could_extract_minus_sign(self) -> bool:
        return False  # see Expr.could_extract_minus_sign

    def is_same(a, b, approx=None):
        """Return True if a and b are structurally the same, else False.
        If `approx` is supplied, it will be used to test whether two
        numbers are the same or not. By default, only numbers of the
        same type will compare equal, so S.Half != Float(0.5).

        Examples
        ========

        In SymPy (unlike Python) two numbers do not compare the same if they are
        not of the same type:

        >>> from sympy import S
        >>> 2.0 == S(2)
        False
        >>> 0.5 == S.Half
        False

        By supplying a function with which to compare two numbers, such
        differences can be ignored. e.g. `equal_valued` will return True
        for decimal numbers having a denominator that is a power of 2,
        regardless of precision.

        >>> from sympy import Float
        >>> from sympy.core.numbers import equal_valued
        >>> (S.Half/4).is_same(Float(0.125, 1), equal_valued)
        True
        >>> Float(1, 2).is_same(Float(1, 10), equal_valued)
        True

        But decimals without a power of 2 denominator will compare
        as not being the same.

        >>> Float(0.1, 9).is_same(Float(0.1, 10), equal_valued)
        False

        But arbitrary differences can be ignored by supplying a function
        to test the equivalence of two numbers:

        >>> import math
        >>> Float(0.1, 9).is_same(Float(0.1, 10), math.isclose)
        True

        Other objects might compare the same even though types are not the
        same. This routine will only return True if two expressions are
        identical in terms of class types.

        >>> from sympy import eye, Basic
        >>> eye(1) == S(eye(1))  # mutable vs immutable
        True
        >>> Basic.is_same(eye(1), S(eye(1)))
        False

        """
        from .numbers import Number
        from .traversal import postorder_traversal as pot
        for t in zip_longest(pot(a), pot(b)):
            if None in t:
                return False
            a, b = t
            if isinstance(a, Number):
                if not isinstance(b, Number):
                    return False
                if approx:
                    return approx(a, b)
            if not (a == b and a.__class__ == b.__class__):
                return False
        return True

_aresame = Basic.is_same  # for sake of others importing this

# key used by Mul and Add to make canonical args
_args_sortkey = cmp_to_key(Basic.compare)

# For all Basic subclasses _prepare_class_assumptions is called by
# Basic.__init_subclass__ but that method is not called for Basic itself so we
# call the function here instead.
_prepare_class_assumptions(Basic)


class Atom(Basic):
    """
    A parent class for atomic things. An atom is an expression with no subexpressions.

    Examples
    ========

    Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """

    is_Atom = True

    __slots__ = ()

    def matches(self, expr, repl_dict=None, old=False):
        if self == expr:
            if repl_dict is None:
                return {}
            return repl_dict.copy()

    def xreplace(self, rule, hack2=False):
        return rule.get(self, self)

    def doit(self, **hints):
        return self

    @classmethod
    def class_key(cls):
        return 2, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (str(self),)), S.One.sort_key(), S.One

    def _eval_simplify(self, **kwargs):
        return self

    @property
    def _sorted_args(self):
        # this is here as a safeguard against accidentally using _sorted_args
        # on Atoms -- they cannot be rebuilt as atom.func(*atom._sorted_args)
        # since there are no args. So the calling routine should be checking
        # to see that this property is not called for Atoms.
        raise AttributeError('Atoms have no args. It might be necessary'
        ' to make a check for Atoms in the calling code.')


def _atomic(e, recursive=False):
    """Return atom-like quantities as far as substitution is
    concerned: Derivatives, Functions and Symbols. Do not
    return any 'atoms' that are inside such quantities unless
    they also appear outside, too, unless `recursive` is True.

    Examples
    ========

    >>> from sympy import Derivative, Function, cos
    >>> from sympy.abc import x, y
    >>> from sympy.core.basic import _atomic
    >>> f = Function('f')
    >>> _atomic(x + y)
    {x, y}
    >>> _atomic(x + f(y))
    {x, f(y)}
    >>> _atomic(Derivative(f(x), x) + cos(x) + y)
    {y, cos(x), Derivative(f(x), x)}

    """
    pot = _preorder_traversal(e)
    seen = set()
    if isinstance(e, Basic):
        free = getattr(e, "free_symbols", None)
        if free is None:
            return {e}
    else:
        return set()
    from .symbol import Symbol
    from .function import Derivative, Function
    atoms = set()
    for p in pot:
        if p in seen:
            pot.skip()
            continue
        seen.add(p)
        if isinstance(p, Symbol) and p in free:
            atoms.add(p)
        elif isinstance(p, (Derivative, Function)):
            if not recursive:
                pot.skip()
            atoms.add(p)
    return atoms


def _make_find_query(query):
    """Convert the argument of Basic.find() into a callable"""
    try:
        query = _sympify(query)
    except SympifyError:
        pass
    if isinstance(query, type):
        return lambda expr: isinstance(expr, query)
    elif isinstance(query, Basic):
        return lambda expr: expr.match(query) is not None
    return query

# Delayed to avoid cyclic import
from .singleton import S
from .traversal import (preorder_traversal as _preorder_traversal,
   iterargs, iterfreeargs)

preorder_traversal = deprecated(
    """
    Using preorder_traversal from the sympy.core.basic submodule is
    deprecated.

    Instead, use preorder_traversal from the top-level sympy namespace, like

        sympy.preorder_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved",
)(_preorder_traversal)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\dtypes\dtypes.py ===
"""
Define extension dtypes.
"""
from __future__ import annotations

from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
import re
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import numpy as np
import pytz

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.interval import Interval
from pandas._libs.properties import cache_readonly
from pandas._libs.tslibs import (
    BaseOffset,
    NaT,
    NaTType,
    Period,
    Timedelta,
    Timestamp,
    timezones,
    to_offset,
    tz_compare,
)
from pandas._libs.tslibs.dtypes import (
    PeriodDtypeBase,
    abbrev_to_npy_unit,
)
from pandas._libs.tslibs.offsets import BDay
from pandas.compat import pa_version_under10p1
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import (
    ExtensionDtype,
    StorageExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCIndex,
    ABCRangeIndex,
)
from pandas.core.dtypes.inference import (
    is_bool,
    is_list_like,
)

from pandas.util import capitalize_first_letter

if not pa_version_under10p1:
    import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from datetime import tzinfo

    import pyarrow as pa  # noqa: TCH004

    from pandas._typing import (
        Dtype,
        DtypeObj,
        IntervalClosedType,
        Ordered,
        Self,
        npt,
        type_t,
    )

    from pandas import (
        Categorical,
        CategoricalIndex,
        DatetimeIndex,
        Index,
        IntervalIndex,
        PeriodIndex,
    )
    from pandas.core.arrays import (
        BaseMaskedArray,
        DatetimeArray,
        IntervalArray,
        NumpyExtensionArray,
        PeriodArray,
        SparseArray,
    )
    from pandas.core.arrays.arrow import ArrowExtensionArray

str_type = str


class PandasExtensionDtype(ExtensionDtype):
    """
    A np.dtype duck-typed class, suitable for holding a custom dtype.

    THIS IS NOT A REAL NUMPY DTYPE
    """

    type: Any
    kind: Any
    # The Any type annotations above are here only because mypy seems to have a
    # problem dealing with multiple inheritance from PandasExtensionDtype
    # and ExtensionDtype's @properties in the subclasses below. The kind and
    # type variables in those subclasses are explicitly typed below.
    subdtype = None
    str: str_type
    num = 100
    shape: tuple[int, ...] = ()
    itemsize = 8
    base: DtypeObj | None = None
    isbuiltin = 0
    isnative = 0
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}

    def __repr__(self) -> str_type:
        """
        Return a string representation for a particular object.
        """
        return str(self)

    def __hash__(self) -> int:
        raise NotImplementedError("sub-classes should implement an __hash__ method")

    def __getstate__(self) -> dict[str_type, Any]:
        # pickle support; we don't want to pickle the cache
        return {k: getattr(self, k, None) for k in self._metadata}

    @classmethod
    def reset_cache(cls) -> None:
        """clear the cache"""
        cls._cache_dtypes = {}


class CategoricalDtypeType(type):
    """
    the type of CategoricalDtype, this metaclass determines subclass ability
    """


@register_extension_dtype
class CategoricalDtype(PandasExtensionDtype, ExtensionDtype):
    """
    Type for categorical data with the categories and orderedness.

    Parameters
    ----------
    categories : sequence, optional
        Must be unique, and must not contain any nulls.
        The categories are stored in an Index,
        and if an index is provided the dtype of that index will be used.
    ordered : bool or None, default False
        Whether or not this categorical is treated as a ordered categorical.
        None can be used to maintain the ordered value of existing categoricals when
        used in operations that combine categoricals, e.g. astype, and will resolve to
        False if there is no existing ordered to maintain.

    Attributes
    ----------
    categories
    ordered

    Methods
    -------
    None

    See Also
    --------
    Categorical : Represent a categorical variable in classic R / S-plus fashion.

    Notes
    -----
    This class is useful for specifying the type of a ``Categorical``
    independent of the values. See :ref:`categorical.categoricaldtype`
    for more.

    Examples
    --------
    >>> t = pd.CategoricalDtype(categories=['b', 'a'], ordered=True)
    >>> pd.Series(['a', 'b', 'a', 'c'], dtype=t)
    0      a
    1      b
    2      a
    3    NaN
    dtype: category
    Categories (2, object): ['b' < 'a']

    An empty CategoricalDtype with a specific dtype can be created
    by providing an empty index. As follows,

    >>> pd.CategoricalDtype(pd.DatetimeIndex([])).categories.dtype
    dtype('<M8[ns]')
    """

    # TODO: Document public vs. private API
    name = "category"
    type: type[CategoricalDtypeType] = CategoricalDtypeType
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    _metadata = ("categories", "ordered")
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _supports_2d = False
    _can_fast_transpose = False

    def __init__(self, categories=None, ordered: Ordered = False) -> None:
        self._finalize(categories, ordered, fastpath=False)

    @classmethod
    def _from_fastpath(
        cls, categories=None, ordered: bool | None = None
    ) -> CategoricalDtype:
        self = cls.__new__(cls)
        self._finalize(categories, ordered, fastpath=True)
        return self

    @classmethod
    def _from_categorical_dtype(
        cls, dtype: CategoricalDtype, categories=None, ordered: Ordered | None = None
    ) -> CategoricalDtype:
        if categories is ordered is None:
            return dtype
        if categories is None:
            categories = dtype.categories
        if ordered is None:
            ordered = dtype.ordered
        return cls(categories, ordered)

    @classmethod
    def _from_values_or_dtype(
        cls,
        values=None,
        categories=None,
        ordered: bool | None = None,
        dtype: Dtype | None = None,
    ) -> CategoricalDtype:
        """
        Construct dtype from the input parameters used in :class:`Categorical`.

        This constructor method specifically does not do the factorization
        step, if that is needed to find the categories. This constructor may
        therefore return ``CategoricalDtype(categories=None, ordered=None)``,
        which may not be useful. Additional steps may therefore have to be
        taken to create the final dtype.

        The return dtype is specified from the inputs in this prioritized
        order:
        1. if dtype is a CategoricalDtype, return dtype
        2. if dtype is the string 'category', create a CategoricalDtype from
           the supplied categories and ordered parameters, and return that.
        3. if values is a categorical, use value.dtype, but override it with
           categories and ordered if either/both of those are not None.
        4. if dtype is None and values is not a categorical, construct the
           dtype from categories and ordered, even if either of those is None.

        Parameters
        ----------
        values : list-like, optional
            The list-like must be 1-dimensional.
        categories : list-like, optional
            Categories for the CategoricalDtype.
        ordered : bool, optional
            Designating if the categories are ordered.
        dtype : CategoricalDtype or the string "category", optional
            If ``CategoricalDtype``, cannot be used together with
            `categories` or `ordered`.

        Returns
        -------
        CategoricalDtype

        Examples
        --------
        >>> pd.CategoricalDtype._from_values_or_dtype()
        CategoricalDtype(categories=None, ordered=None, categories_dtype=None)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     categories=['a', 'b'], ordered=True
        ... )
        CategoricalDtype(categories=['a', 'b'], ordered=True, categories_dtype=object)
        >>> dtype1 = pd.CategoricalDtype(['a', 'b'], ordered=True)
        >>> dtype2 = pd.CategoricalDtype(['x', 'y'], ordered=False)
        >>> c = pd.Categorical([0, 1], dtype=dtype1)
        >>> pd.CategoricalDtype._from_values_or_dtype(
        ...     c, ['x', 'y'], ordered=True, dtype=dtype2
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Cannot specify `categories` or `ordered` together with
        `dtype`.

        The supplied dtype takes precedence over values' dtype:

        >>> pd.CategoricalDtype._from_values_or_dtype(c, dtype=dtype2)
        CategoricalDtype(categories=['x', 'y'], ordered=False, categories_dtype=object)
        """

        if dtype is not None:
            # The dtype argument takes precedence over values.dtype (if any)
            if isinstance(dtype, str):
                if dtype == "category":
                    if ordered is None and cls.is_dtype(values):
                        # GH#49309 preserve orderedness
                        ordered = values.dtype.ordered

                    dtype = CategoricalDtype(categories, ordered)
                else:
                    raise ValueError(f"Unknown dtype {repr(dtype)}")
            elif categories is not None or ordered is not None:
                raise ValueError(
                    "Cannot specify `categories` or `ordered` together with `dtype`."
                )
            elif not isinstance(dtype, CategoricalDtype):
                raise ValueError(f"Cannot not construct CategoricalDtype from {dtype}")
        elif cls.is_dtype(values):
            # If no "dtype" was passed, use the one from "values", but honor
            # the "ordered" and "categories" arguments
            dtype = values.dtype._from_categorical_dtype(
                values.dtype, categories, ordered
            )
        else:
            # If dtype=None and values is not categorical, create a new dtype.
            # Note: This could potentially have categories=None and
            # ordered=None.
            dtype = CategoricalDtype(categories, ordered)

        return cast(CategoricalDtype, dtype)

    @classmethod
    def construct_from_string(cls, string: str_type) -> CategoricalDtype:
        """
        Construct a CategoricalDtype from a string.

        Parameters
        ----------
        string : str
            Must be the string "category" in order to be successfully constructed.

        Returns
        -------
        CategoricalDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a CategoricalDtype cannot be constructed from the input.
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if string != cls.name:
            raise TypeError(f"Cannot construct a 'CategoricalDtype' from '{string}'")

        # need ordered=None to ensure that operations specifying dtype="category" don't
        # override the ordered value for existing categoricals
        return cls(ordered=None)

    def _finalize(self, categories, ordered: Ordered, fastpath: bool = False) -> None:
        if ordered is not None:
            self.validate_ordered(ordered)

        if categories is not None:
            categories = self.validate_categories(categories, fastpath=fastpath)

        self._categories = categories
        self._ordered = ordered

    def __setstate__(self, state: MutableMapping[str_type, Any]) -> None:
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._categories = state.pop("categories", None)
        self._ordered = state.pop("ordered", False)

    def __hash__(self) -> int:
        # _hash_categories returns a uint64, so use the negative
        # space for when we have unknown categories to avoid a conflict
        if self.categories is None:
            if self.ordered:
                return -1
            else:
                return -2
        # We *do* want to include the real self.ordered here
        return int(self._hash_categories)

    def __eq__(self, other: object) -> bool:
        """
        Rules for CDT equality:
        1) Any CDT is equal to the string 'category'
        2) Any CDT is equal to itself
        3) Any CDT is equal to a CDT with categories=None regardless of ordered
        4) A CDT with ordered=True is only equal to another CDT with
           ordered=True and identical categories in the same order
        5) A CDT with ordered={False, None} is only equal to another CDT with
           ordered={False, None} and identical categories, but same order is
           not required. There is no distinction between False/None.
        6) Any other comparison returns False
        """
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not (hasattr(other, "ordered") and hasattr(other, "categories")):
            return False
        elif self.categories is None or other.categories is None:
            # For non-fully-initialized dtypes, these are only equal to
            #  - the string "category" (handled above)
            #  - other CategoricalDtype with categories=None
            return self.categories is other.categories
        elif self.ordered or other.ordered:
            # At least one has ordered=True; equal if both have ordered=True
            # and the same values for categories in the same order.
            return (self.ordered == other.ordered) and self.categories.equals(
                other.categories
            )
        else:
            # Neither has ordered=True; equal if both have the same categories,
            # but same order is not necessary.  There is no distinction between
            # ordered=False and ordered=None: CDT(., False) and CDT(., None)
            # will be equal if they have the same categories.
            left = self.categories
            right = other.categories

            # GH#36280 the ordering of checks here is for performance
            if not left.dtype == right.dtype:
                return False

            if len(left) != len(right):
                return False

            if self.categories.equals(other.categories):
                # Check and see if they happen to be identical categories
                return True

            if left.dtype != object:
                # Faster than calculating hash
                indexer = left.get_indexer(right)
                # Because left and right have the same length and are unique,
                #  `indexer` not having any -1s implies that there is a
                #  bijection between `left` and `right`.
                return bool((indexer != -1).all())

            # With object-dtype we need a comparison that identifies
            #  e.g. int(2) as distinct from float(2)
            return set(left) == set(right)

    def __repr__(self) -> str_type:
        if self.categories is None:
            data = "None"
            dtype = "None"
        else:
            data = self.categories._format_data(name=type(self).__name__)
            if isinstance(self.categories, ABCRangeIndex):
                data = str(self.categories._range)
            data = data.rstrip(", ")
            dtype = self.categories.dtype

        return (
            f"CategoricalDtype(categories={data}, ordered={self.ordered}, "
            f"categories_dtype={dtype})"
        )

    @cache_readonly
    def _hash_categories(self) -> int:
        from pandas.core.util.hashing import (
            combine_hash_arrays,
            hash_array,
            hash_tuples,
        )

        categories = self.categories
        ordered = self.ordered

        if len(categories) and isinstance(categories[0], tuple):
            # assumes if any individual category is a tuple, then all our. ATM
            # I don't really want to support just some of the categories being
            # tuples.
            cat_list = list(categories)  # breaks if a np.array of categories
            cat_array = hash_tuples(cat_list)
        else:
            if categories.dtype == "O" and len({type(x) for x in categories}) != 1:
                # TODO: hash_array doesn't handle mixed types. It casts
                # everything to a str first, which means we treat
                # {'1', '2'} the same as {'1', 2}
                # find a better solution
                hashed = hash((tuple(categories), ordered))
                return hashed

            if DatetimeTZDtype.is_dtype(categories.dtype):
                # Avoid future warning.
                categories = categories.view("datetime64[ns]")

            cat_array = hash_array(np.asarray(categories), categorize=False)
        if ordered:
            cat_array = np.vstack(
                [cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)]
            )
        else:
            cat_array = np.array([cat_array])
        combined_hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
        return np.bitwise_xor.reduce(combined_hashed)

    @classmethod
    def construct_array_type(cls) -> type_t[Categorical]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas import Categorical

        return Categorical

    @staticmethod
    def validate_ordered(ordered: Ordered) -> None:
        """
        Validates that we have a valid ordered parameter. If
        it is not a boolean, a TypeError will be raised.

        Parameters
        ----------
        ordered : object
            The parameter to be verified.

        Raises
        ------
        TypeError
            If 'ordered' is not a boolean.
        """
        if not is_bool(ordered):
            raise TypeError("'ordered' must either be 'True' or 'False'")

    @staticmethod
    def validate_categories(categories, fastpath: bool = False) -> Index:
        """
        Validates that we have good categories

        Parameters
        ----------
        categories : array-like
        fastpath : bool
            Whether to skip nan and uniqueness checks

        Returns
        -------
        categories : Index
        """
        from pandas.core.indexes.base import Index

        if not fastpath and not is_list_like(categories):
            raise TypeError(
                f"Parameter 'categories' must be list-like, was {repr(categories)}"
            )
        if not isinstance(categories, ABCIndex):
            categories = Index._with_infer(categories, tupleize_cols=False)

        if not fastpath:
            if categories.hasnans:
                raise ValueError("Categorical categories cannot be null")

            if not categories.is_unique:
                raise ValueError("Categorical categories must be unique")

        if isinstance(categories, ABCCategoricalIndex):
            categories = categories.categories

        return categories

    def update_dtype(self, dtype: str_type | CategoricalDtype) -> CategoricalDtype:
        """
        Returns a CategoricalDtype with categories and ordered taken from dtype
        if specified, otherwise falling back to self if unspecified

        Parameters
        ----------
        dtype : CategoricalDtype

        Returns
        -------
        new_dtype : CategoricalDtype
        """
        if isinstance(dtype, str) and dtype == "category":
            # dtype='category' should not change anything
            return self
        elif not self.is_dtype(dtype):
            raise ValueError(
                f"a CategoricalDtype must be passed to perform an update, "
                f"got {repr(dtype)}"
            )
        else:
            # from here on, dtype is a CategoricalDtype
            dtype = cast(CategoricalDtype, dtype)

        # update categories/ordered unless they've been explicitly passed as None
        new_categories = (
            dtype.categories if dtype.categories is not None else self.categories
        )
        new_ordered = dtype.ordered if dtype.ordered is not None else self.ordered

        return CategoricalDtype(new_categories, new_ordered)

    @property
    def categories(self) -> Index:
        """
        An ``Index`` containing the unique categories allowed.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=True)
        >>> cat_type.categories
        Index(['a', 'b'], dtype='object')
        """
        return self._categories

    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=True)
        >>> cat_type.ordered
        True

        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=False)
        >>> cat_type.ordered
        False
        """
        return self._ordered

    @property
    def _is_boolean(self) -> bool:
        from pandas.core.dtypes.common import is_bool_dtype

        return is_bool_dtype(self.categories)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # check if we have all categorical dtype with identical categories
        if all(isinstance(x, CategoricalDtype) for x in dtypes):
            first = dtypes[0]
            if all(first == other for other in dtypes[1:]):
                return first

        # special case non-initialized categorical
        # TODO we should figure out the expected return value in general
        non_init_cats = [
            isinstance(x, CategoricalDtype) and x.categories is None for x in dtypes
        ]
        if all(non_init_cats):
            return self
        elif any(non_init_cats):
            return None

        # categorical is aware of Sparse -> extract sparse subdtypes
        dtypes = [x.subtype if isinstance(x, SparseDtype) else x for x in dtypes]
        # extract the categories' dtype
        non_cat_dtypes = [
            x.categories.dtype if isinstance(x, CategoricalDtype) else x for x in dtypes
        ]
        # TODO should categorical always give an answer?
        from pandas.core.dtypes.cast import find_common_type

        return find_common_type(non_cat_dtypes)

    @cache_readonly
    def index_class(self) -> type_t[CategoricalIndex]:
        from pandas import CategoricalIndex

        return CategoricalIndex


@register_extension_dtype
class DatetimeTZDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for timezone-aware datetime data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    unit : str, default "ns"
        The precision of the datetime data. Currently limited
        to ``"ns"``.
    tz : str, int, or datetime.tzinfo
        The timezone.

    Attributes
    ----------
    unit
    tz

    Methods
    -------
    None

    Raises
    ------
    ZoneInfoNotFoundError
        When the requested timezone cannot be found.

    Examples
    --------
    >>> from zoneinfo import ZoneInfo
    >>> pd.DatetimeTZDtype(tz=ZoneInfo('UTC'))
    datetime64[ns, UTC]

    >>> pd.DatetimeTZDtype(tz=ZoneInfo('Europe/Paris'))
    datetime64[ns, Europe/Paris]
    """

    type: type[Timestamp] = Timestamp
    kind: str_type = "M"
    num = 101
    _metadata = ("unit", "tz")
    _match = re.compile(r"(datetime64|M8)\[(?P<unit>.+), (?P<tz>.+)\]")
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _supports_2d = True
    _can_fast_transpose = True

    @property
    def na_value(self) -> NaTType:
        return NaT

    @cache_readonly
    def base(self) -> DtypeObj:  # type: ignore[override]
        return np.dtype(f"M8[{self.unit}]")

    # error: Signature of "str" incompatible with supertype "PandasExtensionDtype"
    @cache_readonly
    def str(self) -> str:  # type: ignore[override]
        return f"|M8[{self.unit}]"

    def __init__(self, unit: str_type | DatetimeTZDtype = "ns", tz=None) -> None:
        if isinstance(unit, DatetimeTZDtype):
            # error: "str" has no attribute "tz"
            unit, tz = unit.unit, unit.tz  # type: ignore[attr-defined]

        if unit != "ns":
            if isinstance(unit, str) and tz is None:
                # maybe a string like datetime64[ns, tz], which we support for
                # now.
                result = type(self).construct_from_string(unit)
                unit = result.unit
                tz = result.tz
                msg = (
                    f"Passing a dtype alias like 'datetime64[ns, {tz}]' "
                    "to DatetimeTZDtype is no longer supported. Use "
                    "'DatetimeTZDtype.construct_from_string()' instead."
                )
                raise ValueError(msg)
            if unit not in ["s", "ms", "us", "ns"]:
                raise ValueError("DatetimeTZDtype only supports s, ms, us, ns units")

        if tz:
            tz = timezones.maybe_get_tz(tz)
            tz = timezones.tz_standardize(tz)
        elif tz is not None:
            raise pytz.UnknownTimeZoneError(tz)
        if tz is None:
            raise TypeError("A 'tz' is required.")

        self._unit = unit
        self._tz = tz

    @cache_readonly
    def _creso(self) -> int:
        """
        The NPY_DATETIMEUNIT corresponding to this dtype's resolution.
        """
        return abbrev_to_npy_unit(self.unit)

    @property
    def unit(self) -> str_type:
        """
        The precision of the datetime data.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo('America/Los_Angeles'))
        >>> dtype.unit
        'ns'
        """
        return self._unit

    @property
    def tz(self) -> tzinfo:
        """
        The timezone.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo('America/Los_Angeles'))
        >>> dtype.tz
        zoneinfo.ZoneInfo(key='America/Los_Angeles')
        """
        return self._tz

    @classmethod
    def construct_array_type(cls) -> type_t[DatetimeArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import DatetimeArray

        return DatetimeArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> DatetimeTZDtype:
        """
        Construct a DatetimeTZDtype from a string.

        Parameters
        ----------
        string : str
            The string alias for this DatetimeTZDtype.
            Should be formatted like ``datetime64[ns, <tz>]``,
            where ``<tz>`` is the timezone name.

        Examples
        --------
        >>> DatetimeTZDtype.construct_from_string('datetime64[ns, UTC]')
        datetime64[ns, UTC]
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        match = cls._match.match(string)
        if match:
            d = match.groupdict()
            try:
                return cls(unit=d["unit"], tz=d["tz"])
            except (KeyError, TypeError, ValueError) as err:
                # KeyError if maybe_get_tz tries and fails to get a
                #  pytz timezone (actually pytz.UnknownTimeZoneError).
                # TypeError if we pass a nonsense tz;
                # ValueError if we pass a unit other than "ns"
                raise TypeError(msg) from err
        raise TypeError(msg)

    def __str__(self) -> str_type:
        return f"datetime64[{self.unit}, {self.tz}]"

    @property
    def name(self) -> str_type:
        """A string representation of the dtype."""
        return str(self)

    def __hash__(self) -> int:
        # make myself hashable
        # TODO: update this.
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            if other.startswith("M8["):
                other = f"datetime64[{other[3:]}"
            return other == self.name

        return (
            isinstance(other, DatetimeTZDtype)
            and self.unit == other.unit
            and tz_compare(self.tz, other.tz)
        )

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> DatetimeArray:
        """
        Construct DatetimeArray from pyarrow Array/ChunkedArray.

        Note: If the units in the pyarrow Array are the same as this
        DatetimeDtype, then values corresponding to the integer representation
        of ``NaT`` (e.g. one nanosecond before :attr:`pandas.Timestamp.min`)
        are converted to ``NaT``, regardless of the null indicator in the
        pyarrow array.

        Parameters
        ----------
        array : pyarrow.Array or pyarrow.ChunkedArray
            The Arrow array to convert to DatetimeArray.

        Returns
        -------
        extension array : DatetimeArray
        """
        import pyarrow

        from pandas.core.arrays import DatetimeArray

        array = array.cast(pyarrow.timestamp(unit=self._unit), safe=True)

        if isinstance(array, pyarrow.Array):
            np_arr = array.to_numpy(zero_copy_only=False)
        else:
            np_arr = array.to_numpy()

        return DatetimeArray._simple_new(np_arr, dtype=self)

    def __setstate__(self, state) -> None:
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._tz = state["tz"]
        self._unit = state["unit"]

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if all(isinstance(t, DatetimeTZDtype) and t.tz == self.tz for t in dtypes):
            np_dtype = np.max([cast(DatetimeTZDtype, t).base for t in [self, *dtypes]])
            unit = np.datetime_data(np_dtype)[0]
            return type(self)(unit=unit, tz=self.tz)
        return super()._get_common_dtype(dtypes)

    @cache_readonly
    def index_class(self) -> type_t[DatetimeIndex]:
        from pandas import DatetimeIndex

        return DatetimeIndex


@register_extension_dtype
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    """
    An ExtensionDtype for Period data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    freq : str or DateOffset
        The frequency of this PeriodDtype.

    Attributes
    ----------
    freq

    Methods
    -------
    None

    Examples
    --------
    >>> pd.PeriodDtype(freq='D')
    period[D]

    >>> pd.PeriodDtype(freq=pd.offsets.MonthEnd())
    period[M]
    """

    type: type[Period] = Period
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    num = 102
    _metadata = ("freq",)
    _match = re.compile(r"(P|p)eriod\[(?P<freq>.+)\]")
    # error: Incompatible types in assignment (expression has type
    # "Dict[int, PandasExtensionDtype]", base class "PandasExtensionDtype"
    # defined the type as "Dict[str, PandasExtensionDtype]")  [assignment]
    _cache_dtypes: dict[BaseOffset, int] = {}  # type: ignore[assignment]
    __hash__ = PeriodDtypeBase.__hash__
    _freq: BaseOffset
    _supports_2d = True
    _can_fast_transpose = True

    def __new__(cls, freq) -> PeriodDtype:  # noqa: PYI034
        """
        Parameters
        ----------
        freq : PeriodDtype, BaseOffset, or string
        """
        if isinstance(freq, PeriodDtype):
            return freq

        if not isinstance(freq, BaseOffset):
            freq = cls._parse_dtype_strict(freq)

        if isinstance(freq, BDay):
            # GH#53446
            # TODO(3.0): enforcing this will close GH#10575
            warnings.warn(
                "PeriodDtype[B] is deprecated and will be removed in a future "
                "version. Use a DatetimeIndex with freq='B' instead",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        try:
            dtype_code = cls._cache_dtypes[freq]
        except KeyError:
            dtype_code = freq._period_dtype_code
            cls._cache_dtypes[freq] = dtype_code
        u = PeriodDtypeBase.__new__(cls, dtype_code, freq.n)
        u._freq = freq
        return u

    def __reduce__(self) -> tuple[type_t[Self], tuple[str_type]]:
        return type(self), (self.name,)

    @property
    def freq(self) -> BaseOffset:
        """
        The frequency object of this PeriodDtype.

        Examples
        --------
        >>> dtype = pd.PeriodDtype(freq='D')
        >>> dtype.freq
        <Day>
        """
        return self._freq

    @classmethod
    def _parse_dtype_strict(cls, freq: str_type) -> BaseOffset:
        if isinstance(freq, str):  # note: freq is already of type str!
            if freq.startswith(("Period[", "period[")):
                m = cls._match.search(freq)
                if m is not None:
                    freq = m.group("freq")

            freq_offset = to_offset(freq, is_period=True)
            if freq_offset is not None:
                return freq_offset

        raise TypeError(
            "PeriodDtype argument should be string or BaseOffset, "
            f"got {type(freq).__name__}"
        )

    @classmethod
    def construct_from_string(cls, string: str_type) -> PeriodDtype:
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        if (
            isinstance(string, str)
            and (string.startswith(("period[", "Period[")))
            or isinstance(string, BaseOffset)
        ):
            # do not parse string like U as period[U]
            # avoid tuple to be regarded as freq
            try:
                return cls(freq=string)
            except ValueError:
                pass
        if isinstance(string, str):
            msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        else:
            msg = f"'construct_from_string' expects a string, got {type(string)}"
        raise TypeError(msg)

    def __str__(self) -> str_type:
        return self.name

    @property
    def name(self) -> str_type:
        return f"period[{self._freqstr}]"

    @property
    def na_value(self) -> NaTType:
        return NaT

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other in [self.name, capitalize_first_letter(self.name)]

        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            # PeriodDtype can be instantiated from freq string like "U",
            # but doesn't regard freq str like "U" as dtype.
            if dtype.startswith(("period[", "Period[")):
                try:
                    return cls._parse_dtype_strict(dtype) is not None
                except ValueError:
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    @classmethod
    def construct_array_type(cls) -> type_t[PeriodArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import PeriodArray

        return PeriodArray

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> PeriodArray:
        """
        Construct PeriodArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays import PeriodArray
        from pandas.core.arrays.arrow._arrow_utils import (
            pyarrow_array_to_numpy_and_mask,
        )

        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks

        results = []
        for arr in chunks:
            data, mask = pyarrow_array_to_numpy_and_mask(arr, dtype=np.dtype(np.int64))
            parr = PeriodArray(data.copy(), dtype=self, copy=False)
            # error: Invalid index type "ndarray[Any, dtype[bool_]]" for "PeriodArray";
            # expected type "Union[int, Sequence[int], Sequence[bool], slice]"
            parr[~mask] = NaT  # type: ignore[index]
            results.append(parr)

        if not results:
            return PeriodArray(np.array([], dtype="int64"), dtype=self, copy=False)
        return PeriodArray._concat_same_type(results)

    @cache_readonly
    def index_class(self) -> type_t[PeriodIndex]:
        from pandas import PeriodIndex

        return PeriodIndex


@register_extension_dtype
class IntervalDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for Interval data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    subtype : str, np.dtype
        The dtype of the Interval bounds.

    Attributes
    ----------
    subtype

    Methods
    -------
    None

    Examples
    --------
    >>> pd.IntervalDtype(subtype='int64', closed='both')
    interval[int64, both]
    """

    name = "interval"
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    num = 103
    _metadata = (
        "subtype",
        "closed",
    )

    _match = re.compile(
        r"(I|i)nterval\[(?P<subtype>[^,]+(\[.+\])?)"
        r"(, (?P<closed>(right|left|both|neither)))?\]"
    )

    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _subtype: None | np.dtype
    _closed: IntervalClosedType | None

    def __init__(self, subtype=None, closed: IntervalClosedType | None = None) -> None:
        from pandas.core.dtypes.common import (
            is_string_dtype,
            pandas_dtype,
        )

        if closed is not None and closed not in {"right", "left", "both", "neither"}:
            raise ValueError("closed must be one of 'right', 'left', 'both', 'neither'")

        if isinstance(subtype, IntervalDtype):
            if closed is not None and closed != subtype.closed:
                raise ValueError(
                    "dtype.closed and 'closed' do not match. "
                    "Try IntervalDtype(dtype.subtype, closed) instead."
                )
            self._subtype = subtype._subtype
            self._closed = subtype._closed
        elif subtype is None:
            # we are called as an empty constructor
            # generally for pickle compat
            self._subtype = None
            self._closed = closed
        elif isinstance(subtype, str) and subtype.lower() == "interval":
            self._subtype = None
            self._closed = closed
        else:
            if isinstance(subtype, str):
                m = IntervalDtype._match.search(subtype)
                if m is not None:
                    gd = m.groupdict()
                    subtype = gd["subtype"]
                    if gd.get("closed", None) is not None:
                        if closed is not None:
                            if closed != gd["closed"]:
                                raise ValueError(
                                    "'closed' keyword does not match value "
                                    "specified in dtype string"
                                )
                        closed = gd["closed"]  # type: ignore[assignment]

            try:
                subtype = pandas_dtype(subtype)
            except TypeError as err:
                raise TypeError("could not construct IntervalDtype") from err
            if CategoricalDtype.is_dtype(subtype) or is_string_dtype(subtype):
                # GH 19016
                msg = (
                    "category, object, and string subtypes are not supported "
                    "for IntervalDtype"
                )
                raise TypeError(msg)
            self._subtype = subtype
            self._closed = closed

    @cache_readonly
    def _can_hold_na(self) -> bool:
        subtype = self._subtype
        if subtype is None:
            # partially-initialized
            raise NotImplementedError(
                "_can_hold_na is not defined for partially-initialized IntervalDtype"
            )
        if subtype.kind in "iu":
            return False
        return True

    @property
    def closed(self) -> IntervalClosedType:
        return self._closed  # type: ignore[return-value]

    @property
    def subtype(self):
        """
        The dtype of the Interval bounds.

        Examples
        --------
        >>> dtype = pd.IntervalDtype(subtype='int64', closed='both')
        >>> dtype.subtype
        dtype('int64')
        """
        return self._subtype

    @classmethod
    def construct_array_type(cls) -> type[IntervalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import IntervalArray

        return IntervalArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> IntervalDtype:
        """
        attempt to construct this type from a string, raise a TypeError
        if its not possible
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string.lower() == "interval" or cls._match.search(string) is not None:
            return cls(string)

        msg = (
            f"Cannot construct a 'IntervalDtype' from '{string}'.\n\n"
            "Incorrectly formatted string passed to constructor. "
            "Valid formats include Interval or Interval[dtype] "
            "where dtype is numeric, datetime, or timedelta"
        )
        raise TypeError(msg)

    @property
    def type(self) -> type[Interval]:
        return Interval

    def __str__(self) -> str_type:
        if self.subtype is None:
            return "interval"
        if self.closed is None:
            # Only partially initialized GH#38394
            return f"interval[{self.subtype}]"
        return f"interval[{self.subtype}, {self.closed}]"

    def __hash__(self) -> int:
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other.lower() in (self.name.lower(), str(self).lower())
        elif not isinstance(other, IntervalDtype):
            return False
        elif self.subtype is None or other.subtype is None:
            # None should match any subtype
            return True
        elif self.closed != other.closed:
            return False
        else:
            return self.subtype == other.subtype

    def __setstate__(self, state) -> None:
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._subtype = state["subtype"]

        # backward-compat older pickles won't have "closed" key
        self._closed = state.pop("closed", None)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            if dtype.lower().startswith("interval"):
                try:
                    return cls.construct_from_string(dtype) is not None
                except (ValueError, TypeError):
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> IntervalArray:
        """
        Construct IntervalArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays import IntervalArray

        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks

        results = []
        for arr in chunks:
            if isinstance(arr, pyarrow.ExtensionArray):
                arr = arr.storage
            left = np.asarray(arr.field("left"), dtype=self.subtype)
            right = np.asarray(arr.field("right"), dtype=self.subtype)
            iarr = IntervalArray.from_arrays(left, right, closed=self.closed)
            results.append(iarr)

        if not results:
            return IntervalArray.from_arrays(
                np.array([], dtype=self.subtype),
                np.array([], dtype=self.subtype),
                closed=self.closed,
            )
        return IntervalArray._concat_same_type(results)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if not all(isinstance(x, IntervalDtype) for x in dtypes):
            return None

        closed = cast("IntervalDtype", dtypes[0]).closed
        if not all(cast("IntervalDtype", x).closed == closed for x in dtypes):
            return np.dtype(object)

        from pandas.core.dtypes.cast import find_common_type

        common = find_common_type([cast("IntervalDtype", x).subtype for x in dtypes])
        if common == object:
            return np.dtype(object)
        return IntervalDtype(common, closed=closed)

    @cache_readonly
    def index_class(self) -> type_t[IntervalIndex]:
        from pandas import IntervalIndex

        return IntervalIndex


class NumpyEADtype(ExtensionDtype):
    """
    A Pandas ExtensionDtype for NumPy dtypes.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    dtype : object
        Object to be converted to a NumPy data type object.

    See Also
    --------
    numpy.dtype
    """

    _metadata = ("_dtype",)
    _supports_2d = False
    _can_fast_transpose = False

    def __init__(self, dtype: npt.DTypeLike | NumpyEADtype | None) -> None:
        if isinstance(dtype, NumpyEADtype):
            # make constructor idempotent
            dtype = dtype.numpy_dtype
        self._dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f"NumpyEADtype({repr(self.name)})"

    @property
    def numpy_dtype(self) -> np.dtype:
        """
        The NumPy dtype this NumpyEADtype wraps.
        """
        return self._dtype

    @property
    def name(self) -> str:
        """
        A bit-width name for this data-type.
        """
        return self._dtype.name

    @property
    def type(self) -> type[np.generic]:
        """
        The type object used to instantiate a scalar of this NumPy data-type.
        """
        return self._dtype.type

    @property
    def _is_numeric(self) -> bool:
        # exclude object, str, unicode, void.
        return self.kind in set("biufc")

    @property
    def _is_boolean(self) -> bool:
        return self.kind == "b"

    @classmethod
    def construct_from_string(cls, string: str) -> NumpyEADtype:
        try:
            dtype = np.dtype(string)
        except TypeError as err:
            if not isinstance(string, str):
                msg = f"'construct_from_string' expects a string, got {type(string)}"
            else:
                msg = f"Cannot construct a 'NumpyEADtype' from '{string}'"
            raise TypeError(msg) from err
        return cls(dtype)

    @classmethod
    def construct_array_type(cls) -> type_t[NumpyExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import NumpyExtensionArray

        return NumpyExtensionArray

    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV') identifying the general kind of data.
        """
        return self._dtype.kind

    @property
    def itemsize(self) -> int:
        """
        The element size of this data-type object.
        """
        return self._dtype.itemsize


class BaseMaskedDtype(ExtensionDtype):
    """
    Base class for dtypes for BaseMaskedArray subclasses.
    """

    base = None
    type: type

    @property
    def na_value(self) -> libmissing.NAType:
        return libmissing.NA

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of our numpy dtype"""
        return np.dtype(self.type)

    @cache_readonly
    def kind(self) -> str:
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls) -> type_t[BaseMaskedArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        raise NotImplementedError

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> BaseMaskedDtype:
        """
        Construct the MaskedDtype corresponding to the given numpy dtype.
        """
        if dtype.kind == "b":
            from pandas.core.arrays.boolean import BooleanDtype

            return BooleanDtype()
        elif dtype.kind in "iu":
            from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE

            return NUMPY_INT_TO_DTYPE[dtype]
        elif dtype.kind == "f":
            from pandas.core.arrays.floating import NUMPY_FLOAT_TO_DTYPE

            return NUMPY_FLOAT_TO_DTYPE[dtype]
        else:
            raise NotImplementedError(dtype)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # We unwrap any masked dtypes, find the common dtype we would use
        #  for that, then re-mask the result.
        from pandas.core.dtypes.cast import find_common_type

        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, BaseMaskedDtype) else dtype
                for dtype in dtypes
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            # If we ever support e.g. Masked[DatetimeArray] then this will change
            return None
        try:
            return type(self).from_numpy_dtype(new_dtype)
        except (KeyError, NotImplementedError):
            return None


@register_extension_dtype
class SparseDtype(ExtensionDtype):
    """
    Dtype for data stored in :class:`SparseArray`.

    This dtype implements the pandas ExtensionDtype interface.

    Parameters
    ----------
    dtype : str, ExtensionDtype, numpy.dtype, type, default numpy.float64
        The dtype of the underlying array storing the non-fill value values.
    fill_value : scalar, optional
        The scalar value not stored in the SparseArray. By default, this
        depends on `dtype`.

        =========== ==========
        dtype       na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        ``False``
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The default value may be overridden by specifying a `fill_value`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> ser = pd.Series([1, 0, 0], dtype=pd.SparseDtype(dtype=int, fill_value=0))
    >>> ser
    0    1
    1    0
    2    0
    dtype: Sparse[int64, 0]
    >>> ser.sparse.density
    0.3333333333333333
    """

    _is_immutable = True

    # We include `_is_na_fill_value` in the metadata to avoid hash collisions
    # between SparseDtype(float, 0.0) and SparseDtype(float, nan).
    # Without is_na_fill_value in the comparison, those would be equal since
    # hash(nan) is (sometimes?) 0.
    _metadata = ("_dtype", "_fill_value", "_is_na_fill_value")

    def __init__(self, dtype: Dtype = np.float64, fill_value: Any = None) -> None:
        if isinstance(dtype, type(self)):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype

        from pandas.core.dtypes.common import (
            is_string_dtype,
            pandas_dtype,
        )
        from pandas.core.dtypes.missing import na_value_for_dtype

        dtype = pandas_dtype(dtype)
        if is_string_dtype(dtype):
            dtype = np.dtype("object")
        if not isinstance(dtype, np.dtype):
            # GH#53160
            raise TypeError("SparseDtype subtype must be a numpy dtype")

        if fill_value is None:
            fill_value = na_value_for_dtype(dtype)

        self._dtype = dtype
        self._fill_value = fill_value
        self._check_fill_value()

    def __hash__(self) -> int:
        # Python3 doesn't inherit __hash__ when a base class overrides
        # __eq__, so we explicitly do it here.
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        # We have to override __eq__ to handle NA values in _metadata.
        # The base class does simple == checks, which fail for NA.
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False

        if isinstance(other, type(self)):
            subtype = self.subtype == other.subtype
            if self._is_na_fill_value:
                # this case is complicated by two things:
                # SparseDtype(float, float(nan)) == SparseDtype(float, np.nan)
                # SparseDtype(float, np.nan)     != SparseDtype(float, pd.NaT)
                # i.e. we want to treat any floating-point NaN as equal, but
                # not a floating-point NaN and a datetime NaT.
                fill_value = (
                    other._is_na_fill_value
                    and isinstance(self.fill_value, type(other.fill_value))
                    or isinstance(other.fill_value, type(self.fill_value))
                )
            else:
                with warnings.catch_warnings():
                    # Ignore spurious numpy warning
                    warnings.filterwarnings(
                        "ignore",
                        "elementwise comparison failed",
                        category=DeprecationWarning,
                    )

                    fill_value = self.fill_value == other.fill_value

            return subtype and fill_value
        return False

    @property
    def fill_value(self):
        """
        The fill value of the array.

        Converting the SparseArray to a dense ndarray will fill the
        array with this value.

        .. warning::

           It's possible to end up with a SparseArray that has ``fill_value``
           values in ``sp_values``. This can occur, for example, when setting
           ``SparseArray.fill_value`` directly.
        """
        return self._fill_value

    def _check_fill_value(self) -> None:
        if not lib.is_scalar(self._fill_value):
            raise ValueError(
                f"fill_value must be a scalar. Got {self._fill_value} instead"
            )

        from pandas.core.dtypes.cast import can_hold_element
        from pandas.core.dtypes.missing import (
            is_valid_na_for_dtype,
            isna,
        )

        from pandas.core.construction import ensure_wrapped_if_datetimelike

        # GH#23124 require fill_value and subtype to match
        val = self._fill_value
        if isna(val):
            if not is_valid_na_for_dtype(val, self.subtype):
                warnings.warn(
                    "Allowing arbitrary scalar fill_value in SparseDtype is "
                    "deprecated. In a future version, the fill_value must be "
                    "a valid value for the SparseDtype.subtype.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
        else:
            dummy = np.empty(0, dtype=self.subtype)
            dummy = ensure_wrapped_if_datetimelike(dummy)

            if not can_hold_element(dummy, val):
                warnings.warn(
                    "Allowing arbitrary scalar fill_value in SparseDtype is "
                    "deprecated. In a future version, the fill_value must be "
                    "a valid value for the SparseDtype.subtype.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

    @property
    def _is_na_fill_value(self) -> bool:
        from pandas import isna

        return isna(self.fill_value)

    @property
    def _is_numeric(self) -> bool:
        return self.subtype != object

    @property
    def _is_boolean(self) -> bool:
        return self.subtype.kind == "b"

    @property
    def kind(self) -> str:
        """
        The sparse kind. Either 'integer', or 'block'.
        """
        return self.subtype.kind

    @property
    def type(self):
        return self.subtype.type

    @property
    def subtype(self):
        return self._dtype

    @property
    def name(self) -> str:
        return f"Sparse[{self.subtype.name}, {repr(self.fill_value)}]"

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def construct_array_type(cls) -> type_t[SparseArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.sparse.array import SparseArray

        return SparseArray

    @classmethod
    def construct_from_string(cls, string: str) -> SparseDtype:
        """
        Construct a SparseDtype from a string form.

        Parameters
        ----------
        string : str
            Can take the following forms.

            string           dtype
            ================ ============================
            'int'            SparseDtype[np.int64, 0]
            'Sparse'         SparseDtype[np.float64, nan]
            'Sparse[int]'    SparseDtype[np.int64, 0]
            'Sparse[int, 0]' SparseDtype[np.int64, 0]
            ================ ============================

            It is not possible to specify non-default fill values
            with a string. An argument like ``'Sparse[int, 1]'``
            will raise a ``TypeError`` because the default fill value
            for integers is 0.

        Returns
        -------
        SparseDtype
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        msg = f"Cannot construct a 'SparseDtype' from '{string}'"
        if string.startswith("Sparse"):
            try:
                sub_type, has_fill_value = cls._parse_subtype(string)
            except ValueError as err:
                raise TypeError(msg) from err
            else:
                result = SparseDtype(sub_type)
                msg = (
                    f"Cannot construct a 'SparseDtype' from '{string}'.\n\nIt "
                    "looks like the fill_value in the string is not "
                    "the default for the dtype. Non-default fill_values "
                    "are not supported. Use the 'SparseDtype()' "
                    "constructor instead."
                )
                if has_fill_value and str(result) != string:
                    raise TypeError(msg)
                return result
        else:
            raise TypeError(msg)

    @staticmethod
    def _parse_subtype(dtype: str) -> tuple[str, bool]:
        """
        Parse a string to get the subtype

        Parameters
        ----------
        dtype : str
            A string like

            * Sparse[subtype]
            * Sparse[subtype, fill_value]

        Returns
        -------
        subtype : str

        Raises
        ------
        ValueError
            When the subtype cannot be extracted.
        """
        xpr = re.compile(r"Sparse\[(?P<subtype>[^,]*)(, )?(?P<fill_value>.*?)?\]$")
        m = xpr.match(dtype)
        has_fill_value = False
        if m:
            subtype = m.groupdict()["subtype"]
            has_fill_value = bool(m.groupdict()["fill_value"])
        elif dtype == "Sparse":
            subtype = "float64"
        else:
            raise ValueError(f"Cannot parse {dtype}")
        return subtype, has_fill_value

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        dtype = getattr(dtype, "dtype", dtype)
        if isinstance(dtype, str) and dtype.startswith("Sparse"):
            sub_type, _ = cls._parse_subtype(dtype)
            dtype = np.dtype(sub_type)
        elif isinstance(dtype, cls):
            return True
        return isinstance(dtype, np.dtype) or dtype == "Sparse"

    def update_dtype(self, dtype) -> SparseDtype:
        """
        Convert the SparseDtype to a new dtype.

        This takes care of converting the ``fill_value``.

        Parameters
        ----------
        dtype : Union[str, numpy.dtype, SparseDtype]
            The new dtype to use.

            * For a SparseDtype, it is simply returned
            * For a NumPy dtype (or str), the current fill value
              is converted to the new dtype, and a SparseDtype
              with `dtype` and the new fill value is returned.

        Returns
        -------
        SparseDtype
            A new SparseDtype with the correct `dtype` and fill value
            for that `dtype`.

        Raises
        ------
        ValueError
            When the current fill value cannot be converted to the
            new `dtype` (e.g. trying to convert ``np.nan`` to an
            integer dtype).


        Examples
        --------
        >>> SparseDtype(int, 0).update_dtype(float)
        Sparse[float64, 0.0]

        >>> SparseDtype(int, 1).update_dtype(SparseDtype(float, np.nan))
        Sparse[float64, nan]
        """
        from pandas.core.dtypes.astype import astype_array
        from pandas.core.dtypes.common import pandas_dtype

        cls = type(self)
        dtype = pandas_dtype(dtype)

        if not isinstance(dtype, cls):
            if not isinstance(dtype, np.dtype):
                raise TypeError("sparse arrays of extension dtypes not supported")

            fv_asarray = np.atleast_1d(np.array(self.fill_value))
            fvarr = astype_array(fv_asarray, dtype)
            # NB: not fv_0d.item(), as that casts dt64->int
            fill_value = fvarr[0]
            dtype = cls(dtype, fill_value=fill_value)

        return dtype

    @property
    def _subtype_with_str(self):
        """
        Whether the SparseDtype's subtype should be considered ``str``.

        Typically, pandas will store string data in an object-dtype array.
        When converting values to a dtype, e.g. in ``.astype``, we need to
        be more specific, we need the actual underlying type.

        Returns
        -------
        >>> SparseDtype(int, 1)._subtype_with_str
        dtype('int64')

        >>> SparseDtype(object, 1)._subtype_with_str
        dtype('O')

        >>> dtype = SparseDtype(str, '')
        >>> dtype.subtype
        dtype('O')

        >>> dtype._subtype_with_str
        <class 'str'>
        """
        if isinstance(self.fill_value, str):
            return type(self.fill_value)
        return self.subtype

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # TODO for now only handle SparseDtypes and numpy dtypes => extend
        # with other compatible extension dtypes
        from pandas.core.dtypes.cast import np_find_common_type

        if any(
            isinstance(x, ExtensionDtype) and not isinstance(x, SparseDtype)
            for x in dtypes
        ):
            return None

        fill_values = [x.fill_value for x in dtypes if isinstance(x, SparseDtype)]
        fill_value = fill_values[0]

        from pandas import isna

        # np.nan isn't a singleton, so we may end up with multiple
        # NaNs here, so we ignore the all NA case too.
        if not (len(set(fill_values)) == 1 or isna(fill_values).all()):
            warnings.warn(
                "Concatenating sparse arrays with multiple fill "
                f"values: '{fill_values}'. Picking the first and "
                "converting the rest.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

        np_dtypes = (x.subtype if isinstance(x, SparseDtype) else x for x in dtypes)
        return SparseDtype(np_find_common_type(*np_dtypes), fill_value=fill_value)


@register_extension_dtype
class ArrowDtype(StorageExtensionDtype):
    """
    An ExtensionDtype for PyArrow data types.

    .. warning::

       ArrowDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    While most ``dtype`` arguments can accept the "string"
    constructor, e.g. ``"int64[pyarrow]"``, ArrowDtype is useful
    if the data type contains parameters like ``pyarrow.timestamp``.

    Parameters
    ----------
    pyarrow_dtype : pa.DataType
        An instance of a `pyarrow.DataType <https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions>`__.

    Attributes
    ----------
    pyarrow_dtype

    Methods
    -------
    None

    Returns
    -------
    ArrowDtype

    Examples
    --------
    >>> import pyarrow as pa
    >>> pd.ArrowDtype(pa.int64())
    int64[pyarrow]

    Types with parameters must be constructed with ArrowDtype.

    >>> pd.ArrowDtype(pa.timestamp("s", tz="America/New_York"))
    timestamp[s, tz=America/New_York][pyarrow]
    >>> pd.ArrowDtype(pa.list_(pa.int64()))
    list<item: int64>[pyarrow]
    """

    _metadata = ("storage", "pyarrow_dtype")  # type: ignore[assignment]

    def __init__(self, pyarrow_dtype: pa.DataType) -> None:
        super().__init__("pyarrow")
        if pa_version_under10p1:
            raise ImportError("pyarrow>=10.0.1 is required for ArrowDtype")
        if not isinstance(pyarrow_dtype, pa.DataType):
            raise ValueError(
                f"pyarrow_dtype ({pyarrow_dtype}) must be an instance "
                f"of a pyarrow.DataType. Got {type(pyarrow_dtype)} instead."
            )
        self.pyarrow_dtype = pyarrow_dtype

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return super().__eq__(other)
        return self.pyarrow_dtype == other.pyarrow_dtype

    @property
    def type(self):
        """
        Returns associated scalar type.
        """
        pa_type = self.pyarrow_dtype
        if pa.types.is_integer(pa_type):
            return int
        elif pa.types.is_floating(pa_type):
            return float
        elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            return str
        elif (
            pa.types.is_binary(pa_type)
            or pa.types.is_fixed_size_binary(pa_type)
            or pa.types.is_large_binary(pa_type)
        ):
            return bytes
        elif pa.types.is_boolean(pa_type):
            return bool
        elif pa.types.is_duration(pa_type):
            if pa_type.unit == "ns":
                return Timedelta
            else:
                return timedelta
        elif pa.types.is_timestamp(pa_type):
            if pa_type.unit == "ns":
                return Timestamp
            else:
                return datetime
        elif pa.types.is_date(pa_type):
            return date
        elif pa.types.is_time(pa_type):
            return time
        elif pa.types.is_decimal(pa_type):
            return Decimal
        elif pa.types.is_dictionary(pa_type):
            # TODO: Potentially change this & CategoricalDtype.type to
            #  something more representative of the scalar
            return CategoricalDtypeType
        elif pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            return list
        elif pa.types.is_fixed_size_list(pa_type):
            return list
        elif pa.types.is_map(pa_type):
            return list
        elif pa.types.is_struct(pa_type):
            return dict
        elif pa.types.is_null(pa_type):
            # TODO: None? pd.NA? pa.null?
            return type(pa_type)
        elif isinstance(pa_type, pa.ExtensionType):
            return type(self)(pa_type.storage_type).type
        raise NotImplementedError(pa_type)

    @property
    def name(self) -> str:  # type: ignore[override]
        """
        A string identifying the data type.
        """
        return f"{str(self.pyarrow_dtype)}[{self.storage}]"

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of the related numpy dtype"""
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # pa.timestamp(unit).to_pandas_dtype() returns ns units
            # regardless of the pyarrow timestamp units.
            # This can be removed if/when pyarrow addresses it:
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"datetime64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_duration(self.pyarrow_dtype):
            # pa.duration(unit).to_pandas_dtype() returns ns units
            # regardless of the pyarrow duration units
            # This can be removed if/when pyarrow addresses it:
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"timedelta64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_string(self.pyarrow_dtype) or pa.types.is_large_string(
            self.pyarrow_dtype
        ):
            # pa.string().to_pandas_dtype() = object which we don't want
            return np.dtype(str)
        try:
            return np.dtype(self.pyarrow_dtype.to_pandas_dtype())
        except (NotImplementedError, TypeError):
            return np.dtype(object)

    @cache_readonly
    def kind(self) -> str:
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # To mirror DatetimeTZDtype
            return "M"
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls) -> type_t[ArrowExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.arrow import ArrowExtensionArray

        return ArrowExtensionArray

    @classmethod
    def construct_from_string(cls, string: str) -> ArrowDtype:
        """
        Construct this type from a string.

        Parameters
        ----------
        string : str
            string should follow the format f"{pyarrow_type}[pyarrow]"
            e.g. int64[pyarrow]
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if not string.endswith("[pyarrow]"):
            raise TypeError(f"'{string}' must end with '[pyarrow]'")
        if string in ("string[pyarrow]", "str[pyarrow]"):
            # Ensure Registry.find skips ArrowDtype to use StringDtype instead
            raise TypeError("string[pyarrow] should be constructed by StringDtype")

        base_type = string[:-9]  # get rid of "[pyarrow]"
        try:
            pa_dtype = pa.type_for_alias(base_type)
        except ValueError as err:
            has_parameters = re.search(r"[\[\(].*[\]\)]", base_type)
            if has_parameters:
                # Fallback to try common temporal types
                try:
                    return cls._parse_temporal_dtype_string(base_type)
                except (NotImplementedError, ValueError):
                    # Fall through to raise with nice exception message below
                    pass

                raise NotImplementedError(
                    "Passing pyarrow type specific parameters "
                    f"({has_parameters.group()}) in the string is not supported. "
                    "Please construct an ArrowDtype object with a pyarrow_dtype "
                    "instance with specific parameters."
                ) from err
            raise TypeError(f"'{base_type}' is not a valid pyarrow data type.") from err
        return cls(pa_dtype)

    # TODO(arrow#33642): This can be removed once supported by pyarrow
    @classmethod
    def _parse_temporal_dtype_string(cls, string: str) -> ArrowDtype:
        """
        Construct a temporal ArrowDtype from string.
        """
        # we assume
        #  1) "[pyarrow]" has already been stripped from the end of our string.
        #  2) we know "[" is present
        head, tail = string.split("[", 1)

        if not tail.endswith("]"):
            raise ValueError
        tail = tail[:-1]

        if head == "timestamp":
            assert "," in tail  # otherwise type_for_alias should work
            unit, tz = tail.split(",", 1)
            unit = unit.strip()
            tz = tz.strip()
            if tz.startswith("tz="):
                tz = tz[3:]

            pa_type = pa.timestamp(unit, tz=tz)
            dtype = cls(pa_type)
            return dtype

        raise NotImplementedError(string)

    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.
        """
        # TODO: pa.types.is_boolean?
        return (
            pa.types.is_integer(self.pyarrow_dtype)
            or pa.types.is_floating(self.pyarrow_dtype)
            or pa.types.is_decimal(self.pyarrow_dtype)
        )

    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.
        """
        return pa.types.is_boolean(self.pyarrow_dtype)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # We unwrap any masked dtypes, find the common dtype we would use
        #  for that, then re-mask the result.
        # Mirrors BaseMaskedDtype
        from pandas.core.dtypes.cast import find_common_type

        null_dtype = type(self)(pa.null())

        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, ArrowDtype) else dtype
                for dtype in dtypes
                if dtype != null_dtype
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            return None
        try:
            pa_dtype = pa.from_numpy_dtype(new_dtype)
            return type(self)(pa_dtype)
        except NotImplementedError:
            return None

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray):
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        array_class = self.construct_array_type()
        arr = array.cast(self.pyarrow_dtype, safe=True)
        return array_class(arr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\ma\extras.py ===
"""
Masked arrays add-ons.

A collection of utilities for `numpy.ma`.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu

"""
__all__ = [
    'apply_along_axis', 'apply_over_axes', 'atleast_1d', 'atleast_2d',
    'atleast_3d', 'average', 'clump_masked', 'clump_unmasked', 'column_stack',
    'compress_cols', 'compress_nd', 'compress_rowcols', 'compress_rows',
    'count_masked', 'corrcoef', 'cov', 'diagflat', 'dot', 'dstack', 'ediff1d',
    'flatnotmasked_contiguous', 'flatnotmasked_edges', 'hsplit', 'hstack',
    'isin', 'in1d', 'intersect1d', 'mask_cols', 'mask_rowcols', 'mask_rows',
    'masked_all', 'masked_all_like', 'median', 'mr_', 'ndenumerate',
    'notmasked_contiguous', 'notmasked_edges', 'polyfit', 'row_stack',
    'setdiff1d', 'setxor1d', 'stack', 'unique', 'union1d', 'vander', 'vstack',
    ]

import itertools
import warnings

import numpy as np
from numpy import array as nxarray
from numpy import ndarray
from numpy.lib._function_base_impl import _ureduce
from numpy.lib._index_tricks_impl import AxisConcatenator
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple

from . import core as ma
from .core import (  # noqa: F401
    MAError,
    MaskedArray,
    add,
    array,
    asarray,
    concatenate,
    count,
    dot,
    filled,
    get_masked_subclass,
    getdata,
    getmask,
    getmaskarray,
    make_mask_descr,
    mask_or,
    masked,
    masked_array,
    nomask,
    ones,
    sort,
    zeros,
)


def issequence(seq):
    """
    Is seq a sequence (ndarray, list or tuple)?

    """
    return isinstance(seq, (ndarray, tuple, list))


def count_masked(arr, axis=None):
    """
    Count the number of masked elements along the given axis.

    Parameters
    ----------
    arr : array_like
        An array with (possibly) masked elements.
    axis : int, optional
        Axis along which to count. If None (default), a flattened
        version of the array is used.

    Returns
    -------
    count : int, ndarray
        The total number of masked elements (axis=None) or the number
        of masked elements along each slice of the given axis.

    See Also
    --------
    MaskedArray.count : Count non-masked elements.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(9).reshape((3,3))
    >>> a = np.ma.array(a)
    >>> a[1, 0] = np.ma.masked
    >>> a[1, 2] = np.ma.masked
    >>> a[2, 1] = np.ma.masked
    >>> a
    masked_array(
      data=[[0, 1, 2],
            [--, 4, --],
            [6, --, 8]],
      mask=[[False, False, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=999999)
    >>> np.ma.count_masked(a)
    3

    When the `axis` keyword is used an array is returned.

    >>> np.ma.count_masked(a, axis=0)
    array([1, 1, 1])
    >>> np.ma.count_masked(a, axis=1)
    array([0, 2, 1])

    """
    m = getmaskarray(arr)
    return m.sum(axis)


def masked_all(shape, dtype=float):
    """
    Empty masked array with all elements masked.

    Return an empty masked array of the given shape and dtype, where all the
    data are masked.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the required MaskedArray, e.g., ``(2, 3)`` or ``2``.
    dtype : dtype, optional
        Data type of the output.

    Returns
    -------
    a : MaskedArray
        A masked array with all data masked.

    See Also
    --------
    masked_all_like : Empty masked array modelled on an existing array.

    Notes
    -----
    Unlike other masked array creation functions (e.g. `numpy.ma.zeros`,
    `numpy.ma.ones`, `numpy.ma.full`), `masked_all` does not initialize the
    values of the array, and may therefore be marginally faster. However,
    the values stored in the newly allocated array are arbitrary. For
    reproducible behavior, be sure to set each element of the array before
    reading.

    Examples
    --------
    >>> import numpy as np
    >>> np.ma.masked_all((3, 3))
    masked_array(
      data=[[--, --, --],
            [--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True],
            [ True,  True,  True]],
      fill_value=1e+20,
      dtype=float64)

    The `dtype` parameter defines the underlying data type.

    >>> a = np.ma.masked_all((3, 3))
    >>> a.dtype
    dtype('float64')
    >>> a = np.ma.masked_all((3, 3), dtype=np.int32)
    >>> a.dtype
    dtype('int32')

    """
    a = masked_array(np.empty(shape, dtype),
                     mask=np.ones(shape, make_mask_descr(dtype)))
    return a


def masked_all_like(arr):
    """
    Empty masked array with the properties of an existing array.

    Return an empty masked array of the same shape and dtype as
    the array `arr`, where all the data are masked.

    Parameters
    ----------
    arr : ndarray
        An array describing the shape and dtype of the required MaskedArray.

    Returns
    -------
    a : MaskedArray
        A masked array with all data masked.

    Raises
    ------
    AttributeError
        If `arr` doesn't have a shape attribute (i.e. not an ndarray)

    See Also
    --------
    masked_all : Empty masked array with all elements masked.

    Notes
    -----
    Unlike other masked array creation functions (e.g. `numpy.ma.zeros_like`,
    `numpy.ma.ones_like`, `numpy.ma.full_like`), `masked_all_like` does not
    initialize the values of the array, and may therefore be marginally
    faster. However, the values stored in the newly allocated array are
    arbitrary. For reproducible behavior, be sure to set each element of the
    array before reading.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.zeros((2, 3), dtype=np.float32)
    >>> arr
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    >>> np.ma.masked_all_like(arr)
    masked_array(
      data=[[--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True]],
      fill_value=np.float64(1e+20),
      dtype=float32)

    The dtype of the masked array matches the dtype of `arr`.

    >>> arr.dtype
    dtype('float32')
    >>> np.ma.masked_all_like(arr).dtype
    dtype('float32')

    """
    a = np.empty_like(arr).view(MaskedArray)
    a._mask = np.ones(a.shape, dtype=make_mask_descr(a.dtype))
    return a


#####--------------------------------------------------------------------------
#---- --- Standard functions ---
#####--------------------------------------------------------------------------
class _fromnxfunction:
    """
    Defines a wrapper to adapt NumPy functions to masked arrays.


    An instance of `_fromnxfunction` can be called with the same parameters
    as the wrapped NumPy function. The docstring of `newfunc` is adapted from
    the wrapped function as well, see `getdoc`.

    This class should not be used directly. Instead, one of its extensions that
    provides support for a specific type of input should be used.

    Parameters
    ----------
    funcname : str
        The name of the function to be adapted. The function should be
        in the NumPy namespace (i.e. ``np.funcname``).

    """

    def __init__(self, funcname):
        self.__name__ = funcname
        self.__qualname__ = funcname
        self.__doc__ = self.getdoc()

    def getdoc(self):
        """
        Retrieve the docstring and signature from the function.

        The ``__doc__`` attribute of the function is used as the docstring for
        the new masked array version of the function. A note on application
        of the function to the mask is appended.

        Parameters
        ----------
        None

        """
        npfunc = getattr(np, self.__name__, None)
        doc = getattr(npfunc, '__doc__', None)
        if doc:
            sig = ma.get_object_signature(npfunc)
            doc = ma.doc_note(doc, "The function is applied to both the _data "
                                   "and the _mask, if any.")
            if sig:
                sig = self.__name__ + sig + "\n\n"
            return sig + doc
        return

    def __call__(self, *args, **params):
        pass


class _fromnxfunction_single(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with a single array
    argument followed by auxiliary args that are passed verbatim for
    both the data and mask calls.
    """
    def __call__(self, x, *args, **params):
        func = getattr(np, self.__name__)
        if isinstance(x, ndarray):
            _d = func(x.__array__(), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            return masked_array(_d, mask=_m)
        else:
            _d = func(np.asarray(x), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            return masked_array(_d, mask=_m)


class _fromnxfunction_seq(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with a single sequence
    of arrays followed by auxiliary args that are passed verbatim for
    both the data and mask calls.
    """
    def __call__(self, x, *args, **params):
        func = getattr(np, self.__name__)
        _d = func(tuple(np.asarray(a) for a in x), *args, **params)
        _m = func(tuple(getmaskarray(a) for a in x), *args, **params)
        return masked_array(_d, mask=_m)


class _fromnxfunction_args(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with multiple array
    arguments. The first non-array-like input marks the beginning of the
    arguments that are passed verbatim for both the data and mask calls.
    Array arguments are processed independently and the results are
    returned in a list. If only one array is found, the return value is
    just the processed array instead of a list.
    """
    def __call__(self, *args, **params):
        func = getattr(np, self.__name__)
        arrays = []
        args = list(args)
        while len(args) > 0 and issequence(args[0]):
            arrays.append(args.pop(0))
        res = []
        for x in arrays:
            _d = func(np.asarray(x), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            res.append(masked_array(_d, mask=_m))
        if len(arrays) == 1:
            return res[0]
        return res


class _fromnxfunction_allargs(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with multiple array
    arguments. Similar to `_fromnxfunction_args` except that all args
    are converted to arrays even if they are not so already. This makes
    it possible to process scalars as 1-D arrays. Only keyword arguments
    are passed through verbatim for the data and mask calls. Arrays
    arguments are processed independently and the results are returned
    in a list. If only one arg is present, the return value is just the
    processed array instead of a list.
    """
    def __call__(self, *args, **params):
        func = getattr(np, self.__name__)
        res = []
        for x in args:
            _d = func(np.asarray(x), **params)
            _m = func(getmaskarray(x), **params)
            res.append(masked_array(_d, mask=_m))
        if len(args) == 1:
            return res[0]
        return res


atleast_1d = _fromnxfunction_allargs('atleast_1d')
atleast_2d = _fromnxfunction_allargs('atleast_2d')
atleast_3d = _fromnxfunction_allargs('atleast_3d')

vstack = row_stack = _fromnxfunction_seq('vstack')
hstack = _fromnxfunction_seq('hstack')
column_stack = _fromnxfunction_seq('column_stack')
dstack = _fromnxfunction_seq('dstack')
stack = _fromnxfunction_seq('stack')

hsplit = _fromnxfunction_single('hsplit')

diagflat = _fromnxfunction_single('diagflat')


#####--------------------------------------------------------------------------
#----
#####--------------------------------------------------------------------------
def flatten_inplace(seq):
    """Flatten a sequence in place."""
    k = 0
    while (k != len(seq)):
        while hasattr(seq[k], '__iter__'):
            seq[k:(k + 1)] = seq[k]
        k += 1
    return seq


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    (This docstring should be overwritten)
    """
    arr = array(arr, copy=False, subok=True)
    nd = arr.ndim
    axis = normalize_axis_index(axis, nd)
    ind = [0] * (nd - 1)
    i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    indlist.remove(axis)
    i[axis] = slice(None, None)
    outshape = np.asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
    #  if res is a number, then we have a smaller output array
    asscalar = np.isscalar(res)
    if not asscalar:
        try:
            len(res)
        except TypeError:
            asscalar = True
    # Note: we shouldn't set the dtype of the output from the first result
    # so we force the type to object, and build a list of dtypes.  We'll
    # just take the largest, to avoid some downcasting
    dtypes = []
    if asscalar:
        dtypes.append(np.asarray(res).dtype)
        outarr = zeros(outshape, object)
        outarr[tuple(ind)] = res
        Ntot = np.prod(outshape)
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(ind)] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    else:
        res = array(res, copy=False, subok=True)
        j = i.copy()
        j[axis] = ([slice(None, None)] * res.ndim)
        j.put(indlist, ind)
        Ntot = np.prod(outshape)
        holdshape = outshape
        outshape = list(arr.shape)
        outshape[axis] = res.shape
        dtypes.append(asarray(res).dtype)
        outshape = flatten_inplace(outshape)
        outarr = zeros(outshape, object)
        outarr[tuple(flatten_inplace(j.tolist()))] = res
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            j.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(flatten_inplace(j.tolist()))] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    max_dtypes = np.dtype(np.asarray(dtypes).max())
    if not hasattr(arr, '_mask'):
        result = np.asarray(outarr, dtype=max_dtypes)
    else:
        result = asarray(outarr, dtype=max_dtypes)
        result.fill_value = ma.default_fill_value(result)
    return result


apply_along_axis.__doc__ = np.apply_along_axis.__doc__


def apply_over_axes(func, a, axes):
    """
    (This docstring will be overwritten)
    """
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = ma.expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning "
                        "an array of the correct shape")
    return val


if apply_over_axes.__doc__ is not None:
    apply_over_axes.__doc__ = np.apply_over_axes.__doc__[
        :np.apply_over_axes.__doc__.find('Notes')].rstrip() + \
    """

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.arange(24).reshape(2,3,4)
    >>> a[:,0,1] = np.ma.masked
    >>> a[:,1,:] = np.ma.masked
    >>> a
    masked_array(
      data=[[[0, --, 2, 3],
             [--, --, --, --],
             [8, 9, 10, 11]],
            [[12, --, 14, 15],
             [--, --, --, --],
             [20, 21, 22, 23]]],
      mask=[[[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]],
            [[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]]],
      fill_value=999999)
    >>> np.ma.apply_over_axes(np.ma.sum, a, [0,2])
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.ma.sum(a, axis=(0,2)).reshape((1,-1,1))
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)
    """


def average(a, axis=None, weights=None, returned=False, *,
            keepdims=np._NoValue):
    """
    Return the weighted average of array over the given axis.

    Parameters
    ----------
    a : array_like
        Data to be averaged.
        Masked entries are not taken into account in the computation.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        `axis=None`, will average over all of the elements of the input array.
        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The array of weights must be the same shape as `a` if no axis is
        specified, otherwise the weights must have dimensions and shape
        consistent with `a` along the specified axis.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        The calculation is::

            avg = sum(a * weights) / sum(weights)

        where the sum is over all included elements.
        The only constraint on the values of `weights` is that `sum(weights)`
        must not be 0.
    returned : bool, optional
        Flag indicating whether a tuple ``(result, sum of weights)``
        should be returned as output (True), or just the result (False).
        Default is False.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        *Note:* `keepdims` will not work with instances of `numpy.matrix`
        or other classes whose methods do not support `keepdims`.

        .. versionadded:: 1.23.0

    Returns
    -------
    average, [sum_of_weights] : (tuple of) scalar or MaskedArray
        The average along the specified axis. When returned is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. The return type is `np.float64`
        if `a` is of integer type and floats smaller than `float64`, or the
        input data-type, otherwise. If returned, `sum_of_weights` is always
        `float64`.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When `weights` does not have the same shape as `a`, and `axis=None`.
    ValueError
        When `weights` does not have dimensions and shape consistent with `a`
        along specified `axis`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array([1., 2., 3., 4.], mask=[False, False, True, True])
    >>> np.ma.average(a, weights=[3, 1, 0, 0])
    1.25

    >>> x = np.ma.arange(6.).reshape(3, 2)
    >>> x
    masked_array(
      data=[[0., 1.],
            [2., 3.],
            [4., 5.]],
      mask=False,
      fill_value=1e+20)
    >>> data = np.arange(8).reshape((2, 2, 2))
    >>> data
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.ma.average(data, axis=(0, 1), weights=[[1./4, 3./4], [1., 1./2]])
    masked_array(data=[3.4, 4.4],
             mask=[False, False],
       fill_value=1e+20)
    >>> np.ma.average(data, axis=0, weights=[[1./4, 3./4], [1., 1./2]])
    Traceback (most recent call last):
        ...
    ValueError: Shape of weights must be consistent
    with shape of a along specified axis.

    >>> avg, sumweights = np.ma.average(x, axis=0, weights=[1, 2, 3],
    ...                                 returned=True)
    >>> avg
    masked_array(data=[2.6666666666666665, 3.6666666666666665],
                 mask=[False, False],
           fill_value=1e+20)

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.ma.average(x, axis=1, keepdims=True)
    masked_array(
      data=[[0.5],
            [2.5],
            [4.5]],
      mask=False,
      fill_value=1e+20)
    """
    a = asarray(a)
    m = getmask(a)

    if axis is not None:
        axis = normalize_axis_tuple(axis, a.ndim, argname="axis")

    if keepdims is np._NoValue:
        # Don't pass on the keepdims argument if one wasn't given.
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}

    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        scl = avg.dtype.type(a.count(axis))
    else:
        wgt = asarray(weights)

        if issubclass(a.dtype.type, (np.integer, np.bool)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.shape != tuple(a.shape[ax] for ax in axis):
                raise ValueError(
                    "Shape of weights must be consistent with "
                    "shape of a along specified axis.")

            # setup wgt to broadcast along axis
            wgt = wgt.transpose(np.argsort(axis))
            wgt = wgt.reshape(tuple((s if ax in axis else 1)
                                    for ax, s in enumerate(a.shape)))

        if m is not nomask:
            wgt = wgt * (~a.mask)
            wgt.mask |= a.mask

        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        avg = np.multiply(a, wgt,
                          dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int, optional
        Axis along which the medians are computed. The default (None) is
        to compute the median along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow use of memory of input array (a) for
        calculations. The input array will be modified by the call to
        median. This will save memory when you do not need to preserve
        the contents of the input array. Treat the input as undefined,
        but it will probably be fully or partially sorted. Default is
        False. Note that, if `overwrite_input` is True, and the input
        is not already an `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    median : ndarray
        A new array holding the result is returned unless out is
        specified, in which case a reference to out is returned.
        Return data-type is `float64` for integers and floats smaller than
        `float64`, or the input data-type, otherwise.

    See Also
    --------
    mean

    Notes
    -----
    Given a vector ``V`` with ``N`` non masked values, the median of ``V``
    is the middle value of a sorted copy of ``V`` (``Vs``) - i.e.
    ``Vs[(N-1)/2]``, when ``N`` is odd, or ``{Vs[N/2 - 1] + Vs[N/2]}/2``
    when ``N`` is even.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array(np.arange(8), mask=[0]*4 + [1]*4)
    >>> np.ma.median(x)
    1.5

    >>> x = np.ma.array(np.arange(10).reshape(2, 5), mask=[0]*6 + [1]*4)
    >>> np.ma.median(x)
    2.5
    >>> np.ma.median(x, axis=-1, overwrite_input=True)
    masked_array(data=[2.0, 5.0],
                 mask=[False, False],
           fill_value=1e+20)

    """
    if not hasattr(a, 'mask'):
        m = np.median(getdata(a, subok=True), axis=axis,
                      out=out, overwrite_input=overwrite_input,
                      keepdims=keepdims)
        if isinstance(m, np.ndarray) and 1 <= m.ndim:
            return masked_array(m, copy=False)
        else:
            return m

    return _ureduce(a, func=_median, keepdims=keepdims, axis=axis, out=out,
                    overwrite_input=overwrite_input)


def _median(a, axis=None, out=None, overwrite_input=False):
    # when an unmasked NaN is present return it, so we need to sort the NaN
    # values behind the mask
    if np.issubdtype(a.dtype, np.inexact):
        fill_value = np.inf
    else:
        fill_value = None
    if overwrite_input:
        if axis is None:
            asorted = a.ravel()
            asorted.sort(fill_value=fill_value)
        else:
            a.sort(axis=axis, fill_value=fill_value)
            asorted = a
    else:
        asorted = sort(a, axis=axis, fill_value=fill_value)

    if axis is None:
        axis = 0
    else:
        axis = normalize_axis_index(axis, asorted.ndim)

    if asorted.shape[axis] == 0:
        # for empty axis integer indices fail so use slicing to get same result
        # as median (which is mean of empty slice = nan)
        indexer = [slice(None)] * asorted.ndim
        indexer[axis] = slice(0, 0)
        indexer = tuple(indexer)
        return np.ma.mean(asorted[indexer], axis=axis, out=out)

    if asorted.ndim == 1:
        idx, odd = divmod(count(asorted), 2)
        mid = asorted[idx + odd - 1:idx + 1]
        if np.issubdtype(asorted.dtype, np.inexact) and asorted.size > 0:
            # avoid inf / x = masked
            s = mid.sum(out=out)
            if not odd:
                s = np.true_divide(s, 2., casting='safe', out=out)
            s = np.lib._utils_impl._median_nancheck(asorted, s, axis)
        else:
            s = mid.mean(out=out)

        # if result is masked either the input contained enough
        # minimum_fill_value so that it would be the median or all values
        # masked
        if np.ma.is_masked(s) and not np.all(asorted.mask):
            return np.ma.minimum_fill_value(asorted)
        return s

    counts = count(asorted, axis=axis, keepdims=True)
    h = counts // 2

    # duplicate high if odd number of elements so mean does nothing
    odd = counts % 2 == 1
    l = np.where(odd, h, h - 1)

    lh = np.concatenate([l, h], axis=axis)

    # get low and high median
    low_high = np.take_along_axis(asorted, lh, axis=axis)

    def replace_masked(s):
        # Replace masked entries with minimum_full_value unless it all values
        # are masked. This is required as the sort order of values equal or
        # larger than the fill value is undefined and a valid value placed
        # elsewhere, e.g. [4, --, inf].
        if np.ma.is_masked(s):
            rep = (~np.all(asorted.mask, axis=axis, keepdims=True)) & s.mask
            s.data[rep] = np.ma.minimum_fill_value(asorted)
            s.mask[rep] = False

    replace_masked(low_high)

    if np.issubdtype(asorted.dtype, np.inexact):
        # avoid inf / x = masked
        s = np.ma.sum(low_high, axis=axis, out=out)
        np.true_divide(s.data, 2., casting='unsafe', out=s.data)

        s = np.lib._utils_impl._median_nancheck(asorted, s, axis)
    else:
        s = np.ma.mean(low_high, axis=axis, out=out)

    return s


def compress_nd(x, axis=None):
    """Suppress slices from multiple dimensions which contain masked values.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on. If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with `mask`
        set to `nomask`.
    axis : tuple of ints or int, optional
        Which dimensions to suppress slices from can be configured with this
        parameter.
        - If axis is a tuple of ints, those are the axes to suppress slices from.
        - If axis is an int, then that is the only axis to suppress slices from.
        - If axis is None, all axis are selected.

    Returns
    -------
    compress_array : ndarray
        The compressed array.

    Examples
    --------
    >>> import numpy as np
    >>> arr = [[1, 2], [3, 4]]
    >>> mask = [[0, 1], [0, 0]]
    >>> x = np.ma.array(arr, mask=mask)
    >>> np.ma.compress_nd(x, axis=0)
    array([[3, 4]])
    >>> np.ma.compress_nd(x, axis=1)
    array([[1],
           [3]])
    >>> np.ma.compress_nd(x)
    array([[3]])

    """
    x = asarray(x)
    m = getmask(x)
    # Set axis to tuple of ints
    if axis is None:
        axis = tuple(range(x.ndim))
    else:
        axis = normalize_axis_tuple(axis, x.ndim)

    # Nothing is masked: return x
    if m is nomask or not m.any():
        return x._data
    # All is masked: return empty
    if m.all():
        return nxarray([])
    # Filter elements through boolean indexing
    data = x._data
    for ax in axis:
        axes = tuple(list(range(ax)) + list(range(ax + 1, x.ndim)))
        data = data[(slice(None),) * ax + (~m.any(axis=axes),)]
    return data


def compress_rowcols(x, axis=None):
    """
    Suppress the rows and/or columns of a 2-D array that contain
    masked values.

    The suppression behavior is selected with the `axis` parameter.

    - If axis is None, both rows and columns are suppressed.
    - If axis is 0, only rows are suppressed.
    - If axis is 1 or -1, only columns are suppressed.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on.  If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.
    axis : int, optional
        Axis along which to perform the operation. Default is None.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> x
    masked_array(
      data=[[--, 1, 2],
            [--, 4, 5],
            [6, 7, 8]],
      mask=[[ True, False, False],
            [ True, False, False],
            [False, False, False]],
      fill_value=999999)

    >>> np.ma.compress_rowcols(x)
    array([[7, 8]])
    >>> np.ma.compress_rowcols(x, 0)
    array([[6, 7, 8]])
    >>> np.ma.compress_rowcols(x, 1)
    array([[1, 2],
           [4, 5],
           [7, 8]])

    """
    if asarray(x).ndim != 2:
        raise NotImplementedError("compress_rowcols works for 2D arrays only.")
    return compress_nd(x, axis=axis)


def compress_rows(a):
    """
    Suppress whole rows of a 2-D array that contain masked values.

    This is equivalent to ``np.ma.compress_rowcols(a, 0)``, see
    `compress_rowcols` for details.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on. If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    See Also
    --------
    compress_rowcols

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> np.ma.compress_rows(a)
    array([[6, 7, 8]])

    """
    a = asarray(a)
    if a.ndim != 2:
        raise NotImplementedError("compress_rows works for 2D arrays only.")
    return compress_rowcols(a, 0)


def compress_cols(a):
    """
    Suppress whole columns of a 2-D array that contain masked values.

    This is equivalent to ``np.ma.compress_rowcols(a, 1)``, see
    `compress_rowcols` for details.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on.  If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    See Also
    --------
    compress_rowcols

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> np.ma.compress_cols(a)
    array([[1, 2],
           [4, 5],
           [7, 8]])

    """
    a = asarray(a)
    if a.ndim != 2:
        raise NotImplementedError("compress_cols works for 2D arrays only.")
    return compress_rowcols(a, 1)


def mask_rowcols(a, axis=None):
    """
    Mask rows and/or columns of a 2D array that contain masked values.

    Mask whole rows and/or columns of a 2D array that contain
    masked values.  The masking behavior is selected using the
    `axis` parameter.

      - If `axis` is None, rows *and* columns are masked.
      - If `axis` is 0, only rows are masked.
      - If `axis` is 1 or -1, only columns are masked.

    Parameters
    ----------
    a : array_like, MaskedArray
        The array to mask.  If not a MaskedArray instance (or if no array
        elements are masked), the result is a MaskedArray with `mask` set
        to `nomask` (False). Must be a 2D array.
    axis : int, optional
        Axis along which to perform the operation. If None, applies to a
        flattened version of the array.

    Returns
    -------
    a : MaskedArray
        A modified version of the input array, masked depending on the value
        of the `axis` parameter.

    Raises
    ------
    NotImplementedError
        If input array `a` is not 2D.

    See Also
    --------
    mask_rows : Mask rows of a 2D array that contain masked values.
    mask_cols : Mask cols of a 2D array that contain masked values.
    masked_where : Mask where a condition is met.

    Notes
    -----
    The input array's mask is modified by this function.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)
    >>> np.ma.mask_rowcols(a)
    masked_array(
      data=[[0, --, 0],
            [--, --, --],
            [0, --, 0]],
      mask=[[False,  True, False],
            [ True,  True,  True],
            [False,  True, False]],
      fill_value=1)

    """
    a = array(a, subok=False)
    if a.ndim != 2:
        raise NotImplementedError("mask_rowcols works for 2D arrays only.")
    m = getmask(a)
    # Nothing is masked: return a
    if m is nomask or not m.any():
        return a
    maskedval = m.nonzero()
    a._mask = a._mask.copy()
    if not axis:
        a[np.unique(maskedval[0])] = masked
    if axis in [None, 1, -1]:
        a[:, np.unique(maskedval[1])] = masked
    return a


def mask_rows(a, axis=np._NoValue):
    """
    Mask rows of a 2D array that contain masked values.

    This function is a shortcut to ``mask_rowcols`` with `axis` equal to 0.

    See Also
    --------
    mask_rowcols : Mask rows and/or columns of a 2D array.
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)

    >>> np.ma.mask_rows(a)
    masked_array(
      data=[[0, 0, 0],
            [--, --, --],
            [0, 0, 0]],
      mask=[[False, False, False],
            [ True,  True,  True],
            [False, False, False]],
      fill_value=1)

    """
    if axis is not np._NoValue:
        # remove the axis argument when this deprecation expires
        # NumPy 1.18.0, 2019-11-28
        warnings.warn(
            "The axis argument has always been ignored, in future passing it "
            "will raise TypeError", DeprecationWarning, stacklevel=2)
    return mask_rowcols(a, 0)


def mask_cols(a, axis=np._NoValue):
    """
    Mask columns of a 2D array that contain masked values.

    This function is a shortcut to ``mask_rowcols`` with `axis` equal to 1.

    See Also
    --------
    mask_rowcols : Mask rows and/or columns of a 2D array.
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)
    >>> np.ma.mask_cols(a)
    masked_array(
      data=[[0, --, 0],
            [0, --, 0],
            [0, --, 0]],
      mask=[[False,  True, False],
            [False,  True, False],
            [False,  True, False]],
      fill_value=1)

    """
    if axis is not np._NoValue:
        # remove the axis argument when this deprecation expires
        # NumPy 1.18.0, 2019-11-28
        warnings.warn(
            "The axis argument has always been ignored, in future passing it "
            "will raise TypeError", DeprecationWarning, stacklevel=2)
    return mask_rowcols(a, 1)


#####--------------------------------------------------------------------------
#---- --- arraysetops ---
#####--------------------------------------------------------------------------

def ediff1d(arr, to_end=None, to_begin=None):
    """
    Compute the differences between consecutive elements of an array.

    This function is the equivalent of `numpy.ediff1d` that takes masked
    values into account, see `numpy.ediff1d` for details.

    See Also
    --------
    numpy.ediff1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.ma.array([1, 2, 4, 7, 0])
    >>> np.ma.ediff1d(arr)
    masked_array(data=[ 1,  2,  3, -7],
                 mask=False,
           fill_value=999999)

    """
    arr = ma.asanyarray(arr).flat
    ed = arr[1:] - arr[:-1]
    arrays = [ed]
    #
    if to_begin is not None:
        arrays.insert(0, to_begin)
    if to_end is not None:
        arrays.append(to_end)
    #
    if len(arrays) != 1:
        # We'll save ourselves a copy of a potentially large array in the common
        # case where neither to_begin or to_end was given.
        ed = hstack(arrays)
    #
    return ed


def unique(ar1, return_index=False, return_inverse=False):
    """
    Finds the unique elements of an array.

    Masked values are considered the same element (masked). The output array
    is always a masked array. See `numpy.unique` for more details.

    See Also
    --------
    numpy.unique : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> a = [1, 2, 1000, 2, 3]
    >>> mask = [0, 0, 1, 0, 0]
    >>> masked_a = np.ma.masked_array(a, mask)
    >>> masked_a
    masked_array(data=[1, 2, --, 2, 3],
                mask=[False, False,  True, False, False],
        fill_value=999999)
    >>> np.ma.unique(masked_a)
    masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999)
    >>> np.ma.unique(masked_a, return_index=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 4, 2]))
    >>> np.ma.unique(masked_a, return_inverse=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 3, 1, 2]))
    >>> np.ma.unique(masked_a, return_index=True, return_inverse=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 4, 2]), array([0, 1, 3, 1, 2]))
    """
    output = np.unique(ar1,
                       return_index=return_index,
                       return_inverse=return_inverse)
    if isinstance(output, tuple):
        output = list(output)
        output[0] = output[0].view(MaskedArray)
        output = tuple(output)
    else:
        output = output.view(MaskedArray)
    return output


def intersect1d(ar1, ar2, assume_unique=False):
    """
    Returns the unique elements common to both arrays.

    Masked values are considered equal one to the other.
    The output is always a masked array.

    See `numpy.intersect1d` for more details.

    See Also
    --------
    numpy.intersect1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([1, 3, 3, 3], mask=[0, 0, 0, 1])
    >>> y = np.ma.array([3, 1, 1, 1], mask=[0, 0, 0, 1])
    >>> np.ma.intersect1d(x, y)
    masked_array(data=[1, 3, --],
                 mask=[False, False,  True],
           fill_value=999999)

    """
    if assume_unique:
        aux = ma.concatenate((ar1, ar2))
    else:
        # Might be faster than unique( intersect1d( ar1, ar2 ) )?
        aux = ma.concatenate((unique(ar1), unique(ar2)))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


def setxor1d(ar1, ar2, assume_unique=False):
    """
    Set exclusive-or of 1-D arrays with unique elements.

    The output is always a masked array. See `numpy.setxor1d` for more details.

    See Also
    --------
    numpy.setxor1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> ar1 = np.ma.array([1, 2, 3, 2, 4])
    >>> ar2 = np.ma.array([2, 3, 5, 7, 5])
    >>> np.ma.setxor1d(ar1, ar2)
    masked_array(data=[1, 4, 5, 7],
                 mask=False,
           fill_value=999999)

    """
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)

    aux = ma.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux
    aux.sort()
    auxf = aux.filled()
#    flag = ediff1d( aux, to_end = 1, to_begin = 1 ) == 0
    flag = ma.concatenate(([True], (auxf[1:] != auxf[:-1]), [True]))
#    flag2 = ediff1d( flag ) == 0
    flag2 = (flag[1:] == flag[:-1])
    return aux[flag2]


def in1d(ar1, ar2, assume_unique=False, invert=False):
    """
    Test whether each element of an array is also present in a second
    array.

    The output is always a masked array. See `numpy.in1d` for more details.

    We recommend using :func:`isin` instead of `in1d` for new code.

    See Also
    --------
    isin       : Version of this function that preserves the shape of ar1.
    numpy.in1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> ar1 = np.ma.array([0, 1, 2, 5, 0])
    >>> ar2 = [0, 2]
    >>> np.ma.in1d(ar1, ar2)
    masked_array(data=[ True, False,  True, False,  True],
                 mask=False,
           fill_value=True)

    """
    if not assume_unique:
        ar1, rev_idx = unique(ar1, return_inverse=True)
        ar2 = unique(ar2)

    ar = ma.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = ma.concatenate((bool_ar, [invert]))
    indx = order.argsort(kind='mergesort')[:len(ar1)]

    if assume_unique:
        return flag[indx]
    else:
        return flag[indx][rev_idx]


def isin(element, test_elements, assume_unique=False, invert=False):
    """
    Calculates `element in test_elements`, broadcasting over
    `element` only.

    The output is always a masked array of the same shape as `element`.
    See `numpy.isin` for more details.

    See Also
    --------
    in1d       : Flattened version of this function.
    numpy.isin : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> element = np.ma.array([1, 2, 3, 4, 5, 6])
    >>> test_elements = [0, 2]
    >>> np.ma.isin(element, test_elements)
    masked_array(data=[False,  True, False, False, False, False],
                 mask=False,
           fill_value=True)

    """
    element = ma.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)


def union1d(ar1, ar2):
    """
    Union of two arrays.

    The output is always a masked array. See `numpy.union1d` for more details.

    See Also
    --------
    numpy.union1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> ar1 = np.ma.array([1, 2, 3, 4])
    >>> ar2 = np.ma.array([3, 4, 5, 6])
    >>> np.ma.union1d(ar1, ar2)
    masked_array(data=[1, 2, 3, 4, 5, 6],
             mask=False,
       fill_value=999999)

    """
    return unique(ma.concatenate((ar1, ar2), axis=None))


def setdiff1d(ar1, ar2, assume_unique=False):
    """
    Set difference of 1D arrays with unique elements.

    The output is always a masked array. See `numpy.setdiff1d` for more
    details.

    See Also
    --------
    numpy.setdiff1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
    >>> np.ma.setdiff1d(x, [1, 2])
    masked_array(data=[3, --],
                 mask=[False,  True],
           fill_value=999999)

    """
    if assume_unique:
        ar1 = ma.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]


###############################################################################
#                                Covariance                                   #
###############################################################################


def _covhelper(x, y=None, rowvar=True, allow_masked=True):
    """
    Private function for the computation of covariance and correlation
    coefficients.

    """
    x = ma.array(x, ndmin=2, copy=True, dtype=float)
    xmask = ma.getmaskarray(x)
    # Quick exit if we can't process masked data
    if not allow_masked and xmask.any():
        raise ValueError("Cannot process masked data.")
    #
    if x.shape[0] == 1:
        rowvar = True
    # Make sure that rowvar is either 0 or 1
    rowvar = int(bool(rowvar))
    axis = 1 - rowvar
    if rowvar:
        tup = (slice(None), None)
    else:
        tup = (None, slice(None))
    #
    if y is None:
        # Check if we can guarantee that the integers in the (N - ddof)
        # normalisation can be accurately represented with single-precision
        # before computing the dot product.
        if x.shape[0] > 2 ** 24 or x.shape[1] > 2 ** 24:
            xnm_dtype = np.float64
        else:
            xnm_dtype = np.float32
        xnotmask = np.logical_not(xmask).astype(xnm_dtype)
    else:
        y = array(y, copy=False, ndmin=2, dtype=float)
        ymask = ma.getmaskarray(y)
        if not allow_masked and ymask.any():
            raise ValueError("Cannot process masked data.")
        if xmask.any() or ymask.any():
            if y.shape == x.shape:
                # Define some common mask
                common_mask = np.logical_or(xmask, ymask)
                if common_mask is not nomask:
                    xmask = x._mask = y._mask = ymask = common_mask
                    x._sharedmask = False
                    y._sharedmask = False
        x = ma.concatenate((x, y), axis)
        # Check if we can guarantee that the integers in the (N - ddof)
        # normalisation can be accurately represented with single-precision
        # before computing the dot product.
        if x.shape[0] > 2 ** 24 or x.shape[1] > 2 ** 24:
            xnm_dtype = np.float64
        else:
            xnm_dtype = np.float32
        xnotmask = np.logical_not(np.concatenate((xmask, ymask), axis)).astype(
            xnm_dtype
        )
    x -= x.mean(axis=rowvar)[tup]
    return (x, xnotmask, rowvar)


def cov(x, y=None, rowvar=True, bias=False, allow_masked=True, ddof=None):
    """
    Estimate the covariance matrix.

    Except for the handling of missing data this function does the same as
    `numpy.cov`. For more details and examples, see `numpy.cov`.

    By default, masked values are recognized as such. If `x` and `y` have the
    same shape, a common mask is allocated: if ``x[i,j]`` is masked, then
    ``y[i,j]`` will also be masked.
    Setting `allow_masked` to False will raise an exception if values are
    missing in either of the input arrays.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N-1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. This keyword can be overridden by
        the keyword ``ddof`` in numpy versions >= 1.5.
    allow_masked : bool, optional
        If True, masked values are propagated pair-wise: if a value is masked
        in `x`, the corresponding value is masked in `y`.
        If False, raises a `ValueError` exception when some values are missing.
    ddof : {None, int}, optional
        If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
        the number of observations; this overrides the value implied by
        ``bias``. The default value is ``None``.

    Raises
    ------
    ValueError
        Raised if some values are missing and `allow_masked` is False.

    See Also
    --------
    numpy.cov

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([[0, 1], [1, 1]], mask=[0, 1, 0, 1])
    >>> y = np.ma.array([[1, 0], [0, 1]], mask=[0, 0, 1, 1])
    >>> np.ma.cov(x, y)
    masked_array(
    data=[[--, --, --, --],
          [--, --, --, --],
          [--, --, --, --],
          [--, --, --, --]],
    mask=[[ True,  True,  True,  True],
          [ True,  True,  True,  True],
          [ True,  True,  True,  True],
          [ True,  True,  True,  True]],
    fill_value=1e+20,
    dtype=float64)

    """
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be an integer")
    # Set up ddof
    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1

    (x, xnotmask, rowvar) = _covhelper(x, y, rowvar, allow_masked)
    if not rowvar:
        fact = np.dot(xnotmask.T, xnotmask) - ddof
        mask = np.less_equal(fact, 0, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.dot(filled(x.T, 0), filled(x.conj(), 0)) / fact
        result = ma.array(data, mask=mask).squeeze()
    else:
        fact = np.dot(xnotmask, xnotmask.T) - ddof
        mask = np.less_equal(fact, 0, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.dot(filled(x, 0), filled(x.T.conj(), 0)) / fact
        result = ma.array(data, mask=mask).squeeze()
    return result


def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, allow_masked=True,
             ddof=np._NoValue):
    """
    Return Pearson product-moment correlation coefficients.

    Except for the handling of missing data this function does the same as
    `numpy.corrcoef`. For more details and examples, see `numpy.corrcoef`.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    allow_masked : bool, optional
        If True, masked values are propagated pair-wise: if a value is masked
        in `x`, the corresponding value is masked in `y`.
        If False, raises an exception.  Because `bias` is deprecated, this
        argument needs to be treated as keyword only to avoid a warning.
    ddof : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0

    See Also
    --------
    numpy.corrcoef : Equivalent function in top-level NumPy module.
    cov : Estimate the covariance matrix.

    Notes
    -----
    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([[0, 1], [1, 1]], mask=[0, 1, 0, 1])
    >>> np.ma.corrcoef(x)
    masked_array(
      data=[[--, --],
            [--, --]],
      mask=[[ True,  True],
            [ True,  True]],
      fill_value=1e+20,
      dtype=float64)

    """
    msg = 'bias and ddof have no effect and are deprecated'
    if bias is not np._NoValue or ddof is not np._NoValue:
        # 2015-03-15, 1.10
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # Estimate the covariance matrix.
    corr = cov(x, y, rowvar, allow_masked=allow_masked)
    # The non-masked version returns a masked value for a scalar.
    try:
        std = ma.sqrt(ma.diagonal(corr))
    except ValueError:
        return ma.MaskedConstant()
    corr /= ma.multiply.outer(std, std)
    return corr

#####--------------------------------------------------------------------------
#---- --- Concatenation helpers ---
#####--------------------------------------------------------------------------

class MAxisConcatenator(AxisConcatenator):
    """
    Translate slice objects to concatenation along an axis.

    For documentation on usage, see `mr_class`.

    See Also
    --------
    mr_class

    """
    __slots__ = ()

    concatenate = staticmethod(concatenate)

    @classmethod
    def makemat(cls, arr):
        # There used to be a view as np.matrix here, but we may eventually
        # deprecate that class. In preparation, we use the unmasked version
        # to construct the matrix (with copy=False for backwards compatibility
        # with the .view)
        data = super().makemat(arr.data, copy=False)
        return array(data, mask=arr.mask)

    def __getitem__(self, key):
        # matrix builder syntax, like 'a, b; c, d'
        if isinstance(key, str):
            raise MAError("Unavailable for masked array.")

        return super().__getitem__(key)


class mr_class(MAxisConcatenator):
    """
    Translate slice objects to concatenation along the first axis.

    This is the masked array version of `r_`.

    See Also
    --------
    r_

    Examples
    --------
    >>> import numpy as np
    >>> np.ma.mr_[np.ma.array([1,2,3]), 0, 0, np.ma.array([4,5,6])]
    masked_array(data=[1, 2, 3, ..., 4, 5, 6],
                 mask=False,
           fill_value=999999)

    """
    __slots__ = ()

    def __init__(self):
        MAxisConcatenator.__init__(self, 0)


mr_ = mr_class()


#####--------------------------------------------------------------------------
#---- Find unmasked data ---
#####--------------------------------------------------------------------------

def ndenumerate(a, compressed=True):
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values,
    skipping elements that are masked. With `compressed=False`,
    `ma.masked` is yielded as the value of masked elements. This
    behavior differs from that of `numpy.ndenumerate`, which yields the
    value of the underlying data array.

    Notes
    -----
    .. versionadded:: 1.23.0

    Parameters
    ----------
    a : array_like
        An array with (possibly) masked elements.
    compressed : bool, optional
        If True (default), masked elements are skipped.

    See Also
    --------
    numpy.ndenumerate : Equivalent function ignoring any mask.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.arange(9).reshape((3, 3))
    >>> a[1, 0] = np.ma.masked
    >>> a[1, 2] = np.ma.masked
    >>> a[2, 1] = np.ma.masked
    >>> a
    masked_array(
      data=[[0, 1, 2],
            [--, 4, --],
            [6, --, 8]],
      mask=[[False, False, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=999999)
    >>> for index, x in np.ma.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 1) 4
    (2, 0) 6
    (2, 2) 8

    >>> for index, x in np.ma.ndenumerate(a, compressed=False):
    ...     print(index, x)
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) --
    (1, 1) 4
    (1, 2) --
    (2, 0) 6
    (2, 1) --
    (2, 2) 8
    """
    for it, mask in zip(np.ndenumerate(a), getmaskarray(a).flat):
        if not mask:
            yield it
        elif not compressed:
            yield it[0], masked


def flatnotmasked_edges(a):
    """
    Find the indices of the first and last unmasked values.

    Expects a 1-D `MaskedArray`, returns None if all values are masked.

    Parameters
    ----------
    a : array_like
        Input 1-D `MaskedArray`

    Returns
    -------
    edges : ndarray or None
        The indices of first and last non-masked value in the array.
        Returns None if all values are masked.

    See Also
    --------
    flatnotmasked_contiguous, notmasked_contiguous, notmasked_edges
    clump_masked, clump_unmasked

    Notes
    -----
    Only accepts 1-D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.arange(10)
    >>> np.ma.flatnotmasked_edges(a)
    array([0, 9])

    >>> mask = (a < 3) | (a > 8) | (a == 5)
    >>> a[mask] = np.ma.masked
    >>> np.array(a[~a.mask])
    array([3, 4, 6, 7, 8])

    >>> np.ma.flatnotmasked_edges(a)
    array([3, 8])

    >>> a[:] = np.ma.masked
    >>> print(np.ma.flatnotmasked_edges(a))
    None

    """
    m = getmask(a)
    if m is nomask or not np.any(m):
        return np.array([0, a.size - 1])
    unmasked = np.flatnonzero(~m)
    if len(unmasked) > 0:
        return unmasked[[0, -1]]
    else:
        return None


def notmasked_edges(a, axis=None):
    """
    Find the indices of the first and last unmasked values along an axis.

    If all values are masked, return None.  Otherwise, return a list
    of two tuples, corresponding to the indices of the first and last
    unmasked values respectively.

    Parameters
    ----------
    a : array_like
        The input array.
    axis : int, optional
        Axis along which to perform the operation.
        If None (default), applies to a flattened version of the array.

    Returns
    -------
    edges : ndarray or list
        An array of start and end indexes if there are any masked data in
        the array. If there are no masked data in the array, `edges` is a
        list of the first and last index.

    See Also
    --------
    flatnotmasked_contiguous, flatnotmasked_edges, notmasked_contiguous
    clump_masked, clump_unmasked

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(9).reshape((3, 3))
    >>> m = np.zeros_like(a)
    >>> m[1:, 1:] = 1

    >>> am = np.ma.array(a, mask=m)
    >>> np.array(am[~am.mask])
    array([0, 1, 2, 3, 6])

    >>> np.ma.notmasked_edges(am)
    array([0, 6])

    """
    a = asarray(a)
    if axis is None or a.ndim == 1:
        return flatnotmasked_edges(a)
    m = getmaskarray(a)
    idx = array(np.indices(a.shape), mask=np.asarray([m] * a.ndim))
    return [tuple(idx[i].min(axis).compressed() for i in range(a.ndim)),
            tuple(idx[i].max(axis).compressed() for i in range(a.ndim)), ]


def flatnotmasked_contiguous(a):
    """
    Find contiguous unmasked data in a masked array.

    Parameters
    ----------
    a : array_like
        The input array.

    Returns
    -------
    slice_list : list
        A sorted sequence of `slice` objects (start index, end index).

    See Also
    --------
    flatnotmasked_edges, notmasked_contiguous, notmasked_edges
    clump_masked, clump_unmasked

    Notes
    -----
    Only accepts 2-D arrays at most.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.arange(10)
    >>> np.ma.flatnotmasked_contiguous(a)
    [slice(0, 10, None)]

    >>> mask = (a < 3) | (a > 8) | (a == 5)
    >>> a[mask] = np.ma.masked
    >>> np.array(a[~a.mask])
    array([3, 4, 6, 7, 8])

    >>> np.ma.flatnotmasked_contiguous(a)
    [slice(3, 5, None), slice(6, 9, None)]
    >>> a[:] = np.ma.masked
    >>> np.ma.flatnotmasked_contiguous(a)
    []

    """
    m = getmask(a)
    if m is nomask:
        return [slice(0, a.size)]
    i = 0
    result = []
    for (k, g) in itertools.groupby(m.ravel()):
        n = len(list(g))
        if not k:
            result.append(slice(i, i + n))
        i += n
    return result


def notmasked_contiguous(a, axis=None):
    """
    Find contiguous unmasked data in a masked array along the given axis.

    Parameters
    ----------
    a : array_like
        The input array.
    axis : int, optional
        Axis along which to perform the operation.
        If None (default), applies to a flattened version of the array, and this
        is the same as `flatnotmasked_contiguous`.

    Returns
    -------
    endpoints : list
        A list of slices (start and end indexes) of unmasked indexes
        in the array.

        If the input is 2d and axis is specified, the result is a list of lists.

    See Also
    --------
    flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
    clump_masked, clump_unmasked

    Notes
    -----
    Only accepts 2-D arrays at most.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(12).reshape((3, 4))
    >>> mask = np.zeros_like(a)
    >>> mask[1:, :-1] = 1; mask[0, 1] = 1; mask[-1, 0] = 0
    >>> ma = np.ma.array(a, mask=mask)
    >>> ma
    masked_array(
      data=[[0, --, 2, 3],
            [--, --, --, 7],
            [8, --, --, 11]],
      mask=[[False,  True, False, False],
            [ True,  True,  True, False],
            [False,  True,  True, False]],
      fill_value=999999)
    >>> np.array(ma[~ma.mask])
    array([ 0,  2,  3,  7, 8, 11])

    >>> np.ma.notmasked_contiguous(ma)
    [slice(0, 1, None), slice(2, 4, None), slice(7, 9, None), slice(11, 12, None)]

    >>> np.ma.notmasked_contiguous(ma, axis=0)
    [[slice(0, 1, None), slice(2, 3, None)], [], [slice(0, 1, None)], [slice(0, 3, None)]]

    >>> np.ma.notmasked_contiguous(ma, axis=1)
    [[slice(0, 1, None), slice(2, 4, None)], [slice(3, 4, None)], [slice(0, 1, None), slice(3, 4, None)]]

    """  # noqa: E501
    a = asarray(a)
    nd = a.ndim
    if nd > 2:
        raise NotImplementedError("Currently limited to at most 2D array.")
    if axis is None or nd == 1:
        return flatnotmasked_contiguous(a)
    #
    result = []
    #
    other = (axis + 1) % 2
    idx = [0, 0]
    idx[axis] = slice(None, None)
    #
    for i in range(a.shape[other]):
        idx[other] = i
        result.append(flatnotmasked_contiguous(a[tuple(idx)]))
    return result


def _ezclump(mask):
    """
    Finds the clumps (groups of data with the same values) for a 1D bool array.

    Returns a series of slices.
    """
    if mask.ndim > 1:
        mask = mask.ravel()
    idx = (mask[1:] ^ mask[:-1]).nonzero()
    idx = idx[0] + 1

    if mask[0]:
        if len(idx) == 0:
            return [slice(0, mask.size)]

        r = [slice(0, idx[0])]
        r.extend((slice(left, right)
                  for left, right in zip(idx[1:-1:2], idx[2::2])))
    else:
        if len(idx) == 0:
            return []

        r = [slice(left, right) for left, right in zip(idx[:-1:2], idx[1::2])]

    if mask[-1]:
        r.append(slice(idx[-1], mask.size))
    return r


def clump_unmasked(a):
    """
    Return list of slices corresponding to the unmasked clumps of a 1-D array.
    (A "clump" is defined as a contiguous region of the array).

    Parameters
    ----------
    a : ndarray
        A one-dimensional masked array.

    Returns
    -------
    slices : list of slice
        The list of slices, one for each continuous region of unmasked
        elements in `a`.

    See Also
    --------
    flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
    notmasked_contiguous, clump_masked

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.masked_array(np.arange(10))
    >>> a[[0, 1, 2, 6, 8, 9]] = np.ma.masked
    >>> np.ma.clump_unmasked(a)
    [slice(3, 6, None), slice(7, 8, None)]

    """
    mask = getattr(a, '_mask', nomask)
    if mask is nomask:
        return [slice(0, a.size)]
    return _ezclump(~mask)


def clump_masked(a):
    """
    Returns a list of slices corresponding to the masked clumps of a 1-D array.
    (A "clump" is defined as a contiguous region of the array).

    Parameters
    ----------
    a : ndarray
        A one-dimensional masked array.

    Returns
    -------
    slices : list of slice
        The list of slices, one for each continuous region of masked elements
        in `a`.

    See Also
    --------
    flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
    notmasked_contiguous, clump_unmasked

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.masked_array(np.arange(10))
    >>> a[[0, 1, 2, 6, 8, 9]] = np.ma.masked
    >>> np.ma.clump_masked(a)
    [slice(0, 3, None), slice(6, 7, None), slice(8, 10, None)]

    """
    mask = ma.getmask(a)
    if mask is nomask:
        return []
    return _ezclump(mask)


###############################################################################
#                              Polynomial fit                                 #
###############################################################################


def vander(x, n=None):
    """
    Masked values in the input array result in rows of zeros.

    """
    _vander = np.vander(x, n)
    m = getmask(x)
    if m is not nomask:
        _vander[m] = 0
    return _vander


vander.__doc__ = ma.doc_note(np.vander.__doc__, vander.__doc__)


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    Any masked values in x is propagated in y, and vice-versa.

    """
    x = asarray(x)
    y = asarray(y)

    m = getmask(x)
    if y.ndim == 1:
        m = mask_or(m, getmask(y))
    elif y.ndim == 2:
        my = getmask(mask_rows(y))
        if my is not nomask:
            m = mask_or(m, my[:, 0])
    else:
        raise TypeError("Expected a 1D or 2D array for y!")

    if w is not None:
        w = asarray(w)
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        m = mask_or(m, getmask(w))

    if m is not nomask:
        not_m = ~m
        if w is not None:
            w = w[not_m]
        return np.polyfit(x[not_m], y[not_m], deg, rcond, full, w, cov)
    else:
        return np.polyfit(x, y, deg, rcond, full, w, cov)


polyfit.__doc__ = ma.doc_note(np.polyfit.__doc__, polyfit.__doc__)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\storage.py ===
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
class SharedStoragePrivateAggregationConfig:
    '''
    Represents a dictionary object passed in as privateAggregationConfig to
    run or selectURL.
    '''
    #: Configures the maximum size allowed for filtering IDs.
    filtering_id_max_bytes: int

    #: The chosen aggregation service deployment.
    aggregation_coordinator_origin: typing.Optional[str] = None

    #: The context ID provided.
    context_id: typing.Optional[str] = None

    #: The limit on the number of contributions in the final report.
    max_contributions: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['filteringIdMaxBytes'] = self.filtering_id_max_bytes
        if self.aggregation_coordinator_origin is not None:
            json['aggregationCoordinatorOrigin'] = self.aggregation_coordinator_origin
        if self.context_id is not None:
            json['contextId'] = self.context_id
        if self.max_contributions is not None:
            json['maxContributions'] = self.max_contributions
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filtering_id_max_bytes=int(json['filteringIdMaxBytes']),
            aggregation_coordinator_origin=str(json['aggregationCoordinatorOrigin']) if 'aggregationCoordinatorOrigin' in json else None,
            context_id=str(json['contextId']) if 'contextId' in json else None,
            max_contributions=int(json['maxContributions']) if 'maxContributions' in json else None,
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
    #: Present only for SharedStorageAccessMethods: addModule and
    #: createWorklet.
    script_source_url: typing.Optional[str] = None

    #: String denoting "context-origin", "script-origin", or a custom
    #: origin to be used as the worklet's data origin.
    #: Present only for SharedStorageAccessMethod: createWorklet.
    data_origin: typing.Optional[str] = None

    #: Name of the registered operation to be run.
    #: Present only for SharedStorageAccessMethods: run and selectURL.
    operation_name: typing.Optional[str] = None

    #: Whether or not to keep the worket alive for future run or selectURL
    #: calls.
    #: Present only for SharedStorageAccessMethods: run and selectURL.
    keep_alive: typing.Optional[bool] = None

    #: Configures the private aggregation options.
    #: Present only for SharedStorageAccessMethods: run and selectURL.
    private_aggregation_config: typing.Optional[SharedStoragePrivateAggregationConfig] = None

    #: The operation's serialized data in bytes (converted to a string).
    #: Present only for SharedStorageAccessMethods: run and selectURL.
    #: TODO(crbug.com/401011862): Consider updating this parameter to binary.
    serialized_data: typing.Optional[str] = None

    #: Array of candidate URLs' specs, along with any associated metadata.
    #: Present only for SharedStorageAccessMethod: selectURL.
    urls_with_metadata: typing.Optional[typing.List[SharedStorageUrlWithMetadata]] = None

    #: Spec of the URN:UUID generated for a selectURL call.
    #: Present only for SharedStorageAccessMethod: selectURL.
    urn_uuid: typing.Optional[str] = None

    #: Key for a specific entry in an origin's shared storage.
    #: Present only for SharedStorageAccessMethods: set, append, delete, and
    #: get.
    key: typing.Optional[str] = None

    #: Value for a specific entry in an origin's shared storage.
    #: Present only for SharedStorageAccessMethods: set and append.
    value: typing.Optional[str] = None

    #: Whether or not to set an entry for a key if that key is already present.
    #: Present only for SharedStorageAccessMethod: set.
    ignore_if_present: typing.Optional[bool] = None

    #: If the method is called on a worklet, or as part of
    #: a worklet script, it will have an ID for the associated worklet.
    #: Present only for SharedStorageAccessMethods: addModule, createWorklet,
    #: run, selectURL, and any other SharedStorageAccessMethod when the
    #: SharedStorageAccessScope is worklet.
    worklet_id: typing.Optional[str] = None

    #: Name of the lock to be acquired, if present.
    #: Optionally present only for SharedStorageAccessMethods: batchUpdate,
    #: set, append, delete, and clear.
    with_lock: typing.Optional[str] = None

    #: If the method has been called as part of a batchUpdate, then this
    #: number identifies the batch to which it belongs.
    #: Optionally present only for SharedStorageAccessMethods:
    #: batchUpdate (required), set, append, delete, and clear.
    batch_update_id: typing.Optional[str] = None

    #: Number of modifier methods sent in batch.
    #: Present only for SharedStorageAccessMethod: batchUpdate.
    batch_size: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        if self.script_source_url is not None:
            json['scriptSourceUrl'] = self.script_source_url
        if self.data_origin is not None:
            json['dataOrigin'] = self.data_origin
        if self.operation_name is not None:
            json['operationName'] = self.operation_name
        if self.keep_alive is not None:
            json['keepAlive'] = self.keep_alive
        if self.private_aggregation_config is not None:
            json['privateAggregationConfig'] = self.private_aggregation_config.to_json()
        if self.serialized_data is not None:
            json['serializedData'] = self.serialized_data
        if self.urls_with_metadata is not None:
            json['urlsWithMetadata'] = [i.to_json() for i in self.urls_with_metadata]
        if self.urn_uuid is not None:
            json['urnUuid'] = self.urn_uuid
        if self.key is not None:
            json['key'] = self.key
        if self.value is not None:
            json['value'] = self.value
        if self.ignore_if_present is not None:
            json['ignoreIfPresent'] = self.ignore_if_present
        if self.worklet_id is not None:
            json['workletId'] = self.worklet_id
        if self.with_lock is not None:
            json['withLock'] = self.with_lock
        if self.batch_update_id is not None:
            json['batchUpdateId'] = self.batch_update_id
        if self.batch_size is not None:
            json['batchSize'] = self.batch_size
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            script_source_url=str(json['scriptSourceUrl']) if 'scriptSourceUrl' in json else None,
            data_origin=str(json['dataOrigin']) if 'dataOrigin' in json else None,
            operation_name=str(json['operationName']) if 'operationName' in json else None,
            keep_alive=bool(json['keepAlive']) if 'keepAlive' in json else None,
            private_aggregation_config=SharedStoragePrivateAggregationConfig.from_json(json['privateAggregationConfig']) if 'privateAggregationConfig' in json else None,
            serialized_data=str(json['serializedData']) if 'serializedData' in json else None,
            urls_with_metadata=[SharedStorageUrlWithMetadata.from_json(i) for i in json['urlsWithMetadata']] if 'urlsWithMetadata' in json else None,
            urn_uuid=str(json['urnUuid']) if 'urnUuid' in json else None,
            key=str(json['key']) if 'key' in json else None,
            value=str(json['value']) if 'value' in json else None,
            ignore_if_present=bool(json['ignoreIfPresent']) if 'ignoreIfPresent' in json else None,
            worklet_id=str(json['workletId']) if 'workletId' in json else None,
            with_lock=str(json['withLock']) if 'withLock' in json else None,
            batch_update_id=str(json['batchUpdateId']) if 'batchUpdateId' in json else None,
            batch_size=int(json['batchSize']) if 'batchSize' in json else None,
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
class AttributionReportingNamedBudgetDef:
    name: str

    budget: int

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['budget'] = self.budget
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            budget=int(json['budget']),
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

    named_budgets: typing.List[AttributionReportingNamedBudgetDef]

    debug_reporting: bool

    event_level_epsilon: float

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
        json['namedBudgets'] = [i.to_json() for i in self.named_budgets]
        json['debugReporting'] = self.debug_reporting
        json['eventLevelEpsilon'] = self.event_level_epsilon
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
            named_budgets=[AttributionReportingNamedBudgetDef.from_json(i) for i in json['namedBudgets']],
            debug_reporting=bool(json['debugReporting']),
            event_level_epsilon=float(json['eventLevelEpsilon']),
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
class AttributionReportingNamedBudgetCandidate:
    filters: AttributionReportingFilterPair

    name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['filters'] = self.filters.to_json()
        if self.name is not None:
            json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            filters=AttributionReportingFilterPair.from_json(json['filters']),
            name=str(json['name']) if 'name' in json else None,
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

    named_budgets: typing.List[AttributionReportingNamedBudgetCandidate]

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
        json['namedBudgets'] = [i.to_json() for i in self.named_budgets]
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
            named_budgets=[AttributionReportingNamedBudgetCandidate.from_json(i) for i in json['namedBudgets']],
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


def set_protected_audience_k_anonymity(
        owner: str,
        name: str,
        hashes: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    :param owner:
    :param name:
    :param hashes:
    '''
    params: T_JSON_DICT = dict()
    params['owner'] = owner
    params['name'] = name
    params['hashes'] = [i for i in hashes]
    cmd_dict: T_JSON_DICT = {
        'method': 'Storage.setProtectedAudienceKAnonymity',
        'params': params,
    }
    json = yield cmd_dict


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

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\langhelpers.py ===
# util/langhelpers.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Routines to help with the creation, loading and introspection of
modules, classes, hierarchies, attributes, functions, and methods.

"""
from __future__ import annotations

import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings

from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_F = TypeVar("_F", bound=Callable[..., Any])
_MP = TypeVar("_MP", bound="memoized_property[Any]")
_MA = TypeVar("_MA", bound="HasMemoized.memoized_attribute[Any]")
_HP = TypeVar("_HP", bound="hybridproperty[Any]")
_HM = TypeVar("_HM", bound="hybridmethod[Any]")


if compat.py314:
    # vendor a minimal form of get_annotations per
    # https://github.com/python/cpython/issues/133684#issuecomment-2863841891

    from annotationlib import call_annotate_function  # type: ignore
    from annotationlib import Format

    def _get_and_call_annotate(obj, format):  # noqa: A002
        annotate = getattr(obj, "__annotate__", None)
        if annotate is not None:
            ann = call_annotate_function(annotate, format, owner=obj)
            if not isinstance(ann, dict):
                raise ValueError(f"{obj!r}.__annotate__ returned a non-dict")
            return ann
        return None

    # this is ported from py3.13.0a7
    _BASE_GET_ANNOTATIONS = type.__dict__["__annotations__"].__get__  # type: ignore  # noqa: E501

    def _get_dunder_annotations(obj):
        if isinstance(obj, type):
            try:
                ann = _BASE_GET_ANNOTATIONS(obj)
            except AttributeError:
                # For static types, the descriptor raises AttributeError.
                return {}
        else:
            ann = getattr(obj, "__annotations__", None)
            if ann is None:
                return {}

        if not isinstance(ann, dict):
            raise ValueError(
                f"{obj!r}.__annotations__ is neither a dict nor None"
            )
        return dict(ann)

    def _vendored_get_annotations(
        obj: Any, *, format: Format  # noqa: A002
    ) -> Mapping[str, Any]:
        """A sparse implementation of annotationlib.get_annotations()"""

        try:
            ann = _get_dunder_annotations(obj)
        except Exception:
            pass
        else:
            if ann is not None:
                return dict(ann)

        # But if __annotations__ threw a NameError, we try calling __annotate__
        ann = _get_and_call_annotate(obj, format)
        if ann is None:
            # If that didn't work either, we have a very weird object:
            # evaluating
            # __annotations__ threw NameError and there is no __annotate__.
            # In that case,
            # we fall back to trying __annotations__ again.
            ann = _get_dunder_annotations(obj)

        if ann is None:
            if isinstance(obj, type) or callable(obj):
                return {}
            raise TypeError(f"{obj!r} does not have annotations")

        if not ann:
            return {}

        return dict(ann)

    def get_annotations(obj: Any) -> Mapping[str, Any]:
        # FORWARDREF has the effect of giving us ForwardRefs and not
        # actually trying to evaluate the annotations.  We need this so
        # that the annotations act as much like
        # "from __future__ import annotations" as possible, which is going
        # away in future python as a separate mode
        return _vendored_get_annotations(obj, format=Format.FORWARDREF)

elif compat.py310:

    def get_annotations(obj: Any) -> Mapping[str, Any]:
        return inspect.get_annotations(obj)

else:

    def get_annotations(obj: Any) -> Mapping[str, Any]:
        # it's been observed that cls.__annotations__ can be non present.
        # it's not clear what causes this, running under tox py37/38 it
        # happens, running straight pytest it doesnt

        # https://docs.python.org/3/howto/annotations.html#annotations-howto
        if isinstance(obj, type):
            ann = obj.__dict__.get("__annotations__", None)
        else:
            ann = getattr(obj, "__annotations__", None)

        if ann is None:
            return _collections.EMPTY_DICT
        else:
            return cast("Mapping[str, Any]", ann)


def md5_hex(x: Any) -> str:
    x = x.encode("utf-8")
    m = compat.md5_not_for_security()
    m.update(x)
    return cast(str, m.hexdigest())


class safe_reraise:
    """Reraise an exception after invoking some
    handler code.

    Stores the existing exception info before
    invoking so that it is maintained across a potential
    coroutine context switch.

    e.g.::

        try:
            sess.commit()
        except:
            with safe_reraise():
                sess.rollback()

    TODO: we should at some point evaluate current behaviors in this regard
    based on current greenlet, gevent/eventlet implementations in Python 3, and
    also see the degree to which our own asyncio (based on greenlet also) is
    impacted by this. .rollback() will cause IO / context switch to occur in
    all these scenarios; what happens to the exception context from an
    "except:" block if we don't explicitly store it? Original issue was #2703.

    """

    __slots__ = ("_exc_info",)

    _exc_info: Union[
        None,
        Tuple[
            Type[BaseException],
            BaseException,
            types.TracebackType,
        ],
        Tuple[None, None, None],
    ]

    def __enter__(self) -> None:
        self._exc_info = sys.exc_info()

    def __exit__(
        self,
        type_: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> NoReturn:
        assert self._exc_info is not None
        # see #2703 for notes
        if type_ is None:
            exc_type, exc_value, exc_tb = self._exc_info
            assert exc_value is not None
            self._exc_info = None  # remove potential circular references
            raise exc_value.with_traceback(exc_tb)
        else:
            self._exc_info = None  # remove potential circular references
            assert value is not None
            raise value.with_traceback(traceback)


def walk_subclasses(cls: Type[_T]) -> Iterator[Type[_T]]:
    seen: Set[Any] = set()

    stack = [cls]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        else:
            seen.add(cls)
        stack.extend(cls.__subclasses__())
        yield cls


def string_or_unprintable(element: Any) -> str:
    if isinstance(element, str):
        return element
    else:
        try:
            return str(element)
        except Exception:
            return "unprintable element %r" % element


def clsname_as_plain_name(
    cls: Type[Any], use_name: Optional[str] = None
) -> str:
    name = use_name or cls.__name__
    return " ".join(n.lower() for n in re.findall(r"([A-Z][a-z]+|SQL)", name))


def method_is_overridden(
    instance_or_cls: Union[Type[Any], object],
    against_method: Callable[..., Any],
) -> bool:
    """Return True if the two class methods don't match."""

    if not isinstance(instance_or_cls, type):
        current_cls = instance_or_cls.__class__
    else:
        current_cls = instance_or_cls

    method_name = against_method.__name__

    current_method: types.MethodType = getattr(current_cls, method_name)

    return current_method != against_method


def decode_slice(slc: slice) -> Tuple[Any, ...]:
    """decode a slice object as sent to __getitem__.

    takes into account the 2.5 __index__() method, basically.

    """
    ret: List[Any] = []
    for x in slc.start, slc.stop, slc.step:
        if hasattr(x, "__index__"):
            x = x.__index__()
        ret.append(x)
    return tuple(ret)


def _unique_symbols(used: Sequence[str], *bases: str) -> Iterator[str]:
    used_set = set(used)
    for base in bases:
        pool = itertools.chain(
            (base,),
            map(lambda i: base + str(i), range(1000)),
        )
        for sym in pool:
            if sym not in used_set:
                used_set.add(sym)
                yield sym
                break
        else:
            raise NameError("exhausted namespace for symbol base %s" % base)


def map_bits(fn: Callable[[int], Any], n: int) -> Iterator[Any]:
    """Call the given function given each nonzero bit from n."""

    while n:
        b = n & (~n + 1)
        yield fn(b)
        n ^= b


_Fn = TypeVar("_Fn", bound="Callable[..., Any]")

# this seems to be in flux in recent mypy versions


def decorator(target: Callable[..., Any]) -> Callable[[_Fn], _Fn]:
    """A signature-matching decorator factory."""

    def decorate(fn: _Fn) -> _Fn:
        if not inspect.isfunction(fn) and not inspect.ismethod(fn):
            raise Exception("not a decoratable function")

        # Python 3.14 defer creating __annotations__ until its used.
        # We do not want to create __annotations__ now.
        annofunc = getattr(fn, "__annotate__", None)
        if annofunc is not None:
            fn.__annotate__ = None  # type: ignore[union-attr]
            try:
                spec = compat.inspect_getfullargspec(fn)
            finally:
                fn.__annotate__ = annofunc  # type: ignore[union-attr]
        else:
            spec = compat.inspect_getfullargspec(fn)

        # Do not generate code for annotations.
        # update_wrapper() copies the annotation from fn to decorated.
        # We use dummy defaults for code generation to avoid having
        # copy of large globals for compiling.
        # We copy __defaults__ and __kwdefaults__ from fn to decorated.
        empty_defaults = (None,) * len(spec.defaults or ())
        empty_kwdefaults = dict.fromkeys(spec.kwonlydefaults or ())
        spec = spec._replace(
            annotations={},
            defaults=empty_defaults,
            kwonlydefaults=empty_kwdefaults,
        )

        names = (
            tuple(cast("Tuple[str, ...]", spec[0]))
            + cast("Tuple[str, ...]", spec[1:3])
            + (fn.__name__,)
        )
        targ_name, fn_name = _unique_symbols(names, "target", "fn")

        metadata: Dict[str, Optional[str]] = dict(target=targ_name, fn=fn_name)
        metadata.update(format_argspec_plus(spec, grouped=False))
        metadata["name"] = fn.__name__

        if inspect.iscoroutinefunction(fn):
            metadata["prefix"] = "async "
            metadata["target_prefix"] = "await "
        else:
            metadata["prefix"] = ""
            metadata["target_prefix"] = ""

        # look for __ positional arguments.  This is a convention in
        # SQLAlchemy that arguments should be passed positionally
        # rather than as keyword
        # arguments.  note that apply_pos doesn't currently work in all cases
        # such as when a kw-only indicator "*" is present, which is why
        # we limit the use of this to just that case we can detect.  As we add
        # more kinds of methods that use @decorator, things may have to
        # be further improved in this area
        if "__" in repr(spec[0]):
            code = (
                """\
%(prefix)sdef %(name)s%(grouped_args)s:
    return %(target_prefix)s%(target)s(%(fn)s, %(apply_pos)s)
"""
                % metadata
            )
        else:
            code = (
                """\
%(prefix)sdef %(name)s%(grouped_args)s:
    return %(target_prefix)s%(target)s(%(fn)s, %(apply_kw)s)
"""
                % metadata
            )

        env: Dict[str, Any] = {
            targ_name: target,
            fn_name: fn,
            "__name__": fn.__module__,
        }

        decorated = cast(
            types.FunctionType,
            _exec_code_in_env(code, env, fn.__name__),
        )
        decorated.__defaults__ = fn.__defaults__
        decorated.__kwdefaults__ = fn.__kwdefaults__  # type: ignore
        return update_wrapper(decorated, fn)  # type: ignore[return-value]

    return update_wrapper(decorate, target)  # type: ignore[return-value]


def _exec_code_in_env(
    code: Union[str, types.CodeType], env: Dict[str, Any], fn_name: str
) -> Callable[..., Any]:
    exec(code, env)
    return env[fn_name]  # type: ignore[no-any-return]


_PF = TypeVar("_PF")
_TE = TypeVar("_TE")


class PluginLoader:
    def __init__(
        self, group: str, auto_fn: Optional[Callable[..., Any]] = None
    ):
        self.group = group
        self.impls: Dict[str, Any] = {}
        self.auto_fn = auto_fn

    def clear(self):
        self.impls.clear()

    def load(self, name: str) -> Any:
        if name in self.impls:
            return self.impls[name]()

        if self.auto_fn:
            loader = self.auto_fn(name)
            if loader:
                self.impls[name] = loader
                return loader()

        for impl in compat.importlib_metadata_get(self.group):
            if impl.name == name:
                self.impls[name] = impl.load
                return impl.load()

        raise exc.NoSuchModuleError(
            "Can't load plugin: %s:%s" % (self.group, name)
        )

    def register(self, name: str, modulepath: str, objname: str) -> None:
        def load():
            mod = __import__(modulepath)
            for token in modulepath.split(".")[1:]:
                mod = getattr(mod, token)
            return getattr(mod, objname)

        self.impls[name] = load

    def deregister(self, name: str) -> None:
        del self.impls[name]


def _inspect_func_args(fn):
    try:
        co_varkeywords = inspect.CO_VARKEYWORDS
    except AttributeError:
        # https://docs.python.org/3/library/inspect.html
        # The flags are specific to CPython, and may not be defined in other
        # Python implementations. Furthermore, the flags are an implementation
        # detail, and can be removed or deprecated in future Python releases.
        spec = compat.inspect_getfullargspec(fn)
        return spec[0], bool(spec[2])
    else:
        # use fn.__code__ plus flags to reduce method call overhead
        co = fn.__code__
        nargs = co.co_argcount
        return (
            list(co.co_varnames[:nargs]),
            bool(co.co_flags & co_varkeywords),
        )


@overload
def get_cls_kwargs(
    cls: type,
    *,
    _set: Optional[Set[str]] = None,
    raiseerr: Literal[True] = ...,
) -> Set[str]: ...


@overload
def get_cls_kwargs(
    cls: type, *, _set: Optional[Set[str]] = None, raiseerr: bool = False
) -> Optional[Set[str]]: ...


def get_cls_kwargs(
    cls: type, *, _set: Optional[Set[str]] = None, raiseerr: bool = False
) -> Optional[Set[str]]:
    r"""Return the full set of inherited kwargs for the given `cls`.

    Probes a class's __init__ method, collecting all named arguments.  If the
    __init__ defines a \**kwargs catch-all, then the constructor is presumed
    to pass along unrecognized keywords to its base classes, and the
    collection process is repeated recursively on each of the bases.

    Uses a subset of inspect.getfullargspec() to cut down on method overhead,
    as this is used within the Core typing system to create copies of type
    objects which is a performance-sensitive operation.

    No anonymous tuple arguments please !

    """
    toplevel = _set is None
    if toplevel:
        _set = set()
    assert _set is not None

    ctr = cls.__dict__.get("__init__", False)

    has_init = (
        ctr
        and isinstance(ctr, types.FunctionType)
        and isinstance(ctr.__code__, types.CodeType)
    )

    if has_init:
        names, has_kw = _inspect_func_args(ctr)
        _set.update(names)

        if not has_kw and not toplevel:
            if raiseerr:
                raise TypeError(
                    f"given cls {cls} doesn't have an __init__ method"
                )
            else:
                return None
    else:
        has_kw = False

    if not has_init or has_kw:
        for c in cls.__bases__:
            if get_cls_kwargs(c, _set=_set) is None:
                break

    _set.discard("self")
    return _set


def get_func_kwargs(func: Callable[..., Any]) -> List[str]:
    """Return the set of legal kwargs for the given `func`.

    Uses getargspec so is safe to call for methods, functions,
    etc.

    """

    return compat.inspect_getfullargspec(func)[0]


def get_callable_argspec(
    fn: Callable[..., Any], no_self: bool = False, _is_init: bool = False
) -> compat.FullArgSpec:
    """Return the argument signature for any callable.

    All pure-Python callables are accepted, including
    functions, methods, classes, objects with __call__;
    builtins and other edge cases like functools.partial() objects
    raise a TypeError.

    """
    if inspect.isbuiltin(fn):
        raise TypeError("Can't inspect builtin: %s" % fn)
    elif inspect.isfunction(fn):
        if _is_init and no_self:
            spec = compat.inspect_getfullargspec(fn)
            return compat.FullArgSpec(
                spec.args[1:],
                spec.varargs,
                spec.varkw,
                spec.defaults,
                spec.kwonlyargs,
                spec.kwonlydefaults,
                spec.annotations,
            )
        else:
            return compat.inspect_getfullargspec(fn)
    elif inspect.ismethod(fn):
        if no_self and (_is_init or fn.__self__):
            spec = compat.inspect_getfullargspec(fn.__func__)
            return compat.FullArgSpec(
                spec.args[1:],
                spec.varargs,
                spec.varkw,
                spec.defaults,
                spec.kwonlyargs,
                spec.kwonlydefaults,
                spec.annotations,
            )
        else:
            return compat.inspect_getfullargspec(fn.__func__)
    elif inspect.isclass(fn):
        return get_callable_argspec(
            fn.__init__, no_self=no_self, _is_init=True
        )
    elif hasattr(fn, "__func__"):
        return compat.inspect_getfullargspec(fn.__func__)
    elif hasattr(fn, "__call__"):
        if inspect.ismethod(fn.__call__):
            return get_callable_argspec(fn.__call__, no_self=no_self)
        else:
            raise TypeError("Can't inspect callable: %s" % fn)
    else:
        raise TypeError("Can't inspect callable: %s" % fn)


def format_argspec_plus(
    fn: Union[Callable[..., Any], compat.FullArgSpec], grouped: bool = True
) -> Dict[str, Optional[str]]:
    """Returns a dictionary of formatted, introspected function arguments.

    A enhanced variant of inspect.formatargspec to support code generation.

    fn
       An inspectable callable or tuple of inspect getargspec() results.
    grouped
      Defaults to True; include (parens, around, argument) lists

    Returns:

    args
      Full inspect.formatargspec for fn
    self_arg
      The name of the first positional argument, varargs[0], or None
      if the function defines no positional arguments.
    apply_pos
      args, re-written in calling rather than receiving syntax.  Arguments are
      passed positionally.
    apply_kw
      Like apply_pos, except keyword-ish args are passed as keywords.
    apply_pos_proxied
      Like apply_pos but omits the self/cls argument

    Example::

      >>> format_argspec_plus(lambda self, a, b, c=3, **d: 123)
      {'grouped_args': '(self, a, b, c=3, **d)',
       'self_arg': 'self',
       'apply_kw': '(self, a, b, c=c, **d)',
       'apply_pos': '(self, a, b, c, **d)'}

    """
    if callable(fn):
        spec = compat.inspect_getfullargspec(fn)
    else:
        spec = fn

    args = compat.inspect_formatargspec(*spec)

    apply_pos = compat.inspect_formatargspec(
        spec[0], spec[1], spec[2], None, spec[4]
    )

    if spec[0]:
        self_arg = spec[0][0]

        apply_pos_proxied = compat.inspect_formatargspec(
            spec[0][1:], spec[1], spec[2], None, spec[4]
        )

    elif spec[1]:
        # I'm not sure what this is
        self_arg = "%s[0]" % spec[1]

        apply_pos_proxied = apply_pos
    else:
        self_arg = None
        apply_pos_proxied = apply_pos

    num_defaults = 0
    if spec[3]:
        num_defaults += len(cast(Tuple[Any], spec[3]))
    if spec[4]:
        num_defaults += len(spec[4])

    name_args = spec[0] + spec[4]

    defaulted_vals: Union[List[str], Tuple[()]]

    if num_defaults:
        defaulted_vals = name_args[0 - num_defaults :]
    else:
        defaulted_vals = ()

    apply_kw = compat.inspect_formatargspec(
        name_args,
        spec[1],
        spec[2],
        defaulted_vals,
        formatvalue=lambda x: "=" + str(x),
    )

    if spec[0]:
        apply_kw_proxied = compat.inspect_formatargspec(
            name_args[1:],
            spec[1],
            spec[2],
            defaulted_vals,
            formatvalue=lambda x: "=" + str(x),
        )
    else:
        apply_kw_proxied = apply_kw

    if grouped:
        return dict(
            grouped_args=args,
            self_arg=self_arg,
            apply_pos=apply_pos,
            apply_kw=apply_kw,
            apply_pos_proxied=apply_pos_proxied,
            apply_kw_proxied=apply_kw_proxied,
        )
    else:
        return dict(
            grouped_args=args,
            self_arg=self_arg,
            apply_pos=apply_pos[1:-1],
            apply_kw=apply_kw[1:-1],
            apply_pos_proxied=apply_pos_proxied[1:-1],
            apply_kw_proxied=apply_kw_proxied[1:-1],
        )


def format_argspec_init(method, grouped=True):
    """format_argspec_plus with considerations for typical __init__ methods

    Wraps format_argspec_plus with error handling strategies for typical
    __init__ cases:

    .. sourcecode:: text

      object.__init__ -> (self)
      other unreflectable (usually C) -> (self, *args, **kwargs)

    """
    if method is object.__init__:
        grouped_args = "(self)"
        args = "(self)" if grouped else "self"
        proxied = "()" if grouped else ""
    else:
        try:
            return format_argspec_plus(method, grouped=grouped)
        except TypeError:
            grouped_args = "(self, *args, **kwargs)"
            args = grouped_args if grouped else "self, *args, **kwargs"
            proxied = "(*args, **kwargs)" if grouped else "*args, **kwargs"
    return dict(
        self_arg="self",
        grouped_args=grouped_args,
        apply_pos=args,
        apply_kw=args,
        apply_pos_proxied=proxied,
        apply_kw_proxied=proxied,
    )


def create_proxy_methods(
    target_cls: Type[Any],
    target_cls_sphinx_name: str,
    proxy_cls_sphinx_name: str,
    classmethods: Sequence[str] = (),
    methods: Sequence[str] = (),
    attributes: Sequence[str] = (),
    use_intermediate_variable: Sequence[str] = (),
) -> Callable[[_T], _T]:
    """A class decorator indicating attributes should refer to a proxy
    class.

    This decorator is now a "marker" that does nothing at runtime.  Instead,
    it is consumed by the tools/generate_proxy_methods.py script to
    statically generate proxy methods and attributes that are fully
    recognized by typing tools such as mypy.

    """

    def decorate(cls):
        return cls

    return decorate


def getargspec_init(method):
    """inspect.getargspec with considerations for typical __init__ methods

    Wraps inspect.getargspec with error handling for typical __init__ cases:

    .. sourcecode:: text

      object.__init__ -> (self)
      other unreflectable (usually C) -> (self, *args, **kwargs)

    """
    try:
        return compat.inspect_getfullargspec(method)
    except TypeError:
        if method is object.__init__:
            return (["self"], None, None, None)
        else:
            return (["self"], "args", "kwargs", None)


def unbound_method_to_callable(func_or_cls):
    """Adjust the incoming callable such that a 'self' argument is not
    required.

    """

    if isinstance(func_or_cls, types.MethodType) and not func_or_cls.__self__:
        return func_or_cls.__func__
    else:
        return func_or_cls


def generic_repr(
    obj: Any,
    additional_kw: Sequence[Tuple[str, Any]] = (),
    to_inspect: Optional[Union[object, List[object]]] = None,
    omit_kwarg: Sequence[str] = (),
) -> str:
    """Produce a __repr__() based on direct association of the __init__()
    specification vs. same-named attributes present.

    """
    if to_inspect is None:
        to_inspect = [obj]
    else:
        to_inspect = _collections.to_list(to_inspect)

    missing = object()

    pos_args = []
    kw_args: _collections.OrderedDict[str, Any] = _collections.OrderedDict()
    vargs = None
    for i, insp in enumerate(to_inspect):
        try:
            spec = compat.inspect_getfullargspec(insp.__init__)
        except TypeError:
            continue
        else:
            default_len = len(spec.defaults) if spec.defaults else 0
            if i == 0:
                if spec.varargs:
                    vargs = spec.varargs
                if default_len:
                    pos_args.extend(spec.args[1:-default_len])
                else:
                    pos_args.extend(spec.args[1:])
            else:
                kw_args.update(
                    [(arg, missing) for arg in spec.args[1:-default_len]]
                )

            if default_len:
                assert spec.defaults
                kw_args.update(
                    [
                        (arg, default)
                        for arg, default in zip(
                            spec.args[-default_len:], spec.defaults
                        )
                    ]
                )
    output: List[str] = []

    output.extend(repr(getattr(obj, arg, None)) for arg in pos_args)

    if vargs is not None and hasattr(obj, vargs):
        output.extend([repr(val) for val in getattr(obj, vargs)])

    for arg, defval in kw_args.items():
        if arg in omit_kwarg:
            continue
        try:
            val = getattr(obj, arg, missing)
            if val is not missing and val != defval:
                output.append("%s=%r" % (arg, val))
        except Exception:
            pass

    if additional_kw:
        for arg, defval in additional_kw:
            try:
                val = getattr(obj, arg, missing)
                if val is not missing and val != defval:
                    output.append("%s=%r" % (arg, val))
            except Exception:
                pass

    return "%s(%s)" % (obj.__class__.__name__, ", ".join(output))


class portable_instancemethod:
    """Turn an instancemethod into a (parent, name) pair
    to produce a serializable callable.

    """

    __slots__ = "target", "name", "kwargs", "__weakref__"

    def __getstate__(self):
        return {
            "target": self.target,
            "name": self.name,
            "kwargs": self.kwargs,
        }

    def __setstate__(self, state):
        self.target = state["target"]
        self.name = state["name"]
        self.kwargs = state.get("kwargs", ())

    def __init__(self, meth, kwargs=()):
        self.target = meth.__self__
        self.name = meth.__name__
        self.kwargs = kwargs

    def __call__(self, *arg, **kw):
        kw.update(self.kwargs)
        return getattr(self.target, self.name)(*arg, **kw)


def class_hierarchy(cls):
    """Return an unordered sequence of all classes related to cls.

    Traverses diamond hierarchies.

    Fibs slightly: subclasses of builtin types are not returned.  Thus
    class_hierarchy(class A(object)) returns (A, object), not A plus every
    class systemwide that derives from object.

    """

    hier = {cls}
    process = list(cls.__mro__)
    while process:
        c = process.pop()
        bases = (_ for _ in c.__bases__ if _ not in hier)

        for b in bases:
            process.append(b)
            hier.add(b)

        if c.__module__ == "builtins" or not hasattr(c, "__subclasses__"):
            continue

        for s in [
            _
            for _ in (
                c.__subclasses__()
                if not issubclass(c, type)
                else c.__subclasses__(c)
            )
            if _ not in hier
        ]:
            process.append(s)
            hier.add(s)
    return list(hier)


def iterate_attributes(cls):
    """iterate all the keys and attributes associated
    with a class, without using getattr().

    Does not use getattr() so that class-sensitive
    descriptors (i.e. property.__get__()) are not called.

    """
    keys = dir(cls)
    for key in keys:
        for c in cls.__mro__:
            if key in c.__dict__:
                yield (key, c.__dict__[key])
                break


def monkeypatch_proxied_specials(
    into_cls,
    from_cls,
    skip=None,
    only=None,
    name="self.proxy",
    from_instance=None,
):
    """Automates delegation of __specials__ for a proxying type."""

    if only:
        dunders = only
    else:
        if skip is None:
            skip = (
                "__slots__",
                "__del__",
                "__getattribute__",
                "__metaclass__",
                "__getstate__",
                "__setstate__",
            )
        dunders = [
            m
            for m in dir(from_cls)
            if (
                m.startswith("__")
                and m.endswith("__")
                and not hasattr(into_cls, m)
                and m not in skip
            )
        ]

    for method in dunders:
        try:
            maybe_fn = getattr(from_cls, method)
            if not hasattr(maybe_fn, "__call__"):
                continue
            maybe_fn = getattr(maybe_fn, "__func__", maybe_fn)
            fn = cast(types.FunctionType, maybe_fn)

        except AttributeError:
            continue
        try:
            spec = compat.inspect_getfullargspec(fn)
            fn_args = compat.inspect_formatargspec(spec[0])
            d_args = compat.inspect_formatargspec(spec[0][1:])
        except TypeError:
            fn_args = "(self, *args, **kw)"
            d_args = "(*args, **kw)"

        py = (
            "def %(method)s%(fn_args)s: "
            "return %(name)s.%(method)s%(d_args)s" % locals()
        )

        env: Dict[str, types.FunctionType] = (
            from_instance is not None and {name: from_instance} or {}
        )
        exec(py, env)
        try:
            env[method].__defaults__ = fn.__defaults__
        except AttributeError:
            pass
        setattr(into_cls, method, env[method])


def methods_equivalent(meth1, meth2):
    """Return True if the two methods are the same implementation."""

    return getattr(meth1, "__func__", meth1) is getattr(
        meth2, "__func__", meth2
    )


def as_interface(obj, cls=None, methods=None, required=None):
    """Ensure basic interface compliance for an instance or dict of callables.

    Checks that ``obj`` implements public methods of ``cls`` or has members
    listed in ``methods``. If ``required`` is not supplied, implementing at
    least one interface method is sufficient. Methods present on ``obj`` that
    are not in the interface are ignored.

    If ``obj`` is a dict and ``dict`` does not meet the interface
    requirements, the keys of the dictionary are inspected. Keys present in
    ``obj`` that are not in the interface will raise TypeErrors.

    Raises TypeError if ``obj`` does not meet the interface criteria.

    In all passing cases, an object with callable members is returned.  In the
    simple case, ``obj`` is returned as-is; if dict processing kicks in then
    an anonymous class is returned.

    obj
      A type, instance, or dictionary of callables.
    cls
      Optional, a type.  All public methods of cls are considered the
      interface.  An ``obj`` instance of cls will always pass, ignoring
      ``required``..
    methods
      Optional, a sequence of method names to consider as the interface.
    required
      Optional, a sequence of mandatory implementations. If omitted, an
      ``obj`` that provides at least one interface method is considered
      sufficient.  As a convenience, required may be a type, in which case
      all public methods of the type are required.

    """
    if not cls and not methods:
        raise TypeError("a class or collection of method names are required")

    if isinstance(cls, type) and isinstance(obj, cls):
        return obj

    interface = set(methods or [m for m in dir(cls) if not m.startswith("_")])
    implemented = set(dir(obj))

    complies = operator.ge
    if isinstance(required, type):
        required = interface
    elif not required:
        required = set()
        complies = operator.gt
    else:
        required = set(required)

    if complies(implemented.intersection(interface), required):
        return obj

    # No dict duck typing here.
    if not isinstance(obj, dict):
        qualifier = complies is operator.gt and "any of" or "all of"
        raise TypeError(
            "%r does not implement %s: %s"
            % (obj, qualifier, ", ".join(interface))
        )

    class AnonymousInterface:
        """A callable-holding shell."""

    if cls:
        AnonymousInterface.__name__ = "Anonymous" + cls.__name__
    found = set()

    for method, impl in dictlike_iteritems(obj):
        if method not in interface:
            raise TypeError("%r: unknown in this interface" % method)
        if not callable(impl):
            raise TypeError("%r=%r is not callable" % (method, impl))
        setattr(AnonymousInterface, method, staticmethod(impl))
        found.add(method)

    if complies(found, required):
        return AnonymousInterface

    raise TypeError(
        "dictionary does not contain required keys %s"
        % ", ".join(required - found)
    )


_GFD = TypeVar("_GFD", bound="generic_fn_descriptor[Any]")


class generic_fn_descriptor(Generic[_T_co]):
    """Descriptor which proxies a function when the attribute is not
    present in dict

    This superclass is organized in a particular way with "memoized" and
    "non-memoized" implementation classes that are hidden from type checkers,
    as Mypy seems to not be able to handle seeing multiple kinds of descriptor
    classes used for the same attribute.

    """

    fget: Callable[..., _T_co]
    __doc__: Optional[str]
    __name__: str

    def __init__(self, fget: Callable[..., _T_co], doc: Optional[str] = None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__

    @overload
    def __get__(self: _GFD, obj: None, cls: Any) -> _GFD: ...

    @overload
    def __get__(self, obj: object, cls: Any) -> _T_co: ...

    def __get__(self: _GFD, obj: Any, cls: Any) -> Union[_GFD, _T_co]:
        raise NotImplementedError()

    if TYPE_CHECKING:

        def __set__(self, instance: Any, value: Any) -> None: ...

        def __delete__(self, instance: Any) -> None: ...

    def _reset(self, obj: Any) -> None:
        raise NotImplementedError()

    @classmethod
    def reset(cls, obj: Any, name: str) -> None:
        raise NotImplementedError()


class _non_memoized_property(generic_fn_descriptor[_T_co]):
    """a plain descriptor that proxies a function.

    primary rationale is to provide a plain attribute that's
    compatible with memoized_property which is also recognized as equivalent
    by mypy.

    """

    if not TYPE_CHECKING:

        def __get__(self, obj, cls):
            if obj is None:
                return self
            return self.fget(obj)


class _memoized_property(generic_fn_descriptor[_T_co]):
    """A read-only @property that is only evaluated once."""

    if not TYPE_CHECKING:

        def __get__(self, obj, cls):
            if obj is None:
                return self
            obj.__dict__[self.__name__] = result = self.fget(obj)
            return result

    def _reset(self, obj):
        _memoized_property.reset(obj, self.__name__)

    @classmethod
    def reset(cls, obj, name):
        obj.__dict__.pop(name, None)


# despite many attempts to get Mypy to recognize an overridden descriptor
# where one is memoized and the other isn't, there seems to be no reliable
# way other than completely deceiving the type checker into thinking there
# is just one single descriptor type everywhere.  Otherwise, if a superclass
# has non-memoized and subclass has memoized, that requires
# "class memoized(non_memoized)".  but then if a superclass has memoized and
# superclass has non-memoized, the class hierarchy of the descriptors
# would need to be reversed; "class non_memoized(memoized)".  so there's no
# way to achieve this.
# additional issues, RO properties:
# https://github.com/python/mypy/issues/12440
if TYPE_CHECKING:
    # allow memoized and non-memoized to be freely mixed by having them
    # be the same class
    memoized_property = generic_fn_descriptor
    non_memoized_property = generic_fn_descriptor

    # for read only situations, mypy only sees @property as read only.
    # read only is needed when a subtype specializes the return type
    # of a property, meaning assignment needs to be disallowed
    ro_memoized_property = property
    ro_non_memoized_property = property

else:
    memoized_property = ro_memoized_property = _memoized_property
    non_memoized_property = ro_non_memoized_property = _non_memoized_property


def memoized_instancemethod(fn: _F) -> _F:
    """Decorate a method memoize its return value.

    Best applied to no-arg methods: memoization is not sensitive to
    argument values, and will always return the same value even when
    called with different arguments.

    """

    def oneshot(self, *args, **kw):
        result = fn(self, *args, **kw)

        def memo(*a, **kw):
            return result

        memo.__name__ = fn.__name__
        memo.__doc__ = fn.__doc__
        self.__dict__[fn.__name__] = memo
        return result

    return update_wrapper(oneshot, fn)  # type: ignore


class HasMemoized:
    """A mixin class that maintains the names of memoized elements in a
    collection for easy cache clearing, generative, etc.

    """

    if not TYPE_CHECKING:
        # support classes that want to have __slots__ with an explicit
        # slot for __dict__.  not sure if that requires base __slots__ here.
        __slots__ = ()

    _memoized_keys: FrozenSet[str] = frozenset()

    def _reset_memoizations(self) -> None:
        for elem in self._memoized_keys:
            self.__dict__.pop(elem, None)

    def _assert_no_memoizations(self) -> None:
        for elem in self._memoized_keys:
            assert elem not in self.__dict__

    def _set_memoized_attribute(self, key: str, value: Any) -> None:
        self.__dict__[key] = value
        self._memoized_keys |= {key}

    class memoized_attribute(memoized_property[_T]):
        """A read-only @property that is only evaluated once.

        :meta private:

        """

        fget: Callable[..., _T]
        __doc__: Optional[str]
        __name__: str

        def __init__(self, fget: Callable[..., _T], doc: Optional[str] = None):
            self.fget = fget
            self.__doc__ = doc or fget.__doc__
            self.__name__ = fget.__name__

        @overload
        def __get__(self: _MA, obj: None, cls: Any) -> _MA: ...

        @overload
        def __get__(self, obj: Any, cls: Any) -> _T: ...

        def __get__(self, obj, cls):
            if obj is None:
                return self
            obj.__dict__[self.__name__] = result = self.fget(obj)
            obj._memoized_keys |= {self.__name__}
            return result

    @classmethod
    def memoized_instancemethod(cls, fn: _F) -> _F:
        """Decorate a method memoize its return value.

        :meta private:

        """

        def oneshot(self: Any, *args: Any, **kw: Any) -> Any:
            result = fn(self, *args, **kw)

            def memo(*a, **kw):
                return result

            memo.__name__ = fn.__name__
            memo.__doc__ = fn.__doc__
            self.__dict__[fn.__name__] = memo
            self._memoized_keys |= {fn.__name__}
            return result

        return update_wrapper(oneshot, fn)  # type: ignore


if TYPE_CHECKING:
    HasMemoized_ro_memoized_attribute = property
else:
    HasMemoized_ro_memoized_attribute = HasMemoized.memoized_attribute


class MemoizedSlots:
    """Apply memoized items to an object using a __getattr__ scheme.

    This allows the functionality of memoized_property and
    memoized_instancemethod to be available to a class using __slots__.

    """

    __slots__ = ()

    def _fallback_getattr(self, key):
        raise AttributeError(key)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_memoized_attr_") or key.startswith(
            "_memoized_method_"
        ):
            raise AttributeError(key)
        # to avoid recursion errors when interacting with other __getattr__
        # schemes that refer to this one, when testing for memoized method
        # look at __class__ only rather than going into __getattr__ again.
        elif hasattr(self.__class__, f"_memoized_attr_{key}"):
            value = getattr(self, f"_memoized_attr_{key}")()
            setattr(self, key, value)
            return value
        elif hasattr(self.__class__, f"_memoized_method_{key}"):
            fn = getattr(self, f"_memoized_method_{key}")

            def oneshot(*args, **kw):
                result = fn(*args, **kw)

                def memo(*a, **kw):
                    return result

                memo.__name__ = fn.__name__
                memo.__doc__ = fn.__doc__
                setattr(self, key, memo)
                return result

            oneshot.__doc__ = fn.__doc__
            return oneshot
        else:
            return self._fallback_getattr(key)


# from paste.deploy.converters
def asbool(obj: Any) -> bool:
    if isinstance(obj, str):
        obj = obj.strip().lower()
        if obj in ["true", "yes", "on", "y", "t", "1"]:
            return True
        elif obj in ["false", "no", "off", "n", "f", "0"]:
            return False
        else:
            raise ValueError("String is not true/false: %r" % obj)
    return bool(obj)


def bool_or_str(*text: str) -> Callable[[str], Union[str, bool]]:
    """Return a callable that will evaluate a string as
    boolean, or one of a set of "alternate" string values.

    """

    def bool_or_value(obj: str) -> Union[str, bool]:
        if obj in text:
            return obj
        else:
            return asbool(obj)

    return bool_or_value


def asint(value: Any) -> Optional[int]:
    """Coerce to integer."""

    if value is None:
        return value
    return int(value)


def coerce_kw_type(
    kw: Dict[str, Any],
    key: str,
    type_: Type[Any],
    flexi_bool: bool = True,
    dest: Optional[Dict[str, Any]] = None,
) -> None:
    r"""If 'key' is present in dict 'kw', coerce its value to type 'type\_' if
    necessary.  If 'flexi_bool' is True, the string '0' is considered false
    when coercing to boolean.
    """

    if dest is None:
        dest = kw

    if (
        key in kw
        and (not isinstance(type_, type) or not isinstance(kw[key], type_))
        and kw[key] is not None
    ):
        if type_ is bool and flexi_bool:
            dest[key] = asbool(kw[key])
        else:
            dest[key] = type_(kw[key])


def constructor_key(obj: Any, cls: Type[Any]) -> Tuple[Any, ...]:
    """Produce a tuple structure that is cacheable using the __dict__ of
    obj to retrieve values

    """
    names = get_cls_kwargs(cls)
    return (cls,) + tuple(
        (k, obj.__dict__[k]) for k in names if k in obj.__dict__
    )


def constructor_copy(obj: _T, cls: Type[_T], *args: Any, **kw: Any) -> _T:
    """Instantiate cls using the __dict__ of obj as constructor arguments.

    Uses inspect to match the named arguments of ``cls``.

    """

    names = get_cls_kwargs(cls)
    kw.update(
        (k, obj.__dict__[k]) for k in names.difference(kw) if k in obj.__dict__
    )
    return cls(*args, **kw)


def counter() -> Callable[[], int]:
    """Return a threadsafe counter function."""

    lock = threading.Lock()
    counter = itertools.count(1)

    # avoid the 2to3 "next" transformation...
    def _next():
        with lock:
            return next(counter)

    return _next


def duck_type_collection(
    specimen: Any, default: Optional[Type[Any]] = None
) -> Optional[Type[Any]]:
    """Given an instance or class, guess if it is or is acting as one of
    the basic collection types: list, set and dict.  If the __emulates__
    property is present, return that preferentially.
    """

    if hasattr(specimen, "__emulates__"):
        # canonicalize set vs sets.Set to a standard: the builtin set
        if specimen.__emulates__ is not None and issubclass(
            specimen.__emulates__, set
        ):
            return set
        else:
            return specimen.__emulates__  # type: ignore

    isa = issubclass if isinstance(specimen, type) else isinstance
    if isa(specimen, list):
        return list
    elif isa(specimen, set):
        return set
    elif isa(specimen, dict):
        return dict

    if hasattr(specimen, "append"):
        return list
    elif hasattr(specimen, "add"):
        return set
    elif hasattr(specimen, "set"):
        return dict
    else:
        return default


def assert_arg_type(
    arg: Any, argtype: Union[Tuple[Type[Any], ...], Type[Any]], name: str
) -> Any:
    if isinstance(arg, argtype):
        return arg
    else:
        if isinstance(argtype, tuple):
            raise exc.ArgumentError(
                "Argument '%s' is expected to be one of type %s, got '%s'"
                % (name, " or ".join("'%s'" % a for a in argtype), type(arg))
            )
        else:
            raise exc.ArgumentError(
                "Argument '%s' is expected to be of type '%s', got '%s'"
                % (name, argtype, type(arg))
            )


def dictlike_iteritems(dictlike):
    """Return a (key, value) iterator for almost any dict-like object."""

    if hasattr(dictlike, "items"):
        return list(dictlike.items())

    getter = getattr(dictlike, "__getitem__", getattr(dictlike, "get", None))
    if getter is None:
        raise TypeError("Object '%r' is not dict-like" % dictlike)

    if hasattr(dictlike, "iterkeys"):

        def iterator():
            for key in dictlike.iterkeys():
                assert getter is not None
                yield key, getter(key)

        return iterator()
    elif hasattr(dictlike, "keys"):
        return iter((key, getter(key)) for key in dictlike.keys())
    else:
        raise TypeError("Object '%r' is not dict-like" % dictlike)


class classproperty(property):
    """A decorator that behaves like @property except that operates
    on classes rather than instances.

    The decorator is currently special when using the declarative
    module, but note that the
    :class:`~.sqlalchemy.ext.declarative.declared_attr`
    decorator should be used for this purpose with declarative.

    """

    fget: Callable[[Any], Any]

    def __init__(self, fget: Callable[[Any], Any], *arg: Any, **kw: Any):
        super().__init__(fget, *arg, **kw)
        self.__doc__ = fget.__doc__

    def __get__(self, obj: Any, cls: Optional[type] = None) -> Any:
        return self.fget(cls)


class hybridproperty(Generic[_T]):
    def __init__(self, func: Callable[..., _T]):
        self.func = func
        self.clslevel = func

    def __get__(self, instance: Any, owner: Any) -> _T:
        if instance is None:
            clsval = self.clslevel(owner)
            return clsval
        else:
            return self.func(instance)

    def classlevel(self, func: Callable[..., Any]) -> hybridproperty[_T]:
        self.clslevel = func
        return self


class rw_hybridproperty(Generic[_T]):
    def __init__(self, func: Callable[..., _T]):
        self.func = func
        self.clslevel = func
        self.setfn: Optional[Callable[..., Any]] = None

    def __get__(self, instance: Any, owner: Any) -> _T:
        if instance is None:
            clsval = self.clslevel(owner)
            return clsval
        else:
            return self.func(instance)

    def __set__(self, instance: Any, value: Any) -> None:
        assert self.setfn is not None
        self.setfn(instance, value)

    def setter(self, func: Callable[..., Any]) -> rw_hybridproperty[_T]:
        self.setfn = func
        return self

    def classlevel(self, func: Callable[..., Any]) -> rw_hybridproperty[_T]:
        self.clslevel = func
        return self


class hybridmethod(Generic[_T]):
    """Decorate a function as cls- or instance- level."""

    def __init__(self, func: Callable[..., _T]):
        self.func = self.__func__ = func
        self.clslevel = func

    def __get__(self, instance: Any, owner: Any) -> Callable[..., _T]:
        if instance is None:
            return self.clslevel.__get__(owner, owner.__class__)  # type:ignore
        else:
            return self.func.__get__(instance, owner)  # type:ignore

    def classlevel(self, func: Callable[..., Any]) -> hybridmethod[_T]:
        self.clslevel = func
        return self


class symbol(int):
    """A constant symbol.

    >>> symbol("foo") is symbol("foo")
    True
    >>> symbol("foo")
    <symbol 'foo>

    A slight refinement of the MAGICCOOKIE=object() pattern.  The primary
    advantage of symbol() is its repr().  They are also singletons.

    Repeated calls of symbol('name') will all return the same instance.

    """

    name: str

    symbols: Dict[str, symbol] = {}
    _lock = threading.Lock()

    def __new__(
        cls,
        name: str,
        doc: Optional[str] = None,
        canonical: Optional[int] = None,
    ) -> symbol:
        with cls._lock:
            sym = cls.symbols.get(name)
            if sym is None:
                assert isinstance(name, str)
                if canonical is None:
                    canonical = hash(name)
                sym = int.__new__(symbol, canonical)
                sym.name = name
                if doc:
                    sym.__doc__ = doc

                # NOTE: we should ultimately get rid of this global thing,
                # however, currently it is to support pickling.  The best
                # change would be when we are on py3.11 at a minimum, we
                # switch to stdlib enum.IntFlag.
                cls.symbols[name] = sym
            else:
                if canonical and canonical != sym:
                    raise TypeError(
                        f"Can't replace canonical symbol for {name!r} "
                        f"with new int value {canonical}"
                    )
            return sym

    def __reduce__(self):
        return symbol, (self.name, "x", int(self))

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"symbol({self.name!r})"


class _IntFlagMeta(type):
    def __init__(
        cls,
        classname: str,
        bases: Tuple[Type[Any], ...],
        dict_: Dict[str, Any],
        **kw: Any,
    ) -> None:
        items: List[symbol]
        cls._items = items = []
        for k, v in dict_.items():
            if re.match(r"^__.*__$", k):
                continue
            if isinstance(v, int):
                sym = symbol(k, canonical=v)
            elif not k.startswith("_"):
                raise TypeError("Expected integer values for IntFlag")
            else:
                continue
            setattr(cls, k, sym)
            items.append(sym)

        cls.__members__ = _collections.immutabledict(
            {sym.name: sym for sym in items}
        )

    def __iter__(self) -> Iterator[symbol]:
        raise NotImplementedError(
            "iter not implemented to ensure compatibility with "
            "Python 3.11 IntFlag.  Please use __members__.  See "
            "https://github.com/python/cpython/issues/99304"
        )


class _FastIntFlag(metaclass=_IntFlagMeta):
    """An 'IntFlag' copycat that isn't slow when performing bitwise
    operations.

    the ``FastIntFlag`` class will return ``enum.IntFlag`` under TYPE_CHECKING
    and ``_FastIntFlag`` otherwise.

    """


if TYPE_CHECKING:
    from enum import IntFlag

    FastIntFlag = IntFlag
else:
    FastIntFlag = _FastIntFlag


_E = TypeVar("_E", bound=enum.Enum)


def parse_user_argument_for_enum(
    arg: Any,
    choices: Dict[_E, List[Any]],
    name: str,
    resolve_symbol_names: bool = False,
) -> Optional[_E]:
    """Given a user parameter, parse the parameter into a chosen value
    from a list of choice objects, typically Enum values.

    The user argument can be a string name that matches the name of a
    symbol, or the symbol object itself, or any number of alternate choices
    such as True/False/ None etc.

    :param arg: the user argument.
    :param choices: dictionary of enum values to lists of possible
        entries for each.
    :param name: name of the argument.   Used in an :class:`.ArgumentError`
        that is raised if the parameter doesn't match any available argument.

    """
    for enum_value, choice in choices.items():
        if arg is enum_value:
            return enum_value
        elif resolve_symbol_names and arg == enum_value.name:
            return enum_value
        elif arg in choice:
            return enum_value

    if arg is None:
        return None

    raise exc.ArgumentError(f"Invalid value for '{name}': {arg!r}")


_creation_order = 1


def set_creation_order(instance: Any) -> None:
    """Assign a '_creation_order' sequence to the given instance.

    This allows multiple instances to be sorted in order of creation
    (typically within a single thread; the counter is not particularly
    threadsafe).

    """
    global _creation_order
    instance._creation_order = _creation_order
    _creation_order += 1


def warn_exception(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """executes the given function, catches all exceptions and converts to
    a warning.

    """
    try:
        return func(*args, **kwargs)
    except Exception:
        warn("%s('%s') ignored" % sys.exc_info()[0:2])


def ellipses_string(value, len_=25):
    try:
        if len(value) > len_:
            return "%s..." % value[0:len_]
        else:
            return value
    except TypeError:
        return value


class _hash_limit_string(str):
    """A string subclass that can only be hashed on a maximum amount
    of unique values.

    This is used for warnings so that we can send out parameterized warnings
    without the __warningregistry__ of the module,  or the non-overridable
    "once" registry within warnings.py, overloading memory,


    """

    _hash: int

    def __new__(
        cls, value: str, num: int, args: Sequence[Any]
    ) -> _hash_limit_string:
        interpolated = (value % args) + (
            " (this warning may be suppressed after %d occurrences)" % num
        )
        self = super().__new__(cls, interpolated)
        self._hash = hash("%s_%d" % (value, hash(interpolated) % num))
        return self

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)


def warn(msg: str, code: Optional[str] = None) -> None:
    """Issue a warning.

    If msg is a string, :class:`.exc.SAWarning` is used as
    the category.

    """
    if code:
        _warnings_warn(exc.SAWarning(msg, code=code))
    else:
        _warnings_warn(msg, exc.SAWarning)


def warn_limited(msg: str, args: Sequence[Any]) -> None:
    """Issue a warning with a parameterized string, limiting the number
    of registrations.

    """
    if args:
        msg = _hash_limit_string(msg, 10, args)
    _warnings_warn(msg, exc.SAWarning)


_warning_tags: Dict[CodeType, Tuple[str, Type[Warning]]] = {}


def tag_method_for_warnings(
    message: str, category: Type[Warning]
) -> Callable[[_F], _F]:
    def go(fn):
        _warning_tags[fn.__code__] = (message, category)
        return fn

    return go


_not_sa_pattern = re.compile(r"^(?:sqlalchemy\.(?!testing)|alembic\.)")


def _warnings_warn(
    message: Union[str, Warning],
    category: Optional[Type[Warning]] = None,
    stacklevel: int = 2,
) -> None:
    # adjust the given stacklevel to be outside of SQLAlchemy
    try:
        frame = sys._getframe(stacklevel)
    except ValueError:
        # being called from less than 3 (or given) stacklevels, weird,
        # but don't crash
        stacklevel = 0
    except:
        # _getframe() doesn't work, weird interpreter issue, weird,
        # ok, but don't crash
        stacklevel = 0
    else:
        stacklevel_found = warning_tag_found = False
        while frame is not None:
            # using __name__ here requires that we have __name__ in the
            # __globals__ of the decorated string functions we make also.
            # we generate this using {"__name__": fn.__module__}
            if not stacklevel_found and not re.match(
                _not_sa_pattern, frame.f_globals.get("__name__", "")
            ):
                # stop incrementing stack level if an out-of-SQLA line
                # were found.
                stacklevel_found = True

                # however, for the warning tag thing, we have to keep
                # scanning up the whole traceback

            if frame.f_code in _warning_tags:
                warning_tag_found = True
                (_suffix, _category) = _warning_tags[frame.f_code]
                category = category or _category
                message = f"{message} ({_suffix})"

            frame = frame.f_back  # type: ignore[assignment]

            if not stacklevel_found:
                stacklevel += 1
            elif stacklevel_found and warning_tag_found:
                break

    if category is not None:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
    else:
        warnings.warn(message, stacklevel=stacklevel + 1)


def only_once(
    fn: Callable[..., _T], retry_on_exception: bool
) -> Callable[..., Optional[_T]]:
    """Decorate the given function to be a no-op after it is called exactly
    once."""

    once = [fn]

    def go(*arg: Any, **kw: Any) -> Optional[_T]:
        # strong reference fn so that it isn't garbage collected,
        # which interferes with the event system's expectations
        strong_fn = fn  # noqa
        if once:
            once_fn = once.pop()
            try:
                return once_fn(*arg, **kw)
            except:
                if retry_on_exception:
                    once.insert(0, once_fn)
                raise

        return None

    return go


_SQLA_RE = re.compile(r"sqlalchemy/([a-z_]+/){0,2}[a-z_]+\.py")
_UNITTEST_RE = re.compile(r"unit(?:2|test2?/)")


def chop_traceback(
    tb: List[str],
    exclude_prefix: re.Pattern[str] = _UNITTEST_RE,
    exclude_suffix: re.Pattern[str] = _SQLA_RE,
) -> List[str]:
    """Chop extraneous lines off beginning and end of a traceback.

    :param tb:
      a list of traceback lines as returned by ``traceback.format_stack()``

    :param exclude_prefix:
      a regular expression object matching lines to skip at beginning of
      ``tb``

    :param exclude_suffix:
      a regular expression object matching lines to skip at end of ``tb``
    """
    start = 0
    end = len(tb) - 1
    while start <= end and exclude_prefix.search(tb[start]):
        start += 1
    while start <= end and exclude_suffix.search(tb[end]):
        end -= 1
    return tb[start : end + 1]


NoneType = type(None)


def attrsetter(attrname):
    code = "def set(obj, value):    obj.%s = value" % attrname
    env = locals().copy()
    exec(code, env)
    return env["set"]


_dunders = re.compile("^__.+__$")


class TypingOnly:
    """A mixin class that marks a class as 'typing only', meaning it has
    absolutely no methods, attributes, or runtime functionality whatsoever.

    """

    __slots__ = ()

    def __init_subclass__(cls) -> None:
        if TypingOnly in cls.__bases__:
            remaining = {
                name for name in cls.__dict__ if not _dunders.match(name)
            }
            if remaining:
                raise AssertionError(
                    f"Class {cls} directly inherits TypingOnly but has "
                    f"additional attributes {remaining}."
                )
        super().__init_subclass__()


class EnsureKWArg:
    r"""Apply translation of functions to accept \**kw arguments if they
    don't already.

    Used to ensure cross-compatibility with third party legacy code, for things
    like compiler visit methods that need to accept ``**kw`` arguments,
    but may have been copied from old code that didn't accept them.

    """

    ensure_kwarg: str
    """a regular expression that indicates method names for which the method
    should accept ``**kw`` arguments.

    The class will scan for methods matching the name template and decorate
    them if necessary to ensure ``**kw`` parameters are accepted.

    """

    def __init_subclass__(cls) -> None:
        fn_reg = cls.ensure_kwarg
        clsdict = cls.__dict__
        if fn_reg:
            for key in clsdict:
                m = re.match(fn_reg, key)
                if m:
                    fn = clsdict[key]
                    spec = compat.inspect_getfullargspec(fn)
                    if not spec.varkw:
                        wrapped = cls._wrap_w_kw(fn)
                        setattr(cls, key, wrapped)
        super().__init_subclass__()

    @classmethod
    def _wrap_w_kw(cls, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrap(*arg: Any, **kw: Any) -> Any:
            return fn(*arg)

        return update_wrapper(wrap, fn)


def wrap_callable(wrapper, fn):
    """Augment functools.update_wrapper() to work with objects with
    a ``__call__()`` method.

    :param fn:
      object with __call__ method

    """
    if hasattr(fn, "__name__"):
        return update_wrapper(wrapper, fn)
    else:
        _f = wrapper
        _f.__name__ = fn.__class__.__name__
        if hasattr(fn, "__module__"):
            _f.__module__ = fn.__module__

        if hasattr(fn.__call__, "__doc__") and fn.__call__.__doc__:
            _f.__doc__ = fn.__call__.__doc__
        elif fn.__doc__:
            _f.__doc__ = fn.__doc__

        return _f


def quoted_token_parser(value):
    """Parse a dotted identifier with accommodation for quoted names.

    Includes support for SQL-style double quotes as a literal character.

    E.g.::

        >>> quoted_token_parser("name")
        ["name"]
        >>> quoted_token_parser("schema.name")
        ["schema", "name"]
        >>> quoted_token_parser('"Schema"."Name"')
        ['Schema', 'Name']
        >>> quoted_token_parser('"Schema"."Name""Foo"')
        ['Schema', 'Name""Foo']

    """

    if '"' not in value:
        return value.split(".")

    # 0 = outside of quotes
    # 1 = inside of quotes
    state = 0
    result: List[List[str]] = [[]]
    idx = 0
    lv = len(value)
    while idx < lv:
        char = value[idx]
        if char == '"':
            if state == 1 and idx < lv - 1 and value[idx + 1] == '"':
                result[-1].append('"')
                idx += 1
            else:
                state ^= 1
        elif char == "." and state == 0:
            result.append([])
        else:
            result[-1].append(char)
        idx += 1

    return ["".join(token) for token in result]


def add_parameter_text(params: Any, text: str) -> Callable[[_F], _F]:
    params = _collections.to_list(params)

    def decorate(fn):
        doc = fn.__doc__ is not None and fn.__doc__ or ""
        if doc:
            doc = inject_param_text(doc, {param: text for param in params})
        fn.__doc__ = doc
        return fn

    return decorate


def _dedent_docstring(text: str) -> str:
    split_text = text.split("\n", 1)
    if len(split_text) == 1:
        return text
    else:
        firstline, remaining = split_text
    if not firstline.startswith(" "):
        return firstline + "\n" + textwrap.dedent(remaining)
    else:
        return textwrap.dedent(text)


def inject_docstring_text(
    given_doctext: Optional[str], injecttext: str, pos: int
) -> str:
    doctext: str = _dedent_docstring(given_doctext or "")
    lines = doctext.split("\n")
    if len(lines) == 1:
        lines.append("")
    injectlines = textwrap.dedent(injecttext).split("\n")
    if injectlines[0]:
        injectlines.insert(0, "")

    blanks = [num for num, line in enumerate(lines) if not line.strip()]
    blanks.insert(0, 0)

    inject_pos = blanks[min(pos, len(blanks) - 1)]

    lines = lines[0:inject_pos] + injectlines + lines[inject_pos:]
    return "\n".join(lines)


_param_reg = re.compile(r"(\s+):param (.+?):")


def inject_param_text(doctext: str, inject_params: Dict[str, str]) -> str:
    doclines = collections.deque(doctext.splitlines())
    lines = []

    # TODO: this is not working for params like ":param case_sensitive=True:"

    to_inject = None
    while doclines:
        line = doclines.popleft()

        m = _param_reg.match(line)

        if to_inject is None:
            if m:
                param = m.group(2).lstrip("*")
                if param in inject_params:
                    # default indent to that of :param: plus one
                    indent = " " * len(m.group(1)) + " "

                    # but if the next line has text, use that line's
                    # indentation
                    if doclines:
                        m2 = re.match(r"(\s+)\S", doclines[0])
                        if m2:
                            indent = " " * len(m2.group(1))

                    to_inject = indent + inject_params[param]
        elif m:
            lines.extend(["\n", to_inject, "\n"])
            to_inject = None
        elif not line.rstrip():
            lines.extend([line, to_inject, "\n"])
            to_inject = None
        elif line.endswith("::"):
            # TODO: this still won't cover if the code example itself has
            # blank lines in it, need to detect those via indentation.
            lines.extend([line, doclines.popleft()])
            continue
        lines.append(line)

    return "\n".join(lines)


def repr_tuple_names(names: List[str]) -> Optional[str]:
    """Trims a list of strings from the middle and return a string of up to
    four elements. Strings greater than 11 characters will be truncated"""
    if len(names) == 0:
        return None
    flag = len(names) <= 4
    names = names[0:4] if flag else names[0:3] + names[-1:]
    res = ["%s.." % name[:11] if len(name) > 11 else name for name in names]
    if flag:
        return ", ".join(res)
    else:
        return "%s, ..., %s" % (", ".join(res[0:3]), res[-1])


def has_compiled_ext(raise_=False):
    if HAS_CYEXTENSION:
        return True
    elif raise_:
        raise ImportError(
            "cython extensions were expected to be installed, "
            "but are not present"
        )
    else:
        return False


class _Missing(enum.Enum):
    Missing = enum.auto()


Missing = _Missing.Missing
MissingOr = Union[_T, Literal[_Missing.Missing]]