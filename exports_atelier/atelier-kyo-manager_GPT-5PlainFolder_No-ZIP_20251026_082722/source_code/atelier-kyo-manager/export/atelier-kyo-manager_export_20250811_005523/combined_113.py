
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\delta_functions.py ===
from sympy.core import S, diff
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.complexes import im, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.utilities.misc import filldedent


###############################################################################
################################ DELTA FUNCTION ###############################
###############################################################################


class DiracDelta(DefinedFunction):
    r"""
    The DiracDelta function and its derivatives.

    Explanation
    ===========

    DiracDelta is not an ordinary function. It can be rigorously defined either
    as a distribution or as a measure.

    DiracDelta only makes sense in definite integrals, and in particular,
    integrals of the form ``Integral(f(x)*DiracDelta(x - x0), (x, a, b))``,
    where it equals ``f(x0)`` if ``a <= x0 <= b`` and ``0`` otherwise. Formally,
    DiracDelta acts in some ways like a function that is ``0`` everywhere except
    at ``0``, but in many ways it also does not. It can often be useful to treat
    DiracDelta in formal ways, building up and manipulating expressions with
    delta functions (which may eventually be integrated), but care must be taken
    to not treat it as a real function. SymPy's ``oo`` is similar. It only
    truly makes sense formally in certain contexts (such as integration limits),
    but SymPy allows its use everywhere, and it tries to be consistent with
    operations on it (like ``1/oo``), but it is easy to get into trouble and get
    wrong results if ``oo`` is treated too much like a number. Similarly, if
    DiracDelta is treated too much like a function, it is easy to get wrong or
    nonsensical results.

    DiracDelta function has the following properties:

    1) $\frac{d}{d x} \theta(x) = \delta(x)$
    2) $\int_{-\infty}^\infty \delta(x - a)f(x)\, dx = f(a)$ and $\int_{a-
       \epsilon}^{a+\epsilon} \delta(x - a)f(x)\, dx = f(a)$
    3) $\delta(x) = 0$ for all $x \neq 0$
    4) $\delta(g(x)) = \sum_i \frac{\delta(x - x_i)}{\|g'(x_i)\|}$ where $x_i$
       are the roots of $g$
    5) $\delta(-x) = \delta(x)$

    Derivatives of ``k``-th order of DiracDelta have the following properties:

    6) $\delta(x, k) = 0$ for all $x \neq 0$
    7) $\delta(-x, k) = -\delta(x, k)$ for odd $k$
    8) $\delta(-x, k) = \delta(x, k)$ for even $k$

    Examples
    ========

    >>> from sympy import DiracDelta, diff, pi
    >>> from sympy.abc import x, y

    >>> DiracDelta(x)
    DiracDelta(x)
    >>> DiracDelta(1)
    0
    >>> DiracDelta(-1)
    0
    >>> DiracDelta(pi)
    0
    >>> DiracDelta(x - 4).subs(x, 4)
    DiracDelta(0)
    >>> diff(DiracDelta(x))
    DiracDelta(x, 1)
    >>> diff(DiracDelta(x - 1), x, 2)
    DiracDelta(x - 1, 2)
    >>> diff(DiracDelta(x**2 - 1), x, 2)
    2*(2*x**2*DiracDelta(x**2 - 1, 2) + DiracDelta(x**2 - 1, 1))
    >>> DiracDelta(3*x).is_simple(x)
    True
    >>> DiracDelta(x**2).is_simple(x)
    False
    >>> DiracDelta((x**2 - 1)*y).expand(diracdelta=True, wrt=x)
    DiracDelta(x - 1)/(2*Abs(y)) + DiracDelta(x + 1)/(2*Abs(y))

    See Also
    ========

    Heaviside
    sympy.simplify.simplify.simplify, is_simple
    sympy.functions.special.tensor_functions.KroneckerDelta

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DeltaFunction.html

    """

    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a DiracDelta Function.

        Explanation
        ===========

        The difference between ``diff()`` and ``fdiff()`` is: ``diff()`` is the
        user-level function and ``fdiff()`` is an object method. ``fdiff()`` is
        a convenience method available in the ``Function`` class. It returns
        the derivative of the function without considering the chain rule.
        ``diff(function, x)`` calls ``Function._eval_derivative`` which in turn
        calls ``fdiff()`` internally to compute the derivative of the function.

        Examples
        ========

        >>> from sympy import DiracDelta, diff
        >>> from sympy.abc import x

        >>> DiracDelta(x).fdiff()
        DiracDelta(x, 1)

        >>> DiracDelta(x, 1).fdiff()
        DiracDelta(x, 2)

        >>> DiracDelta(x**2 - 1).fdiff()
        DiracDelta(x**2 - 1, 1)

        >>> diff(DiracDelta(x, 1)).fdiff()
        DiracDelta(x, 3)

        Parameters
        ==========

        argindex : integer
            degree of derivative

        """
        if argindex == 1:
            #I didn't know if there is a better way to handle default arguments
            k = 0
            if len(self.args) > 1:
                k = self.args[1]
            return self.func(self.args[0], k + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg, k=S.Zero):
        """
        Returns a simplified form or a value of DiracDelta depending on the
        argument passed by the DiracDelta object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the ``DiracDelta``
        class is about to be instantiated and it returns either some simplified
        instance or the unevaluated instance depending on the argument passed.
        In other words, ``eval()`` method is not needed to be called explicitly,
        it is being called and evaluated once the object is called.

        Examples
        ========

        >>> from sympy import DiracDelta, S
        >>> from sympy.abc import x

        >>> DiracDelta(x)
        DiracDelta(x)

        >>> DiracDelta(-x, 1)
        -DiracDelta(x, 1)

        >>> DiracDelta(1)
        0

        >>> DiracDelta(5, 1)
        0

        >>> DiracDelta(0)
        DiracDelta(0)

        >>> DiracDelta(-1)
        0

        >>> DiracDelta(S.NaN)
        nan

        >>> DiracDelta(x - 100).subs(x, 5)
        0

        >>> DiracDelta(x - 100).subs(x, 100)
        DiracDelta(0)

        Parameters
        ==========

        k : integer
            order of derivative

        arg : argument passed to DiracDelta

        """
        if not k.is_Integer or k.is_negative:
            raise ValueError("Error: the second argument of DiracDelta must be \
            a non-negative integer, %s given instead." % (k,))
        if arg is S.NaN:
            return S.NaN
        if arg.is_nonzero:
            return S.Zero
        if fuzzy_not(im(arg).is_zero):
            raise ValueError(filldedent('''
                Function defined only for Real Values.
                Complex part: %s  found in %s .''' % (
                repr(im(arg)), repr(arg))))
        c, nc = arg.args_cnc()
        if c and c[0] is S.NegativeOne:
            # keep this fast and simple instead of using
            # could_extract_minus_sign
            if k.is_odd:
                return -cls(-arg, k)
            elif k.is_even:
                return cls(-arg, k) if k else cls(-arg)
        elif k.is_zero:
            return cls(arg, evaluate=False)

    def _eval_expand_diracdelta(self, **hints):
        """
        Compute a simplified representation of the function using
        property number 4. Pass ``wrt`` as a hint to expand the expression
        with respect to a particular variable.

        Explanation
        ===========

        ``wrt`` is:

        - a variable with respect to which a DiracDelta expression will
        get expanded.

        Examples
        ========

        >>> from sympy import DiracDelta
        >>> from sympy.abc import x, y

        >>> DiracDelta(x*y).expand(diracdelta=True, wrt=x)
        DiracDelta(x)/Abs(y)
        >>> DiracDelta(x*y).expand(diracdelta=True, wrt=y)
        DiracDelta(y)/Abs(x)

        >>> DiracDelta(x**2 + x - 2).expand(diracdelta=True, wrt=x)
        DiracDelta(x - 1)/3 + DiracDelta(x + 2)/3

        See Also
        ========

        is_simple, Diracdelta

        """
        wrt = hints.get('wrt', None)
        if wrt is None:
            free = self.free_symbols
            if len(free) == 1:
                wrt = free.pop()
            else:
                raise TypeError(filldedent('''
            When there is more than 1 free symbol or variable in the expression,
            the 'wrt' keyword is required as a hint to expand when using the
            DiracDelta hint.'''))

        if not self.args[0].has(wrt) or (len(self.args) > 1 and self.args[1] != 0 ):
            return self
        try:
            argroots = roots(self.args[0], wrt)
            result = 0
            valid = True
            darg = abs(diff(self.args[0], wrt))
            for r, m in argroots.items():
                if r.is_real is not False and m == 1:
                    result += self.func(wrt - r)/darg.subs(wrt, r)
                else:
                    # don't handle non-real and if m != 1 then
                    # a polynomial will have a zero in the derivative (darg)
                    # at r
                    valid = False
                    break
            if valid:
                return result
        except PolynomialError:
            pass
        return self

    def is_simple(self, x):
        """
        Tells whether the argument(args[0]) of DiracDelta is a linear
        expression in *x*.

        Examples
        ========

        >>> from sympy import DiracDelta, cos
        >>> from sympy.abc import x, y

        >>> DiracDelta(x*y).is_simple(x)
        True
        >>> DiracDelta(x*y).is_simple(y)
        True

        >>> DiracDelta(x**2 + x - 2).is_simple(x)
        False

        >>> DiracDelta(cos(x)).is_simple(x)
        False

        Parameters
        ==========

        x : can be a symbol

        See Also
        ========

        sympy.simplify.simplify.simplify, DiracDelta

        """
        p = self.args[0].as_poly(x)
        if p:
            return p.degree() == 1
        return False

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        """
        Represents DiracDelta in a piecewise form.

        Examples
        ========

        >>> from sympy import DiracDelta, Piecewise, Symbol
        >>> x = Symbol('x')

        >>> DiracDelta(x).rewrite(Piecewise)
        Piecewise((DiracDelta(0), Eq(x, 0)), (0, True))

        >>> DiracDelta(x - 5).rewrite(Piecewise)
        Piecewise((DiracDelta(0), Eq(x, 5)), (0, True))

        >>> DiracDelta(x**2 - 5).rewrite(Piecewise)
           Piecewise((DiracDelta(0), Eq(x**2, 5)), (0, True))

        >>> DiracDelta(x - 5, 4).rewrite(Piecewise)
        DiracDelta(x - 5, 4)

        """
        if len(args) == 1:
            return Piecewise((DiracDelta(0), Eq(args[0], 0)), (0, True))

    def _eval_rewrite_as_SingularityFunction(self, *args, **kwargs):
        """
        Returns the DiracDelta expression written in the form of Singularity
        Functions.

        """
        from sympy.solvers import solve
        from sympy.functions.special.singularity_functions import SingularityFunction
        if self == DiracDelta(0):
            return SingularityFunction(0, 0, -1)
        if self == DiracDelta(0, 1):
            return SingularityFunction(0, 0, -2)
        free = self.free_symbols
        if len(free) == 1:
            x = (free.pop())
            if len(args) == 1:
                return SingularityFunction(x, solve(args[0], x)[0], -1)
            return SingularityFunction(x, solve(args[0], x)[0], -args[1] - 1)
        else:
            # I don't know how to handle the case for DiracDelta expressions
            # having arguments with more than one variable.
            raise TypeError(filldedent('''
                rewrite(SingularityFunction) does not support
                arguments with more that one variable.'''))


###############################################################################
############################## HEAVISIDE FUNCTION #############################
###############################################################################


class Heaviside(DefinedFunction):
    r"""
    Heaviside step function.

    Explanation
    ===========

    The Heaviside step function has the following properties:

    1) $\frac{d}{d x} \theta(x) = \delta(x)$
    2) $\theta(x) = \begin{cases} 0 & \text{for}\: x < 0 \\ \frac{1}{2} &
       \text{for}\: x = 0 \\1 & \text{for}\: x > 0 \end{cases}$
    3) $\frac{d}{d x} \max(x, 0) = \theta(x)$

    Heaviside(x) is printed as $\theta(x)$ with the SymPy LaTeX printer.

    The value at 0 is set differently in different fields. SymPy uses 1/2,
    which is a convention from electronics and signal processing, and is
    consistent with solving improper integrals by Fourier transform and
    convolution.

    To specify a different value of Heaviside at ``x=0``, a second argument
    can be given. Using ``Heaviside(x, nan)`` gives an expression that will
    evaluate to nan for x=0.

    .. versionchanged:: 1.9 ``Heaviside(0)`` now returns 1/2 (before: undefined)

    Examples
    ========

    >>> from sympy import Heaviside, nan
    >>> from sympy.abc import x
    >>> Heaviside(9)
    1
    >>> Heaviside(-9)
    0
    >>> Heaviside(0)
    1/2
    >>> Heaviside(0, nan)
    nan
    >>> (Heaviside(x) + 1).replace(Heaviside(x), Heaviside(x, 1))
    Heaviside(x, 1) + 1

    See Also
    ========

    DiracDelta

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HeavisideStepFunction.html
    .. [2] https://dlmf.nist.gov/1.16#iv

    """

    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a Heaviside Function.

        Examples
        ========

        >>> from sympy import Heaviside, diff
        >>> from sympy.abc import x

        >>> Heaviside(x).fdiff()
        DiracDelta(x)

        >>> Heaviside(x**2 - 1).fdiff()
        DiracDelta(x**2 - 1)

        >>> diff(Heaviside(x)).fdiff()
        DiracDelta(x, 1)

        Parameters
        ==========

        argindex : integer
            order of derivative

        """
        if argindex == 1:
            return DiracDelta(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def __new__(cls, arg, H0=S.Half, **options):
        if isinstance(H0, Heaviside) and len(H0.args) == 1:
            H0 = S.Half
        return super(cls, cls).__new__(cls, arg, H0, **options)

    @property
    def pargs(self):
        """Args without default S.Half"""
        args = self.args
        if args[1] is S.Half:
            args = args[:1]
        return args

    @classmethod
    def eval(cls, arg, H0=S.Half):
        """
        Returns a simplified form or a value of Heaviside depending on the
        argument passed by the Heaviside object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the ``Heaviside``
        class is about to be instantiated and it returns either some simplified
        instance or the unevaluated instance depending on the argument passed.
        In other words, ``eval()`` method is not needed to be called explicitly,
        it is being called and evaluated once the object is called.

        Examples
        ========

        >>> from sympy import Heaviside, S
        >>> from sympy.abc import x

        >>> Heaviside(x)
        Heaviside(x)

        >>> Heaviside(19)
        1

        >>> Heaviside(0)
        1/2

        >>> Heaviside(0, 1)
        1

        >>> Heaviside(-5)
        0

        >>> Heaviside(S.NaN)
        nan

        >>> Heaviside(x - 100).subs(x, 5)
        0

        >>> Heaviside(x - 100).subs(x, 105)
        1

        Parameters
        ==========

        arg : argument passed by Heaviside object

        H0 : value of Heaviside(0)

        """
        if arg.is_extended_negative:
            return S.Zero
        elif arg.is_extended_positive:
            return S.One
        elif arg.is_zero:
            return H0
        elif arg is S.NaN:
            return S.NaN
        elif fuzzy_not(im(arg).is_zero):
            raise ValueError("Function defined only for Real Values. Complex part: %s  found in %s ." % (repr(im(arg)), repr(arg)) )

    def _eval_rewrite_as_Piecewise(self, arg, H0=None, **kwargs):
        """
        Represents Heaviside in a Piecewise form.

        Examples
        ========

        >>> from sympy import Heaviside, Piecewise, Symbol, nan
        >>> x = Symbol('x')

        >>> Heaviside(x).rewrite(Piecewise)
        Piecewise((0, x < 0), (1/2, Eq(x, 0)), (1, True))

        >>> Heaviside(x,nan).rewrite(Piecewise)
        Piecewise((0, x < 0), (nan, Eq(x, 0)), (1, True))

        >>> Heaviside(x - 5).rewrite(Piecewise)
        Piecewise((0, x < 5), (1/2, Eq(x, 5)), (1, True))

        >>> Heaviside(x**2 - 1).rewrite(Piecewise)
        Piecewise((0, x**2 < 1), (1/2, Eq(x**2, 1)), (1, True))

        """
        if H0 == 0:
            return Piecewise((0, arg <= 0), (1, True))
        if H0 == 1:
            return Piecewise((0, arg < 0), (1, True))
        return Piecewise((0, arg < 0), (H0, Eq(arg, 0)), (1, True))

    def _eval_rewrite_as_sign(self, arg, H0=S.Half, **kwargs):
        """
        Represents the Heaviside function in the form of sign function.

        Explanation
        ===========

        The value of Heaviside(0) must be 1/2 for rewriting as sign to be
        strictly equivalent. For easier usage, we also allow this rewriting
        when Heaviside(0) is undefined.

        Examples
        ========

        >>> from sympy import Heaviside, Symbol, sign, nan
        >>> x = Symbol('x', real=True)
        >>> y = Symbol('y')

        >>> Heaviside(x).rewrite(sign)
        sign(x)/2 + 1/2

        >>> Heaviside(x, 0).rewrite(sign)
        Piecewise((sign(x)/2 + 1/2, Ne(x, 0)), (0, True))

        >>> Heaviside(x, nan).rewrite(sign)
        Piecewise((sign(x)/2 + 1/2, Ne(x, 0)), (nan, True))

        >>> Heaviside(x - 2).rewrite(sign)
        sign(x - 2)/2 + 1/2

        >>> Heaviside(x**2 - 2*x + 1).rewrite(sign)
        sign(x**2 - 2*x + 1)/2 + 1/2

        >>> Heaviside(y).rewrite(sign)
        Heaviside(y)

        >>> Heaviside(y**2 - 2*y + 1).rewrite(sign)
        Heaviside(y**2 - 2*y + 1)

        See Also
        ========

        sign

        """
        if arg.is_extended_real:
            pw1 = Piecewise(
                ((sign(arg) + 1)/2, Ne(arg, 0)),
                (Heaviside(0, H0=H0), True))
            pw2 = Piecewise(
                ((sign(arg) + 1)/2, Eq(Heaviside(0, H0=H0), S.Half)),
                (pw1, True))
            return pw2

    def _eval_rewrite_as_SingularityFunction(self, args, H0=S.Half, **kwargs):
        """
        Returns the Heaviside expression written in the form of Singularity
        Functions.

        """
        from sympy.solvers import solve
        from sympy.functions.special.singularity_functions import SingularityFunction
        if self == Heaviside(0):
            return SingularityFunction(0, 0, 0)
        free = self.free_symbols
        if len(free) == 1:
            x = (free.pop())
            return SingularityFunction(x, solve(args, x)[0], 0)
            # TODO
            # ((x - 5)**3*Heaviside(x - 5)).rewrite(SingularityFunction) should output
            # SingularityFunction(x, 5, 0) instead of (x - 5)**3*SingularityFunction(x, 5, 0)
        else:
            # I don't know how to handle the case for Heaviside expressions
            # having arguments with more than one variable.
            raise TypeError(filldedent('''
                rewrite(SingularityFunction) does not
                support arguments with more that one variable.'''))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\headers.py ===
from __future__ import annotations

import collections.abc as cabc
import re
import typing as t

from .._internal import _missing
from ..exceptions import BadRequestKeyError
from .mixins import ImmutableHeadersMixin
from .structures import iter_multi_items
from .structures import MultiDict

if t.TYPE_CHECKING:
    import typing_extensions as te
    from _typeshed.wsgi import WSGIEnvironment

T = t.TypeVar("T")


class Headers:
    """An object that stores some headers. It has a dict-like interface,
    but is ordered, can store the same key multiple times, and iterating
    yields ``(key, value)`` pairs instead of only keys.

    This data structure is useful if you want a nicer way to handle WSGI
    headers which are stored as tuples in a list.

    From Werkzeug 0.3 onwards, the :exc:`KeyError` raised by this class is
    also a subclass of the :class:`~exceptions.BadRequest` HTTP exception
    and will render a page for a ``400 BAD REQUEST`` if caught in a
    catch-all for HTTP exceptions.

    Headers is mostly compatible with the Python :class:`wsgiref.headers.Headers`
    class, with the exception of `__getitem__`.  :mod:`wsgiref` will return
    `None` for ``headers['missing']``, whereas :class:`Headers` will raise
    a :class:`KeyError`.

    To create a new ``Headers`` object, pass it a list, dict, or
    other ``Headers`` object with default values. These values are
    validated the same way values added later are.

    :param defaults: The list of default values for the :class:`Headers`.

    .. versionchanged:: 3.1
        Implement ``|`` and ``|=`` operators.

    .. versionchanged:: 2.1.0
        Default values are validated the same as values added later.

    .. versionchanged:: 0.9
       This data structure now stores unicode values similar to how the
       multi dicts do it.  The main difference is that bytes can be set as
       well which will automatically be latin1 decoded.

    .. versionchanged:: 0.9
       The :meth:`linked` function was removed without replacement as it
       was an API that does not support the changes to the encoding model.
    """

    def __init__(
        self,
        defaults: (
            Headers
            | MultiDict[str, t.Any]
            | cabc.Mapping[str, t.Any | list[t.Any] | tuple[t.Any, ...] | set[t.Any]]
            | cabc.Iterable[tuple[str, t.Any]]
            | None
        ) = None,
    ) -> None:
        self._list: list[tuple[str, str]] = []

        if defaults is not None:
            self.extend(defaults)

    @t.overload
    def __getitem__(self, key: str) -> str: ...
    @t.overload
    def __getitem__(self, key: int) -> tuple[str, str]: ...
    @t.overload
    def __getitem__(self, key: slice) -> te.Self: ...
    def __getitem__(self, key: str | int | slice) -> str | tuple[str, str] | te.Self:
        if isinstance(key, str):
            return self._get_key(key)

        if isinstance(key, int):
            return self._list[key]

        return self.__class__(self._list[key])

    def _get_key(self, key: str) -> str:
        ikey = key.lower()

        for k, v in self._list:
            if k.lower() == ikey:
                return v

        raise BadRequestKeyError(key)

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        def lowered(item: tuple[str, ...]) -> tuple[str, ...]:
            return item[0].lower(), *item[1:]

        return set(map(lowered, other._list)) == set(map(lowered, self._list))  # type: ignore[attr-defined]

    __hash__ = None  # type: ignore[assignment]

    @t.overload
    def get(self, key: str) -> str | None: ...
    @t.overload
    def get(self, key: str, default: str) -> str: ...
    @t.overload
    def get(self, key: str, default: T) -> str | T: ...
    @t.overload
    def get(self, key: str, type: cabc.Callable[[str], T]) -> T | None: ...
    @t.overload
    def get(self, key: str, default: T, type: cabc.Callable[[str], T]) -> T: ...
    def get(  # type: ignore[misc]
        self,
        key: str,
        default: str | T | None = None,
        type: cabc.Callable[[str], T] | None = None,
    ) -> str | T | None:
        """Return the default value if the requested data doesn't exist.
        If `type` is provided and is a callable it should convert the value,
        return it or raise a :exc:`ValueError` if that is not possible.  In
        this case the function will return the default as if the value was not
        found:

        >>> d = Headers([('Content-Length', '42')])
        >>> d.get('Content-Length', type=int)
        42

        :param key: The key to be looked up.
        :param default: The default value to be returned if the key can't
                        be looked up.  If not further specified `None` is
                        returned.
        :param type: A callable that is used to cast the value in the
                     :class:`Headers`.  If a :exc:`ValueError` is raised
                     by this callable the default value is returned.

        .. versionchanged:: 3.0
            The ``as_bytes`` parameter was removed.

        .. versionchanged:: 0.9
            The ``as_bytes`` parameter was added.
        """
        try:
            rv = self._get_key(key)
        except KeyError:
            return default

        if type is None:
            return rv

        try:
            return type(rv)
        except ValueError:
            return default

    @t.overload
    def getlist(self, key: str) -> list[str]: ...
    @t.overload
    def getlist(self, key: str, type: cabc.Callable[[str], T]) -> list[T]: ...
    def getlist(
        self, key: str, type: cabc.Callable[[str], T] | None = None
    ) -> list[str] | list[T]:
        """Return the list of items for a given key. If that key is not in the
        :class:`Headers`, the return value will be an empty list.  Just like
        :meth:`get`, :meth:`getlist` accepts a `type` parameter.  All items will
        be converted with the callable defined there.

        :param key: The key to be looked up.
        :param type: A callable that is used to cast the value in the
                     :class:`Headers`.  If a :exc:`ValueError` is raised
                     by this callable the value will be removed from the list.
        :return: a :class:`list` of all the values for the key.

        .. versionchanged:: 3.0
            The ``as_bytes`` parameter was removed.

        .. versionchanged:: 0.9
            The ``as_bytes`` parameter was added.
        """
        ikey = key.lower()

        if type is not None:
            result = []

            for k, v in self:
                if k.lower() == ikey:
                    try:
                        result.append(type(v))
                    except ValueError:
                        continue

            return result

        return [v for k, v in self if k.lower() == ikey]

    def get_all(self, name: str) -> list[str]:
        """Return a list of all the values for the named field.

        This method is compatible with the :mod:`wsgiref`
        :meth:`~wsgiref.headers.Headers.get_all` method.
        """
        return self.getlist(name)

    def items(self, lower: bool = False) -> t.Iterable[tuple[str, str]]:
        for key, value in self:
            if lower:
                key = key.lower()
            yield key, value

    def keys(self, lower: bool = False) -> t.Iterable[str]:
        for key, _ in self.items(lower):
            yield key

    def values(self) -> t.Iterable[str]:
        for _, value in self.items():
            yield value

    def extend(
        self,
        arg: (
            Headers
            | MultiDict[str, t.Any]
            | cabc.Mapping[str, t.Any | list[t.Any] | tuple[t.Any, ...] | set[t.Any]]
            | cabc.Iterable[tuple[str, t.Any]]
            | None
        ) = None,
        /,
        **kwargs: str,
    ) -> None:
        """Extend headers in this object with items from another object
        containing header items as well as keyword arguments.

        To replace existing keys instead of extending, use
        :meth:`update` instead.

        If provided, the first argument can be another :class:`Headers`
        object, a :class:`MultiDict`, :class:`dict`, or iterable of
        pairs.

        .. versionchanged:: 1.0
            Support :class:`MultiDict`. Allow passing ``kwargs``.
        """
        if arg is not None:
            for key, value in iter_multi_items(arg):
                self.add(key, value)

        for key, value in iter_multi_items(kwargs):
            self.add(key, value)

    def __delitem__(self, key: str | int | slice) -> None:
        if isinstance(key, str):
            self._del_key(key)
            return

        del self._list[key]

    def _del_key(self, key: str) -> None:
        key = key.lower()
        new = []

        for k, v in self._list:
            if k.lower() != key:
                new.append((k, v))

        self._list[:] = new

    def remove(self, key: str) -> None:
        """Remove a key.

        :param key: The key to be removed.
        """
        return self._del_key(key)

    @t.overload
    def pop(self) -> tuple[str, str]: ...
    @t.overload
    def pop(self, key: str) -> str: ...
    @t.overload
    def pop(self, key: int | None = ...) -> tuple[str, str]: ...
    @t.overload
    def pop(self, key: str, default: str) -> str: ...
    @t.overload
    def pop(self, key: str, default: T) -> str | T: ...
    def pop(
        self,
        key: str | int | None = None,
        default: str | T = _missing,  # type: ignore[assignment]
    ) -> str | tuple[str, str] | T:
        """Removes and returns a key or index.

        :param key: The key to be popped.  If this is an integer the item at
                    that position is removed, if it's a string the value for
                    that key is.  If the key is omitted or `None` the last
                    item is removed.
        :return: an item.
        """
        if key is None:
            return self._list.pop()

        if isinstance(key, int):
            return self._list.pop(key)

        try:
            rv = self._get_key(key)
        except KeyError:
            if default is not _missing:
                return default

            raise

        self.remove(key)
        return rv

    def popitem(self) -> tuple[str, str]:
        """Removes a key or index and returns a (key, value) item."""
        return self._list.pop()

    def __contains__(self, key: str) -> bool:
        """Check if a key is present."""
        try:
            self._get_key(key)
        except KeyError:
            return False

        return True

    def __iter__(self) -> t.Iterator[tuple[str, str]]:
        """Yield ``(key, value)`` tuples."""
        return iter(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def add(self, key: str, value: t.Any, /, **kwargs: t.Any) -> None:
        """Add a new header tuple to the list.

        Keyword arguments can specify additional parameters for the header
        value, with underscores converted to dashes::

        >>> d = Headers()
        >>> d.add('Content-Type', 'text/plain')
        >>> d.add('Content-Disposition', 'attachment', filename='foo.png')

        The keyword argument dumping uses :func:`dump_options_header`
        behind the scenes.

        .. versionchanged:: 0.4.1
            keyword arguments were added for :mod:`wsgiref` compatibility.
        """
        if kwargs:
            value = _options_header_vkw(value, kwargs)

        value_str = _str_header_value(value)
        self._list.append((key, value_str))

    def add_header(self, key: str, value: t.Any, /, **kwargs: t.Any) -> None:
        """Add a new header tuple to the list.

        An alias for :meth:`add` for compatibility with the :mod:`wsgiref`
        :meth:`~wsgiref.headers.Headers.add_header` method.
        """
        self.add(key, value, **kwargs)

    def clear(self) -> None:
        """Clears all headers."""
        self._list.clear()

    def set(self, key: str, value: t.Any, /, **kwargs: t.Any) -> None:
        """Remove all header tuples for `key` and add a new one.  The newly
        added key either appears at the end of the list if there was no
        entry or replaces the first one.

        Keyword arguments can specify additional parameters for the header
        value, with underscores converted to dashes.  See :meth:`add` for
        more information.

        .. versionchanged:: 0.6.1
           :meth:`set` now accepts the same arguments as :meth:`add`.

        :param key: The key to be inserted.
        :param value: The value to be inserted.
        """
        if kwargs:
            value = _options_header_vkw(value, kwargs)

        value_str = _str_header_value(value)

        if not self._list:
            self._list.append((key, value_str))
            return

        iter_list = iter(self._list)
        ikey = key.lower()

        for idx, (old_key, _) in enumerate(iter_list):
            if old_key.lower() == ikey:
                # replace first occurrence
                self._list[idx] = (key, value_str)
                break
        else:
            # no existing occurrences
            self._list.append((key, value_str))
            return

        # remove remaining occurrences
        self._list[idx + 1 :] = [t for t in iter_list if t[0].lower() != ikey]

    def setlist(self, key: str, values: cabc.Iterable[t.Any]) -> None:
        """Remove any existing values for a header and add new ones.

        :param key: The header key to set.
        :param values: An iterable of values to set for the key.

        .. versionadded:: 1.0
        """
        if values:
            values_iter = iter(values)
            self.set(key, next(values_iter))

            for value in values_iter:
                self.add(key, value)
        else:
            self.remove(key)

    def setdefault(self, key: str, default: t.Any) -> str:
        """Return the first value for the key if it is in the headers,
        otherwise set the header to the value given by ``default`` and
        return that.

        :param key: The header key to get.
        :param default: The value to set for the key if it is not in the
            headers.
        """
        try:
            return self._get_key(key)
        except KeyError:
            pass

        self.set(key, default)
        return self._get_key(key)

    def setlistdefault(self, key: str, default: cabc.Iterable[t.Any]) -> list[str]:
        """Return the list of values for the key if it is in the
        headers, otherwise set the header to the list of values given
        by ``default`` and return that.

        Unlike :meth:`MultiDict.setlistdefault`, modifying the returned
        list will not affect the headers.

        :param key: The header key to get.
        :param default: An iterable of values to set for the key if it
            is not in the headers.

        .. versionadded:: 1.0
        """
        if key not in self:
            self.setlist(key, default)

        return self.getlist(key)

    @t.overload
    def __setitem__(self, key: str, value: t.Any) -> None: ...
    @t.overload
    def __setitem__(self, key: int, value: tuple[str, t.Any]) -> None: ...
    @t.overload
    def __setitem__(
        self, key: slice, value: cabc.Iterable[tuple[str, t.Any]]
    ) -> None: ...
    def __setitem__(
        self,
        key: str | int | slice,
        value: t.Any | tuple[str, t.Any] | cabc.Iterable[tuple[str, t.Any]],
    ) -> None:
        """Like :meth:`set` but also supports index/slice based setting."""
        if isinstance(key, str):
            self.set(key, value)
        elif isinstance(key, int):
            self._list[key] = value[0], _str_header_value(value[1])  # type: ignore[index]
        else:
            self._list[key] = [(k, _str_header_value(v)) for k, v in value]  # type: ignore[misc]

    def update(
        self,
        arg: (
            Headers
            | MultiDict[str, t.Any]
            | cabc.Mapping[
                str, t.Any | list[t.Any] | tuple[t.Any, ...] | cabc.Set[t.Any]
            ]
            | cabc.Iterable[tuple[str, t.Any]]
            | None
        ) = None,
        /,
        **kwargs: t.Any | list[t.Any] | tuple[t.Any, ...] | cabc.Set[t.Any],
    ) -> None:
        """Replace headers in this object with items from another
        headers object and keyword arguments.

        To extend existing keys instead of replacing, use :meth:`extend`
        instead.

        If provided, the first argument can be another :class:`Headers`
        object, a :class:`MultiDict`, :class:`dict`, or iterable of
        pairs.

        .. versionadded:: 1.0
        """
        if arg is not None:
            if isinstance(arg, (Headers, MultiDict)):
                for key in arg.keys():
                    self.setlist(key, arg.getlist(key))
            elif isinstance(arg, cabc.Mapping):
                for key, value in arg.items():
                    if isinstance(value, (list, tuple, set)):
                        self.setlist(key, value)
                    else:
                        self.set(key, value)
            else:
                for key, value in arg:
                    self.set(key, value)

        for key, value in kwargs.items():
            if isinstance(value, (list, tuple, set)):
                self.setlist(key, value)
            else:
                self.set(key, value)

    def __or__(
        self,
        other: cabc.Mapping[
            str, t.Any | list[t.Any] | tuple[t.Any, ...] | cabc.Set[t.Any]
        ],
    ) -> te.Self:
        if not isinstance(other, cabc.Mapping):
            return NotImplemented

        rv = self.copy()
        rv.update(other)
        return rv

    def __ior__(
        self,
        other: (
            cabc.Mapping[str, t.Any | list[t.Any] | tuple[t.Any, ...] | cabc.Set[t.Any]]
            | cabc.Iterable[tuple[str, t.Any]]
        ),
    ) -> te.Self:
        if not isinstance(other, (cabc.Mapping, cabc.Iterable)):
            return NotImplemented

        self.update(other)
        return self

    def to_wsgi_list(self) -> list[tuple[str, str]]:
        """Convert the headers into a list suitable for WSGI.

        :return: list
        """
        return list(self)

    def copy(self) -> te.Self:
        return self.__class__(self._list)

    def __copy__(self) -> te.Self:
        return self.copy()

    def __str__(self) -> str:
        """Returns formatted headers suitable for HTTP transmission."""
        strs = []
        for key, value in self.to_wsgi_list():
            strs.append(f"{key}: {value}")
        strs.append("\r\n")
        return "\r\n".join(strs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)!r})"


def _options_header_vkw(value: str, kw: dict[str, t.Any]) -> str:
    return http.dump_options_header(
        value, {k.replace("_", "-"): v for k, v in kw.items()}
    )


_newline_re = re.compile(r"[\r\n]")


def _str_header_value(value: t.Any) -> str:
    if not isinstance(value, str):
        value = str(value)

    if _newline_re.search(value) is not None:
        raise ValueError("Header values must not contain newline characters.")

    return value  # type: ignore[no-any-return]


class EnvironHeaders(ImmutableHeadersMixin, Headers):  # type: ignore[misc]
    """Read only version of the headers from a WSGI environment.  This
    provides the same interface as `Headers` and is constructed from
    a WSGI environment.
    From Werkzeug 0.3 onwards, the `KeyError` raised by this class is also a
    subclass of the :exc:`~exceptions.BadRequest` HTTP exception and will
    render a page for a ``400 BAD REQUEST`` if caught in a catch-all for
    HTTP exceptions.
    """

    def __init__(self, environ: WSGIEnvironment) -> None:
        super().__init__()
        self.environ = environ

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvironHeaders):
            return NotImplemented

        return self.environ is other.environ

    __hash__ = None  # type: ignore[assignment]

    def __getitem__(self, key: str) -> str:  # type: ignore[override]
        return self._get_key(key)

    def _get_key(self, key: str) -> str:
        if not isinstance(key, str):
            raise BadRequestKeyError(key)

        key = key.upper().replace("-", "_")

        if key in {"CONTENT_TYPE", "CONTENT_LENGTH"}:
            return self.environ[key]  # type: ignore[no-any-return]

        return self.environ[f"HTTP_{key}"]  # type: ignore[no-any-return]

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> cabc.Iterator[tuple[str, str]]:
        for key, value in self.environ.items():
            if key.startswith("HTTP_") and key not in {
                "HTTP_CONTENT_TYPE",
                "HTTP_CONTENT_LENGTH",
            }:
                yield key[5:].replace("_", "-").title(), value
            elif key in {"CONTENT_TYPE", "CONTENT_LENGTH"} and value:
                yield key.replace("_", "-").title(), value

    def copy(self) -> t.NoReturn:
        raise TypeError(f"cannot create {type(self).__name__!r} copies")

    def __or__(self, other: t.Any) -> t.NoReturn:
        raise TypeError(f"cannot create {type(self).__name__!r} copies")


# circular dependencies
from .. import http

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_win32_console.py ===
"""Light wrapper around the Win32 Console API - this module should only be imported on Windows

The API that this module wraps is documented at https://docs.microsoft.com/en-us/windows/console/console-functions
"""

import ctypes
import sys
from typing import Any

windll: Any = None
if sys.platform == "win32":
    windll = ctypes.LibraryLoader(ctypes.WinDLL)
else:
    raise ImportError(f"{__name__} can only be imported on Windows")

import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast

from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style

STDOUT = -11
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 4

COORD = wintypes._COORD


class LegacyWindowsError(Exception):
    pass


class WindowsCoordinates(NamedTuple):
    """Coordinates in the Windows Console API are (y, x), not (x, y).
    This class is intended to prevent that confusion.
    Rows and columns are indexed from 0.
    This class can be used in place of wintypes._COORD in arguments and argtypes.
    """

    row: int
    col: int

    @classmethod
    def from_param(cls, value: "WindowsCoordinates") -> COORD:
        """Converts a WindowsCoordinates into a wintypes _COORD structure.
        This classmethod is internally called by ctypes to perform the conversion.

        Args:
            value (WindowsCoordinates): The input coordinates to convert.

        Returns:
            wintypes._COORD: The converted coordinates struct.
        """
        return COORD(value.col, value.row)


class CONSOLE_SCREEN_BUFFER_INFO(Structure):
    _fields_ = [
        ("dwSize", COORD),
        ("dwCursorPosition", COORD),
        ("wAttributes", wintypes.WORD),
        ("srWindow", wintypes.SMALL_RECT),
        ("dwMaximumWindowSize", COORD),
    ]


class CONSOLE_CURSOR_INFO(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("bVisible", wintypes.BOOL)]


_GetStdHandle = windll.kernel32.GetStdHandle
_GetStdHandle.argtypes = [
    wintypes.DWORD,
]
_GetStdHandle.restype = wintypes.HANDLE


def GetStdHandle(handle: int = STDOUT) -> wintypes.HANDLE:
    """Retrieves a handle to the specified standard device (standard input, standard output, or standard error).

    Args:
        handle (int): Integer identifier for the handle. Defaults to -11 (stdout).

    Returns:
        wintypes.HANDLE: The handle
    """
    return cast(wintypes.HANDLE, _GetStdHandle(handle))


_GetConsoleMode = windll.kernel32.GetConsoleMode
_GetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
_GetConsoleMode.restype = wintypes.BOOL


def GetConsoleMode(std_handle: wintypes.HANDLE) -> int:
    """Retrieves the current input mode of a console's input buffer
    or the current output mode of a console screen buffer.

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.

    Raises:
        LegacyWindowsError: If any error occurs while calling the Windows console API.

    Returns:
        int: Value representing the current console mode as documented at
            https://docs.microsoft.com/en-us/windows/console/getconsolemode#parameters
    """

    console_mode = wintypes.DWORD()
    success = bool(_GetConsoleMode(std_handle, console_mode))
    if not success:
        raise LegacyWindowsError("Unable to get legacy Windows Console Mode")
    return console_mode.value


_FillConsoleOutputCharacterW = windll.kernel32.FillConsoleOutputCharacterW
_FillConsoleOutputCharacterW.argtypes = [
    wintypes.HANDLE,
    ctypes.c_char,
    wintypes.DWORD,
    cast(Type[COORD], WindowsCoordinates),
    ctypes.POINTER(wintypes.DWORD),
]
_FillConsoleOutputCharacterW.restype = wintypes.BOOL


def FillConsoleOutputCharacter(
    std_handle: wintypes.HANDLE,
    char: str,
    length: int,
    start: WindowsCoordinates,
) -> int:
    """Writes a character to the console screen buffer a specified number of times, beginning at the specified coordinates.

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
        char (str): The character to write. Must be a string of length 1.
        length (int): The number of times to write the character.
        start (WindowsCoordinates): The coordinates to start writing at.

    Returns:
        int: The number of characters written.
    """
    character = ctypes.c_char(char.encode())
    num_characters = wintypes.DWORD(length)
    num_written = wintypes.DWORD(0)
    _FillConsoleOutputCharacterW(
        std_handle,
        character,
        num_characters,
        start,
        byref(num_written),
    )
    return num_written.value


_FillConsoleOutputAttribute = windll.kernel32.FillConsoleOutputAttribute
_FillConsoleOutputAttribute.argtypes = [
    wintypes.HANDLE,
    wintypes.WORD,
    wintypes.DWORD,
    cast(Type[COORD], WindowsCoordinates),
    ctypes.POINTER(wintypes.DWORD),
]
_FillConsoleOutputAttribute.restype = wintypes.BOOL


def FillConsoleOutputAttribute(
    std_handle: wintypes.HANDLE,
    attributes: int,
    length: int,
    start: WindowsCoordinates,
) -> int:
    """Sets the character attributes for a specified number of character cells,
    beginning at the specified coordinates in a screen buffer.

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
        attributes (int): Integer value representing the foreground and background colours of the cells.
        length (int): The number of cells to set the output attribute of.
        start (WindowsCoordinates): The coordinates of the first cell whose attributes are to be set.

    Returns:
        int: The number of cells whose attributes were actually set.
    """
    num_cells = wintypes.DWORD(length)
    style_attrs = wintypes.WORD(attributes)
    num_written = wintypes.DWORD(0)
    _FillConsoleOutputAttribute(
        std_handle, style_attrs, num_cells, start, byref(num_written)
    )
    return num_written.value


_SetConsoleTextAttribute = windll.kernel32.SetConsoleTextAttribute
_SetConsoleTextAttribute.argtypes = [
    wintypes.HANDLE,
    wintypes.WORD,
]
_SetConsoleTextAttribute.restype = wintypes.BOOL


def SetConsoleTextAttribute(
    std_handle: wintypes.HANDLE, attributes: wintypes.WORD
) -> bool:
    """Set the colour attributes for all text written after this function is called.

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
        attributes (int): Integer value representing the foreground and background colours.


    Returns:
        bool: True if the attribute was set successfully, otherwise False.
    """
    return bool(_SetConsoleTextAttribute(std_handle, attributes))


_GetConsoleScreenBufferInfo = windll.kernel32.GetConsoleScreenBufferInfo
_GetConsoleScreenBufferInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(CONSOLE_SCREEN_BUFFER_INFO),
]
_GetConsoleScreenBufferInfo.restype = wintypes.BOOL


def GetConsoleScreenBufferInfo(
    std_handle: wintypes.HANDLE,
) -> CONSOLE_SCREEN_BUFFER_INFO:
    """Retrieves information about the specified console screen buffer.

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.

    Returns:
        CONSOLE_SCREEN_BUFFER_INFO: A CONSOLE_SCREEN_BUFFER_INFO ctype struct contain information about
            screen size, cursor position, colour attributes, and more."""
    console_screen_buffer_info = CONSOLE_SCREEN_BUFFER_INFO()
    _GetConsoleScreenBufferInfo(std_handle, byref(console_screen_buffer_info))
    return console_screen_buffer_info


_SetConsoleCursorPosition = windll.kernel32.SetConsoleCursorPosition
_SetConsoleCursorPosition.argtypes = [
    wintypes.HANDLE,
    cast(Type[COORD], WindowsCoordinates),
]
_SetConsoleCursorPosition.restype = wintypes.BOOL


def SetConsoleCursorPosition(
    std_handle: wintypes.HANDLE, coords: WindowsCoordinates
) -> bool:
    """Set the position of the cursor in the console screen

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
        coords (WindowsCoordinates): The coordinates to move the cursor to.

    Returns:
        bool: True if the function succeeds, otherwise False.
    """
    return bool(_SetConsoleCursorPosition(std_handle, coords))


_GetConsoleCursorInfo = windll.kernel32.GetConsoleCursorInfo
_GetConsoleCursorInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(CONSOLE_CURSOR_INFO),
]
_GetConsoleCursorInfo.restype = wintypes.BOOL


def GetConsoleCursorInfo(
    std_handle: wintypes.HANDLE, cursor_info: CONSOLE_CURSOR_INFO
) -> bool:
    """Get the cursor info - used to get cursor visibility and width

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
        cursor_info (CONSOLE_CURSOR_INFO): CONSOLE_CURSOR_INFO ctype struct that receives information
            about the console's cursor.

    Returns:
          bool: True if the function succeeds, otherwise False.
    """
    return bool(_GetConsoleCursorInfo(std_handle, byref(cursor_info)))


_SetConsoleCursorInfo = windll.kernel32.SetConsoleCursorInfo
_SetConsoleCursorInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(CONSOLE_CURSOR_INFO),
]
_SetConsoleCursorInfo.restype = wintypes.BOOL


def SetConsoleCursorInfo(
    std_handle: wintypes.HANDLE, cursor_info: CONSOLE_CURSOR_INFO
) -> bool:
    """Set the cursor info - used for adjusting cursor visibility and width

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
        cursor_info (CONSOLE_CURSOR_INFO): CONSOLE_CURSOR_INFO ctype struct containing the new cursor info.

    Returns:
          bool: True if the function succeeds, otherwise False.
    """
    return bool(_SetConsoleCursorInfo(std_handle, byref(cursor_info)))


_SetConsoleTitle = windll.kernel32.SetConsoleTitleW
_SetConsoleTitle.argtypes = [wintypes.LPCWSTR]
_SetConsoleTitle.restype = wintypes.BOOL


def SetConsoleTitle(title: str) -> bool:
    """Sets the title of the current console window

    Args:
        title (str): The new title of the console window.

    Returns:
        bool: True if the function succeeds, otherwise False.
    """
    return bool(_SetConsoleTitle(title))


class LegacyWindowsTerm:
    """This class allows interaction with the legacy Windows Console API. It should only be used in the context
    of environments where virtual terminal processing is not available. However, if it is used in a Windows environment,
    the entire API should work.

    Args:
        file (IO[str]): The file which the Windows Console API HANDLE is retrieved from, defaults to sys.stdout.
    """

    BRIGHT_BIT = 8

    # Indices are ANSI color numbers, values are the corresponding Windows Console API color numbers
    ANSI_TO_WINDOWS = [
        0,  # black                      The Windows colours are defined in wincon.h as follows:
        4,  # red                         define FOREGROUND_BLUE            0x0001 -- 0000 0001
        2,  # green                       define FOREGROUND_GREEN           0x0002 -- 0000 0010
        6,  # yellow                      define FOREGROUND_RED             0x0004 -- 0000 0100
        1,  # blue                        define FOREGROUND_INTENSITY       0x0008 -- 0000 1000
        5,  # magenta                     define BACKGROUND_BLUE            0x0010 -- 0001 0000
        3,  # cyan                        define BACKGROUND_GREEN           0x0020 -- 0010 0000
        7,  # white                       define BACKGROUND_RED             0x0040 -- 0100 0000
        8,  # bright black (grey)         define BACKGROUND_INTENSITY       0x0080 -- 1000 0000
        12,  # bright red
        10,  # bright green
        14,  # bright yellow
        9,  # bright blue
        13,  # bright magenta
        11,  # bright cyan
        15,  # bright white
    ]

    def __init__(self, file: "IO[str]") -> None:
        handle = GetStdHandle(STDOUT)
        self._handle = handle
        default_text = GetConsoleScreenBufferInfo(handle).wAttributes
        self._default_text = default_text

        self._default_fore = default_text & 7
        self._default_back = (default_text >> 4) & 7
        self._default_attrs = self._default_fore | (self._default_back << 4)

        self._file = file
        self.write = file.write
        self.flush = file.flush

    @property
    def cursor_position(self) -> WindowsCoordinates:
        """Returns the current position of the cursor (0-based)

        Returns:
            WindowsCoordinates: The current cursor position.
        """
        coord: COORD = GetConsoleScreenBufferInfo(self._handle).dwCursorPosition
        return WindowsCoordinates(row=coord.Y, col=coord.X)

    @property
    def screen_size(self) -> WindowsCoordinates:
        """Returns the current size of the console screen buffer, in character columns and rows

        Returns:
            WindowsCoordinates: The width and height of the screen as WindowsCoordinates.
        """
        screen_size: COORD = GetConsoleScreenBufferInfo(self._handle).dwSize
        return WindowsCoordinates(row=screen_size.Y, col=screen_size.X)

    def write_text(self, text: str) -> None:
        """Write text directly to the terminal without any modification of styles

        Args:
            text (str): The text to write to the console
        """
        self.write(text)
        self.flush()

    def write_styled(self, text: str, style: Style) -> None:
        """Write styled text to the terminal.

        Args:
            text (str): The text to write
            style (Style): The style of the text
        """
        color = style.color
        bgcolor = style.bgcolor
        if style.reverse:
            color, bgcolor = bgcolor, color

        if color:
            fore = color.downgrade(ColorSystem.WINDOWS).number
            fore = fore if fore is not None else 7  # Default to ANSI 7: White
            if style.bold:
                fore = fore | self.BRIGHT_BIT
            if style.dim:
                fore = fore & ~self.BRIGHT_BIT
            fore = self.ANSI_TO_WINDOWS[fore]
        else:
            fore = self._default_fore

        if bgcolor:
            back = bgcolor.downgrade(ColorSystem.WINDOWS).number
            back = back if back is not None else 0  # Default to ANSI 0: Black
            back = self.ANSI_TO_WINDOWS[back]
        else:
            back = self._default_back

        assert fore is not None
        assert back is not None

        SetConsoleTextAttribute(
            self._handle, attributes=ctypes.c_ushort(fore | (back << 4))
        )
        self.write_text(text)
        SetConsoleTextAttribute(self._handle, attributes=self._default_text)

    def move_cursor_to(self, new_position: WindowsCoordinates) -> None:
        """Set the position of the cursor

        Args:
            new_position (WindowsCoordinates): The WindowsCoordinates representing the new position of the cursor.
        """
        if new_position.col < 0 or new_position.row < 0:
            return
        SetConsoleCursorPosition(self._handle, coords=new_position)

    def erase_line(self) -> None:
        """Erase all content on the line the cursor is currently located at"""
        screen_size = self.screen_size
        cursor_position = self.cursor_position
        cells_to_erase = screen_size.col
        start_coordinates = WindowsCoordinates(row=cursor_position.row, col=0)
        FillConsoleOutputCharacter(
            self._handle, " ", length=cells_to_erase, start=start_coordinates
        )
        FillConsoleOutputAttribute(
            self._handle,
            self._default_attrs,
            length=cells_to_erase,
            start=start_coordinates,
        )

    def erase_end_of_line(self) -> None:
        """Erase all content from the cursor position to the end of that line"""
        cursor_position = self.cursor_position
        cells_to_erase = self.screen_size.col - cursor_position.col
        FillConsoleOutputCharacter(
            self._handle, " ", length=cells_to_erase, start=cursor_position
        )
        FillConsoleOutputAttribute(
            self._handle,
            self._default_attrs,
            length=cells_to_erase,
            start=cursor_position,
        )

    def erase_start_of_line(self) -> None:
        """Erase all content from the cursor position to the start of that line"""
        row, col = self.cursor_position
        start = WindowsCoordinates(row, 0)
        FillConsoleOutputCharacter(self._handle, " ", length=col, start=start)
        FillConsoleOutputAttribute(
            self._handle, self._default_attrs, length=col, start=start
        )

    def move_cursor_up(self) -> None:
        """Move the cursor up a single cell"""
        cursor_position = self.cursor_position
        SetConsoleCursorPosition(
            self._handle,
            coords=WindowsCoordinates(
                row=cursor_position.row - 1, col=cursor_position.col
            ),
        )

    def move_cursor_down(self) -> None:
        """Move the cursor down a single cell"""
        cursor_position = self.cursor_position
        SetConsoleCursorPosition(
            self._handle,
            coords=WindowsCoordinates(
                row=cursor_position.row + 1,
                col=cursor_position.col,
            ),
        )

    def move_cursor_forward(self) -> None:
        """Move the cursor forward a single cell. Wrap to the next line if required."""
        row, col = self.cursor_position
        if col == self.screen_size.col - 1:
            row += 1
            col = 0
        else:
            col += 1
        SetConsoleCursorPosition(
            self._handle, coords=WindowsCoordinates(row=row, col=col)
        )

    def move_cursor_to_column(self, column: int) -> None:
        """Move cursor to the column specified by the zero-based column index, staying on the same row

        Args:
            column (int): The zero-based column index to move the cursor to.
        """
        row, _ = self.cursor_position
        SetConsoleCursorPosition(self._handle, coords=WindowsCoordinates(row, column))

    def move_cursor_backward(self) -> None:
        """Move the cursor backward a single cell. Wrap to the previous line if required."""
        row, col = self.cursor_position
        if col == 0:
            row -= 1
            col = self.screen_size.col - 1
        else:
            col -= 1
        SetConsoleCursorPosition(
            self._handle, coords=WindowsCoordinates(row=row, col=col)
        )

    def hide_cursor(self) -> None:
        """Hide the cursor"""
        current_cursor_size = self._get_cursor_size()
        invisible_cursor = CONSOLE_CURSOR_INFO(dwSize=current_cursor_size, bVisible=0)
        SetConsoleCursorInfo(self._handle, cursor_info=invisible_cursor)

    def show_cursor(self) -> None:
        """Show the cursor"""
        current_cursor_size = self._get_cursor_size()
        visible_cursor = CONSOLE_CURSOR_INFO(dwSize=current_cursor_size, bVisible=1)
        SetConsoleCursorInfo(self._handle, cursor_info=visible_cursor)

    def set_title(self, title: str) -> None:
        """Set the title of the terminal window

        Args:
            title (str): The new title of the console window
        """
        assert len(title) < 255, "Console title must be less than 255 characters"
        SetConsoleTitle(title)

    def _get_cursor_size(self) -> int:
        """Get the percentage of the character cell that is filled by the cursor"""
        cursor_info = CONSOLE_CURSOR_INFO()
        GetConsoleCursorInfo(self._handle, cursor_info=cursor_info)
        return int(cursor_info.dwSize)


if __name__ == "__main__":
    handle = GetStdHandle()

    from pip._vendor.rich.console import Console

    console = Console()

    term = LegacyWindowsTerm(sys.stdout)
    term.set_title("Win32 Console Examples")

    style = Style(color="black", bgcolor="red")

    heading = Style.parse("black on green")

    # Check colour output
    console.rule("Checking colour output")
    console.print("[on red]on red!")
    console.print("[blue]blue!")
    console.print("[yellow]yellow!")
    console.print("[bold yellow]bold yellow!")
    console.print("[bright_yellow]bright_yellow!")
    console.print("[dim bright_yellow]dim bright_yellow!")
    console.print("[italic cyan]italic cyan!")
    console.print("[bold white on blue]bold white on blue!")
    console.print("[reverse bold white on blue]reverse bold white on blue!")
    console.print("[bold black on cyan]bold black on cyan!")
    console.print("[black on green]black on green!")
    console.print("[blue on green]blue on green!")
    console.print("[white on black]white on black!")
    console.print("[black on white]black on white!")
    console.print("[#1BB152 on #DA812D]#1BB152 on #DA812D!")

    # Check cursor movement
    console.rule("Checking cursor movement")
    console.print()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text("went back and wrapped to prev line")
    time.sleep(1)
    term.move_cursor_up()
    term.write_text("we go up")
    time.sleep(1)
    term.move_cursor_down()
    term.write_text("and down")
    time.sleep(1)
    term.move_cursor_up()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text("we went up and back 2")
    time.sleep(1)
    term.move_cursor_down()
    term.move_cursor_backward()
    term.move_cursor_backward()
    term.write_text("we went down and back 2")
    time.sleep(1)

    # Check erasing of lines
    term.hide_cursor()
    console.print()
    console.rule("Checking line erasing")
    console.print("\n...Deleting to the start of the line...")
    term.write_text("The red arrow shows the cursor location, and direction of erase")
    time.sleep(1)
    term.move_cursor_to_column(16)
    term.write_styled("<", Style.parse("black on red"))
    term.move_cursor_backward()
    time.sleep(1)
    term.erase_start_of_line()
    time.sleep(1)

    console.print("\n\n...And to the end of the line...")
    term.write_text("The red arrow shows the cursor location, and direction of erase")
    time.sleep(1)

    term.move_cursor_to_column(16)
    term.write_styled(">", Style.parse("black on red"))
    time.sleep(1)
    term.erase_end_of_line()
    time.sleep(1)

    console.print("\n\n...Now the whole line will be erased...")
    term.write_styled("I'm going to disappear!", style=Style.parse("black on cyan"))
    time.sleep(1)
    term.erase_line()

    term.show_cursor()
    print("\n")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\h11\_connection.py ===
# This contains the main Connection class. Everything in h11 revolves around
# this.
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    overload,
    Tuple,
    Type,
    Union,
)

from ._events import (
    ConnectionClosed,
    Data,
    EndOfMessage,
    Event,
    InformationalResponse,
    Request,
    Response,
)
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
    _SWITCH_CONNECT,
    _SWITCH_UPGRADE,
    CLIENT,
    ConnectionState,
    DONE,
    ERROR,
    MIGHT_SWITCH_PROTOCOL,
    SEND_BODY,
    SERVER,
    SWITCHED_PROTOCOL,
)
from ._util import (  # Import the internal things we need
    LocalProtocolError,
    RemoteProtocolError,
    Sentinel,
)
from ._writers import WRITERS, WritersType

# Everything in __all__ gets re-exported as part of the h11 public API.
__all__ = ["Connection", "NEED_DATA", "PAUSED"]


class NEED_DATA(Sentinel, metaclass=Sentinel):
    pass


class PAUSED(Sentinel, metaclass=Sentinel):
    pass


# If we ever have this much buffered without it making a complete parseable
# event, we error out. The only time we really buffer is when reading the
# request/response line + headers together, so this is effectively the limit on
# the size of that.
#
# Some precedents for defaults:
# - node.js: 80 * 1024
# - tomcat: 8 * 1024
# - IIS: 16 * 1024
# - Apache: <8 KiB per line>
DEFAULT_MAX_INCOMPLETE_EVENT_SIZE = 16 * 1024


# RFC 7230's rules for connection lifecycles:
# - If either side says they want to close the connection, then the connection
#   must close.
# - HTTP/1.1 defaults to keep-alive unless someone says Connection: close
# - HTTP/1.0 defaults to close unless both sides say Connection: keep-alive
#   (and even this is a mess -- e.g. if you're implementing a proxy then
#   sending Connection: keep-alive is forbidden).
#
# We simplify life by simply not supporting keep-alive with HTTP/1.0 peers. So
# our rule is:
# - If someone says Connection: close, we will close
# - If someone uses HTTP/1.0, we will close.
def _keep_alive(event: Union[Request, Response]) -> bool:
    connection = get_comma_header(event.headers, b"connection")
    if b"close" in connection:
        return False
    if getattr(event, "http_version", b"1.1") < b"1.1":
        return False
    return True


def _body_framing(
    request_method: bytes, event: Union[Request, Response]
) -> Tuple[str, Union[Tuple[()], Tuple[int]]]:
    # Called when we enter SEND_BODY to figure out framing information for
    # this body.
    #
    # These are the only two events that can trigger a SEND_BODY state:
    assert type(event) in (Request, Response)
    # Returns one of:
    #
    #    ("content-length", count)
    #    ("chunked", ())
    #    ("http/1.0", ())
    #
    # which are (lookup key, *args) for constructing body reader/writer
    # objects.
    #
    # Reference: https://tools.ietf.org/html/rfc7230#section-3.3.3
    #
    # Step 1: some responses always have an empty body, regardless of what the
    # headers say.
    if type(event) is Response:
        if (
            event.status_code in (204, 304)
            or request_method == b"HEAD"
            or (request_method == b"CONNECT" and 200 <= event.status_code < 300)
        ):
            return ("content-length", (0,))
        # Section 3.3.3 also lists another case -- responses with status_code
        # < 200. For us these are InformationalResponses, not Responses, so
        # they can't get into this function in the first place.
        assert event.status_code >= 200

    # Step 2: check for Transfer-Encoding (T-E beats C-L):
    transfer_encodings = get_comma_header(event.headers, b"transfer-encoding")
    if transfer_encodings:
        assert transfer_encodings == [b"chunked"]
        return ("chunked", ())

    # Step 3: check for Content-Length
    content_lengths = get_comma_header(event.headers, b"content-length")
    if content_lengths:
        return ("content-length", (int(content_lengths[0]),))

    # Step 4: no applicable headers; fallback/default depends on type
    if type(event) is Request:
        return ("content-length", (0,))
    else:
        return ("http/1.0", ())


################################################################
#
# The main Connection class
#
################################################################


class Connection:
    """An object encapsulating the state of an HTTP connection.

    Args:
        our_role: If you're implementing a client, pass :data:`h11.CLIENT`. If
            you're implementing a server, pass :data:`h11.SERVER`.

        max_incomplete_event_size (int):
            The maximum number of bytes we're willing to buffer of an
            incomplete event. In practice this mostly sets a limit on the
            maximum size of the request/response line + headers. If this is
            exceeded, then :meth:`next_event` will raise
            :exc:`RemoteProtocolError`.

    """

    def __init__(
        self,
        our_role: Type[Sentinel],
        max_incomplete_event_size: int = DEFAULT_MAX_INCOMPLETE_EVENT_SIZE,
    ) -> None:
        self._max_incomplete_event_size = max_incomplete_event_size
        # State and role tracking
        if our_role not in (CLIENT, SERVER):
            raise ValueError(f"expected CLIENT or SERVER, not {our_role!r}")
        self.our_role = our_role
        self.their_role: Type[Sentinel]
        if our_role is CLIENT:
            self.their_role = SERVER
        else:
            self.their_role = CLIENT
        self._cstate = ConnectionState()

        # Callables for converting data->events or vice-versa given the
        # current state
        self._writer = self._get_io_object(self.our_role, None, WRITERS)
        self._reader = self._get_io_object(self.their_role, None, READERS)

        # Holds any unprocessed received data
        self._receive_buffer = ReceiveBuffer()
        # If this is true, then it indicates that the incoming connection was
        # closed *after* the end of whatever's in self._receive_buffer:
        self._receive_buffer_closed = False

        # Extra bits of state that don't fit into the state machine.
        #
        # These two are only used to interpret framing headers for figuring
        # out how to read/write response bodies. their_http_version is also
        # made available as a convenient public API.
        self.their_http_version: Optional[bytes] = None
        self._request_method: Optional[bytes] = None
        # This is pure flow-control and doesn't at all affect the set of legal
        # transitions, so no need to bother ConnectionState with it:
        self.client_is_waiting_for_100_continue = False

    @property
    def states(self) -> Dict[Type[Sentinel], Type[Sentinel]]:
        """A dictionary like::

           {CLIENT: <client state>, SERVER: <server state>}

        See :ref:`state-machine` for details.

        """
        return dict(self._cstate.states)

    @property
    def our_state(self) -> Type[Sentinel]:
        """The current state of whichever role we are playing. See
        :ref:`state-machine` for details.
        """
        return self._cstate.states[self.our_role]

    @property
    def their_state(self) -> Type[Sentinel]:
        """The current state of whichever role we are NOT playing. See
        :ref:`state-machine` for details.
        """
        return self._cstate.states[self.their_role]

    @property
    def they_are_waiting_for_100_continue(self) -> bool:
        return self.their_role is CLIENT and self.client_is_waiting_for_100_continue

    def start_next_cycle(self) -> None:
        """Attempt to reset our connection state for a new request/response
        cycle.

        If both client and server are in :data:`DONE` state, then resets them
        both to :data:`IDLE` state in preparation for a new request/response
        cycle on this same connection. Otherwise, raises a
        :exc:`LocalProtocolError`.

        See :ref:`keepalive-and-pipelining`.

        """
        old_states = dict(self._cstate.states)
        self._cstate.start_next_cycle()
        self._request_method = None
        # self.their_http_version gets left alone, since it presumably lasts
        # beyond a single request/response cycle
        assert not self.client_is_waiting_for_100_continue
        self._respond_to_state_changes(old_states)

    def _process_error(self, role: Type[Sentinel]) -> None:
        old_states = dict(self._cstate.states)
        self._cstate.process_error(role)
        self._respond_to_state_changes(old_states)

    def _server_switch_event(self, event: Event) -> Optional[Type[Sentinel]]:
        if type(event) is InformationalResponse and event.status_code == 101:
            return _SWITCH_UPGRADE
        if type(event) is Response:
            if (
                _SWITCH_CONNECT in self._cstate.pending_switch_proposals
                and 200 <= event.status_code < 300
            ):
                return _SWITCH_CONNECT
        return None

    # All events go through here
    def _process_event(self, role: Type[Sentinel], event: Event) -> None:
        # First, pass the event through the state machine to make sure it
        # succeeds.
        old_states = dict(self._cstate.states)
        if role is CLIENT and type(event) is Request:
            if event.method == b"CONNECT":
                self._cstate.process_client_switch_proposal(_SWITCH_CONNECT)
            if get_comma_header(event.headers, b"upgrade"):
                self._cstate.process_client_switch_proposal(_SWITCH_UPGRADE)
        server_switch_event = None
        if role is SERVER:
            server_switch_event = self._server_switch_event(event)
        self._cstate.process_event(role, type(event), server_switch_event)

        # Then perform the updates triggered by it.

        if type(event) is Request:
            self._request_method = event.method

        if role is self.their_role and type(event) in (
            Request,
            Response,
            InformationalResponse,
        ):
            event = cast(Union[Request, Response, InformationalResponse], event)
            self.their_http_version = event.http_version

        # Keep alive handling
        #
        # RFC 7230 doesn't really say what one should do if Connection: close
        # shows up on a 1xx InformationalResponse. I think the idea is that
        # this is not supposed to happen. In any case, if it does happen, we
        # ignore it.
        if type(event) in (Request, Response) and not _keep_alive(
            cast(Union[Request, Response], event)
        ):
            self._cstate.process_keep_alive_disabled()

        # 100-continue
        if type(event) is Request and has_expect_100_continue(event):
            self.client_is_waiting_for_100_continue = True
        if type(event) in (InformationalResponse, Response):
            self.client_is_waiting_for_100_continue = False
        if role is CLIENT and type(event) in (Data, EndOfMessage):
            self.client_is_waiting_for_100_continue = False

        self._respond_to_state_changes(old_states, event)

    def _get_io_object(
        self,
        role: Type[Sentinel],
        event: Optional[Event],
        io_dict: Union[ReadersType, WritersType],
    ) -> Optional[Callable[..., Any]]:
        # event may be None; it's only used when entering SEND_BODY
        state = self._cstate.states[role]
        if state is SEND_BODY:
            # Special case: the io_dict has a dict of reader/writer factories
            # that depend on the request/response framing.
            framing_type, args = _body_framing(
                cast(bytes, self._request_method), cast(Union[Request, Response], event)
            )
            return io_dict[SEND_BODY][framing_type](*args)  # type: ignore[index]
        else:
            # General case: the io_dict just has the appropriate reader/writer
            # for this state
            return io_dict.get((role, state))  # type: ignore[return-value]

    # This must be called after any action that might have caused
    # self._cstate.states to change.
    def _respond_to_state_changes(
        self,
        old_states: Dict[Type[Sentinel], Type[Sentinel]],
        event: Optional[Event] = None,
    ) -> None:
        # Update reader/writer
        if self.our_state != old_states[self.our_role]:
            self._writer = self._get_io_object(self.our_role, event, WRITERS)
        if self.their_state != old_states[self.their_role]:
            self._reader = self._get_io_object(self.their_role, event, READERS)

    @property
    def trailing_data(self) -> Tuple[bytes, bool]:
        """Data that has been received, but not yet processed, represented as
        a tuple with two elements, where the first is a byte-string containing
        the unprocessed data itself, and the second is a bool that is True if
        the receive connection was closed.

        See :ref:`switching-protocols` for discussion of why you'd want this.
        """
        return (bytes(self._receive_buffer), self._receive_buffer_closed)

    def receive_data(self, data: bytes) -> None:
        """Add data to our internal receive buffer.

        This does not actually do any processing on the data, just stores
        it. To trigger processing, you have to call :meth:`next_event`.

        Args:
            data (:term:`bytes-like object`):
                The new data that was just received.

                Special case: If *data* is an empty byte-string like ``b""``,
                then this indicates that the remote side has closed the
                connection (end of file). Normally this is convenient, because
                standard Python APIs like :meth:`file.read` or
                :meth:`socket.recv` use ``b""`` to indicate end-of-file, while
                other failures to read are indicated using other mechanisms
                like raising :exc:`TimeoutError`. When using such an API you
                can just blindly pass through whatever you get from ``read``
                to :meth:`receive_data`, and everything will work.

                But, if you have an API where reading an empty string is a
                valid non-EOF condition, then you need to be aware of this and
                make sure to check for such strings and avoid passing them to
                :meth:`receive_data`.

        Returns:
            Nothing, but after calling this you should call :meth:`next_event`
            to parse the newly received data.

        Raises:
            RuntimeError:
                Raised if you pass an empty *data*, indicating EOF, and then
                pass a non-empty *data*, indicating more data that somehow
                arrived after the EOF.

                (Calling ``receive_data(b"")`` multiple times is fine,
                and equivalent to calling it once.)

        """
        if data:
            if self._receive_buffer_closed:
                raise RuntimeError("received close, then received more data?")
            self._receive_buffer += data
        else:
            self._receive_buffer_closed = True

    def _extract_next_receive_event(
        self,
    ) -> Union[Event, Type[NEED_DATA], Type[PAUSED]]:
        state = self.their_state
        # We don't pause immediately when they enter DONE, because even in
        # DONE state we can still process a ConnectionClosed() event. But
        # if we have data in our buffer, then we definitely aren't getting
        # a ConnectionClosed() immediately and we need to pause.
        if state is DONE and self._receive_buffer:
            return PAUSED
        if state is MIGHT_SWITCH_PROTOCOL or state is SWITCHED_PROTOCOL:
            return PAUSED
        assert self._reader is not None
        event = self._reader(self._receive_buffer)
        if event is None:
            if not self._receive_buffer and self._receive_buffer_closed:
                # In some unusual cases (basically just HTTP/1.0 bodies), EOF
                # triggers an actual protocol event; in that case, we want to
                # return that event, and then the state will change and we'll
                # get called again to generate the actual ConnectionClosed().
                if hasattr(self._reader, "read_eof"):
                    event = self._reader.read_eof()
                else:
                    event = ConnectionClosed()
        if event is None:
            event = NEED_DATA
        return event  # type: ignore[no-any-return]

    def next_event(self) -> Union[Event, Type[NEED_DATA], Type[PAUSED]]:
        """Parse the next event out of our receive buffer, update our internal
        state, and return it.

        This is a mutating operation -- think of it like calling :func:`next`
        on an iterator.

        Returns:
            : One of three things:

            1) An event object -- see :ref:`events`.

            2) The special constant :data:`NEED_DATA`, which indicates that
               you need to read more data from your socket and pass it to
               :meth:`receive_data` before this method will be able to return
               any more events.

            3) The special constant :data:`PAUSED`, which indicates that we
               are not in a state where we can process incoming data (usually
               because the peer has finished their part of the current
               request/response cycle, and you have not yet called
               :meth:`start_next_cycle`). See :ref:`flow-control` for details.

        Raises:
            RemoteProtocolError:
                The peer has misbehaved. You should close the connection
                (possibly after sending some kind of 4xx response).

        Once this method returns :class:`ConnectionClosed` once, then all
        subsequent calls will also return :class:`ConnectionClosed`.

        If this method raises any exception besides :exc:`RemoteProtocolError`
        then that's a bug -- if it happens please file a bug report!

        If this method raises any exception then it also sets
        :attr:`Connection.their_state` to :data:`ERROR` -- see
        :ref:`error-handling` for discussion.

        """

        if self.their_state is ERROR:
            raise RemoteProtocolError("Can't receive data when peer state is ERROR")
        try:
            event = self._extract_next_receive_event()
            if event not in [NEED_DATA, PAUSED]:
                self._process_event(self.their_role, cast(Event, event))
            if event is NEED_DATA:
                if len(self._receive_buffer) > self._max_incomplete_event_size:
                    # 431 is "Request header fields too large" which is pretty
                    # much the only situation where we can get here
                    raise RemoteProtocolError(
                        "Receive buffer too long", error_status_hint=431
                    )
                if self._receive_buffer_closed:
                    # We're still trying to complete some event, but that's
                    # never going to happen because no more data is coming
                    raise RemoteProtocolError("peer unexpectedly closed connection")
            return event
        except BaseException as exc:
            self._process_error(self.their_role)
            if isinstance(exc, LocalProtocolError):
                exc._reraise_as_remote_protocol_error()
            else:
                raise

    @overload
    def send(self, event: ConnectionClosed) -> None:
        ...

    @overload
    def send(
        self, event: Union[Request, InformationalResponse, Response, Data, EndOfMessage]
    ) -> bytes:
        ...

    @overload
    def send(self, event: Event) -> Optional[bytes]:
        ...

    def send(self, event: Event) -> Optional[bytes]:
        """Convert a high-level event into bytes that can be sent to the peer,
        while updating our internal state machine.

        Args:
            event: The :ref:`event <events>` to send.

        Returns:
            If ``type(event) is ConnectionClosed``, then returns
            ``None``. Otherwise, returns a :term:`bytes-like object`.

        Raises:
            LocalProtocolError:
                Sending this event at this time would violate our
                understanding of the HTTP/1.1 protocol.

        If this method raises any exception then it also sets
        :attr:`Connection.our_state` to :data:`ERROR` -- see
        :ref:`error-handling` for discussion.

        """
        data_list = self.send_with_data_passthrough(event)
        if data_list is None:
            return None
        else:
            return b"".join(data_list)

    def send_with_data_passthrough(self, event: Event) -> Optional[List[bytes]]:
        """Identical to :meth:`send`, except that in situations where
        :meth:`send` returns a single :term:`bytes-like object`, this instead
        returns a list of them -- and when sending a :class:`Data` event, this
        list is guaranteed to contain the exact object you passed in as
        :attr:`Data.data`. See :ref:`sendfile` for discussion.

        """
        if self.our_state is ERROR:
            raise LocalProtocolError("Can't send data when our state is ERROR")
        try:
            if type(event) is Response:
                event = self._clean_up_response_headers_for_sending(event)
            # We want to call _process_event before calling the writer,
            # because if someone tries to do something invalid then this will
            # give a sensible error message, while our writers all just assume
            # they will only receive valid events. But, _process_event might
            # change self._writer. So we have to do a little dance:
            writer = self._writer
            self._process_event(self.our_role, event)
            if type(event) is ConnectionClosed:
                return None
            else:
                # In any situation where writer is None, process_event should
                # have raised ProtocolError
                assert writer is not None
                data_list: List[bytes] = []
                writer(event, data_list.append)
                return data_list
        except:
            self._process_error(self.our_role)
            raise

    def send_failed(self) -> None:
        """Notify the state machine that we failed to send the data it gave
        us.

        This causes :attr:`Connection.our_state` to immediately become
        :data:`ERROR` -- see :ref:`error-handling` for discussion.

        """
        self._process_error(self.our_role)

    # When sending a Response, we take responsibility for a few things:
    #
    # - Sometimes you MUST set Connection: close. We take care of those
    #   times. (You can also set it yourself if you want, and if you do then
    #   we'll respect that and close the connection at the right time. But you
    #   don't have to worry about that unless you want to.)
    #
    # - The user has to set Content-Length if they want it. Otherwise, for
    #   responses that have bodies (e.g. not HEAD), then we will automatically
    #   select the right mechanism for streaming a body of unknown length,
    #   which depends on depending on the peer's HTTP version.
    #
    # This function's *only* responsibility is making sure headers are set up
    # right -- everything downstream just looks at the headers. There are no
    # side channels.
    def _clean_up_response_headers_for_sending(self, response: Response) -> Response:
        assert type(response) is Response

        headers = response.headers
        need_close = False

        # HEAD requests need some special handling: they always act like they
        # have Content-Length: 0, and that's how _body_framing treats
        # them. But their headers are supposed to match what we would send if
        # the request was a GET. (Technically there is one deviation allowed:
        # we're allowed to leave out the framing headers -- see
        # https://tools.ietf.org/html/rfc7231#section-4.3.2 . But it's just as
        # easy to get them right.)
        method_for_choosing_headers = cast(bytes, self._request_method)
        if method_for_choosing_headers == b"HEAD":
            method_for_choosing_headers = b"GET"
        framing_type, _ = _body_framing(method_for_choosing_headers, response)
        if framing_type in ("chunked", "http/1.0"):
            # This response has a body of unknown length.
            # If our peer is HTTP/1.1, we use Transfer-Encoding: chunked
            # If our peer is HTTP/1.0, we use no framing headers, and close the
            # connection afterwards.
            #
            # Make sure to clear Content-Length (in principle user could have
            # set both and then we ignored Content-Length b/c
            # Transfer-Encoding overwrote it -- this would be naughty of them,
            # but the HTTP spec says that if our peer does this then we have
            # to fix it instead of erroring out, so we'll accord the user the
            # same respect).
            headers = set_comma_header(headers, b"content-length", [])
            if self.their_http_version is None or self.their_http_version < b"1.1":
                # Either we never got a valid request and are sending back an
                # error (their_http_version is None), so we assume the worst;
                # or else we did get a valid HTTP/1.0 request, so we know that
                # they don't understand chunked encoding.
                headers = set_comma_header(headers, b"transfer-encoding", [])
                # This is actually redundant ATM, since currently we
                # unconditionally disable keep-alive when talking to HTTP/1.0
                # peers. But let's be defensive just in case we add
                # Connection: keep-alive support later:
                if self._request_method != b"HEAD":
                    need_close = True
            else:
                headers = set_comma_header(headers, b"transfer-encoding", [b"chunked"])

        if not self._cstate.keep_alive or need_close:
            # Make sure Connection: close is set
            connection = set(get_comma_header(headers, b"connection"))
            connection.discard(b"keep-alive")
            connection.add(b"close")
            headers = set_comma_header(headers, b"connection", sorted(connection))

        return Response(
            headers=headers,
            status_code=response.status_code,
            http_version=response.http_version,
            reason=response.reason,
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\fnodes.py ===
"""
AST nodes specific to Fortran.

The functions defined in this module allows the user to express functions such as ``dsign``
as a SymPy function for symbolic manipulation.
"""

from __future__ import annotations
from sympy.codegen.ast import (
    Attribute, CodeBlock, FunctionCall, Node, none, String,
    Token, _mk_Tuple, Variable
)
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Str
from sympy.core.sympify import sympify
from sympy.logic import true, false
from sympy.utilities.iterables import iterable



pure = Attribute('pure')
elemental = Attribute('elemental')  # (all elemental procedures are also pure)

intent_in = Attribute('intent_in')
intent_out = Attribute('intent_out')
intent_inout = Attribute('intent_inout')

allocatable = Attribute('allocatable')

class Program(Token):
    """ Represents a 'program' block in Fortran.

    Examples
    ========

    >>> from sympy.codegen.ast import Print
    >>> from sympy.codegen.fnodes import Program
    >>> prog = Program('myprogram', [Print([42])])
    >>> from sympy import fcode
    >>> print(fcode(prog, source_format='free'))
    program myprogram
        print *, 42
    end program

    """
    __slots__ = _fields = ('name', 'body')
    _construct_name = String
    _construct_body = staticmethod(lambda body: CodeBlock(*body))


class use_rename(Token):
    """ Represents a renaming in a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use_rename, use
    >>> from sympy import fcode
    >>> ren = use_rename("thingy", "convolution2d")
    >>> print(fcode(ren, source_format='free'))
    thingy => convolution2d
    >>> full = use('signallib', only=['snr', ren])
    >>> print(fcode(full, source_format='free'))
    use signallib, only: snr, thingy => convolution2d

    """
    __slots__ = _fields = ('local', 'original')
    _construct_local = String
    _construct_original = String

def _name(arg):
    if hasattr(arg, 'name'):
        return arg.name
    else:
        return String(arg)

class use(Token):
    """ Represents a use statement in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import use
    >>> from sympy import fcode
    >>> fcode(use('signallib'), source_format='free')
    'use signallib'
    >>> fcode(use('signallib', [('metric', 'snr')]), source_format='free')
    'use signallib, metric => snr'
    >>> fcode(use('signallib', only=['snr', 'convolution2d']), source_format='free')
    'use signallib, only: snr, convolution2d'

    """
    __slots__ = _fields = ('namespace', 'rename', 'only')
    defaults = {'rename': none, 'only': none}
    _construct_namespace = staticmethod(_name)
    _construct_rename = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else use_rename(*arg) for arg in args]))
    _construct_only = staticmethod(lambda args: Tuple(*[arg if isinstance(arg, use_rename) else _name(arg) for arg in args]))


class Module(Token):
    """ Represents a module in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Module
    >>> from sympy import fcode
    >>> print(fcode(Module('signallib', ['implicit none'], []), source_format='free'))
    module signallib
    implicit none
    <BLANKLINE>
    contains
    <BLANKLINE>
    <BLANKLINE>
    end module

    """
    __slots__ = _fields = ('name', 'declarations', 'definitions')
    defaults = {'declarations': Tuple()}
    _construct_name = String

    @classmethod
    def _construct_declarations(cls, args):
        args = [Str(arg) if isinstance(arg, str) else arg for arg in args]
        return CodeBlock(*args)

    _construct_definitions = staticmethod(lambda arg: CodeBlock(*arg))


class Subroutine(Node):
    """ Represents a subroutine in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import Print
    >>> from sympy.codegen.fnodes import Subroutine
    >>> x, y = symbols('x y', real=True)
    >>> sub = Subroutine('mysub', [x, y], [Print([x**2 + y**2, x*y])])
    >>> print(fcode(sub, source_format='free', standard=2003))
    subroutine mysub(x, y)
    real*8 :: x
    real*8 :: y
    print *, x**2 + y**2, x*y
    end subroutine

    """
    __slots__ = ('name', 'parameters', 'body')
    _fields = __slots__ + Node._fields
    _construct_name = String
    _construct_parameters = staticmethod(lambda params: Tuple(*map(Variable.deduced, params)))

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)

class SubroutineCall(Token):
    """ Represents a call to a subroutine in Fortran.

    Examples
    ========

    >>> from sympy.codegen.fnodes import SubroutineCall
    >>> from sympy import fcode
    >>> fcode(SubroutineCall('mysub', 'x y'.split()))
    '       call mysub(x, y)'

    """
    __slots__ = _fields = ('name', 'subroutine_args')
    _construct_name = staticmethod(_name)
    _construct_subroutine_args = staticmethod(_mk_Tuple)


class Do(Token):
    """ Represents a Do loop in in Fortran.

    Examples
    ========

    >>> from sympy import fcode, symbols
    >>> from sympy.codegen.ast import aug_assign, Print
    >>> from sympy.codegen.fnodes import Do
    >>> i, n = symbols('i n', integer=True)
    >>> r = symbols('r', real=True)
    >>> body = [aug_assign(r, '+', 1/i), Print([i, r])]
    >>> do1 = Do(body, i, 1, n)
    >>> print(fcode(do1, source_format='free'))
    do i = 1, n
        r = r + 1d0/i
        print *, i, r
    end do
    >>> do2 = Do(body, i, 1, n, 2)
    >>> print(fcode(do2, source_format='free'))
    do i = 1, n, 2
        r = r + 1d0/i
        print *, i, r
    end do

    """

    __slots__ = _fields = ('body', 'counter', 'first', 'last', 'step', 'concurrent')
    defaults = {'step': Integer(1), 'concurrent': false}
    _construct_body = staticmethod(lambda body: CodeBlock(*body))
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)
    _construct_concurrent = staticmethod(lambda arg: true if arg else false)


class ArrayConstructor(Token):
    """ Represents an array constructor.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import ArrayConstructor
    >>> ac = ArrayConstructor([1, 2, 3])
    >>> fcode(ac, standard=95, source_format='free')
    '(/1, 2, 3/)'
    >>> fcode(ac, standard=2003, source_format='free')
    '[1, 2, 3]'

    """
    __slots__ = _fields = ('elements',)
    _construct_elements = staticmethod(_mk_Tuple)


class ImpliedDoLoop(Token):
    """ Represents an implied do loop in Fortran.

    Examples
    ========

    >>> from sympy import Symbol, fcode
    >>> from sympy.codegen.fnodes import ImpliedDoLoop, ArrayConstructor
    >>> i = Symbol('i', integer=True)
    >>> idl = ImpliedDoLoop(i**3, i, -3, 3, 2)  # -27, -1, 1, 27
    >>> ac = ArrayConstructor([-28, idl, 28]) # -28, -27, -1, 1, 27, 28
    >>> fcode(ac, standard=2003, source_format='free')
    '[-28, (i**3, i = -3, 3, 2), 28]'

    """
    __slots__ = _fields = ('expr', 'counter', 'first', 'last', 'step')
    defaults = {'step': Integer(1)}
    _construct_expr = staticmethod(sympify)
    _construct_counter = staticmethod(sympify)
    _construct_first = staticmethod(sympify)
    _construct_last = staticmethod(sympify)
    _construct_step = staticmethod(sympify)


class Extent(Basic):
    """ Represents a dimension extent.

    Examples
    ========

    >>> from sympy.codegen.fnodes import Extent
    >>> e = Extent(-3, 3)  # -3, -2, -1, 0, 1, 2, 3
    >>> from sympy import fcode
    >>> fcode(e, source_format='free')
    '-3:3'
    >>> from sympy.codegen.ast import Variable, real
    >>> from sympy.codegen.fnodes import dimension, intent_out
    >>> dim = dimension(e, e)
    >>> arr = Variable('x', real, attrs=[dim, intent_out])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'real*8, dimension(-3:3, -3:3), intent(out) :: x'

    """
    def __new__(cls, *args):
        if len(args) == 2:
            low, high = args
            return Basic.__new__(cls, sympify(low), sympify(high))
        elif len(args) == 0 or (len(args) == 1 and args[0] in (':', None)):
            return Basic.__new__(cls)  # assumed shape
        else:
            raise ValueError("Expected 0 or 2 args (or one argument == None or ':')")

    def _sympystr(self, printer):
        if len(self.args) == 0:
            return ':'
        return ":".join(str(arg) for arg in self.args)

assumed_extent = Extent() # or Extent(':'), Extent(None)


def dimension(*args):
    """ Creates a 'dimension' Attribute with (up to 7) extents.

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import dimension, intent_in
    >>> dim = dimension('2', ':')  # 2 rows, runtime determined number of columns
    >>> from sympy.codegen.ast import Variable, integer
    >>> arr = Variable('a', integer, attrs=[dim, intent_in])
    >>> fcode(arr.as_Declaration(), source_format='free', standard=2003)
    'integer*4, dimension(2, :), intent(in) :: a'

    """
    if len(args) > 7:
        raise ValueError("Fortran only supports up to 7 dimensional arrays")
    parameters = []
    for arg in args:
        if isinstance(arg, Extent):
            parameters.append(arg)
        elif isinstance(arg, str):
            if arg == ':':
                parameters.append(Extent())
            else:
                parameters.append(String(arg))
        elif iterable(arg):
            parameters.append(Extent(*arg))
        else:
            parameters.append(sympify(arg))
    if len(args) == 0:
        raise ValueError("Need at least one dimension")
    return Attribute('dimension', parameters)


assumed_size = dimension('*')

def array(symbol, dim, intent=None, *, attrs=(), value=None, type=None):
    """ Convenience function for creating a Variable instance for a Fortran array.

    Parameters
    ==========

    symbol : symbol
    dim : Attribute or iterable
        If dim is an ``Attribute`` it need to have the name 'dimension'. If it is
        not an ``Attribute``, then it is passed to :func:`dimension` as ``*dim``
    intent : str
        One of: 'in', 'out', 'inout' or None
    \\*\\*kwargs:
        Keyword arguments for ``Variable`` ('type' & 'value')

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.ast import integer, real
    >>> from sympy.codegen.fnodes import array
    >>> arr = array('a', '*', 'in', type=integer)
    >>> print(fcode(arr.as_Declaration(), source_format='free', standard=2003))
    integer*4, dimension(*), intent(in) :: a
    >>> x = array('x', [3, ':', ':'], intent='out', type=real)
    >>> print(fcode(x.as_Declaration(value=1), source_format='free', standard=2003))
    real*8, dimension(3, :, :), intent(out) :: x = 1

    """
    if isinstance(dim, Attribute):
        if str(dim.name) != 'dimension':
            raise ValueError("Got an unexpected Attribute argument as dim: %s" % str(dim))
    else:
        dim = dimension(*dim)

    attrs = list(attrs) + [dim]
    if intent is not None:
        if intent not in (intent_in, intent_out, intent_inout):
            intent = {'in': intent_in, 'out': intent_out, 'inout': intent_inout}[intent]
        attrs.append(intent)
    if type is None:
        return Variable.deduced(symbol, value=value, attrs=attrs)
    else:
        return Variable(symbol, type, value=value, attrs=attrs)

def _printable(arg):
    return String(arg) if isinstance(arg, str) else sympify(arg)


def allocated(array):
    """ Creates an AST node for a function call to Fortran's "allocated(...)"

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import allocated
    >>> alloc = allocated('x')
    >>> fcode(alloc, source_format='free')
    'allocated(x)'

    """
    return FunctionCall('allocated', [_printable(array)])


def lbound(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "lbound(...)"

    Parameters
    ==========

    array : Symbol or String
    dim : expr
    kind : expr

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import lbound
    >>> lb = lbound('arr', dim=2)
    >>> fcode(lb, source_format='free')
    'lbound(arr, 2)'

    """
    return FunctionCall(
        'lbound',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def ubound(array, dim=None, kind=None):
    return FunctionCall(
        'ubound',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def shape(source, kind=None):
    """ Creates an AST node for a function call to Fortran's "shape(...)"

    Parameters
    ==========

    source : Symbol or String
    kind : expr

    Examples
    ========

    >>> from sympy import fcode
    >>> from sympy.codegen.fnodes import shape
    >>> shp = shape('x')
    >>> fcode(shp, source_format='free')
    'shape(x)'

    """
    return FunctionCall(
        'shape',
        [_printable(source)] +
        ([_printable(kind)] if kind else [])
    )


def size(array, dim=None, kind=None):
    """ Creates an AST node for a function call to Fortran's "size(...)"

    Examples
    ========

    >>> from sympy import fcode, Symbol
    >>> from sympy.codegen.ast import FunctionDefinition, real, Return
    >>> from sympy.codegen.fnodes import array, sum_, size
    >>> a = Symbol('a', real=True)
    >>> body = [Return((sum_(a**2)/size(a))**.5)]
    >>> arr = array(a, dim=[':'], intent='in')
    >>> fd = FunctionDefinition(real, 'rms', [arr], body)
    >>> print(fcode(fd, source_format='free', standard=2003))
    real*8 function rms(a)
    real*8, dimension(:), intent(in) :: a
    rms = sqrt(sum(a**2)*1d0/size(a))
    end function

    """
    return FunctionCall(
        'size',
        [_printable(array)] +
        ([_printable(dim)] if dim else []) +
        ([_printable(kind)] if kind else [])
    )


def reshape(source, shape, pad=None, order=None):
    """ Creates an AST node for a function call to Fortran's "reshape(...)"

    Parameters
    ==========

    source : Symbol or String
    shape : ArrayExpr

    """
    return FunctionCall(
        'reshape',
        [_printable(source), _printable(shape)] +
        ([_printable(pad)] if pad else []) +
        ([_printable(order)] if pad else [])
    )


def bind_C(name=None):
    """ Creates an Attribute ``bind_C`` with a name.

    Parameters
    ==========

    name : str

    Examples
    ========

    >>> from sympy import fcode, Symbol
    >>> from sympy.codegen.ast import FunctionDefinition, real, Return
    >>> from sympy.codegen.fnodes import array, sum_, bind_C
    >>> a = Symbol('a', real=True)
    >>> s = Symbol('s', integer=True)
    >>> arr = array(a, dim=[s], intent='in')
    >>> body = [Return((sum_(a**2)/s)**.5)]
    >>> fd = FunctionDefinition(real, 'rms', [arr, s], body, attrs=[bind_C('rms')])
    >>> print(fcode(fd, source_format='free', standard=2003))
    real*8 function rms(a, s) bind(C, name="rms")
    real*8, dimension(s), intent(in) :: a
    integer*4 :: s
    rms = sqrt(sum(a**2)/s)
    end function

    """
    return Attribute('bind_C', [String(name)] if name else [])

class GoTo(Token):
    """ Represents a goto statement in Fortran

    Examples
    ========

    >>> from sympy.codegen.fnodes import GoTo
    >>> go = GoTo([10, 20, 30], 'i')
    >>> from sympy import fcode
    >>> fcode(go, source_format='free')
    'go to (10, 20, 30), i'

    """
    __slots__ = _fields = ('labels', 'expr')
    defaults = {'expr': none}
    _construct_labels = staticmethod(_mk_Tuple)
    _construct_expr = staticmethod(sympify)


class FortranReturn(Token):
    """ AST node explicitly mapped to a fortran "return".

    Explanation
    ===========

    Because a return statement in fortran is different from C, and
    in order to aid reuse of our codegen ASTs the ordinary
    ``.codegen.ast.Return`` is interpreted as assignment to
    the result variable of the function. If one for some reason needs
    to generate a fortran RETURN statement, this node should be used.

    Examples
    ========

    >>> from sympy.codegen.fnodes import FortranReturn
    >>> from sympy import fcode
    >>> fcode(FortranReturn('x'))
    '       return x'

    """
    __slots__ = _fields = ('return_value',)
    defaults = {'return_value': none}
    _construct_return_value = staticmethod(sympify)


class FFunction(Function):
    _required_standard = 77

    def _fcode(self, printer):
        name = self.__class__.__name__
        if printer._settings['standard'] < self._required_standard:
            raise NotImplementedError("%s requires Fortran %d or newer" %
                                      (name, self._required_standard))
        return '{}({})'.format(name, ', '.join(map(printer._print, self.args)))


class F95Function(FFunction):
    _required_standard = 95


class isign(FFunction):
    """ Fortran sign intrinsic for integer arguments. """
    nargs = 2


class dsign(FFunction):
    """ Fortran sign intrinsic for double precision arguments. """
    nargs = 2


class cmplx(FFunction):
    """ Fortran complex conversion function. """
    nargs = 2  # may be extended to (2, 3) at a later point


class kind(FFunction):
    """ Fortran kind function. """
    nargs = 1


class merge(F95Function):
    """ Fortran merge function """
    nargs = 3


class _literal(Float):
    _token: str
    _decimals: int

    def _fcode(self, printer, *args, **kwargs):
        mantissa, sgnd_ex = ('%.{}e'.format(self._decimals) % self).split('e')
        mantissa = mantissa.strip('0').rstrip('.')
        ex_sgn, ex_num = sgnd_ex[0], sgnd_ex[1:].lstrip('0')
        ex_sgn = '' if ex_sgn == '+' else ex_sgn
        return (mantissa or '0') + self._token + ex_sgn + (ex_num or '0')


class literal_sp(_literal):
    """ Fortran single precision real literal """
    _token = 'e'
    _decimals = 9


class literal_dp(_literal):
    """ Fortran double precision real literal """
    _token = 'd'
    _decimals = 17


class sum_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)


class product_(Token, Expr):
    __slots__ = _fields = ('array', 'dim', 'mask')
    defaults = {'dim': none, 'mask': none}
    _construct_array = staticmethod(sympify)
    _construct_dim = staticmethod(sympify)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\common.py ===
"""
Misc tools for implementing data structures

Note: pandas.core.common is *not* part of the public API.
"""
from __future__ import annotations

import builtins
from collections import (
    abc,
    defaultdict,
)
from collections.abc import (
    Collection,
    Generator,
    Hashable,
    Iterable,
    Sequence,
)
import contextlib
from functools import partial
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.compat.numpy import np_version_gte1p24

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
)
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.inference import iterable_not_string

if TYPE_CHECKING:
    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        NpDtype,
        RandomState,
        T,
    )

    from pandas import Index


def flatten(line):
    """
    Flatten an arbitrarily nested sequence.

    Parameters
    ----------
    line : sequence
        The non string sequence to flatten

    Notes
    -----
    This doesn't consider strings sequences.

    Returns
    -------
    flattened : generator
    """
    for element in line:
        if iterable_not_string(element):
            yield from flatten(element)
        else:
            yield element


def consensus_name_attr(objs):
    name = objs[0].name
    for obj in objs[1:]:
        try:
            if obj.name != name:
                name = None
        except ValueError:
            name = None
    return name


def is_bool_indexer(key: Any) -> bool:
    """
    Check whether `key` is a valid boolean indexer.

    Parameters
    ----------
    key : Any
        Only list-likes may be considered boolean indexers.
        All other types are not considered a boolean indexer.
        For array-like input, boolean ndarrays or ExtensionArrays
        with ``_is_boolean`` set are considered boolean indexers.

    Returns
    -------
    bool
        Whether `key` is a valid boolean indexer.

    Raises
    ------
    ValueError
        When the array is an object-dtype ndarray or ExtensionArray
        and contains missing values.

    See Also
    --------
    check_array_indexer : Check that `key` is a valid array to index,
        and convert to an ndarray.
    """
    if isinstance(
        key, (ABCSeries, np.ndarray, ABCIndex, ABCExtensionArray)
    ) and not isinstance(key, ABCMultiIndex):
        if key.dtype == np.object_:
            key_array = np.asarray(key)

            if not lib.is_bool_array(key_array):
                na_msg = "Cannot mask with non-boolean array containing NA / NaN values"
                if lib.is_bool_array(key_array, skipna=True):
                    # Don't raise on e.g. ["A", "B", np.nan], see
                    #  test_loc_getitem_list_of_labels_categoricalindex_with_na
                    raise ValueError(na_msg)
                return False
            return True
        elif is_bool_dtype(key.dtype):
            return True
    elif isinstance(key, list):
        # check if np.array(key).dtype would be bool
        if len(key) > 0:
            if type(key) is not list:  # noqa: E721
                # GH#42461 cython will raise TypeError if we pass a subclass
                key = list(key)
            return lib.is_bool_list(key)

    return False


def cast_scalar_indexer(val):
    """
    Disallow indexing with a float key, even if that key is a round number.

    Parameters
    ----------
    val : scalar

    Returns
    -------
    outval : scalar
    """
    # assumes lib.is_scalar(val)
    if lib.is_float(val) and val.is_integer():
        raise IndexError(
            # GH#34193
            "Indexing with a float is no longer supported. Manually convert "
            "to an integer key instead."
        )
    return val


def not_none(*args):
    """
    Returns a generator consisting of the arguments that are not None.
    """
    return (arg for arg in args if arg is not None)


def any_none(*args) -> bool:
    """
    Returns a boolean indicating if any argument is None.
    """
    return any(arg is None for arg in args)


def all_none(*args) -> bool:
    """
    Returns a boolean indicating if all arguments are None.
    """
    return all(arg is None for arg in args)


def any_not_none(*args) -> bool:
    """
    Returns a boolean indicating if any argument is not None.
    """
    return any(arg is not None for arg in args)


def all_not_none(*args) -> bool:
    """
    Returns a boolean indicating if all arguments are not None.
    """
    return all(arg is not None for arg in args)


def count_not_none(*args) -> int:
    """
    Returns the count of arguments that are not None.
    """
    return sum(x is not None for x in args)


@overload
def asarray_tuplesafe(
    values: ArrayLike | list | tuple | zip, dtype: NpDtype | None = ...
) -> np.ndarray:
    # ExtensionArray can only be returned when values is an Index, all other iterables
    # will return np.ndarray. Unfortunately "all other" cannot be encoded in a type
    # signature, so instead we special-case some common types.
    ...


@overload
def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = ...) -> ArrayLike:
    ...


def asarray_tuplesafe(values: Iterable, dtype: NpDtype | None = None) -> ArrayLike:
    if not (isinstance(values, (list, tuple)) or hasattr(values, "__array__")):
        values = list(values)
    elif isinstance(values, ABCIndex):
        return values._values
    elif isinstance(values, ABCSeries):
        return values._values

    if isinstance(values, list) and dtype in [np.object_, object]:
        return construct_1d_object_array_from_listlike(values)

    try:
        with warnings.catch_warnings():
            # Can remove warning filter once NumPy 1.24 is min version
            if not np_version_gte1p24:
                warnings.simplefilter("ignore", np.VisibleDeprecationWarning)
            result = np.asarray(values, dtype=dtype)
    except ValueError:
        # Using try/except since it's more performant than checking is_list_like
        # over each element
        # error: Argument 1 to "construct_1d_object_array_from_listlike"
        # has incompatible type "Iterable[Any]"; expected "Sized"
        return construct_1d_object_array_from_listlike(values)  # type: ignore[arg-type]

    if issubclass(result.dtype.type, str):
        result = np.asarray(values, dtype=object)

    if result.ndim == 2:
        # Avoid building an array of arrays:
        values = [tuple(x) for x in values]
        result = construct_1d_object_array_from_listlike(values)

    return result


def index_labels_to_array(
    labels: np.ndarray | Iterable, dtype: NpDtype | None = None
) -> np.ndarray:
    """
    Transform label or iterable of labels to array, for use in Index.

    Parameters
    ----------
    dtype : dtype
        If specified, use as dtype of the resulting array, otherwise infer.

    Returns
    -------
    array
    """
    if isinstance(labels, (str, tuple)):
        labels = [labels]

    if not isinstance(labels, (list, np.ndarray)):
        try:
            labels = list(labels)
        except TypeError:  # non-iterable
            labels = [labels]

    labels = asarray_tuplesafe(labels, dtype=dtype)

    return labels


def maybe_make_list(obj):
    if obj is not None and not isinstance(obj, (tuple, list)):
        return [obj]
    return obj


def maybe_iterable_to_list(obj: Iterable[T] | T) -> Collection[T] | T:
    """
    If obj is Iterable but not list-like, consume into list.
    """
    if isinstance(obj, abc.Iterable) and not isinstance(obj, abc.Sized):
        return list(obj)
    obj = cast(Collection, obj)
    return obj


def is_null_slice(obj) -> bool:
    """
    We have a null slice.
    """
    return (
        isinstance(obj, slice)
        and obj.start is None
        and obj.stop is None
        and obj.step is None
    )


def is_empty_slice(obj) -> bool:
    """
    We have an empty slice, e.g. no values are selected.
    """
    return (
        isinstance(obj, slice)
        and obj.start is not None
        and obj.stop is not None
        and obj.start == obj.stop
    )


def is_true_slices(line) -> list[bool]:
    """
    Find non-trivial slices in "line": return a list of booleans with same length.
    """
    return [isinstance(k, slice) and not is_null_slice(k) for k in line]


# TODO: used only once in indexing; belongs elsewhere?
def is_full_slice(obj, line: int) -> bool:
    """
    We have a full length slice.
    """
    return (
        isinstance(obj, slice)
        and obj.start == 0
        and obj.stop == line
        and obj.step is None
    )


def get_callable_name(obj):
    # typical case has name
    if hasattr(obj, "__name__"):
        return getattr(obj, "__name__")
    # some objects don't; could recurse
    if isinstance(obj, partial):
        return get_callable_name(obj.func)
    # fall back to class name
    if callable(obj):
        return type(obj).__name__
    # everything failed (probably because the argument
    # wasn't actually callable); we return None
    # instead of the empty string in this case to allow
    # distinguishing between no name and a name of ''
    return None


def apply_if_callable(maybe_callable, obj, **kwargs):
    """
    Evaluate possibly callable input using obj and kwargs if it is callable,
    otherwise return as it is.

    Parameters
    ----------
    maybe_callable : possibly a callable
    obj : NDFrame
    **kwargs
    """
    if callable(maybe_callable):
        return maybe_callable(obj, **kwargs)

    return maybe_callable


def standardize_mapping(into):
    """
    Helper function to standardize a supplied mapping.

    Parameters
    ----------
    into : instance or subclass of collections.abc.Mapping
        Must be a class, an initialized collections.defaultdict,
        or an instance of a collections.abc.Mapping subclass.

    Returns
    -------
    mapping : a collections.abc.Mapping subclass or other constructor
        a callable object that can accept an iterator to create
        the desired Mapping.

    See Also
    --------
    DataFrame.to_dict
    Series.to_dict
    """
    if not inspect.isclass(into):
        if isinstance(into, defaultdict):
            return partial(defaultdict, into.default_factory)
        into = type(into)
    if not issubclass(into, abc.Mapping):
        raise TypeError(f"unsupported type: {into}")
    if into == defaultdict:
        raise TypeError("to_dict() only accepts initialized defaultdicts")
    return into


@overload
def random_state(state: np.random.Generator) -> np.random.Generator:
    ...


@overload
def random_state(
    state: int | np.ndarray | np.random.BitGenerator | np.random.RandomState | None,
) -> np.random.RandomState:
    ...


def random_state(state: RandomState | None = None):
    """
    Helper function for processing random_state arguments.

    Parameters
    ----------
    state : int, array-like, BitGenerator, Generator, np.random.RandomState, None.
        If receives an int, array-like, or BitGenerator, passes to
        np.random.RandomState() as seed.
        If receives an np.random RandomState or Generator, just returns that unchanged.
        If receives `None`, returns np.random.
        If receives anything else, raises an informative ValueError.

        Default None.

    Returns
    -------
    np.random.RandomState or np.random.Generator. If state is None, returns np.random

    """
    if is_integer(state) or isinstance(state, (np.ndarray, np.random.BitGenerator)):
        return np.random.RandomState(state)
    elif isinstance(state, np.random.RandomState):
        return state
    elif isinstance(state, np.random.Generator):
        return state
    elif state is None:
        return np.random
    else:
        raise ValueError(
            "random_state must be an integer, array-like, a BitGenerator, Generator, "
            "a numpy RandomState, or None"
        )


def pipe(
    obj, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
) -> T:
    """
    Apply a function ``func`` to object ``obj`` either by passing obj as the
    first argument to the function or, in the case that the func is a tuple,
    interpret the first element of the tuple as a function and pass the obj to
    that function as a keyword argument whose key is the value of the second
    element of the tuple.

    Parameters
    ----------
    func : callable or tuple of (callable, str)
        Function to apply to this object or, alternatively, a
        ``(callable, data_keyword)`` tuple where ``data_keyword`` is a
        string indicating the keyword of ``callable`` that expects the
        object.
    *args : iterable, optional
        Positional arguments passed into ``func``.
    **kwargs : dict, optional
        A dictionary of keyword arguments passed into ``func``.

    Returns
    -------
    object : the return type of ``func``.
    """
    if isinstance(func, tuple):
        func, target = func
        if target in kwargs:
            msg = f"{target} is both the pipe target and a keyword argument"
            raise ValueError(msg)
        kwargs[target] = obj
        return func(*args, **kwargs)
    else:
        return func(obj, *args, **kwargs)


def get_rename_function(mapper):
    """
    Returns a function that will map names/labels, dependent if mapper
    is a dict, Series or just a function.
    """

    def f(x):
        if x in mapper:
            return mapper[x]
        else:
            return x

    return f if isinstance(mapper, (abc.Mapping, ABCSeries)) else mapper


def convert_to_list_like(
    values: Hashable | Iterable | AnyArrayLike,
) -> list | AnyArrayLike:
    """
    Convert list-like or scalar input to list-like. List, numpy and pandas array-like
    inputs are returned unmodified whereas others are converted to list.
    """
    if isinstance(values, (list, np.ndarray, ABCIndex, ABCSeries, ABCExtensionArray)):
        return values
    elif isinstance(values, abc.Iterable) and not isinstance(values, str):
        return list(values)

    return [values]


@contextlib.contextmanager
def temp_setattr(
    obj, attr: str, value, condition: bool = True
) -> Generator[None, None, None]:
    """
    Temporarily set attribute on an object.

    Parameters
    ----------
    obj : object
        Object whose attribute will be modified.
    attr : str
        Attribute to modify.
    value : Any
        Value to temporarily set attribute to.
    condition : bool, default True
        Whether to set the attribute. Provided in order to not have to
        conditionally use this context manager.

    Yields
    ------
    object : obj with modified attribute.
    """
    if condition:
        old_value = getattr(obj, attr)
        setattr(obj, attr, value)
    try:
        yield obj
    finally:
        if condition:
            setattr(obj, attr, old_value)


def require_length_match(data, index: Index) -> None:
    """
    Check the length of data matches the length of the index.
    """
    if len(data) != len(index):
        raise ValueError(
            "Length of values "
            f"({len(data)}) "
            "does not match length of index "
            f"({len(index)})"
        )


# the ufuncs np.maximum.reduce and np.minimum.reduce default to axis=0,
#  whereas np.min and np.max (which directly call obj.min and obj.max)
#  default to axis=None.
_builtin_table = {
    builtins.sum: np.sum,
    builtins.max: np.maximum.reduce,
    builtins.min: np.minimum.reduce,
}

# GH#53425: Only for deprecation
_builtin_table_alias = {
    builtins.sum: "np.sum",
    builtins.max: "np.maximum.reduce",
    builtins.min: "np.minimum.reduce",
}

_cython_table = {
    builtins.sum: "sum",
    builtins.max: "max",
    builtins.min: "min",
    np.all: "all",
    np.any: "any",
    np.sum: "sum",
    np.nansum: "sum",
    np.mean: "mean",
    np.nanmean: "mean",
    np.prod: "prod",
    np.nanprod: "prod",
    np.std: "std",
    np.nanstd: "std",
    np.var: "var",
    np.nanvar: "var",
    np.median: "median",
    np.nanmedian: "median",
    np.max: "max",
    np.nanmax: "max",
    np.min: "min",
    np.nanmin: "min",
    np.cumprod: "cumprod",
    np.nancumprod: "cumprod",
    np.cumsum: "cumsum",
    np.nancumsum: "cumsum",
}


def get_cython_func(arg: Callable) -> str | None:
    """
    if we define an internal function for this argument, return it
    """
    return _cython_table.get(arg)


def is_builtin_func(arg):
    """
    if we define a builtin function for this argument, return it,
    otherwise return the arg
    """
    return _builtin_table.get(arg, arg)


def fill_missing_names(names: Sequence[Hashable | None]) -> list[Hashable]:
    """
    If a name is missing then replace it by level_n, where n is the count

    .. versionadded:: 1.4.0

    Parameters
    ----------
    names : list-like
        list of column names or None values.

    Returns
    -------
    list
        list of column names with the None values replaced.
    """
    return [f"level_{i}" if name is None else name for i, name in enumerate(names)]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\_compilation\compilation.py ===
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from sysconfig import get_config_var, get_config_vars, get_path

from .runners import (
    CCompilerRunner,
    CppCompilerRunner,
    FortranCompilerRunner
)
from .util import (
    get_abspath, make_dirs, copy, Glob, ArbitraryDepthGlob,
    glob_at_depth, import_module_from_file, pyx_is_cplus,
    sha256_of_string, sha256_of_file, CompileError
)

if os.name == 'posix':
    objext = '.o'
elif os.name == 'nt':
    objext = '.obj'
else:
    warnings.warn("Unknown os.name: {}".format(os.name))
    objext = '.o'


def compile_sources(files, Runner=None, destdir=None, cwd=None, keep_dir_struct=False,
                    per_file_kwargs=None, **kwargs):
    """ Compile source code files to object files.

    Parameters
    ==========

    files : iterable of str
        Paths to source files, if ``cwd`` is given, the paths are taken as relative.
    Runner: CompilerRunner subclass (optional)
        Could be e.g. ``FortranCompilerRunner``. Will be inferred from filename
        extensions if missing.
    destdir: str
        Output directory, if cwd is given, the path is taken as relative.
    cwd: str
        Working directory. Specify to have compiler run in other directory.
        also used as root of relative paths.
    keep_dir_struct: bool
        Reproduce directory structure in `destdir`. default: ``False``
    per_file_kwargs: dict
        Dict mapping instances in ``files`` to keyword arguments.
    \\*\\*kwargs: dict
        Default keyword arguments to pass to ``Runner``.

    Returns
    =======
    List of strings (paths of object files).
    """
    _per_file_kwargs = {}

    if per_file_kwargs is not None:
        for k, v in per_file_kwargs.items():
            if isinstance(k, Glob):
                for path in glob.glob(k.pathname):
                    _per_file_kwargs[path] = v
            elif isinstance(k, ArbitraryDepthGlob):
                for path in glob_at_depth(k.filename, cwd):
                    _per_file_kwargs[path] = v
            else:
                _per_file_kwargs[k] = v

    # Set up destination directory
    destdir = destdir or '.'
    if not os.path.isdir(destdir):
        if os.path.exists(destdir):
            raise OSError("{} is not a directory".format(destdir))
        else:
            make_dirs(destdir)
    if cwd is None:
        cwd = '.'
        for f in files:
            copy(f, destdir, only_update=True, dest_is_dir=True)

    # Compile files and return list of paths to the objects
    dstpaths = []
    for f in files:
        if keep_dir_struct:
            name, ext = os.path.splitext(f)
        else:
            name, ext = os.path.splitext(os.path.basename(f))
        file_kwargs = kwargs.copy()
        file_kwargs.update(_per_file_kwargs.get(f, {}))
        dstpaths.append(src2obj(f, Runner, cwd=cwd, **file_kwargs))
    return dstpaths


def get_mixed_fort_c_linker(vendor=None, cplus=False, cwd=None):
    vendor = vendor or os.environ.get('SYMPY_COMPILER_VENDOR', 'gnu')

    if vendor.lower() == 'intel':
        if cplus:
            return (FortranCompilerRunner,
                    {'flags': ['-nofor_main', '-cxxlib']}, vendor)
        else:
            return (FortranCompilerRunner,
                    {'flags': ['-nofor_main']}, vendor)
    elif vendor.lower() == 'gnu' or 'llvm':
        if cplus:
            return (CppCompilerRunner,
                    {'lib_options': ['fortran']}, vendor)
        else:
            return (FortranCompilerRunner,
                    {}, vendor)
    else:
        raise ValueError("No vendor found.")


def link(obj_files, out_file=None, shared=False, Runner=None,
         cwd=None, cplus=False, fort=False, extra_objs=None, **kwargs):
    """ Link object files.

    Parameters
    ==========

    obj_files: iterable of str
        Paths to object files.
    out_file: str (optional)
        Path to executable/shared library, if ``None`` it will be
        deduced from the last item in obj_files.
    shared: bool
        Generate a shared library?
    Runner: CompilerRunner subclass (optional)
        If not given the ``cplus`` and ``fort`` flags will be inspected
        (fallback is the C compiler).
    cwd: str
        Path to the root of relative paths and working directory for compiler.
    cplus: bool
        C++ objects? default: ``False``.
    fort: bool
        Fortran objects? default: ``False``.
    extra_objs: list
        List of paths to extra object files / static libraries.
    \\*\\*kwargs: dict
        Keyword arguments passed to ``Runner``.

    Returns
    =======

    The absolute path to the generated shared object / executable.

    """
    if out_file is None:
        out_file, ext = os.path.splitext(os.path.basename(obj_files[-1]))
        if shared:
            out_file += get_config_var('EXT_SUFFIX')

    if not Runner:
        if fort:
            Runner, extra_kwargs, vendor = \
                get_mixed_fort_c_linker(
                    vendor=kwargs.get('vendor', None),
                    cplus=cplus,
                    cwd=cwd,
                )
            for k, v in extra_kwargs.items():
                if k in kwargs:
                    kwargs[k].expand(v)
                else:
                    kwargs[k] = v
        else:
            if cplus:
                Runner = CppCompilerRunner
            else:
                Runner = CCompilerRunner

    flags = kwargs.pop('flags', [])
    if shared:
        if '-shared' not in flags:
            flags.append('-shared')
    run_linker = kwargs.pop('run_linker', True)
    if not run_linker:
        raise ValueError("run_linker was set to False (nonsensical).")

    out_file = get_abspath(out_file, cwd=cwd)
    runner = Runner(obj_files+(extra_objs or []), out_file, flags, cwd=cwd, **kwargs)
    runner.run()
    return out_file


def link_py_so(obj_files, so_file=None, cwd=None, libraries=None,
               cplus=False, fort=False, extra_objs=None, **kwargs):
    """ Link Python extension module (shared object) for importing

    Parameters
    ==========

    obj_files: iterable of str
        Paths to object files to be linked.
    so_file: str
        Name (path) of shared object file to create. If not specified it will
        have the basname of the last object file in `obj_files` but with the
        extension '.so' (Unix).
    cwd: path string
        Root of relative paths and working directory of linker.
    libraries: iterable of strings
        Libraries to link against, e.g. ['m'].
    cplus: bool
        Any C++ objects? default: ``False``.
    fort: bool
        Any Fortran objects? default: ``False``.
    extra_objs: list
        List of paths of extra object files / static libraries to link against.
    kwargs**: dict
        Keyword arguments passed to ``link(...)``.

    Returns
    =======

    Absolute path to the generate shared object.
    """
    libraries = libraries or []

    include_dirs = kwargs.pop('include_dirs', [])
    library_dirs = kwargs.pop('library_dirs', [])

    # Add Python include and library directories
    # PY_LDFLAGS does not available on all python implementations
    # e.g. when with pypy, so it's LDFLAGS we need to use
    if sys.platform == "win32":
        warnings.warn("Windows not yet supported.")
    elif sys.platform == 'darwin':
        cfgDict = get_config_vars()
        kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
        library_dirs += [cfgDict['LIBDIR']]

        # In macOS, linker needs to compile frameworks
        # e.g. "-framework CoreFoundation"
        is_framework = False
        for opt in cfgDict['LIBS'].split():
            if is_framework:
                kwargs['linkline'] = kwargs.get('linkline', []) + ['-framework', opt]
                is_framework = False
            elif opt.startswith('-l'):
                libraries.append(opt[2:])
            elif opt.startswith('-framework'):
                is_framework = True
        # The python library is not included in LIBS
        libfile = cfgDict['LIBRARY']
        libname = ".".join(libfile.split('.')[:-1])[3:]
        libraries.append(libname)

    elif sys.platform[:3] == 'aix':
        # Don't use the default code below
        pass
    else:
        if get_config_var('Py_ENABLE_SHARED'):
            cfgDict = get_config_vars()
            kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
            library_dirs += [cfgDict['LIBDIR']]
            for opt in cfgDict['BLDLIBRARY'].split():
                if opt.startswith('-l'):
                    libraries += [opt[2:]]
        else:
            pass

    flags = kwargs.pop('flags', [])
    needed_flags = ('-pthread',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)

    return link(obj_files, shared=True, flags=flags, cwd=cwd, cplus=cplus, fort=fort,
                include_dirs=include_dirs, libraries=libraries,
                library_dirs=library_dirs, extra_objs=extra_objs, **kwargs)


def simple_cythonize(src, destdir=None, cwd=None, **cy_kwargs):
    """ Generates a C file from a Cython source file.

    Parameters
    ==========

    src: str
        Path to Cython source.
    destdir: str (optional)
        Path to output directory (default: '.').
    cwd: path string (optional)
        Root of relative paths (default: '.').
    **cy_kwargs:
        Second argument passed to cy_compile. Generates a .cpp file if ``cplus=True`` in ``cy_kwargs``,
        else a .c file.
    """
    from Cython.Compiler.Main import (
        default_options, CompilationOptions
    )
    from Cython.Compiler.Main import compile as cy_compile

    assert src.lower().endswith('.pyx') or src.lower().endswith('.py')
    cwd = cwd or '.'
    destdir = destdir or '.'

    ext = '.cpp' if cy_kwargs.get('cplus', False) else '.c'
    c_name = os.path.splitext(os.path.basename(src))[0] + ext

    dstfile = os.path.join(destdir, c_name)

    if cwd:
        ori_dir = os.getcwd()
    else:
        ori_dir = '.'
    os.chdir(cwd)
    try:
        cy_options = CompilationOptions(default_options)
        cy_options.__dict__.update(cy_kwargs)
        # Set language_level if not set by cy_kwargs
        # as not setting it is deprecated
        if 'language_level' not in cy_kwargs:
            cy_options.__dict__['language_level'] = 3
        cy_result = cy_compile([src], cy_options)
        if cy_result.num_errors > 0:
            raise ValueError("Cython compilation failed.")

        # Move generated C file to destination
        # In macOS, the generated C file is in the same directory as the source
        # but the /var is a symlink to /private/var, so we need to use realpath
        if os.path.realpath(os.path.dirname(src)) != os.path.realpath(destdir):
            if os.path.exists(dstfile):
                os.unlink(dstfile)
            shutil.move(os.path.join(os.path.dirname(src), c_name), destdir)
    finally:
        os.chdir(ori_dir)
    return dstfile


extension_mapping = {
    '.c': (CCompilerRunner, None),
    '.cpp': (CppCompilerRunner, None),
    '.cxx': (CppCompilerRunner, None),
    '.f': (FortranCompilerRunner, None),
    '.for': (FortranCompilerRunner, None),
    '.ftn': (FortranCompilerRunner, None),
    '.f90': (FortranCompilerRunner, None),  # ifort only knows about .f90
    '.f95': (FortranCompilerRunner, 'f95'),
    '.f03': (FortranCompilerRunner, 'f2003'),
    '.f08': (FortranCompilerRunner, 'f2008'),
}


def src2obj(srcpath, Runner=None, objpath=None, cwd=None, inc_py=False, **kwargs):
    """ Compiles a source code file to an object file.

    Files ending with '.pyx' assumed to be cython files and
    are dispatched to pyx2obj.

    Parameters
    ==========

    srcpath: str
        Path to source file.
    Runner: CompilerRunner subclass (optional)
        If ``None``: deduced from extension of srcpath.
    objpath : str (optional)
        Path to generated object. If ``None``: deduced from ``srcpath``.
    cwd: str (optional)
        Working directory and root of relative paths. If ``None``: current dir.
    inc_py: bool
        Add Python include path to kwarg "include_dirs". Default: False
    \\*\\*kwargs: dict
        keyword arguments passed to Runner or pyx2obj

    """
    name, ext = os.path.splitext(os.path.basename(srcpath))
    if objpath is None:
        if os.path.isabs(srcpath):
            objpath = '.'
        else:
            objpath = os.path.dirname(srcpath)
            objpath = objpath or '.'  # avoid objpath == ''

    if os.path.isdir(objpath):
        objpath = os.path.join(objpath, name + objext)

    include_dirs = kwargs.pop('include_dirs', [])
    if inc_py:
        py_inc_dir = get_path('include')
        if py_inc_dir not in include_dirs:
            include_dirs.append(py_inc_dir)

    if ext.lower() == '.pyx':
        return pyx2obj(srcpath, objpath=objpath, include_dirs=include_dirs, cwd=cwd,
                       **kwargs)

    if Runner is None:
        Runner, std = extension_mapping[ext.lower()]
        if 'std' not in kwargs:
            kwargs['std'] = std

    flags = kwargs.pop('flags', [])
    needed_flags = ('-fPIC',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)

    # src2obj implies not running the linker...
    run_linker = kwargs.pop('run_linker', False)
    if run_linker:
        raise CompileError("src2obj called with run_linker=True")

    runner = Runner([srcpath], objpath, include_dirs=include_dirs,
                    run_linker=run_linker, cwd=cwd, flags=flags, **kwargs)
    runner.run()
    return objpath


def pyx2obj(pyxpath, objpath=None, destdir=None, cwd=None,
            include_dirs=None, cy_kwargs=None, cplus=None, **kwargs):
    """
    Convenience function

    If cwd is specified, pyxpath and dst are taken to be relative
    If only_update is set to `True` the modification time is checked
    and compilation is only run if the source is newer than the
    destination

    Parameters
    ==========

    pyxpath: str
        Path to Cython source file.
    objpath: str (optional)
        Path to object file to generate.
    destdir: str (optional)
        Directory to put generated C file. When ``None``: directory of ``objpath``.
    cwd: str (optional)
        Working directory and root of relative paths.
    include_dirs: iterable of path strings (optional)
        Passed onto src2obj and via cy_kwargs['include_path']
        to simple_cythonize.
    cy_kwargs: dict (optional)
        Keyword arguments passed onto `simple_cythonize`
    cplus: bool (optional)
        Indicate whether C++ is used. default: auto-detect using ``.util.pyx_is_cplus``.
    compile_kwargs: dict
        keyword arguments passed onto src2obj

    Returns
    =======

    Absolute path of generated object file.

    """
    assert pyxpath.endswith('.pyx')
    cwd = cwd or '.'
    objpath = objpath or '.'
    destdir = destdir or os.path.dirname(objpath)

    abs_objpath = get_abspath(objpath, cwd=cwd)

    if os.path.isdir(abs_objpath):
        pyx_fname = os.path.basename(pyxpath)
        name, ext = os.path.splitext(pyx_fname)
        objpath = os.path.join(objpath, name + objext)

    cy_kwargs = cy_kwargs or {}
    cy_kwargs['output_dir'] = cwd
    if cplus is None:
        cplus = pyx_is_cplus(pyxpath)
    cy_kwargs['cplus'] = cplus

    interm_c_file = simple_cythonize(pyxpath, destdir=destdir, cwd=cwd, **cy_kwargs)

    include_dirs = include_dirs or []
    flags = kwargs.pop('flags', [])
    needed_flags = ('-fwrapv', '-pthread', '-fPIC')
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)

    options = kwargs.pop('options', [])

    if kwargs.pop('strict_aliasing', False):
        raise CompileError("Cython requires strict aliasing to be disabled.")

    # Let's be explicit about standard
    if cplus:
        std = kwargs.pop('std', 'c++98')
    else:
        std = kwargs.pop('std', 'c99')

    return src2obj(interm_c_file, objpath=objpath, cwd=cwd,
                   include_dirs=include_dirs, flags=flags, std=std,
                   options=options, inc_py=True, strict_aliasing=False,
                   **kwargs)


def _any_X(srcs, cls):
    for src in srcs:
        name, ext = os.path.splitext(src)
        key = ext.lower()
        if key in extension_mapping:
            if extension_mapping[key][0] == cls:
                return True
    return False


def any_fortran_src(srcs):
    return _any_X(srcs, FortranCompilerRunner)


def any_cplus_src(srcs):
    return _any_X(srcs, CppCompilerRunner)


def compile_link_import_py_ext(sources, extname=None, build_dir='.', compile_kwargs=None,
                               link_kwargs=None, extra_objs=None):
    """ Compiles sources to a shared object (Python extension) and imports it

    Sources in ``sources`` which is imported. If shared object is newer than the sources, they
    are not recompiled but instead it is imported.

    Parameters
    ==========

    sources : list of strings
        List of paths to sources.
    extname : string
        Name of extension (default: ``None``).
        If ``None``: taken from the last file in ``sources`` without extension.
    build_dir: str
        Path to directory in which objects files etc. are generated.
    compile_kwargs: dict
        keyword arguments passed to ``compile_sources``
    link_kwargs: dict
        keyword arguments passed to ``link_py_so``
    extra_objs: list
        List of paths to (prebuilt) object files / static libraries to link against.

    Returns
    =======

    The imported module from of the Python extension.
    """
    if extname is None:
        extname = os.path.splitext(os.path.basename(sources[-1]))[0]

    compile_kwargs = compile_kwargs or {}
    link_kwargs = link_kwargs or {}

    try:
        mod = import_module_from_file(os.path.join(build_dir, extname), sources)
    except ImportError:
        objs = compile_sources(list(map(get_abspath, sources)), destdir=build_dir,
                               cwd=build_dir, **compile_kwargs)
        so = link_py_so(objs, cwd=build_dir, fort=any_fortran_src(sources),
                        cplus=any_cplus_src(sources), extra_objs=extra_objs, **link_kwargs)
        mod = import_module_from_file(so)
    return mod


def _write_sources_to_build_dir(sources, build_dir):
    build_dir = build_dir or tempfile.mkdtemp()
    if not os.path.isdir(build_dir):
        raise OSError("Non-existent directory: ", build_dir)

    source_files = []
    for name, src in sources:
        dest = os.path.join(build_dir, name)
        differs = True
        sha256_in_mem = sha256_of_string(src.encode('utf-8')).hexdigest()
        if os.path.exists(dest):
            if os.path.exists(dest + '.sha256'):
                sha256_on_disk = Path(dest + '.sha256').read_text()
            else:
                sha256_on_disk = sha256_of_file(dest).hexdigest()

            differs = sha256_on_disk != sha256_in_mem
        if differs:
            with open(dest, 'wt') as fh:
                fh.write(src)
            with open(dest + '.sha256', 'wt') as fh:
                fh.write(sha256_in_mem)
        source_files.append(dest)
    return source_files, build_dir


def compile_link_import_strings(sources, build_dir=None, **kwargs):
    """ Compiles, links and imports extension module from source.

    Parameters
    ==========

    sources : iterable of name/source pair tuples
    build_dir : string (default: None)
        Path. ``None`` implies use a temporary directory.
    **kwargs:
        Keyword arguments passed onto `compile_link_import_py_ext`.

    Returns
    =======

    mod : module
        The compiled and imported extension module.
    info : dict
        Containing ``build_dir`` as 'build_dir'.

    """
    source_files, build_dir = _write_sources_to_build_dir(sources, build_dir)
    mod = compile_link_import_py_ext(source_files, build_dir=build_dir, **kwargs)
    info = {"build_dir": build_dir}
    return mod, info


def compile_run_strings(sources, build_dir=None, clean=False, compile_kwargs=None, link_kwargs=None):
    """ Compiles, links and runs a program built from sources.

    Parameters
    ==========

    sources : iterable of name/source pair tuples
    build_dir : string (default: None)
        Path. ``None`` implies use a temporary directory.
    clean : bool
        Whether to remove build_dir after use. This will only have an
        effect if ``build_dir`` is ``None`` (which creates a temporary directory).
        Passing ``clean == True`` and ``build_dir != None`` raises a ``ValueError``.
        This will also set ``build_dir`` in returned info dictionary to ``None``.
    compile_kwargs: dict
        Keyword arguments passed onto ``compile_sources``
    link_kwargs: dict
        Keyword arguments passed onto ``link``

    Returns
    =======

    (stdout, stderr): pair of strings
    info: dict
        Containing exit status as 'exit_status' and ``build_dir`` as 'build_dir'

    """
    if clean and build_dir is not None:
        raise ValueError("Automatic removal of build_dir is only available for temporary directory.")
    try:
        source_files, build_dir = _write_sources_to_build_dir(sources, build_dir)
        objs = compile_sources(list(map(get_abspath, source_files)), destdir=build_dir,
                               cwd=build_dir, **(compile_kwargs or {}))
        prog = link(objs, cwd=build_dir,
                    fort=any_fortran_src(source_files),
                    cplus=any_cplus_src(source_files), **(link_kwargs or {}))
        p = subprocess.Popen([prog], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        exit_status = p.wait()
        stdout, stderr = [txt.decode('utf-8') for txt in p.communicate()]
    finally:
        if clean and os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
            build_dir = None
    info = {"exit_status": exit_status, "build_dir": build_dir}
    return (stdout, stderr), info

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\tags.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import logging
import platform
import re
import struct
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    cast,
)

from . import _manylinux, _musllinux

logger = logging.getLogger(__name__)

PythonVersion = Sequence[int]
AppleVersion = Tuple[int, int]

INTERPRETER_SHORT_NAMES: dict[str, str] = {
    "python": "py",  # Generic.
    "cpython": "cp",
    "pypy": "pp",
    "ironpython": "ip",
    "jython": "jy",
}


_32_BIT_INTERPRETER = struct.calcsize("P") == 4


class Tag:
    """
    A representation of the tag triple for a wheel.

    Instances are considered immutable and thus are hashable. Equality checking
    is also supported.
    """

    __slots__ = ["_abi", "_hash", "_interpreter", "_platform"]

    def __init__(self, interpreter: str, abi: str, platform: str) -> None:
        self._interpreter = interpreter.lower()
        self._abi = abi.lower()
        self._platform = platform.lower()
        # The __hash__ of every single element in a Set[Tag] will be evaluated each time
        # that a set calls its `.disjoint()` method, which may be called hundreds of
        # times when scanning a page of links for packages with tags matching that
        # Set[Tag]. Pre-computing the value here produces significant speedups for
        # downstream consumers.
        self._hash = hash((self._interpreter, self._abi, self._platform))

    @property
    def interpreter(self) -> str:
        return self._interpreter

    @property
    def abi(self) -> str:
        return self._abi

    @property
    def platform(self) -> str:
        return self._platform

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented

        return (
            (self._hash == other._hash)  # Short-circuit ASAP for perf reasons.
            and (self._platform == other._platform)
            and (self._abi == other._abi)
            and (self._interpreter == other._interpreter)
        )

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"{self._interpreter}-{self._abi}-{self._platform}"

    def __repr__(self) -> str:
        return f"<{self} @ {id(self)}>"


def parse_tag(tag: str) -> frozenset[Tag]:
    """
    Parses the provided tag (e.g. `py3-none-any`) into a frozenset of Tag instances.

    Returning a set is required due to the possibility that the tag is a
    compressed tag set.
    """
    tags = set()
    interpreters, abis, platforms = tag.split("-")
    for interpreter in interpreters.split("."):
        for abi in abis.split("."):
            for platform_ in platforms.split("."):
                tags.add(Tag(interpreter, abi, platform_))
    return frozenset(tags)


def _get_config_var(name: str, warn: bool = False) -> int | str | None:
    value: int | str | None = sysconfig.get_config_var(name)
    if value is None and warn:
        logger.debug(
            "Config variable '%s' is unset, Python ABI tag may be incorrect", name
        )
    return value


def _normalize_string(string: str) -> str:
    return string.replace(".", "_").replace("-", "_").replace(" ", "_")


def _is_threaded_cpython(abis: list[str]) -> bool:
    """
    Determine if the ABI corresponds to a threaded (`--disable-gil`) build.

    The threaded builds are indicated by a "t" in the abiflags.
    """
    if len(abis) == 0:
        return False
    # expect e.g., cp313
    m = re.match(r"cp\d+(.*)", abis[0])
    if not m:
        return False
    abiflags = m.group(1)
    return "t" in abiflags


def _abi3_applies(python_version: PythonVersion, threading: bool) -> bool:
    """
    Determine if the Python version supports abi3.

    PEP 384 was first implemented in Python 3.2. The threaded (`--disable-gil`)
    builds do not support abi3.
    """
    return len(python_version) > 1 and tuple(python_version) >= (3, 2) and not threading


def _cpython_abis(py_version: PythonVersion, warn: bool = False) -> list[str]:
    py_version = tuple(py_version)  # To allow for version comparison.
    abis = []
    version = _version_nodot(py_version[:2])
    threading = debug = pymalloc = ucs4 = ""
    with_debug = _get_config_var("Py_DEBUG", warn)
    has_refcount = hasattr(sys, "gettotalrefcount")
    # Windows doesn't set Py_DEBUG, so checking for support of debug-compiled
    # extension modules is the best option.
    # https://github.com/pypa/pip/issues/3383#issuecomment-173267692
    has_ext = "_d.pyd" in EXTENSION_SUFFIXES
    if with_debug or (with_debug is None and (has_refcount or has_ext)):
        debug = "d"
    if py_version >= (3, 13) and _get_config_var("Py_GIL_DISABLED", warn):
        threading = "t"
    if py_version < (3, 8):
        with_pymalloc = _get_config_var("WITH_PYMALLOC", warn)
        if with_pymalloc or with_pymalloc is None:
            pymalloc = "m"
        if py_version < (3, 3):
            unicode_size = _get_config_var("Py_UNICODE_SIZE", warn)
            if unicode_size == 4 or (
                unicode_size is None and sys.maxunicode == 0x10FFFF
            ):
                ucs4 = "u"
    elif debug:
        # Debug builds can also load "normal" extension modules.
        # We can also assume no UCS-4 or pymalloc requirement.
        abis.append(f"cp{version}{threading}")
    abis.insert(0, f"cp{version}{threading}{debug}{pymalloc}{ucs4}")
    return abis


def cpython_tags(
    python_version: PythonVersion | None = None,
    abis: Iterable[str] | None = None,
    platforms: Iterable[str] | None = None,
    *,
    warn: bool = False,
) -> Iterator[Tag]:
    """
    Yields the tags for a CPython interpreter.

    The tags consist of:
    - cp<python_version>-<abi>-<platform>
    - cp<python_version>-abi3-<platform>
    - cp<python_version>-none-<platform>
    - cp<less than python_version>-abi3-<platform>  # Older Python versions down to 3.2.

    If python_version only specifies a major version then user-provided ABIs and
    the 'none' ABItag will be used.

    If 'abi3' or 'none' are specified in 'abis' then they will be yielded at
    their normal position and not at the beginning.
    """
    if not python_version:
        python_version = sys.version_info[:2]

    interpreter = f"cp{_version_nodot(python_version[:2])}"

    if abis is None:
        if len(python_version) > 1:
            abis = _cpython_abis(python_version, warn)
        else:
            abis = []
    abis = list(abis)
    # 'abi3' and 'none' are explicitly handled later.
    for explicit_abi in ("abi3", "none"):
        try:
            abis.remove(explicit_abi)
        except ValueError:
            pass

    platforms = list(platforms or platform_tags())
    for abi in abis:
        for platform_ in platforms:
            yield Tag(interpreter, abi, platform_)

    threading = _is_threaded_cpython(abis)
    use_abi3 = _abi3_applies(python_version, threading)
    if use_abi3:
        yield from (Tag(interpreter, "abi3", platform_) for platform_ in platforms)
    yield from (Tag(interpreter, "none", platform_) for platform_ in platforms)

    if use_abi3:
        for minor_version in range(python_version[1] - 1, 1, -1):
            for platform_ in platforms:
                version = _version_nodot((python_version[0], minor_version))
                interpreter = f"cp{version}"
                yield Tag(interpreter, "abi3", platform_)


def _generic_abi() -> list[str]:
    """
    Return the ABI tag based on EXT_SUFFIX.
    """
    # The following are examples of `EXT_SUFFIX`.
    # We want to keep the parts which are related to the ABI and remove the
    # parts which are related to the platform:
    # - linux:   '.cpython-310-x86_64-linux-gnu.so' => cp310
    # - mac:     '.cpython-310-darwin.so'           => cp310
    # - win:     '.cp310-win_amd64.pyd'             => cp310
    # - win:     '.pyd'                             => cp37 (uses _cpython_abis())
    # - pypy:    '.pypy38-pp73-x86_64-linux-gnu.so' => pypy38_pp73
    # - graalpy: '.graalpy-38-native-x86_64-darwin.dylib'
    #                                               => graalpy_38_native

    ext_suffix = _get_config_var("EXT_SUFFIX", warn=True)
    if not isinstance(ext_suffix, str) or ext_suffix[0] != ".":
        raise SystemError("invalid sysconfig.get_config_var('EXT_SUFFIX')")
    parts = ext_suffix.split(".")
    if len(parts) < 3:
        # CPython3.7 and earlier uses ".pyd" on Windows.
        return _cpython_abis(sys.version_info[:2])
    soabi = parts[1]
    if soabi.startswith("cpython"):
        # non-windows
        abi = "cp" + soabi.split("-")[1]
    elif soabi.startswith("cp"):
        # windows
        abi = soabi.split("-")[0]
    elif soabi.startswith("pypy"):
        abi = "-".join(soabi.split("-")[:2])
    elif soabi.startswith("graalpy"):
        abi = "-".join(soabi.split("-")[:3])
    elif soabi:
        # pyston, ironpython, others?
        abi = soabi
    else:
        return []
    return [_normalize_string(abi)]


def generic_tags(
    interpreter: str | None = None,
    abis: Iterable[str] | None = None,
    platforms: Iterable[str] | None = None,
    *,
    warn: bool = False,
) -> Iterator[Tag]:
    """
    Yields the tags for a generic interpreter.

    The tags consist of:
    - <interpreter>-<abi>-<platform>

    The "none" ABI will be added if it was not explicitly provided.
    """
    if not interpreter:
        interp_name = interpreter_name()
        interp_version = interpreter_version(warn=warn)
        interpreter = "".join([interp_name, interp_version])
    if abis is None:
        abis = _generic_abi()
    else:
        abis = list(abis)
    platforms = list(platforms or platform_tags())
    if "none" not in abis:
        abis.append("none")
    for abi in abis:
        for platform_ in platforms:
            yield Tag(interpreter, abi, platform_)


def _py_interpreter_range(py_version: PythonVersion) -> Iterator[str]:
    """
    Yields Python versions in descending order.

    After the latest version, the major-only version will be yielded, and then
    all previous versions of that major version.
    """
    if len(py_version) > 1:
        yield f"py{_version_nodot(py_version[:2])}"
    yield f"py{py_version[0]}"
    if len(py_version) > 1:
        for minor in range(py_version[1] - 1, -1, -1):
            yield f"py{_version_nodot((py_version[0], minor))}"


def compatible_tags(
    python_version: PythonVersion | None = None,
    interpreter: str | None = None,
    platforms: Iterable[str] | None = None,
) -> Iterator[Tag]:
    """
    Yields the sequence of tags that are compatible with a specific version of Python.

    The tags consist of:
    - py*-none-<platform>
    - <interpreter>-none-any  # ... if `interpreter` is provided.
    - py*-none-any
    """
    if not python_version:
        python_version = sys.version_info[:2]
    platforms = list(platforms or platform_tags())
    for version in _py_interpreter_range(python_version):
        for platform_ in platforms:
            yield Tag(version, "none", platform_)
    if interpreter:
        yield Tag(interpreter, "none", "any")
    for version in _py_interpreter_range(python_version):
        yield Tag(version, "none", "any")


def _mac_arch(arch: str, is_32bit: bool = _32_BIT_INTERPRETER) -> str:
    if not is_32bit:
        return arch

    if arch.startswith("ppc"):
        return "ppc"

    return "i386"


def _mac_binary_formats(version: AppleVersion, cpu_arch: str) -> list[str]:
    formats = [cpu_arch]
    if cpu_arch == "x86_64":
        if version < (10, 4):
            return []
        formats.extend(["intel", "fat64", "fat32"])

    elif cpu_arch == "i386":
        if version < (10, 4):
            return []
        formats.extend(["intel", "fat32", "fat"])

    elif cpu_arch == "ppc64":
        # TODO: Need to care about 32-bit PPC for ppc64 through 10.2?
        if version > (10, 5) or version < (10, 4):
            return []
        formats.append("fat64")

    elif cpu_arch == "ppc":
        if version > (10, 6):
            return []
        formats.extend(["fat32", "fat"])

    if cpu_arch in {"arm64", "x86_64"}:
        formats.append("universal2")

    if cpu_arch in {"x86_64", "i386", "ppc64", "ppc", "intel"}:
        formats.append("universal")

    return formats


def mac_platforms(
    version: AppleVersion | None = None, arch: str | None = None
) -> Iterator[str]:
    """
    Yields the platform tags for a macOS system.

    The `version` parameter is a two-item tuple specifying the macOS version to
    generate platform tags for. The `arch` parameter is the CPU architecture to
    generate platform tags for. Both parameters default to the appropriate value
    for the current system.
    """
    version_str, _, cpu_arch = platform.mac_ver()
    if version is None:
        version = cast("AppleVersion", tuple(map(int, version_str.split(".")[:2])))
        if version == (10, 16):
            # When built against an older macOS SDK, Python will report macOS 10.16
            # instead of the real version.
            version_str = subprocess.run(
                [
                    sys.executable,
                    "-sS",
                    "-c",
                    "import platform; print(platform.mac_ver()[0])",
                ],
                check=True,
                env={"SYSTEM_VERSION_COMPAT": "0"},
                stdout=subprocess.PIPE,
                text=True,
            ).stdout
            version = cast("AppleVersion", tuple(map(int, version_str.split(".")[:2])))
    else:
        version = version
    if arch is None:
        arch = _mac_arch(cpu_arch)
    else:
        arch = arch

    if (10, 0) <= version and version < (11, 0):
        # Prior to Mac OS 11, each yearly release of Mac OS bumped the
        # "minor" version number.  The major version was always 10.
        major_version = 10
        for minor_version in range(version[1], -1, -1):
            compat_version = major_version, minor_version
            binary_formats = _mac_binary_formats(compat_version, arch)
            for binary_format in binary_formats:
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"

    if version >= (11, 0):
        # Starting with Mac OS 11, each yearly release bumps the major version
        # number.   The minor versions are now the midyear updates.
        minor_version = 0
        for major_version in range(version[0], 10, -1):
            compat_version = major_version, minor_version
            binary_formats = _mac_binary_formats(compat_version, arch)
            for binary_format in binary_formats:
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"

    if version >= (11, 0):
        # Mac OS 11 on x86_64 is compatible with binaries from previous releases.
        # Arm64 support was introduced in 11.0, so no Arm binaries from previous
        # releases exist.
        #
        # However, the "universal2" binary format can have a
        # macOS version earlier than 11.0 when the x86_64 part of the binary supports
        # that version of macOS.
        major_version = 10
        if arch == "x86_64":
            for minor_version in range(16, 3, -1):
                compat_version = major_version, minor_version
                binary_formats = _mac_binary_formats(compat_version, arch)
                for binary_format in binary_formats:
                    yield f"macosx_{major_version}_{minor_version}_{binary_format}"
        else:
            for minor_version in range(16, 3, -1):
                compat_version = major_version, minor_version
                binary_format = "universal2"
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"


def ios_platforms(
    version: AppleVersion | None = None, multiarch: str | None = None
) -> Iterator[str]:
    """
    Yields the platform tags for an iOS system.

    :param version: A two-item tuple specifying the iOS version to generate
        platform tags for. Defaults to the current iOS version.
    :param multiarch: The CPU architecture+ABI to generate platform tags for -
        (the value used by `sys.implementation._multiarch` e.g.,
        `arm64_iphoneos` or `x84_64_iphonesimulator`). Defaults to the current
        multiarch value.
    """
    if version is None:
        # if iOS is the current platform, ios_ver *must* be defined. However,
        # it won't exist for CPython versions before 3.13, which causes a mypy
        # error.
        _, release, _, _ = platform.ios_ver()  # type: ignore[attr-defined, unused-ignore]
        version = cast("AppleVersion", tuple(map(int, release.split(".")[:2])))

    if multiarch is None:
        multiarch = sys.implementation._multiarch
    multiarch = multiarch.replace("-", "_")

    ios_platform_template = "ios_{major}_{minor}_{multiarch}"

    # Consider any iOS major.minor version from the version requested, down to
    # 12.0. 12.0 is the first iOS version that is known to have enough features
    # to support CPython. Consider every possible minor release up to X.9. There
    # highest the minor has ever gone is 8 (14.8 and 15.8) but having some extra
    # candidates that won't ever match doesn't really hurt, and it saves us from
    # having to keep an explicit list of known iOS versions in the code. Return
    # the results descending order of version number.

    # If the requested major version is less than 12, there won't be any matches.
    if version[0] < 12:
        return

    # Consider the actual X.Y version that was requested.
    yield ios_platform_template.format(
        major=version[0], minor=version[1], multiarch=multiarch
    )

    # Consider every minor version from X.0 to the minor version prior to the
    # version requested by the platform.
    for minor in range(version[1] - 1, -1, -1):
        yield ios_platform_template.format(
            major=version[0], minor=minor, multiarch=multiarch
        )

    for major in range(version[0] - 1, 11, -1):
        for minor in range(9, -1, -1):
            yield ios_platform_template.format(
                major=major, minor=minor, multiarch=multiarch
            )


def android_platforms(
    api_level: int | None = None, abi: str | None = None
) -> Iterator[str]:
    """
    Yields the :attr:`~Tag.platform` tags for Android. If this function is invoked on
    non-Android platforms, the ``api_level`` and ``abi`` arguments are required.

    :param int api_level: The maximum `API level
        <https://developer.android.com/tools/releases/platforms>`__ to return. Defaults
        to the current system's version, as returned by ``platform.android_ver``.
    :param str abi: The `Android ABI <https://developer.android.com/ndk/guides/abis>`__,
        e.g. ``arm64_v8a``. Defaults to the current system's ABI , as returned by
        ``sysconfig.get_platform``. Hyphens and periods will be replaced with
        underscores.
    """
    if platform.system() != "Android" and (api_level is None or abi is None):
        raise TypeError(
            "on non-Android platforms, the api_level and abi arguments are required"
        )

    if api_level is None:
        # Python 3.13 was the first version to return platform.system() == "Android",
        # and also the first version to define platform.android_ver().
        api_level = platform.android_ver().api_level  # type: ignore[attr-defined]

    if abi is None:
        abi = sysconfig.get_platform().split("-")[-1]
    abi = _normalize_string(abi)

    # 16 is the minimum API level known to have enough features to support CPython
    # without major patching. Yield every API level from the maximum down to the
    # minimum, inclusive.
    min_api_level = 16
    for ver in range(api_level, min_api_level - 1, -1):
        yield f"android_{ver}_{abi}"


def _linux_platforms(is_32bit: bool = _32_BIT_INTERPRETER) -> Iterator[str]:
    linux = _normalize_string(sysconfig.get_platform())
    if not linux.startswith("linux_"):
        # we should never be here, just yield the sysconfig one and return
        yield linux
        return
    if is_32bit:
        if linux == "linux_x86_64":
            linux = "linux_i686"
        elif linux == "linux_aarch64":
            linux = "linux_armv8l"
    _, arch = linux.split("_", 1)
    archs = {"armv8l": ["armv8l", "armv7l"]}.get(arch, [arch])
    yield from _manylinux.platform_tags(archs)
    yield from _musllinux.platform_tags(archs)
    for arch in archs:
        yield f"linux_{arch}"


def _generic_platforms() -> Iterator[str]:
    yield _normalize_string(sysconfig.get_platform())


def platform_tags() -> Iterator[str]:
    """
    Provides the platform tags for this installation.
    """
    if platform.system() == "Darwin":
        return mac_platforms()
    elif platform.system() == "iOS":
        return ios_platforms()
    elif platform.system() == "Android":
        return android_platforms()
    elif platform.system() == "Linux":
        return _linux_platforms()
    else:
        return _generic_platforms()


def interpreter_name() -> str:
    """
    Returns the name of the running interpreter.

    Some implementations have a reserved, two-letter abbreviation which will
    be returned when appropriate.
    """
    name = sys.implementation.name
    return INTERPRETER_SHORT_NAMES.get(name) or name


def interpreter_version(*, warn: bool = False) -> str:
    """
    Returns the version of the running interpreter.
    """
    version = _get_config_var("py_version_nodot", warn=warn)
    if version:
        version = str(version)
    else:
        version = _version_nodot(sys.version_info[:2])
    return version


def _version_nodot(version: PythonVersion) -> str:
    return "".join(map(str, version))


def sys_tags(*, warn: bool = False) -> Iterator[Tag]:
    """
    Returns the sequence of tag triples for the running interpreter.

    The order of the sequence corresponds to priority order for the
    interpreter, from most to least important.
    """

    interp_name = interpreter_name()
    if interp_name == "cp":
        yield from cpython_tags(warn=warn)
    else:
        yield from generic_tags()

    if interp_name == "pp":
        interp = "pp3"
    elif interp_name == "cp":
        interp = "cp" + interpreter_version(warn=warn)
    else:
        interp = None
    yield from compatible_tags(interpreter=interp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\tags.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import logging
import platform
import re
import struct
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
    Iterable,
    Iterator,
    Sequence,
    Tuple,
    cast,
)

from . import _manylinux, _musllinux

logger = logging.getLogger(__name__)

PythonVersion = Sequence[int]
AppleVersion = Tuple[int, int]

INTERPRETER_SHORT_NAMES: dict[str, str] = {
    "python": "py",  # Generic.
    "cpython": "cp",
    "pypy": "pp",
    "ironpython": "ip",
    "jython": "jy",
}


_32_BIT_INTERPRETER = struct.calcsize("P") == 4


class Tag:
    """
    A representation of the tag triple for a wheel.

    Instances are considered immutable and thus are hashable. Equality checking
    is also supported.
    """

    __slots__ = ["_abi", "_hash", "_interpreter", "_platform"]

    def __init__(self, interpreter: str, abi: str, platform: str) -> None:
        self._interpreter = interpreter.lower()
        self._abi = abi.lower()
        self._platform = platform.lower()
        # The __hash__ of every single element in a Set[Tag] will be evaluated each time
        # that a set calls its `.disjoint()` method, which may be called hundreds of
        # times when scanning a page of links for packages with tags matching that
        # Set[Tag]. Pre-computing the value here produces significant speedups for
        # downstream consumers.
        self._hash = hash((self._interpreter, self._abi, self._platform))

    @property
    def interpreter(self) -> str:
        return self._interpreter

    @property
    def abi(self) -> str:
        return self._abi

    @property
    def platform(self) -> str:
        return self._platform

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented

        return (
            (self._hash == other._hash)  # Short-circuit ASAP for perf reasons.
            and (self._platform == other._platform)
            and (self._abi == other._abi)
            and (self._interpreter == other._interpreter)
        )

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"{self._interpreter}-{self._abi}-{self._platform}"

    def __repr__(self) -> str:
        return f"<{self} @ {id(self)}>"


def parse_tag(tag: str) -> frozenset[Tag]:
    """
    Parses the provided tag (e.g. `py3-none-any`) into a frozenset of Tag instances.

    Returning a set is required due to the possibility that the tag is a
    compressed tag set.
    """
    tags = set()
    interpreters, abis, platforms = tag.split("-")
    for interpreter in interpreters.split("."):
        for abi in abis.split("."):
            for platform_ in platforms.split("."):
                tags.add(Tag(interpreter, abi, platform_))
    return frozenset(tags)


def _get_config_var(name: str, warn: bool = False) -> int | str | None:
    value: int | str | None = sysconfig.get_config_var(name)
    if value is None and warn:
        logger.debug(
            "Config variable '%s' is unset, Python ABI tag may be incorrect", name
        )
    return value


def _normalize_string(string: str) -> str:
    return string.replace(".", "_").replace("-", "_").replace(" ", "_")


def _is_threaded_cpython(abis: list[str]) -> bool:
    """
    Determine if the ABI corresponds to a threaded (`--disable-gil`) build.

    The threaded builds are indicated by a "t" in the abiflags.
    """
    if len(abis) == 0:
        return False
    # expect e.g., cp313
    m = re.match(r"cp\d+(.*)", abis[0])
    if not m:
        return False
    abiflags = m.group(1)
    return "t" in abiflags


def _abi3_applies(python_version: PythonVersion, threading: bool) -> bool:
    """
    Determine if the Python version supports abi3.

    PEP 384 was first implemented in Python 3.2. The threaded (`--disable-gil`)
    builds do not support abi3.
    """
    return len(python_version) > 1 and tuple(python_version) >= (3, 2) and not threading


def _cpython_abis(py_version: PythonVersion, warn: bool = False) -> list[str]:
    py_version = tuple(py_version)  # To allow for version comparison.
    abis = []
    version = _version_nodot(py_version[:2])
    threading = debug = pymalloc = ucs4 = ""
    with_debug = _get_config_var("Py_DEBUG", warn)
    has_refcount = hasattr(sys, "gettotalrefcount")
    # Windows doesn't set Py_DEBUG, so checking for support of debug-compiled
    # extension modules is the best option.
    # https://github.com/pypa/pip/issues/3383#issuecomment-173267692
    has_ext = "_d.pyd" in EXTENSION_SUFFIXES
    if with_debug or (with_debug is None and (has_refcount or has_ext)):
        debug = "d"
    if py_version >= (3, 13) and _get_config_var("Py_GIL_DISABLED", warn):
        threading = "t"
    if py_version < (3, 8):
        with_pymalloc = _get_config_var("WITH_PYMALLOC", warn)
        if with_pymalloc or with_pymalloc is None:
            pymalloc = "m"
        if py_version < (3, 3):
            unicode_size = _get_config_var("Py_UNICODE_SIZE", warn)
            if unicode_size == 4 or (
                unicode_size is None and sys.maxunicode == 0x10FFFF
            ):
                ucs4 = "u"
    elif debug:
        # Debug builds can also load "normal" extension modules.
        # We can also assume no UCS-4 or pymalloc requirement.
        abis.append(f"cp{version}{threading}")
    abis.insert(0, f"cp{version}{threading}{debug}{pymalloc}{ucs4}")
    return abis


def cpython_tags(
    python_version: PythonVersion | None = None,
    abis: Iterable[str] | None = None,
    platforms: Iterable[str] | None = None,
    *,
    warn: bool = False,
) -> Iterator[Tag]:
    """
    Yields the tags for a CPython interpreter.

    The tags consist of:
    - cp<python_version>-<abi>-<platform>
    - cp<python_version>-abi3-<platform>
    - cp<python_version>-none-<platform>
    - cp<less than python_version>-abi3-<platform>  # Older Python versions down to 3.2.

    If python_version only specifies a major version then user-provided ABIs and
    the 'none' ABItag will be used.

    If 'abi3' or 'none' are specified in 'abis' then they will be yielded at
    their normal position and not at the beginning.
    """
    if not python_version:
        python_version = sys.version_info[:2]

    interpreter = f"cp{_version_nodot(python_version[:2])}"

    if abis is None:
        if len(python_version) > 1:
            abis = _cpython_abis(python_version, warn)
        else:
            abis = []
    abis = list(abis)
    # 'abi3' and 'none' are explicitly handled later.
    for explicit_abi in ("abi3", "none"):
        try:
            abis.remove(explicit_abi)
        except ValueError:
            pass

    platforms = list(platforms or platform_tags())
    for abi in abis:
        for platform_ in platforms:
            yield Tag(interpreter, abi, platform_)

    threading = _is_threaded_cpython(abis)
    use_abi3 = _abi3_applies(python_version, threading)
    if use_abi3:
        yield from (Tag(interpreter, "abi3", platform_) for platform_ in platforms)
    yield from (Tag(interpreter, "none", platform_) for platform_ in platforms)

    if use_abi3:
        for minor_version in range(python_version[1] - 1, 1, -1):
            for platform_ in platforms:
                version = _version_nodot((python_version[0], minor_version))
                interpreter = f"cp{version}"
                yield Tag(interpreter, "abi3", platform_)


def _generic_abi() -> list[str]:
    """
    Return the ABI tag based on EXT_SUFFIX.
    """
    # The following are examples of `EXT_SUFFIX`.
    # We want to keep the parts which are related to the ABI and remove the
    # parts which are related to the platform:
    # - linux:   '.cpython-310-x86_64-linux-gnu.so' => cp310
    # - mac:     '.cpython-310-darwin.so'           => cp310
    # - win:     '.cp310-win_amd64.pyd'             => cp310
    # - win:     '.pyd'                             => cp37 (uses _cpython_abis())
    # - pypy:    '.pypy38-pp73-x86_64-linux-gnu.so' => pypy38_pp73
    # - graalpy: '.graalpy-38-native-x86_64-darwin.dylib'
    #                                               => graalpy_38_native

    ext_suffix = _get_config_var("EXT_SUFFIX", warn=True)
    if not isinstance(ext_suffix, str) or ext_suffix[0] != ".":
        raise SystemError("invalid sysconfig.get_config_var('EXT_SUFFIX')")
    parts = ext_suffix.split(".")
    if len(parts) < 3:
        # CPython3.7 and earlier uses ".pyd" on Windows.
        return _cpython_abis(sys.version_info[:2])
    soabi = parts[1]
    if soabi.startswith("cpython"):
        # non-windows
        abi = "cp" + soabi.split("-")[1]
    elif soabi.startswith("cp"):
        # windows
        abi = soabi.split("-")[0]
    elif soabi.startswith("pypy"):
        abi = "-".join(soabi.split("-")[:2])
    elif soabi.startswith("graalpy"):
        abi = "-".join(soabi.split("-")[:3])
    elif soabi:
        # pyston, ironpython, others?
        abi = soabi
    else:
        return []
    return [_normalize_string(abi)]


def generic_tags(
    interpreter: str | None = None,
    abis: Iterable[str] | None = None,
    platforms: Iterable[str] | None = None,
    *,
    warn: bool = False,
) -> Iterator[Tag]:
    """
    Yields the tags for a generic interpreter.

    The tags consist of:
    - <interpreter>-<abi>-<platform>

    The "none" ABI will be added if it was not explicitly provided.
    """
    if not interpreter:
        interp_name = interpreter_name()
        interp_version = interpreter_version(warn=warn)
        interpreter = "".join([interp_name, interp_version])
    if abis is None:
        abis = _generic_abi()
    else:
        abis = list(abis)
    platforms = list(platforms or platform_tags())
    if "none" not in abis:
        abis.append("none")
    for abi in abis:
        for platform_ in platforms:
            yield Tag(interpreter, abi, platform_)


def _py_interpreter_range(py_version: PythonVersion) -> Iterator[str]:
    """
    Yields Python versions in descending order.

    After the latest version, the major-only version will be yielded, and then
    all previous versions of that major version.
    """
    if len(py_version) > 1:
        yield f"py{_version_nodot(py_version[:2])}"
    yield f"py{py_version[0]}"
    if len(py_version) > 1:
        for minor in range(py_version[1] - 1, -1, -1):
            yield f"py{_version_nodot((py_version[0], minor))}"


def compatible_tags(
    python_version: PythonVersion | None = None,
    interpreter: str | None = None,
    platforms: Iterable[str] | None = None,
) -> Iterator[Tag]:
    """
    Yields the sequence of tags that are compatible with a specific version of Python.

    The tags consist of:
    - py*-none-<platform>
    - <interpreter>-none-any  # ... if `interpreter` is provided.
    - py*-none-any
    """
    if not python_version:
        python_version = sys.version_info[:2]
    platforms = list(platforms or platform_tags())
    for version in _py_interpreter_range(python_version):
        for platform_ in platforms:
            yield Tag(version, "none", platform_)
    if interpreter:
        yield Tag(interpreter, "none", "any")
    for version in _py_interpreter_range(python_version):
        yield Tag(version, "none", "any")


def _mac_arch(arch: str, is_32bit: bool = _32_BIT_INTERPRETER) -> str:
    if not is_32bit:
        return arch

    if arch.startswith("ppc"):
        return "ppc"

    return "i386"


def _mac_binary_formats(version: AppleVersion, cpu_arch: str) -> list[str]:
    formats = [cpu_arch]
    if cpu_arch == "x86_64":
        if version < (10, 4):
            return []
        formats.extend(["intel", "fat64", "fat32"])

    elif cpu_arch == "i386":
        if version < (10, 4):
            return []
        formats.extend(["intel", "fat32", "fat"])

    elif cpu_arch == "ppc64":
        # TODO: Need to care about 32-bit PPC for ppc64 through 10.2?
        if version > (10, 5) or version < (10, 4):
            return []
        formats.append("fat64")

    elif cpu_arch == "ppc":
        if version > (10, 6):
            return []
        formats.extend(["fat32", "fat"])

    if cpu_arch in {"arm64", "x86_64"}:
        formats.append("universal2")

    if cpu_arch in {"x86_64", "i386", "ppc64", "ppc", "intel"}:
        formats.append("universal")

    return formats


def mac_platforms(
    version: AppleVersion | None = None, arch: str | None = None
) -> Iterator[str]:
    """
    Yields the platform tags for a macOS system.

    The `version` parameter is a two-item tuple specifying the macOS version to
    generate platform tags for. The `arch` parameter is the CPU architecture to
    generate platform tags for. Both parameters default to the appropriate value
    for the current system.
    """
    version_str, _, cpu_arch = platform.mac_ver()
    if version is None:
        version = cast("AppleVersion", tuple(map(int, version_str.split(".")[:2])))
        if version == (10, 16):
            # When built against an older macOS SDK, Python will report macOS 10.16
            # instead of the real version.
            version_str = subprocess.run(
                [
                    sys.executable,
                    "-sS",
                    "-c",
                    "import platform; print(platform.mac_ver()[0])",
                ],
                check=True,
                env={"SYSTEM_VERSION_COMPAT": "0"},
                stdout=subprocess.PIPE,
                text=True,
            ).stdout
            version = cast("AppleVersion", tuple(map(int, version_str.split(".")[:2])))
    else:
        version = version
    if arch is None:
        arch = _mac_arch(cpu_arch)
    else:
        arch = arch

    if (10, 0) <= version and version < (11, 0):
        # Prior to Mac OS 11, each yearly release of Mac OS bumped the
        # "minor" version number.  The major version was always 10.
        major_version = 10
        for minor_version in range(version[1], -1, -1):
            compat_version = major_version, minor_version
            binary_formats = _mac_binary_formats(compat_version, arch)
            for binary_format in binary_formats:
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"

    if version >= (11, 0):
        # Starting with Mac OS 11, each yearly release bumps the major version
        # number.   The minor versions are now the midyear updates.
        minor_version = 0
        for major_version in range(version[0], 10, -1):
            compat_version = major_version, minor_version
            binary_formats = _mac_binary_formats(compat_version, arch)
            for binary_format in binary_formats:
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"

    if version >= (11, 0):
        # Mac OS 11 on x86_64 is compatible with binaries from previous releases.
        # Arm64 support was introduced in 11.0, so no Arm binaries from previous
        # releases exist.
        #
        # However, the "universal2" binary format can have a
        # macOS version earlier than 11.0 when the x86_64 part of the binary supports
        # that version of macOS.
        major_version = 10
        if arch == "x86_64":
            for minor_version in range(16, 3, -1):
                compat_version = major_version, minor_version
                binary_formats = _mac_binary_formats(compat_version, arch)
                for binary_format in binary_formats:
                    yield f"macosx_{major_version}_{minor_version}_{binary_format}"
        else:
            for minor_version in range(16, 3, -1):
                compat_version = major_version, minor_version
                binary_format = "universal2"
                yield f"macosx_{major_version}_{minor_version}_{binary_format}"


def ios_platforms(
    version: AppleVersion | None = None, multiarch: str | None = None
) -> Iterator[str]:
    """
    Yields the platform tags for an iOS system.

    :param version: A two-item tuple specifying the iOS version to generate
        platform tags for. Defaults to the current iOS version.
    :param multiarch: The CPU architecture+ABI to generate platform tags for -
        (the value used by `sys.implementation._multiarch` e.g.,
        `arm64_iphoneos` or `x84_64_iphonesimulator`). Defaults to the current
        multiarch value.
    """
    if version is None:
        # if iOS is the current platform, ios_ver *must* be defined. However,
        # it won't exist for CPython versions before 3.13, which causes a mypy
        # error.
        _, release, _, _ = platform.ios_ver()  # type: ignore[attr-defined, unused-ignore]
        version = cast("AppleVersion", tuple(map(int, release.split(".")[:2])))

    if multiarch is None:
        multiarch = sys.implementation._multiarch
    multiarch = multiarch.replace("-", "_")

    ios_platform_template = "ios_{major}_{minor}_{multiarch}"

    # Consider any iOS major.minor version from the version requested, down to
    # 12.0. 12.0 is the first iOS version that is known to have enough features
    # to support CPython. Consider every possible minor release up to X.9. There
    # highest the minor has ever gone is 8 (14.8 and 15.8) but having some extra
    # candidates that won't ever match doesn't really hurt, and it saves us from
    # having to keep an explicit list of known iOS versions in the code. Return
    # the results descending order of version number.

    # If the requested major version is less than 12, there won't be any matches.
    if version[0] < 12:
        return

    # Consider the actual X.Y version that was requested.
    yield ios_platform_template.format(
        major=version[0], minor=version[1], multiarch=multiarch
    )

    # Consider every minor version from X.0 to the minor version prior to the
    # version requested by the platform.
    for minor in range(version[1] - 1, -1, -1):
        yield ios_platform_template.format(
            major=version[0], minor=minor, multiarch=multiarch
        )

    for major in range(version[0] - 1, 11, -1):
        for minor in range(9, -1, -1):
            yield ios_platform_template.format(
                major=major, minor=minor, multiarch=multiarch
            )


def android_platforms(
    api_level: int | None = None, abi: str | None = None
) -> Iterator[str]:
    """
    Yields the :attr:`~Tag.platform` tags for Android. If this function is invoked on
    non-Android platforms, the ``api_level`` and ``abi`` arguments are required.

    :param int api_level: The maximum `API level
        <https://developer.android.com/tools/releases/platforms>`__ to return. Defaults
        to the current system's version, as returned by ``platform.android_ver``.
    :param str abi: The `Android ABI <https://developer.android.com/ndk/guides/abis>`__,
        e.g. ``arm64_v8a``. Defaults to the current system's ABI , as returned by
        ``sysconfig.get_platform``. Hyphens and periods will be replaced with
        underscores.
    """
    if platform.system() != "Android" and (api_level is None or abi is None):
        raise TypeError(
            "on non-Android platforms, the api_level and abi arguments are required"
        )

    if api_level is None:
        # Python 3.13 was the first version to return platform.system() == "Android",
        # and also the first version to define platform.android_ver().
        api_level = platform.android_ver().api_level  # type: ignore[attr-defined]

    if abi is None:
        abi = sysconfig.get_platform().split("-")[-1]
    abi = _normalize_string(abi)

    # 16 is the minimum API level known to have enough features to support CPython
    # without major patching. Yield every API level from the maximum down to the
    # minimum, inclusive.
    min_api_level = 16
    for ver in range(api_level, min_api_level - 1, -1):
        yield f"android_{ver}_{abi}"


def _linux_platforms(is_32bit: bool = _32_BIT_INTERPRETER) -> Iterator[str]:
    linux = _normalize_string(sysconfig.get_platform())
    if not linux.startswith("linux_"):
        # we should never be here, just yield the sysconfig one and return
        yield linux
        return
    if is_32bit:
        if linux == "linux_x86_64":
            linux = "linux_i686"
        elif linux == "linux_aarch64":
            linux = "linux_armv8l"
    _, arch = linux.split("_", 1)
    archs = {"armv8l": ["armv8l", "armv7l"]}.get(arch, [arch])
    yield from _manylinux.platform_tags(archs)
    yield from _musllinux.platform_tags(archs)
    for arch in archs:
        yield f"linux_{arch}"


def _generic_platforms() -> Iterator[str]:
    yield _normalize_string(sysconfig.get_platform())


def platform_tags() -> Iterator[str]:
    """
    Provides the platform tags for this installation.
    """
    if platform.system() == "Darwin":
        return mac_platforms()
    elif platform.system() == "iOS":
        return ios_platforms()
    elif platform.system() == "Android":
        return android_platforms()
    elif platform.system() == "Linux":
        return _linux_platforms()
    else:
        return _generic_platforms()


def interpreter_name() -> str:
    """
    Returns the name of the running interpreter.

    Some implementations have a reserved, two-letter abbreviation which will
    be returned when appropriate.
    """
    name = sys.implementation.name
    return INTERPRETER_SHORT_NAMES.get(name) or name


def interpreter_version(*, warn: bool = False) -> str:
    """
    Returns the version of the running interpreter.
    """
    version = _get_config_var("py_version_nodot", warn=warn)
    if version:
        version = str(version)
    else:
        version = _version_nodot(sys.version_info[:2])
    return version


def _version_nodot(version: PythonVersion) -> str:
    return "".join(map(str, version))


def sys_tags(*, warn: bool = False) -> Iterator[Tag]:
    """
    Returns the sequence of tag triples for the running interpreter.

    The order of the sequence corresponds to priority order for the
    interpreter, from most to least important.
    """

    interp_name = interpreter_name()
    if interp_name == "cp":
        yield from cpython_tags(warn=warn)
    else:
        yield from generic_tags()

    if interp_name == "pp":
        interp = "pp3"
    elif interp_name == "cp":
        interp = "cp" + interpreter_version(warn=warn)
    else:
        interp = None
    yield from compatible_tags(interpreter=interp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\event\attr.py ===
# event/attr.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Attribute implementation for _Dispatch classes.

The various listener targets for a particular event class are represented
as attributes, which refer to collections of listeners to be fired off.
These collections can exist at the class level as well as at the instance
level.  An event is fired off using code like this::

    some_object.dispatch.first_connect(arg1, arg2)

Above, ``some_object.dispatch`` would be an instance of ``_Dispatch`` and
``first_connect`` is typically an instance of ``_ListenerCollection``
if event listeners are present, or ``_EmptyListener`` if none are present.

The attribute mechanics here spend effort trying to ensure listener functions
are available with a minimum of function call overhead, that unnecessary
objects aren't created (i.e. many empty per-instance listener collections),
as well as that everything is garbage collectable when owning references are
lost.  Other features such as "propagation" of listener functions across
many ``_Dispatch`` instances, "joining" of multiple ``_Dispatch`` instances,
as well as support for subclass propagation (e.g. events assigned to
``Pool`` vs. ``QueuePool``) are all implemented here.

"""
from __future__ import annotations

import collections
from itertools import chain
import threading
from types import TracebackType
import typing
from typing import Any
from typing import cast
from typing import Collection
from typing import Deque
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import MutableMapping
from typing import MutableSequence
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref

from . import legacy
from . import registry
from .registry import _ET
from .registry import _EventKey
from .registry import _ListenerFnType
from .. import exc
from .. import util
from ..util.concurrency import AsyncAdaptedLock
from ..util.typing import Protocol

_T = TypeVar("_T", bound=Any)

if typing.TYPE_CHECKING:
    from .base import _Dispatch
    from .base import _DispatchCommon
    from .base import _HasEventsDispatch


class RefCollection(util.MemoizedSlots, Generic[_ET]):
    __slots__ = ("ref",)

    ref: weakref.ref[RefCollection[_ET]]

    def _memoized_attr_ref(self) -> weakref.ref[RefCollection[_ET]]:
        return weakref.ref(self, registry._collection_gced)


class _empty_collection(Collection[_T]):
    def append(self, element: _T) -> None:
        pass

    def appendleft(self, element: _T) -> None:
        pass

    def extend(self, other: Sequence[_T]) -> None:
        pass

    def remove(self, element: _T) -> None:
        pass

    def __contains__(self, element: Any) -> bool:
        return False

    def __iter__(self) -> Iterator[_T]:
        return iter([])

    def clear(self) -> None:
        pass

    def __len__(self) -> int:
        return 0


_ListenerFnSequenceType = Union[Deque[_T], _empty_collection[_T]]


class _ClsLevelDispatch(RefCollection[_ET]):
    """Class-level events on :class:`._Dispatch` classes."""

    __slots__ = (
        "clsname",
        "name",
        "arg_names",
        "has_kw",
        "legacy_signatures",
        "_clslevel",
        "__weakref__",
    )

    clsname: str
    name: str
    arg_names: Sequence[str]
    has_kw: bool
    legacy_signatures: MutableSequence[legacy._LegacySignatureType]
    _clslevel: MutableMapping[
        Type[_ET], _ListenerFnSequenceType[_ListenerFnType]
    ]

    def __init__(
        self,
        parent_dispatch_cls: Type[_HasEventsDispatch[_ET]],
        fn: _ListenerFnType,
    ):
        self.name = fn.__name__
        self.clsname = parent_dispatch_cls.__name__
        argspec = util.inspect_getfullargspec(fn)
        self.arg_names = argspec.args[1:]
        self.has_kw = bool(argspec.varkw)
        self.legacy_signatures = list(
            reversed(
                sorted(
                    getattr(fn, "_legacy_signatures", []), key=lambda s: s[0]
                )
            )
        )
        fn.__doc__ = legacy._augment_fn_docs(self, parent_dispatch_cls, fn)

        self._clslevel = weakref.WeakKeyDictionary()

    def _adjust_fn_spec(
        self, fn: _ListenerFnType, named: bool
    ) -> _ListenerFnType:
        if named:
            fn = self._wrap_fn_for_kw(fn)
        if self.legacy_signatures:
            try:
                argspec = util.get_callable_argspec(fn, no_self=True)
            except TypeError:
                pass
            else:
                fn = legacy._wrap_fn_for_legacy(self, fn, argspec)
        return fn

    def _wrap_fn_for_kw(self, fn: _ListenerFnType) -> _ListenerFnType:
        def wrap_kw(*args: Any, **kw: Any) -> Any:
            argdict = dict(zip(self.arg_names, args))
            argdict.update(kw)
            return fn(**argdict)

        return wrap_kw

    def _do_insert_or_append(
        self, event_key: _EventKey[_ET], is_append: bool
    ) -> None:
        target = event_key.dispatch_target
        assert isinstance(
            target, type
        ), "Class-level Event targets must be classes."
        if not getattr(target, "_sa_propagate_class_events", True):
            raise exc.InvalidRequestError(
                f"Can't assign an event directly to the {target} class"
            )

        cls: Type[_ET]

        for cls in util.walk_subclasses(target):
            if cls is not target and cls not in self._clslevel:
                self.update_subclass(cls)
            else:
                if cls not in self._clslevel:
                    self.update_subclass(cls)
                if is_append:
                    self._clslevel[cls].append(event_key._listen_fn)
                else:
                    self._clslevel[cls].appendleft(event_key._listen_fn)
        registry._stored_in_collection(event_key, self)

    def insert(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        self._do_insert_or_append(event_key, is_append=False)

    def append(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        self._do_insert_or_append(event_key, is_append=True)

    def update_subclass(self, target: Type[_ET]) -> None:
        if target not in self._clslevel:
            if getattr(target, "_sa_propagate_class_events", True):
                self._clslevel[target] = collections.deque()
            else:
                self._clslevel[target] = _empty_collection()

        clslevel = self._clslevel[target]
        cls: Type[_ET]
        for cls in target.__mro__[1:]:
            if cls in self._clslevel:
                clslevel.extend(
                    [fn for fn in self._clslevel[cls] if fn not in clslevel]
                )

    def remove(self, event_key: _EventKey[_ET]) -> None:
        target = event_key.dispatch_target
        cls: Type[_ET]
        for cls in util.walk_subclasses(target):
            if cls in self._clslevel:
                self._clslevel[cls].remove(event_key._listen_fn)
        registry._removed_from_collection(event_key, self)

    def clear(self) -> None:
        """Clear all class level listeners"""

        to_clear: Set[_ListenerFnType] = set()
        for dispatcher in self._clslevel.values():
            to_clear.update(dispatcher)
            dispatcher.clear()
        registry._clear(self, to_clear)

    def for_modify(self, obj: _Dispatch[_ET]) -> _ClsLevelDispatch[_ET]:
        """Return an event collection which can be modified.

        For _ClsLevelDispatch at the class level of
        a dispatcher, this returns self.

        """
        return self


class _InstanceLevelDispatch(RefCollection[_ET], Collection[_ListenerFnType]):
    __slots__ = ()

    parent: _ClsLevelDispatch[_ET]

    def _adjust_fn_spec(
        self, fn: _ListenerFnType, named: bool
    ) -> _ListenerFnType:
        return self.parent._adjust_fn_spec(fn, named)

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[_ListenerFnType]:
        raise NotImplementedError()

    def __bool__(self) -> bool:
        raise NotImplementedError()

    def exec_once(self, *args: Any, **kw: Any) -> None:
        raise NotImplementedError()

    def exec_once_unless_exception(self, *args: Any, **kw: Any) -> None:
        raise NotImplementedError()

    def _exec_w_sync_on_first_run(self, *args: Any, **kw: Any) -> None:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kw: Any) -> None:
        raise NotImplementedError()

    def insert(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        raise NotImplementedError()

    def append(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        raise NotImplementedError()

    def remove(self, event_key: _EventKey[_ET]) -> None:
        raise NotImplementedError()

    def for_modify(
        self, obj: _DispatchCommon[_ET]
    ) -> _InstanceLevelDispatch[_ET]:
        """Return an event collection which can be modified.

        For _ClsLevelDispatch at the class level of
        a dispatcher, this returns self.

        """
        return self


class _EmptyListener(_InstanceLevelDispatch[_ET]):
    """Serves as a proxy interface to the events
    served by a _ClsLevelDispatch, when there are no
    instance-level events present.

    Is replaced by _ListenerCollection when instance-level
    events are added.

    """

    __slots__ = "parent", "parent_listeners", "name"

    propagate: FrozenSet[_ListenerFnType] = frozenset()
    listeners: Tuple[()] = ()
    parent: _ClsLevelDispatch[_ET]
    parent_listeners: _ListenerFnSequenceType[_ListenerFnType]
    name: str

    def __init__(self, parent: _ClsLevelDispatch[_ET], target_cls: Type[_ET]):
        if target_cls not in parent._clslevel:
            parent.update_subclass(target_cls)
        self.parent = parent
        self.parent_listeners = parent._clslevel[target_cls]
        self.name = parent.name

    def for_modify(
        self, obj: _DispatchCommon[_ET]
    ) -> _ListenerCollection[_ET]:
        """Return an event collection which can be modified.

        For _EmptyListener at the instance level of
        a dispatcher, this generates a new
        _ListenerCollection, applies it to the instance,
        and returns it.

        """
        obj = cast("_Dispatch[_ET]", obj)

        assert obj._instance_cls is not None
        result = _ListenerCollection(self.parent, obj._instance_cls)
        if getattr(obj, self.name) is self:
            setattr(obj, self.name, result)
        else:
            assert isinstance(getattr(obj, self.name), _JoinedListener)
        return result

    def _needs_modify(self, *args: Any, **kw: Any) -> NoReturn:
        raise NotImplementedError("need to call for_modify()")

    def exec_once(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def exec_once_unless_exception(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def insert(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def append(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def remove(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def clear(self, *args: Any, **kw: Any) -> NoReturn:
        self._needs_modify(*args, **kw)

    def __call__(self, *args: Any, **kw: Any) -> None:
        """Execute this event."""

        for fn in self.parent_listeners:
            fn(*args, **kw)

    def __contains__(self, item: Any) -> bool:
        return item in self.parent_listeners

    def __len__(self) -> int:
        return len(self.parent_listeners)

    def __iter__(self) -> Iterator[_ListenerFnType]:
        return iter(self.parent_listeners)

    def __bool__(self) -> bool:
        return bool(self.parent_listeners)


class _MutexProtocol(Protocol):
    def __enter__(self) -> bool: ...

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]: ...


class _CompoundListener(_InstanceLevelDispatch[_ET]):
    __slots__ = (
        "_exec_once_mutex",
        "_exec_once",
        "_exec_w_sync_once",
        "_is_asyncio",
    )

    _exec_once_mutex: _MutexProtocol
    parent_listeners: Collection[_ListenerFnType]
    listeners: Collection[_ListenerFnType]
    _exec_once: bool
    _exec_w_sync_once: bool

    def __init__(self, *arg: Any, **kw: Any):
        super().__init__(*arg, **kw)
        self._is_asyncio = False

    def _set_asyncio(self) -> None:
        self._is_asyncio = True

    def _memoized_attr__exec_once_mutex(self) -> _MutexProtocol:
        if self._is_asyncio:
            return AsyncAdaptedLock()
        else:
            return threading.Lock()

    def _exec_once_impl(
        self, retry_on_exception: bool, *args: Any, **kw: Any
    ) -> None:
        with self._exec_once_mutex:
            if not self._exec_once:
                try:
                    self(*args, **kw)
                    exception = False
                except:
                    exception = True
                    raise
                finally:
                    if not exception or not retry_on_exception:
                        self._exec_once = True

    def exec_once(self, *args: Any, **kw: Any) -> None:
        """Execute this event, but only if it has not been
        executed already for this collection."""

        if not self._exec_once:
            self._exec_once_impl(False, *args, **kw)

    def exec_once_unless_exception(self, *args: Any, **kw: Any) -> None:
        """Execute this event, but only if it has not been
        executed already for this collection, or was called
        by a previous exec_once_unless_exception call and
        raised an exception.

        If exec_once was already called, then this method will never run
        the callable regardless of whether it raised or not.

        .. versionadded:: 1.3.8

        """
        if not self._exec_once:
            self._exec_once_impl(True, *args, **kw)

    def _exec_w_sync_on_first_run(self, *args: Any, **kw: Any) -> None:
        """Execute this event, and use a mutex if it has not been
        executed already for this collection, or was called
        by a previous _exec_w_sync_on_first_run call and
        raised an exception.

        If _exec_w_sync_on_first_run was already called and didn't raise an
        exception, then a mutex is not used.

        .. versionadded:: 1.4.11

        """
        if not self._exec_w_sync_once:
            with self._exec_once_mutex:
                try:
                    self(*args, **kw)
                except:
                    raise
                else:
                    self._exec_w_sync_once = True
        else:
            self(*args, **kw)

    def __call__(self, *args: Any, **kw: Any) -> None:
        """Execute this event."""

        for fn in self.parent_listeners:
            fn(*args, **kw)
        for fn in self.listeners:
            fn(*args, **kw)

    def __contains__(self, item: Any) -> bool:
        return item in self.parent_listeners or item in self.listeners

    def __len__(self) -> int:
        return len(self.parent_listeners) + len(self.listeners)

    def __iter__(self) -> Iterator[_ListenerFnType]:
        return chain(self.parent_listeners, self.listeners)

    def __bool__(self) -> bool:
        return bool(self.listeners or self.parent_listeners)


class _ListenerCollection(_CompoundListener[_ET]):
    """Instance-level attributes on instances of :class:`._Dispatch`.

    Represents a collection of listeners.

    As of 0.7.9, _ListenerCollection is only first
    created via the _EmptyListener.for_modify() method.

    """

    __slots__ = (
        "parent_listeners",
        "parent",
        "name",
        "listeners",
        "propagate",
        "__weakref__",
    )

    parent_listeners: Collection[_ListenerFnType]
    parent: _ClsLevelDispatch[_ET]
    name: str
    listeners: Deque[_ListenerFnType]
    propagate: Set[_ListenerFnType]

    def __init__(self, parent: _ClsLevelDispatch[_ET], target_cls: Type[_ET]):
        super().__init__()
        if target_cls not in parent._clslevel:
            parent.update_subclass(target_cls)
        self._exec_once = False
        self._exec_w_sync_once = False
        self.parent_listeners = parent._clslevel[target_cls]
        self.parent = parent
        self.name = parent.name
        self.listeners = collections.deque()
        self.propagate = set()

    def for_modify(
        self, obj: _DispatchCommon[_ET]
    ) -> _ListenerCollection[_ET]:
        """Return an event collection which can be modified.

        For _ListenerCollection at the instance level of
        a dispatcher, this returns self.

        """
        return self

    def _update(
        self, other: _ListenerCollection[_ET], only_propagate: bool = True
    ) -> None:
        """Populate from the listeners in another :class:`_Dispatch`
        object."""
        existing_listeners = self.listeners
        existing_listener_set = set(existing_listeners)
        self.propagate.update(other.propagate)
        other_listeners = [
            l
            for l in other.listeners
            if l not in existing_listener_set
            and not only_propagate
            or l in self.propagate
        ]

        existing_listeners.extend(other_listeners)

        if other._is_asyncio:
            self._set_asyncio()

        to_associate = other.propagate.union(other_listeners)
        registry._stored_in_collection_multi(self, other, to_associate)

    def insert(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        if event_key.prepend_to_list(self, self.listeners):
            if propagate:
                self.propagate.add(event_key._listen_fn)

    def append(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        if event_key.append_to_list(self, self.listeners):
            if propagate:
                self.propagate.add(event_key._listen_fn)

    def remove(self, event_key: _EventKey[_ET]) -> None:
        self.listeners.remove(event_key._listen_fn)
        self.propagate.discard(event_key._listen_fn)
        registry._removed_from_collection(event_key, self)

    def clear(self) -> None:
        registry._clear(self, self.listeners)
        self.propagate.clear()
        self.listeners.clear()


class _JoinedListener(_CompoundListener[_ET]):
    __slots__ = "parent_dispatch", "name", "local", "parent_listeners"

    parent_dispatch: _DispatchCommon[_ET]
    name: str
    local: _InstanceLevelDispatch[_ET]
    parent_listeners: Collection[_ListenerFnType]

    def __init__(
        self,
        parent_dispatch: _DispatchCommon[_ET],
        name: str,
        local: _EmptyListener[_ET],
    ):
        self._exec_once = False
        self.parent_dispatch = parent_dispatch
        self.name = name
        self.local = local
        self.parent_listeners = self.local

    if not typing.TYPE_CHECKING:
        # first error, I don't really understand:
        # Signature of "listeners" incompatible with
        # supertype "_CompoundListener"  [override]
        # the name / return type are exactly the same
        # second error is getattr_isn't typed, the cast() here
        # adds too much method overhead
        @property
        def listeners(self) -> Collection[_ListenerFnType]:
            return getattr(self.parent_dispatch, self.name)

    def _adjust_fn_spec(
        self, fn: _ListenerFnType, named: bool
    ) -> _ListenerFnType:
        return self.local._adjust_fn_spec(fn, named)

    def for_modify(self, obj: _DispatchCommon[_ET]) -> _JoinedListener[_ET]:
        self.local = self.parent_listeners = self.local.for_modify(obj)
        return self

    def insert(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        self.local.insert(event_key, propagate)

    def append(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        self.local.append(event_key, propagate)

    def remove(self, event_key: _EventKey[_ET]) -> None:
        self.local.remove(event_key)

    def clear(self) -> None:
        raise NotImplementedError()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tests\test_sync.py ===
from __future__ import annotations

import re
import weakref
from typing import TYPE_CHECKING, Callable, Union

import pytest

from trio.testing import Matcher, RaisesGroup

from .. import _core
from .._core._parking_lot import GLOBAL_PARKING_LOT_BREAKER
from .._sync import *
from .._timeouts import sleep_forever
from ..testing import assert_checkpoints, wait_all_tasks_blocked

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


async def test_Event() -> None:
    e = Event()
    assert not e.is_set()
    assert e.statistics().tasks_waiting == 0

    e.set()
    assert e.is_set()
    with assert_checkpoints():
        await e.wait()

    e = Event()

    record = []

    async def child() -> None:
        record.append("sleeping")
        await e.wait()
        record.append("woken")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(child)
        nursery.start_soon(child)
        await wait_all_tasks_blocked()
        assert record == ["sleeping", "sleeping"]
        assert e.statistics().tasks_waiting == 2
        e.set()
        await wait_all_tasks_blocked()
        assert record == ["sleeping", "sleeping", "woken", "woken"]


async def test_CapacityLimiter() -> None:
    with pytest.raises(TypeError):
        CapacityLimiter(1.0)
    with pytest.raises(ValueError, match=r"^total_tokens must be >= 1$"):
        CapacityLimiter(-1)
    c = CapacityLimiter(2)
    repr(c)  # smoke test
    assert c.total_tokens == 2
    assert c.borrowed_tokens == 0
    assert c.available_tokens == 2
    with pytest.raises(RuntimeError):
        c.release()
    assert c.borrowed_tokens == 0
    c.acquire_nowait()
    assert c.borrowed_tokens == 1
    assert c.available_tokens == 1

    stats = c.statistics()
    assert stats.borrowed_tokens == 1
    assert stats.total_tokens == 2
    assert stats.borrowers == [_core.current_task()]
    assert stats.tasks_waiting == 0

    # Can't re-acquire when we already have it
    with pytest.raises(RuntimeError):
        c.acquire_nowait()
    assert c.borrowed_tokens == 1
    with pytest.raises(RuntimeError):
        await c.acquire()
    assert c.borrowed_tokens == 1

    # We can acquire on behalf of someone else though
    with assert_checkpoints():
        await c.acquire_on_behalf_of("someone")

    # But then we've run out of capacity
    assert c.borrowed_tokens == 2
    with pytest.raises(_core.WouldBlock):
        c.acquire_on_behalf_of_nowait("third party")

    assert set(c.statistics().borrowers) == {_core.current_task(), "someone"}

    # Until we release one
    c.release_on_behalf_of(_core.current_task())
    assert c.statistics().borrowers == ["someone"]

    c.release_on_behalf_of("someone")
    assert c.borrowed_tokens == 0
    with assert_checkpoints():
        async with c:
            assert c.borrowed_tokens == 1

    async with _core.open_nursery() as nursery:
        await c.acquire_on_behalf_of("value 1")
        await c.acquire_on_behalf_of("value 2")
        nursery.start_soon(c.acquire_on_behalf_of, "value 3")
        await wait_all_tasks_blocked()
        assert c.borrowed_tokens == 2
        assert c.statistics().tasks_waiting == 1
        c.release_on_behalf_of("value 2")
        # Fairness:
        assert c.borrowed_tokens == 2
        with pytest.raises(_core.WouldBlock):
            c.acquire_nowait()

    c.release_on_behalf_of("value 3")
    c.release_on_behalf_of("value 1")


async def test_CapacityLimiter_inf() -> None:
    from math import inf

    c = CapacityLimiter(inf)
    repr(c)  # smoke test
    assert c.total_tokens == inf
    assert c.borrowed_tokens == 0
    assert c.available_tokens == inf
    with pytest.raises(RuntimeError):
        c.release()
    assert c.borrowed_tokens == 0
    c.acquire_nowait()
    assert c.borrowed_tokens == 1
    assert c.available_tokens == inf


async def test_CapacityLimiter_change_total_tokens() -> None:
    c = CapacityLimiter(2)

    with pytest.raises(TypeError):
        c.total_tokens = 1.0

    with pytest.raises(ValueError, match=r"^total_tokens must be >= 1$"):
        c.total_tokens = 0

    with pytest.raises(ValueError, match=r"^total_tokens must be >= 1$"):
        c.total_tokens = -10

    assert c.total_tokens == 2

    async with _core.open_nursery() as nursery:
        for i in range(5):
            nursery.start_soon(c.acquire_on_behalf_of, i)
            await wait_all_tasks_blocked()
        assert set(c.statistics().borrowers) == {0, 1}
        assert c.statistics().tasks_waiting == 3
        c.total_tokens += 2
        assert set(c.statistics().borrowers) == {0, 1, 2, 3}
        assert c.statistics().tasks_waiting == 1
        c.total_tokens -= 3
        assert c.borrowed_tokens == 4
        assert c.total_tokens == 1
        c.release_on_behalf_of(0)
        c.release_on_behalf_of(1)
        c.release_on_behalf_of(2)
        assert set(c.statistics().borrowers) == {3}
        assert c.statistics().tasks_waiting == 1
        c.release_on_behalf_of(3)
        assert set(c.statistics().borrowers) == {4}
        assert c.statistics().tasks_waiting == 0


# regression test for issue #548
async def test_CapacityLimiter_memleak_548() -> None:
    limiter = CapacityLimiter(total_tokens=1)
    await limiter.acquire()

    async with _core.open_nursery() as n:
        n.start_soon(limiter.acquire)
        await wait_all_tasks_blocked()  # give it a chance to run the task
        n.cancel_scope.cancel()

    # if this is 1, the acquire call (despite being killed) is still there in the task, and will
    # leak memory all the while the limiter is active
    assert len(limiter._pending_borrowers) == 0


async def test_Semaphore() -> None:
    with pytest.raises(TypeError):
        Semaphore(1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^initial value must be >= 0$"):
        Semaphore(-1)
    s = Semaphore(1)
    repr(s)  # smoke test
    assert s.value == 1
    assert s.max_value is None
    s.release()
    assert s.value == 2
    assert s.statistics().tasks_waiting == 0
    s.acquire_nowait()
    assert s.value == 1
    with assert_checkpoints():
        await s.acquire()
    assert s.value == 0
    with pytest.raises(_core.WouldBlock):
        s.acquire_nowait()

    s.release()
    assert s.value == 1
    with assert_checkpoints():
        async with s:
            assert s.value == 0
    assert s.value == 1
    s.acquire_nowait()

    record = []

    async def do_acquire(s: Semaphore) -> None:
        record.append("started")
        await s.acquire()
        record.append("finished")

    async with _core.open_nursery() as nursery:
        nursery.start_soon(do_acquire, s)
        await wait_all_tasks_blocked()
        assert record == ["started"]
        assert s.value == 0
        s.release()
        # Fairness:
        assert s.value == 0
        with pytest.raises(_core.WouldBlock):
            s.acquire_nowait()
    assert record == ["started", "finished"]


def test_Semaphore_bounded() -> None:
    with pytest.raises(TypeError):
        Semaphore(1, max_value=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"^max_values must be >= initial_value$"):
        Semaphore(2, max_value=1)
    bs = Semaphore(1, max_value=1)
    assert bs.max_value == 1
    repr(bs)  # smoke test
    with pytest.raises(ValueError, match=r"^semaphore released too many times$"):
        bs.release()
    assert bs.value == 1
    bs.acquire_nowait()
    assert bs.value == 0
    bs.release()
    assert bs.value == 1


@pytest.mark.parametrize("lockcls", [Lock, StrictFIFOLock], ids=lambda fn: fn.__name__)
async def test_Lock_and_StrictFIFOLock(
    lockcls: type[Lock | StrictFIFOLock],
) -> None:
    l = lockcls()  # noqa
    assert not l.locked()

    # make sure locks can be weakref'ed (gh-331)
    r = weakref.ref(l)
    assert r() is l

    repr(l)  # smoke test
    # make sure repr uses the right name for subclasses
    assert lockcls.__name__ in repr(l)
    with assert_checkpoints():
        async with l:
            assert l.locked()
            repr(l)  # smoke test (repr branches on locked/unlocked)
    assert not l.locked()
    l.acquire_nowait()
    assert l.locked()
    l.release()
    assert not l.locked()
    with assert_checkpoints():
        await l.acquire()
    assert l.locked()
    l.release()
    assert not l.locked()

    l.acquire_nowait()
    with pytest.raises(RuntimeError):
        # Error out if we already own the lock
        l.acquire_nowait()
    l.release()
    with pytest.raises(RuntimeError):
        # Error out if we don't own the lock
        l.release()

    holder_task = None

    async def holder() -> None:
        nonlocal holder_task
        holder_task = _core.current_task()
        async with l:
            await sleep_forever()

    async with _core.open_nursery() as nursery:
        assert not l.locked()
        nursery.start_soon(holder)
        await wait_all_tasks_blocked()
        assert l.locked()
        # WouldBlock if someone else holds the lock
        with pytest.raises(_core.WouldBlock):
            l.acquire_nowait()
        # Can't release a lock someone else holds
        with pytest.raises(RuntimeError):
            l.release()

        statistics = l.statistics()
        print(statistics)
        assert statistics.locked
        assert statistics.owner is holder_task
        assert statistics.tasks_waiting == 0

        nursery.start_soon(holder)
        await wait_all_tasks_blocked()
        statistics = l.statistics()
        print(statistics)
        assert statistics.tasks_waiting == 1

        nursery.cancel_scope.cancel()

    statistics = l.statistics()
    assert not statistics.locked
    assert statistics.owner is None
    assert statistics.tasks_waiting == 0


async def test_Condition() -> None:
    with pytest.raises(TypeError):
        Condition(Semaphore(1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Condition(StrictFIFOLock)  # type: ignore[arg-type]
    l = Lock()  # noqa
    c = Condition(l)
    assert not l.locked()
    assert not c.locked()
    with assert_checkpoints():
        await c.acquire()
    assert l.locked()
    assert c.locked()

    c = Condition()
    assert not c.locked()
    c.acquire_nowait()
    assert c.locked()
    with pytest.raises(RuntimeError):
        c.acquire_nowait()
    c.release()

    with pytest.raises(RuntimeError):
        # Can't wait without holding the lock
        await c.wait()
    with pytest.raises(RuntimeError):
        # Can't notify without holding the lock
        c.notify()
    with pytest.raises(RuntimeError):
        # Can't notify without holding the lock
        c.notify_all()

    finished_waiters = set()

    async def waiter(i: int) -> None:
        async with c:
            await c.wait()
        finished_waiters.add(i)

    async with _core.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(waiter, i)
            await wait_all_tasks_blocked()
        async with c:
            c.notify()
        assert c.locked()
        await wait_all_tasks_blocked()
        assert finished_waiters == {0}
        async with c:
            c.notify_all()
        await wait_all_tasks_blocked()
        assert finished_waiters == {0, 1, 2}

    finished_waiters = set()
    async with _core.open_nursery() as nursery:
        for i in range(3):
            nursery.start_soon(waiter, i)
            await wait_all_tasks_blocked()
        async with c:
            c.notify(2)
            statistics = c.statistics()
            print(statistics)
            assert statistics.tasks_waiting == 1
            assert statistics.lock_statistics.tasks_waiting == 2
        # exiting the context manager hands off the lock to the first task
        assert c.statistics().lock_statistics.tasks_waiting == 1

        await wait_all_tasks_blocked()
        assert finished_waiters == {0, 1}

        async with c:
            c.notify_all()

    # After being cancelled still hold the lock (!)
    # (Note that c.__aexit__ checks that we hold the lock as well)
    with _core.CancelScope() as scope:
        async with c:
            scope.cancel()
            try:
                await c.wait()
            finally:
                assert c.locked()


from .._channel import open_memory_channel
from .._sync import AsyncContextManagerMixin

# Three ways of implementing a Lock in terms of a channel. Used to let us put
# the channel through the generic lock tests.


class ChannelLock1(AsyncContextManagerMixin):
    def __init__(self, capacity: int) -> None:
        self.s, self.r = open_memory_channel[None](capacity)
        for _ in range(capacity - 1):
            self.s.send_nowait(None)

    def acquire_nowait(self) -> None:
        self.s.send_nowait(None)

    async def acquire(self) -> None:
        await self.s.send(None)

    def release(self) -> None:
        self.r.receive_nowait()


class ChannelLock2(AsyncContextManagerMixin):
    def __init__(self) -> None:
        self.s, self.r = open_memory_channel[None](10)
        self.s.send_nowait(None)

    def acquire_nowait(self) -> None:
        self.r.receive_nowait()

    async def acquire(self) -> None:
        await self.r.receive()

    def release(self) -> None:
        self.s.send_nowait(None)


class ChannelLock3(AsyncContextManagerMixin):
    def __init__(self) -> None:
        self.s, self.r = open_memory_channel[None](0)
        # self.acquired is true when one task acquires the lock and
        # only becomes false when it's released and no tasks are
        # waiting to acquire.
        self.acquired = False

    def acquire_nowait(self) -> None:
        assert not self.acquired
        self.acquired = True

    async def acquire(self) -> None:
        if self.acquired:
            await self.s.send(None)
        else:
            self.acquired = True
            await _core.checkpoint()

    def release(self) -> None:
        try:
            self.r.receive_nowait()
        except _core.WouldBlock:
            assert self.acquired
            self.acquired = False


lock_factories = [
    lambda: CapacityLimiter(1),
    lambda: Semaphore(1),
    Lock,
    StrictFIFOLock,
    lambda: ChannelLock1(10),
    lambda: ChannelLock1(1),
    ChannelLock2,
    ChannelLock3,
]
lock_factory_names = [
    "CapacityLimiter(1)",
    "Semaphore(1)",
    "Lock",
    "StrictFIFOLock",
    "ChannelLock1(10)",
    "ChannelLock1(1)",
    "ChannelLock2",
    "ChannelLock3",
]

generic_lock_test = pytest.mark.parametrize(
    "lock_factory",
    lock_factories,
    ids=lock_factory_names,
)

LockLike: TypeAlias = Union[
    CapacityLimiter,
    Semaphore,
    Lock,
    StrictFIFOLock,
    ChannelLock1,
    ChannelLock2,
    ChannelLock3,
]
LockFactory: TypeAlias = Callable[[], LockLike]


# Spawn a bunch of workers that take a lock and then yield; make sure that
# only one worker is ever in the critical section at a time.
@generic_lock_test
async def test_generic_lock_exclusion(lock_factory: LockFactory) -> None:
    LOOPS = 10
    WORKERS = 5
    in_critical_section = False
    acquires = 0

    async def worker(lock_like: LockLike) -> None:
        nonlocal in_critical_section, acquires
        for _ in range(LOOPS):
            async with lock_like:
                acquires += 1
                assert not in_critical_section
                in_critical_section = True
                await _core.checkpoint()
                await _core.checkpoint()
                assert in_critical_section
                in_critical_section = False

    async with _core.open_nursery() as nursery:
        lock_like = lock_factory()
        for _ in range(WORKERS):
            nursery.start_soon(worker, lock_like)
    assert not in_critical_section
    assert acquires == LOOPS * WORKERS


# Several workers queue on the same lock; make sure they each get it, in
# order.
@generic_lock_test
async def test_generic_lock_fifo_fairness(lock_factory: LockFactory) -> None:
    initial_order = []
    record = []
    LOOPS = 5

    async def loopy(name: int, lock_like: LockLike) -> None:
        # Record the order each task was initially scheduled in
        initial_order.append(name)
        for _ in range(LOOPS):
            async with lock_like:
                record.append(name)

    lock_like = lock_factory()
    async with _core.open_nursery() as nursery:
        nursery.start_soon(loopy, 1, lock_like)
        nursery.start_soon(loopy, 2, lock_like)
        nursery.start_soon(loopy, 3, lock_like)
    # The first three could be in any order due to scheduling randomness,
    # but after that they should repeat in the same order
    for i in range(LOOPS):
        assert record[3 * i : 3 * (i + 1)] == initial_order


@generic_lock_test
async def test_generic_lock_acquire_nowait_blocks_acquire(
    lock_factory: LockFactory,
) -> None:
    lock_like = lock_factory()

    record = []

    async def lock_taker() -> None:
        record.append("started")
        async with lock_like:
            pass
        record.append("finished")

    async with _core.open_nursery() as nursery:
        lock_like.acquire_nowait()
        nursery.start_soon(lock_taker)
        await wait_all_tasks_blocked()
        assert record == ["started"]
        lock_like.release()


async def test_lock_acquire_unowned_lock() -> None:
    """Test that trying to acquire a lock whose owner has exited raises an error.
    see https://github.com/python-trio/trio/issues/3035
    """
    assert not GLOBAL_PARKING_LOT_BREAKER
    lock = trio.Lock()
    async with trio.open_nursery() as nursery:
        nursery.start_soon(lock.acquire)
    owner_str = re.escape(str(lock._lot.broken_by[0]))
    with pytest.raises(
        trio.BrokenResourceError,
        match=f"^Owner of this lock exited without releasing: {owner_str}$",
    ):
        await lock.acquire()
    assert not GLOBAL_PARKING_LOT_BREAKER


async def test_lock_multiple_acquire() -> None:
    """Test for error if awaiting on a lock whose owner exits without releasing.
    see https://github.com/python-trio/trio/issues/3035"""
    assert not GLOBAL_PARKING_LOT_BREAKER
    lock = trio.Lock()
    with RaisesGroup(
        Matcher(
            trio.BrokenResourceError,
            match="^Owner of this lock exited without releasing: ",
        ),
    ):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(lock.acquire)
            nursery.start_soon(lock.acquire)
    assert not GLOBAL_PARKING_LOT_BREAKER


async def test_lock_handover() -> None:
    assert not GLOBAL_PARKING_LOT_BREAKER
    child_task: Task | None = None
    lock = trio.Lock()

    # this task acquires the lock
    lock.acquire_nowait()
    assert {
        _core.current_task(): [
            lock._lot,
        ],
    } == GLOBAL_PARKING_LOT_BREAKER

    async with trio.open_nursery() as nursery:
        nursery.start_soon(lock.acquire)
        await wait_all_tasks_blocked()

        # hand over the lock to the child task
        lock.release()

        # check values, and get the identifier out of the dict for later check
        assert len(GLOBAL_PARKING_LOT_BREAKER) == 1
        child_task = next(iter(GLOBAL_PARKING_LOT_BREAKER))
        assert GLOBAL_PARKING_LOT_BREAKER[child_task] == [lock._lot]

    assert lock._lot.broken_by == [child_task]
    assert not GLOBAL_PARKING_LOT_BREAKER

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\webelement.py ===
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
from __future__ import annotations

import os
import pkgutil
import warnings
import zipfile
from abc import ABCMeta
from base64 import b64decode, encodebytes
from hashlib import md5 as md5_hash
from io import BytesIO
from typing import List

from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.utils import keys_to_typing

from .command import Command
from .shadowroot import ShadowRoot

# TODO: When moving to supporting python 3.9 as the minimum version we can
# use built in importlib_resources.files.
getAttribute_js = None
isDisplayed_js = None


def _load_js():
    global getAttribute_js
    global isDisplayed_js
    _pkg = ".".join(__name__.split(".")[:-1])
    getAttribute_js = pkgutil.get_data(_pkg, "getAttribute.js").decode("utf8")
    isDisplayed_js = pkgutil.get_data(_pkg, "isDisplayed.js").decode("utf8")


class BaseWebElement(metaclass=ABCMeta):
    """Abstract Base Class for WebElement.

    ABC's will allow custom types to be registered as a WebElement to
    pass type checks.
    """

    pass


class WebElement(BaseWebElement):
    """Represents a DOM element.

    Generally, all interesting operations that interact with a document will be
    performed through this interface.

    All method calls will do a freshness check to ensure that the element
    reference is still valid.  This essentially determines whether the
    element is still attached to the DOM.  If this test fails, then an
    ``StaleElementReferenceException`` is thrown, and all future calls to this
    instance will fail.
    """

    def __init__(self, parent, id_) -> None:
        self._parent = parent
        self._id = id_

    def __repr__(self):
        return f'<{type(self).__module__}.{type(self).__name__} (session="{self.session_id}", element="{self._id}")>'

    @property
    def session_id(self) -> str:
        return self._parent.session_id

    @property
    def tag_name(self) -> str:
        """This element's ``tagName`` property.

        Returns:
        --------
        str : The tag name of the element.

        Example:
        --------
        >>> element = driver.find_element(By.ID, "foo")
        """
        return self._execute(Command.GET_ELEMENT_TAG_NAME)["value"]

    @property
    def text(self) -> str:
        """The text of the element.

        Returns:
        --------
        str : The text of the element.

        Example:
        --------
        >>> element = driver.find_element(By.ID, "foo")
        >>> print(element.text)
        """
        return self._execute(Command.GET_ELEMENT_TEXT)["value"]

    def click(self) -> None:
        """Clicks the element.

        Example:
        --------
        >>> element = driver.find_element(By.ID, "foo")
        >>> element.click()
        """
        self._execute(Command.CLICK_ELEMENT)

    def submit(self) -> None:
        """Submits a form.

        Example:
        --------
        >>> form = driver.find_element(By.NAME, "login")
        >>> form.submit()
        """
        script = (
            "/* submitForm */var form = arguments[0];\n"
            'while (form.nodeName != "FORM" && form.parentNode) {\n'
            "  form = form.parentNode;\n"
            "}\n"
            "if (!form) { throw Error('Unable to find containing form element'); }\n"
            "if (!form.ownerDocument) { throw Error('Unable to find owning document'); }\n"
            "var e = form.ownerDocument.createEvent('Event');\n"
            "e.initEvent('submit', true, true);\n"
            "if (form.dispatchEvent(e)) { HTMLFormElement.prototype.submit.call(form) }\n"
        )

        try:
            self._parent.execute_script(script, self)
        except JavascriptException as exc:
            raise WebDriverException("To submit an element, it must be nested inside a form element") from exc

    def clear(self) -> None:
        """Clears the text if it's a text entry element.

        Example:
        --------
        >>> text_field = driver.find_element(By.NAME, "username")
        >>> text_field.clear()
        """
        self._execute(Command.CLEAR_ELEMENT)

    def get_property(self, name) -> str | bool | WebElement | dict:
        """Gets the given property of the element.

        Parameters:
        -----------
        name : str
            - Name of the property to retrieve.

        Returns:
        -------
        str | bool | WebElement | dict : The value of the property.

        Example:
        -------
        >>> text_length = target_element.get_property("text_length")
        """
        try:
            return self._execute(Command.GET_ELEMENT_PROPERTY, {"name": name})["value"]
        except WebDriverException:
            # if we hit an end point that doesn't understand getElementProperty lets fake it
            return self.parent.execute_script("return arguments[0][arguments[1]]", self, name)

    def get_dom_attribute(self, name) -> str:
        """Gets the given attribute of the element. Unlike
        :func:`~selenium.webdriver.remote.BaseWebElement.get_attribute`, this
        method only returns attributes declared in the element's HTML markup.

        Parameters:
        -----------
        name : str
            - Name of the attribute to retrieve.

        Returns:
        -------
        str : The value of the attribute.

        Example:
        -------
        >>> text_length = target_element.get_dom_attribute("class")
        """
        return self._execute(Command.GET_ELEMENT_ATTRIBUTE, {"name": name})["value"]

    def get_attribute(self, name) -> str | None:
        """Gets the given attribute or property of the element.

        This method will first try to return the value of a property with the
        given name. If a property with that name doesn't exist, it returns the
        value of the attribute with the same name. If there's no attribute with
        that name, ``None`` is returned.

        Values which are considered truthy, that is equals "true" or "false",
        are returned as booleans.  All other non-``None`` values are returned
        as strings.  For attributes or properties which do not exist, ``None``
        is returned.

        To obtain the exact value of the attribute or property,
        use :func:`~selenium.webdriver.remote.BaseWebElement.get_dom_attribute` or
        :func:`~selenium.webdriver.remote.BaseWebElement.get_property` methods respectively.

        Parameters:
        -----------
        name : str
            - Name of the attribute/property to retrieve.

        Returns:
        -------
        str | bool | None : The value of the attribute/property.

        Example:
        --------
        >>> # Check if the "active" CSS class is applied to an element.
        >>> is_active = "active" in target_element.get_attribute("class")
        """
        if getAttribute_js is None:
            _load_js()
        attribute_value = self.parent.execute_script(
            f"/* getAttribute */return ({getAttribute_js}).apply(null, arguments);", self, name
        )
        return attribute_value

    def is_selected(self) -> bool:
        """Returns whether the element is selected.

        Example:
        --------
        >>> is_selected = element.is_selected()

        Notes:
        ------
            - This method is generally used on checkboxes, options in a select
            and radio buttons.
        """
        return self._execute(Command.IS_ELEMENT_SELECTED)["value"]

    def is_enabled(self) -> bool:
        """Returns whether the element is enabled.

        Example:
        --------
        >>> is_enabled = element.is_enabled()
        """
        return self._execute(Command.IS_ELEMENT_ENABLED)["value"]

    def send_keys(self, *value: str) -> None:
        """Simulates typing into the element.

        Parameters:
        -----------
        value : str
            - A string for typing, or setting form fields.  For setting
            file inputs, this could be a local file path.

        Notes:
        ------
            - Use this to send simple key events or to fill out form fields
            - This can also be used to set file inputs.

        Examples:
        --------
        To send a simple key event::
        >>> form_textfield = driver.find_element(By.NAME, "username")
        >>> form_textfield.send_keys("admin")

        or to set a file input field::
        >>> file_input = driver.find_element(By.NAME, "profilePic")
        >>> file_input.send_keys("path/to/profilepic.gif")
        >>> # Generally it's better to wrap the file path in one of the methods
        >>> # in os.path to return the actual path to support cross OS testing.
        >>> # file_input.send_keys(os.path.abspath("path/to/profilepic.gif"))
        >>> # When using Cygwin, the path need to be provided in Windows format.
        >>> # file_input.send_keys(f"C:/cygwin{os.path.abspath('path/to/profilepic.gif').replace('/', '\\')}")
        """
        # transfer file to another machine only if remote driver is used
        # the same behaviour as for java binding
        if self.parent._is_remote:
            local_files = list(
                map(
                    lambda keys_to_send: self.parent.file_detector.is_local_file(str(keys_to_send)),
                    "".join(map(str, value)).split("\n"),
                )
            )
            if None not in local_files:
                remote_files = []
                for file in local_files:
                    remote_files.append(self._upload(file))
                value = tuple("\n".join(remote_files))

        self._execute(
            Command.SEND_KEYS_TO_ELEMENT, {"text": "".join(keys_to_typing(value)), "value": keys_to_typing(value)}
        )

    @property
    def shadow_root(self) -> ShadowRoot:
        """Returns a shadow root of the element if there is one or an error.
        Only works from Chromium 96, Firefox 96, and Safari 16.4 onwards.

        Returns:
        -------
        ShadowRoot : object

        Raises:
        -------
        NoSuchShadowRoot - if no shadow root was attached to element

        Example:
        --------
        >>> try:
        ...     shadow_root = element.shadow_root
        >>> except NoSuchShadowRoot:
        ...     print("No shadow root attached to element")
        """
        return self._execute(Command.GET_SHADOW_ROOT)["value"]

    # RenderedWebElement Items
    def is_displayed(self) -> bool:
        """Whether the element is visible to a user.

        Example:
        --------
        >>> is_displayed = element.is_displayed()
        """
        # Only go into this conditional for browsers that don't use the atom themselves
        if isDisplayed_js is None:
            _load_js()
        return self.parent.execute_script(f"/* isDisplayed */return ({isDisplayed_js}).apply(null, arguments);", self)

    @property
    def location_once_scrolled_into_view(self) -> dict:
        """THIS PROPERTY MAY CHANGE WITHOUT WARNING. Use this to discover where
        on the screen an element is so that we can click it. This method should
        cause the element to be scrolled into view.

        Returns:
        --------
        dict: the top lefthand corner location on the screen, or zero
            coordinates if the element is not visible.

        Example:
        --------
        >>> loc = element.location_once_scrolled_into_view
        """
        old_loc = self._execute(
            Command.W3C_EXECUTE_SCRIPT,
            {
                "script": "arguments[0].scrollIntoView(true); return arguments[0].getBoundingClientRect()",
                "args": [self],
            },
        )["value"]
        return {"x": round(old_loc["x"]), "y": round(old_loc["y"])}

    @property
    def size(self) -> dict:
        """The size of the element.

        Returns:
        --------
        dict: The width and height of the element.

        Example:
        --------
        >>> size = element.size
        """
        size = self._execute(Command.GET_ELEMENT_RECT)["value"]
        new_size = {"height": size["height"], "width": size["width"]}
        return new_size

    def value_of_css_property(self, property_name) -> str:
        """The value of a CSS property.

        Parameters:
        -----------
        property_name : str
            - The name of the CSS property to get the value of.

        Returns:
        --------
        str : The value of the CSS property.

        Example:
        --------
        >>> value = element.value_of_css_property("color")
        """
        return self._execute(Command.GET_ELEMENT_VALUE_OF_CSS_PROPERTY, {"propertyName": property_name})["value"]

    @property
    def location(self) -> dict:
        """The location of the element in the renderable canvas.

        Returns:
        --------
        dict: The x and y coordinates of the element.

        Example:
        --------
        >>> loc = element.location
        """
        old_loc = self._execute(Command.GET_ELEMENT_RECT)["value"]
        new_loc = {"x": round(old_loc["x"]), "y": round(old_loc["y"])}
        return new_loc

    @property
    def rect(self) -> dict:
        """A dictionary with the size and location of the element.

        Returns:
        --------
        dict: The size and location of the element.

        Example:
        --------
        >>> rect = element.rect
        """
        return self._execute(Command.GET_ELEMENT_RECT)["value"]

    @property
    def aria_role(self) -> str:
        """Returns the ARIA role of the current web element.

        Returns:
        --------
        str : The ARIA role of the element.

        Example:
        --------
        >>> role = element.aria_role
        """
        return self._execute(Command.GET_ELEMENT_ARIA_ROLE)["value"]

    @property
    def accessible_name(self) -> str:
        """Returns the ARIA Level of the current webelement.

        Returns:
        --------
        str : The ARIA Level of the element.

        Example:
        --------
        >>> name = element.accessible_name
        """
        return self._execute(Command.GET_ELEMENT_ARIA_LABEL)["value"]

    @property
    def screenshot_as_base64(self) -> str:
        """Gets the screenshot of the current element as a base64 encoded
        string.

        Returns:
        --------
        str : The screenshot of the element as a base64 encoded string.

        Example:
        --------
        >>> img_b64 = element.screenshot_as_base64
        """
        return self._execute(Command.ELEMENT_SCREENSHOT)["value"]

    @property
    def screenshot_as_png(self) -> bytes:
        """Gets the screenshot of the current element as a binary data.

        Returns:
        --------
        bytes : The screenshot of the element as binary data.

        Example:
        --------
        >>> element_png = element.screenshot_as_png
        """
        return b64decode(self.screenshot_as_base64.encode("ascii"))

    def screenshot(self, filename) -> bool:
        """Saves a screenshot of the current element to a PNG image file.
        Returns False if there is any IOError, else returns True. Use full
        paths in your filename.

        Returns:
        --------
        bool : True if the screenshot was saved successfully, False otherwise.

        Parameters:
        -----------
        filename : str
            The full path you wish to save your screenshot to. This
            should end with a `.png` extension.

        Element:
        --------
        >>> element.screenshot("/Screenshots/foo.png")
        """
        if not filename.lower().endswith(".png"):
            warnings.warn(
                "name used for saved screenshot does not match file type. It should end with a `.png` extension",
                UserWarning,
            )
        png = self.screenshot_as_png
        try:
            with open(filename, "wb") as f:
                f.write(png)
        except OSError:
            return False
        finally:
            del png
        return True

    @property
    def parent(self):
        """Internal reference to the WebDriver instance this element was found
        from.

        Example:
        --------
        >>> element = driver.find_element(By.ID, "foo")
        >>> parent_element = element.parent
        """
        return self._parent

    @property
    def id(self) -> str:
        """Internal ID used by selenium.

        This is mainly for internal use. Simple use cases such as checking if 2
        webelements refer to the same element, can be done using ``==``::

        Example:
        --------
        >>> if element1 == element2:
        ...     print("These 2 are equal")
        """
        return self._id

    def __eq__(self, element):
        return hasattr(element, "id") and self._id == element.id

    def __ne__(self, element):
        return not self.__eq__(element)

    # Private Methods
    def _execute(self, command, params=None):
        """Executes a command against the underlying HTML element.

        Parameters:
        -----------
        command : any
            The name of the command to _execute as a string.

        params : dict
            A dictionary of named Parameters to send with the command.

        Returns:
        -------
          The command's JSON response loaded into a dictionary object.
        """
        if not params:
            params = {}
        params["id"] = self._id
        return self._parent.execute(command, params)

    def find_element(self, by=By.ID, value=None) -> WebElement:
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
        by, value = self._parent.locator_converter.convert(by, value)
        return self._execute(Command.FIND_CHILD_ELEMENT, {"using": by, "value": value})["value"]

    def find_elements(self, by=By.ID, value=None) -> List[WebElement]:
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
        >>> element = driver.find_elements(By.ID, "foo")

        Returns:
        -------
        List[WebElement]
            list of `WebElements` matching locator strategy found on the page.
        """
        by, value = self._parent.locator_converter.convert(by, value)
        return self._execute(Command.FIND_CHILD_ELEMENTS, {"using": by, "value": value})["value"]

    def __hash__(self) -> int:
        return int(md5_hash(self._id.encode("utf-8")).hexdigest(), 16)

    def _upload(self, filename):
        fp = BytesIO()
        zipped = zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED)
        zipped.write(filename, os.path.split(filename)[1])
        zipped.close()
        content = encodebytes(fp.getvalue())
        if not isinstance(content, str):
            content = content.decode("utf-8")
        try:
            return self._execute(Command.UPLOAD_FILE, {"file": content})["value"]
        except WebDriverException as e:
            if "Unrecognized command: POST" in str(e):
                return filename
            if "Command not found: POST " in str(e):
                return filename
            if '{"status":405,"value":["GET","HEAD","DELETE"]}' in str(e):
                return filename
            raise

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\operator_type_usage_processors.py ===
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import json
from abc import ABC, abstractmethod

import ort_flatbuffers_py.fbs as fbs

from .types import FbsTypeInfo, value_name_to_typestr


def _create_op_key(domain: str, optype: str):
    return f"{domain}:{optype}"


def _ort_constant_for_domain(domain: str):
    """
    Map a string domain value to the internal ONNX Runtime constant for that domain.
    :param domain: Domain string to map.
    :return: Internal ONNX Runtime constant
    """

    # constants are defined in <ORT root>/include/onnxruntime/core/graph/constants.h
    # This list is limited to just the domains we have processors for
    domain_to_constant_map = {"ai.onnx": "kOnnxDomain", "ai.onnx.ml": "kMLDomain", "com.microsoft": "kMSDomain"}

    if domain not in domain_to_constant_map:
        raise ValueError(f"Domain {domain} not found in map to ONNX Runtime constant. Please update map.")

    return domain_to_constant_map[domain]


def _reg_type_to_cpp_type(reg_type: str):
    if reg_type == "string":
        return "std::string"
    return reg_type


def _split_reg_types(reg_types_str: str):
    """
    Split on underscores but append "_t" to the previous element.
    """
    tokens = reg_types_str.split("_")
    reg_types = []
    for token in tokens:
        if token == "t" and len(reg_types) > 0:
            reg_types[-1] += "_t"
        else:
            reg_types += [token]
    return reg_types


class TypeUsageProcessor(ABC):
    """
    Abstract base class for processors which implement operator specific logic to determine the type or types required.
    """

    def __init__(self, domain: str, optype: str):
        self.domain = domain
        self.optype = optype
        self.name = _create_op_key(domain, optype)

    @abstractmethod
    def process_node(self, node: fbs.Node, value_name_to_typeinfo: dict):
        pass

    def is_typed_registration_needed(self, type_in_registration: str, globally_allowed_types: set[str] | None):
        """
        Given the string from a kernel registration, determine if the registration is required or not.
        :param type_in_registration: Type string from kernel registration
        :param globally_allowed_types: Optional set of globally allowed types. If provided, these types take precedence
                                       in determining the required types.
        :return: True is required. False if not.
        """
        # Not all operators have typed registrations, so this is optionally implemented by derived classes
        raise RuntimeError(f"Did not expect processor for {self.name} to have typed registrations.")

    def get_cpp_entry(self):
        """
        Get the C++ code that specifies this operator's required types.
        :return: List with any applicable C++ code for this operator's required types. One line per entry.
        """
        # Not applicable for some ops, so return no lines by default.
        return []

    @abstractmethod
    def to_config_entry(self):
        """
        Generate a configuration file entry in JSON format with the required types for the operator.
        :return: JSON string with required type information.
        """

    @abstractmethod
    def from_config_entry(self, entry: str):
        """
        Re-create the types required from a configuration file entry created with to_config_entry.
        NOTE: Any existing type information should be cleared prior to re-creating from a config file entry.
        :param entry: Configuration file entry
        """


class DefaultTypeUsageProcessor(TypeUsageProcessor):
    """
    Operator processor which tracks the types used for selected input/s and/or output/s.
    """

    def __init__(
        self,
        domain: str,
        optype: str,
        inputs: [int] = [0],  # noqa: B006
        outputs: [int] = [],  # noqa: B006
        required_input_types: dict[int, set[str]] = {},  # noqa: B006
        required_output_types: dict[int, set[str]] = {},  # noqa: B006
    ):
        """
        Create DefaultTypeUsageProcessor. Types for one or more inputs and/or outputs can be tracked by the processor.
        The default is to track the types required for input 0, as this is the most common use case in ONNX.

        Required input and output types may be specified. These are only applicable to is_typed_registration_needed().
        If a registration type matches a required type, the typed registration is needed.
        There is a separate mechanism for specifying required types from C++ for kernels with untyped registration.

        :param domain: Operator domain.
        :param optype: Operator name.
        :param inputs: Inputs to track. Zero based index. May be empty.
        :param outputs: Outputs to track. Zero based index. May be empty.
        :param required_input_types: Required input types. May be empty.
        :param required_output_types: Required output types. May be empty.
        """
        super().__init__(domain, optype)
        self._input_types = {}
        self._output_types = {}

        for i in inputs:
            self._input_types[i] = set()

        for o in outputs:
            self._output_types[o] = set()

        if not inputs and not outputs:
            raise ValueError("At least one input or output must be tracked")

        self._required_input_types = required_input_types
        self._required_output_types = required_output_types

    def _is_type_enabled(self, reg_type, index, required_types, allowed_type_set):
        cpp_type = _reg_type_to_cpp_type(reg_type)
        return cpp_type in required_types.get(index, set()) or cpp_type in allowed_type_set

    def is_input_type_enabled(self, reg_type, index, allowed_type_set=None):
        """Whether input type is enabled based on required and allowed types."""
        if allowed_type_set is None:
            allowed_type_set = self._input_types[index]
        return self._is_type_enabled(reg_type, index, self._required_input_types, allowed_type_set)

    def is_output_type_enabled(self, reg_type, index, allowed_type_set=None):
        """Whether output type is enabled based on required and allowed types."""
        if allowed_type_set is None:
            allowed_type_set = self._output_types[index]
        return self._is_type_enabled(reg_type, index, self._required_output_types, allowed_type_set)

    def process_node(self, node: fbs.Node, value_name_to_typeinfo: dict):
        for i in self._input_types:
            if i >= node.InputsLength():
                # Some operators have fewer inputs in earlier versions where data that was as an attribute
                # become an input in later versions to allow it to be dynamically provided. Allow for that.
                # e.g. Slice-1 had attributes for the indices, and Slice-10 moved those to be inputs
                # raise RuntimeError('Node has {} outputs. Tracker for {} incorrectly configured as it requires {}.'
                #                    .format(node.OutputsLength(), self.name, o))
                pass
            else:
                type_str = value_name_to_typestr(node.Inputs(i), value_name_to_typeinfo)
                self._input_types[i].add(type_str)

        for o in self._output_types:
            # Don't know of any ops where the number of outputs changed across versions, so require a valid length
            if o >= node.OutputsLength():
                raise RuntimeError(
                    f"Node has {node.OutputsLength()} outputs. Tracker for {self.name} incorrectly configured as it requires {o}."
                )

            type_str = value_name_to_typestr(node.Outputs(o), value_name_to_typeinfo)
            self._output_types[o].add(type_str)

    def is_typed_registration_needed(self, type_in_registration: str, globally_allowed_types: set[str] | None):
        if 0 not in self._input_types:
            # currently all standard typed registrations are for input 0.
            # custom registrations can be handled by operator specific processors (e.g. OneHotProcessor below).
            raise RuntimeError(f"Expected typed registration to use type from input 0. Node:{self.name}")

        return self.is_input_type_enabled(type_in_registration, 0, globally_allowed_types)

    def get_cpp_entry(self):
        entries = []
        domain = _ort_constant_for_domain(self.domain)
        for i in sorted(self._input_types.keys()):
            if self._input_types[i]:
                entries.append(
                    "ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES({}, {}, Input, {}, {});".format(
                        domain, self.optype, i, ", ".join(sorted(self._input_types[i]))
                    )
                )

        for o in sorted(self._output_types.keys()):
            if self._output_types[o]:
                entries.append(
                    "ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES({}, {}, Output, {}, {});".format(
                        domain, self.optype, o, ", ".join(sorted(self._output_types[o]))
                    )
                )

        return entries

    def to_config_entry(self):
        # convert the sets of types to lists so they can easily written out using the json model
        aggregate_info = {"inputs": {}, "outputs": {}}

        # filter out empty entries and sort the types
        for i in sorted(self._input_types.keys()):
            if self._input_types[i]:
                aggregate_info["inputs"][i] = sorted(self._input_types[i])

        for o in sorted(self._output_types.keys()):
            if self._output_types[o]:
                aggregate_info["outputs"][o] = sorted(self._output_types[o])

        # remove any empty keys
        if not aggregate_info["inputs"]:
            aggregate_info.pop("inputs")
        if not aggregate_info["outputs"]:
            aggregate_info.pop("outputs")

        entry = json.dumps(aggregate_info) if aggregate_info else None
        return entry

    def from_config_entry(self, entry: str):
        self._input_types.clear()
        self._output_types.clear()

        aggregate_info = json.loads(entry)
        if "inputs" in aggregate_info:
            for i_str, values in aggregate_info["inputs"].items():
                self._input_types[int(i_str)] = set(values)

        if "outputs" in aggregate_info:
            for o_str, values in aggregate_info["outputs"].items():
                self._output_types[int(o_str)] = set(values)


class Input1TypedRegistrationProcessor(DefaultTypeUsageProcessor):
    """
    Processor for operators where the second input type is used in a typed kernel registration.
    """

    def __init__(self, domain: str, optype: str):
        # init with tracking of input 1 only.
        super().__init__(domain, optype, inputs=[1], outputs=[])

    def is_typed_registration_needed(self, type_in_registration: str, globally_allowed_types: set[str] | None):
        return self.is_input_type_enabled(type_in_registration, 1, globally_allowed_types)


class Output0TypedRegistrationProcessor(DefaultTypeUsageProcessor):
    """
    Processor for operators where the first output type is used in a typed kernel registration.
    """

    def __init__(self, domain: str, optype: str):
        # init with tracking of output 0 only.
        super().__init__(domain, optype, inputs=[], outputs=[0])

    def is_typed_registration_needed(self, type_in_registration: str, globally_allowed_types: set[str] | None):
        return self.is_output_type_enabled(type_in_registration, 0, globally_allowed_types)


class OneHotProcessor(TypeUsageProcessor):
    """
    Processor for the OneHot operator, which requires custom logic as the type registration key is a concatenation of
    the three types involved instead of a single type name.
    """

    def __init__(self):
        super().__init__("ai.onnx", "OneHot")
        self._triples = set()

    def process_node(self, node: fbs.Node, value_name_to_typeinfo: dict):
        type0 = value_name_to_typestr(node.Inputs(0), value_name_to_typeinfo)
        type1 = value_name_to_typestr(node.Inputs(1), value_name_to_typeinfo)
        type2 = value_name_to_typestr(node.Inputs(2), value_name_to_typeinfo)
        # types in kernel registration are ordered this way: input (T1), output (T3), depth (T2)
        key = (type0, type2, type1)
        self._triples.add(key)

    def is_typed_registration_needed(self, type_in_registration: str, globally_allowed_types: set[str] | None):
        # the OneHot registration involves a concatenation of the 3 types involved
        reg_types = tuple([_reg_type_to_cpp_type(reg_type) for reg_type in _split_reg_types(type_in_registration)])
        if globally_allowed_types is not None:
            return all(reg_type in globally_allowed_types for reg_type in reg_types)
        else:
            return reg_types in self._triples

    def to_config_entry(self):
        if not self._triples:
            return None

        aggregate_info = {"custom": sorted(self._triples)}
        entry = json.dumps(aggregate_info)
        return entry

    def from_config_entry(self, entry: str):
        self._triples.clear()
        aggregate_info = json.loads(entry)
        if "custom" in aggregate_info:
            self._triples = {tuple(triple) for triple in aggregate_info["custom"]}


def _create_operator_type_usage_processors():
    """
    Create a set of processors that determine the required types for all enabled operators.
    :return: Dictionary of operator key to processor. Key is 'domain:operator (e.g. ai.onnx:Cast)'.
    """
    operator_processors = {}

    def add(processor):
        if processor.name in operator_processors:
            raise RuntimeError("Duplicate processor for " + processor.name)

        operator_processors[processor.name] = processor

    # Starting with ops from:
    #   - Priority 1P models
    #   - Mobilenet + SSD Mobilenet + MobileBert
    #   - some known large kernels
    #
    # Ops we are ignoring currently so as not to produce meaningless/unused output:
    # - Implementation is type agnostic:
    #    ai.onnx: If, Loop, Reshape, Scan, Shape, Squeeze, Tile, Unsqueeze
    #    com.microsoft: DynamicQuantizeMatMul, MatMulIntegerToFloat
    # - Only one type supported in the ORT implementation:
    #    ai.onnx: NonMaxSuppression
    #    com.microsoft: FusedConv, FusedGemm, FusedMatMul
    # - Implementation does not have any significant type specific code:
    #    ai.onnx: Concat, Flatten, Not, Reshape, Shape, Squeeze, Unsqueeze
    #
    default_processor_onnx_ops = [
        "Abs",
        "ArgMax",
        "ArgMin",
        "AveragePool",
        "BatchNormalization",
        "BitShift",
        "Ceil",
        "Clip",
        "Conv",
        "CumSum",
        "Exp",
        "Expand",
        "Floor",
        "Gemm",
        "IsNaN",
        "Log",
        "LogSoftmax",
        "LpNormalization",
        "MatMul",
        "Max",
        "MaxPool",
        "Mean",
        "Min",
        "NonZero",
        "Pad",
        "QLinearConv",
        "QLinearMatMul",
        "Range",
        "Reciprocal",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
        "Relu",
        "Resize",
        "ReverseSequence",
        "RoiAlign",
        "Round",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "Shrink",
        "Sigmoid",
        "Sign",
        "Sin",
        "Softmax",
        "Split",
        "SplitToSequence",
        "Sqrt",
        "Sum",
        "Tanh",
        "TopK",
        "Transpose",
        "Unique",
    ]

    # ops that are used to manipulate shapes or indices so require int32_t and int64_t to be available
    default_processor_onnx_ops_requiring_ints_for_input_0 = [
        "Add",
        "Concat",
        "Div",
        "Equal",
        "Greater",
        "Less",
        "Mul",
        "Neg",  # used in tflite TransposeConv conversion
        "Sub",
    ]

    # NOTE: QLinearConv has ONNX and internal implementations
    internal_ops = ["QLinearAdd", "QLinearMul", "QLinearConv"]

    # TODO - review and add ML ops as needed
    # ML Op notes.
    #  CastMap: Switch on value type of input map type, and output type
    #  DictVectorizer: Templatized on key+value of input so need to handle like OneHot with custom processor
    #  LabelEncoder: Implementation switches on input and output types (only supports string and int64 in T1 and T2)
    #  LinearClassifier: Internal switch on input type and also switch on output type
    #  SVMClassifier: ditto
    #  TreeEnsembleClassifier: Templatized on input type and also switch on output type
    #  ZipMap: Switch on output type (derived from attributes)
    default_processor_onnxml_ops = []

    [add(DefaultTypeUsageProcessor("ai.onnx", op)) for op in default_processor_onnx_ops]
    [
        add(DefaultTypeUsageProcessor("ai.onnx", op, required_input_types={0: {"int32_t", "int64_t"}}))
        for op in default_processor_onnx_ops_requiring_ints_for_input_0
    ]
    [add(DefaultTypeUsageProcessor("ai.onnx.ml", op)) for op in default_processor_onnxml_ops]
    [add(DefaultTypeUsageProcessor("com.microsoft", op)) for op in internal_ops]

    #
    # Operators that require custom handling
    #

    # Cast switches on types of input 0 and output 0
    add(DefaultTypeUsageProcessor("ai.onnx", "Cast", inputs=[0], outputs=[0]))

    # Operators that switch on the type of input 0 and 1
    add(DefaultTypeUsageProcessor("ai.onnx", "Gather", inputs=[0, 1]))
    add(DefaultTypeUsageProcessor("ai.onnx", "GatherElements", inputs=[0, 1]))
    add(DefaultTypeUsageProcessor("ai.onnx", "Pow", inputs=[0, 1]))
    add(DefaultTypeUsageProcessor("ai.onnx", "Slice", inputs=[0, 1]))

    # Operators that switch on output type
    add(DefaultTypeUsageProcessor("ai.onnx", "ConstantOfShape", inputs=[], outputs=[0]))

    # Random generator ops produce new data so we track the output type
    onnx_random_ops = ["RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Multinomial"]
    [add(DefaultTypeUsageProcessor("ai.onnx", op, inputs=[], outputs=[0])) for op in onnx_random_ops]

    # Where always has a boolean first input so track the second input type for typed registration
    add(Input1TypedRegistrationProcessor("ai.onnx", "Where"))

    # we only support 'float' as input for [Dynamic]QuantizeLinear so just track the output type
    # as that's what is used in the typed registration
    add(Output0TypedRegistrationProcessor("ai.onnx", "QuantizeLinear"))
    add(Output0TypedRegistrationProcessor("ai.onnx", "DynamicQuantizeLinear"))

    # make sure all the dequantize types are enabled. we use int32_t for parts of GEMM and Conv so just
    # enabling int8 and uint8 is not enough.
    # TODO: Only apply required types to the global type list and ignore if it's model based per-op type reduction
    add(
        DefaultTypeUsageProcessor(
            "ai.onnx", "DequantizeLinear", inputs=[0], required_input_types={0: {"int8_t", "uint8_t", "int32_t"}}
        )
    )

    # OneHot concatenates type strings into a triple in the typed registration
    #   e.g. float_int64_t_int64_t
    add(OneHotProcessor())

    return operator_processors


class OpTypeImplFilterInterface(ABC):
    """
    Class that filters operator implementations based on type.
    """

    @abstractmethod
    def is_typed_registration_needed(self, domain: str, optype: str, type_registration_str: str):
        """
        Given the string from a kernel registration, determine if the registration is required or not.
        :param domain: Operator domain.
        :param optype: Operator type.
        :param type_registration_str: Type string from kernel registration
        :return: True is required. False if not.
        """

    @abstractmethod
    def get_cpp_entries(self):
        """
        Get the C++ code that specifies the operator types to enable.
        :return: List of strings. One line of C++ code per entry.
        """


class OperatorTypeUsageManager:
    """
    Class to manage the operator type usage processors.
    TODO: Currently the type tracking is not specific to a version of the operator.
    It's unclear how/where version specific logic could/should be added, and it would add significant complexity
    to track types on a per-version basis. Not clear there's enough benefit from doing so either.
    """

    def __init__(self):
        self._all_operator_processors = _create_operator_type_usage_processors()  # all possible processors
        self._operator_processors = {}  # processors we have actually used so we can limit output to be meaningful

    def _get_op_processor(self, key):
        "Add the processor to _operator_processors as it is about to be used."
        processor = None
        if key in self._all_operator_processors:
            if key not in self._operator_processors:
                self._operator_processors[key] = self._all_operator_processors[key]

            processor = self._operator_processors[key]

        return processor

    def process_node(self, node: fbs.Node, value_name_to_typeinfo: dict):
        """
        Process a Node and record info on the types used.
        :param node: Node from ORT format model
        :param value_name_to_typeinfo: Map of value names to TypeInfo instances
        """
        optype = node.OpType().decode()
        domain = node.Domain().decode() or "ai.onnx"  # empty domain defaults to ai.onnx

        key = _create_op_key(domain, optype)
        op_processor = self._get_op_processor(key)
        if op_processor:
            op_processor.process_node(node, value_name_to_typeinfo)

    def get_config_entry(self, domain: str, optype: str):
        """
        Get the config entry specifying the types for this operator.
        :param domain: Operator domain.
        :param optype: Operator type.
        :return: JSON string with type info if available, else None
        """
        key = _create_op_key(domain, optype)
        config_str = None
        if key in self._operator_processors:
            config_str = self._operator_processors[key].to_config_entry()

        return config_str

    def restore_from_config_entry(self, domain: str, optype: str, config_entry: str):
        """
        Restore the per-operator type information from a configuration file entry.
        :param domain: Operator domain.
        :param optype: Operator type.
        :param config_entry: JSON string with type info as created by get_config_entry
        """
        key = _create_op_key(domain, optype)
        op_processor = self._get_op_processor(key)
        if op_processor:
            op_processor.from_config_entry(config_entry)

    def debug_dump(self):
        print("C++ code that will be emitted:")
        [print(cpp_line) for cpp_line in self.get_cpp_entries()]

        print("Config file type information that will be returned by get_config_entry:")
        for key in sorted(self._operator_processors.keys()):
            entry = self._operator_processors[key].to_config_entry()
            if entry:
                print(f"{key} -> {entry}")

                # roundtrip test to validate that we can initialize the processor from the entry and get the
                # same values back
                self._operator_processors[key].from_config_entry(entry)
                assert entry == self._operator_processors[key].to_config_entry()

    class _OpTypeImplFilter(OpTypeImplFilterInterface):
        def __init__(self, manager):
            self._manager = manager

        def is_typed_registration_needed(self, domain: str, optype: str, type_registration_str: str):
            needed = True  # we keep the registration unless the per-operator processor says not to
            key = _create_op_key(domain, optype)
            if key in self._manager._operator_processors:
                needed = self._manager._operator_processors[key].is_typed_registration_needed(
                    type_in_registration=type_registration_str, globally_allowed_types=None
                )

            return needed

        def get_cpp_entries(self):
            entries = []
            for key in sorted(self._manager._operator_processors.keys()):
                entries.extend(self._manager._operator_processors[key].get_cpp_entry())

            return entries

    def make_op_type_impl_filter(self):
        """
        Creates an OpTypeImplFilterInterface instance from this manager.
        Filtering uses the manager's operator type usage processor state.
        """
        return OperatorTypeUsageManager._OpTypeImplFilter(self)


class GloballyAllowedTypesOpTypeImplFilter(OpTypeImplFilterInterface):
    """
    Operator implementation filter which uses globally allowed types.
    """

    _valid_allowed_types = set(FbsTypeInfo.tensordatatype_to_string.values())  # noqa: RUF012

    def __init__(self, globally_allowed_types: set[str]):
        self._operator_processors = _create_operator_type_usage_processors()

        if not globally_allowed_types.issubset(self._valid_allowed_types):
            raise ValueError(
                f"Globally allowed types must all be valid. Invalid types: {sorted(globally_allowed_types - self._valid_allowed_types)}"
            )

        self._globally_allowed_types = globally_allowed_types

    def is_typed_registration_needed(self, domain: str, optype: str, type_registration_str: str):
        key = _create_op_key(domain, optype)
        if key in self._operator_processors:
            needed = self._operator_processors[key].is_typed_registration_needed(
                type_in_registration=type_registration_str, globally_allowed_types=self._globally_allowed_types
            )
        else:
            needed = _reg_type_to_cpp_type(type_registration_str) in self._globally_allowed_types

        return needed

    def get_cpp_entries(self):
        return [
            "ORT_SPECIFY_OP_KERNEL_GLOBAL_ALLOWED_TYPES({});".format(", ".join(sorted(self._globally_allowed_types)))
        ]

    def global_type_list(self):
        return self._globally_allowed_types

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\hilbert.py ===
"""Hilbert spaces for quantum mechanics.

Authors:
* Brian Granger
* Matt Curry
"""

from functools import reduce

from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.sets.sets import Interval
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.qexpr import QuantumError


__all__ = [
    'HilbertSpaceError',
    'HilbertSpace',
    'TensorProductHilbertSpace',
    'TensorPowerHilbertSpace',
    'DirectSumHilbertSpace',
    'ComplexSpace',
    'L2',
    'FockSpace'
]

#-----------------------------------------------------------------------------
# Main objects
#-----------------------------------------------------------------------------


class HilbertSpaceError(QuantumError):
    pass

#-----------------------------------------------------------------------------
# Main objects
#-----------------------------------------------------------------------------


class HilbertSpace(Basic):
    """An abstract Hilbert space for quantum mechanics.

    In short, a Hilbert space is an abstract vector space that is complete
    with inner products defined [1]_.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import HilbertSpace
    >>> hs = HilbertSpace()
    >>> hs
    H

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space
    """

    def __new__(cls):
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        """Return the Hilbert dimension of the space."""
        raise NotImplementedError('This Hilbert space has no dimension.')

    def __add__(self, other):
        return DirectSumHilbertSpace(self, other)

    def __radd__(self, other):
        return DirectSumHilbertSpace(other, self)

    def __mul__(self, other):
        return TensorProductHilbertSpace(self, other)

    def __rmul__(self, other):
        return TensorProductHilbertSpace(other, self)

    def __pow__(self, other, mod=None):
        if mod is not None:
            raise ValueError('The third argument to __pow__ is not supported \
            for Hilbert spaces.')
        return TensorPowerHilbertSpace(self, other)

    def __contains__(self, other):
        """Is the operator or state in this Hilbert space.

        This is checked by comparing the classes of the Hilbert spaces, not
        the instances. This is to allow Hilbert Spaces with symbolic
        dimensions.
        """
        if other.hilbert_space.__class__ == self.__class__:
            return True
        else:
            return False

    def _sympystr(self, printer, *args):
        return 'H'

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER H}'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        return r'\mathcal{H}'


class ComplexSpace(HilbertSpace):
    """Finite dimensional Hilbert space of complex vectors.

    The elements of this Hilbert space are n-dimensional complex valued
    vectors with the usual inner product that takes the complex conjugate
    of the vector on the right.

    A classic example of this type of Hilbert space is spin-1/2, which is
    ``ComplexSpace(2)``. Generalizing to spin-s, the space is
    ``ComplexSpace(2*s+1)``.  Quantum computing with N qubits is done with the
    direct product space ``ComplexSpace(2)**N``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.hilbert import ComplexSpace
    >>> c1 = ComplexSpace(2)
    >>> c1
    C(2)
    >>> c1.dimension
    2

    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> c2
    C(n)
    >>> c2.dimension
    n

    """

    def __new__(cls, dimension):
        dimension = sympify(dimension)
        r = cls.eval(dimension)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, dimension)
        return obj

    @classmethod
    def eval(cls, dimension):
        if len(dimension.atoms()) == 1:
            if not (dimension.is_Integer and dimension > 0 or dimension is S.Infinity
            or dimension.is_Symbol):
                raise TypeError('The dimension of a ComplexSpace can only'
                                'be a positive integer, oo, or a Symbol: %r'
                                % dimension)
        else:
            for dim in dimension.atoms():
                if not (dim.is_Integer or dim is S.Infinity or dim.is_Symbol):
                    raise TypeError('The dimension of a ComplexSpace can only'
                                    ' contain integers, oo, or a Symbol: %r'
                                    % dim)

    @property
    def dimension(self):
        return self.args[0]

    def _sympyrepr(self, printer, *args):
        return "%s(%s)" % (self.__class__.__name__,
                           printer._print(self.dimension, *args))

    def _sympystr(self, printer, *args):
        return "C(%s)" % printer._print(self.dimension, *args)

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER C}'
        pform_exp = printer._print(self.dimension, *args)
        pform_base = prettyForm(ustr)
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        return r'\mathcal{C}^{%s}' % printer._print(self.dimension, *args)


class L2(HilbertSpace):
    """The Hilbert space of square integrable functions on an interval.

    An L2 object takes in a single SymPy Interval argument which represents
    the interval its functions (vectors) are defined on.

    Examples
    ========

    >>> from sympy import Interval, oo
    >>> from sympy.physics.quantum.hilbert import L2
    >>> hs = L2(Interval(0,oo))
    >>> hs
    L2(Interval(0, oo))
    >>> hs.dimension
    oo
    >>> hs.interval
    Interval(0, oo)

    """

    def __new__(cls, interval):
        if not isinstance(interval, Interval):
            raise TypeError('L2 interval must be an Interval instance: %r'
            % interval)
        obj = Basic.__new__(cls, interval)
        return obj

    @property
    def dimension(self):
        return S.Infinity

    @property
    def interval(self):
        return self.args[0]

    def _sympyrepr(self, printer, *args):
        return "L2(%s)" % printer._print(self.interval, *args)

    def _sympystr(self, printer, *args):
        return "L2(%s)" % printer._print(self.interval, *args)

    def _pretty(self, printer, *args):
        pform_exp = prettyForm('2')
        pform_base = prettyForm('L')
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        interval = printer._print(self.interval, *args)
        return r'{\mathcal{L}^2}\left( %s \right)' % interval


class FockSpace(HilbertSpace):
    """The Hilbert space for second quantization.

    Technically, this Hilbert space is a infinite direct sum of direct
    products of single particle Hilbert spaces [1]_. This is a mess, so we have
    a class to represent it directly.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import FockSpace
    >>> hs = FockSpace()
    >>> hs
    F
    >>> hs.dimension
    oo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fock_space
    """

    def __new__(cls):
        obj = Basic.__new__(cls)
        return obj

    @property
    def dimension(self):
        return S.Infinity

    def _sympyrepr(self, printer, *args):
        return "FockSpace()"

    def _sympystr(self, printer, *args):
        return "F"

    def _pretty(self, printer, *args):
        ustr = '\N{LATIN CAPITAL LETTER F}'
        return prettyForm(ustr)

    def _latex(self, printer, *args):
        return r'\mathcal{F}'


class TensorProductHilbertSpace(HilbertSpace):
    """A tensor product of Hilbert spaces [1]_.

    The tensor product between Hilbert spaces is represented by the
    operator ``*`` Products of the same Hilbert space will be combined into
    tensor powers.

    A ``TensorProductHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. In addition, multiplication of
    ``HilbertSpace`` objects will automatically return this tensor product
    object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c*f
    >>> hs
    C(2)*F
    >>> hs.dimension
    oo
    >>> hs.spaces
    (C(2), F)

    >>> c1 = ComplexSpace(2)
    >>> n = symbols('n')
    >>> c2 = ComplexSpace(n)
    >>> hs = c1*c2
    >>> hs
    C(2)*C(n)
    >>> hs.dimension
    2*n

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, *args)
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct product."""
        new_args = []
        recall = False
        #flatten arguments
        for arg in args:
            if isinstance(arg, TensorProductHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, (HilbertSpace, TensorPowerHilbertSpace)):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be multiplied by \
                other Hilbert spaces: %r' % arg)
        #combine like arguments into direct powers
        comb_args = []
        prev_arg = None
        for new_arg in new_args:
            if prev_arg is not None:
                if isinstance(new_arg, TensorPowerHilbertSpace) and \
                    isinstance(prev_arg, TensorPowerHilbertSpace) and \
                        new_arg.base == prev_arg.base:
                    prev_arg = new_arg.base**(new_arg.exp + prev_arg.exp)
                elif isinstance(new_arg, TensorPowerHilbertSpace) and \
                        new_arg.base == prev_arg:
                    prev_arg = prev_arg**(new_arg.exp + 1)
                elif isinstance(prev_arg, TensorPowerHilbertSpace) and \
                        new_arg == prev_arg.base:
                    prev_arg = new_arg**(prev_arg.exp + 1)
                elif new_arg == prev_arg:
                    prev_arg = new_arg**2
                else:
                    comb_args.append(prev_arg)
                    prev_arg = new_arg
            elif prev_arg is None:
                prev_arg = new_arg
        comb_args.append(prev_arg)
        if recall:
            return TensorProductHilbertSpace(*comb_args)
        elif len(comb_args) == 1:
            return TensorPowerHilbertSpace(comb_args[0].base, comb_args[0].exp)
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x*y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this tensor product."""
        return self.args

    def _spaces_printer(self, printer, *args):
        spaces_strs = []
        for arg in self.args:
            s = printer._print(arg, *args)
            if isinstance(arg, DirectSumHilbertSpace):
                s = '(%s)' % s
            spaces_strs.append(s)
        return spaces_strs

    def _sympyrepr(self, printer, *args):
        spaces_reprs = self._spaces_printer(printer, *args)
        return "TensorProductHilbertSpace(%s)" % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        spaces_strs = self._spaces_printer(printer, *args)
        return '*'.join(spaces_strs)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                          TensorProductHilbertSpace)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right(' ' + '\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right(' x '))
        return pform

    def _latex(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            arg_s = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                 TensorProductHilbertSpace)):
                arg_s = r'\left(%s\right)' % arg_s
            s = s + arg_s
            if i != length - 1:
                s = s + r'\otimes '
        return s


class DirectSumHilbertSpace(HilbertSpace):
    """A direct sum of Hilbert spaces [1]_.

    This class uses the ``+`` operator to represent direct sums between
    different Hilbert spaces.

    A ``DirectSumHilbertSpace`` object takes in an arbitrary number of
    ``HilbertSpace`` objects as its arguments. Also, addition of
    ``HilbertSpace`` objects will automatically return a direct sum object.

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace

    >>> c = ComplexSpace(2)
    >>> f = FockSpace()
    >>> hs = c+f
    >>> hs
    C(2)+F
    >>> hs.dimension
    oo
    >>> list(hs.spaces)
    [C(2), F]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Direct_sums
    """
    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        obj = Basic.__new__(cls, *args)
        return obj

    @classmethod
    def eval(cls, args):
        """Evaluates the direct product."""
        new_args = []
        recall = False
        #flatten arguments
        for arg in args:
            if isinstance(arg, DirectSumHilbertSpace):
                new_args.extend(arg.args)
                recall = True
            elif isinstance(arg, HilbertSpace):
                new_args.append(arg)
            else:
                raise TypeError('Hilbert spaces can only be summed with other \
                Hilbert spaces: %r' % arg)
        if recall:
            return DirectSumHilbertSpace(*new_args)
        else:
            return None

    @property
    def dimension(self):
        arg_list = [arg.dimension for arg in self.args]
        if S.Infinity in arg_list:
            return S.Infinity
        else:
            return reduce(lambda x, y: x + y, arg_list)

    @property
    def spaces(self):
        """A tuple of the Hilbert spaces in this direct sum."""
        return self.args

    def _sympyrepr(self, printer, *args):
        spaces_reprs = [printer._print(arg, *args) for arg in self.args]
        return "DirectSumHilbertSpace(%s)" % ','.join(spaces_reprs)

    def _sympystr(self, printer, *args):
        spaces_strs = [printer._print(arg, *args) for arg in self.args]
        return '+'.join(spaces_strs)

    def _pretty(self, printer, *args):
        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                          TensorProductHilbertSpace)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right(' \N{CIRCLED PLUS} '))
                else:
                    pform = prettyForm(*pform.right(' + '))
        return pform

    def _latex(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            arg_s = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (DirectSumHilbertSpace,
                 TensorProductHilbertSpace)):
                arg_s = r'\left(%s\right)' % arg_s
            s = s + arg_s
            if i != length - 1:
                s = s + r'\oplus '
        return s


class TensorPowerHilbertSpace(HilbertSpace):
    """An exponentiated Hilbert space [1]_.

    Tensor powers (repeated tensor products) are represented by the
    operator ``**`` Identical Hilbert spaces that are multiplied together
    will be automatically combined into a single tensor power object.

    Any Hilbert space, product, or sum may be raised to a tensor power. The
    ``TensorPowerHilbertSpace`` takes two arguments: the Hilbert space; and the
    tensor power (number).

    Examples
    ========

    >>> from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace
    >>> from sympy import symbols

    >>> n = symbols('n')
    >>> c = ComplexSpace(2)
    >>> hs = c**n
    >>> hs
    C(2)**n
    >>> hs.dimension
    2**n

    >>> c = ComplexSpace(2)
    >>> c*c
    C(2)**2
    >>> f = FockSpace()
    >>> c*f*f
    C(2)*F**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hilbert_space#Tensor_products
    """

    def __new__(cls, *args):
        r = cls.eval(args)
        if isinstance(r, Basic):
            return r
        return Basic.__new__(cls, *r)

    @classmethod
    def eval(cls, args):
        new_args = args[0], sympify(args[1])
        exp = new_args[1]
        #simplify hs**1 -> hs
        if exp is S.One:
            return args[0]
        #simplify hs**0 -> 1
        if exp is S.Zero:
            return S.One
        #check (and allow) for hs**(x+42+y...) case
        if len(exp.atoms()) == 1:
            if not (exp.is_Integer and exp >= 0 or exp.is_Symbol):
                raise ValueError('Hilbert spaces can only be raised to \
                positive integers or Symbols: %r' % exp)
        else:
            for power in exp.atoms():
                if not (power.is_Integer or power.is_Symbol):
                    raise ValueError('Tensor powers can only contain integers \
                    or Symbols: %r' % power)
        return new_args

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

    @property
    def dimension(self):
        if self.base.dimension is S.Infinity:
            return S.Infinity
        else:
            return self.base.dimension**self.exp

    def _sympyrepr(self, printer, *args):
        return "TensorPowerHilbertSpace(%s,%s)" % (printer._print(self.base,
        *args), printer._print(self.exp, *args))

    def _sympystr(self, printer, *args):
        return "%s**%s" % (printer._print(self.base, *args),
        printer._print(self.exp, *args))

    def _pretty(self, printer, *args):
        pform_exp = printer._print(self.exp, *args)
        if printer._use_unicode:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('\N{N-ARY CIRCLED TIMES OPERATOR}')))
        else:
            pform_exp = prettyForm(*pform_exp.left(prettyForm('x')))
        pform_base = printer._print(self.base, *args)
        return pform_base**pform_exp

    def _latex(self, printer, *args):
        base = printer._print(self.base, *args)
        exp = printer._print(self.exp, *args)
        return r'{%s}^{\otimes %s}' % (base, exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\operator.py ===
"""Quantum mechanical operators.

TODO:

* Fix early 0 in apply_operators.
* Debug and test apply_operators.
* Get cse working with classes in this file.
* Doctests and documentation of special methods for InnerProduct, Commutator,
  AntiCommutator, represent, apply_operators.
"""
from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, expand)
from sympy.core.mul import Mul
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.kind import OperatorKind
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
from sympy.matrices import eye
from sympy.utilities.exceptions import sympy_deprecation_warning



__all__ = [
    'Operator',
    'HermitianOperator',
    'UnitaryOperator',
    'IdentityOperator',
    'OuterProduct',
    'DifferentialOperator'
]

#-----------------------------------------------------------------------------
# Operators and outer products
#-----------------------------------------------------------------------------


class Operator(QExpr):
    """Base class for non-commuting quantum operators.

    An operator maps between quantum states [1]_. In quantum mechanics,
    observables (including, but not limited to, measured physical values) are
    represented as Hermitian operators [2]_.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    Create an operator and examine its attributes::

        >>> from sympy.physics.quantum import Operator
        >>> from sympy import I
        >>> A = Operator('A')
        >>> A
        A
        >>> A.hilbert_space
        H
        >>> A.label
        (A,)
        >>> A.is_commutative
        False

    Create another operator and do some arithmetic operations::

        >>> B = Operator('B')
        >>> C = 2*A*A + I*B
        >>> C
        2*A**2 + I*B

    Operators do not commute::

        >>> A.is_commutative
        False
        >>> B.is_commutative
        False
        >>> A*B == B*A
        False

    Polymonials of operators respect the commutation properties::

        >>> e = (A+B)**3
        >>> e.expand()
        A*B*A + A*B**2 + A**2*B + A**3 + B*A*B + B*A**2 + B**2*A + B**3

    Operator inverses are handle symbolically::

        >>> A.inv()
        A**(-1)
        >>> A*A.inv()
        1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Operator_%28physics%29
    .. [2] https://en.wikipedia.org/wiki/Observable
    """
    is_hermitian: Optional[bool] = None
    is_unitary: Optional[bool] = None
    @classmethod
    def default_args(self):
        return ("O",)

    kind = OperatorKind

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    _label_separator = ','

    def _print_operator_name(self, printer, *args):
        return self.__class__.__name__

    _print_operator_name_latex = _print_operator_name

    def _print_operator_name_pretty(self, printer, *args):
        return prettyForm(self.__class__.__name__)

    def _print_contents(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label(printer, *args)
        else:
            return '%s(%s)' % (
                self._print_operator_name(printer, *args),
                self._print_label(printer, *args)
            )

    def _print_contents_pretty(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label_pretty(printer, *args)
        else:
            pform = self._print_operator_name_pretty(printer, *args)
            label_pform = self._print_label_pretty(printer, *args)
            label_pform = prettyForm(
                *label_pform.parens(left='(', right=')')
            )
            pform = prettyForm(*pform.right(label_pform))
            return pform

    def _print_contents_latex(self, printer, *args):
        if len(self.label) == 1:
            return self._print_label_latex(printer, *args)
        else:
            return r'%s\left(%s\right)' % (
                self._print_operator_name_latex(printer, *args),
                self._print_label_latex(printer, *args)
            )

    #-------------------------------------------------------------------------
    # _eval_* methods
    #-------------------------------------------------------------------------

    def _eval_commutator(self, other, **options):
        """Evaluate [self, other] if known, return None if not known."""
        return dispatch_method(self, '_eval_commutator', other, **options)

    def _eval_anticommutator(self, other, **options):
        """Evaluate [self, other] if known."""
        return dispatch_method(self, '_eval_anticommutator', other, **options)

    #-------------------------------------------------------------------------
    # Operator application
    #-------------------------------------------------------------------------

    def _apply_operator(self, ket, **options):
        return dispatch_method(self, '_apply_operator', ket, **options)

    def _apply_from_right_to(self, bra, **options):
        return None

    def matrix_element(self, *args):
        raise NotImplementedError('matrix_elements is not defined')

    def inverse(self):
        return self._eval_inverse()

    inv = inverse

    def _eval_inverse(self):
        return self**(-1)


class HermitianOperator(Operator):
    """A Hermitian operator that satisfies H == Dagger(H).

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, HermitianOperator
    >>> H = HermitianOperator('H')
    >>> Dagger(H)
    H
    """

    is_hermitian = True

    def _eval_inverse(self):
        if isinstance(self, UnitaryOperator):
            return self
        else:
            return Operator._eval_inverse(self)

    def _eval_power(self, exp):
        if isinstance(self, UnitaryOperator):
            # so all eigenvalues of self are 1 or -1
            if exp.is_even:
                from sympy.core.singleton import S
                return S.One # is identity, see Issue 24153.
            elif exp.is_odd:
                return self
        # No simplification in all other cases
        return Operator._eval_power(self, exp)


class UnitaryOperator(Operator):
    """A unitary operator that satisfies U*Dagger(U) == 1.

    Parameters
    ==========

    args : tuple
        The list of numbers or parameters that uniquely specify the
        operator. For time-dependent operators, this will include the time.

    Examples
    ========

    >>> from sympy.physics.quantum import Dagger, UnitaryOperator
    >>> U = UnitaryOperator('U')
    >>> U*Dagger(U)
    1
    """
    is_unitary = True
    def _eval_adjoint(self):
        return self._eval_inverse()


class IdentityOperator(Operator):
    """An identity operator I that satisfies op * I == I * op == op for any
    operator op.

    .. deprecated:: 1.14.
        Use the scalar S.One instead as the multiplicative identity for
        operators and states.

    Parameters
    ==========

    N : Integer
        Optional parameter that specifies the dimension of the Hilbert space
        of operator. This is used when generating a matrix representation.

    Examples
    ========

    >>> from sympy.physics.quantum import IdentityOperator
    >>> IdentityOperator() # doctest: +SKIP
    I
    """
    is_hermitian = True
    is_unitary = True
    @property
    def dimension(self):
        return self.N

    @classmethod
    def default_args(self):
        return (oo,)

    def __init__(self, *args, **hints):
        sympy_deprecation_warning(
            """
            IdentityOperator has been deprecated. In the future, please use
            S.One as the identity for quantum operators and states.
            """,
            deprecated_since_version="1.14",
            active_deprecations_target='deprecated-operator-identity',
        )
        if not len(args) in (0, 1):
            raise ValueError('0 or 1 parameters expected, got %s' % args)

        self.N = args[0] if (len(args) == 1 and args[0]) else oo

    def _eval_commutator(self, other, **hints):
        return S.Zero

    def _eval_anticommutator(self, other, **hints):
        return 2 * other

    def _eval_inverse(self):
        return self

    def _eval_adjoint(self):
        return self

    def _apply_operator(self, ket, **options):
        return ket

    def _apply_from_right_to(self, bra, **options):
        return bra

    def _eval_power(self, exp):
        return self

    def _print_contents(self, printer, *args):
        return 'I'

    def _print_contents_pretty(self, printer, *args):
        return prettyForm('I')

    def _print_contents_latex(self, printer, *args):
        return r'{\mathcal{I}}'

    def _represent_default_basis(self, **options):
        if not self.N or self.N == oo:
            raise NotImplementedError('Cannot represent infinite dimensional' +
                                      ' identity operator as a matrix')

        format = options.get('format', 'sympy')
        if format != 'sympy':
            raise NotImplementedError('Representation in format ' +
                                      '%s not implemented.' % format)

        return eye(self.N)


class OuterProduct(Operator):
    """An unevaluated outer product between a ket and bra.

    This constructs an outer product between any subclass of ``KetBase`` and
    ``BraBase`` as ``|a><b|``. An ``OuterProduct`` inherits from Operator as they act as
    operators in quantum expressions.  For reference see [1]_.

    Parameters
    ==========

    ket : KetBase
        The ket on the left side of the outer product.
    bar : BraBase
        The bra on the right side of the outer product.

    Examples
    ========

    Create a simple outer product by hand and take its dagger::

        >>> from sympy.physics.quantum import Ket, Bra, OuterProduct, Dagger

        >>> k = Ket('k')
        >>> b = Bra('b')
        >>> op = OuterProduct(k, b)
        >>> op
        |k><b|
        >>> op.hilbert_space
        H
        >>> op.ket
        |k>
        >>> op.bra
        <b|
        >>> Dagger(op)
        |b><k|

    In quantum expressions, outer products will be automatically
    identified and created::

        >>> k*b
        |k><b|

    However, the creation of inner products always has higher priority than that of
    outer products:

        >>> b*k*b
        <b|k>*<b|

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Outer_product
    """
    is_commutative = False

    def __new__(cls, *args, **old_assumptions):
        from sympy.physics.quantum.state import KetBase, BraBase

        if len(args) != 2:
            raise ValueError('2 parameters expected, got %d' % len(args))

        ket_expr = expand(args[0])
        bra_expr = expand(args[1])

        if (isinstance(ket_expr, (KetBase, Mul)) and
                isinstance(bra_expr, (BraBase, Mul))):
            ket_c, kets = ket_expr.args_cnc()
            bra_c, bras = bra_expr.args_cnc()

            if len(kets) != 1 or not isinstance(kets[0], KetBase):
                raise TypeError('KetBase subclass expected'
                                ', got: %r' % Mul(*kets))

            if len(bras) != 1 or not isinstance(bras[0], BraBase):
                raise TypeError('BraBase subclass expected'
                                ', got: %r' % Mul(*bras))

            if not kets[0].dual_class() == bras[0].__class__:
                raise TypeError(
                    'ket and bra are not dual classes: %r, %r' %
                    (kets[0].__class__, bras[0].__class__)
                    )

            # TODO: make sure the hilbert spaces of the bra and ket are
            # compatible
            obj = Expr.__new__(cls, *(kets[0], bras[0]), **old_assumptions)
            obj.hilbert_space = kets[0].hilbert_space
            return Mul(*(ket_c + bra_c)) * obj

        op_terms = []
        if isinstance(ket_expr, Add) and isinstance(bra_expr, Add):
            for ket_term in ket_expr.args:
                for bra_term in bra_expr.args:
                    op_terms.append(OuterProduct(ket_term, bra_term,
                                                 **old_assumptions))
        elif isinstance(ket_expr, Add):
            for ket_term in ket_expr.args:
                op_terms.append(OuterProduct(ket_term, bra_expr,
                                             **old_assumptions))
        elif isinstance(bra_expr, Add):
            for bra_term in bra_expr.args:
                op_terms.append(OuterProduct(ket_expr, bra_term,
                                             **old_assumptions))
        else:
            raise TypeError(
                'Expected ket and bra expression, got: %r, %r' %
                (ket_expr, bra_expr)
                )

        return Add(*op_terms)

    @property
    def ket(self):
        """Return the ket on the left side of the outer product."""
        return self.args[0]

    @property
    def bra(self):
        """Return the bra on the right side of the outer product."""
        return self.args[1]

    def _eval_adjoint(self):
        return OuterProduct(Dagger(self.bra), Dagger(self.ket))

    def _sympystr(self, printer, *args):
        return printer._print(self.ket) + printer._print(self.bra)

    def _sympyrepr(self, printer, *args):
        return '%s(%s,%s)' % (self.__class__.__name__,
            printer._print(self.ket, *args), printer._print(self.bra, *args))

    def _pretty(self, printer, *args):
        pform = self.ket._pretty(printer, *args)
        return prettyForm(*pform.right(self.bra._pretty(printer, *args)))

    def _latex(self, printer, *args):
        k = printer._print(self.ket, *args)
        b = printer._print(self.bra, *args)
        return k + b

    def _represent(self, **options):
        k = self.ket._represent(**options)
        b = self.bra._represent(**options)
        return k*b

    def _eval_trace(self, **kwargs):
        # TODO if operands are tensorproducts this may be will be handled
        # differently.

        return self.ket._eval_trace(self.bra, **kwargs)


class DifferentialOperator(Operator):
    """An operator for representing the differential operator, i.e. d/dx

    It is initialized by passing two arguments. The first is an arbitrary
    expression that involves a function, such as ``Derivative(f(x), x)``. The
    second is the function (e.g. ``f(x)``) which we are to replace with the
    ``Wavefunction`` that this ``DifferentialOperator`` is applied to.

    Parameters
    ==========

    expr : Expr
           The arbitrary expression which the appropriate Wavefunction is to be
           substituted into

    func : Expr
           A function (e.g. f(x)) which is to be replaced with the appropriate
           Wavefunction when this DifferentialOperator is applied

    Examples
    ========

    You can define a completely arbitrary expression and specify where the
    Wavefunction is to be substituted

    >>> from sympy import Derivative, Function, Symbol
    >>> from sympy.physics.quantum.operator import DifferentialOperator
    >>> from sympy.physics.quantum.state import Wavefunction
    >>> from sympy.physics.quantum.qapply import qapply
    >>> f = Function('f')
    >>> x = Symbol('x')
    >>> d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
    >>> w = Wavefunction(x**2, x)
    >>> d.function
    f(x)
    >>> d.variables
    (x,)
    >>> qapply(d*w)
    Wavefunction(2, x)

    """

    @property
    def variables(self):
        """
        Returns the variables with which the function in the specified
        arbitrary expression is evaluated

        Examples
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Symbol, Function, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(1/x*Derivative(f(x), x), f(x))
        >>> d.variables
        (x,)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.variables
        (x, y)
        """

        return self.args[-1].args

    @property
    def function(self):
        """
        Returns the function which is to be replaced with the Wavefunction

        Examples
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Function, Symbol, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(Derivative(f(x), x), f(x))
        >>> d.function
        f(x)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.function
        f(x, y)
        """

        return self.args[-1]

    @property
    def expr(self):
        """
        Returns the arbitrary expression which is to have the Wavefunction
        substituted into it

        Examples
        ========

        >>> from sympy.physics.quantum.operator import DifferentialOperator
        >>> from sympy import Function, Symbol, Derivative
        >>> x = Symbol('x')
        >>> f = Function('f')
        >>> d = DifferentialOperator(Derivative(f(x), x), f(x))
        >>> d.expr
        Derivative(f(x), x)
        >>> y = Symbol('y')
        >>> d = DifferentialOperator(Derivative(f(x, y), x) +
        ...                          Derivative(f(x, y), y), f(x, y))
        >>> d.expr
        Derivative(f(x, y), x) + Derivative(f(x, y), y)
        """

        return self.args[0]

    @property
    def free_symbols(self):
        """
        Return the free symbols of the expression.
        """

        return self.expr.free_symbols

    def _apply_operator_Wavefunction(self, func, **options):
        from sympy.physics.quantum.state import Wavefunction
        var = self.variables
        wf_vars = func.args[1:]

        f = self.function
        new_expr = self.expr.subs(f, func(*var))
        new_expr = new_expr.doit()

        return Wavefunction(new_expr, *wf_vars)

    def _eval_derivative(self, symbol):
        new_expr = Derivative(self.expr, symbol)
        return DifferentialOperator(new_expr, self.args[-1])

    #-------------------------------------------------------------------------
    # Printing
    #-------------------------------------------------------------------------

    def _print(self, printer, *args):
        return '%s(%s)' % (
            self._print_operator_name(printer, *args),
            self._print_label(printer, *args)
        )

    def _print_pretty(self, printer, *args):
        pform = self._print_operator_name_pretty(printer, *args)
        label_pform = self._print_label_pretty(printer, *args)
        label_pform = prettyForm(
            *label_pform.parens(left='(', right=')')
        )
        pform = prettyForm(*pform.right(label_pform))
        return pform

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\local.py ===
from __future__ import annotations

import copy
import math
import operator
import typing as t
from contextvars import ContextVar
from functools import partial
from functools import update_wrapper
from operator import attrgetter

from .wsgi import ClosingIterator

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment

T = t.TypeVar("T")
F = t.TypeVar("F", bound=t.Callable[..., t.Any])


def release_local(local: Local | LocalStack[t.Any]) -> None:
    """Release the data for the current context in a :class:`Local` or
    :class:`LocalStack` without using a :class:`LocalManager`.

    This should not be needed for modern use cases, and may be removed
    in the future.

    .. versionadded:: 0.6.1
    """
    local.__release_local__()


class Local:
    """Create a namespace of context-local data. This wraps a
    :class:`ContextVar` containing a :class:`dict` value.

    This may incur a performance penalty compared to using individual
    context vars, as it has to copy data to avoid mutating the dict
    between nested contexts.

    :param context_var: The :class:`~contextvars.ContextVar` to use as
        storage for this local. If not given, one will be created.
        Context vars not created at the global scope may interfere with
        garbage collection.

    .. versionchanged:: 2.0
        Uses ``ContextVar`` instead of a custom storage implementation.
    """

    __slots__ = ("__storage",)

    def __init__(self, context_var: ContextVar[dict[str, t.Any]] | None = None) -> None:
        if context_var is None:
            # A ContextVar not created at global scope interferes with
            # Python's garbage collection. However, a local only makes
            # sense defined at the global scope as well, in which case
            # the GC issue doesn't seem relevant.
            context_var = ContextVar(f"werkzeug.Local<{id(self)}>.storage")

        object.__setattr__(self, "_Local__storage", context_var)

    def __iter__(self) -> t.Iterator[tuple[str, t.Any]]:
        return iter(self.__storage.get({}).items())

    def __call__(
        self, name: str, *, unbound_message: str | None = None
    ) -> LocalProxy[t.Any]:
        """Create a :class:`LocalProxy` that access an attribute on this
        local namespace.

        :param name: Proxy this attribute.
        :param unbound_message: The error message that the proxy will
            show if the attribute isn't set.
        """
        return LocalProxy(self, name, unbound_message=unbound_message)

    def __release_local__(self) -> None:
        self.__storage.set({})

    def __getattr__(self, name: str) -> t.Any:
        values = self.__storage.get({})

        if name in values:
            return values[name]

        raise AttributeError(name)

    def __setattr__(self, name: str, value: t.Any) -> None:
        values = self.__storage.get({}).copy()
        values[name] = value
        self.__storage.set(values)

    def __delattr__(self, name: str) -> None:
        values = self.__storage.get({})

        if name in values:
            values = values.copy()
            del values[name]
            self.__storage.set(values)
        else:
            raise AttributeError(name)


class LocalStack(t.Generic[T]):
    """Create a stack of context-local data. This wraps a
    :class:`ContextVar` containing a :class:`list` value.

    This may incur a performance penalty compared to using individual
    context vars, as it has to copy data to avoid mutating the list
    between nested contexts.

    :param context_var: The :class:`~contextvars.ContextVar` to use as
        storage for this local. If not given, one will be created.
        Context vars not created at the global scope may interfere with
        garbage collection.

    .. versionchanged:: 2.0
        Uses ``ContextVar`` instead of a custom storage implementation.

    .. versionadded:: 0.6.1
    """

    __slots__ = ("_storage",)

    def __init__(self, context_var: ContextVar[list[T]] | None = None) -> None:
        if context_var is None:
            # A ContextVar not created at global scope interferes with
            # Python's garbage collection. However, a local only makes
            # sense defined at the global scope as well, in which case
            # the GC issue doesn't seem relevant.
            context_var = ContextVar(f"werkzeug.LocalStack<{id(self)}>.storage")

        self._storage = context_var

    def __release_local__(self) -> None:
        self._storage.set([])

    def push(self, obj: T) -> list[T]:
        """Add a new item to the top of the stack."""
        stack = self._storage.get([]).copy()
        stack.append(obj)
        self._storage.set(stack)
        return stack

    def pop(self) -> T | None:
        """Remove the top item from the stack and return it. If the
        stack is empty, return ``None``.
        """
        stack = self._storage.get([])

        if len(stack) == 0:
            return None

        rv = stack[-1]
        self._storage.set(stack[:-1])
        return rv

    @property
    def top(self) -> T | None:
        """The topmost item on the stack.  If the stack is empty,
        `None` is returned.
        """
        stack = self._storage.get([])

        if len(stack) == 0:
            return None

        return stack[-1]

    def __call__(
        self, name: str | None = None, *, unbound_message: str | None = None
    ) -> LocalProxy[t.Any]:
        """Create a :class:`LocalProxy` that accesses the top of this
        local stack.

        :param name: If given, the proxy access this attribute of the
            top item, rather than the item itself.
        :param unbound_message: The error message that the proxy will
            show if the stack is empty.
        """
        return LocalProxy(self, name, unbound_message=unbound_message)


class LocalManager:
    """Manage releasing the data for the current context in one or more
    :class:`Local` and :class:`LocalStack` objects.

    This should not be needed for modern use cases, and may be removed
    in the future.

    :param locals: A local or list of locals to manage.

    .. versionchanged:: 2.1
        The ``ident_func`` was removed.

    .. versionchanged:: 0.7
        The ``ident_func`` parameter was added.

    .. versionchanged:: 0.6.1
        The :func:`release_local` function can be used instead of a
        manager.
    """

    __slots__ = ("locals",)

    def __init__(
        self,
        locals: None
        | (Local | LocalStack[t.Any] | t.Iterable[Local | LocalStack[t.Any]]) = None,
    ) -> None:
        if locals is None:
            self.locals = []
        elif isinstance(locals, Local):
            self.locals = [locals]
        else:
            self.locals = list(locals)  # type: ignore[arg-type]

    def cleanup(self) -> None:
        """Release the data in the locals for this context. Call this at
        the end of each request or use :meth:`make_middleware`.
        """
        for local in self.locals:
            release_local(local)

    def make_middleware(self, app: WSGIApplication) -> WSGIApplication:
        """Wrap a WSGI application so that local data is released
        automatically after the response has been sent for a request.
        """

        def application(
            environ: WSGIEnvironment, start_response: StartResponse
        ) -> t.Iterable[bytes]:
            return ClosingIterator(app(environ, start_response), self.cleanup)

        return application

    def middleware(self, func: WSGIApplication) -> WSGIApplication:
        """Like :meth:`make_middleware` but used as a decorator on the
        WSGI application function.

        .. code-block:: python

            @manager.middleware
            def application(environ, start_response):
                ...
        """
        return update_wrapper(self.make_middleware(func), func)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} storages: {len(self.locals)}>"


class _ProxyLookup:
    """Descriptor that handles proxied attribute lookup for
    :class:`LocalProxy`.

    :param f: The built-in function this attribute is accessed through.
        Instead of looking up the special method, the function call
        is redone on the object.
    :param fallback: Return this function if the proxy is unbound
        instead of raising a :exc:`RuntimeError`.
    :param is_attr: This proxied name is an attribute, not a function.
        Call the fallback immediately to get the value.
    :param class_value: Value to return when accessed from the
        ``LocalProxy`` class directly. Used for ``__doc__`` so building
        docs still works.
    """

    __slots__ = ("bind_f", "fallback", "is_attr", "class_value", "name")

    def __init__(
        self,
        f: t.Callable[..., t.Any] | None = None,
        fallback: t.Callable[[LocalProxy[t.Any]], t.Any] | None = None,
        class_value: t.Any | None = None,
        is_attr: bool = False,
    ) -> None:
        bind_f: t.Callable[[LocalProxy[t.Any], t.Any], t.Callable[..., t.Any]] | None

        if hasattr(f, "__get__"):
            # A Python function, can be turned into a bound method.

            def bind_f(
                instance: LocalProxy[t.Any], obj: t.Any
            ) -> t.Callable[..., t.Any]:
                return f.__get__(obj, type(obj))  # type: ignore

        elif f is not None:
            # A C function, use partial to bind the first argument.

            def bind_f(
                instance: LocalProxy[t.Any], obj: t.Any
            ) -> t.Callable[..., t.Any]:
                return partial(f, obj)

        else:
            # Use getattr, which will produce a bound method.
            bind_f = None

        self.bind_f = bind_f
        self.fallback = fallback
        self.class_value = class_value
        self.is_attr = is_attr

    def __set_name__(self, owner: LocalProxy[t.Any], name: str) -> None:
        self.name = name

    def __get__(self, instance: LocalProxy[t.Any], owner: type | None = None) -> t.Any:
        if instance is None:
            if self.class_value is not None:
                return self.class_value

            return self

        try:
            obj = instance._get_current_object()
        except RuntimeError:
            if self.fallback is None:
                raise

            fallback = self.fallback.__get__(instance, owner)

            if self.is_attr:
                # __class__ and __doc__ are attributes, not methods.
                # Call the fallback to get the value.
                return fallback()

            return fallback

        if self.bind_f is not None:
            return self.bind_f(instance, obj)

        return getattr(obj, self.name)

    def __repr__(self) -> str:
        return f"proxy {self.name}"

    def __call__(
        self, instance: LocalProxy[t.Any], *args: t.Any, **kwargs: t.Any
    ) -> t.Any:
        """Support calling unbound methods from the class. For example,
        this happens with ``copy.copy``, which does
        ``type(x).__copy__(x)``. ``type(x)`` can't be proxied, so it
        returns the proxy type and descriptor.
        """
        return self.__get__(instance, type(instance))(*args, **kwargs)


class _ProxyIOp(_ProxyLookup):
    """Look up an augmented assignment method on a proxied object. The
    method is wrapped to return the proxy instead of the object.
    """

    __slots__ = ()

    def __init__(
        self,
        f: t.Callable[..., t.Any] | None = None,
        fallback: t.Callable[[LocalProxy[t.Any]], t.Any] | None = None,
    ) -> None:
        super().__init__(f, fallback)

        def bind_f(instance: LocalProxy[t.Any], obj: t.Any) -> t.Callable[..., t.Any]:
            def i_op(self: t.Any, other: t.Any) -> LocalProxy[t.Any]:
                f(self, other)  # type: ignore
                return instance

            return i_op.__get__(obj, type(obj))  # type: ignore

        self.bind_f = bind_f


def _l_to_r_op(op: F) -> F:
    """Swap the argument order to turn an l-op into an r-op."""

    def r_op(obj: t.Any, other: t.Any) -> t.Any:
        return op(other, obj)

    return t.cast(F, r_op)


def _identity(o: T) -> T:
    return o


class LocalProxy(t.Generic[T]):
    """A proxy to the object bound to a context-local object. All
    operations on the proxy are forwarded to the bound object. If no
    object is bound, a ``RuntimeError`` is raised.

    :param local: The context-local object that provides the proxied
        object.
    :param name: Proxy this attribute from the proxied object.
    :param unbound_message: The error message to show if the
        context-local object is unbound.

    Proxy a :class:`~contextvars.ContextVar` to make it easier to
    access. Pass a name to proxy that attribute.

    .. code-block:: python

        _request_var = ContextVar("request")
        request = LocalProxy(_request_var)
        session = LocalProxy(_request_var, "session")

    Proxy an attribute on a :class:`Local` namespace by calling the
    local with the attribute name:

    .. code-block:: python

        data = Local()
        user = data("user")

    Proxy the top item on a :class:`LocalStack` by calling the local.
    Pass a name to proxy that attribute.

    .. code-block::

        app_stack = LocalStack()
        current_app = app_stack()
        g = app_stack("g")

    Pass a function to proxy the return value from that function. This
    was previously used to access attributes of local objects before
    that was supported directly.

    .. code-block:: python

        session = LocalProxy(lambda: request.session)

    ``__repr__`` and ``__class__`` are proxied, so ``repr(x)`` and
    ``isinstance(x, cls)`` will look like the proxied object. Use
    ``issubclass(type(x), LocalProxy)`` to check if an object is a
    proxy.

    .. code-block:: python

        repr(user)  # <User admin>
        isinstance(user, User)  # True
        issubclass(type(user), LocalProxy)  # True

    .. versionchanged:: 2.2.2
        ``__wrapped__`` is set when wrapping an object, not only when
        wrapping a function, to prevent doctest from failing.

    .. versionchanged:: 2.2
        Can proxy a ``ContextVar`` or ``LocalStack`` directly.

    .. versionchanged:: 2.2
        The ``name`` parameter can be used with any proxied object, not
        only ``Local``.

    .. versionchanged:: 2.2
        Added the ``unbound_message`` parameter.

    .. versionchanged:: 2.0
        Updated proxied attributes and methods to reflect the current
        data model.

    .. versionchanged:: 0.6.1
        The class can be instantiated with a callable.
    """

    __slots__ = ("__wrapped", "_get_current_object")

    _get_current_object: t.Callable[[], T]
    """Return the current object this proxy is bound to. If the proxy is
    unbound, this raises a ``RuntimeError``.

    This should be used if you need to pass the object to something that
    doesn't understand the proxy. It can also be useful for performance
    if you are accessing the object multiple times in a function, rather
    than going through the proxy multiple times.
    """

    def __init__(
        self,
        local: ContextVar[T] | Local | LocalStack[T] | t.Callable[[], T],
        name: str | None = None,
        *,
        unbound_message: str | None = None,
    ) -> None:
        if name is None:
            get_name = _identity
        else:
            get_name = attrgetter(name)  # type: ignore[assignment]

        if unbound_message is None:
            unbound_message = "object is not bound"

        if isinstance(local, Local):
            if name is None:
                raise TypeError("'name' is required when proxying a 'Local' object.")

            def _get_current_object() -> T:
                try:
                    return get_name(local)  # type: ignore[return-value]
                except AttributeError:
                    raise RuntimeError(unbound_message) from None

        elif isinstance(local, LocalStack):

            def _get_current_object() -> T:
                obj = local.top

                if obj is None:
                    raise RuntimeError(unbound_message)

                return get_name(obj)

        elif isinstance(local, ContextVar):

            def _get_current_object() -> T:
                try:
                    obj = local.get()
                except LookupError:
                    raise RuntimeError(unbound_message) from None

                return get_name(obj)

        elif callable(local):

            def _get_current_object() -> T:
                return get_name(local())

        else:
            raise TypeError(f"Don't know how to proxy '{type(local)}'.")

        object.__setattr__(self, "_LocalProxy__wrapped", local)
        object.__setattr__(self, "_get_current_object", _get_current_object)

    __doc__ = _ProxyLookup(  # type: ignore[assignment]
        class_value=__doc__, fallback=lambda self: type(self).__doc__, is_attr=True
    )
    __wrapped__ = _ProxyLookup(
        fallback=lambda self: self._LocalProxy__wrapped,  # type: ignore[attr-defined]
        is_attr=True,
    )
    # __del__ should only delete the proxy
    __repr__ = _ProxyLookup(  # type: ignore[assignment]
        repr, fallback=lambda self: f"<{type(self).__name__} unbound>"
    )
    __str__ = _ProxyLookup(str)  # type: ignore[assignment]
    __bytes__ = _ProxyLookup(bytes)
    __format__ = _ProxyLookup()  # type: ignore[assignment]
    __lt__ = _ProxyLookup(operator.lt)
    __le__ = _ProxyLookup(operator.le)
    __eq__ = _ProxyLookup(operator.eq)  # type: ignore[assignment]
    __ne__ = _ProxyLookup(operator.ne)  # type: ignore[assignment]
    __gt__ = _ProxyLookup(operator.gt)
    __ge__ = _ProxyLookup(operator.ge)
    __hash__ = _ProxyLookup(hash)  # type: ignore[assignment]
    __bool__ = _ProxyLookup(bool, fallback=lambda self: False)
    __getattr__ = _ProxyLookup(getattr)
    # __getattribute__ triggered through __getattr__
    __setattr__ = _ProxyLookup(setattr)  # type: ignore[assignment]
    __delattr__ = _ProxyLookup(delattr)  # type: ignore[assignment]
    __dir__ = _ProxyLookup(dir, fallback=lambda self: [])  # type: ignore[assignment]
    # __get__ (proxying descriptor not supported)
    # __set__ (descriptor)
    # __delete__ (descriptor)
    # __set_name__ (descriptor)
    # __objclass__ (descriptor)
    # __slots__ used by proxy itself
    # __dict__ (__getattr__)
    # __weakref__ (__getattr__)
    # __init_subclass__ (proxying metaclass not supported)
    # __prepare__ (metaclass)
    __class__ = _ProxyLookup(fallback=lambda self: type(self), is_attr=True)  # type: ignore[assignment]
    __instancecheck__ = _ProxyLookup(lambda self, other: isinstance(other, self))
    __subclasscheck__ = _ProxyLookup(lambda self, other: issubclass(other, self))
    # __class_getitem__ triggered through __getitem__
    __call__ = _ProxyLookup(lambda self, *args, **kwargs: self(*args, **kwargs))
    __len__ = _ProxyLookup(len)
    __length_hint__ = _ProxyLookup(operator.length_hint)
    __getitem__ = _ProxyLookup(operator.getitem)
    __setitem__ = _ProxyLookup(operator.setitem)
    __delitem__ = _ProxyLookup(operator.delitem)
    # __missing__ triggered through __getitem__
    __iter__ = _ProxyLookup(iter)
    __next__ = _ProxyLookup(next)
    __reversed__ = _ProxyLookup(reversed)
    __contains__ = _ProxyLookup(operator.contains)
    __add__ = _ProxyLookup(operator.add)
    __sub__ = _ProxyLookup(operator.sub)
    __mul__ = _ProxyLookup(operator.mul)
    __matmul__ = _ProxyLookup(operator.matmul)
    __truediv__ = _ProxyLookup(operator.truediv)
    __floordiv__ = _ProxyLookup(operator.floordiv)
    __mod__ = _ProxyLookup(operator.mod)
    __divmod__ = _ProxyLookup(divmod)
    __pow__ = _ProxyLookup(pow)
    __lshift__ = _ProxyLookup(operator.lshift)
    __rshift__ = _ProxyLookup(operator.rshift)
    __and__ = _ProxyLookup(operator.and_)
    __xor__ = _ProxyLookup(operator.xor)
    __or__ = _ProxyLookup(operator.or_)
    __radd__ = _ProxyLookup(_l_to_r_op(operator.add))
    __rsub__ = _ProxyLookup(_l_to_r_op(operator.sub))
    __rmul__ = _ProxyLookup(_l_to_r_op(operator.mul))
    __rmatmul__ = _ProxyLookup(_l_to_r_op(operator.matmul))
    __rtruediv__ = _ProxyLookup(_l_to_r_op(operator.truediv))
    __rfloordiv__ = _ProxyLookup(_l_to_r_op(operator.floordiv))
    __rmod__ = _ProxyLookup(_l_to_r_op(operator.mod))
    __rdivmod__ = _ProxyLookup(_l_to_r_op(divmod))
    __rpow__ = _ProxyLookup(_l_to_r_op(pow))
    __rlshift__ = _ProxyLookup(_l_to_r_op(operator.lshift))
    __rrshift__ = _ProxyLookup(_l_to_r_op(operator.rshift))
    __rand__ = _ProxyLookup(_l_to_r_op(operator.and_))
    __rxor__ = _ProxyLookup(_l_to_r_op(operator.xor))
    __ror__ = _ProxyLookup(_l_to_r_op(operator.or_))
    __iadd__ = _ProxyIOp(operator.iadd)
    __isub__ = _ProxyIOp(operator.isub)
    __imul__ = _ProxyIOp(operator.imul)
    __imatmul__ = _ProxyIOp(operator.imatmul)
    __itruediv__ = _ProxyIOp(operator.itruediv)
    __ifloordiv__ = _ProxyIOp(operator.ifloordiv)
    __imod__ = _ProxyIOp(operator.imod)
    __ipow__ = _ProxyIOp(operator.ipow)
    __ilshift__ = _ProxyIOp(operator.ilshift)
    __irshift__ = _ProxyIOp(operator.irshift)
    __iand__ = _ProxyIOp(operator.iand)
    __ixor__ = _ProxyIOp(operator.ixor)
    __ior__ = _ProxyIOp(operator.ior)
    __neg__ = _ProxyLookup(operator.neg)
    __pos__ = _ProxyLookup(operator.pos)
    __abs__ = _ProxyLookup(abs)
    __invert__ = _ProxyLookup(operator.invert)
    __complex__ = _ProxyLookup(complex)
    __int__ = _ProxyLookup(int)
    __float__ = _ProxyLookup(float)
    __index__ = _ProxyLookup(operator.index)
    __round__ = _ProxyLookup(round)
    __trunc__ = _ProxyLookup(math.trunc)
    __floor__ = _ProxyLookup(math.floor)
    __ceil__ = _ProxyLookup(math.ceil)
    __enter__ = _ProxyLookup()
    __exit__ = _ProxyLookup()
    __await__ = _ProxyLookup()
    __aiter__ = _ProxyLookup()
    __anext__ = _ProxyLookup()
    __aenter__ = _ProxyLookup()
    __aexit__ = _ProxyLookup()
    __copy__ = _ProxyLookup(copy.copy)
    __deepcopy__ = _ProxyLookup(copy.deepcopy)
    # __getnewargs_ex__ (pickle through proxy not supported)
    # __getnewargs__ (pickle)
    # __getstate__ (pickle)
    # __setstate__ (pickle)
    # __reduce__ (pickle)
    # __reduce_ex__ (pickle)