
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\emoji.py ===
import sys
from typing import TYPE_CHECKING, Optional, Union

from .jupyter import JupyterMixin
from .segment import Segment
from .style import Style
from ._emoji_codes import EMOJI
from ._emoji_replace import _emoji_replace

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from pip._vendor.typing_extensions import Literal  # pragma: no cover


if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderResult


EmojiVariant = Literal["emoji", "text"]


class NoEmoji(Exception):
    """No emoji by that name."""


class Emoji(JupyterMixin):
    __slots__ = ["name", "style", "_char", "variant"]

    VARIANTS = {"text": "\uFE0E", "emoji": "\uFE0F"}

    def __init__(
        self,
        name: str,
        style: Union[str, Style] = "none",
        variant: Optional[EmojiVariant] = None,
    ) -> None:
        """A single emoji character.

        Args:
            name (str): Name of emoji.
            style (Union[str, Style], optional): Optional style. Defaults to None.

        Raises:
            NoEmoji: If the emoji doesn't exist.
        """
        self.name = name
        self.style = style
        self.variant = variant
        try:
            self._char = EMOJI[name]
        except KeyError:
            raise NoEmoji(f"No emoji called {name!r}")
        if variant is not None:
            self._char += self.VARIANTS.get(variant, "")

    @classmethod
    def replace(cls, text: str) -> str:
        """Replace emoji markup with corresponding unicode characters.

        Args:
            text (str): A string with emojis codes, e.g. "Hello :smiley:!"

        Returns:
            str: A string with emoji codes replaces with actual emoji.
        """
        return _emoji_replace(text)

    def __repr__(self) -> str:
        return f"<emoji {self.name!r}>"

    def __str__(self) -> str:
        return self._char

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        yield Segment(self._char, console.get_style(self.style))


if __name__ == "__main__":  # pragma: no cover
    import sys

    from pip._vendor.rich.columns import Columns
    from pip._vendor.rich.console import Console

    console = Console(record=True)

    columns = Columns(
        (f":{name}: {name}" for name in sorted(EMOJI.keys()) if "\u200D" not in name),
        column_first=True,
    )

    console.print(columns)
    if len(sys.argv) > 1:
        console.save_html(sys.argv[1])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\conftest.py ===
import sys

sys._running_pytest = True  # type: ignore
from sympy.external.importtools import version_tuple

import pytest
from sympy.core.cache import clear_cache, USE_CACHE
from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.misc import ARCH
import re

try:
    import hypothesis

    hypothesis.settings.register_profile("sympy_hypothesis_profile", deadline=None)
    hypothesis.settings.load_profile("sympy_hypothesis_profile")
except ImportError:
    raise ImportError(
        "hypothesis is a required dependency to run the SymPy test suite. "
        "Install it with 'pip install hypothesis' or 'conda install -c conda-forge hypothesis'"
    )


sp = re.compile(r"([0-9]+)/([1-9][0-9]*)")


def process_split(config, items):
    split = config.getoption("--split")
    if not split:
        return
    m = sp.match(split)
    if not m:
        raise ValueError(
            "split must be a string of the form a/b " "where a and b are ints."
        )
    i, t = map(int, m.groups())
    start, end = (i - 1) * len(items) // t, i * len(items) // t

    if i < t:
        # remove elements from end of list first
        del items[end:]
    del items[:start]


def pytest_report_header(config):
    s = "architecture: %s\n" % ARCH
    s += "cache:        %s\n" % USE_CACHE
    version = ""
    if GROUND_TYPES == "gmpy":
        import gmpy2

        version = gmpy2.version()
    elif GROUND_TYPES == "flint":
        try:
            from flint import __version__
        except ImportError:
            version = "unknown"
        else:
            version = f'(python-flint=={__version__})'
    s += "ground types: %s %s\n" % (GROUND_TYPES, version)
    return s


def pytest_terminal_summary(terminalreporter):
    if terminalreporter.stats.get("error", None) or terminalreporter.stats.get(
        "failed", None
    ):
        terminalreporter.write_sep(" ", "DO *NOT* COMMIT!", red=True, bold=True)


def pytest_addoption(parser):
    parser.addoption("--split", action="store", default="", help="split tests")


def pytest_collection_modifyitems(config, items):
    """pytest hook."""
    # handle splits
    process_split(config, items)


@pytest.fixture(autouse=True, scope="module")
def file_clear_cache():
    clear_cache()


@pytest.fixture(autouse=True, scope="module")
def check_disabled(request):
    if getattr(request.module, "disabled", False):
        pytest.skip("test requirements not met.")
    elif getattr(request.module, "ipython", False):
        # need to check version and options for ipython tests
        if (
            version_tuple(pytest.__version__) < version_tuple("2.6.3")
            and pytest.config.getvalue("-s") != "no"
        ):
            pytest.skip("run py.test with -s or upgrade to newer version.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\csrf\core.py ===
from wtforms.fields import HiddenField
from wtforms.validators import ValidationError

__all__ = ("CSRFTokenField", "CSRF")


class CSRFTokenField(HiddenField):
    """
    A subclass of HiddenField designed for sending the CSRF token that is used
    for most CSRF protection schemes.

    Notably different from a normal field, this field always renders the
    current token regardless of the submitted value, and also will not be
    populated over to object data via populate_obj
    """

    current_token = None

    def __init__(self, *args, **kw):
        self.csrf_impl = kw.pop("csrf_impl")
        super().__init__(*args, **kw)

    def _value(self):
        """
        We want to always return the current token on render, regardless of
        whether a good or bad token was passed.
        """
        return self.current_token

    def populate_obj(self, *args):
        """
        Don't populate objects with the CSRF token
        """
        pass

    def pre_validate(self, form):
        """
        Handle validation of this token field.
        """
        self.csrf_impl.validate_csrf_token(form, self)

    def process(self, *args, **kwargs):
        super().process(*args, **kwargs)
        self.current_token = self.csrf_impl.generate_csrf_token(self)


class CSRF:
    field_class = CSRFTokenField

    def setup_form(self, form):
        """
        Receive the form we're attached to and set up fields.

        The default implementation creates a single field of
        type :attr:`field_class` with name taken from the
        ``csrf_field_name`` of the class meta.

        :param form:
            The form instance we're attaching to.
        :return:
            A sequence of `(field_name, unbound_field)` 2-tuples which
            are unbound fields to be added to the form.
        """
        meta = form.meta
        field_name = meta.csrf_field_name
        unbound_field = self.field_class(label="CSRF Token", csrf_impl=self)
        return [(field_name, unbound_field)]

    def generate_csrf_token(self, csrf_token_field):
        """
        Implementations must override this to provide a method with which one
        can get a CSRF token for this form.

        A CSRF token is usually a string that is generated deterministically
        based on some sort of user data, though it can be anything which you
        can validate on a subsequent request.

        :param csrf_token_field:
            The field which is being used for CSRF.
        :return:
            A generated CSRF string.
        """
        raise NotImplementedError()

    def validate_csrf_token(self, form, field):
        """
        Override this method to provide custom CSRF validation logic.

        The default CSRF validation logic simply checks if the recently
        generated token equals the one we received as formdata.

        :param form: The form which has this CSRF token.
        :param field: The CSRF token field.
        """
        if field.current_token != field.data:
            raise ValidationError(field.gettext("Invalid CSRF Token."))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\exceptions.py ===
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import ClassVar


class FrozenError(AttributeError):
    """
    A frozen/immutable instance or attribute have been attempted to be
    modified.

    It mirrors the behavior of ``namedtuples`` by using the same error message
    and subclassing `AttributeError`.

    .. versionadded:: 20.1.0
    """

    msg = "can't set attribute"
    args: ClassVar[tuple[str]] = [msg]


class FrozenInstanceError(FrozenError):
    """
    A frozen instance has been attempted to be modified.

    .. versionadded:: 16.1.0
    """


class FrozenAttributeError(FrozenError):
    """
    A frozen attribute has been attempted to be modified.

    .. versionadded:: 20.1.0
    """


class AttrsAttributeNotFoundError(ValueError):
    """
    An *attrs* function couldn't find an attribute that the user asked for.

    .. versionadded:: 16.2.0
    """


class NotAnAttrsClassError(ValueError):
    """
    A non-*attrs* class has been passed into an *attrs* function.

    .. versionadded:: 16.2.0
    """


class DefaultAlreadySetError(RuntimeError):
    """
    A default has been set when defining the field and is attempted to be reset
    using the decorator.

    .. versionadded:: 17.1.0
    """


class UnannotatedAttributeError(RuntimeError):
    """
    A class with ``auto_attribs=True`` has a field without a type annotation.

    .. versionadded:: 17.3.0
    """


class PythonTooOldError(RuntimeError):
    """
    It was attempted to use an *attrs* feature that requires a newer Python
    version.

    .. versionadded:: 18.2.0
    """


class NotCallableError(TypeError):
    """
    A field requiring a callable has been set with a value that is not
    callable.

    .. versionadded:: 19.2.0
    """

    def __init__(self, msg, value):
        super(TypeError, self).__init__(msg, value)
        self.msg = msg
        self.value = value

    def __str__(self):
        return str(self.msg)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_utils\__init__.py ===
"""
This is a module for defining private helpers which do not depend on the
rest of NumPy.

Everything in here must be self-contained so that it can be
imported anywhere else without creating circular imports.
If a utility requires the import of NumPy, it probably belongs
in ``numpy._core``.
"""

import functools
import warnings

from ._convertions import asbytes, asunicode


def set_module(module):
    """Private decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module('numpy')
        def example():
            pass

        assert example.__module__ == 'numpy'
    """
    def decorator(func):
        if module is not None:
            if isinstance(func, type):
                try:
                    func._module_source = func.__module__
                except (AttributeError):
                    pass

            func.__module__ = module
        return func
    return decorator


def _rename_parameter(old_names, new_names, dep_version=None):
    """
    Generate decorator for backward-compatible keyword renaming.

    Apply the decorator generated by `_rename_parameter` to functions with a
    renamed parameter to maintain backward-compatibility.

    After decoration, the function behaves as follows:
    If only the new parameter is passed into the function, behave as usual.
    If only the old parameter is passed into the function (as a keyword), raise
    a DeprecationWarning if `dep_version` is provided, and behave as usual
    otherwise.
    If both old and new parameters are passed into the function, raise a
    DeprecationWarning if `dep_version` is provided, and raise the appropriate
    TypeError (function got multiple values for argument).

    Parameters
    ----------
    old_names : list of str
        Old names of parameters
    new_name : list of str
        New names of parameters
    dep_version : str, optional
        Version of NumPy in which old parameter was deprecated in the format
        'X.Y.Z'. If supplied, the deprecation message will indicate that
        support for the old parameter will be removed in version 'X.Y+2.Z'

    Notes
    -----
    Untested with functions that accept *args. Probably won't work as written.

    """
    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            __tracebackhide__ = True  # Hide traceback for py.test
            for old_name, new_name in zip(old_names, new_names):
                if old_name in kwargs:
                    if dep_version:
                        end_version = dep_version.split('.')
                        end_version[1] = str(int(end_version[1]) + 2)
                        end_version = '.'.join(end_version)
                        msg = (f"Use of keyword argument `{old_name}` is "
                               f"deprecated and replaced by `{new_name}`. "
                               f"Support for `{old_name}` will be removed "
                               f"in NumPy {end_version}.")
                        warnings.warn(msg, DeprecationWarning, stacklevel=2)
                    if new_name in kwargs:
                        msg = (f"{fun.__name__}() got multiple values for "
                               f"argument now known as `{new_name}`")
                        raise TypeError(msg)
                    kwargs[new_name] = kwargs.pop(old_name)
            return fun(*args, **kwargs)
        return wrapper
    return decorator

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\differential.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors import (
    Typed,
    Sequence,
    Alias,
)
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.styles import (
    Font,
    Fill,
    Border,
    Alignment,
    Protection,
    )
from .numbers import NumberFormat


class DifferentialStyle(Serialisable):

    tagname = "dxf"

    __elements__ = ("font", "numFmt", "fill", "alignment", "border", "protection")

    font = Typed(expected_type=Font, allow_none=True)
    numFmt = Typed(expected_type=NumberFormat, allow_none=True)
    fill = Typed(expected_type=Fill, allow_none=True)
    alignment = Typed(expected_type=Alignment, allow_none=True)
    border = Typed(expected_type=Border, allow_none=True)
    protection = Typed(expected_type=Protection, allow_none=True)

    def __init__(self,
                 font=None,
                 numFmt=None,
                 fill=None,
                 alignment=None,
                 border=None,
                 protection=None,
                 extLst=None,
                ):
        self.font = font
        self.numFmt = numFmt
        self.fill = fill
        self.alignment = alignment
        self.border = border
        self.protection = protection
        self.extLst = extLst


class DifferentialStyleList(Serialisable):
    """
    Dedupable container for differential styles.
    """

    tagname = "dxfs"

    dxf = Sequence(expected_type=DifferentialStyle)
    styles = Alias("dxf")
    __attrs__ = ("count",)


    def __init__(self, dxf=(), count=None):
        self.dxf = dxf


    def append(self, dxf):
        """
        Check to see whether style already exists and append it if does not.
        """
        if not isinstance(dxf, DifferentialStyle):
            raise TypeError('expected ' + str(DifferentialStyle))
        if dxf in self.styles:
            return
        self.styles.append(dxf)


    def add(self, dxf):
        """
        Add a differential style and return its index
        """
        self.append(dxf)
        return self.styles.index(dxf)


    def __bool__(self):
        return bool(self.styles)


    def __getitem__(self, idx):
        return self.styles[idx]


    @property
    def count(self):
        return len(self.dxf)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sniffio\_impl.py ===
from contextvars import ContextVar
from typing import Optional
import sys
import threading

current_async_library_cvar = ContextVar(
    "current_async_library_cvar", default=None
)  # type: ContextVar[Optional[str]]


class _ThreadLocal(threading.local):
    # Since threading.local provides no explicit mechanism is for setting
    # a default for a value, a custom class with a class attribute is used
    # instead.
    name = None  # type: Optional[str]


thread_local = _ThreadLocal()


class AsyncLibraryNotFoundError(RuntimeError):
    pass


def current_async_library() -> str:
    """Detect which async library is currently running.

    The following libraries are currently supported:

    ================   ===========  ============================
    Library             Requires     Magic string
    ================   ===========  ============================
    **Trio**            Trio v0.6+   ``"trio"``
    **Curio**           -            ``"curio"``
    **asyncio**                      ``"asyncio"``
    **Trio-asyncio**    v0.8.2+     ``"trio"`` or ``"asyncio"``,
                                    depending on current mode
    ================   ===========  ============================

    Returns:
      A string like ``"trio"``.

    Raises:
      AsyncLibraryNotFoundError: if called from synchronous context,
        or if the current async library was not recognized.

    Examples:

        .. code-block:: python3

           from sniffio import current_async_library

           async def generic_sleep(seconds):
               library = current_async_library()
               if library == "trio":
                   import trio
                   await trio.sleep(seconds)
               elif library == "asyncio":
                   import asyncio
                   await asyncio.sleep(seconds)
               # ... and so on ...
               else:
                   raise RuntimeError(f"Unsupported library {library!r}")

    """
    value = thread_local.name
    if value is not None:
        return value

    value = current_async_library_cvar.get()
    if value is not None:
        return value

    # Need to sniff for asyncio
    if "asyncio" in sys.modules:
        import asyncio
        try:
            current_task = asyncio.current_task  # type: ignore[attr-defined]
        except AttributeError:
            current_task = asyncio.Task.current_task  # type: ignore[attr-defined]
        try:
            if current_task() is not None:
                return "asyncio"
        except RuntimeError:
            pass

    # Sniff for curio (for now)
    if 'curio' in sys.modules:
        from curio.meta import curio_running
        if curio_running():
            return 'curio'

    raise AsyncLibraryNotFoundError(
        "unknown async library, or not in async context"
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\interactive\traversal.py ===
from sympy.core.basic import Basic
from sympy.printing import pprint

import random

def interactive_traversal(expr):
    """Traverse a tree asking a user which branch to choose. """

    RED, BRED = '\033[0;31m', '\033[1;31m'
    GREEN, BGREEN = '\033[0;32m', '\033[1;32m'
    YELLOW, BYELLOW = '\033[0;33m', '\033[1;33m'  # noqa
    BLUE, BBLUE = '\033[0;34m', '\033[1;34m'      # noqa
    MAGENTA, BMAGENTA = '\033[0;35m', '\033[1;35m'# noqa
    CYAN, BCYAN = '\033[0;36m', '\033[1;36m'      # noqa
    END = '\033[0m'

    def cprint(*args):
        print("".join(map(str, args)) + END)

    def _interactive_traversal(expr, stage):
        if stage > 0:
            print()

        cprint("Current expression (stage ", BYELLOW, stage, END, "):")
        print(BCYAN)
        pprint(expr)
        print(END)

        if isinstance(expr, Basic):
            if expr.is_Add:
                args = expr.as_ordered_terms()
            elif expr.is_Mul:
                args = expr.as_ordered_factors()
            else:
                args = expr.args
        elif hasattr(expr, "__iter__"):
            args = list(expr)
        else:
            return expr

        n_args = len(args)

        if not n_args:
            return expr

        for i, arg in enumerate(args):
            cprint(GREEN, "[", BGREEN, i, GREEN, "] ", BLUE, type(arg), END)
            pprint(arg)
            print()

        if n_args == 1:
            choices = '0'
        else:
            choices = '0-%d' % (n_args - 1)

        try:
            choice = input("Your choice [%s,f,l,r,d,?]: " % choices)
        except EOFError:
            result = expr
            print()
        else:
            if choice == '?':
                cprint(RED, "%s - select subexpression with the given index" %
                       choices)
                cprint(RED, "f - select the first subexpression")
                cprint(RED, "l - select the last subexpression")
                cprint(RED, "r - select a random subexpression")
                cprint(RED, "d - done\n")

                result = _interactive_traversal(expr, stage)
            elif choice in ('d', ''):
                result = expr
            elif choice == 'f':
                result = _interactive_traversal(args[0], stage + 1)
            elif choice == 'l':
                result = _interactive_traversal(args[-1], stage + 1)
            elif choice == 'r':
                result = _interactive_traversal(random.choice(args), stage + 1)
            else:
                try:
                    choice = int(choice)
                except ValueError:
                    cprint(BRED,
                           "Choice must be a number in %s range\n" % choices)
                    result = _interactive_traversal(expr, stage)
                else:
                    if choice < 0 or choice >= n_args:
                        cprint(BRED, "Choice must be in %s range\n" % choices)
                        result = _interactive_traversal(expr, stage)
                    else:
                        result = _interactive_traversal(args[choice], stage + 1)

        return result

    return _interactive_traversal(expr, 0)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\sho.py ===
from sympy.core import S, pi, Rational
from sympy.functions import assoc_laguerre, sqrt, exp, factorial, factorial2


def R_nl(n, l, nu, r):
    """
    Returns the radial wavefunction R_{nl} for a 3d isotropic harmonic
    oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.  Corresponds to the number of nodes in
        the wavefunction.  ``n >= 0``
    l :
        The quantum number for orbital angular momentum.
    nu :
        mass-scaled frequency: nu = m*omega/(2*hbar) where `m` is the mass
        and `omega` the frequency of the oscillator.
        (in atomic units ``nu == omega/2``)
    r :
        Radial coordinate.

    Examples
    ========

    >>> from sympy.physics.sho import R_nl
    >>> from sympy.abc import r, nu, l
    >>> R_nl(0, 0, 1, r)
    2*2**(3/4)*exp(-r**2)/pi**(1/4)
    >>> R_nl(1, 0, 1, r)
    4*2**(1/4)*sqrt(3)*(3/2 - 2*r**2)*exp(-r**2)/(3*pi**(1/4))

    l, nu and r may be symbolic:

    >>> R_nl(0, 0, nu, r)
    2*2**(3/4)*sqrt(nu**(3/2))*exp(-nu*r**2)/pi**(1/4)
    >>> R_nl(0, l, 1, r)
    r**l*sqrt(2**(l + 3/2)*2**(l + 2)/factorial2(2*l + 1))*exp(-r**2)/pi**(1/4)

    The normalization of the radial wavefunction is:

    >>> from sympy import Integral, oo
    >>> Integral(R_nl(0, 0, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000
    >>> Integral(R_nl(1, 0, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000
    >>> Integral(R_nl(1, 1, 1, r)**2*r**2, (r, 0, oo)).n()
    1.00000000000000

    """
    n, l, nu, r = map(S, [n, l, nu, r])

    # formula uses n >= 1 (instead of nodal n >= 0)
    n = n + 1
    C = sqrt(
            ((2*nu)**(l + Rational(3, 2))*2**(n + l + 1)*factorial(n - 1))/
            (sqrt(pi)*(factorial2(2*n + 2*l - 1)))
    )
    return C*r**(l)*exp(-nu*r**2)*assoc_laguerre(n - 1, l + S.Half, 2*nu*r**2)


def E_nl(n, l, hw):
    """
    Returns the Energy of an isotropic harmonic oscillator.

    Parameters
    ==========

    n :
        The "nodal" quantum number.
    l :
        The orbital angular momentum.
    hw :
        The harmonic oscillator parameter.

    Notes
    =====

    The unit of the returned value matches the unit of hw, since the energy is
    calculated as:

        E_nl = (2*n + l + 3/2)*hw

    Examples
    ========

    >>> from sympy.physics.sho import E_nl
    >>> from sympy import symbols
    >>> x, y, z = symbols('x, y, z')
    >>> E_nl(x, y, z)
    z*(2*x + y + 3/2)
    """
    return (2*n + l + Rational(3, 2))*hw

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\dagger.py ===
"""Hermitian conjugation."""

from sympy.core import Expr, sympify
from sympy.functions.elementary.complexes import adjoint

__all__ = [
    'Dagger'
]


class Dagger(adjoint):
    """General Hermitian conjugate operation.

    Explanation
    ===========

    Take the Hermetian conjugate of an argument [1]_. For matrices this
    operation is equivalent to transpose and complex conjugate [2]_.

    Parameters
    ==========

    arg : Expr
        The SymPy expression that we want to take the dagger of.
    evaluate : bool
        Whether the resulting expression should be directly evaluated.

    Examples
    ========

    Daggering various quantum objects:

        >>> from sympy.physics.quantum.dagger import Dagger
        >>> from sympy.physics.quantum.state import Ket, Bra
        >>> from sympy.physics.quantum.operator import Operator
        >>> Dagger(Ket('psi'))
        <psi|
        >>> Dagger(Bra('phi'))
        |phi>
        >>> Dagger(Operator('A'))
        Dagger(A)

    Inner and outer products::

        >>> from sympy.physics.quantum import InnerProduct, OuterProduct
        >>> Dagger(InnerProduct(Bra('a'), Ket('b')))
        <b|a>
        >>> Dagger(OuterProduct(Ket('a'), Bra('b')))
        |b><a|

    Powers, sums and products::

        >>> A = Operator('A')
        >>> B = Operator('B')
        >>> Dagger(A*B)
        Dagger(B)*Dagger(A)
        >>> Dagger(A+B)
        Dagger(A) + Dagger(B)
        >>> Dagger(A**2)
        Dagger(A)**2

    Dagger also seamlessly handles complex numbers and matrices::

        >>> from sympy import Matrix, I
        >>> m = Matrix([[1,I],[2,I]])
        >>> m
        Matrix([
        [1, I],
        [2, I]])
        >>> Dagger(m)
        Matrix([
        [ 1,  2],
        [-I, -I]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermitian_adjoint
    .. [2] https://en.wikipedia.org/wiki/Hermitian_transpose
    """

    @property
    def kind(self):
        """Find the kind of a dagger of something (just the kind of the something)."""
        return self.args[0].kind

    def __new__(cls, arg, evaluate=True):
        if hasattr(arg, 'adjoint') and evaluate:
            return arg.adjoint()
        elif hasattr(arg, 'conjugate') and hasattr(arg, 'transpose') and evaluate:
            return arg.conjugate().transpose()
        return Expr.__new__(cls, sympify(arg))

adjoint.__name__ = "Dagger"
adjoint._sympyrepr = lambda a, b: "Dagger(%s)" % b._print(a.args[0])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\_compat.py ===
# SPDX-License-Identifier: MIT

import inspect
import platform
import sys
import threading

from collections.abc import Mapping, Sequence  # noqa: F401
from typing import _GenericAlias


PYPY = platform.python_implementation() == "PyPy"
PY_3_9_PLUS = sys.version_info[:2] >= (3, 9)
PY_3_10_PLUS = sys.version_info[:2] >= (3, 10)
PY_3_11_PLUS = sys.version_info[:2] >= (3, 11)
PY_3_12_PLUS = sys.version_info[:2] >= (3, 12)
PY_3_13_PLUS = sys.version_info[:2] >= (3, 13)
PY_3_14_PLUS = sys.version_info[:2] >= (3, 14)


if PY_3_14_PLUS:  # pragma: no cover
    import annotationlib

    _get_annotations = annotationlib.get_annotations

else:

    def _get_annotations(cls):
        """
        Get annotations for *cls*.
        """
        return cls.__dict__.get("__annotations__", {})


class _AnnotationExtractor:
    """
    Extract type annotations from a callable, returning None whenever there
    is none.
    """

    __slots__ = ["sig"]

    def __init__(self, callable):
        try:
            self.sig = inspect.signature(callable)
        except (ValueError, TypeError):  # inspect failed
            self.sig = None

    def get_first_param_type(self):
        """
        Return the type annotation of the first argument if it's not empty.
        """
        if not self.sig:
            return None

        params = list(self.sig.parameters.values())
        if params and params[0].annotation is not inspect.Parameter.empty:
            return params[0].annotation

        return None

    def get_return_type(self):
        """
        Return the return type if it's not empty.
        """
        if (
            self.sig
            and self.sig.return_annotation is not inspect.Signature.empty
        ):
            return self.sig.return_annotation

        return None


# Thread-local global to track attrs instances which are already being repr'd.
# This is needed because there is no other (thread-safe) way to pass info
# about the instances that are already being repr'd through the call stack
# in order to ensure we don't perform infinite recursion.
#
# For instance, if an instance contains a dict which contains that instance,
# we need to know that we're already repr'ing the outside instance from within
# the dict's repr() call.
#
# This lives here rather than in _make.py so that the functions in _make.py
# don't have a direct reference to the thread-local in their globals dict.
# If they have such a reference, it breaks cloudpickle.
repr_context = threading.local()


def get_generic_base(cl):
    """If this is a generic class (A[str]), return the generic base for it."""
    if cl.__class__ is _GenericAlias:
        return cl.__origin__
    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_nbit_base.py ===
"""A module with the precisions of generic `~numpy.number` types."""
from typing import final

from numpy._utils import set_module


@final  # Disallow the creation of arbitrary `NBitBase` subclasses
@set_module("numpy.typing")
class NBitBase:
    """
    A type representing `numpy.number` precision during static type checking.

    Used exclusively for the purpose of static type checking, `NBitBase`
    represents the base of a hierarchical set of subclasses.
    Each subsequent subclass is herein used for representing a lower level
    of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

    .. versionadded:: 1.20

    .. deprecated:: 2.3
        Use ``@typing.overload`` or a ``TypeVar`` with a scalar-type as upper
        bound, instead.

    Examples
    --------
    Below is a typical usage example: `NBitBase` is herein used for annotating
    a function that takes a float and integer of arbitrary precision
    as arguments and returns a new float of whichever precision is largest
    (*e.g.* ``np.float16 + np.int64 -> np.float64``).

    .. code-block:: python

        >>> from typing import TypeVar, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> S = TypeVar("S", bound=npt.NBitBase)
        >>> T = TypeVar("T", bound=npt.NBitBase)

        >>> def add(a: np.floating[S], b: np.integer[T]) -> np.floating[S | T]:
        ...     return a + b

        >>> a = np.float16()
        >>> b = np.int64()
        >>> out = add(a, b)

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
        ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
        ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

    """
    # Deprecated in NumPy 2.3, 2025-05-01

    def __init_subclass__(cls) -> None:
        allowed_names = {
            "NBitBase", "_128Bit", "_96Bit", "_64Bit", "_32Bit", "_16Bit", "_8Bit"
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()

@final
@set_module("numpy._typing")
# Silence errors about subclassing a `@final`-decorated class
class _128Bit(NBitBase):  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    pass

@final
@set_module("numpy._typing")
class _96Bit(_128Bit):  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    pass

@final
@set_module("numpy._typing")
class _64Bit(_96Bit):  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    pass

@final
@set_module("numpy._typing")
class _32Bit(_64Bit):  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    pass

@final
@set_module("numpy._typing")
class _16Bit(_32Bit):  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    pass

@final
@set_module("numpy._typing")
class _8Bit(_16Bit):  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\table.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Float,
    Bool,
    Set,
    Integer,
    NoneSet,
    String,
    Sequence
)

from .colors import Color


class TableStyleElement(Serialisable):

    tagname = "tableStyleElement"

    type = Set(values=(['wholeTable', 'headerRow', 'totalRow', 'firstColumn',
                        'lastColumn', 'firstRowStripe', 'secondRowStripe', 'firstColumnStripe',
                        'secondColumnStripe', 'firstHeaderCell', 'lastHeaderCell',
                        'firstTotalCell', 'lastTotalCell', 'firstSubtotalColumn',
                        'secondSubtotalColumn', 'thirdSubtotalColumn', 'firstSubtotalRow',
                        'secondSubtotalRow', 'thirdSubtotalRow', 'blankRow',
                        'firstColumnSubheading', 'secondColumnSubheading',
                        'thirdColumnSubheading', 'firstRowSubheading', 'secondRowSubheading',
                        'thirdRowSubheading', 'pageFieldLabels', 'pageFieldValues']))
    size = Integer(allow_none=True)
    dxfId = Integer(allow_none=True)

    def __init__(self,
                 type=None,
                 size=None,
                 dxfId=None,
                ):
        self.type = type
        self.size = size
        self.dxfId = dxfId


class TableStyle(Serialisable):

    tagname = "tableStyle"

    name = String()
    pivot = Bool(allow_none=True)
    table = Bool(allow_none=True)
    count = Integer(allow_none=True)
    tableStyleElement = Sequence(expected_type=TableStyleElement, allow_none=True)

    __elements__ = ('tableStyleElement',)

    def __init__(self,
                 name=None,
                 pivot=None,
                 table=None,
                 count=None,
                 tableStyleElement=(),
                ):
        self.name = name
        self.pivot = pivot
        self.table = table
        self.count = count
        self.tableStyleElement = tableStyleElement


class TableStyleList(Serialisable):

    tagname = "tableStyles"

    defaultTableStyle = String(allow_none=True)
    defaultPivotStyle = String(allow_none=True)
    tableStyle = Sequence(expected_type=TableStyle, allow_none=True)

    __elements__ = ('tableStyle',)
    __attrs__ = ("count", "defaultTableStyle", "defaultPivotStyle")

    def __init__(self,
                 count=None,
                 defaultTableStyle="TableStyleMedium9",
                 defaultPivotStyle="PivotStyleLight16",
                 tableStyle=(),
                ):
        self.defaultTableStyle = defaultTableStyle
        self.defaultPivotStyle = defaultPivotStyle
        self.tableStyle = tableStyle


    @property
    def count(self):
        return len(self.tableStyle)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\pagebreak.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Integer,
    Bool,
    Sequence,
)


class Break(Serialisable):

    tagname = "brk"

    id = Integer(allow_none=True)
    min = Integer(allow_none=True)
    max = Integer(allow_none=True)
    man = Bool(allow_none=True)
    pt = Bool(allow_none=True)

    def __init__(self,
                 id=0,
                 min=0,
                 max=16383,
                 man=True,
                 pt=None,
                ):
        self.id = id
        self.min = min
        self.max = max
        self.man = man
        self.pt = pt


class RowBreak(Serialisable):

    tagname = "rowBreaks"

    count = Integer(allow_none=True)
    manualBreakCount = Integer(allow_none=True)
    brk = Sequence(expected_type=Break, allow_none=True)

    __elements__ = ('brk',)
    __attrs__ = ("count", "manualBreakCount",)

    def __init__(self,
                 count=None,
                 manualBreakCount=None,
                 brk=(),
                ):
        self.brk = brk


    def __bool__(self):
        return len(self.brk) > 0


    def __len__(self):
        return len(self.brk)


    @property
    def count(self):
        return len(self)


    @property
    def manualBreakCount(self):
        return len(self)


    def append(self, brk=None):
        """
        Add a page break
        """
        vals = list(self.brk)
        if not isinstance(brk, Break):
            brk = Break(id=self.count+1)
        vals.append(brk)
        self.brk = vals


PageBreak = RowBreak


class ColBreak(RowBreak):

    tagname = "colBreaks"

    count = RowBreak.count
    manualBreakCount = RowBreak.manualBreakCount
    brk = RowBreak.brk

    __attrs__ = RowBreak.__attrs__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\_util.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas.compat import (
    pa_version_under18p0,
    pa_version_under19p0,
)
from pandas.compat._optional import import_optional_dependency

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow

    from pandas._typing import DtypeBackend


def _arrow_dtype_mapping() -> dict:
    pa = import_optional_dependency("pyarrow")
    return {
        pa.int8(): pd.Int8Dtype(),
        pa.int16(): pd.Int16Dtype(),
        pa.int32(): pd.Int32Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.uint8(): pd.UInt8Dtype(),
        pa.uint16(): pd.UInt16Dtype(),
        pa.uint32(): pd.UInt32Dtype(),
        pa.uint64(): pd.UInt64Dtype(),
        pa.bool_(): pd.BooleanDtype(),
        pa.string(): pd.StringDtype(),
        pa.float32(): pd.Float32Dtype(),
        pa.float64(): pd.Float64Dtype(),
        pa.string(): pd.StringDtype(),
        pa.large_string(): pd.StringDtype(),
    }


def _arrow_string_types_mapper() -> Callable:
    pa = import_optional_dependency("pyarrow")

    mapping = {
        pa.string(): pd.StringDtype(na_value=np.nan),
        pa.large_string(): pd.StringDtype(na_value=np.nan),
    }
    if not pa_version_under18p0:
        mapping[pa.string_view()] = pd.StringDtype(na_value=np.nan)

    return mapping.get


def arrow_table_to_pandas(
    table: pyarrow.Table,
    dtype_backend: DtypeBackend | Literal["numpy"] | lib.NoDefault = lib.no_default,
    null_to_int64: bool = False,
    to_pandas_kwargs: dict | None = None,
) -> pd.DataFrame:
    if to_pandas_kwargs is None:
        to_pandas_kwargs = {}

    pa = import_optional_dependency("pyarrow")

    types_mapper: type[pd.ArrowDtype] | None | Callable
    if dtype_backend == "numpy_nullable":
        mapping = _arrow_dtype_mapping()
        if null_to_int64:
            # Modify the default mapping to also map null to Int64
            # (to match other engines - only for CSV parser)
            mapping[pa.null()] = pd.Int64Dtype()
        types_mapper = mapping.get
    elif dtype_backend == "pyarrow":
        types_mapper = pd.ArrowDtype
    elif using_string_dtype():
        if pa_version_under19p0:
            types_mapper = _arrow_string_types_mapper()
        else:
            types_mapper = None
    elif dtype_backend is lib.no_default or dtype_backend == "numpy":
        types_mapper = None
    else:
        raise NotImplementedError

    df = table.to_pandas(types_mapper=types_mapper, **to_pandas_kwargs)
    return df

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\console.py ===
"""
Internal module for console introspection
"""
from __future__ import annotations

from shutil import get_terminal_size


def get_console_size() -> tuple[int | None, int | None]:
    """
    Return console size as tuple = (width, height).

    Returns (None,None) in non-interactive session.
    """
    from pandas import get_option

    display_width = get_option("display.width")
    display_height = get_option("display.max_rows")

    # Consider
    # interactive shell terminal, can detect term size
    # interactive non-shell terminal (ipnb/ipqtconsole), cannot detect term
    # size non-interactive script, should disregard term size

    # in addition
    # width,height have default values, but setting to 'None' signals
    # should use Auto-Detection, But only in interactive shell-terminal.
    # Simple. yeah.

    if in_interactive_session():
        if in_ipython_frontend():
            # sane defaults for interactive non-shell terminal
            # match default for width,height in config_init
            from pandas._config.config import get_default_val

            terminal_width = get_default_val("display.width")
            terminal_height = get_default_val("display.max_rows")
        else:
            # pure terminal
            terminal_width, terminal_height = get_terminal_size()
    else:
        terminal_width, terminal_height = None, None

    # Note if the User sets width/Height to None (auto-detection)
    # and we're in a script (non-inter), this will return (None,None)
    # caller needs to deal.
    return display_width or terminal_width, display_height or terminal_height


# ----------------------------------------------------------------------
# Detect our environment


def in_interactive_session() -> bool:
    """
    Check if we're running in an interactive shell.

    Returns
    -------
    bool
        True if running under python/ipython interactive shell.
    """
    from pandas import get_option

    def check_main():
        try:
            import __main__ as main
        except ModuleNotFoundError:
            return get_option("mode.sim_interactive")
        return not hasattr(main, "__file__") or get_option("mode.sim_interactive")

    try:
        # error: Name '__IPYTHON__' is not defined
        return __IPYTHON__ or check_main()  # type: ignore[name-defined]
    except NameError:
        return check_main()


def in_ipython_frontend() -> bool:
    """
    Check if we're inside an IPython zmq frontend.

    Returns
    -------
    bool
    """
    try:
        # error: Name 'get_ipython' is not defined
        ip = get_ipython()  # type: ignore[name-defined]
        return "zmq" in str(type(ip)).lower()
    except NameError:
        pass

    return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_log_render.py ===
from datetime import datetime
from typing import Iterable, List, Optional, TYPE_CHECKING, Union, Callable


from .text import Text, TextType

if TYPE_CHECKING:
    from .console import Console, ConsoleRenderable, RenderableType
    from .table import Table

FormatTimeCallable = Callable[[datetime], Text]


class LogRender:
    def __init__(
        self,
        show_time: bool = True,
        show_level: bool = False,
        show_path: bool = True,
        time_format: Union[str, FormatTimeCallable] = "[%x %X]",
        omit_repeated_times: bool = True,
        level_width: Optional[int] = 8,
    ) -> None:
        self.show_time = show_time
        self.show_level = show_level
        self.show_path = show_path
        self.time_format = time_format
        self.omit_repeated_times = omit_repeated_times
        self.level_width = level_width
        self._last_time: Optional[Text] = None

    def __call__(
        self,
        console: "Console",
        renderables: Iterable["ConsoleRenderable"],
        log_time: Optional[datetime] = None,
        time_format: Optional[Union[str, FormatTimeCallable]] = None,
        level: TextType = "",
        path: Optional[str] = None,
        line_no: Optional[int] = None,
        link_path: Optional[str] = None,
    ) -> "Table":
        from .containers import Renderables
        from .table import Table

        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_time:
            output.add_column(style="log.time")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)
        output.add_column(ratio=1, style="log.message", overflow="fold")
        if self.show_path and path:
            output.add_column(style="log.path")
        row: List["RenderableType"] = []
        if self.show_time:
            log_time = log_time or console.get_datetime()
            time_format = time_format or self.time_format
            if callable(time_format):
                log_time_display = time_format(log_time)
            else:
                log_time_display = Text(log_time.strftime(time_format))
            if log_time_display == self._last_time and self.omit_repeated_times:
                row.append(Text(" " * len(log_time_display)))
            else:
                row.append(log_time_display)
                self._last_time = log_time_display
        if self.show_level:
            row.append(level)

        row.append(Renderables(renderables))
        if self.show_path and path:
            path_text = Text()
            path_text.append(
                path, style=f"link file://{link_path}" if link_path else ""
            )
            if line_no:
                path_text.append(":")
                path_text.append(
                    f"{line_no}",
                    style=f"link file://{link_path}#{line_no}" if link_path else "",
                )
            row.append(path_text)

        output.add_row(*row)
        return output


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich.console import Console

    c = Console()
    c.print("[on blue]Hello", justify="right")
    c.log("[on blue]hello", justify="right")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\pring.py ===
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.constants import hbar


def wavefunction(n, x):
    """
    Returns the wavefunction for particle on ring.

    Parameters
    ==========

    n : The quantum number.
        Here ``n`` can be positive as well as negative
        which can be used to describe the direction of motion of particle.
    x :
        The angle.

    Examples
    ========

    >>> from sympy.physics.pring import wavefunction
    >>> from sympy import Symbol, integrate, pi
    >>> x=Symbol("x")
    >>> wavefunction(1, x)
    sqrt(2)*exp(I*x)/(2*sqrt(pi))
    >>> wavefunction(2, x)
    sqrt(2)*exp(2*I*x)/(2*sqrt(pi))
    >>> wavefunction(3, x)
    sqrt(2)*exp(3*I*x)/(2*sqrt(pi))

    The normalization of the wavefunction is:

    >>> integrate(wavefunction(2, x)*wavefunction(-2, x), (x, 0, 2*pi))
    1
    >>> integrate(wavefunction(4, x)*wavefunction(-4, x), (x, 0, 2*pi))
    1

    References
    ==========

    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum
           Mechanics (4th ed.).  Pages 71-73.

    """
    # sympify arguments
    n, x = S(n), S(x)
    return exp(n * I * x) / sqrt(2 * pi)


def energy(n, m, r):
    """
    Returns the energy of the state corresponding to quantum number ``n``.

    E=(n**2 * (hcross)**2) / (2 * m * r**2)

    Parameters
    ==========

    n :
        The quantum number.
    m :
        Mass of the particle.
    r :
        Radius of circle.

    Examples
    ========

    >>> from sympy.physics.pring import energy
    >>> from sympy import Symbol
    >>> m=Symbol("m")
    >>> r=Symbol("r")
    >>> energy(1, m, r)
    hbar**2/(2*m*r**2)
    >>> energy(2, m, r)
    2*hbar**2/(m*r**2)
    >>> energy(-2, 2.0, 3.0)
    0.111111111111111*hbar**2

    References
    ==========

    .. [1] Atkins, Peter W.; Friedman, Ronald (2005). Molecular Quantum
           Mechanics (4th ed.).  Pages 71-73.

    """
    n, m, r = S(n), S(m), S(r)
    if n.is_integer:
        return (n**2 * hbar**2) / (2 * m * r**2)
    else:
        raise ValueError("'n' must be integer")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\body_base.py ===
from abc import ABC, abstractmethod
from sympy import Symbol, sympify
from sympy.physics.vector import Point

__all__ = ['BodyBase']


class BodyBase(ABC):
    """Abstract class for body type objects."""
    def __init__(self, name, masscenter=None, mass=None):
        # Note: If frame=None, no auto-generated frame is created, because a
        # Particle does not need to have a frame by default.
        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name
        if mass is None:
            mass = Symbol(f'{name}_mass')
        if masscenter is None:
            masscenter = Point(f'{name}_masscenter')
        self.mass = mass
        self.masscenter = masscenter
        self.potential_energy = 0
        self.points = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return (f'{self.__class__.__name__}({repr(self.name)}, masscenter='
                f'{repr(self.masscenter)}, mass={repr(self.mass)})')

    @property
    def name(self):
        """The name of the body."""
        return self._name

    @property
    def masscenter(self):
        """The body's center of mass."""
        return self._masscenter

    @masscenter.setter
    def masscenter(self, point):
        if not isinstance(point, Point):
            raise TypeError("The body's center of mass must be a Point object.")
        self._masscenter = point

    @property
    def mass(self):
        """The body's mass."""
        return self._mass

    @mass.setter
    def mass(self, mass):
        self._mass = sympify(mass)

    @property
    def potential_energy(self):
        """The potential energy of the body.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point
        >>> from sympy import symbols
        >>> m, g, h = symbols('m g h')
        >>> O = Point('O')
        >>> P = Particle('P', O, m)
        >>> P.potential_energy = m * g * h
        >>> P.potential_energy
        g*h*m

        """
        return self._potential_energy

    @potential_energy.setter
    def potential_energy(self, scalar):
        self._potential_energy = sympify(scalar)

    @abstractmethod
    def kinetic_energy(self, frame):
        pass

    @abstractmethod
    def linear_momentum(self, frame):
        pass

    @abstractmethod
    def angular_momentum(self, point, frame):
        pass

    @abstractmethod
    def parallel_axis(self, point, frame):
        pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\lll.py ===
from __future__ import annotations

from math import floor as mfloor

from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError


def _ddm_lll(x, delta=QQ(3, 4), return_transform=False):
    if QQ(1, 4) >= delta or delta >= QQ(1, 1):
        raise DMValueError("delta must lie in range (0.25, 1)")
    if x.shape[0] > x.shape[1]:
        raise DMShapeError("input matrix must have shape (m, n) with m <= n")
    if x.domain != ZZ:
        raise DMDomainError("input matrix domain must be ZZ")
    m = x.shape[0]
    n = x.shape[1]
    k = 1
    y = x.copy()
    y_star = x.zeros((m, n), QQ)
    mu = x.zeros((m, m), QQ)
    g_star = [QQ(0, 1) for _ in range(m)]
    half = QQ(1, 2)
    T = x.eye(m, ZZ) if return_transform else None
    linear_dependent_error = "input matrix contains linearly dependent rows"

    def closest_integer(x):
        return ZZ(mfloor(x + half))

    def lovasz_condition(k: int) -> bool:
        return g_star[k] >= ((delta - mu[k][k - 1] ** 2) * g_star[k - 1])

    def mu_small(k: int, j: int) -> bool:
        return abs(mu[k][j]) <= half

    def dot_rows(x, y, rows: tuple[int, int]):
        return sum(x[rows[0]][z] * y[rows[1]][z] for z in range(x.shape[1]))

    def reduce_row(T, mu, y, rows: tuple[int, int]):
        r = closest_integer(mu[rows[0]][rows[1]])
        y[rows[0]] = [y[rows[0]][z] - r * y[rows[1]][z] for z in range(n)]
        mu[rows[0]][:rows[1]] = [mu[rows[0]][z] - r * mu[rows[1]][z] for z in range(rows[1])]
        mu[rows[0]][rows[1]] -= r
        if return_transform:
            T[rows[0]] = [T[rows[0]][z] - r * T[rows[1]][z] for z in range(m)]

    for i in range(m):
        y_star[i] = [QQ.convert_from(z, ZZ) for z in y[i]]
        for j in range(i):
            row_dot = dot_rows(y, y_star, (i, j))
            try:
                mu[i][j] = row_dot / g_star[j]
            except ZeroDivisionError:
                raise DMRankError(linear_dependent_error)
            y_star[i] = [y_star[i][z] - mu[i][j] * y_star[j][z] for z in range(n)]
        g_star[i] = dot_rows(y_star, y_star, (i, i))
    while k < m:
        if not mu_small(k, k - 1):
            reduce_row(T, mu, y, (k, k - 1))
        if lovasz_condition(k):
            for l in range(k - 2, -1, -1):
                if not mu_small(k, l):
                    reduce_row(T, mu, y, (k, l))
            k += 1
        else:
            nu = mu[k][k - 1]
            alpha = g_star[k] + nu ** 2 * g_star[k - 1]
            try:
                beta = g_star[k - 1] / alpha
            except ZeroDivisionError:
                raise DMRankError(linear_dependent_error)
            mu[k][k - 1] = nu * beta
            g_star[k] = g_star[k] * beta
            g_star[k - 1] = alpha
            y[k], y[k - 1] = y[k - 1], y[k]
            mu[k][:k - 1], mu[k - 1][:k - 1] = mu[k - 1][:k - 1], mu[k][:k - 1]
            for i in range(k + 1, m):
                xi = mu[i][k]
                mu[i][k] = mu[i][k - 1] - nu * xi
                mu[i][k - 1] = mu[k][k - 1] * mu[i][k] + xi
            if return_transform:
                T[k], T[k - 1] = T[k - 1], T[k]
            k = max(k - 1, 1)
    assert all(lovasz_condition(i) for i in range(1, m))
    assert all(mu_small(i, j) for i in range(m) for j in range(i))
    return y, T


def ddm_lll(x, delta=QQ(3, 4)):
    return _ddm_lll(x, delta=delta, return_transform=False)[0]


def ddm_lll_transform(x, delta=QQ(3, 4)):
    return _ddm_lll(x, delta=delta, return_transform=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\file_manager.py ===
import os
import re
import tarfile
import zipfile

from webdriver_manager.core.archive import Archive, LinuxZipFileWithPermissions
from webdriver_manager.core.os_manager import OperationSystemManager


class File(object):
    def __init__(self, stream, file_name):
        self.content = stream.content
        self.__stream = stream
        self.file_name = file_name
        self.__temp_name = "driver"
        self.__regex_filename = r"""filename.+"(.+)"|filename.+''(.+)|filename=([\w.-]+)"""

    @property
    def filename(self) -> str:
        if self.file_name:
            return self.file_name
        try:
            content = self.__stream.headers["content-disposition"]

            content_disposition_list = re.split(";", content)
            filenames = [re.findall(self.__regex_filename, element) for element in content_disposition_list]
            filename = next(filter(None, next(filter(None, next(filter(None, filenames))))))
        except KeyError:
            filename = f"{self.__temp_name}.zip"
        except (IndexError, StopIteration):
            filename = f"{self.__temp_name}.exe"

        if '"' in filename:
            filename = filename.replace('"', "")

        return filename


class FileManager(object):

    def __init__(self, os_system_manager: OperationSystemManager):
        self._os_system_manager = os_system_manager

    def save_archive_file(self, file: File, directory: str):
        os.makedirs(directory, exist_ok=True)

        archive_path = f"{directory}{os.sep}{file.filename}"
        with open(archive_path, "wb") as code:
            code.write(file.content)
        if not os.path.exists(archive_path):
            raise FileExistsError(f"No file has been saved on such path {archive_path}")
        return Archive(archive_path)

    def unpack_archive(self, archive_file: Archive, target_dir):
        file_path = archive_file.file_path
        if file_path.endswith(".zip"):
            return self.__extract_zip(archive_file, target_dir)
        elif file_path.endswith(".tar.gz"):
            return self.__extract_tar_file(archive_file, target_dir)

    def __extract_zip(self, archive_file, to_directory):
        zip_class = (LinuxZipFileWithPermissions if self._os_system_manager.get_os_name() == "linux" else zipfile.ZipFile)
        archive = zip_class(archive_file.file_path)
        try:
            archive.extractall(to_directory)
        except Exception as e:
            if e.args[0] not in [26, 13] and e.args[1] not in [
                "Text file busy",
                "Permission denied",
            ]:
                raise e
            file_names = []
            for n in archive.namelist():
                if "/" not in n:
                    file_names.append(n)
                else:
                    file_path, file_name = n.split("/")
                    full_file_path = os.path.join(to_directory, file_path)
                    source = os.path.join(full_file_path, file_name)
                    destination = os.path.join(to_directory, file_name)
                    os.replace(source, destination)
                    file_names.append(file_name)
            return sorted(file_names, key=lambda x: x.lower())
        return archive.namelist()

    def __extract_tar_file(self, archive_file, to_directory):
        try:
            tar = tarfile.open(archive_file.file_path, mode="r:gz")
        except tarfile.ReadError:
            tar = tarfile.open(archive_file.file_path, mode="r:bz2")
        members = tar.getmembers()
        tar.extractall(to_directory)
        tar.close()
        return [x.name for x in members]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_exceptions.py ===
"""
_exceptions.py
websocket - WebSocket client library for Python

Copyright 2024 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


class WebSocketException(Exception):
    """
    WebSocket exception class.
    """

    pass


class WebSocketProtocolException(WebSocketException):
    """
    If the WebSocket protocol is invalid, this exception will be raised.
    """

    pass


class WebSocketPayloadException(WebSocketException):
    """
    If the WebSocket payload is invalid, this exception will be raised.
    """

    pass


class WebSocketConnectionClosedException(WebSocketException):
    """
    If remote host closed the connection or some network error happened,
    this exception will be raised.
    """

    pass


class WebSocketTimeoutException(WebSocketException):
    """
    WebSocketTimeoutException will be raised at socket timeout during read/write data.
    """

    pass


class WebSocketProxyException(WebSocketException):
    """
    WebSocketProxyException will be raised when proxy error occurred.
    """

    pass


class WebSocketBadStatusException(WebSocketException):
    """
    WebSocketBadStatusException will be raised when we get bad handshake status code.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        status_message=None,
        resp_headers=None,
        resp_body=None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.resp_headers = resp_headers
        self.resp_body = resp_body


class WebSocketAddressException(WebSocketException):
    """
    If the websocket address info cannot be found, this exception will be raised.
    """

    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wsproto\__init__.py ===
"""
wsproto
~~~~~~~

A WebSocket implementation.
"""
from typing import Generator, Optional, Union

from .connection import Connection, ConnectionState, ConnectionType
from .events import Event
from .handshake import H11Handshake
from .typing import Headers

__version__ = "1.2.0"


class WSConnection:
    """
    Represents the local end of a WebSocket connection to a remote peer.
    """

    def __init__(self, connection_type: ConnectionType) -> None:
        """
        Constructor

        :param wsproto.connection.ConnectionType connection_type: Controls
            whether the library behaves as a client or as a server.
        """
        self.client = connection_type is ConnectionType.CLIENT
        self.handshake = H11Handshake(connection_type)
        self.connection: Optional[Connection] = None

    @property
    def state(self) -> ConnectionState:
        """
        :returns: Connection state
        :rtype: wsproto.connection.ConnectionState
        """
        if self.connection is None:
            return self.handshake.state
        return self.connection.state

    def initiate_upgrade_connection(
        self, headers: Headers, path: Union[bytes, str]
    ) -> None:
        self.handshake.initiate_upgrade_connection(headers, path)

    def send(self, event: Event) -> bytes:
        """
        Generate network data for the specified event.

        When you want to communicate with a WebSocket peer, you should construct
        an event and pass it to this method. This method will return the bytes
        that you should send to the peer.

        :param wsproto.events.Event event: The event to generate data for
        :returns bytes: The data to send to the peer
        """
        data = b""
        if self.connection is None:
            data += self.handshake.send(event)
            self.connection = self.handshake.connection
        else:
            data += self.connection.send(event)
        return data

    def receive_data(self, data: Optional[bytes]) -> None:
        """
        Feed network data into the connection instance.

        After calling this method, you should call :meth:`events` to see if the
        received data triggered any new events.

        :param bytes data: Data received from remote peer
        """
        if self.connection is None:
            self.handshake.receive_data(data)
            self.connection = self.handshake.connection
        else:
            self.connection.receive_data(data)

    def events(self) -> Generator[Event, None, None]:
        """
        A generator that yields pending events.

        Each event is an instance of a subclass of
        :class:`wsproto.events.Event`.
        """
        yield from self.handshake.events()
        if self.connection is not None:
            yield from self.connection.events()


__all__ = ("ConnectionType", "WSConnection")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tests\reshape\merge\test_merge.py ===
from datetime import (
    date,
    datetime,
    timedelta,
)
import re

import numpy as np
import pytest

from pandas.core.dtypes.common import (
    is_object_dtype,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

import pandas as pd
from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
    MergeError,
    merge,
)


def get_test_data(ngroups=8, n=50):
    unique_groups = list(range(ngroups))
    arr = np.asarray(np.tile(unique_groups, n // ngroups))

    if len(arr) < n:
        arr = np.asarray(list(arr) + unique_groups[: n - len(arr)])

    np.random.default_rng(2).shuffle(arr)
    return arr


def get_series():
    return [
        Series([1], dtype="int64"),
        Series([1], dtype="Int64"),
        Series([1.23]),
        Series(["foo"]),
        Series([True]),
        Series([pd.Timestamp("2018-01-01")]),
        Series([pd.Timestamp("2018-01-01", tz="US/Eastern")]),
    ]


def get_series_na():
    return [
        Series([np.nan], dtype="Int64"),
        Series([np.nan], dtype="float"),
        Series([np.nan], dtype="object"),
        Series([pd.NaT]),
    ]


@pytest.fixture(params=get_series(), ids=lambda x: x.dtype.name)
def series_of_dtype(request):
    """
    A parametrized fixture returning a variety of Series of different
    dtypes
    """
    return request.param


@pytest.fixture(params=get_series(), ids=lambda x: x.dtype.name)
def series_of_dtype2(request):
    """
    A duplicate of the series_of_dtype fixture, so that it can be used
    twice by a single function
    """
    return request.param


@pytest.fixture(params=get_series_na(), ids=lambda x: x.dtype.name)
def series_of_dtype_all_na(request):
    """
    A parametrized fixture returning a variety of Series with all NA
    values
    """
    return request.param


@pytest.fixture
def dfs_for_indicator():
    df1 = DataFrame({"col1": [0, 1], "col_conflict": [1, 2], "col_left": ["a", "b"]})
    df2 = DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col_conflict": [1, 2, 3, 4, 5],
            "col_right": [2, 2, 2, 2, 2],
        }
    )
    return df1, df2


class TestMerge:
    @pytest.fixture
    def df(self):
        df = DataFrame(
            {
                "key1": get_test_data(),
                "key2": get_test_data(),
                "data1": np.random.default_rng(2).standard_normal(50),
                "data2": np.random.default_rng(2).standard_normal(50),
            }
        )

        # exclude a couple keys for fun
        df = df[df["key2"] > 1]
        return df

    @pytest.fixture
    def df2(self):
        return DataFrame(
            {
                "key1": get_test_data(n=10),
                "key2": get_test_data(ngroups=4, n=10),
                "value": np.random.default_rng(2).standard_normal(10),
            }
        )

    @pytest.fixture
    def left(self):
        return DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )

    @pytest.fixture
    def right(self):
        return DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )

    def test_merge_inner_join_empty(self):
        # GH 15328
        df_empty = DataFrame()
        df_a = DataFrame({"a": [1, 2]}, index=[0, 1], dtype="int64")
        result = merge(df_empty, df_a, left_index=True, right_index=True)
        expected = DataFrame({"a": []}, dtype="int64")
        tm.assert_frame_equal(result, expected)

    def test_merge_common(self, df, df2):
        joined = merge(df, df2)
        exp = merge(df, df2, on=["key1", "key2"])
        tm.assert_frame_equal(joined, exp)

    def test_merge_non_string_columns(self):
        # https://github.com/pandas-dev/pandas/issues/17962
        # Checks that method runs for non string column names
        left = DataFrame(
            {0: [1, 0, 1, 0], 1: [0, 1, 0, 0], 2: [0, 0, 2, 0], 3: [1, 0, 0, 3]}
        )

        right = left.astype(float)
        expected = left
        result = merge(left, right)
        tm.assert_frame_equal(expected, result)

    def test_merge_index_as_on_arg(self, df, df2):
        # GH14355

        left = df.set_index("key1")
        right = df2.set_index("key1")
        result = merge(left, right, on="key1")
        expected = merge(df, df2, on="key1").set_index("key1")
        tm.assert_frame_equal(result, expected)

    def test_merge_index_singlekey_right_vs_left(self):
        left = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )

        merged1 = merge(
            left, right, left_on="key", right_index=True, how="left", sort=False
        )
        merged2 = merge(
            right, left, right_on="key", left_index=True, how="right", sort=False
        )
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

        merged1 = merge(
            left, right, left_on="key", right_index=True, how="left", sort=True
        )
        merged2 = merge(
            right, left, right_on="key", left_index=True, how="right", sort=True
        )
        tm.assert_frame_equal(merged1, merged2.loc[:, merged1.columns])

    def test_merge_index_singlekey_inner(self):
        left = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "e", "a"],
                "v1": np.random.default_rng(2).standard_normal(7),
            }
        )
        right = DataFrame(
            {"v2": np.random.default_rng(2).standard_normal(4)},
            index=["d", "b", "c", "a"],
        )

        # inner join
        result = merge(left, right, left_on="key", right_index=True, how="inner")
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected)

        result = merge(right, left, right_on="key", left_index=True, how="inner")
        expected = left.join(right, on="key").loc[result.index]
        tm.assert_frame_equal(result, expected.loc[:, result.columns])

    def test_merge_misspecified(self, df, df2, left, right):
        msg = "Must pass right_on or right_index=True"
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, left_index=True)
        msg = "Must pass left_on or left_index=True"
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, right, right_index=True)

        msg = (
            'Can only pass argument "on" OR "left_on" and "right_on", not '
            "a combination of both"
        )
        with pytest.raises(pd.errors.MergeError, match=msg):
            merge(left, left, left_on="key", on="key")

        msg = r"len\(right_on\) must equal len\(left_on\)"
        with pytest.raises(ValueError, match=msg):
            merge(df, df2, left_on=["key1"], right_on=["key1", "key2"])

    def test_index_and_on_parameters_confusion(self, df, df2):
        msg = "right_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=False,
                right_index=["key1", "key2"],
            )
        msg = "left_index parameter must be of type bool, not <class 'list'>"
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=["key1", "key2"],
                right_index=False,
            )
        with pytest.raises(ValueError, match=msg):
            merge(
                df,
                df2,
                how="left",
                left_index=["key1", "key2"],
                right_index=["key1", "key2"],
            )

    def test_merge_overlap(self, left):
        merged = merge(left, left, on="key")
        exp_len = (left["key"].value_counts() ** 2).sum()
        assert len(merged) == exp_len
        assert "v1_x" in merged
        assert "v1_y" in merged

    def test_merge_different_column_key_names(self):
        left = DataFrame({"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 4]})
        right = DataFrame({"rkey": ["foo", "bar", "qux", "foo"], "value": [5, 6, 7, 8]})

        merged = left.merge(
            right, left_on="lkey", right_on="rkey", how="outer", sort=True
        )

        exp = Series(["bar", "baz", "foo", "foo", "foo", "foo", np.nan], name="lkey")
        tm.assert_series_equal(merged["lkey"], exp)

        exp = Series(["bar", np.nan, "foo", "foo", "foo", "foo", "qux"], name="rkey")
        tm.assert_series_equal(merged["rkey"], exp)

        exp = Series([2, 3, 1, 1, 4, 4, np.nan], name="value_x")
        tm.assert_series_equal(merged["value_x"], exp)

        exp = Series([6, np.nan, 5, 8, 5, 8, 7], name="value_y")
        tm.assert_series_equal(merged["value_y"], exp)

    def test_merge_copy(self):
        left = DataFrame({"a": 0, "b": 1}, index=range(10))
        right = DataFrame({"c": "foo", "d": "bar"}, index=range(10))

        merged = merge(left, right, left_index=True, right_index=True, copy=True)

        merged["a"] = 6
        assert (left["a"] == 0).all()

        merged["d"] = "peekaboo"
        assert (right["d"] == "bar").all()

    def test_merge_nocopy(self, using_array_manager, using_infer_string):
        left = DataFrame({"a": 0, "b": 1}, index=range(10))
        right = DataFrame({"c": "foo", "d": "bar"}, index=range(10))

        merged = merge(left, right, left_index=True, right_index=True, copy=False)

        assert np.shares_memory(merged["a"]._values, left["a"]._values)
        if not using_infer_string:
            assert np.shares_memory(merged["d"]._values, right["d"]._values)

    def test_intelligently_handle_join_key(self):
        # #733, be a bit more 1337 about not returning unconsolidated DataFrame

        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": list(range(5))}, columns=["value", "key"]
        )
        right = DataFrame({"key": [1, 1, 2, 3, 4, 5], "rvalue": list(range(6))})

        joined = merge(left, right, on="key", how="outer")
        expected = DataFrame(
            {
                "key": [1, 1, 1, 1, 2, 2, 3, 4, 5],
                "value": np.array([0, 0, 1, 1, 2, 3, 4, np.nan, np.nan]),
                "rvalue": [0, 1, 0, 1, 2, 2, 3, 4, 5],
            },
            columns=["value", "key", "rvalue"],
        )
        tm.assert_frame_equal(joined, expected)

    def test_merge_join_key_dtype_cast(self):
        # #8596

        df1 = DataFrame({"key": [1], "v1": [10]})
        df2 = DataFrame({"key": [2], "v1": [20]})
        df = merge(df1, df2, how="outer")
        assert df["key"].dtype == "int64"

        df1 = DataFrame({"key": [True], "v1": [1]})
        df2 = DataFrame({"key": [False], "v1": [0]})
        df = merge(df1, df2, how="outer")

        # GH13169
        # GH#40073
        assert df["key"].dtype == "bool"

        df1 = DataFrame({"val": [1]})
        df2 = DataFrame({"val": [2]})
        lkey = np.array([1])
        rkey = np.array([2])
        df = merge(df1, df2, left_on=lkey, right_on=rkey, how="outer")
        assert df["key_0"].dtype == np.dtype(int)

    def test_handle_join_key_pass_array(self):
        left = DataFrame(
            {"key": [1, 1, 2, 2, 3], "value": np.arange(5)},
            columns=["value", "key"],
            dtype="int64",
        )
        right = DataFrame({"rvalue": np.arange(6)}, dtype="int64")
        key = np.array([1, 1, 2, 3, 4, 5], dtype="int64")

        merged = merge(left, right, left_on="key", right_on=key, how="outer")
        merged2 = merge(right, left, left_on=key, right_on="key", how="outer")

        tm.assert_series_equal(merged["key"], merged2["key"])
        assert merged["key"].notna().all()
        assert merged2["key"].notna().all()

        left = DataFrame({"value": np.arange(5)}, columns=["value"])
        right = DataFrame({"rvalue": np.arange(6)})
        lkey = np.array([1, 1, 2, 2, 3])
        rkey = np.array([1, 1, 2, 3, 4, 5])

        merged = merge(left, right, left_on=lkey, right_on=rkey, how="outer")
        expected = Series([1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=int, name="key_0")
        tm.assert_series_equal(merged["key_0"], expected)

        left = DataFrame({"value": np.arange(3)})
        right = DataFrame({"rvalue": np.arange(6)})

        key = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
        merged = merge(left, right, left_index=True, right_on=key, how="outer")
        tm.assert_series_equal(merged["key_0"], Series(key, name="key_0"))

    def test_no_overlap_more_informative_error(self):
        dt = datetime.now()
        df1 = DataFrame({"x": ["a"]}, index=[dt])

        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])

        msg = (
            "No common columns to perform merge on. "
            f"Merge options: left_on={None}, right_on={None}, "
            f"left_index={False}, right_index={False}"
        )

        with pytest.raises(MergeError, match=msg):
            merge(df1, df2)

    def test_merge_non_unique_indexes(self):
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        dt4 = datetime(2012, 5, 4)

        df1 = DataFrame({"x": ["a"]}, index=[dt])
        df2 = DataFrame({"y": ["b", "c"]}, index=[dt, dt])
        _check_merge(df1, df2)

        # Not monotonic
        df1 = DataFrame({"x": ["a", "b", "q"]}, index=[dt2, dt, dt4])
        df2 = DataFrame(
            {"y": ["c", "d", "e", "f", "g", "h"]}, index=[dt3, dt3, dt2, dt2, dt, dt]
        )
        _check_merge(df1, df2)

        df1 = DataFrame({"x": ["a", "b"]}, index=[dt, dt])
        df2 = DataFrame({"y": ["c", "d"]}, index=[dt, dt])
        _check_merge(df1, df2)

    def test_merge_non_unique_index_many_to_many(self):
        dt = datetime(2012, 5, 1)
        dt2 = datetime(2012, 5, 2)
        dt3 = datetime(2012, 5, 3)
        df1 = DataFrame({"x": ["a", "b", "c", "d"]}, index=[dt2, dt2, dt, dt])
        df2 = DataFrame(
            {"y": ["e", "f", "g", " h", "i"]}, index=[dt2, dt2, dt3, dt, dt]
        )
        _check_merge(df1, df2)

    def test_left_merge_empty_dataframe(self):
        left = DataFrame({"key": [1], "value": [2]})
        right = DataFrame({"key": []})

        result = merge(left, right, on="key", how="left")
        tm.assert_frame_equal(result, left)

        result = merge(right, left, on="key", how="right")
        tm.assert_frame_equal(result, left)

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_merge_empty_dataframe(self, index, how):
        # GH52777
        left = DataFrame([], index=index[:0])
        right = left.copy()

        result = left.join(right, how=how)
        tm.assert_frame_equal(result, left)

    @pytest.mark.parametrize(
        "kwarg",
        [
            {"left_index": True, "right_index": True},
            {"left_index": True, "right_on": "x"},
            {"left_on": "a", "right_index": True},
            {"left_on": "a", "right_on": "x"},
        ],
    )
    def test_merge_left_empty_right_empty(self, join_type, kwarg):
        # GH 10824
        left = DataFrame(columns=["a", "b", "c"])
        right = DataFrame(columns=["x", "y", "z"])

        exp_in = DataFrame(columns=["a", "b", "c", "x", "y", "z"], dtype=object)

        result = merge(left, right, how=join_type, **kwarg)
        tm.assert_frame_equal(result, exp_in)

    def test_merge_left_empty_right_notempty(self):
        # GH 10824
        left = DataFrame(columns=["a", "b", "c"])
        right = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["x", "y", "z"])

        exp_out = DataFrame(
            {
                "a": np.array([np.nan] * 3, dtype=object),
                "b": np.array([np.nan] * 3, dtype=object),
                "c": np.array([np.nan] * 3, dtype=object),
                "x": [1, 4, 7],
                "y": [2, 5, 8],
                "z": [3, 6, 9],
            },
            columns=["a", "b", "c", "x", "y", "z"],
        )
        exp_in = exp_out[0:0]  # make empty DataFrame keeping dtype

        def check1(exp, kwarg):
            result = merge(left, right, how="inner", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="left", **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp, kwarg):
            result = merge(left, right, how="right", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="outer", **kwarg)
            tm.assert_frame_equal(result, exp)

        for kwarg in [
            {"left_index": True, "right_index": True},
            {"left_index": True, "right_on": "x"},
        ]:
            check1(exp_in, kwarg)
            check2(exp_out, kwarg)

        kwarg = {"left_on": "a", "right_index": True}
        check1(exp_in, kwarg)
        exp_out["a"] = [0, 1, 2]
        check2(exp_out, kwarg)

        kwarg = {"left_on": "a", "right_on": "x"}
        check1(exp_in, kwarg)
        exp_out["a"] = np.array([np.nan] * 3, dtype=object)
        check2(exp_out, kwarg)

    def test_merge_left_notempty_right_empty(self):
        # GH 10824
        left = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
        right = DataFrame(columns=["x", "y", "z"])

        exp_out = DataFrame(
            {
                "a": [1, 4, 7],
                "b": [2, 5, 8],
                "c": [3, 6, 9],
                "x": np.array([np.nan] * 3, dtype=object),
                "y": np.array([np.nan] * 3, dtype=object),
                "z": np.array([np.nan] * 3, dtype=object),
            },
            columns=["a", "b", "c", "x", "y", "z"],
        )
        exp_in = exp_out[0:0]  # make empty DataFrame keeping dtype
        # result will have object dtype
        exp_in.index = exp_in.index.astype(object)

        def check1(exp, kwarg):
            result = merge(left, right, how="inner", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="right", **kwarg)
            tm.assert_frame_equal(result, exp)

        def check2(exp, kwarg):
            result = merge(left, right, how="left", **kwarg)
            tm.assert_frame_equal(result, exp)
            result = merge(left, right, how="outer", **kwarg)
            tm.assert_frame_equal(result, exp)

            # TODO: should the next loop be un-indented? doing so breaks this test
            for kwarg in [
                {"left_index": True, "right_index": True},
                {"left_index": True, "right_on": "x"},
                {"left_on": "a", "right_index": True},
                {"left_on": "a", "right_on": "x"},
            ]:
                check1(exp_in, kwarg)
                check2(exp_out, kwarg)

    def test_merge_empty_frame(self, series_of_dtype, series_of_dtype2):
        # GH 25183
        df = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype2},
            columns=["key", "value"],
        )
        df_empty = df[:0]
        expected = DataFrame(
            {
                "key": Series(dtype=df.dtypes["key"]),
                "value_x": Series(dtype=df.dtypes["value"]),
                "value_y": Series(dtype=df.dtypes["value"]),
            },
            columns=["key", "value_x", "value_y"],
        )
        actual = df_empty.merge(df, on="key")
        tm.assert_frame_equal(actual, expected)

    def test_merge_all_na_column(self, series_of_dtype, series_of_dtype_all_na):
        # GH 25183
        df_left = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype_all_na},
            columns=["key", "value"],
        )
        df_right = DataFrame(
            {"key": series_of_dtype, "value": series_of_dtype_all_na},
            columns=["key", "value"],
        )
        expected = DataFrame(
            {
                "key": series_of_dtype,
                "value_x": series_of_dtype_all_na,
                "value_y": series_of_dtype_all_na,
            },
            columns=["key", "value_x", "value_y"],
        )
        actual = df_left.merge(df_right, on="key")
        tm.assert_frame_equal(actual, expected)

    def test_merge_nosort(self):
        # GH#2098

        d = {
            "var1": np.random.default_rng(2).integers(0, 10, size=10),
            "var2": np.random.default_rng(2).integers(0, 10, size=10),
            "var3": [
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2010, 2, 3),
                datetime(2012, 1, 12),
                datetime(2011, 2, 4),
                datetime(2012, 4, 3),
                datetime(2012, 3, 4),
                datetime(2008, 5, 1),
                datetime(2010, 2, 3),
                datetime(2012, 2, 3),
            ],
        }
        df = DataFrame.from_dict(d)
        var3 = df.var3.unique()
        var3 = np.sort(var3)
        new = DataFrame.from_dict(
            {"var3": var3, "var8": np.random.default_rng(2).random(7)}
        )

        result = df.merge(new, on="var3", sort=False)
        exp = merge(df, new, on="var3", sort=False)
        tm.assert_frame_equal(result, exp)

        assert (df.var3.unique() == result.var3.unique()).all()

    @pytest.mark.parametrize(
        ("sort", "values"), [(False, [1, 1, 0, 1, 1]), (True, [0, 1, 1, 1, 1])]
    )
    @pytest.mark.parametrize("how", ["left", "right"])
    def test_merge_same_order_left_right(self, sort, values, how):
        # GH#35382
        df = DataFrame({"a": [1, 0, 1]})

        result = df.merge(df, on="a", how=how, sort=sort)
        expected = DataFrame(values, columns=["a"])
        tm.assert_frame_equal(result, expected)

    def test_merge_nan_right(self):
        df1 = DataFrame({"i1": [0, 1], "i2": [0, 1]})
        df2 = DataFrame({"i1": [0], "i3": [0]})
        result = df1.join(df2, on="i1", rsuffix="_")
        expected = (
            DataFrame(
                {
                    "i1": {0: 0.0, 1: 1},
                    "i2": {0: 0, 1: 1},
                    "i1_": {0: 0, 1: np.nan},
                    "i3": {0: 0.0, 1: np.nan},
                    None: {0: 0, 1: 0},
                },
                columns=Index(["i1", "i2", "i1_", "i3", None], dtype=object),
            )
            .set_index(None)
            .reset_index()[["i1", "i2", "i1_", "i3"]]
        )
        result.columns = result.columns.astype("object")
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_merge_nan_right2(self):
        df1 = DataFrame({"i1": [0, 1], "i2": [0.5, 1.5]})
        df2 = DataFrame({"i1": [0], "i3": [0.7]})
        result = df1.join(df2, rsuffix="_", on="i1")
        expected = DataFrame(
            {
                "i1": {0: 0, 1: 1},
                "i1_": {0: 0.0, 1: np.nan},
                "i2": {0: 0.5, 1: 1.5},
                "i3": {0: 0.69999999999999996, 1: np.nan},
            }
        )[["i1", "i2", "i1_", "i3"]]
        tm.assert_frame_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_merge_type(self, df, df2):
        class NotADataFrame(DataFrame):
            @property
            def _constructor(self):
                return NotADataFrame

        nad = NotADataFrame(df)
        result = nad.merge(df2, on="key1")

        assert isinstance(result, NotADataFrame)

    def test_join_append_timedeltas(self, using_array_manager):
        # timedelta64 issues with join/merge
        # GH 5695

        d = DataFrame.from_dict(
            {"d": [datetime(2013, 11, 5, 5, 56)], "t": [timedelta(0, 22500)]}
        )
        df = DataFrame(columns=list("dt"))
        msg = "The behavior of DataFrame concatenation with empty or all-NA entries"
        warn = FutureWarning
        if using_array_manager:
            warn = None
        with tm.assert_produces_warning(warn, match=msg):
            df = concat([df, d], ignore_index=True)
            result = concat([df, d], ignore_index=True)
        expected = DataFrame(
            {
                "d": [datetime(2013, 11, 5, 5, 56), datetime(2013, 11, 5, 5, 56)],
                "t": [timedelta(0, 22500), timedelta(0, 22500)],
            }
        )
        if using_array_manager:
            # TODO(ArrayManager) decide on exact casting rules in concat
            expected = expected.astype(object)
        tm.assert_frame_equal(result, expected)

    def test_join_append_timedeltas2(self):
        # timedelta64 issues with join/merge
        # GH 5695
        td = np.timedelta64(300000000)
        lhs = DataFrame(Series([td, td], index=["A", "B"]))
        rhs = DataFrame(Series([td], index=["A"]))

        result = lhs.join(rhs, rsuffix="r", how="left")
        expected = DataFrame(
            {
                "0": Series([td, td], index=list("AB")),
                "0r": Series([td, pd.NaT], index=list("AB")),
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_other_datetime_unit(self, unit):
        # GH 13389
        df1 = DataFrame({"entity_id": [101, 102]})
        ser = Series([None, None], index=[101, 102], name="days")

        dtype = f"datetime64[{unit}]"

        if unit in ["D", "h", "m"]:
            # not supported so we cast to the nearest supported unit, seconds
            exp_dtype = "datetime64[s]"
        else:
            exp_dtype = dtype
        df2 = ser.astype(exp_dtype).to_frame("days")
        assert df2["days"].dtype == exp_dtype

        result = df1.merge(df2, left_on="entity_id", right_index=True)

        days = np.array(["nat", "nat"], dtype=exp_dtype)
        days = pd.core.arrays.DatetimeArray._simple_new(days, dtype=days.dtype)
        exp = DataFrame(
            {
                "entity_id": [101, 102],
                "days": days,
            },
            columns=["entity_id", "days"],
        )
        assert exp["days"].dtype == exp_dtype
        tm.assert_frame_equal(result, exp)

    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_other_timedelta_unit(self, unit):
        # GH 13389
        df1 = DataFrame({"entity_id": [101, 102]})
        ser = Series([None, None], index=[101, 102], name="days")

        dtype = f"m8[{unit}]"
        if unit in ["D", "h", "m"]:
            # We cannot astype, instead do nearest supported unit, i.e. "s"
            msg = "Supported resolutions are 's', 'ms', 'us', 'ns'"
            with pytest.raises(ValueError, match=msg):
                ser.astype(dtype)

            df2 = ser.astype("m8[s]").to_frame("days")
        else:
            df2 = ser.astype(dtype).to_frame("days")
            assert df2["days"].dtype == dtype

        result = df1.merge(df2, left_on="entity_id", right_index=True)

        exp = DataFrame(
            {"entity_id": [101, 102], "days": np.array(["nat", "nat"], dtype=dtype)},
            columns=["entity_id", "days"],
        )
        tm.assert_frame_equal(result, exp)

    def test_overlapping_columns_error_message(self):
        df = DataFrame({"key": [1, 2, 3], "v1": [4, 5, 6], "v2": [7, 8, 9]})
        df2 = DataFrame({"key": [1, 2, 3], "v1": [4, 5, 6], "v2": [7, 8, 9]})

        df.columns = ["key", "foo", "foo"]
        df2.columns = ["key", "bar", "bar"]
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "v1": [4, 5, 6],
                "v2": [7, 8, 9],
                "v3": [4, 5, 6],
                "v4": [7, 8, 9],
            }
        )
        expected.columns = ["key", "foo", "foo", "bar", "bar"]
        tm.assert_frame_equal(merge(df, df2), expected)

        # #2649, #10639
        df2.columns = ["key1", "foo", "foo"]
        msg = r"Data columns not unique: Index\(\['foo'\], dtype='object|str'\)"
        with pytest.raises(MergeError, match=msg):
            merge(df, df2)

    def test_merge_on_datetime64tz(self):
        # GH11405
        left = DataFrame(
            {
                "key": pd.date_range("20151010", periods=2, tz="US/Eastern"),
                "value": [1, 2],
            }
        )
        right = DataFrame(
            {
                "key": pd.date_range("20151011", periods=3, tz="US/Eastern"),
                "value": [1, 2, 3],
            }
        )

        expected = DataFrame(
            {
                "key": pd.date_range("20151010", periods=4, tz="US/Eastern"),
                "value_x": [1, 2, np.nan, np.nan],
                "value_y": [np.nan, 1, 2, 3],
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_values(self):
        left = DataFrame(
            {
                "key": [1, 2],
                "value": pd.date_range("20151010", periods=2, tz="US/Eastern"),
            }
        )
        right = DataFrame(
            {
                "key": [2, 3],
                "value": pd.date_range("20151011", periods=2, tz="US/Eastern"),
            }
        )
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "value_x": list(pd.date_range("20151010", periods=2, tz="US/Eastern"))
                + [pd.NaT],
                "value_y": [pd.NaT]
                + list(pd.date_range("20151011", periods=2, tz="US/Eastern")),
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)
        assert result["value_x"].dtype == "datetime64[ns, US/Eastern]"
        assert result["value_y"].dtype == "datetime64[ns, US/Eastern]"

    def test_merge_on_datetime64tz_empty(self):
        # https://github.com/pandas-dev/pandas/issues/25014
        dtz = pd.DatetimeTZDtype(tz="UTC")
        right = DataFrame(
            {
                "date": DatetimeIndex(["2018"], dtype=dtz),
                "value": [4.0],
                "date2": DatetimeIndex(["2019"], dtype=dtz),
            },
            columns=["date", "value", "date2"],
        )
        left = right[:0]
        result = left.merge(right, on="date")
        expected = DataFrame(
            {
                "date": Series(dtype=dtz),
                "value_x": Series(dtype=float),
                "date2_x": Series(dtype=dtz),
                "value_y": Series(dtype=float),
                "date2_y": Series(dtype=dtz),
            },
            columns=["date", "value_x", "date2_x", "value_y", "date2_y"],
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_datetime64tz_with_dst_transition(self):
        # GH 18885
        df1 = DataFrame(
            pd.date_range("2017-10-29 01:00", periods=4, freq="h", tz="Europe/Madrid"),
            columns=["date"],
        )
        df1["value"] = 1
        df2 = DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2017-10-29 03:00:00",
                        "2017-10-29 04:00:00",
                        "2017-10-29 05:00:00",
                    ]
                ),
                "value": 2,
            }
        )
        df2["date"] = df2["date"].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid")
        result = merge(df1, df2, how="outer", on="date")
        expected = DataFrame(
            {
                "date": pd.date_range(
                    "2017-10-29 01:00", periods=7, freq="h", tz="Europe/Madrid"
                ),
                "value_x": [1] * 4 + [np.nan] * 3,
                "value_y": [np.nan] * 4 + [2] * 3,
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_non_unique_period_index(self):
        # GH #16871
        index = pd.period_range("2016-01-01", periods=16, freq="M")
        df = DataFrame(list(range(len(index))), index=index, columns=["pnum"])
        df2 = concat([df, df])
        result = df.merge(df2, left_index=True, right_index=True, how="inner")
        expected = DataFrame(
            np.tile(np.arange(16, dtype=np.int64).repeat(2).reshape(-1, 1), 2),
            columns=["pnum_x", "pnum_y"],
            index=df2.sort_index().index,
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_on_periods(self):
        left = DataFrame(
            {"key": pd.period_range("20151010", periods=2, freq="D"), "value": [1, 2]}
        )
        right = DataFrame(
            {
                "key": pd.period_range("20151011", periods=3, freq="D"),
                "value": [1, 2, 3],
            }
        )

        expected = DataFrame(
            {
                "key": pd.period_range("20151010", periods=4, freq="D"),
                "value_x": [1, 2, np.nan, np.nan],
                "value_y": [np.nan, 1, 2, 3],
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)

    def test_merge_period_values(self):
        left = DataFrame(
            {"key": [1, 2], "value": pd.period_range("20151010", periods=2, freq="D")}
        )
        right = DataFrame(
            {"key": [2, 3], "value": pd.period_range("20151011", periods=2, freq="D")}
        )

        exp_x = pd.period_range("20151010", periods=2, freq="D")
        exp_y = pd.period_range("20151011", periods=2, freq="D")
        expected = DataFrame(
            {
                "key": [1, 2, 3],
                "value_x": list(exp_x) + [pd.NaT],
                "value_y": [pd.NaT] + list(exp_y),
            }
        )
        result = merge(left, right, on="key", how="outer")
        tm.assert_frame_equal(result, expected)
        assert result["value_x"].dtype == "Period[D]"
        assert result["value_y"].dtype == "Period[D]"

    def test_indicator(self, dfs_for_indicator):
        # PR #10054. xref #7412 and closes #8790.
        df1, df2 = dfs_for_indicator
        df1_copy = df1.copy()

        df2_copy = df2.copy()

        df_result = DataFrame(
            {
                "col1": [0, 1, 2, 3, 4, 5],
                "col_conflict_x": [1, 2, np.nan, np.nan, np.nan, np.nan],
                "col_left": ["a", "b", np.nan, np.nan, np.nan, np.nan],
                "col_conflict_y": [np.nan, 1, 2, 3, 4, 5],
                "col_right": [np.nan, 2, 2, 2, 2, 2],
            }
        )
        df_result["_merge"] = Categorical(
            [
                "left_only",
                "both",
                "right_only",
                "right_only",
                "right_only",
                "right_only",
            ],
            categories=["left_only", "right_only", "both"],
        )

        df_result = df_result[
            [
                "col1",
                "col_conflict_x",
                "col_left",
                "col_conflict_y",
                "col_right",
                "_merge",
            ]
        ]

        test = merge(df1, df2, on="col1", how="outer", indicator=True)
        tm.assert_frame_equal(test, df_result)
        test = df1.merge(df2, on="col1", how="outer", indicator=True)
        tm.assert_frame_equal(test, df_result)

        # No side effects
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)

        # Check with custom name
        df_result_custom_name = df_result
        df_result_custom_name = df_result_custom_name.rename(
            columns={"_merge": "custom_name"}
        )

        test_custom_name = merge(
            df1, df2, on="col1", how="outer", indicator="custom_name"
        )
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)
        test_custom_name = df1.merge(
            df2, on="col1", how="outer", indicator="custom_name"
        )
        tm.assert_frame_equal(test_custom_name, df_result_custom_name)

    def test_merge_indicator_arg_validation(self, dfs_for_indicator):
        # Check only accepts strings and booleans
        df1, df2 = dfs_for_indicator

        msg = "indicator option can only accept boolean or string arguments"
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on="col1", how="outer", indicator=5)
        with pytest.raises(ValueError, match=msg):
            df1.merge(df2, on="col1", how="outer", indicator=5)

    def test_merge_indicator_result_integrity(self, dfs_for_indicator):
        # Check result integrity
        df1, df2 = dfs_for_indicator

        test2 = merge(df1, df2, on="col1", how="left", indicator=True)
        assert (test2._merge != "right_only").all()
        test2 = df1.merge(df2, on="col1", how="left", indicator=True)
        assert (test2._merge != "right_only").all()

        test3 = merge(df1, df2, on="col1", how="right", indicator=True)
        assert (test3._merge != "left_only").all()
        test3 = df1.merge(df2, on="col1", how="right", indicator=True)
        assert (test3._merge != "left_only").all()

        test4 = merge(df1, df2, on="col1", how="inner", indicator=True)
        assert (test4._merge == "both").all()
        test4 = df1.merge(df2, on="col1", how="inner", indicator=True)
        assert (test4._merge == "both").all()

    def test_merge_indicator_invalid(self, dfs_for_indicator):
        # Check if working name in df
        df1, _ = dfs_for_indicator

        for i in ["_right_indicator", "_left_indicator", "_merge"]:
            df_badcolumn = DataFrame({"col1": [1, 2], i: [2, 2]})

            msg = (
                "Cannot use `indicator=True` option when data contains a "
                f"column named {i}|"
                "Cannot use name of an existing column for indicator column"
            )
            with pytest.raises(ValueError, match=msg):
                merge(df1, df_badcolumn, on="col1", how="outer", indicator=True)
            with pytest.raises(ValueError, match=msg):
                df1.merge(df_badcolumn, on="col1", how="outer", indicator=True)

        # Check for name conflict with custom name
        df_badcolumn = DataFrame({"col1": [1, 2], "custom_column_name": [2, 2]})

        msg = "Cannot use name of an existing column for indicator column"
        with pytest.raises(ValueError, match=msg):
            merge(
                df1,
                df_badcolumn,
                on="col1",
                how="outer",
                indicator="custom_column_name",
            )
        with pytest.raises(ValueError, match=msg):
            df1.merge(
                df_badcolumn, on="col1", how="outer", indicator="custom_column_name"
            )

    def test_merge_indicator_multiple_columns(self):
        # Merge on multiple columns
        df3 = DataFrame({"col1": [0, 1], "col2": ["a", "b"]})

        df4 = DataFrame({"col1": [1, 1, 3], "col2": ["b", "x", "y"]})

        hand_coded_result = DataFrame(
            {"col1": [0, 1, 1, 3], "col2": ["a", "b", "x", "y"]}
        )
        hand_coded_result["_merge"] = Categorical(
            ["left_only", "both", "right_only", "right_only"],
            categories=["left_only", "right_only", "both"],
        )

        test5 = merge(df3, df4, on=["col1", "col2"], how="outer", indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)
        test5 = df3.merge(df4, on=["col1", "col2"], how="outer", indicator=True)
        tm.assert_frame_equal(test5, hand_coded_result)

    def test_validation(self):
        left = DataFrame(
            {"a": ["a", "b", "c", "d"], "b": ["cat", "dog", "weasel", "horse"]},
            index=range(4),
        )

        right = DataFrame(
            {
                "a": ["a", "b", "c", "d", "e"],
                "c": ["meow", "bark", "um... weasel noise?", "nay", "chirp"],
            },
            index=range(5),
        )

        # Make sure no side effects.
        left_copy = left.copy()
        right_copy = right.copy()

        result = merge(left, right, left_index=True, right_index=True, validate="1:1")
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)

        # make sure merge still correct
        expected = DataFrame(
            {
                "a_x": ["a", "b", "c", "d"],
                "b": ["cat", "dog", "weasel", "horse"],
                "a_y": ["a", "b", "c", "d"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            index=range(4),
            columns=["a_x", "b", "a_y", "c"],
        )

        result = merge(
            left, right, left_index=True, right_index=True, validate="one_to_one"
        )
        tm.assert_frame_equal(result, expected)

        expected_2 = DataFrame(
            {
                "a": ["a", "b", "c", "d"],
                "b": ["cat", "dog", "weasel", "horse"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            index=range(4),
        )

        result = merge(left, right, on="a", validate="1:1")
        tm.assert_frame_equal(left, left_copy)
        tm.assert_frame_equal(right, right_copy)
        tm.assert_frame_equal(result, expected_2)

        result = merge(left, right, on="a", validate="one_to_one")
        tm.assert_frame_equal(result, expected_2)

        # One index, one column
        expected_3 = DataFrame(
            {
                "b": ["cat", "dog", "weasel", "horse"],
                "a": ["a", "b", "c", "d"],
                "c": ["meow", "bark", "um... weasel noise?", "nay"],
            },
            columns=["b", "a", "c"],
            index=range(4),
        )

        left_index_reset = left.set_index("a")
        result = merge(
            left_index_reset,
            right,
            left_index=True,
            right_on="a",
            validate="one_to_one",
        )
        tm.assert_frame_equal(result, expected_3)

        # Dups on right
        right_w_dups = concat([right, DataFrame({"a": ["e"], "c": ["moo"]}, index=[4])])
        merge(
            left,
            right_w_dups,
            left_index=True,
            right_index=True,
            validate="one_to_many",
        )

        msg = "Merge keys are not unique in right dataset; not a one-to-one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left,
                right_w_dups,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )

        with pytest.raises(MergeError, match=msg):
            merge(left, right_w_dups, on="a", validate="one_to_one")

        # Dups on left
        left_w_dups = concat(
            [left, DataFrame({"a": ["a"], "c": ["cow"]}, index=[3])], sort=True
        )
        merge(
            left_w_dups,
            right,
            left_index=True,
            right_index=True,
            validate="many_to_one",
        )

        msg = "Merge keys are not unique in left dataset; not a one-to-one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left_w_dups,
                right,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )

        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right, on="a", validate="one_to_one")

        # Dups on both
        merge(left_w_dups, right_w_dups, on="a", validate="many_to_many")

        msg = "Merge keys are not unique in right dataset; not a many-to-one merge"
        with pytest.raises(MergeError, match=msg):
            merge(
                left_w_dups,
                right_w_dups,
                left_index=True,
                right_index=True,
                validate="many_to_one",
            )

        msg = "Merge keys are not unique in left dataset; not a one-to-many merge"
        with pytest.raises(MergeError, match=msg):
            merge(left_w_dups, right_w_dups, on="a", validate="one_to_many")

        # Check invalid arguments
        msg = (
            '"jibberish" is not a valid argument. '
            "Valid arguments are:\n"
            '- "1:1"\n'
            '- "1:m"\n'
            '- "m:1"\n'
            '- "m:m"\n'
            '- "one_to_one"\n'
            '- "one_to_many"\n'
            '- "many_to_one"\n'
            '- "many_to_many"'
        )
        with pytest.raises(ValueError, match=msg):
            merge(left, right, on="a", validate="jibberish")

        # Two column merge, dups in both, but jointly no dups.
        left = DataFrame(
            {
                "a": ["a", "a", "b", "b"],
                "b": [0, 1, 0, 1],
                "c": ["cat", "dog", "weasel", "horse"],
            },
            index=range(4),
        )

        right = DataFrame(
            {
                "a": ["a", "a", "b"],
                "b": [0, 1, 0],
                "d": ["meow", "bark", "um... weasel noise?"],
            },
            index=range(3),
        )

        expected_multi = DataFrame(
            {
                "a": ["a", "a", "b"],
                "b": [0, 1, 0],
                "c": ["cat", "dog", "weasel"],
                "d": ["meow", "bark", "um... weasel noise?"],
            },
            index=range(3),
        )

        msg = (
            "Merge keys are not unique in either left or right dataset; "
            "not a one-to-one merge"
        )
        with pytest.raises(MergeError, match=msg):
            merge(left, right, on="a", validate="1:1")

        result = merge(left, right, on=["a", "b"], validate="1:1")
        tm.assert_frame_equal(result, expected_multi)

    def test_merge_two_empty_df_no_division_error(self):
        # GH17776, PR #17846
        a = DataFrame({"a": [], "b": [], "c": []})
        with np.errstate(divide="raise"):
            merge(a, a, on=("a", "b"))

    @pytest.mark.parametrize("how", ["right", "outer"])
    @pytest.mark.parametrize(
        "index,expected_index",
        [
            (
                CategoricalIndex([1, 2, 4]),
                CategoricalIndex([1, 2, 4, None, None, None]),
            ),
            (
                DatetimeIndex(
                    ["2001-01-01", "2002-02-02", "2003-03-03"], dtype="M8[ns]"
                ),
                DatetimeIndex(
                    ["2001-01-01", "2002-02-02", "2003-03-03", pd.NaT, pd.NaT, pd.NaT],
                    dtype="M8[ns]",
                ),
            ),
            *[
                (
                    Index([1, 2, 3], dtype=dtyp),
                    Index([1, 2, 3, None, None, None], dtype=np.float64),
                )
                for dtyp in tm.ALL_REAL_NUMPY_DTYPES
            ],
            (
                IntervalIndex.from_tuples([(1, 2), (2, 3), (3, 4)]),
                IntervalIndex.from_tuples(
                    [(1, 2), (2, 3), (3, 4), np.nan, np.nan, np.nan]
                ),
            ),
            (
                PeriodIndex(["2001-01-01", "2001-01-02", "2001-01-03"], freq="D"),
                PeriodIndex(
                    ["2001-01-01", "2001-01-02", "2001-01-03", pd.NaT, pd.NaT, pd.NaT],
                    freq="D",
                ),
            ),
            (
                TimedeltaIndex(["1d", "2d", "3d"]),
                TimedeltaIndex(["1d", "2d", "3d", pd.NaT, pd.NaT, pd.NaT]),
            ),
        ],
    )
    def test_merge_on_index_with_more_values(self, how, index, expected_index):
        # GH 24212
        # pd.merge gets [0, 1, 2, -1, -1, -1] as left_indexer, ensure that
        # -1 is interpreted as a missing value instead of the last element
        df1 = DataFrame({"a": [0, 1, 2], "key": [0, 1, 2]}, index=index)
        df2 = DataFrame({"b": [0, 1, 2, 3, 4, 5]})
        result = df1.merge(df2, left_on="key", right_index=True, how=how)
        expected = DataFrame(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [np.nan, 3, 3],
                [np.nan, 4, 4],
                [np.nan, 5, 5],
            ],
            columns=["a", "key", "b"],
        )
        expected.set_index(expected_index, inplace=True)
        tm.assert_frame_equal(result, expected)

    def test_merge_right_index_right(self):
        # Note: the expected output here is probably incorrect.
        # See https://github.com/pandas-dev/pandas/issues/17257 for more.
        # We include this as a regression test for GH-24897.
        left = DataFrame({"a": [1, 2, 3], "key": [0, 1, 1]})
        right = DataFrame({"b": [1, 2, 3]})

        expected = DataFrame(
            {"a": [1, 2, 3, None], "key": [0, 1, 1, 2], "b": [1, 2, 2, 3]},
            columns=["a", "key", "b"],
            index=[0, 1, 2, np.nan],
        )
        result = left.merge(right, left_on="key", right_index=True, how="right")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("how", ["left", "right"])
    def test_merge_preserves_row_order(self, how):
        # GH 27453
        left_df = DataFrame({"animal": ["dog", "pig"], "max_speed": [40, 11]})
        right_df = DataFrame({"animal": ["quetzal", "pig"], "max_speed": [80, 11]})
        result = left_df.merge(right_df, on=["animal", "max_speed"], how=how)
        if how == "right":
            expected = DataFrame({"animal": ["quetzal", "pig"], "max_speed": [80, 11]})
        else:
            expected = DataFrame({"animal": ["dog", "pig"], "max_speed": [40, 11]})
        tm.assert_frame_equal(result, expected)

    def test_merge_take_missing_values_from_index_of_other_dtype(self):
        # GH 24212
        left = DataFrame(
            {
                "a": [1, 2, 3],
                "key": Categorical(["a", "a", "b"], categories=list("abc")),
            }
        )
        right = DataFrame({"b": [1, 2, 3]}, index=CategoricalIndex(["a", "b", "c"]))
        result = left.merge(right, left_on="key", right_index=True, how="right")
        expected = DataFrame(
            {
                "a": [1, 2, 3, None],
                "key": Categorical(["a", "a", "b", "c"]),
                "b": [1, 1, 2, 3],
            },
            index=[0, 1, 2, np.nan],
        )
        expected = expected.reindex(columns=["a", "key", "b"])
        tm.assert_frame_equal(result, expected)

    def test_merge_readonly(self):
        # https://github.com/pandas-dev/pandas/issues/27943
        data1 = DataFrame(
            np.arange(20).reshape((4, 5)) + 1, columns=["a", "b", "c", "d", "e"]
        )
        data2 = DataFrame(
            np.arange(20).reshape((5, 4)) + 1, columns=["a", "b", "x", "y"]
        )

        # make each underlying block array / column array read-only
        for arr in data1._mgr.arrays:
            arr.flags.writeable = False

        data1.merge(data2)  # no error


def _check_merge(x, y):
    for how in ["inner", "left", "outer"]:
        for sort in [True, False]:
            result = x.join(y, how=how, sort=sort)

            expected = merge(x.reset_index(), y.reset_index(), how=how, sort=sort)
            expected = expected.set_index("index")

            # TODO check_names on merge?
            tm.assert_frame_equal(result, expected, check_names=False)


class TestMergeDtypes:
    @pytest.mark.parametrize(
        "right_vals", [["foo", "bar"], Series(["foo", "bar"]).astype("category")]
    )
    def test_different(self, right_vals):
        left = DataFrame(
            {
                "A": ["foo", "bar"],
                "B": Series(["foo", "bar"]).astype("category"),
                "C": [1, 2],
                "D": [1.0, 2.0],
                "E": Series([1, 2], dtype="uint64"),
                "F": Series([1, 2], dtype="int32"),
            }
        )
        right = DataFrame({"A": right_vals})

        # GH 9780
        # We allow merging on object and categorical cols and cast
        # categorical cols to object
        result = merge(left, right, on="A")
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize(
        "d1", [np.int64, np.int32, np.intc, np.int16, np.int8, np.uint8]
    )
    @pytest.mark.parametrize("d2", [np.int64, np.float64, np.float32, np.float16])
    def test_join_multi_dtypes(self, d1, d2):
        dtype1 = np.dtype(d1)
        dtype2 = np.dtype(d2)

        left = DataFrame(
            {
                "k1": np.array([0, 1, 2] * 8, dtype=dtype1),
                "k2": ["foo", "bar"] * 12,
                "v": np.array(np.arange(24), dtype=np.int64),
            }
        )

        index = MultiIndex.from_tuples([(2, "bar"), (1, "foo")])
        right = DataFrame({"v2": np.array([5, 7], dtype=dtype2)}, index=index)

        result = left.join(right, on=["k1", "k2"])

        expected = left.copy()

        if dtype2.kind == "i":
            dtype2 = np.dtype("float64")
        expected["v2"] = np.array(np.nan, dtype=dtype2)
        expected.loc[(expected.k1 == 2) & (expected.k2 == "bar"), "v2"] = 5
        expected.loc[(expected.k1 == 1) & (expected.k2 == "foo"), "v2"] = 7

        tm.assert_frame_equal(result, expected)

        result = left.join(right, on=["k1", "k2"], sort=True)
        expected.sort_values(["k1", "k2"], kind="mergesort", inplace=True)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "int_vals, float_vals, exp_vals",
        [
            ([1, 2, 3], [1.0, 2.0, 3.0], {"X": [1, 2, 3], "Y": [1.0, 2.0, 3.0]}),
            ([1, 2, 3], [1.0, 3.0], {"X": [1, 3], "Y": [1.0, 3.0]}),
            ([1, 2], [1.0, 2.0, 3.0], {"X": [1, 2], "Y": [1.0, 2.0]}),
        ],
    )
    def test_merge_on_ints_floats(self, int_vals, float_vals, exp_vals):
        # GH 16572
        # Check that float column is not cast to object if
        # merging on float and int columns
        A = DataFrame({"X": int_vals})
        B = DataFrame({"Y": float_vals})
        expected = DataFrame(exp_vals)

        result = A.merge(B, left_on="X", right_on="Y")
        tm.assert_frame_equal(result, expected)

        result = B.merge(A, left_on="Y", right_on="X")
        tm.assert_frame_equal(result, expected[["Y", "X"]])

    def test_merge_key_dtype_cast(self):
        # GH 17044
        df1 = DataFrame({"key": [1.0, 2.0], "v1": [10, 20]}, columns=["key", "v1"])
        df2 = DataFrame({"key": [2], "v2": [200]}, columns=["key", "v2"])
        result = df1.merge(df2, on="key", how="left")
        expected = DataFrame(
            {"key": [1.0, 2.0], "v1": [10, 20], "v2": [np.nan, 200.0]},
            columns=["key", "v1", "v2"],
        )
        tm.assert_frame_equal(result, expected)

    def test_merge_on_ints_floats_warning(self):
        # GH 16572
        # merge will produce a warning when merging on int and
        # float columns where the float values are not exactly
        # equal to their int representation
        A = DataFrame({"X": [1, 2, 3]})
        B = DataFrame({"Y": [1.1, 2.5, 3.0]})
        expected = DataFrame({"X": [3], "Y": [3.0]})

        with tm.assert_produces_warning(UserWarning):
            result = A.merge(B, left_on="X", right_on="Y")
            tm.assert_frame_equal(result, expected)

        with tm.assert_produces_warning(UserWarning):
            result = B.merge(A, left_on="Y", right_on="X")
            tm.assert_frame_equal(result, expected[["Y", "X"]])

        # test no warning if float has NaNs
        B = DataFrame({"Y": [np.nan, np.nan, 3.0]})

        with tm.assert_produces_warning(None):
            result = B.merge(A, left_on="Y", right_on="X")
            tm.assert_frame_equal(result, expected[["Y", "X"]])

    def test_merge_incompat_infer_boolean_object(self):
        # GH21119: bool + object bool merge OK
        df1 = DataFrame({"key": Series([True, False], dtype=object)})
        df2 = DataFrame({"key": [True, False]})

        expected = DataFrame({"key": [True, False]}, dtype=object)
        result = merge(df1, df2, on="key")
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on="key")
        tm.assert_frame_equal(result, expected)

    def test_merge_incompat_infer_boolean_object_with_missing(self):
        # GH21119: bool + object bool merge OK
        # with missing value
        df1 = DataFrame({"key": Series([True, False, np.nan], dtype=object)})
        df2 = DataFrame({"key": [True, False]})

        expected = DataFrame({"key": [True, False]}, dtype=object)
        result = merge(df1, df2, on="key")
        tm.assert_frame_equal(result, expected)
        result = merge(df2, df1, on="key")
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "df1_vals, df2_vals",
        [
            # merge on category coerces to object
            ([0, 1, 2], Series(["a", "b", "a"]).astype("category")),
            ([0.0, 1.0, 2.0], Series(["a", "b", "a"]).astype("category")),
            # no not infer
            ([0, 1], Series([False, True], dtype=object)),
            ([0, 1], Series([False, True], dtype=bool)),
        ],
    )
    def test_merge_incompat_dtypes_are_ok(self, df1_vals, df2_vals):
        # these are explicitly allowed incompat merges, that pass thru
        # the result type is dependent on if the values on the rhs are
        # inferred, otherwise these will be coerced to object

        df1 = DataFrame({"A": df1_vals})
        df2 = DataFrame({"A": df2_vals})

        result = merge(df1, df2, on=["A"])
        assert is_object_dtype(result.A.dtype)
        result = merge(df2, df1, on=["A"])
        assert is_object_dtype(result.A.dtype) or is_string_dtype(result.A.dtype)

    @pytest.mark.parametrize(
        "df1_vals, df2_vals",
        [
            # do not infer to numeric
            (Series([1, 2], dtype="uint64"), ["a", "b", "c"]),
            (Series([1, 2], dtype="int32"), ["a", "b", "c"]),
            ([0, 1, 2], ["0", "1", "2"]),
            ([0.0, 1.0, 2.0], ["0", "1", "2"]),
            ([0, 1, 2], ["0", "1", "2"]),
            (
                pd.date_range("1/1/2011", periods=2, freq="D"),
                ["2011-01-01", "2011-01-02"],
            ),
            (pd.date_range("1/1/2011", periods=2, freq="D"), [0, 1]),
            (pd.date_range("1/1/2011", periods=2, freq="D"), [0.0, 1.0]),
            (
                pd.date_range("20130101", periods=3),
                pd.date_range("20130101", periods=3, tz="US/Eastern"),
            ),
        ],
    )
    def test_merge_incompat_dtypes_error(self, df1_vals, df2_vals):
        # GH 9780, GH 15800
        # Raise a ValueError when a user tries to merge on
        # dtypes that are incompatible (e.g., obj and int/float)

        df1 = DataFrame({"A": df1_vals})
        df2 = DataFrame({"A": df2_vals})

        msg = (
            f"You are trying to merge on {df1['A'].dtype} and {df2['A'].dtype} "
            "columns for key 'A'. If you wish to proceed you should use pd.concat"
        )
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df1, df2, on=["A"])

        # Check that error still raised when swapping order of dataframes
        msg = (
            f"You are trying to merge on {df2['A'].dtype} and {df1['A'].dtype} "
            "columns for key 'A'. If you wish to proceed you should use pd.concat"
        )
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df2, df1, on=["A"])

        # Check that error still raised when merging on multiple columns
        # The error message should mention the first incompatible column
        if len(df1_vals) == len(df2_vals):
            # Column A in df1 and df2 is of compatible (the same) dtype
            # Columns B and C in df1 and df2 are of incompatible dtypes
            df3 = DataFrame({"A": df2_vals, "B": df1_vals, "C": df1_vals})
            df4 = DataFrame({"A": df2_vals, "B": df2_vals, "C": df2_vals})

            # Check that error raised correctly when merging all columns A, B, and C
            # The error message should mention key 'B'
            msg = (
                f"You are trying to merge on {df3['B'].dtype} and {df4['B'].dtype} "
                "columns for key 'B'. If you wish to proceed you should use pd.concat"
            )
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4)

            # Check that error raised correctly when merging columns A and C
            # The error message should mention key 'C'
            msg = (
                f"You are trying to merge on {df3['C'].dtype} and {df4['C'].dtype} "
                "columns for key 'C'. If you wish to proceed you should use pd.concat"
            )
            msg = re.escape(msg)
            with pytest.raises(ValueError, match=msg):
                merge(df3, df4, on=["A", "C"])

    @pytest.mark.parametrize(
        "expected_data, how",
        [
            ([1, 2], "outer"),
            ([], "inner"),
            ([2], "right"),
            ([1], "left"),
        ],
    )
    def test_merge_EA_dtype(self, any_numeric_ea_dtype, how, expected_data):
        # GH#40073
        d1 = DataFrame([(1,)], columns=["id"], dtype=any_numeric_ea_dtype)
        d2 = DataFrame([(2,)], columns=["id"], dtype=any_numeric_ea_dtype)
        result = merge(d1, d2, how=how)
        exp_index = RangeIndex(len(expected_data))
        expected = DataFrame(
            expected_data, index=exp_index, columns=["id"], dtype=any_numeric_ea_dtype
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "expected_data, how",
        [
            (["a", "b"], "outer"),
            ([], "inner"),
            (["b"], "right"),
            (["a"], "left"),
        ],
    )
    def test_merge_string_dtype(self, how, expected_data, any_string_dtype):
        # GH#40073
        d1 = DataFrame([("a",)], columns=["id"], dtype=any_string_dtype)
        d2 = DataFrame([("b",)], columns=["id"], dtype=any_string_dtype)
        result = merge(d1, d2, how=how)
        exp_idx = RangeIndex(len(expected_data))
        expected = DataFrame(
            expected_data, index=exp_idx, columns=["id"], dtype=any_string_dtype
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "how, expected_data",
        [
            ("inner", [[True, 1, 4], [False, 5, 3]]),
            ("outer", [[False, 5, 3], [True, 1, 4]]),
            ("left", [[True, 1, 4], [False, 5, 3]]),
            ("right", [[False, 5, 3], [True, 1, 4]]),
        ],
    )
    def test_merge_bool_dtype(self, how, expected_data):
        # GH#40073
        df1 = DataFrame({"A": [True, False], "B": [1, 5]})
        df2 = DataFrame({"A": [False, True], "C": [3, 4]})
        result = merge(df1, df2, how=how)
        expected = DataFrame(expected_data, columns=["A", "B", "C"])
        tm.assert_frame_equal(result, expected)

    def test_merge_ea_with_string(self, join_type, string_dtype):
        # GH 43734 Avoid the use of `assign` with multi-index
        df1 = DataFrame(
            data={
                ("lvl0", "lvl1-a"): ["1", "2", "3", "4", None],
                ("lvl0", "lvl1-b"): ["4", "5", "6", "7", "8"],
            },
            dtype=pd.StringDtype(),
        )
        df1_copy = df1.copy()
        df2 = DataFrame(
            data={
                ("lvl0", "lvl1-a"): ["1", "2", "3", pd.NA, "5"],
                ("lvl0", "lvl1-c"): ["7", "8", "9", pd.NA, "11"],
            },
            dtype=string_dtype,
        )
        df2_copy = df2.copy()
        merged = merge(left=df1, right=df2, on=[("lvl0", "lvl1-a")], how=join_type)

        # No change in df1 and df2
        tm.assert_frame_equal(df1, df1_copy)
        tm.assert_frame_equal(df2, df2_copy)

        # Check the expected types for the merged data frame
        expected = Series(
            [np.dtype("O"), pd.StringDtype(), np.dtype("O")],
            index=MultiIndex.from_tuples(
                [("lvl0", "lvl1-a"), ("lvl0", "lvl1-b"), ("lvl0", "lvl1-c")]
            ),
        )
        tm.assert_series_equal(merged.dtypes, expected)

    @pytest.mark.parametrize(
        "left_empty, how, exp",
        [
            (False, "left", "left"),
            (False, "right", "empty"),
            (False, "inner", "empty"),
            (False, "outer", "left"),
            (False, "cross", "empty_cross"),
            (True, "left", "empty"),
            (True, "right", "right"),
            (True, "inner", "empty"),
            (True, "outer", "right"),
            (True, "cross", "empty_cross"),
        ],
    )
    def test_merge_empty(self, left_empty, how, exp):
        left = DataFrame({"A": [2, 1], "B": [3, 4]})
        right = DataFrame({"A": [1], "C": [5]}, dtype="int64")

        if left_empty:
            left = left.head(0)
        else:
            right = right.head(0)

        result = left.merge(right, how=how)

        if exp == "left":
            expected = DataFrame({"A": [2, 1], "B": [3, 4], "C": [np.nan, np.nan]})
        elif exp == "right":
            expected = DataFrame({"A": [1], "B": [np.nan], "C": [5]})
        elif exp == "empty":
            expected = DataFrame(columns=["A", "B", "C"], dtype="int64")
        elif exp == "empty_cross":
            expected = DataFrame(columns=["A_x", "B", "A_y", "C"], dtype="int64")

        if how == "outer":
            expected = expected.sort_values("A", ignore_index=True)

        tm.assert_frame_equal(result, expected)


@pytest.fixture
def left():
    return DataFrame(
        {
            "X": Series(
                np.random.default_rng(2).choice(["foo", "bar"], size=(10,))
            ).astype(CategoricalDtype(["foo", "bar"])),
            "Y": np.random.default_rng(2).choice(["one", "two", "three"], size=(10,)),
        }
    )


@pytest.fixture
def right():
    return DataFrame(
        {
            "X": Series(["foo", "bar"]).astype(CategoricalDtype(["foo", "bar"])),
            "Z": [1, 2],
        }
    )


class TestMergeCategorical:
    def test_identical(self, left, using_infer_string):
        # merging on the same, should preserve dtypes
        merged = merge(left, left, on="X")
        result = merged.dtypes.sort_index()
        dtype = np.dtype("O") if not using_infer_string else "str"
        expected = Series(
            [CategoricalDtype(categories=["foo", "bar"]), dtype, dtype],
            index=["X", "Y_x", "Y_y"],
        )
        tm.assert_series_equal(result, expected)

    def test_basic(self, left, right, using_infer_string):
        # we have matching Categorical dtypes in X
        # so should preserve the merged column
        merged = merge(left, right, on="X")
        result = merged.dtypes.sort_index()
        dtype = np.dtype("O") if not using_infer_string else "str"
        expected = Series(
            [
                CategoricalDtype(categories=["foo", "bar"]),
                dtype,
                np.dtype("int64"),
            ],
            index=["X", "Y", "Z"],
        )
        tm.assert_series_equal(result, expected)

    def test_merge_categorical(self):
        # GH 9426

        right = DataFrame(
            {
                "c": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"},
                "d": {0: "null", 1: "null", 2: "null", 3: "null", 4: "null"},
            }
        )
        left = DataFrame(
            {
                "a": {0: "f", 1: "f", 2: "f", 3: "f", 4: "f"},
                "b": {0: "g", 1: "g", 2: "g", 3: "g", 4: "g"},
            }
        )
        df = merge(left, right, how="left", left_on="b", right_on="c")

        # object-object
        expected = df.copy()

        # object-cat
        # note that we propagate the category
        # because we don't have any matching rows
        cright = right.copy()
        cright["d"] = cright["d"].astype("category")
        result = merge(left, cright, how="left", left_on="b", right_on="c")
        expected["d"] = expected["d"].astype(CategoricalDtype(["null"]))
        tm.assert_frame_equal(result, expected)

        # cat-object
        cleft = left.copy()
        cleft["b"] = cleft["b"].astype("category")
        result = merge(cleft, cright, how="left", left_on="b", right_on="c")
        tm.assert_frame_equal(result, expected)

        # cat-cat
        cright = right.copy()
        cright["d"] = cright["d"].astype("category")
        cleft = left.copy()
        cleft["b"] = cleft["b"].astype("category")
        result = merge(cleft, cright, how="left", left_on="b", right_on="c")
        tm.assert_frame_equal(result, expected)

    def tests_merge_categorical_unordered_equal(self):
        # GH-19551
        df1 = DataFrame(
            {
                "Foo": Categorical(["A", "B", "C"], categories=["A", "B", "C"]),
                "Left": ["A0", "B0", "C0"],
            }
        )

        df2 = DataFrame(
            {
                "Foo": Categorical(["C", "B", "A"], categories=["C", "B", "A"]),
                "Right": ["C1", "B1", "A1"],
            }
        )
        result = merge(df1, df2, on=["Foo"])
        expected = DataFrame(
            {
                "Foo": Categorical(["A", "B", "C"]),
                "Left": ["A0", "B0", "C0"],
                "Right": ["A1", "B1", "C1"],
            }
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_multiindex_merge_with_unordered_categoricalindex(self, ordered):
        # GH 36973
        pcat = CategoricalDtype(categories=["P2", "P1"], ordered=ordered)
        df1 = DataFrame(
            {
                "id": ["C", "C", "D"],
                "p": Categorical(["P2", "P1", "P2"], dtype=pcat),
                "a": [0, 1, 2],
            }
        ).set_index(["id", "p"])
        df2 = DataFrame(
            {
                "id": ["A", "C", "C"],
                "p": Categorical(["P2", "P2", "P1"], dtype=pcat),
                "d1": [10, 11, 12],
            }
        ).set_index(["id", "p"])
        result = merge(df1, df2, how="left", left_index=True, right_index=True)
        expected = DataFrame(
            {
                "id": ["C", "C", "D"],
                "p": Categorical(["P2", "P1", "P2"], dtype=pcat),
                "a": [0, 1, 2],
                "d1": [11.0, 12.0, np.nan],
            }
        ).set_index(["id", "p"])
        tm.assert_frame_equal(result, expected)

    def test_other_columns(self, left, right, using_infer_string):
        # non-merge columns should preserve if possible
        right = right.assign(Z=right.Z.astype("category"))

        merged = merge(left, right, on="X")
        result = merged.dtypes.sort_index()
        dtype = np.dtype("O") if not using_infer_string else "str"
        expected = Series(
            [
                CategoricalDtype(categories=["foo", "bar"]),
                dtype,
                CategoricalDtype(categories=[1, 2]),
            ],
            index=["X", "Y", "Z"],
        )
        tm.assert_series_equal(result, expected)

        # categories are preserved
        assert left.X.values._categories_match_up_to_permutation(merged.X.values)
        assert right.Z.values._categories_match_up_to_permutation(merged.Z.values)

    @pytest.mark.parametrize(
        "change",
        [
            lambda x: x,
            lambda x: x.astype(CategoricalDtype(["foo", "bar", "bah"])),
            lambda x: x.astype(CategoricalDtype(ordered=True)),
        ],
    )
    def test_dtype_on_merged_different(
        self, change, join_type, left, right, using_infer_string
    ):
        # our merging columns, X now has 2 different dtypes
        # so we must be object as a result

        X = change(right.X.astype("object"))
        right = right.assign(X=X)
        assert isinstance(left.X.values.dtype, CategoricalDtype)
        # assert not left.X.values._categories_match_up_to_permutation(right.X.values)

        merged = merge(left, right, on="X", how=join_type)

        result = merged.dtypes.sort_index()
        dtype = np.dtype("O") if not using_infer_string else "str"
        expected = Series([dtype, dtype, np.dtype("int64")], index=["X", "Y", "Z"])
        tm.assert_series_equal(result, expected)

    def test_self_join_multiple_categories(self):
        # GH 16767
        # non-duplicates should work with multiple categories
        m = 5
        df = DataFrame(
            {
                "a": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] * m,
                "b": ["t", "w", "x", "y", "z"] * 2 * m,
                "c": [
                    letter
                    for each in ["m", "n", "u", "p", "o"]
                    for letter in [each] * 2 * m
                ],
                "d": [
                    letter
                    for each in [
                        "aa",
                        "bb",
                        "cc",
                        "dd",
                        "ee",
                        "ff",
                        "gg",
                        "hh",
                        "ii",
                        "jj",
                    ]
                    for letter in [each] * m
                ],
            }
        )

        # change them all to categorical variables
        df = df.apply(lambda x: x.astype("category"))

        # self-join should equal ourselves
        result = merge(df, df, on=list(df.columns))

        tm.assert_frame_equal(result, df)

    def test_dtype_on_categorical_dates(self):
        # GH 16900
        # dates should not be coerced to ints

        df = DataFrame(
            [[date(2001, 1, 1), 1.1], [date(2001, 1, 2), 1.3]], columns=["date", "num2"]
        )
        df["date"] = df["date"].astype("category")

        df2 = DataFrame(
            [[date(2001, 1, 1), 1.3], [date(2001, 1, 3), 1.4]], columns=["date", "num4"]
        )
        df2["date"] = df2["date"].astype("category")

        expected_outer = DataFrame(
            [
                [pd.Timestamp("2001-01-01").date(), 1.1, 1.3],
                [pd.Timestamp("2001-01-02").date(), 1.3, np.nan],
                [pd.Timestamp("2001-01-03").date(), np.nan, 1.4],
            ],
            columns=["date", "num2", "num4"],
        )
        result_outer = merge(df, df2, how="outer", on=["date"])
        tm.assert_frame_equal(result_outer, expected_outer)

        expected_inner = DataFrame(
            [[pd.Timestamp("2001-01-01").date(), 1.1, 1.3]],
            columns=["date", "num2", "num4"],
        )
        result_inner = merge(df, df2, how="inner", on=["date"])
        tm.assert_frame_equal(result_inner, expected_inner)

    @pytest.mark.parametrize("ordered", [True, False])
    @pytest.mark.parametrize(
        "category_column,categories,expected_categories",
        [
            ([False, True, True, False], [True, False], [True, False]),
            ([2, 1, 1, 2], [1, 2], [1, 2]),
            (["False", "True", "True", "False"], ["True", "False"], ["True", "False"]),
        ],
    )
    def test_merging_with_bool_or_int_cateorical_column(
        self, category_column, categories, expected_categories, ordered
    ):
        # GH 17187
        # merging with a boolean/int categorical column
        df1 = DataFrame({"id": [1, 2, 3, 4], "cat": category_column})
        df1["cat"] = df1["cat"].astype(CategoricalDtype(categories, ordered=ordered))
        df2 = DataFrame({"id": [2, 4], "num": [1, 9]})
        result = df1.merge(df2)
        expected = DataFrame({"id": [2, 4], "cat": expected_categories, "num": [1, 9]})
        expected["cat"] = expected["cat"].astype(
            CategoricalDtype(categories, ordered=ordered)
        )
        tm.assert_frame_equal(expected, result)

    def test_merge_on_int_array(self):
        # GH 23020
        df = DataFrame({"A": Series([1, 2, np.nan], dtype="Int64"), "B": 1})
        result = merge(df, df, on="A")
        expected = DataFrame(
            {"A": Series([1, 2, np.nan], dtype="Int64"), "B_x": 1, "B_y": 1}
        )
        tm.assert_frame_equal(result, expected)


@pytest.fixture
def left_df():
    return DataFrame({"a": [20, 10, 0]}, index=[2, 1, 0])


@pytest.fixture
def right_df():
    return DataFrame({"b": [300, 100, 200]}, index=[3, 1, 2])


class TestMergeOnIndexes:
    @pytest.mark.parametrize(
        "how, sort, expected",
        [
            ("inner", False, DataFrame({"a": [20, 10], "b": [200, 100]}, index=[2, 1])),
            ("inner", True, DataFrame({"a": [10, 20], "b": [100, 200]}, index=[1, 2])),
            (
                "left",
                False,
                DataFrame({"a": [20, 10, 0], "b": [200, 100, np.nan]}, index=[2, 1, 0]),
            ),
            (
                "left",
                True,
                DataFrame({"a": [0, 10, 20], "b": [np.nan, 100, 200]}, index=[0, 1, 2]),
            ),
            (
                "right",
                False,
                DataFrame(
                    {"a": [np.nan, 10, 20], "b": [300, 100, 200]}, index=[3, 1, 2]
                ),
            ),
            (
                "right",
                True,
                DataFrame(
                    {"a": [10, 20, np.nan], "b": [100, 200, 300]}, index=[1, 2, 3]
                ),
            ),
            (
                "outer",
                False,
                DataFrame(
                    {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                    index=[0, 1, 2, 3],
                ),
            ),
            (
                "outer",
                True,
                DataFrame(
                    {"a": [0, 10, 20, np.nan], "b": [np.nan, 100, 200, 300]},
                    index=[0, 1, 2, 3],
                ),
            ),
        ],
    )
    def test_merge_on_indexes(self, left_df, right_df, how, sort, expected):
        result = merge(
            left_df, right_df, left_index=True, right_index=True, how=how, sort=sort
        )
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index",
    [Index([1, 2], dtype=dtyp, name="index_col") for dtyp in tm.ALL_REAL_NUMPY_DTYPES]
    + [
        CategoricalIndex(["A", "B"], categories=["A", "B"], name="index_col"),
        RangeIndex(start=0, stop=2, name="index_col"),
        DatetimeIndex(["2018-01-01", "2018-01-02"], name="index_col"),
    ],
    ids=lambda x: f"{type(x).__name__}[{x.dtype}]",
)
def test_merge_index_types(index):
    # gh-20777
    # assert key access is consistent across index types
    left = DataFrame({"left_data": [1, 2]}, index=index)
    right = DataFrame({"right_data": [1.0, 2.0]}, index=index)

    result = left.merge(right, on=["index_col"])

    expected = DataFrame({"left_data": [1, 2], "right_data": [1.0, 2.0]}, index=index)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "on,left_on,right_on,left_index,right_index,nm",
    [
        (["outer", "inner"], None, None, False, False, "B"),
        (None, None, None, True, True, "B"),
        (None, ["outer", "inner"], None, False, True, "B"),
        (None, None, ["outer", "inner"], True, False, "B"),
        (["outer", "inner"], None, None, False, False, None),
        (None, None, None, True, True, None),
        (None, ["outer", "inner"], None, False, True, None),
        (None, None, ["outer", "inner"], True, False, None),
    ],
)
def test_merge_series(on, left_on, right_on, left_index, right_index, nm):
    # GH 21220
    a = DataFrame(
        {"A": [1, 2, 3, 4]},
        index=MultiIndex.from_product([["a", "b"], [0, 1]], names=["outer", "inner"]),
    )
    b = Series(
        [1, 2, 3, 4],
        index=MultiIndex.from_product([["a", "b"], [1, 2]], names=["outer", "inner"]),
        name=nm,
    )
    expected = DataFrame(
        {"A": [2, 4], "B": [1, 3]},
        index=MultiIndex.from_product([["a", "b"], [1]], names=["outer", "inner"]),
    )
    if nm is not None:
        result = merge(
            a,
            b,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
        )
        tm.assert_frame_equal(result, expected)
    else:
        msg = "Cannot merge a Series without a name"
        with pytest.raises(ValueError, match=msg):
            result = merge(
                a,
                b,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
            )


def test_merge_series_multilevel():
    # GH#47946
    # GH 40993: For raising, enforced in 2.0
    a = DataFrame(
        {"A": [1, 2, 3, 4]},
        index=MultiIndex.from_product([["a", "b"], [0, 1]], names=["outer", "inner"]),
    )
    b = Series(
        [1, 2, 3, 4],
        index=MultiIndex.from_product([["a", "b"], [1, 2]], names=["outer", "inner"]),
        name=("B", "C"),
    )
    with pytest.raises(
        MergeError, match="Not allowed to merge between different levels"
    ):
        merge(a, b, on=["outer", "inner"])


@pytest.mark.parametrize(
    "col1, col2, kwargs, expected_cols",
    [
        (0, 0, {"suffixes": ("", "_dup")}, ["0", "0_dup"]),
        (0, 0, {"suffixes": (None, "_dup")}, [0, "0_dup"]),
        (0, 0, {"suffixes": ("_x", "_y")}, ["0_x", "0_y"]),
        (0, 0, {"suffixes": ["_x", "_y"]}, ["0_x", "0_y"]),
        ("a", 0, {"suffixes": (None, "_y")}, ["a", 0]),
        (0.0, 0.0, {"suffixes": ("_x", None)}, ["0.0_x", 0.0]),
        ("b", "b", {"suffixes": (None, "_y")}, ["b", "b_y"]),
        ("a", "a", {"suffixes": ("_x", None)}, ["a_x", "a"]),
        ("a", "b", {"suffixes": ("_x", None)}, ["a", "b"]),
        ("a", "a", {"suffixes": (None, "_x")}, ["a", "a_x"]),
        (0, 0, {"suffixes": ("_a", None)}, ["0_a", 0]),
        ("a", "a", {}, ["a_x", "a_y"]),
        (0, 0, {}, ["0_x", "0_y"]),
    ],
)
def test_merge_suffix(col1, col2, kwargs, expected_cols):
    # issue: 24782
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [4, 5, 6]})

    expected = DataFrame([[1, 4], [2, 5], [3, 6]], columns=expected_cols)

    result = a.merge(b, left_index=True, right_index=True, **kwargs)
    tm.assert_frame_equal(result, expected)

    result = merge(a, b, left_index=True, right_index=True, **kwargs)
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "how,expected",
    [
        (
            "right",
            DataFrame(
                {"A": [100, 200, 300], "B1": [60, 70, np.nan], "B2": [600, 700, 800]}
            ),
        ),
        (
            "outer",
            DataFrame(
                {
                    "A": [1, 100, 200, 300],
                    "B1": [80, 60, 70, np.nan],
                    "B2": [np.nan, 600, 700, 800],
                }
            ),
        ),
    ],
)
def test_merge_duplicate_suffix(how, expected):
    left_df = DataFrame({"A": [100, 200, 1], "B": [60, 70, 80]})
    right_df = DataFrame({"A": [100, 200, 300], "B": [600, 700, 800]})
    result = merge(left_df, right_df, on="A", how=how, suffixes=("_x", "_x"))
    expected.columns = ["A", "B_x", "B_x"]

    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "col1, col2, suffixes",
    [("a", "a", (None, None)), ("a", "a", ("", None)), (0, 0, (None, ""))],
)
def test_merge_suffix_error(col1, col2, suffixes):
    # issue: 24782
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [3, 4, 5]})

    # TODO: might reconsider current raise behaviour, see issue 24782
    msg = "columns overlap but no suffix specified"
    with pytest.raises(ValueError, match=msg):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


@pytest.mark.parametrize("suffixes", [{"left", "right"}, {"left": 0, "right": 0}])
def test_merge_suffix_raises(suffixes):
    a = DataFrame({"a": [1, 2, 3]})
    b = DataFrame({"b": [3, 4, 5]})

    with pytest.raises(TypeError, match="Passing 'suffixes' as a"):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


@pytest.mark.parametrize(
    "col1, col2, suffixes, msg",
    [
        ("a", "a", ("a", "b", "c"), r"too many values to unpack \(expected 2\)"),
        ("a", "a", tuple("a"), r"not enough values to unpack \(expected 2, got 1\)"),
    ],
)
def test_merge_suffix_length_error(col1, col2, suffixes, msg):
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [3, 4, 5]})

    with pytest.raises(ValueError, match=msg):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)


@pytest.mark.parametrize("cat_dtype", ["one", "two"])
@pytest.mark.parametrize("reverse", [True, False])
def test_merge_equal_cat_dtypes(cat_dtype, reverse):
    # see gh-22501
    cat_dtypes = {
        "one": CategoricalDtype(categories=["a", "b", "c"], ordered=False),
        "two": CategoricalDtype(categories=["a", "b", "c"], ordered=False),
    }

    df1 = DataFrame(
        {"foo": Series(["a", "b", "c"]).astype(cat_dtypes["one"]), "left": [1, 2, 3]}
    ).set_index("foo")

    data_foo = ["a", "b", "c"]
    data_right = [1, 2, 3]

    if reverse:
        data_foo.reverse()
        data_right.reverse()

    df2 = DataFrame(
        {"foo": Series(data_foo).astype(cat_dtypes[cat_dtype]), "right": data_right}
    ).set_index("foo")

    result = df1.merge(df2, left_index=True, right_index=True)

    expected = DataFrame(
        {
            "left": [1, 2, 3],
            "right": [1, 2, 3],
            "foo": Series(["a", "b", "c"]).astype(cat_dtypes["one"]),
        }
    ).set_index("foo")

    tm.assert_frame_equal(result, expected)


def test_merge_equal_cat_dtypes2():
    # see gh-22501
    cat_dtype = CategoricalDtype(categories=["a", "b", "c"], ordered=False)

    # Test Data
    df1 = DataFrame(
        {"foo": Series(["a", "b"]).astype(cat_dtype), "left": [1, 2]}
    ).set_index("foo")

    df2 = DataFrame(
        {"foo": Series(["a", "b", "c"]).astype(cat_dtype), "right": [3, 2, 1]}
    ).set_index("foo")

    result = df1.merge(df2, left_index=True, right_index=True)

    expected = DataFrame(
        {"left": [1, 2], "right": [3, 2], "foo": Series(["a", "b"]).astype(cat_dtype)}
    ).set_index("foo")

    tm.assert_frame_equal(result, expected)


def test_merge_on_cat_and_ext_array():
    # GH 28668
    right = DataFrame(
        {"a": Series([pd.Interval(0, 1), pd.Interval(1, 2)], dtype="interval")}
    )
    left = right.copy()
    left["a"] = left["a"].astype("category")

    result = merge(left, right, how="inner", on="a")
    expected = right.copy()

    tm.assert_frame_equal(result, expected)


def test_merge_multiindex_columns():
    # Issue #28518
    # Verify that merging two dataframes give the expected labels
    # The original cause of this issue come from a bug lexsort_depth and is tested in
    # test_lexsort_depth

    letters = ["a", "b", "c", "d"]
    numbers = ["1", "2", "3"]
    index = MultiIndex.from_product((letters, numbers), names=["outer", "inner"])

    frame_x = DataFrame(columns=index)
    frame_x["id"] = ""
    frame_y = DataFrame(columns=index)
    frame_y["id"] = ""

    l_suf = "_x"
    r_suf = "_y"
    result = frame_x.merge(frame_y, on="id", suffixes=((l_suf, r_suf)))

    # Constructing the expected results
    tuples = [(letter + l_suf, num) for letter in letters for num in numbers]
    tuples += [("id", "")]
    tuples += [(letter + r_suf, num) for letter in letters for num in numbers]

    expected_index = MultiIndex.from_tuples(tuples, names=["outer", "inner"])
    expected = DataFrame(columns=expected_index)

    tm.assert_frame_equal(result, expected, check_dtype=False)


def test_merge_datetime_upcast_dtype():
    # https://github.com/pandas-dev/pandas/issues/31208
    df1 = DataFrame({"x": ["a", "b", "c"], "y": ["1", "2", "4"]})
    df2 = DataFrame(
        {"y": ["1", "2", "3"], "z": pd.to_datetime(["2000", "2001", "2002"])}
    )
    result = merge(df1, df2, how="left", on="y")
    expected = DataFrame(
        {
            "x": ["a", "b", "c"],
            "y": ["1", "2", "4"],
            "z": pd.to_datetime(["2000", "2001", "NaT"]),
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("n_categories", [5, 128])
def test_categorical_non_unique_monotonic(n_categories):
    # GH 28189
    # With n_categories as 5, we test the int8 case is hit in libjoin,
    # with n_categories as 128 we test the int16 case.
    left_index = CategoricalIndex([0] + list(range(n_categories)))
    df1 = DataFrame(range(n_categories + 1), columns=["value"], index=left_index)
    df2 = DataFrame(
        [[6]],
        columns=["value"],
        index=CategoricalIndex([0], categories=list(range(n_categories))),
    )

    result = merge(df1, df2, how="left", left_index=True, right_index=True)
    expected = DataFrame(
        [[i, 6.0] if i < 2 else [i, np.nan] for i in range(n_categories + 1)],
        columns=["value_x", "value_y"],
        index=left_index,
    )
    tm.assert_frame_equal(expected, result)


def test_merge_join_categorical_multiindex():
    # From issue 16627
    a = {
        "Cat1": Categorical(["a", "b", "a", "c", "a", "b"], ["a", "b", "c"]),
        "Int1": [0, 1, 0, 1, 0, 0],
    }
    a = DataFrame(a)

    b = {
        "Cat": Categorical(["a", "b", "c", "a", "b", "c"], ["a", "b", "c"]),
        "Int": [0, 0, 0, 1, 1, 1],
        "Factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    }
    b = DataFrame(b).set_index(["Cat", "Int"])["Factor"]

    expected = merge(
        a,
        b.reset_index(),
        left_on=["Cat1", "Int1"],
        right_on=["Cat", "Int"],
        how="left",
    )
    expected = expected.drop(["Cat", "Int"], axis=1)
    result = a.join(b, on=["Cat1", "Int1"])
    tm.assert_frame_equal(expected, result)

    # Same test, but with ordered categorical
    a = {
        "Cat1": Categorical(
            ["a", "b", "a", "c", "a", "b"], ["b", "a", "c"], ordered=True
        ),
        "Int1": [0, 1, 0, 1, 0, 0],
    }
    a = DataFrame(a)

    b = {
        "Cat": Categorical(
            ["a", "b", "c", "a", "b", "c"], ["b", "a", "c"], ordered=True
        ),
        "Int": [0, 0, 0, 1, 1, 1],
        "Factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    }
    b = DataFrame(b).set_index(["Cat", "Int"])["Factor"]

    expected = merge(
        a,
        b.reset_index(),
        left_on=["Cat1", "Int1"],
        right_on=["Cat", "Int"],
        how="left",
    )
    expected = expected.drop(["Cat", "Int"], axis=1)
    result = a.join(b, on=["Cat1", "Int1"])
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    ("kwargs", "err_msg"),
    [
        ({"left_on": "a", "left_index": True}, ["left_on", "left_index"]),
        ({"right_on": "a", "right_index": True}, ["right_on", "right_index"]),
    ],
)
def test_merge_join_cols_error_reporting_duplicates(func, kwargs, err_msg):
    # GH: 16228
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    msg = rf'Can only pass argument "{err_msg[0]}" OR "{err_msg[1]}" not both\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, **kwargs)


@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    ("kwargs", "err_msg"),
    [
        ({"left_on": "a"}, ["right_on", "right_index"]),
        ({"right_on": "a"}, ["left_on", "left_index"]),
    ],
)
def test_merge_join_cols_error_reporting_missing(func, kwargs, err_msg):
    # GH: 16228
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    msg = rf'Must pass "{err_msg[0]}" OR "{err_msg[1]}"\.'
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, **kwargs)


@pytest.mark.parametrize("func", ["merge", "merge_asof"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"right_index": True},
        {"left_index": True},
    ],
)
def test_merge_join_cols_error_reporting_on_and_index(func, kwargs):
    # GH: 16228
    left = DataFrame({"a": [1, 2], "b": [3, 4]})
    right = DataFrame({"a": [1, 1], "c": [5, 6]})
    msg = (
        r'Can only pass argument "on" OR "left_index" '
        r'and "right_index", not a combination of both\.'
    )
    with pytest.raises(MergeError, match=msg):
        getattr(pd, func)(left, right, on="a", **kwargs)


def test_merge_right_left_index():
    # GH#38616
    left = DataFrame({"x": [1, 1], "z": ["foo", "foo"]})
    right = DataFrame({"x": [1, 1], "z": ["foo", "foo"]})
    result = merge(left, right, how="right", left_index=True, right_on="x")
    expected = DataFrame(
        {
            "x": [1, 1],
            "x_x": [1, 1],
            "z_x": ["foo", "foo"],
            "x_y": [1, 1],
            "z_y": ["foo", "foo"],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_merge_result_empty_index_and_on():
    # GH#33814
    df1 = DataFrame({"a": [1], "b": [2]}).set_index(["a", "b"])
    df2 = DataFrame({"b": [1]}).set_index(["b"])
    expected = DataFrame({"a": [], "b": []}, dtype=np.int64).set_index(["a", "b"])
    result = merge(df1, df2, left_on=["b"], right_index=True)
    tm.assert_frame_equal(result, expected)

    result = merge(df2, df1, left_index=True, right_on=["b"])
    tm.assert_frame_equal(result, expected)


def test_merge_suffixes_produce_dup_columns_raises():
    # GH#22818; Enforced in 2.0
    left = DataFrame({"a": [1, 2, 3], "b": 1, "b_x": 2})
    right = DataFrame({"a": [1, 2, 3], "b": 2})

    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left, right, on="a")

    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(right, left, on="a", suffixes=("_y", "_x"))


def test_merge_duplicate_columns_with_suffix_no_warning():
    # GH#22818
    # Do not raise warning when duplicates are caused by duplicates in origin
    left = DataFrame([[1, 1, 1], [2, 2, 2]], columns=["a", "b", "b"])
    right = DataFrame({"a": [1, 3], "b": 2})
    result = merge(left, right, on="a")
    expected = DataFrame([[1, 1, 1, 2]], columns=["a", "b_x", "b_x", "b_y"])
    tm.assert_frame_equal(result, expected)


def test_merge_duplicate_columns_with_suffix_causing_another_duplicate_raises():
    # GH#22818, Enforced in 2.0
    # This should raise warning because suffixes cause another collision
    left = DataFrame([[1, 1, 1, 1], [2, 2, 2, 2]], columns=["a", "b", "b", "b_x"])
    right = DataFrame({"a": [1, 3], "b": 2})
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left, right, on="a")


def test_merge_string_float_column_result():
    # GH 13353
    df1 = DataFrame([[1, 2], [3, 4]], columns=Index(["a", 114.0]))
    df2 = DataFrame([[9, 10], [11, 12]], columns=["x", "y"])
    result = merge(df2, df1, how="inner", left_index=True, right_index=True)
    expected = DataFrame(
        [[9, 10, 1, 2], [11, 12, 3, 4]], columns=Index(["x", "y", "a", 114.0])
    )
    tm.assert_frame_equal(result, expected)


def test_mergeerror_on_left_index_mismatched_dtypes():
    # GH 22449
    df_1 = DataFrame(data=["X"], columns=["C"], index=[22])
    df_2 = DataFrame(data=["X"], columns=["C"], index=[999])
    with pytest.raises(MergeError, match="Can only pass argument"):
        merge(df_1, df_2, on=["C"], left_index=True)


def test_merge_on_left_categoricalindex():
    # GH#48464 don't raise when left_on is a CategoricalIndex
    ci = CategoricalIndex(range(3))

    right = DataFrame({"A": ci, "B": range(3)})
    left = DataFrame({"C": range(3, 6)})

    res = merge(left, right, left_on=ci, right_on="A")
    expected = merge(left, right, left_on=ci._data, right_on="A")
    tm.assert_frame_equal(res, expected)


@pytest.mark.parametrize("dtype", [None, "Int64"])
def test_merge_outer_with_NaN(dtype):
    # GH#43550
    left = DataFrame({"key": [1, 2], "col1": [1, 2]}, dtype=dtype)
    right = DataFrame({"key": [np.nan, np.nan], "col2": [3, 4]}, dtype=dtype)
    result = merge(left, right, on="key", how="outer")
    expected = DataFrame(
        {
            "key": [1, 2, np.nan, np.nan],
            "col1": [1, 2, np.nan, np.nan],
            "col2": [np.nan, np.nan, 3, 4],
        },
        dtype=dtype,
    )
    tm.assert_frame_equal(result, expected)

    # switch left and right
    result = merge(right, left, on="key", how="outer")
    expected = DataFrame(
        {
            "key": [1, 2, np.nan, np.nan],
            "col2": [np.nan, np.nan, 3, 4],
            "col1": [1, 2, np.nan, np.nan],
        },
        dtype=dtype,
    )
    tm.assert_frame_equal(result, expected)


def test_merge_different_index_names():
    # GH#45094
    left = DataFrame({"a": [1]}, index=Index([1], name="c"))
    right = DataFrame({"a": [1]}, index=Index([1], name="d"))
    result = merge(left, right, left_on="c", right_on="d")
    expected = DataFrame({"a_x": [1], "a_y": 1})
    tm.assert_frame_equal(result, expected)


def test_merge_ea(any_numeric_ea_dtype, join_type):
    # GH#44240
    left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
    right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype)
    result = left.merge(right, how=join_type)
    expected = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2}, dtype=any_numeric_ea_dtype)
    tm.assert_frame_equal(result, expected)


def test_merge_ea_and_non_ea(any_numeric_ea_dtype, join_type):
    # GH#44240
    left = DataFrame({"a": [1, 2, 3], "b": 1}, dtype=any_numeric_ea_dtype)
    right = DataFrame({"a": [1, 2, 3], "c": 2}, dtype=any_numeric_ea_dtype.lower())
    result = left.merge(right, how=join_type)
    expected = DataFrame(
        {
            "a": Series([1, 2, 3], dtype=any_numeric_ea_dtype),
            "b": Series([1, 1, 1], dtype=any_numeric_ea_dtype),
            "c": Series([2, 2, 2], dtype=any_numeric_ea_dtype.lower()),
        }
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64", "int64[pyarrow]"])
def test_merge_arrow_and_numpy_dtypes(dtype):
    # GH#52406
    pytest.importorskip("pyarrow")
    df = DataFrame({"a": [1, 2]}, dtype=dtype)
    df2 = DataFrame({"a": [1, 2]}, dtype="int64[pyarrow]")
    result = df.merge(df2)
    expected = df.copy()
    tm.assert_frame_equal(result, expected)

    result = df2.merge(df)
    expected = df2.copy()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["inner", "left", "outer", "right"])
@pytest.mark.parametrize("tz", [None, "America/Chicago"])
def test_merge_datetime_different_resolution(tz, how):
    # https://github.com/pandas-dev/pandas/issues/53200
    vals = [
        pd.Timestamp(2023, 5, 12, tz=tz),
        pd.Timestamp(2023, 5, 13, tz=tz),
        pd.Timestamp(2023, 5, 14, tz=tz),
    ]
    df1 = DataFrame({"t": vals[:2], "a": [1.0, 2.0]})
    df1["t"] = df1["t"].dt.as_unit("ns")
    df2 = DataFrame({"t": vals[1:], "b": [1.0, 2.0]})
    df2["t"] = df2["t"].dt.as_unit("s")

    expected = DataFrame({"t": vals, "a": [1.0, 2.0, np.nan], "b": [np.nan, 1.0, 2.0]})
    expected["t"] = expected["t"].dt.as_unit("ns")
    if how == "inner":
        expected = expected.iloc[[1]].reset_index(drop=True)
    elif how == "left":
        expected = expected.iloc[[0, 1]]
    elif how == "right":
        expected = expected.iloc[[1, 2]].reset_index(drop=True)

    result = df1.merge(df2, on="t", how=how)
    tm.assert_frame_equal(result, expected)


def test_merge_multiindex_single_level():
    # GH52331
    df = DataFrame({"col": ["A", "B"]})
    df2 = DataFrame(
        data={"b": [100]},
        index=MultiIndex.from_tuples([("A",), ("C",)], names=["col"]),
    )
    expected = DataFrame({"col": ["A", "B"], "b": [100, np.nan]})

    result = df.merge(df2, left_on=["col"], right_index=True, how="left")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("on_index", [True, False])
@pytest.mark.parametrize("left_unique", [True, False])
@pytest.mark.parametrize("left_monotonic", [True, False])
@pytest.mark.parametrize("right_unique", [True, False])
@pytest.mark.parametrize("right_monotonic", [True, False])
def test_merge_combinations(
    how, sort, on_index, left_unique, left_monotonic, right_unique, right_monotonic
):
    # GH 54611
    left = [2, 3]
    if left_unique:
        left.append(4 if left_monotonic else 1)
    else:
        left.append(3 if left_monotonic else 2)

    right = [2, 3]
    if right_unique:
        right.append(4 if right_monotonic else 1)
    else:
        right.append(3 if right_monotonic else 2)

    left = DataFrame({"key": left})
    right = DataFrame({"key": right})

    if on_index:
        left = left.set_index("key")
        right = right.set_index("key")
        on_kwargs = {"left_index": True, "right_index": True}
    else:
        on_kwargs = {"on": "key"}

    result = merge(left, right, how=how, sort=sort, **on_kwargs)

    if on_index:
        left = left.reset_index()
        right = right.reset_index()

    if how in ["left", "right", "inner"]:
        if how in ["left", "inner"]:
            expected, other, other_unique = left, right, right_unique
        else:
            expected, other, other_unique = right, left, left_unique
        if how == "inner":
            keep_values = set(left["key"].values).intersection(right["key"].values)
            keep_mask = expected["key"].isin(keep_values)
            expected = expected[keep_mask]
        if sort:
            expected = expected.sort_values("key")
        if not other_unique:
            other_value_counts = other["key"].value_counts()
            repeats = other_value_counts.reindex(expected["key"].values, fill_value=1)
            repeats = repeats.astype(np.intp)
            expected = expected["key"].repeat(repeats.values)
            expected = expected.to_frame()
    elif how == "outer":
        left_counts = left["key"].value_counts()
        right_counts = right["key"].value_counts()
        expected_counts = left_counts.mul(right_counts, fill_value=1)
        expected_counts = expected_counts.astype(np.intp)
        expected = expected_counts.index.values.repeat(expected_counts.values)
        expected = DataFrame({"key": expected})
        expected = expected.sort_values("key")

    if on_index:
        expected = expected.set_index("key")
    else:
        expected = expected.reset_index(drop=True)

    tm.assert_frame_equal(result, expected)


def test_merge_ea_int_and_float_numpy():
    # GH#46178
    df1 = DataFrame([1.0, np.nan], dtype=pd.Int64Dtype())
    df2 = DataFrame([1.5])
    expected = DataFrame(columns=[0], dtype="Int64")

    with tm.assert_produces_warning(UserWarning, match="You are merging"):
        result = df1.merge(df2)
    tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(UserWarning, match="You are merging"):
        result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype("float64"))

    df2 = DataFrame([1.0])
    expected = DataFrame([1], columns=[0], dtype="Int64")
    result = df1.merge(df2)
    tm.assert_frame_equal(result, expected)

    result = df2.merge(df1)
    tm.assert_frame_equal(result, expected.astype("float64"))


def test_merge_arrow_string_index(any_string_dtype):
    # GH#54894
    pytest.importorskip("pyarrow")
    left = DataFrame({"a": ["a", "b"]}, dtype=any_string_dtype)
    right = DataFrame({"b": 1}, index=Index(["a", "c"], dtype=any_string_dtype))
    result = left.merge(right, left_on="a", right_index=True, how="left")
    expected = DataFrame(
        {"a": Series(["a", "b"], dtype=any_string_dtype), "b": [1, np.nan]}
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("left_empty", [True, False])
@pytest.mark.parametrize("right_empty", [True, False])
def test_merge_empty_frames_column_order(left_empty, right_empty):
    # GH 51929
    df1 = DataFrame(1, index=[0], columns=["A", "B"])
    df2 = DataFrame(1, index=[0], columns=["A", "C", "D"])

    if left_empty:
        df1 = df1.iloc[:0]
    if right_empty:
        df2 = df2.iloc[:0]

    result = merge(df1, df2, on=["A"], how="outer")
    expected = DataFrame(1, index=[0], columns=["A", "B", "C", "D"])
    if left_empty and right_empty:
        expected = expected.iloc[:0]
    elif left_empty:
        expected["B"] = np.nan
    elif right_empty:
        expected[["C", "D"]] = np.nan
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_merge_datetime_and_timedelta(how):
    left = DataFrame({"key": Series([1, None], dtype="datetime64[ns]")})
    right = DataFrame({"key": Series([1], dtype="timedelta64[ns]")})

    msg = (
        f"You are trying to merge on {left['key'].dtype} and {right['key'].dtype} "
        "columns for key 'key'. If you wish to proceed you should use pd.concat"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        left.merge(right, on="key", how=how)

    msg = (
        f"You are trying to merge on {right['key'].dtype} and {left['key'].dtype} "
        "columns for key 'key'. If you wish to proceed you should use pd.concat"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        right.merge(left, on="key", how=how)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\readline.py ===
# -*- coding: UTF-8 -*-
# this file is needed in site-packages to emulate readline
# necessary for rlcompleter since it relies on the existence
# of a readline module


from pyreadline3.rlmain import Readline

__all__ = [
    "parse_and_bind",
    "get_line_buffer",
    "insert_text",
    "clear_history",
    "read_init_file",
    "read_history_file",
    "write_history_file",
    "get_current_history_length",
    "get_history_length",
    "get_history_item",
    "set_history_length",
    "set_startup_hook",
    "set_pre_input_hook",
    "set_completer",
    "get_completer",
    "get_begidx",
    "get_endidx",
    "set_completer_delims",
    "get_completer_delims",
    "add_history",
    "callback_handler_install",
    "callback_handler_remove",
    "callback_read_char",
    "redisplay",
]  # Some other objects are added below


# create a Readline object to contain the state
rl = Readline()

if rl.disable_readline:

    def dummy(completer=""):
        pass

    for funk in __all__:
        globals()[funk] = dummy
else:

    def GetOutputFile():
        """Return the console object used by readline so that it can be
        used for printing in color."""
        return rl.console

    __all__.append("GetOutputFile")

    import pyreadline3.console as console

    # make these available so this looks like the python readline module
    read_init_file = rl.read_init_file
    parse_and_bind = rl.parse_and_bind
    clear_history = rl.clear_history
    add_history = rl.add_history
    insert_text = rl.insert_text

    write_history_file = rl.write_history_file
    read_history_file = rl.read_history_file

    get_completer_delims = rl.get_completer_delims
    get_current_history_length = rl.get_current_history_length
    get_history_length = rl.get_history_length
    get_history_item = rl.get_history_item
    get_line_buffer = rl.get_line_buffer
    set_completer = rl.set_completer
    get_completer = rl.get_completer
    get_begidx = rl.get_begidx
    get_endidx = rl.get_endidx

    set_completer_delims = rl.set_completer_delims
    set_history_length = rl.set_history_length
    set_pre_input_hook = rl.set_pre_input_hook
    set_startup_hook = rl.set_startup_hook

    callback_handler_install = rl.callback_handler_install
    callback_handler_remove = rl.callback_handler_remove
    callback_read_char = rl.callback_read_char

    redisplay = rl.redisplay

    console.install_readline(rl.readline)

__all__.append("rl")
__doc__ = "Importing this module enables command line editing using pyreadline3 for Widnows systems"
# type: ignore

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\typing.py ===
from __future__ import annotations

import collections.abc as cabc
import typing as t

if t.TYPE_CHECKING:  # pragma: no cover
    from _typeshed.wsgi import WSGIApplication  # noqa: F401
    from werkzeug.datastructures import Headers  # noqa: F401
    from werkzeug.sansio.response import Response  # noqa: F401

# The possible types that are directly convertible or are a Response object.
ResponseValue = t.Union[
    "Response",
    str,
    bytes,
    list[t.Any],
    # Only dict is actually accepted, but Mapping allows for TypedDict.
    t.Mapping[str, t.Any],
    t.Iterator[str],
    t.Iterator[bytes],
    cabc.AsyncIterable[str],  # for Quart, until App is generic.
    cabc.AsyncIterable[bytes],
]

# the possible types for an individual HTTP header
# This should be a Union, but mypy doesn't pass unless it's a TypeVar.
HeaderValue = t.Union[str, list[str], tuple[str, ...]]

# the possible types for HTTP headers
HeadersValue = t.Union[
    "Headers",
    t.Mapping[str, HeaderValue],
    t.Sequence[tuple[str, HeaderValue]],
]

# The possible types returned by a route function.
ResponseReturnValue = t.Union[
    ResponseValue,
    tuple[ResponseValue, HeadersValue],
    tuple[ResponseValue, int],
    tuple[ResponseValue, int, HeadersValue],
    "WSGIApplication",
]

# Allow any subclass of werkzeug.Response, such as the one from Flask,
# as a callback argument. Using werkzeug.Response directly makes a
# callback annotated with flask.Response fail type checking.
ResponseClass = t.TypeVar("ResponseClass", bound="Response")

AppOrBlueprintKey = t.Optional[str]  # The App key is None, whereas blueprints are named
AfterRequestCallable = t.Union[
    t.Callable[[ResponseClass], ResponseClass],
    t.Callable[[ResponseClass], t.Awaitable[ResponseClass]],
]
BeforeFirstRequestCallable = t.Union[
    t.Callable[[], None], t.Callable[[], t.Awaitable[None]]
]
BeforeRequestCallable = t.Union[
    t.Callable[[], t.Optional[ResponseReturnValue]],
    t.Callable[[], t.Awaitable[t.Optional[ResponseReturnValue]]],
]
ShellContextProcessorCallable = t.Callable[[], dict[str, t.Any]]
TeardownCallable = t.Union[
    t.Callable[[t.Optional[BaseException]], None],
    t.Callable[[t.Optional[BaseException]], t.Awaitable[None]],
]
TemplateContextProcessorCallable = t.Union[
    t.Callable[[], dict[str, t.Any]],
    t.Callable[[], t.Awaitable[dict[str, t.Any]]],
]
TemplateFilterCallable = t.Callable[..., t.Any]
TemplateGlobalCallable = t.Callable[..., t.Any]
TemplateTestCallable = t.Callable[..., bool]
URLDefaultCallable = t.Callable[[str, dict[str, t.Any]], None]
URLValuePreprocessorCallable = t.Callable[
    [t.Optional[str], t.Optional[dict[str, t.Any]]], None
]

# This should take Exception, but that either breaks typing the argument
# with a specific exception, or decorating multiple times with different
# exceptions (and using a union type on the argument).
# https://github.com/pallets/flask/issues/4095
# https://github.com/pallets/flask/issues/4295
# https://github.com/pallets/flask/issues/4297
ErrorHandlerCallable = t.Union[
    t.Callable[[t.Any], ResponseReturnValue],
    t.Callable[[t.Any], t.Awaitable[ResponseReturnValue]],
]

RouteCallable = t.Union[
    t.Callable[..., ResponseReturnValue],
    t.Callable[..., t.Awaitable[ResponseReturnValue]],
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\usertools.py ===

def monitor(f, input='print', output='print'):
    """
    Returns a wrapped copy of *f* that monitors evaluation by calling
    *input* with every input (*args*, *kwargs*) passed to *f* and
    *output* with every value returned from *f*. The default action
    (specify using the special string value ``'print'``) is to print
    inputs and outputs to stdout, along with the total evaluation
    count::

        >>> from mpmath import *
        >>> mp.dps = 5; mp.pretty = False
        >>> diff(monitor(exp), 1)   # diff will eval f(x-h) and f(x+h)
        in  0 (mpf('0.99999999906867742538452148'),) {}
        out 0 mpf('2.7182818259274480055282064')
        in  1 (mpf('1.0000000009313225746154785'),) {}
        out 1 mpf('2.7182818309906424675501024')
        mpf('2.7182808')

    To disable either the input or the output handler, you may
    pass *None* as argument.

    Custom input and output handlers may be used e.g. to store
    results for later analysis::

        >>> mp.dps = 15
        >>> input = []
        >>> output = []
        >>> findroot(monitor(sin, input.append, output.append), 3.0)
        mpf('3.1415926535897932')
        >>> len(input)  # Count number of evaluations
        9
        >>> print(input[3]); print(output[3])
        ((mpf('3.1415076583334066'),), {})
        8.49952562843408e-5
        >>> print(input[4]); print(output[4])
        ((mpf('3.1415928201669122'),), {})
        -1.66577118985331e-7

    """
    if not input:
        input = lambda v: None
    elif input == 'print':
        incount = [0]
        def input(value):
            args, kwargs = value
            print("in  %s %r %r" % (incount[0], args, kwargs))
            incount[0] += 1
    if not output:
        output = lambda v: None
    elif output == 'print':
        outcount = [0]
        def output(value):
            print("out %s %r" % (outcount[0], value))
            outcount[0] += 1
    def f_monitored(*args, **kwargs):
        input((args, kwargs))
        v = f(*args, **kwargs)
        output(v)
        return v
    return f_monitored

def timing(f, *args, **kwargs):
    """
    Returns time elapsed for evaluating ``f()``. Optionally arguments
    may be passed to time the execution of ``f(*args, **kwargs)``.

    If the first call is very quick, ``f`` is called
    repeatedly and the best time is returned.
    """
    once = kwargs.get('once')
    if 'once' in kwargs:
        del kwargs['once']
    if args or kwargs:
        if len(args) == 1 and not kwargs:
            arg = args[0]
            g = lambda: f(arg)
        else:
            g = lambda: f(*args, **kwargs)
    else:
        g = f
    from timeit import default_timer as clock
    t1=clock(); v=g(); t2=clock(); t=t2-t1
    if t > 0.05 or once:
        return t
    for i in range(3):
        t1=clock();
        # Evaluate multiple times because the timer function
        # has a significant overhead
        g();g();g();g();g();g();g();g();g();g()
        t2=clock()
        t=min(t,(t2-t1)/10)
    return t

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\errors.py ===
#Autogenerated schema
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    String,
    Bool,
    Sequence,
)
from openpyxl.descriptors.excel import CellRange


class Extension(Serialisable):

    tagname = "extension"

    uri = String(allow_none=True)

    def __init__(self,
                 uri=None,
                ):
        self.uri = uri


class ExtensionList(Serialisable):

    tagname = "extensionList"

    # uses element group EG_ExtensionList
    ext = Sequence(expected_type=Extension)

    __elements__ = ('ext',)

    def __init__(self,
                 ext=(),
                ):
        self.ext = ext


class IgnoredError(Serialisable):

    tagname = "ignoredError"

    sqref = CellRange
    evalError = Bool(allow_none=True)
    twoDigitTextYear = Bool(allow_none=True)
    numberStoredAsText = Bool(allow_none=True)
    formula = Bool(allow_none=True)
    formulaRange = Bool(allow_none=True)
    unlockedFormula = Bool(allow_none=True)
    emptyCellReference = Bool(allow_none=True)
    listDataValidation = Bool(allow_none=True)
    calculatedColumn = Bool(allow_none=True)

    def __init__(self,
                 sqref=None,
                 evalError=False,
                 twoDigitTextYear=False,
                 numberStoredAsText=False,
                 formula=False,
                 formulaRange=False,
                 unlockedFormula=False,
                 emptyCellReference=False,
                 listDataValidation=False,
                 calculatedColumn=False,
                ):
        self.sqref = sqref
        self.evalError = evalError
        self.twoDigitTextYear = twoDigitTextYear
        self.numberStoredAsText = numberStoredAsText
        self.formula = formula
        self.formulaRange = formulaRange
        self.unlockedFormula = unlockedFormula
        self.emptyCellReference = emptyCellReference
        self.listDataValidation = listDataValidation
        self.calculatedColumn = calculatedColumn


class IgnoredErrors(Serialisable):

    tagname = "ignoredErrors"

    ignoredError = Sequence(expected_type=IgnoredError)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('ignoredError', 'extLst')

    def __init__(self,
                 ignoredError=(),
                 extLst=None,
                ):
        self.ignoredError = ignoredError
        self.extLst = extLst


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\ops\__init__.py ===
"""
Arithmetic operations for PandasObjects

This is not a public API.
"""
from __future__ import annotations

from pandas.core.ops.array_ops import (
    arithmetic_op,
    comp_method_OBJECT_ARRAY,
    comparison_op,
    fill_binop,
    get_array_op,
    logical_op,
    maybe_prepare_scalar_for_op,
)
from pandas.core.ops.common import (
    get_op_result_name,
    unpack_zerodim_and_defer,
)
from pandas.core.ops.docstrings import make_flex_doc
from pandas.core.ops.invalid import invalid_comparison
from pandas.core.ops.mask_ops import (
    kleene_and,
    kleene_or,
    kleene_xor,
)
from pandas.core.roperator import (
    radd,
    rand_,
    rdiv,
    rdivmod,
    rfloordiv,
    rmod,
    rmul,
    ror_,
    rpow,
    rsub,
    rtruediv,
    rxor,
)

# -----------------------------------------------------------------------------
# constants
ARITHMETIC_BINOPS: set[str] = {
    "add",
    "sub",
    "mul",
    "pow",
    "mod",
    "floordiv",
    "truediv",
    "divmod",
    "radd",
    "rsub",
    "rmul",
    "rpow",
    "rmod",
    "rfloordiv",
    "rtruediv",
    "rdivmod",
}


__all__ = [
    "ARITHMETIC_BINOPS",
    "arithmetic_op",
    "comparison_op",
    "comp_method_OBJECT_ARRAY",
    "invalid_comparison",
    "fill_binop",
    "kleene_and",
    "kleene_or",
    "kleene_xor",
    "logical_op",
    "make_flex_doc",
    "radd",
    "rand_",
    "rdiv",
    "rdivmod",
    "rfloordiv",
    "rmod",
    "rmul",
    "ror_",
    "rpow",
    "rsub",
    "rtruediv",
    "rxor",
    "unpack_zerodim_and_defer",
    "get_op_result_name",
    "maybe_prepare_scalar_for_op",
    "get_array_op",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\__init__.py ===
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.plotting._matplotlib.boxplot import (
    BoxPlot,
    boxplot,
    boxplot_frame,
    boxplot_frame_groupby,
)
from pandas.plotting._matplotlib.converter import (
    deregister,
    register,
)
from pandas.plotting._matplotlib.core import (
    AreaPlot,
    BarhPlot,
    BarPlot,
    HexBinPlot,
    LinePlot,
    PiePlot,
    ScatterPlot,
)
from pandas.plotting._matplotlib.hist import (
    HistPlot,
    KdePlot,
    hist_frame,
    hist_series,
)
from pandas.plotting._matplotlib.misc import (
    andrews_curves,
    autocorrelation_plot,
    bootstrap_plot,
    lag_plot,
    parallel_coordinates,
    radviz,
    scatter_matrix,
)
from pandas.plotting._matplotlib.tools import table

if TYPE_CHECKING:
    from pandas.plotting._matplotlib.core import MPLPlot

PLOT_CLASSES: dict[str, type[MPLPlot]] = {
    "line": LinePlot,
    "bar": BarPlot,
    "barh": BarhPlot,
    "box": BoxPlot,
    "hist": HistPlot,
    "kde": KdePlot,
    "area": AreaPlot,
    "pie": PiePlot,
    "scatter": ScatterPlot,
    "hexbin": HexBinPlot,
}


def plot(data, kind, **kwargs):
    # Importing pyplot at the top of the file (before the converters are
    # registered) causes problems in matplotlib 2 (converters seem to not
    # work)
    import matplotlib.pyplot as plt

    if kwargs.pop("reuse_plot", False):
        ax = kwargs.get("ax")
        if ax is None and len(plt.get_fignums()) > 0:
            with plt.rc_context():
                ax = plt.gca()
            kwargs["ax"] = getattr(ax, "left_ax", ax)
    plot_obj = PLOT_CLASSES[kind](data, **kwargs)
    plot_obj.generate()
    plot_obj.draw()
    return plot_obj.result


__all__ = [
    "plot",
    "hist_series",
    "hist_frame",
    "boxplot",
    "boxplot_frame",
    "boxplot_frame_groupby",
    "table",
    "andrews_curves",
    "autocorrelation_plot",
    "bootstrap_plot",
    "lag_plot",
    "parallel_coordinates",
    "radviz",
    "scatter_matrix",
    "register",
    "deregister",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_testing\_hypothesis.py ===
"""
Hypothesis data generator helpers.
"""
from datetime import datetime

from hypothesis import strategies as st
from hypothesis.extra.dateutil import timezones as dateutil_timezones
from hypothesis.extra.pytz import timezones as pytz_timezones

from pandas.compat import is_platform_windows

import pandas as pd

from pandas.tseries.offsets import (
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BYearBegin,
    BYearEnd,
    MonthBegin,
    MonthEnd,
    QuarterBegin,
    QuarterEnd,
    YearBegin,
    YearEnd,
)

OPTIONAL_INTS = st.lists(st.one_of(st.integers(), st.none()), max_size=10, min_size=3)

OPTIONAL_FLOATS = st.lists(st.one_of(st.floats(), st.none()), max_size=10, min_size=3)

OPTIONAL_TEXT = st.lists(st.one_of(st.none(), st.text()), max_size=10, min_size=3)

OPTIONAL_DICTS = st.lists(
    st.one_of(st.none(), st.dictionaries(st.text(), st.integers())),
    max_size=10,
    min_size=3,
)

OPTIONAL_LISTS = st.lists(
    st.one_of(st.none(), st.lists(st.text(), max_size=10, min_size=3)),
    max_size=10,
    min_size=3,
)

OPTIONAL_ONE_OF_ALL = st.one_of(
    OPTIONAL_DICTS, OPTIONAL_FLOATS, OPTIONAL_INTS, OPTIONAL_LISTS, OPTIONAL_TEXT
)

if is_platform_windows():
    DATETIME_NO_TZ = st.datetimes(min_value=datetime(1900, 1, 1))
else:
    DATETIME_NO_TZ = st.datetimes()

DATETIME_JAN_1_1900_OPTIONAL_TZ = st.datetimes(
    min_value=pd.Timestamp(
        1900, 1, 1
    ).to_pydatetime(),  # pyright: ignore[reportGeneralTypeIssues]
    max_value=pd.Timestamp(
        1900, 1, 1
    ).to_pydatetime(),  # pyright: ignore[reportGeneralTypeIssues]
    timezones=st.one_of(st.none(), dateutil_timezones(), pytz_timezones()),
)

DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ = st.datetimes(
    min_value=pd.Timestamp.min.to_pydatetime(warn=False),
    max_value=pd.Timestamp.max.to_pydatetime(warn=False),
)

INT_NEG_999_TO_POS_999 = st.integers(-999, 999)

# The strategy for each type is registered in conftest.py, as they don't carry
# enough runtime information (e.g. type hints) to infer how to build them.
YQM_OFFSET = st.one_of(
    *map(
        st.from_type,
        [
            MonthBegin,
            MonthEnd,
            BMonthBegin,
            BMonthEnd,
            QuarterBegin,
            QuarterEnd,
            BQuarterBegin,
            BQuarterEnd,
            YearBegin,
            YearEnd,
            BYearBegin,
            BYearEnd,
        ],
    )
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\bar.py ===
from typing import Optional, Union

from .color import Color
from .console import Console, ConsoleOptions, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import Style

# There are left-aligned characters for 1/8 to 7/8, but
# the right-aligned characters exist only for 1/8 and 4/8.
BEGIN_BLOCK_ELEMENTS = ["█", "█", "█", "▐", "▐", "▐", "▕", "▕"]
END_BLOCK_ELEMENTS = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
FULL_BLOCK = "█"


class Bar(JupyterMixin):
    """Renders a solid block bar.

    Args:
        size (float): Value for the end of the bar.
        begin (float): Begin point (between 0 and size, inclusive).
        end (float): End point (between 0 and size, inclusive).
        width (int, optional): Width of the bar, or ``None`` for maximum width. Defaults to None.
        color (Union[Color, str], optional): Color of the bar. Defaults to "default".
        bgcolor (Union[Color, str], optional): Color of bar background. Defaults to "default".
    """

    def __init__(
        self,
        size: float,
        begin: float,
        end: float,
        *,
        width: Optional[int] = None,
        color: Union[Color, str] = "default",
        bgcolor: Union[Color, str] = "default",
    ):
        self.size = size
        self.begin = max(begin, 0)
        self.end = min(end, size)
        self.width = width
        self.style = Style(color=color, bgcolor=bgcolor)

    def __repr__(self) -> str:
        return f"Bar({self.size}, {self.begin}, {self.end})"

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        width = min(
            self.width if self.width is not None else options.max_width,
            options.max_width,
        )

        if self.begin >= self.end:
            yield Segment(" " * width, self.style)
            yield Segment.line()
            return

        prefix_complete_eights = int(width * 8 * self.begin / self.size)
        prefix_bar_count = prefix_complete_eights // 8
        prefix_eights_count = prefix_complete_eights % 8

        body_complete_eights = int(width * 8 * self.end / self.size)
        body_bar_count = body_complete_eights // 8
        body_eights_count = body_complete_eights % 8

        # When start and end fall into the same cell, we ideally should render
        # a symbol that's "center-aligned", but there is no good symbol in Unicode.
        # In this case, we fall back to right-aligned block symbol for simplicity.

        prefix = " " * prefix_bar_count
        if prefix_eights_count:
            prefix += BEGIN_BLOCK_ELEMENTS[prefix_eights_count]

        body = FULL_BLOCK * body_bar_count
        if body_eights_count:
            body += END_BLOCK_ELEMENTS[body_eights_count]

        suffix = " " * (width - len(body))

        yield Segment(prefix + body[len(prefix) :] + suffix, self.style)
        yield Segment.line()

    def __rich_measure__(
        self, console: Console, options: ConsoleOptions
    ) -> Measurement:
        return (
            Measurement(self.width, self.width)
            if self.width is not None
            else Measurement(4, options.max_width)
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_wrap.py ===
from __future__ import annotations

import re
from typing import Iterable

from ._loop import loop_last
from .cells import cell_len, chop_cells

re_word = re.compile(r"\s*\S+\s*")


def words(text: str) -> Iterable[tuple[int, int, str]]:
    """Yields each word from the text as a tuple
    containing (start_index, end_index, word). A "word" in this context may
    include the actual word and any whitespace to the right.
    """
    position = 0
    word_match = re_word.match(text, position)
    while word_match is not None:
        start, end = word_match.span()
        word = word_match.group(0)
        yield start, end, word
        word_match = re_word.match(text, end)


def divide_line(text: str, width: int, fold: bool = True) -> list[int]:
    """Given a string of text, and a width (measured in cells), return a list
    of cell offsets which the string should be split at in order for it to fit
    within the given width.

    Args:
        text: The text to examine.
        width: The available cell width.
        fold: If True, words longer than `width` will be folded onto a new line.

    Returns:
        A list of indices to break the line at.
    """
    break_positions: list[int] = []  # offsets to insert the breaks at
    append = break_positions.append
    cell_offset = 0
    _cell_len = cell_len

    for start, _end, word in words(text):
        word_length = _cell_len(word.rstrip())
        remaining_space = width - cell_offset
        word_fits_remaining_space = remaining_space >= word_length

        if word_fits_remaining_space:
            # Simplest case - the word fits within the remaining width for this line.
            cell_offset += _cell_len(word)
        else:
            # Not enough space remaining for this word on the current line.
            if word_length > width:
                # The word doesn't fit on any line, so we can't simply
                # place it on the next line...
                if fold:
                    # Fold the word across multiple lines.
                    folded_word = chop_cells(word, width=width)
                    for last, line in loop_last(folded_word):
                        if start:
                            append(start)
                        if last:
                            cell_offset = _cell_len(line)
                        else:
                            start += len(line)
                else:
                    # Folding isn't allowed, so crop the word.
                    if start:
                        append(start)
                    cell_offset = _cell_len(word)
            elif cell_offset and start:
                # The word doesn't fit within the remaining space on the current
                # line, but it *can* fit on to the next (empty) line.
                append(start)
                cell_offset = _cell_len(word)

    return break_positions


if __name__ == "__main__":  # pragma: no cover
    from .console import Console

    console = Console(width=10)
    console.print("12345 abcdefghijklmnopqrstuvwyxzABCDEFGHIJKLMNOPQRSTUVWXYZ 12345")
    print(chop_cells("abcdefghijklmnopqrstuvwxyz", 10))

    console = Console(width=20)
    console.rule()
    console.print("TextualはPythonの高速アプリケーション開発フレームワークです")

    console.rule()
    console.print("アプリケーションは1670万色を使用でき")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\__init__.py ===
#-----------------------------------------------------------------
# pycparser: __init__.py
#
# This package file exports some convenience functions for
# interacting with pycparser
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#-----------------------------------------------------------------
__all__ = ['c_lexer', 'c_parser', 'c_ast']
__version__ = '2.22'

import io
from subprocess import check_output
from .c_parser import CParser


def preprocess_file(filename, cpp_path='cpp', cpp_args=''):
    """ Preprocess a file using cpp.

        filename:
            Name of the file you want to preprocess.

        cpp_path:
        cpp_args:
            Refer to the documentation of parse_file for the meaning of these
            arguments.

        When successful, returns the preprocessed file's contents.
        Errors from cpp will be printed out.
    """
    path_list = [cpp_path]
    if isinstance(cpp_args, list):
        path_list += cpp_args
    elif cpp_args != '':
        path_list += [cpp_args]
    path_list += [filename]

    try:
        # Note the use of universal_newlines to treat all newlines
        # as \n for Python's purpose
        text = check_output(path_list, universal_newlines=True)
    except OSError as e:
        raise RuntimeError("Unable to invoke 'cpp'.  " +
            'Make sure its path was passed correctly\n' +
            ('Original error: %s' % e))

    return text


def parse_file(filename, use_cpp=False, cpp_path='cpp', cpp_args='',
               parser=None, encoding=None):
    """ Parse a C file using pycparser.

        filename:
            Name of the file you want to parse.

        use_cpp:
            Set to True if you want to execute the C pre-processor
            on the file prior to parsing it.

        cpp_path:
            If use_cpp is True, this is the path to 'cpp' on your
            system. If no path is provided, it attempts to just
            execute 'cpp', so it must be in your PATH.

        cpp_args:
            If use_cpp is True, set this to the command line arguments strings
            to cpp. Be careful with quotes - it's best to pass a raw string
            (r'') here. For example:
            r'-I../utils/fake_libc_include'
            If several arguments are required, pass a list of strings.

        encoding:
            Encoding to use for the file to parse

        parser:
            Optional parser object to be used instead of the default CParser

        When successful, an AST is returned. ParseError can be
        thrown if the file doesn't parse successfully.

        Errors from cpp will be printed out.
    """
    if use_cpp:
        text = preprocess_file(filename, cpp_path, cpp_args)
    else:
        with io.open(filename, encoding=encoding) as f:
            text = f.read()

    if parser is None:
        parser = CParser()
    return parser.parse(text, filename)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\csrf\session.py ===
"""
A provided CSRF implementation which puts CSRF data in a session.

This can be used fairly comfortably with many `request.session` type
objects, including the Werkzeug/Flask session store, Django sessions, and
potentially other similar objects which use a dict-like API for storing
session keys.

The basic concept is a randomly generated value is stored in the user's
session, and an hmac-sha1 of it (along with an optional expiration time,
for extra security) is used as the value of the csrf_token. If this token
validates with the hmac of the random value + expiration time, and the
expiration time is not passed, the CSRF validation will pass.
"""

import hmac
import os
from datetime import datetime
from datetime import timedelta
from hashlib import sha1

from ..validators import ValidationError
from .core import CSRF

__all__ = ("SessionCSRF",)


class SessionCSRF(CSRF):
    TIME_FORMAT = "%Y%m%d%H%M%S"

    def setup_form(self, form):
        self.form_meta = form.meta
        return super().setup_form(form)

    def generate_csrf_token(self, csrf_token_field):
        meta = self.form_meta
        if meta.csrf_secret is None:
            raise Exception(
                "must set `csrf_secret` on class Meta for SessionCSRF to work"
            )
        if meta.csrf_context is None:
            raise TypeError("Must provide a session-like object as csrf context")

        session = self.session

        if "csrf" not in session:
            session["csrf"] = sha1(os.urandom(64)).hexdigest()

        if self.time_limit:
            expires = (self.now() + self.time_limit).strftime(self.TIME_FORMAT)
            csrf_build = "{}{}".format(session["csrf"], expires)
        else:
            expires = ""
            csrf_build = session["csrf"]

        hmac_csrf = hmac.new(
            meta.csrf_secret, csrf_build.encode("utf8"), digestmod=sha1
        )
        return f"{expires}##{hmac_csrf.hexdigest()}"

    def validate_csrf_token(self, form, field):
        meta = self.form_meta
        if not field.data or "##" not in field.data:
            raise ValidationError(field.gettext("CSRF token missing."))

        expires, hmac_csrf = field.data.split("##", 1)

        check_val = (self.session["csrf"] + expires).encode("utf8")

        hmac_compare = hmac.new(meta.csrf_secret, check_val, digestmod=sha1)
        if hmac_compare.hexdigest() != hmac_csrf:
            raise ValidationError(field.gettext("CSRF failed."))

        if self.time_limit:
            now_formatted = self.now().strftime(self.TIME_FORMAT)
            if now_formatted > expires:
                raise ValidationError(field.gettext("CSRF token expired."))

    def now(self):
        """
        Get the current time. Used for test mocking/overriding mainly.
        """
        return datetime.now()

    @property
    def time_limit(self):
        return getattr(self.form_meta, "csrf_time_limit", timedelta(minutes=30))

    @property
    def session(self):
        return getattr(
            self.form_meta.csrf_context, "session", self.form_meta.csrf_context
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\visitor.py ===
"""API for traversing the AST nodes. Implemented by the compiler and
meta introspection.
"""

import typing as t

from .nodes import Node

if t.TYPE_CHECKING:
    import typing_extensions as te

    class VisitCallable(te.Protocol):
        def __call__(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any: ...


class NodeVisitor:
    """Walks the abstract syntax tree and call visitor functions for every
    node found.  The visitor functions may return values which will be
    forwarded by the `visit` method.

    Per default the visitor functions for the nodes are ``'visit_'`` +
    class name of the node.  So a `TryFinally` node visit function would
    be `visit_TryFinally`.  This behavior can be changed by overriding
    the `get_visitor` function.  If no visitor function exists for a node
    (return value `None`) the `generic_visit` visitor is used instead.
    """

    def get_visitor(self, node: Node) -> "t.Optional[VisitCallable]":
        """Return the visitor function for this node or `None` if no visitor
        exists for this node.  In that case the generic visit function is
        used instead.
        """
        return getattr(self, f"visit_{type(node).__name__}", None)

    def visit(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Visit a node."""
        f = self.get_visitor(node)

        if f is not None:
            return f(node, *args, **kwargs)

        return self.generic_visit(node, *args, **kwargs)

    def generic_visit(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Called if no explicit visitor function exists for a node."""
        for child_node in node.iter_child_nodes():
            self.visit(child_node, *args, **kwargs)


class NodeTransformer(NodeVisitor):
    """Walks the abstract syntax tree and allows modifications of nodes.

    The `NodeTransformer` will walk the AST and use the return value of the
    visitor functions to replace or remove the old node.  If the return
    value of the visitor function is `None` the node will be removed
    from the previous location otherwise it's replaced with the return
    value.  The return value may be the original node in which case no
    replacement takes place.
    """

    def generic_visit(self, node: Node, *args: t.Any, **kwargs: t.Any) -> Node:
        for field, old_value in node.iter_fields():
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, Node):
                        value = self.visit(value, *args, **kwargs)
                        if value is None:
                            continue
                        elif not isinstance(value, Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, Node):
                new_node = self.visit(old_value, *args, **kwargs)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def visit_list(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.List[Node]:
        """As transformers may return lists in some places this method
        can be used to enforce a list as return value.
        """
        rv = self.visit(node, *args, **kwargs)

        if not isinstance(rv, list):
            return [rv]

        return rv

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\drawing.py ===

# Copyright (c) 2010-2024 openpyxl

import math

from openpyxl.utils.units import pixels_to_EMU


class Drawing:
    """ a drawing object - eg container for shapes or charts
        we assume user specifies dimensions in pixels; units are
        converted to EMU in the drawing part
    """

    count = 0

    def __init__(self):

        self.name = ''
        self.description = ''
        self.coordinates = ((1, 2), (16, 8))
        self.left = 0
        self.top = 0
        self._width = 21 # default in px
        self._height = 192 #default in px
        self.resize_proportional = False
        self.rotation = 0
        self.anchortype = "absolute"
        self.anchorcol = 0 # left cell
        self.anchorrow = 0 # top row


    @property
    def width(self):
        return self._width


    @width.setter
    def width(self, w):
        if self.resize_proportional and w:
            ratio = self._height / self._width
            self._height = round(ratio * w)
        self._width = w


    @property
    def height(self):
        return self._height


    @height.setter
    def height(self, h):
        if self.resize_proportional and h:
            ratio = self._width / self._height
            self._width = round(ratio * h)
        self._height = h


    def set_dimension(self, w=0, h=0):

        xratio = w / self._width
        yratio = h / self._height

        if self.resize_proportional and w and h:
            if (xratio * self._height) < h:
                self._height = math.ceil(xratio * self._height)
                self._width = w
            else:
                self._width = math.ceil(yratio * self._width)
                self._height = h


    @property
    def anchor(self):
        from .spreadsheet_drawing import (
            OneCellAnchor,
            TwoCellAnchor,
            AbsoluteAnchor)
        if self.anchortype == "absolute":
            anchor = AbsoluteAnchor()
            anchor.pos.x = pixels_to_EMU(self.left)
            anchor.pos.y = pixels_to_EMU(self.top)

        elif self.anchortype == "oneCell":
            anchor = OneCellAnchor()
            anchor._from.col = self.anchorcol
            anchor._from.row = self.anchorrow

        anchor.ext.width = pixels_to_EMU(self._width)
        anchor.ext.height = pixels_to_EMU(self._height)

        return anchor

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\inspect.py ===
import logging
from optparse import Values
from typing import Any, Dict, List

from pip._vendor.packaging.markers import default_environment
from pip._vendor.rich import print_json

from pip import __version__
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.utils.compat import stdlib_pkgs
from pip._internal.utils.urls import path_to_url

logger = logging.getLogger(__name__)


class InspectCommand(Command):
    """
    Inspect the content of a Python environment and produce a report in JSON format.
    """

    ignore_require_venv = True
    usage = """
      %prog [options]"""

    def add_options(self) -> None:
        self.cmd_opts.add_option(
            "--local",
            action="store_true",
            default=False,
            help=(
                "If in a virtualenv that has global access, do not list "
                "globally-installed packages."
            ),
        )
        self.cmd_opts.add_option(
            "--user",
            dest="user",
            action="store_true",
            default=False,
            help="Only output packages installed in user-site.",
        )
        self.cmd_opts.add_option(cmdoptions.list_path())
        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        cmdoptions.check_list_path_option(options)
        dists = get_environment(options.path).iter_installed_distributions(
            local_only=options.local,
            user_only=options.user,
            skip=set(stdlib_pkgs),
        )
        output = {
            "version": "1",
            "pip_version": __version__,
            "installed": [self._dist_to_dict(dist) for dist in dists],
            "environment": default_environment(),
            # TODO tags? scheme?
        }
        print_json(data=output)
        return SUCCESS

    def _dist_to_dict(self, dist: BaseDistribution) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "metadata": dist.metadata_dict,
            "metadata_location": dist.info_location,
        }
        # direct_url. Note that we don't have download_info (as in the installation
        # report) since it is not recorded in installed metadata.
        direct_url = dist.direct_url
        if direct_url is not None:
            res["direct_url"] = direct_url.to_dict()
        else:
            # Emulate direct_url for legacy editable installs.
            editable_project_location = dist.editable_project_location
            if editable_project_location is not None:
                res["direct_url"] = {
                    "url": path_to_url(editable_project_location),
                    "dir_info": {
                        "editable": True,
                    },
                }
        # installer
        installer = dist.installer
        if dist.installer:
            res["installer"] = installer
        # requested
        if dist.installed_with_dist_info:
            res["requested"] = dist.requested
        return res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\sqlite\json.py ===
# dialects/sqlite/json.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from ... import types as sqltypes


class JSON(sqltypes.JSON):
    """SQLite JSON type.

    SQLite supports JSON as of version 3.9 through its JSON1_ extension. Note
    that JSON1_ is a
    `loadable extension <https://www.sqlite.org/loadext.html>`_ and as such
    may not be available, or may require run-time loading.

    :class:`_sqlite.JSON` is used automatically whenever the base
    :class:`_types.JSON` datatype is used against a SQLite backend.

    .. seealso::

        :class:`_types.JSON` - main documentation for the generic
        cross-platform JSON datatype.

    The :class:`_sqlite.JSON` type supports persistence of JSON values
    as well as the core index operations provided by :class:`_types.JSON`
    datatype, by adapting the operations to render the ``JSON_EXTRACT``
    function wrapped in the ``JSON_QUOTE`` function at the database level.
    Extracted values are quoted in order to ensure that the results are
    always JSON string values.


    .. versionadded:: 1.3


    .. _JSON1: https://www.sqlite.org/json1.html

    """


# Note: these objects currently match exactly those of MySQL, however since
# these are not generalizable to all JSON implementations, remain separately
# implemented for each dialect.
class _FormatTypeMixin:
    def _format_value(self, value):
        raise NotImplementedError()

    def bind_processor(self, dialect):
        super_proc = self.string_bind_processor(dialect)

        def process(value):
            value = self._format_value(value)
            if super_proc:
                value = super_proc(value)
            return value

        return process

    def literal_processor(self, dialect):
        super_proc = self.string_literal_processor(dialect)

        def process(value):
            value = self._format_value(value)
            if super_proc:
                value = super_proc(value)
            return value

        return process


class JSONIndexType(_FormatTypeMixin, sqltypes.JSON.JSONIndexType):
    def _format_value(self, value):
        if isinstance(value, int):
            value = "$[%s]" % value
        else:
            value = '$."%s"' % value
        return value


class JSONPathType(_FormatTypeMixin, sqltypes.JSON.JSONPathType):
    def _format_value(self, value):
        return "$%s" % (
            "".join(
                [
                    "[%s]" % elem if isinstance(elem, int) else '."%s"' % elem
                    for elem in value
                ]
            )
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\benchmarks\bench_numbers.py ===
from sympy.core.numbers import Integer, Rational, pi, oo
from sympy.core.intfunc import integer_nthroot, igcd
from sympy.core.singleton import S

i3 = Integer(3)
i4 = Integer(4)
r34 = Rational(3, 4)
q45 = Rational(4, 5)


def timeit_Integer_create():
    Integer(2)


def timeit_Integer_int():
    int(i3)


def timeit_neg_one():
    -S.One


def timeit_Integer_neg():
    -i3


def timeit_Integer_abs():
    abs(i3)


def timeit_Integer_sub():
    i3 - i3


def timeit_abs_pi():
    abs(pi)


def timeit_neg_oo():
    -oo


def timeit_Integer_add_i1():
    i3 + 1


def timeit_Integer_add_ij():
    i3 + i4


def timeit_Integer_add_Rational():
    i3 + r34


def timeit_Integer_mul_i4():
    i3*4


def timeit_Integer_mul_ij():
    i3*i4


def timeit_Integer_mul_Rational():
    i3*r34


def timeit_Integer_eq_i3():
    i3 == 3


def timeit_Integer_ed_Rational():
    i3 == r34


def timeit_integer_nthroot():
    integer_nthroot(100, 2)


def timeit_number_igcd_23_17():
    igcd(23, 17)


def timeit_number_igcd_60_3600():
    igcd(60, 3600)


def timeit_Rational_add_r1():
    r34 + 1


def timeit_Rational_add_rq():
    r34 + q45

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\python.py ===
import keyword as kw
import sympy
from .repr import ReprPrinter
from .str import StrPrinter

# A list of classes that should be printed using StrPrinter
STRPRINT = ("Add", "Infinity", "Integer", "Mul", "NegativeInfinity", "Pow")


class PythonPrinter(ReprPrinter, StrPrinter):
    """A printer which converts an expression into its Python interpretation."""

    def __init__(self, settings=None):
        super().__init__(settings)
        self.symbols = []
        self.functions = []

        # Create print methods for classes that should use StrPrinter instead
        # of ReprPrinter.
        for name in STRPRINT:
            f_name = "_print_%s" % name
            f = getattr(StrPrinter, f_name)
            setattr(PythonPrinter, f_name, f)

    def _print_Function(self, expr):
        func = expr.func.__name__
        if not hasattr(sympy, func) and func not in self.functions:
            self.functions.append(func)
        return StrPrinter._print_Function(self, expr)

    # procedure (!) for defining symbols which have be defined in print_python()
    def _print_Symbol(self, expr):
        symbol = self._str(expr)
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        return StrPrinter._print_Symbol(self, expr)

    def _print_module(self, expr):
        raise ValueError('Modules in the expression are unacceptable')


def python(expr, **settings):
    """Return Python interpretation of passed expression
    (can be passed to the exec() function without any modifications)"""

    printer = PythonPrinter(settings)
    exprp = printer.doprint(expr)

    result = ''
    # Returning found symbols and functions
    renamings = {}
    for symbolname in printer.symbols:
        # Remove curly braces from subscripted variables
        if '{' in symbolname:
            newsymbolname = symbolname.replace('{', '').replace('}', '')
            renamings[sympy.Symbol(symbolname)] = newsymbolname
        else:
            newsymbolname = symbolname

        # Escape symbol names that are reserved Python keywords
        if kw.iskeyword(newsymbolname):
            while True:
                newsymbolname += "_"
                if (newsymbolname not in printer.symbols and
                        newsymbolname not in printer.functions):
                    renamings[sympy.Symbol(
                        symbolname)] = sympy.Symbol(newsymbolname)
                    break
        result += newsymbolname + ' = Symbol(\'' + symbolname + '\')\n'

    for functionname in printer.functions:
        newfunctionname = functionname
        # Escape function names that are reserved Python keywords
        if kw.iskeyword(newfunctionname):
            while True:
                newfunctionname += "_"
                if (newfunctionname not in printer.symbols and
                        newfunctionname not in printer.functions):
                    renamings[sympy.Function(
                        functionname)] = sympy.Function(newfunctionname)
                    break
        result += newfunctionname + ' = Function(\'' + functionname + '\')\n'

    if renamings:
        exprp = expr.subs(renamings)
    result += 'e = ' + printer._str(exprp)
    return result


def print_python(expr, **settings):
    """Print output of python() function"""
    print(python(expr, **settings))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\KernelTypeStrArgsEntry.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class KernelTypeStrArgsEntry(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = KernelTypeStrArgsEntry()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsKernelTypeStrArgsEntry(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def KernelTypeStrArgsEntryBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # KernelTypeStrArgsEntry
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # KernelTypeStrArgsEntry
    def KernelTypeStr(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # KernelTypeStrArgsEntry
    def Args(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.ArgTypeAndIndex import ArgTypeAndIndex
            obj = ArgTypeAndIndex()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # KernelTypeStrArgsEntry
    def ArgsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # KernelTypeStrArgsEntry
    def ArgsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def KernelTypeStrArgsEntryStart(builder):
    builder.StartObject(2)

def Start(builder):
    KernelTypeStrArgsEntryStart(builder)

def KernelTypeStrArgsEntryAddKernelTypeStr(builder, kernelTypeStr):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(kernelTypeStr), 0)

def AddKernelTypeStr(builder, kernelTypeStr):
    KernelTypeStrArgsEntryAddKernelTypeStr(builder, kernelTypeStr)

def KernelTypeStrArgsEntryAddArgs(builder, args):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(args), 0)

def AddArgs(builder, args):
    KernelTypeStrArgsEntryAddArgs(builder, args)

def KernelTypeStrArgsEntryStartArgsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartArgsVector(builder, numElems: int) -> int:
    return KernelTypeStrArgsEntryStartArgsVector(builder, numElems)

def KernelTypeStrArgsEntryEnd(builder):
    return builder.EndObject()

def End(builder):
    return KernelTypeStrArgsEntryEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\OpIdKernelTypeStrArgsEntry.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class OpIdKernelTypeStrArgsEntry(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = OpIdKernelTypeStrArgsEntry()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsOpIdKernelTypeStrArgsEntry(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def OpIdKernelTypeStrArgsEntryBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # OpIdKernelTypeStrArgsEntry
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # OpIdKernelTypeStrArgsEntry
    def OpId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # OpIdKernelTypeStrArgsEntry
    def KernelTypeStrArgs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.KernelTypeStrArgsEntry import KernelTypeStrArgsEntry
            obj = KernelTypeStrArgsEntry()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # OpIdKernelTypeStrArgsEntry
    def KernelTypeStrArgsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # OpIdKernelTypeStrArgsEntry
    def KernelTypeStrArgsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def OpIdKernelTypeStrArgsEntryStart(builder):
    builder.StartObject(2)

def Start(builder):
    OpIdKernelTypeStrArgsEntryStart(builder)

def OpIdKernelTypeStrArgsEntryAddOpId(builder, opId):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(opId), 0)

def AddOpId(builder, opId):
    OpIdKernelTypeStrArgsEntryAddOpId(builder, opId)

def OpIdKernelTypeStrArgsEntryAddKernelTypeStrArgs(builder, kernelTypeStrArgs):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(kernelTypeStrArgs), 0)

def AddKernelTypeStrArgs(builder, kernelTypeStrArgs):
    OpIdKernelTypeStrArgsEntryAddKernelTypeStrArgs(builder, kernelTypeStrArgs)

def OpIdKernelTypeStrArgsEntryStartKernelTypeStrArgsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartKernelTypeStrArgsVector(builder, numElems: int) -> int:
    return OpIdKernelTypeStrArgsEntryStartKernelTypeStrArgsVector(builder, numElems)

def OpIdKernelTypeStrArgsEntryEnd(builder):
    return builder.EndObject()

def End(builder):
    return OpIdKernelTypeStrArgsEntryEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\ParameterOptimizerState.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ParameterOptimizerState(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ParameterOptimizerState()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsParameterOptimizerState(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ParameterOptimizerStateBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x44\x54\x43", size_prefixed=size_prefixed)

    # ParameterOptimizerState
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ParameterOptimizerState
    def ParamName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ParameterOptimizerState
    def Momentums(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ParameterOptimizerState
    def MomentumsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ParameterOptimizerState
    def MomentumsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def ParameterOptimizerStateStart(builder):
    builder.StartObject(2)

def Start(builder):
    ParameterOptimizerStateStart(builder)

def ParameterOptimizerStateAddParamName(builder, paramName):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(paramName), 0)

def AddParamName(builder, paramName):
    ParameterOptimizerStateAddParamName(builder, paramName)

def ParameterOptimizerStateAddMomentums(builder, momentums):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(momentums), 0)

def AddMomentums(builder, momentums):
    ParameterOptimizerStateAddMomentums(builder, momentums)

def ParameterOptimizerStateStartMomentumsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartMomentumsVector(builder, numElems: int) -> int:
    return ParameterOptimizerStateStartMomentumsVector(builder, numElems)

def ParameterOptimizerStateEnd(builder):
    return builder.EndObject()

def End(builder):
    return ParameterOptimizerStateEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\RuntimeOptimizationRecordContainerEntry.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class RuntimeOptimizationRecordContainerEntry(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RuntimeOptimizationRecordContainerEntry()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRuntimeOptimizationRecordContainerEntry(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def RuntimeOptimizationRecordContainerEntryBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # RuntimeOptimizationRecordContainerEntry
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RuntimeOptimizationRecordContainerEntry
    def OptimizerName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # RuntimeOptimizationRecordContainerEntry
    def RuntimeOptimizationRecords(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.RuntimeOptimizationRecord import RuntimeOptimizationRecord
            obj = RuntimeOptimizationRecord()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # RuntimeOptimizationRecordContainerEntry
    def RuntimeOptimizationRecordsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # RuntimeOptimizationRecordContainerEntry
    def RuntimeOptimizationRecordsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

def RuntimeOptimizationRecordContainerEntryStart(builder):
    builder.StartObject(2)

def Start(builder):
    RuntimeOptimizationRecordContainerEntryStart(builder)

def RuntimeOptimizationRecordContainerEntryAddOptimizerName(builder, optimizerName):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(optimizerName), 0)

def AddOptimizerName(builder, optimizerName):
    RuntimeOptimizationRecordContainerEntryAddOptimizerName(builder, optimizerName)

def RuntimeOptimizationRecordContainerEntryAddRuntimeOptimizationRecords(builder, runtimeOptimizationRecords):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(runtimeOptimizationRecords), 0)

def AddRuntimeOptimizationRecords(builder, runtimeOptimizationRecords):
    RuntimeOptimizationRecordContainerEntryAddRuntimeOptimizationRecords(builder, runtimeOptimizationRecords)

def RuntimeOptimizationRecordContainerEntryStartRuntimeOptimizationRecordsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartRuntimeOptimizationRecordsVector(builder, numElems: int) -> int:
    return RuntimeOptimizationRecordContainerEntryStartRuntimeOptimizationRecordsVector(builder, numElems)

def RuntimeOptimizationRecordContainerEntryEnd(builder):
    return builder.EndObject()

def End(builder):
    return RuntimeOptimizationRecordContainerEntryEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\requirements.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
from __future__ import annotations

from typing import Any, Iterator

from ._parser import parse_requirement as _parse_requirement
from ._tokenizer import ParserSyntaxError
from .markers import Marker, _normalize_extra_values
from .specifiers import SpecifierSet
from .utils import canonicalize_name


class InvalidRequirement(ValueError):
    """
    An invalid requirement was found, users should refer to PEP 508.
    """


class Requirement:
    """Parse a requirement.

    Parse a given requirement string into its parts, such as name, specifier,
    URL, and extras. Raises InvalidRequirement on a badly-formed requirement
    string.
    """

    # TODO: Can we test whether something is contained within a requirement?
    #       If so how do we do that? Do we need to test against the _name_ of
    #       the thing as well as the version? What about the markers?
    # TODO: Can we normalize the name and extra name?

    def __init__(self, requirement_string: str) -> None:
        try:
            parsed = _parse_requirement(requirement_string)
        except ParserSyntaxError as e:
            raise InvalidRequirement(str(e)) from e

        self.name: str = parsed.name
        self.url: str | None = parsed.url or None
        self.extras: set[str] = set(parsed.extras or [])
        self.specifier: SpecifierSet = SpecifierSet(parsed.specifier)
        self.marker: Marker | None = None
        if parsed.marker is not None:
            self.marker = Marker.__new__(Marker)
            self.marker._markers = _normalize_extra_values(parsed.marker)

    def _iter_parts(self, name: str) -> Iterator[str]:
        yield name

        if self.extras:
            formatted_extras = ",".join(sorted(self.extras))
            yield f"[{formatted_extras}]"

        if self.specifier:
            yield str(self.specifier)

        if self.url:
            yield f"@ {self.url}"
            if self.marker:
                yield " "

        if self.marker:
            yield f"; {self.marker}"

    def __str__(self) -> str:
        return "".join(self._iter_parts(self.name))

    def __repr__(self) -> str:
        return f"<Requirement('{self}')>"

    def __hash__(self) -> int:
        return hash(
            (
                self.__class__.__name__,
                *self._iter_parts(canonicalize_name(self.name)),
            )
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Requirement):
            return NotImplemented

        return (
            canonicalize_name(self.name) == canonicalize_name(other.name)
            and self.extras == other.extras
            and self.specifier == other.specifier
            and self.url == other.url
            and self.marker == other.marker
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\tseries\offsets.py ===
from __future__ import annotations

from pandas._libs.tslibs.offsets import (
    FY5253,
    BaseOffset,
    BDay,
    BMonthBegin,
    BMonthEnd,
    BQuarterBegin,
    BQuarterEnd,
    BusinessDay,
    BusinessHour,
    BusinessMonthBegin,
    BusinessMonthEnd,
    BYearBegin,
    BYearEnd,
    CBMonthBegin,
    CBMonthEnd,
    CDay,
    CustomBusinessDay,
    CustomBusinessHour,
    CustomBusinessMonthBegin,
    CustomBusinessMonthEnd,
    DateOffset,
    Day,
    Easter,
    FY5253Quarter,
    Hour,
    LastWeekOfMonth,
    Micro,
    Milli,
    Minute,
    MonthBegin,
    MonthEnd,
    Nano,
    QuarterBegin,
    QuarterEnd,
    Second,
    SemiMonthBegin,
    SemiMonthEnd,
    Tick,
    Week,
    WeekOfMonth,
    YearBegin,
    YearEnd,
)

__all__ = [
    "Day",
    "BaseOffset",
    "BusinessDay",
    "BusinessMonthBegin",
    "BusinessMonthEnd",
    "BDay",
    "CustomBusinessDay",
    "CustomBusinessMonthBegin",
    "CustomBusinessMonthEnd",
    "CDay",
    "CBMonthEnd",
    "CBMonthBegin",
    "MonthBegin",
    "BMonthBegin",
    "MonthEnd",
    "BMonthEnd",
    "SemiMonthEnd",
    "SemiMonthBegin",
    "BusinessHour",
    "CustomBusinessHour",
    "YearBegin",
    "BYearBegin",
    "YearEnd",
    "BYearEnd",
    "QuarterBegin",
    "BQuarterBegin",
    "QuarterEnd",
    "BQuarterEnd",
    "LastWeekOfMonth",
    "FY5253Quarter",
    "FY5253",
    "Week",
    "WeekOfMonth",
    "Easter",
    "Tick",
    "Hour",
    "Minute",
    "Second",
    "Milli",
    "Micro",
    "Nano",
    "DateOffset",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\requirements.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.
from __future__ import annotations

from typing import Any, Iterator

from ._parser import parse_requirement as _parse_requirement
from ._tokenizer import ParserSyntaxError
from .markers import Marker, _normalize_extra_values
from .specifiers import SpecifierSet
from .utils import canonicalize_name


class InvalidRequirement(ValueError):
    """
    An invalid requirement was found, users should refer to PEP 508.
    """


class Requirement:
    """Parse a requirement.

    Parse a given requirement string into its parts, such as name, specifier,
    URL, and extras. Raises InvalidRequirement on a badly-formed requirement
    string.
    """

    # TODO: Can we test whether something is contained within a requirement?
    #       If so how do we do that? Do we need to test against the _name_ of
    #       the thing as well as the version? What about the markers?
    # TODO: Can we normalize the name and extra name?

    def __init__(self, requirement_string: str) -> None:
        try:
            parsed = _parse_requirement(requirement_string)
        except ParserSyntaxError as e:
            raise InvalidRequirement(str(e)) from e

        self.name: str = parsed.name
        self.url: str | None = parsed.url or None
        self.extras: set[str] = set(parsed.extras or [])
        self.specifier: SpecifierSet = SpecifierSet(parsed.specifier)
        self.marker: Marker | None = None
        if parsed.marker is not None:
            self.marker = Marker.__new__(Marker)
            self.marker._markers = _normalize_extra_values(parsed.marker)

    def _iter_parts(self, name: str) -> Iterator[str]:
        yield name

        if self.extras:
            formatted_extras = ",".join(sorted(self.extras))
            yield f"[{formatted_extras}]"

        if self.specifier:
            yield str(self.specifier)

        if self.url:
            yield f"@ {self.url}"
            if self.marker:
                yield " "

        if self.marker:
            yield f"; {self.marker}"

    def __str__(self) -> str:
        return "".join(self._iter_parts(self.name))

    def __repr__(self) -> str:
        return f"<Requirement('{self}')>"

    def __hash__(self) -> int:
        return hash(
            (
                self.__class__.__name__,
                *self._iter_parts(canonicalize_name(self.name)),
            )
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Requirement):
            return NotImplemented

        return (
            canonicalize_name(self.name) == canonicalize_name(other.name)
            and self.extras == other.extras
            and self.specifier == other.specifier
            and self.url == other.url
            and self.marker == other.marker
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\regexopt.py ===
"""
    pygments.regexopt
    ~~~~~~~~~~~~~~~~~

    An algorithm that generates optimized regexes for matching long lists of
    literal strings.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re
from re import escape
from os.path import commonprefix
from itertools import groupby
from operator import itemgetter

CS_ESCAPE = re.compile(r'[\[\^\\\-\]]')
FIRST_ELEMENT = itemgetter(0)


def make_charset(letters):
    return '[' + CS_ESCAPE.sub(lambda m: '\\' + m.group(), ''.join(letters)) + ']'


def regex_opt_inner(strings, open_paren):
    """Return a regex that matches any string in the sorted list of strings."""
    close_paren = open_paren and ')' or ''
    # print strings, repr(open_paren)
    if not strings:
        # print '-> nothing left'
        return ''
    first = strings[0]
    if len(strings) == 1:
        # print '-> only 1 string'
        return open_paren + escape(first) + close_paren
    if not first:
        # print '-> first string empty'
        return open_paren + regex_opt_inner(strings[1:], '(?:') \
            + '?' + close_paren
    if len(first) == 1:
        # multiple one-char strings? make a charset
        oneletter = []
        rest = []
        for s in strings:
            if len(s) == 1:
                oneletter.append(s)
            else:
                rest.append(s)
        if len(oneletter) > 1:  # do we have more than one oneletter string?
            if rest:
                # print '-> 1-character + rest'
                return open_paren + regex_opt_inner(rest, '') + '|' \
                    + make_charset(oneletter) + close_paren
            # print '-> only 1-character'
            return open_paren + make_charset(oneletter) + close_paren
    prefix = commonprefix(strings)
    if prefix:
        plen = len(prefix)
        # we have a prefix for all strings
        # print '-> prefix:', prefix
        return open_paren + escape(prefix) \
            + regex_opt_inner([s[plen:] for s in strings], '(?:') \
            + close_paren
    # is there a suffix?
    strings_rev = [s[::-1] for s in strings]
    suffix = commonprefix(strings_rev)
    if suffix:
        slen = len(suffix)
        # print '-> suffix:', suffix[::-1]
        return open_paren \
            + regex_opt_inner(sorted(s[:-slen] for s in strings), '(?:') \
            + escape(suffix[::-1]) + close_paren
    # recurse on common 1-string prefixes
    # print '-> last resort'
    return open_paren + \
        '|'.join(regex_opt_inner(list(group[1]), '')
                 for group in groupby(strings, lambda s: s[0] == first[0])) \
        + close_paren


def regex_opt(strings, prefix='', suffix=''):
    """Return a compiled regex that matches any string in the given list.

    The strings to match must be literal strings, not regexes.  They will be
    regex-escaped.

    *prefix* and *suffix* are pre- and appended to the final regex.
    """
    strings = sorted(strings)
    return prefix + regex_opt_inner(strings, '(') + suffix

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\fourier.py ===
from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt


class DFT(MatrixExpr):
    r"""
    Returns a discrete Fourier transform matrix. The matrix is scaled
    with :math:`\frac{1}{\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform.

    Examples
    ========

    >>> from sympy.abc import n
    >>> from sympy.matrices.expressions.fourier import DFT
    >>> DFT(3)
    DFT(3)
    >>> DFT(3).as_explicit()
    Matrix([
    [sqrt(3)/3,                sqrt(3)/3,                sqrt(3)/3],
    [sqrt(3)/3, sqrt(3)*exp(-2*I*pi/3)/3,  sqrt(3)*exp(2*I*pi/3)/3],
    [sqrt(3)/3,  sqrt(3)*exp(2*I*pi/3)/3, sqrt(3)*exp(-2*I*pi/3)/3]])
    >>> DFT(n).shape
    (n, n)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/DFT_matrix

    """

    def __new__(cls, n):
        n = _sympify(n)
        cls._check_dim(n)

        obj = super().__new__(cls, n)
        return obj

    n = property(lambda self: self.args[0])  # type: ignore
    shape = property(lambda self: (self.n, self.n))  # type: ignore

    def _entry(self, i, j, **kwargs):
        w = exp(-2*S.Pi*I/self.n)
        return w**(i*j) / sqrt(self.n)

    def _eval_inverse(self):
        return IDFT(self.n)


class IDFT(DFT):
    r"""
    Returns an inverse discrete Fourier transform matrix. The matrix is scaled
    with :math:`\frac{1}{\sqrt{n}}` so that it is unitary.

    Parameters
    ==========

    n : integer or Symbol
        Size of the transform

    Examples
    ========

    >>> from sympy.matrices.expressions.fourier import DFT, IDFT
    >>> IDFT(3)
    IDFT(3)
    >>> IDFT(4)*DFT(4)
    I

    See Also
    ========

    DFT

    """
    def _entry(self, i, j, **kwargs):
        w = exp(-2*S.Pi*I/self.n)
        return w**(-i*j) / sqrt(self.n)

    def _eval_inverse(self):
        return DFT(self.n)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\latex\_build_latex_antlr.py ===
import os
import subprocess
import glob

from sympy.utilities.misc import debug

here = os.path.dirname(__file__)
grammar_file = os.path.abspath(os.path.join(here, "LaTeX.g4"))
dir_latex_antlr = os.path.join(here, "_antlr")

header = '''\
# *** GENERATED BY `setup.py antlr`, DO NOT EDIT BY HAND ***
#
# Generated from ../LaTeX.g4, derived from latex2sympy
#     latex2sympy is licensed under the MIT license
#     https://github.com/augustt198/latex2sympy/blob/master/LICENSE.txt
#
# Generated with antlr4
#    antlr4 is licensed under the BSD-3-Clause License
#    https://github.com/antlr/antlr4/blob/master/LICENSE.txt
'''


def check_antlr_version():
    debug("Checking antlr4 version...")

    try:
        debug(subprocess.check_output(["antlr4"])
              .decode('utf-8').split("\n")[0])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        debug("The 'antlr4' command line tool is not installed, "
              "or not on your PATH.\n"
              "> Please refer to the README.md file for more information.")
        return False


def build_parser(output_dir=dir_latex_antlr):
    check_antlr_version()

    debug("Updating ANTLR-generated code in {}".format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "__init__.py"), "w+") as fp:
        fp.write(header)

    args = [
        "antlr4",
        grammar_file,
        "-o", output_dir,
        # for now, not generating these as latex2sympy did not use them
        "-no-visitor",
        "-no-listener",
    ]

    debug("Running code generation...\n\t$ {}".format(" ".join(args)))
    subprocess.check_output(args, cwd=output_dir)

    debug("Applying headers, removing unnecessary files and renaming...")
    # Handle case insensitive file systems. If the files are already
    # generated, they will be written to latex* but LaTeX*.* won't match them.
    for path in (glob.glob(os.path.join(output_dir, "LaTeX*.*")) or
        glob.glob(os.path.join(output_dir, "latex*.*"))):

        # Remove files ending in .interp or .tokens as they are not needed.
        if not path.endswith(".py"):
            os.unlink(path)
            continue

        new_path = os.path.join(output_dir, os.path.basename(path).lower())
        with open(path, 'r') as f:
            lines = [line.rstrip() + '\n' for line in f]

        os.unlink(path)

        with open(new_path, "w") as out_file:
            offset = 0
            while lines[offset].startswith('#'):
                offset += 1
            out_file.write(header)
            out_file.writelines(lines[offset:])

        debug("\t{}".format(new_path))

    return True


if __name__ == "__main__":
    build_parser()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\utils.py ===
import os
import re

_LEADING_SYMBOL = "#" if os.name == "nt" else "-"

# https://docs.python.org/3/library/datetime.html#technical-detail (see NOTE #9)
_DATETIME_STRIP_ZERO_PADDING_FORMATS_RE = re.compile(
    f"%{_LEADING_SYMBOL}["
    "d"  # day of month
    "m"  # month
    "H"  # hour (24-hour)
    "I"  # hour (12-hour)
    "M"  # minutes
    "S"  # seconds
    "U"  # week of year (Sunday first day of week)
    "W"  # week of year (Monday first day of week)
    "V"  # week of year (ISO 8601)
    "]",
    re.MULTILINE,
)


def clean_datetime_format_for_strptime(formats):
    """
    Remove dashes used to disable zero-padding with strftime formats (for
    compatibility with strptime).
    """
    return [
        re.sub(
            _DATETIME_STRIP_ZERO_PADDING_FORMATS_RE,
            lambda m: m[0].replace(_LEADING_SYMBOL, ""),
            format,
        )
        for format in formats
    ]


class UnsetValue:
    """
    An unset value.

    This is used in situations where a blank value like `None` is acceptable
    usually as the default value of a class variable or function parameter
    (iow, usually when `None` is a valid value.)
    """

    def __str__(self):
        return "<unset value>"

    def __repr__(self):
        return "<unset value>"

    def __bool__(self):
        return False

    def __nonzero__(self):
        return False


unset_value = UnsetValue()


class WebobInputWrapper:
    """
    Wrap a webob MultiDict for use as passing as `formdata` to Field.

    Since for consistency, we have decided in WTForms to support as input a
    small subset of the API provided in common between cgi.FieldStorage,
    Django's QueryDict, and Werkzeug's MultiDict, we need to wrap Webob, the
    only supported framework whose multidict does not fit this API, but is
    nevertheless used by a lot of frameworks.

    While we could write a full wrapper to support all the methods, this will
    undoubtedly result in bugs due to some subtle differences between the
    various wrappers. So we will keep it simple.
    """

    def __init__(self, multidict):
        self._wrapped = multidict

    def __iter__(self):
        return iter(self._wrapped)

    def __len__(self):
        return len(self._wrapped)

    def __contains__(self, name):
        return name in self._wrapped

    def getlist(self, name):
        return self._wrapped.getall(name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\CalTableFlatBuffers\TrtTable.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: CalTableFlatBuffers

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class TrtTable:
    __slots__ = ["_tab"]

    @classmethod
    def GetRootAs(cls, buf, offset=0):  # noqa: N802
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = TrtTable()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsTrtTable(cls, buf, offset=0):  # noqa: N802
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)

    # TrtTable
    def Init(self, buf, pos):  # noqa: N802
        self._tab = flatbuffers.table.Table(buf, pos)

    # TrtTable
    def Dict(self, j):  # noqa: N802
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from onnxruntime.quantization.CalTableFlatBuffers.KeyValue import KeyValue

            obj = KeyValue()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # TrtTable
    def DictLength(self):  # noqa: N802
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # TrtTable
    def DictIsNone(self):  # noqa: N802
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0


def Start(builder):  # noqa: N802
    builder.StartObject(1)


def TrtTableStart(builder):  # noqa: N802
    """This method is deprecated. Please switch to Start."""
    return Start(builder)


def AddDict(builder, dict):  # noqa: N802
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(dict), 0)


def TrtTableAddDict(builder, dict):  # noqa: N802
    """This method is deprecated. Please switch to AddDict."""
    return AddDict(builder, dict)


def StartDictVector(builder, numElems):  # noqa: N802
    return builder.StartVector(4, numElems, 4)


def TrtTableStartDictVector(builder, numElems):  # noqa: N802
    """This method is deprecated. Please switch to Start."""
    return StartDictVector(builder, numElems)


def End(builder):  # noqa: N802
    return builder.EndObject()


def TrtTableEnd(builder):  # noqa: N802
    """This method is deprecated. Please switch to End."""
    return End(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\marker.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Alias,
)

from openpyxl.descriptors.excel import(
    ExtensionList,
    _explicit_none,
)

from openpyxl.descriptors.nested import (
    NestedBool,
    NestedInteger,
    NestedMinMax,
    NestedNoneSet,
)

from .layout import Layout
from .picture import PictureOptions
from .shapes import *
from .text import *
from .error_bar import *


class Marker(Serialisable):

    tagname = "marker"

    symbol = NestedNoneSet(values=(['circle', 'dash', 'diamond', 'dot', 'picture',
                              'plus', 'square', 'star', 'triangle', 'x', 'auto']),
                           to_tree=_explicit_none)
    size = NestedMinMax(min=2, max=72, allow_none=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias('spPr')
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('symbol', 'size', 'spPr')

    def __init__(self,
                 symbol=None,
                 size=None,
                 spPr=None,
                 extLst=None,
                ):
        self.symbol = symbol
        self.size = size
        if spPr is None:
            spPr = GraphicalProperties()
        self.spPr = spPr


class DataPoint(Serialisable):

    tagname = "dPt"

    idx = NestedInteger()
    invertIfNegative = NestedBool(allow_none=True)
    marker = Typed(expected_type=Marker, allow_none=True)
    bubble3D = NestedBool(allow_none=True)
    explosion = NestedInteger(allow_none=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias('spPr')
    pictureOptions = Typed(expected_type=PictureOptions, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('idx', 'invertIfNegative', 'marker', 'bubble3D',
                    'explosion', 'spPr', 'pictureOptions')

    def __init__(self,
                 idx=None,
                 invertIfNegative=None,
                 marker=None,
                 bubble3D=None,
                 explosion=None,
                 spPr=None,
                 pictureOptions=None,
                 extLst=None,
                ):
        self.idx = idx
        self.invertIfNegative = invertIfNegative
        self.marker = marker
        self.bubble3D = bubble3D
        self.explosion = explosion
        if spPr is None:
            spPr = GraphicalProperties()
        self.spPr = spPr
        self.pictureOptions = pictureOptions

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\array_algos\masked_accumulations.py ===
"""
masked_accumulations.py is for accumulation algorithms using a mask-based approach
for missing values.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
)

import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt


def _cum_func(
    func: Callable,
    values: np.ndarray,
    mask: npt.NDArray[np.bool_],
    *,
    skipna: bool = True,
):
    """
    Accumulations for 1D masked array.

    We will modify values in place to replace NAs with the appropriate fill value.

    Parameters
    ----------
    func : np.cumsum, np.cumprod, np.maximum.accumulate, np.minimum.accumulate
    values : np.ndarray
        Numpy array with the values (can be of any dtype that support the
        operation).
    mask : np.ndarray
        Boolean numpy array (True values indicate missing values).
    skipna : bool, default True
        Whether to skip NA.
    """
    dtype_info: np.iinfo | np.finfo
    if values.dtype.kind == "f":
        dtype_info = np.finfo(values.dtype.type)
    elif values.dtype.kind in "iu":
        dtype_info = np.iinfo(values.dtype.type)
    elif values.dtype.kind == "b":
        # Max value of bool is 1, but since we are setting into a boolean
        # array, 255 is fine as well. Min value has to be 0 when setting
        # into the boolean array.
        dtype_info = np.iinfo(np.uint8)
    else:
        raise NotImplementedError(
            f"No masked accumulation defined for dtype {values.dtype.type}"
        )
    try:
        fill_value = {
            np.cumprod: 1,
            np.maximum.accumulate: dtype_info.min,
            np.cumsum: 0,
            np.minimum.accumulate: dtype_info.max,
        }[func]
    except KeyError:
        raise NotImplementedError(
            f"No accumulation for {func} implemented on BaseMaskedArray"
        )

    values[mask] = fill_value

    if not skipna:
        mask = np.maximum.accumulate(mask)

    values = func(values)
    return values, mask


def cumsum(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True):
    return _cum_func(np.cumsum, values, mask, skipna=skipna)


def cumprod(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True):
    return _cum_func(np.cumprod, values, mask, skipna=skipna)


def cummin(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True):
    return _cum_func(np.minimum.accumulate, values, mask, skipna=skipna)


def cummax(values: np.ndarray, mask: npt.NDArray[np.bool_], *, skipna: bool = True):
    return _cum_func(np.maximum.accumulate, values, mask, skipna=skipna)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\common\__init__.py ===
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

from .exceptions import (
    DetachedShadowRootException,
    ElementClickInterceptedException,
    ElementNotInteractableException,
    ElementNotSelectableException,
    ElementNotVisibleException,
    ImeActivationFailedException,
    ImeNotAvailableException,
    InsecureCertificateException,
    InvalidArgumentException,
    InvalidCookieDomainException,
    InvalidCoordinatesException,
    InvalidElementStateException,
    InvalidSelectorException,
    InvalidSessionIdException,
    InvalidSwitchToTargetException,
    JavascriptException,
    MoveTargetOutOfBoundsException,
    NoAlertPresentException,
    NoSuchAttributeException,
    NoSuchCookieException,
    NoSuchDriverException,
    NoSuchElementException,
    NoSuchFrameException,
    NoSuchShadowRootException,
    NoSuchWindowException,
    ScreenshotException,
    SessionNotCreatedException,
    StaleElementReferenceException,
    TimeoutException,
    UnableToSetCookieException,
    UnexpectedAlertPresentException,
    UnexpectedTagNameException,
    UnknownMethodException,
    WebDriverException,
)

__all__ = [
    "DetachedShadowRootException",
    "ElementClickInterceptedException",
    "ElementNotInteractableException",
    "ElementNotSelectableException",
    "ElementNotVisibleException",
    "ImeActivationFailedException",
    "ImeNotAvailableException",
    "InsecureCertificateException",
    "InvalidArgumentException",
    "InvalidCookieDomainException",
    "InvalidCoordinatesException",
    "InvalidElementStateException",
    "InvalidSelectorException",
    "InvalidSessionIdException",
    "InvalidSwitchToTargetException",
    "JavascriptException",
    "MoveTargetOutOfBoundsException",
    "NoAlertPresentException",
    "NoSuchAttributeException",
    "NoSuchCookieException",
    "NoSuchDriverException",
    "NoSuchElementException",
    "NoSuchFrameException",
    "NoSuchShadowRootException",
    "NoSuchWindowException",
    "ScreenshotException",
    "SessionNotCreatedException",
    "StaleElementReferenceException",
    "TimeoutException",
    "UnableToSetCookieException",
    "UnexpectedAlertPresentException",
    "UnexpectedTagNameException",
    "UnknownMethodException",
    "WebDriverException",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\keys.py ===
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
"""The Keys implementation."""


class Keys:
    """Set of special keys codes."""

    NULL = "\ue000"
    CANCEL = "\ue001"  # ^break
    HELP = "\ue002"
    BACKSPACE = "\ue003"
    BACK_SPACE = BACKSPACE
    TAB = "\ue004"
    CLEAR = "\ue005"
    RETURN = "\ue006"
    ENTER = "\ue007"
    SHIFT = "\ue008"
    LEFT_SHIFT = SHIFT
    CONTROL = "\ue009"
    LEFT_CONTROL = CONTROL
    ALT = "\ue00a"
    LEFT_ALT = ALT
    PAUSE = "\ue00b"
    ESCAPE = "\ue00c"
    SPACE = "\ue00d"
    PAGE_UP = "\ue00e"
    PAGE_DOWN = "\ue00f"
    END = "\ue010"
    HOME = "\ue011"
    LEFT = "\ue012"
    ARROW_LEFT = LEFT
    UP = "\ue013"
    ARROW_UP = UP
    RIGHT = "\ue014"
    ARROW_RIGHT = RIGHT
    DOWN = "\ue015"
    ARROW_DOWN = DOWN
    INSERT = "\ue016"
    DELETE = "\ue017"
    SEMICOLON = "\ue018"
    EQUALS = "\ue019"

    NUMPAD0 = "\ue01a"  # number pad keys
    NUMPAD1 = "\ue01b"
    NUMPAD2 = "\ue01c"
    NUMPAD3 = "\ue01d"
    NUMPAD4 = "\ue01e"
    NUMPAD5 = "\ue01f"
    NUMPAD6 = "\ue020"
    NUMPAD7 = "\ue021"
    NUMPAD8 = "\ue022"
    NUMPAD9 = "\ue023"
    MULTIPLY = "\ue024"
    ADD = "\ue025"
    SEPARATOR = "\ue026"
    SUBTRACT = "\ue027"
    DECIMAL = "\ue028"
    DIVIDE = "\ue029"

    F1 = "\ue031"  # function  keys
    F2 = "\ue032"
    F3 = "\ue033"
    F4 = "\ue034"
    F5 = "\ue035"
    F6 = "\ue036"
    F7 = "\ue037"
    F8 = "\ue038"
    F9 = "\ue039"
    F10 = "\ue03a"
    F11 = "\ue03b"
    F12 = "\ue03c"

    META = "\ue03d"
    COMMAND = "\ue03d"
    ZENKAKU_HANKAKU = "\ue040"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\__init__.py ===
__all__ = [
    'vector',

    'CoordinateSym', 'ReferenceFrame', 'Dyadic', 'Vector', 'Point', 'cross',
    'dot', 'express', 'time_derivative', 'outer', 'kinematic_equations',
    'get_motion_params', 'partial_velocity', 'dynamicsymbols', 'vprint',
    'vsstrrepr', 'vsprint', 'vpprint', 'vlatex', 'init_vprinting', 'curl',
    'divergence', 'gradient', 'is_conservative', 'is_solenoidal',
    'scalar_potential', 'scalar_potential_difference',

    'KanesMethod',

    'RigidBody',

    'linear_momentum', 'angular_momentum', 'kinetic_energy', 'potential_energy',
    'Lagrangian', 'mechanics_printing', 'mprint', 'msprint', 'mpprint',
    'mlatex', 'msubs', 'find_dynamicsymbols',

    'inertia', 'inertia_of_point_mass', 'Inertia',

    'Force', 'Torque',

    'Particle',

    'LagrangesMethod',

    'Linearizer',

    'Body',

    'SymbolicSystem', 'System',

    'PinJoint', 'PrismaticJoint', 'CylindricalJoint', 'PlanarJoint',
    'SphericalJoint', 'WeldJoint',

    'JointsMethod',

    'WrappingCylinder', 'WrappingGeometryBase', 'WrappingSphere',

    'PathwayBase', 'LinearPathway', 'ObstacleSetPathway', 'WrappingPathway',

    'ActuatorBase', 'ForceActuator', 'LinearDamper', 'LinearSpring',
    'TorqueActuator', 'DuffingSpring', 'CoulombKineticFriction',
]

from sympy.physics import vector

from sympy.physics.vector import (CoordinateSym, ReferenceFrame, Dyadic, Vector, Point,
        cross, dot, express, time_derivative, outer, kinematic_equations,
        get_motion_params, partial_velocity, dynamicsymbols, vprint,
        vsstrrepr, vsprint, vpprint, vlatex, init_vprinting, curl, divergence,
        gradient, is_conservative, is_solenoidal, scalar_potential,
        scalar_potential_difference)

from .kane import KanesMethod

from .rigidbody import RigidBody

from .functions import (linear_momentum, angular_momentum, kinetic_energy,
                        potential_energy, Lagrangian, mechanics_printing,
                        mprint, msprint, mpprint, mlatex, msubs,
                        find_dynamicsymbols)

from .inertia import inertia, inertia_of_point_mass, Inertia

from .loads import Force, Torque

from .particle import Particle

from .lagrange import LagrangesMethod

from .linearize import Linearizer

from .body import Body

from .system import SymbolicSystem, System

from .jointsmethod import JointsMethod

from .joint import (PinJoint, PrismaticJoint, CylindricalJoint, PlanarJoint,
                    SphericalJoint, WeldJoint)

from .wrapping_geometry import (WrappingCylinder, WrappingGeometryBase,
                                WrappingSphere)

from .pathway import (PathwayBase, LinearPathway, ObstacleSetPathway,
                      WrappingPathway)

from .actuator import (ActuatorBase, ForceActuator, LinearDamper, LinearSpring,
                       TorqueActuator, DuffingSpring, CoulombKineticFriction)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\matrices\eigen.py ===
"""

Routines for computing eigenvectors with DomainMatrix.

"""
from sympy.core.symbol import Dummy

from ..agca.extensions import FiniteExtension
from ..factortools import dup_factor_list
from ..polyroots import roots
from ..polytools import Poly
from ..rootoftools import CRootOf

from .domainmatrix import DomainMatrix


def dom_eigenvects(A, l=Dummy('lambda')):
    charpoly = A.charpoly()
    rows, cols = A.shape
    domain = A.domain
    _, factors = dup_factor_list(charpoly, domain)

    rational_eigenvects = []
    algebraic_eigenvects = []
    for base, exp in factors:
        if len(base) == 2:
            field = domain
            eigenval = -base[1] / base[0]

            EE_items = [
                [eigenval if i == j else field.zero for j in range(cols)]
                for i in range(rows)]
            EE = DomainMatrix(EE_items, (rows, cols), field)

            basis = (A - EE).nullspace(divide_last=True)
            rational_eigenvects.append((field, eigenval, exp, basis))
        else:
            minpoly = Poly.from_list(base, l, domain=domain)
            field = FiniteExtension(minpoly)
            eigenval = field(l)

            AA_items = [
                [Poly.from_list([item], l, domain=domain).rep for item in row]
                for row in A.rep.to_ddm()]
            AA_items = [[field(item) for item in row] for row in AA_items]
            AA = DomainMatrix(AA_items, (rows, cols), field)
            EE_items = [
                [eigenval if i == j else field.zero for j in range(cols)]
                for i in range(rows)]
            EE = DomainMatrix(EE_items, (rows, cols), field)

            basis = (AA - EE).nullspace(divide_last=True)
            algebraic_eigenvects.append((field, minpoly, exp, basis))

    return rational_eigenvects, algebraic_eigenvects


def dom_eigenvects_to_sympy(
    rational_eigenvects, algebraic_eigenvects,
    Matrix, **kwargs
):
    result = []

    for field, eigenvalue, multiplicity, eigenvects in rational_eigenvects:
        eigenvects = eigenvects.rep.to_ddm()
        eigenvalue = field.to_sympy(eigenvalue)
        new_eigenvects = [
            Matrix([field.to_sympy(x) for x in vect])
            for vect in eigenvects]
        result.append((eigenvalue, multiplicity, new_eigenvects))

    for field, minpoly, multiplicity, eigenvects in algebraic_eigenvects:
        eigenvects = eigenvects.rep.to_ddm()
        l = minpoly.gens[0]

        eigenvects = [[field.to_sympy(x) for x in vect] for vect in eigenvects]

        degree = minpoly.degree()
        minpoly = minpoly.as_expr()
        eigenvals = roots(minpoly, l, **kwargs)
        if len(eigenvals) != degree:
            eigenvals = [CRootOf(minpoly, l, idx) for idx in range(degree)]

        for eigenvalue in eigenvals:
            new_eigenvects = [
                Matrix([x.subs(l, eigenvalue) for x in vect])
                for vect in eigenvects]
            result.append((eigenvalue, multiplicity, new_eigenvects))

    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\tests\test_single.py ===
#
# The main tests for the code in single.py are currently located in
# sympy/solvers/tests/test_ode.py
#
r"""
This File contains test functions for the individual hints used for solving ODEs.

Examples of each solver will be returned by _get_examples_ode_sol_name_of_solver.

Examples should have a key 'XFAIL' which stores the list of hints if they are
expected to fail for that hint.

Functions that are for internal use:

1) _ode_solver_test(ode_examples) - It takes a dictionary of examples returned by
   _get_examples method and tests them with their respective hints.

2) _test_particular_example(our_hint, example_name) - It tests the ODE example corresponding
   to the hint provided.

3) _test_all_hints(runxfail=False) - It is used to test all the examples with all the hints
  currently implemented. It calls _test_all_examples_for_one_hint() which outputs whether the
  given hint functions properly if it classifies the ODE example.
  If runxfail flag is set to True then it will only test the examples which are expected to fail.

  Everytime the ODE of a particular solver is added, _test_all_hints() is to be executed to find
  the possible failures of different solver hints.

4) _test_all_examples_for_one_hint(our_hint, all_examples) - It takes hint as argument and checks
   this hint against all the ODE examples and gives output as the number of ODEs matched, number
   of ODEs which were solved correctly, list of ODEs which gives incorrect solution and list of
   ODEs which raises exception.

"""
from sympy.core.function import (Derivative, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (Ei, erfi)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.rootoftools import rootof

from sympy.core import Function, Symbol
from sympy.functions import airyai, airybi, besselj, bessely, lowergamma
from sympy.integrals.risch import NonElementaryIntegral
from sympy.solvers.ode import classify_ode, dsolve
from sympy.solvers.ode.ode import allhints, _remove_redundant_solutions
from sympy.solvers.ode.single import (FirstLinear, ODEMatchError,
    SingleODEProblem, SingleODESolver, NthOrderReducible)

from sympy.solvers.ode.subscheck import checkodesol

from sympy.testing.pytest import raises, slow
import traceback


x = Symbol('x')
u = Symbol('u')
_u = Dummy('u')
y = Symbol('y')
f = Function('f')
g = Function('g')
C1, C2, C3, C4, C5, C6, C7, C8, C9, C10  = symbols('C1:11')
a, b, c = symbols('a b c')


hint_message = """\
Hint did not match the example {example}.

The ODE is:
{eq}.

The expected hint was
{our_hint}\
"""

expected_sol_message = """\
Different solution found from dsolve for example {example}.

The ODE is:
{eq}

The expected solution was
{sol}

What dsolve returned is:
{dsolve_sol}\
"""

checkodesol_msg = """\
solution found is not correct for example {example}.

The ODE is:
{eq}\
"""

dsol_incorrect_msg = """\
solution returned by dsolve is incorrect when using {hint}.

The ODE is:
{eq}

The expected solution was
{sol}

what dsolve returned is:
{dsolve_sol}

You can test this with:

eq = {eq}
sol = dsolve(eq, hint='{hint}')
print(sol)
print(checkodesol(eq, sol))

"""

exception_msg = """\
dsolve raised exception : {e}

when using {hint} for the example {example}

You can test this with:

from sympy.solvers.ode.tests.test_single import _test_an_example

_test_an_example('{hint}', example_name = '{example}')

The ODE is:
{eq}

\
"""

check_hint_msg = """\
Tested hint was : {hint}

Total of {matched} examples matched with this hint.

Out of which {solve} gave correct results.

Examples which gave incorrect results are {unsolve}.

Examples which raised exceptions are {exceptions}
\
"""


def _add_example_keys(func):
    def inner():
        solver=func()
        examples=[]
        for example in solver['examples']:
            temp={
                'eq': solver['examples'][example]['eq'],
                'sol': solver['examples'][example]['sol'],
                'XFAIL': solver['examples'][example].get('XFAIL', []),
                'func': solver['examples'][example].get('func',solver['func']),
                'example_name': example,
                'slow': solver['examples'][example].get('slow', False),
                'simplify_flag':solver['examples'][example].get('simplify_flag',True),
                'checkodesol_XFAIL': solver['examples'][example].get('checkodesol_XFAIL', False),
                'dsolve_too_slow':solver['examples'][example].get('dsolve_too_slow',False),
                'checkodesol_too_slow':solver['examples'][example].get('checkodesol_too_slow',False),
                'hint': solver['hint']
            }
            examples.append(temp)
        return examples
    return inner()


def _ode_solver_test(ode_examples, run_slow_test=False):
    for example in ode_examples:
        if ((not run_slow_test) and example['slow']) or (run_slow_test and (not example['slow'])):
            continue

        result = _test_particular_example(example['hint'], example, solver_flag=True)
        if result['xpass_msg'] != "":
            print(result['xpass_msg'])


def _test_all_hints(runxfail=False):
    all_hints = list(allhints)+["default"]
    all_examples = _get_all_examples()

    for our_hint in all_hints:
        if our_hint.endswith('_Integral') or 'series' in our_hint:
            continue
        _test_all_examples_for_one_hint(our_hint, all_examples, runxfail)


def _test_dummy_sol(expected_sol,dsolve_sol):
    if type(dsolve_sol)==list:
        return any(expected_sol.dummy_eq(sub_dsol) for sub_dsol in dsolve_sol)
    else:
        return expected_sol.dummy_eq(dsolve_sol)


def _test_an_example(our_hint, example_name):
    all_examples = _get_all_examples()
    for example in all_examples:
        if example['example_name'] == example_name:
            _test_particular_example(our_hint, example)


def _test_particular_example(our_hint, ode_example, solver_flag=False):
    eq = ode_example['eq']
    expected_sol = ode_example['sol']
    example = ode_example['example_name']
    xfail = our_hint in ode_example['XFAIL']
    func = ode_example['func']
    result = {'msg': '', 'xpass_msg': ''}
    simplify_flag=ode_example['simplify_flag']
    checkodesol_XFAIL = ode_example['checkodesol_XFAIL']
    dsolve_too_slow = ode_example['dsolve_too_slow']
    checkodesol_too_slow = ode_example['checkodesol_too_slow']
    xpass = True
    if solver_flag:
        if our_hint not in classify_ode(eq, func):
            message = hint_message.format(example=example, eq=eq, our_hint=our_hint)
            raise AssertionError(message)

    if our_hint in classify_ode(eq, func):
        result['match_list'] = example
        try:
            if not (dsolve_too_slow):
                dsolve_sol = dsolve(eq, func, simplify=simplify_flag,hint=our_hint)
            else:
                if len(expected_sol)==1:
                    dsolve_sol = expected_sol[0]
                else:
                    dsolve_sol = expected_sol

        except Exception as e:
            dsolve_sol = []
            result['exception_list'] = example
            if not solver_flag:
                traceback.print_exc()
            result['msg'] = exception_msg.format(e=str(e), hint=our_hint, example=example, eq=eq)
            if solver_flag and not xfail:
                print(result['msg'])
                raise
            xpass = False

        if solver_flag and dsolve_sol!=[]:
            expect_sol_check = False
            if type(dsolve_sol)==list:
                for sub_sol in expected_sol:
                    if sub_sol.has(Dummy):
                        expect_sol_check = not _test_dummy_sol(sub_sol, dsolve_sol)
                    else:
                        expect_sol_check = sub_sol not in dsolve_sol
                    if expect_sol_check:
                        break
            else:
                expect_sol_check = dsolve_sol not in expected_sol
                for sub_sol in expected_sol:
                    if sub_sol.has(Dummy):
                        expect_sol_check = not _test_dummy_sol(sub_sol, dsolve_sol)

            if expect_sol_check:
                message = expected_sol_message.format(example=example, eq=eq, sol=expected_sol, dsolve_sol=dsolve_sol)
                raise AssertionError(message)

            expected_checkodesol = [(True, 0) for i in range(len(expected_sol))]
            if len(expected_sol) == 1:
                expected_checkodesol = (True, 0)

            if not checkodesol_too_slow:
                if not checkodesol_XFAIL:
                    if checkodesol(eq, dsolve_sol, func, solve_for_func=False) != expected_checkodesol:
                        result['unsolve_list'] = example
                        xpass = False
                        message = dsol_incorrect_msg.format(hint=our_hint, eq=eq, sol=expected_sol,dsolve_sol=dsolve_sol)
                        if solver_flag:
                            message = checkodesol_msg.format(example=example, eq=eq)
                            raise AssertionError(message)
                        else:
                            result['msg'] = 'AssertionError: ' + message

        if xpass and xfail:
            result['xpass_msg'] = example + "is now passing for the hint" + our_hint
    return result


def _test_all_examples_for_one_hint(our_hint, all_examples=[], runxfail=None):
    if all_examples == []:
        all_examples = _get_all_examples()
    match_list, unsolve_list, exception_list = [], [], []
    for ode_example in all_examples:
        xfail = our_hint in ode_example['XFAIL']
        if runxfail and not xfail:
            continue
        if xfail:
            continue
        result = _test_particular_example(our_hint, ode_example)
        match_list += result.get('match_list',[])
        unsolve_list += result.get('unsolve_list',[])
        exception_list += result.get('exception_list',[])
        if runxfail is not None:
            msg = result['msg']
            if msg!='':
                print(result['msg'])
            # print(result.get('xpass_msg',''))
    if runxfail is None:
        match_count = len(match_list)
        solved = len(match_list)-len(unsolve_list)-len(exception_list)
        msg = check_hint_msg.format(hint=our_hint, matched=match_count, solve=solved, unsolve=unsolve_list, exceptions=exception_list)
        print(msg)


def test_SingleODESolver():
    # Test that not implemented methods give NotImplementedError
    # Subclasses should override these methods.
    problem = SingleODEProblem(f(x).diff(x), f(x), x)
    solver = SingleODESolver(problem)
    raises(NotImplementedError, lambda: solver.matches())
    raises(NotImplementedError, lambda: solver.get_general_solution())
    raises(NotImplementedError, lambda: solver._matches())
    raises(NotImplementedError, lambda: solver._get_general_solution())

    # This ODE can not be solved by the FirstLinear solver. Here we test that
    # it does not match and the asking for a general solution gives
    # ODEMatchError

    problem = SingleODEProblem(f(x).diff(x) + f(x)*f(x), f(x), x)

    solver = FirstLinear(problem)
    raises(ODEMatchError, lambda: solver.get_general_solution())

    solver = FirstLinear(problem)
    assert solver.matches() is False

    #These are just test for order of ODE

    problem = SingleODEProblem(f(x).diff(x) + f(x), f(x), x)
    assert problem.order == 1

    problem = SingleODEProblem(f(x).diff(x,4) + f(x).diff(x,2) - f(x).diff(x,3), f(x), x)
    assert problem.order == 4

    problem = SingleODEProblem(f(x).diff(x, 3) + f(x).diff(x, 2) - f(x)**2, f(x), x)
    assert problem.is_autonomous == True

    problem = SingleODEProblem(f(x).diff(x, 3) + x*f(x).diff(x, 2) - f(x)**2, f(x), x)
    assert problem.is_autonomous == False


def test_linear_coefficients():
    _ode_solver_test(_get_examples_ode_sol_linear_coefficients)


@slow
def test_1st_homogeneous_coeff_ode():
    #These were marked as test_1st_homogeneous_coeff_corner_case
    eq1 = f(x).diff(x) - f(x)/x
    c1 = classify_ode(eq1, f(x))
    eq2 = x*f(x).diff(x) - f(x)
    c2 = classify_ode(eq2, f(x))
    sdi = "1st_homogeneous_coeff_subs_dep_div_indep"
    sid = "1st_homogeneous_coeff_subs_indep_div_dep"
    assert sid not in c1 and sdi not in c1
    assert sid not in c2 and sdi not in c2
    _ode_solver_test(_get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep)
    _ode_solver_test(_get_examples_ode_sol_1st_homogeneous_coeff_best)


@slow
def test_slow_examples_1st_homogeneous_coeff_ode():
    _ode_solver_test(_get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep, run_slow_test=True)
    _ode_solver_test(_get_examples_ode_sol_1st_homogeneous_coeff_best, run_slow_test=True)


@slow
def test_nth_linear_constant_coeff_homogeneous():
    _ode_solver_test(_get_examples_ode_sol_nth_linear_constant_coeff_homogeneous)


@slow
def test_slow_examples_nth_linear_constant_coeff_homogeneous():
    _ode_solver_test(_get_examples_ode_sol_nth_linear_constant_coeff_homogeneous, run_slow_test=True)


def test_Airy_equation():
    _ode_solver_test(_get_examples_ode_sol_2nd_linear_airy)


@slow
def test_lie_group():
    _ode_solver_test(_get_examples_ode_sol_lie_group)


@slow
def test_separable_reduced():
    df = f(x).diff(x)
    eq = (x / f(x))*df  + tan(x**2*f(x) / (x**2*f(x) - 1))
    assert classify_ode(eq) == ('factorable', 'separable_reduced', 'lie_group',
        'separable_reduced_Integral')
    _ode_solver_test(_get_examples_ode_sol_separable_reduced)


@slow
def test_slow_examples_separable_reduced():
    _ode_solver_test(_get_examples_ode_sol_separable_reduced, run_slow_test=True)


@slow
def test_2nd_2F1_hypergeometric():
    _ode_solver_test(_get_examples_ode_sol_2nd_2F1_hypergeometric)


def test_2nd_2F1_hypergeometric_integral():
    eq = x*(x-1)*f(x).diff(x, 2) + (-1+ S(7)/2*x)*f(x).diff(x) + f(x)
    sol = Eq(f(x), (C1 + C2*Integral(exp(Integral((1 - x/2)/(x*(x - 1)), x))/(1 -
          x/2)**2, x))*exp(Integral(1/(x - 1), x)/4)*exp(-Integral(7/(x -
          1), x)/4)*hyper((S(1)/2, -1), (1,), x))
    assert sol == dsolve(eq, hint='2nd_hypergeometric_Integral')
    assert checkodesol(eq, sol) == (True, 0)


@slow
def test_2nd_nonlinear_autonomous_conserved():
    _ode_solver_test(_get_examples_ode_sol_2nd_nonlinear_autonomous_conserved)


def test_2nd_nonlinear_autonomous_conserved_integral():
    eq = f(x).diff(x, 2) + asin(f(x))
    actual = [Eq(Integral(1/sqrt(C1 - 2*Integral(asin(_u), _u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(C1 - 2*Integral(asin(_u), _u)), (_u, f(x))), C2 - x)]
    solved = dsolve(eq, hint='2nd_nonlinear_autonomous_conserved_Integral', simplify=False)
    for a,s in zip(actual, solved):
        assert a.dummy_eq(s)
    # checkodesol unable to simplify solutions with f(x) in an integral equation
    assert checkodesol(eq, [s.doit() for s in solved]) == [(True, 0), (True, 0)]


@slow
def test_2nd_linear_bessel_equation():
    _ode_solver_test(_get_examples_ode_sol_2nd_linear_bessel)


@slow
def test_nth_algebraic():
    eqn = f(x) + f(x)*f(x).diff(x)
    solns = [Eq(f(x), exp(x)),
             Eq(f(x), C1*exp(C2*x))]
    solns_final =  _remove_redundant_solutions(eqn, solns, 2, x)
    assert solns_final == [Eq(f(x), C1*exp(C2*x))]

    _ode_solver_test(_get_examples_ode_sol_nth_algebraic)


@slow
def test_slow_examples_nth_linear_constant_coeff_var_of_parameters():
    _ode_solver_test(_get_examples_ode_sol_nth_linear_var_of_parameters, run_slow_test=True)


def test_nth_linear_constant_coeff_var_of_parameters():
    _ode_solver_test(_get_examples_ode_sol_nth_linear_var_of_parameters)


@slow
def test_nth_linear_constant_coeff_variation_of_parameters__integral():
    # solve_variation_of_parameters shouldn't attempt to simplify the
    # Wronskian if simplify=False.  If wronskian() ever gets good enough
    # to simplify the result itself, this test might fail.
    our_hint = 'nth_linear_constant_coeff_variation_of_parameters_Integral'
    eq = f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - 2*x - exp(I*x)
    sol_simp = dsolve(eq, f(x), hint=our_hint, simplify=True)
    sol_nsimp = dsolve(eq, f(x), hint=our_hint, simplify=False)
    assert sol_simp != sol_nsimp
    assert checkodesol(eq, sol_simp, order=5, solve_for_func=False) == (True, 0)
    assert checkodesol(eq, sol_simp, order=5, solve_for_func=False) == (True, 0)


@slow
def test_slow_examples_1st_exact():
    _ode_solver_test(_get_examples_ode_sol_1st_exact, run_slow_test=True)


@slow
def test_1st_exact():
    _ode_solver_test(_get_examples_ode_sol_1st_exact)


def test_1st_exact_integral():
    eq = cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x)
    sol_1 = dsolve(eq, f(x), simplify=False, hint='1st_exact_Integral')
    assert checkodesol(eq, sol_1, order=1, solve_for_func=False)


@slow
def test_slow_examples_nth_order_reducible():
    _ode_solver_test(_get_examples_ode_sol_nth_order_reducible, run_slow_test=True)


@slow
def test_slow_examples_nth_linear_constant_coeff_undetermined_coefficients():
    _ode_solver_test(_get_examples_ode_sol_nth_linear_undetermined_coefficients, run_slow_test=True)


@slow
def test_slow_examples_separable():
    _ode_solver_test(_get_examples_ode_sol_separable, run_slow_test=True)


@slow
def test_nth_linear_constant_coeff_undetermined_coefficients():
    #issue-https://github.com/sympy/sympy/issues/5787
    # This test case is to show the classification of imaginary constants under
    # nth_linear_constant_coeff_undetermined_coefficients
    eq = Eq(diff(f(x), x), I*f(x) + S.Half - I)
    our_hint = 'nth_linear_constant_coeff_undetermined_coefficients'
    assert our_hint in classify_ode(eq)
    _ode_solver_test(_get_examples_ode_sol_nth_linear_undetermined_coefficients)


def test_nth_order_reducible():
    F = lambda eq: NthOrderReducible(SingleODEProblem(eq, f(x), x))._matches()
    D = Derivative
    assert F(D(y*f(x), x, y) + D(f(x), x)) == False
    assert F(D(y*f(y), y, y) + D(f(y), y)) == False
    assert F(f(x)*D(f(x), x) + D(f(x), x, 2))== False
    assert F(D(x*f(y), y, 2) + D(u*y*f(x), x, 3)) == False  # no simplification by design
    assert F(D(f(y), y, 2) + D(f(y), y, 3) + D(f(x), x, 4)) == False
    assert F(D(f(x), x, 2) + D(f(x), x, 3)) == True
    _ode_solver_test(_get_examples_ode_sol_nth_order_reducible)


@slow
def test_separable():
    _ode_solver_test(_get_examples_ode_sol_separable)


@slow
def test_factorable():
    assert integrate(-asin(f(2*x)+pi), x) == -Integral(asin(pi + f(2*x)), x)
    _ode_solver_test(_get_examples_ode_sol_factorable)


@slow
def test_slow_examples_factorable():
    _ode_solver_test(_get_examples_ode_sol_factorable, run_slow_test=True)


def test_Riccati_special_minus2():
    _ode_solver_test(_get_examples_ode_sol_riccati)


@slow
def test_1st_rational_riccati():
    _ode_solver_test(_get_examples_ode_sol_1st_rational_riccati)


def test_Bernoulli():
    _ode_solver_test(_get_examples_ode_sol_bernoulli)


def test_1st_linear():
    _ode_solver_test(_get_examples_ode_sol_1st_linear)


def test_almost_linear():
    _ode_solver_test(_get_examples_ode_sol_almost_linear)


@slow
def test_Liouville_ODE():
    hint = 'Liouville'
    not_Liouville1 = classify_ode(diff(f(x), x)/x + f(x)*diff(f(x), x, x)/2 -
        diff(f(x), x)**2/2, f(x))
    not_Liouville2 = classify_ode(diff(f(x), x)/x + diff(f(x), x, x)/2 -
        x*diff(f(x), x)**2/2, f(x))
    assert hint not in not_Liouville1
    assert hint not in not_Liouville2
    assert hint + '_Integral' not in not_Liouville1
    assert hint + '_Integral' not in not_Liouville2

    _ode_solver_test(_get_examples_ode_sol_liouville)


def test_nth_order_linear_euler_eq_homogeneous():
    x, t, a, b, c = symbols('x t a b c')
    y = Function('y')
    our_hint = "nth_linear_euler_eq_homogeneous"

    eq = diff(f(t), t, 4)*t**4 - 13*diff(f(t), t, 2)*t**2 + 36*f(t)
    assert our_hint in classify_ode(eq)

    eq = a*y(t) + b*t*diff(y(t), t) + c*t**2*diff(y(t), t, 2)
    assert our_hint in classify_ode(eq)

    _ode_solver_test(_get_examples_ode_sol_euler_homogeneous)


def test_nth_order_linear_euler_eq_nonhomogeneous_undetermined_coefficients():
    x, t = symbols('x t')
    a, b, c, d = symbols('a b c d', integer=True)
    our_hint = "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients"

    eq = x**4*diff(f(x), x, 4) - 13*x**2*diff(f(x), x, 2) + 36*f(x) + x
    assert our_hint in classify_ode(eq, f(x))

    eq = a*x**2*diff(f(x), x, 2) + b*x*diff(f(x), x) + c*f(x) + d*log(x)
    assert our_hint in classify_ode(eq, f(x))

    _ode_solver_test(_get_examples_ode_sol_euler_undetermined_coeff)


@slow
def test_nth_order_linear_euler_eq_nonhomogeneous_variation_of_parameters():
    x, t = symbols('x, t')
    a, b, c, d = symbols('a, b, c, d', integer=True)
    our_hint = "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters"

    eq = Eq(x**2*diff(f(x),x,2) - 8*x*diff(f(x),x) + 12*f(x), x**2)
    assert our_hint in classify_ode(eq, f(x))

    eq = Eq(a*x**3*diff(f(x),x,3) + b*x**2*diff(f(x),x,2) + c*x*diff(f(x),x) + d*f(x), x*log(x))
    assert our_hint in classify_ode(eq, f(x))

    _ode_solver_test(_get_examples_ode_sol_euler_var_para)


@_add_example_keys
def _get_examples_ode_sol_euler_homogeneous():
    r1, r2, r3, r4, r5 = [rootof(x**5 - 14*x**4 + 71*x**3 - 154*x**2 + 120*x - 1, n) for n in range(5)]
    return {
            'hint': "nth_linear_euler_eq_homogeneous",
            'func': f(x),
            'examples':{
    'euler_hom_01': {
        'eq': Eq(-3*diff(f(x), x)*x + 2*x**2*diff(f(x), x, x), 0),
        'sol': [Eq(f(x), C1 + C2*x**Rational(5, 2))],
    },

    'euler_hom_02': {
        'eq': Eq(3*f(x) - 5*diff(f(x), x)*x + 2*x**2*diff(f(x), x, x), 0),
        'sol': [Eq(f(x), C1*sqrt(x) + C2*x**3)]
    },

    'euler_hom_03': {
        'eq': Eq(4*f(x) + 5*diff(f(x), x)*x + x**2*diff(f(x), x, x), 0),
        'sol': [Eq(f(x), (C1 + C2*log(x))/x**2)]
    },

    'euler_hom_04': {
        'eq': Eq(6*f(x) - 6*diff(f(x), x)*x + 1*x**2*diff(f(x), x, x) + x**3*diff(f(x), x, x, x), 0),
        'sol': [Eq(f(x), C1/x**2 + C2*x + C3*x**3)]
    },

    'euler_hom_05': {
        'eq': Eq(-125*f(x) + 61*diff(f(x), x)*x - 12*x**2*diff(f(x), x, x) + x**3*diff(f(x), x, x, x), 0),
        'sol': [Eq(f(x), x**5*(C1 + C2*log(x) + C3*log(x)**2))]
    },

    'euler_hom_06': {
        'eq': x**2*diff(f(x), x, 2) + x*diff(f(x), x) - 9*f(x),
        'sol': [Eq(f(x), C1*x**-3 + C2*x**3)]
    },

    'euler_hom_07': {
        'eq': sin(x)*x**2*f(x).diff(x, 2) + sin(x)*x*f(x).diff(x) + sin(x)*f(x),
        'sol': [Eq(f(x), C1*sin(log(x)) + C2*cos(log(x)))],
        'XFAIL': ['2nd_power_series_regular','nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients']
    },

    'euler_hom_08': {
        'eq': x**6 * f(x).diff(x, 6) - x*f(x).diff(x) + f(x),
        'sol': [Eq(f(x), C1*x + C2*x**r1 + C3*x**r2 + C4*x**r3 + C5*x**r4 + C6*x**r5)],
        'checkodesol_XFAIL':True
    },

     #This example is from issue: https://github.com/sympy/sympy/issues/15237 #This example is from issue:
     #  https://github.com/sympy/sympy/issues/15237
    'euler_hom_09': {
        'eq': Derivative(x*f(x), x, x, x),
        'sol': [Eq(f(x), C1 + C2/x + C3*x)],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_euler_undetermined_coeff():
    return {
            'hint': "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients",
            'func': f(x),
            'examples':{
    'euler_undet_01': {
        'eq': Eq(x**2*diff(f(x), x, x) + x*diff(f(x), x), 1),
        'sol': [Eq(f(x), C1 + C2*log(x) + log(x)**2/2)]
    },

    'euler_undet_02': {
        'eq': Eq(x**2*diff(f(x), x, x) - 2*x*diff(f(x), x) + 2*f(x), x**3),
        'sol': [Eq(f(x), x*(C1 + C2*x + Rational(1, 2)*x**2))]
    },

    'euler_undet_03': {
        'eq': Eq(x**2*diff(f(x), x, x) - x*diff(f(x), x) - 3*f(x), log(x)/x),
        'sol': [Eq(f(x), (C1 + C2*x**4 - log(x)**2/8 - log(x)/16)/x)]
    },

    'euler_undet_04': {
        'eq': Eq(x**2*diff(f(x), x, x) + 3*x*diff(f(x), x) - 8*f(x), log(x)**3 - log(x)),
        'sol': [Eq(f(x), C1/x**4 + C2*x**2 - Rational(1,8)*log(x)**3 - Rational(3,32)*log(x)**2 - Rational(1,64)*log(x) - Rational(7, 256))]
    },

    'euler_undet_05': {
        'eq': Eq(x**3*diff(f(x), x, x, x) - 3*x**2*diff(f(x), x, x) + 6*x*diff(f(x), x) - 6*f(x), log(x)),
        'sol': [Eq(f(x), C1*x + C2*x**2 + C3*x**3 - Rational(1, 6)*log(x) - Rational(11, 36))]
    },

    #Below examples were added for the issue: https://github.com/sympy/sympy/issues/5096
    'euler_undet_06': {
        'eq': 2*x**2*f(x).diff(x, 2) + f(x) + sqrt(2*x)*sin(log(2*x)/2),
        'sol': [Eq(f(x), sqrt(x)*(C1*sin(log(x)/2) + C2*cos(log(x)/2) + sqrt(2)*log(x)*cos(log(2*x)/2)/2))]
    },

    'euler_undet_07': {
        'eq': 2*x**2*f(x).diff(x, 2) + f(x) + sin(log(2*x)/2),
        'sol': [Eq(f(x), C1*sqrt(x)*sin(log(x)/2) + C2*sqrt(x)*cos(log(x)/2) - 2*sin(log(2*x)/2)/5 - 4*cos(log(2*x)/2)/5)]
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_euler_var_para():
    return {
            'hint': "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters",
            'func': f(x),
            'examples':{
    'euler_var_01': {
        'eq': Eq(x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x), x**4),
        'sol': [Eq(f(x), x*(C1 + C2*x + x**3/6))]
    },

    'euler_var_02': {
        'eq': Eq(3*x**2*diff(f(x), x, x) + 6*x*diff(f(x), x) - 6*f(x), x**3*exp(x)),
        'sol': [Eq(f(x), C1/x**2 + C2*x + x*exp(x)/3 - 4*exp(x)/3 + 8*exp(x)/(3*x) - 8*exp(x)/(3*x**2))]
    },

    'euler_var_03': {
        'eq': Eq(x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x), x**4*exp(x)),
        'sol':  [Eq(f(x), x*(C1 + C2*x + x*exp(x) - 2*exp(x)))]
    },

    'euler_var_04': {
        'eq': x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - log(x),
        'sol': [Eq(f(x), C1*x + C2*x**2 + log(x)/2 + Rational(3, 4))]
    },

    'euler_var_05': {
        'eq': -exp(x) + (x*Derivative(f(x), (x, 2)) + Derivative(f(x), x))/x,
        'sol': [Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))]
    },

    'euler_var_06': {
        'eq': x**2 * f(x).diff(x, 2) + x * f(x).diff(x) + 4 * f(x) - 1/x,
        'sol': [Eq(f(x), C1*sin(2*log(x)) + C2*cos(2*log(x)) + 1/(5*x))]
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_bernoulli():
    # Type: Bernoulli, f'(x) + p(x)*f(x) == q(x)*f(x)**n
    return {
            'hint': "Bernoulli",
            'func': f(x),
            'examples':{
    'bernoulli_01': {
        'eq': Eq(x*f(x).diff(x) + f(x) - f(x)**2, 0),
        'sol': [Eq(f(x), 1/(C1*x + 1))],
        'XFAIL': ['separable_reduced']
    },

    'bernoulli_02': {
        'eq': f(x).diff(x) - y*f(x),
        'sol': [Eq(f(x), C1*exp(x*y))]
    },

    'bernoulli_03': {
        'eq': f(x)*f(x).diff(x) - 1,
        'sol': [Eq(f(x), -sqrt(C1 + 2*x)), Eq(f(x), sqrt(C1 + 2*x))]
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_riccati():
    # Type: Riccati special alpha = -2, a*dy/dx + b*y**2 + c*y/x +d/x**2
    return {
            'hint': "Riccati_special_minus2",
            'func': f(x),
            'examples':{
    'riccati_01': {
        'eq': 2*f(x).diff(x) + f(x)**2 - f(x)/x + 3*x**(-2),
        'sol': [Eq(f(x), (-sqrt(3)*tan(C1 + sqrt(3)*log(x)/4) + 3)/(2*x))],
    },
    },
    }


@_add_example_keys
def _get_examples_ode_sol_1st_rational_riccati():
    # Type: 1st Order Rational Riccati, dy/dx = a + b*y + c*y**2,
    # a, b, c are rational functions of x
    return {
            'hint': "1st_rational_riccati",
            'func': f(x),
            'examples':{
    # a(x) is a constant
    "rational_riccati_01": {
        "eq": Eq(f(x).diff(x) + f(x)**2 - 2, 0),
        "sol": [Eq(f(x), sqrt(2)*(-C1 - exp(2*sqrt(2)*x))/(C1 - exp(2*sqrt(2)*x)))]
    },
    # a(x) is a constant
    "rational_riccati_02": {
        "eq": f(x)**2 + Derivative(f(x), x) + 4*f(x)/x + 2/x**2,
        "sol": [Eq(f(x), (-2*C1 - x)/(x*(C1 + x)))]
    },
    # a(x) is a constant
    "rational_riccati_03": {
        "eq": 2*x**2*Derivative(f(x), x) - x*(4*f(x) + Derivative(f(x), x) - 4) + (f(x) - 1)*f(x),
        "sol": [Eq(f(x), (C1 + 2*x**2)/(C1 + x))]
    },
    # Constant coefficients
    "rational_riccati_04": {
        "eq": f(x).diff(x) - 6 - 5*f(x) - f(x)**2,
        "sol": [Eq(f(x), (-2*C1 + 3*exp(x))/(C1 - exp(x)))]
    },
    # One pole of multiplicity 2
    "rational_riccati_05": {
        "eq": x**2 - (2*x + 1/x)*f(x) + f(x)**2 + Derivative(f(x), x),
        "sol": [Eq(f(x), x*(C1 + x**2 + 1)/(C1 + x**2 - 1))]
    },
    # One pole of multiplicity 2
    "rational_riccati_06": {
        "eq": x**4*Derivative(f(x), x) + x**2 - x*(2*f(x)**2 + Derivative(f(x), x)) + f(x),
        "sol": [Eq(f(x), x*(C1*x - x + 1)/(C1 + x**2 - 1))]
    },
    # Multiple poles of multiplicity 2
    "rational_riccati_07": {
        "eq": -f(x)**2 + Derivative(f(x), x) + (15*x**2 - 20*x + 7)/((x - 1)**2*(2*x \
            - 1)**2),
        "sol": [Eq(f(x), (9*C1*x - 6*C1 - 15*x**5 + 60*x**4 - 94*x**3 + 72*x**2 - \
            33*x + 8)/(6*C1*x**2 - 9*C1*x + 3*C1 + 6*x**6 - 29*x**5 + 57*x**4 - \
            58*x**3 + 28*x**2 - 3*x - 1))]
    },
    # Imaginary poles
    "rational_riccati_08": {
        "eq": Derivative(f(x), x) + (3*x**2 + 1)*f(x)**2/x + (6*x**2 - x + 3)*f(x)/(x*(x \
            - 1)) + (3*x**2 - 2*x + 2)/(x*(x - 1)**2),
        "sol": [Eq(f(x), (-C1 - x**3 + x**2 - 2*x + 1)/(C1*x - C1 + x**4 - x**3 + x**2 - \
            2*x + 1))],
    },
    # Imaginary coefficients in equation
    "rational_riccati_09": {
        "eq": Derivative(f(x), x) - 2*I*(f(x)**2 + 1)/x,
        "sol": [Eq(f(x), (-I*C1 + I*x**4 + I)/(C1 + x**4 - 1))]
    },
    # Regression: linsolve returning empty solution
    # Large value of m (> 10)
    "rational_riccati_10": {
        "eq": Eq(Derivative(f(x), x), x*f(x)/(S(3)/2 - 2*x) + (x/2 - S(1)/3)*f(x)**2/\
            (2*x/3 - S(1)/2) - S(5)/4 + (281*x**2 - 1260*x + 756)/(16*x**3 - 12*x**2)),
        "sol": [Eq(f(x), (40*C1*x**14 + 28*C1*x**13 + 420*C1*x**12 + 2940*C1*x**11 + \
            18480*C1*x**10 + 103950*C1*x**9 + 519750*C1*x**8 + 2286900*C1*x**7 + \
            8731800*C1*x**6 + 28378350*C1*x**5 + 76403250*C1*x**4 + 163721250*C1*x**3 \
            + 261954000*C1*x**2 + 278326125*C1*x + 147349125*C1 + x*exp(2*x) - 9*exp(2*x) \
            )/(x*(24*C1*x**13 + 140*C1*x**12 + 840*C1*x**11 + 4620*C1*x**10 + 23100*C1*x**9 \
            + 103950*C1*x**8 + 415800*C1*x**7 + 1455300*C1*x**6 + 4365900*C1*x**5 + \
            10914750*C1*x**4 + 21829500*C1*x**3 + 32744250*C1*x**2 + 32744250*C1*x + \
            16372125*C1 - exp(2*x))))]
    }
    }
    }



@_add_example_keys
def _get_examples_ode_sol_1st_linear():
    # Type: first order linear form f'(x)+p(x)f(x)=q(x)
    return {
            'hint': "1st_linear",
            'func': f(x),
            'examples':{
    'linear_01': {
        'eq': Eq(f(x).diff(x) + x*f(x), x**2),
        'sol': [Eq(f(x), (C1 + x*exp(x**2/2)- sqrt(2)*sqrt(pi)*erfi(sqrt(2)*x/2)/2)*exp(-x**2/2))],
    },
    },
    }


@_add_example_keys
def _get_examples_ode_sol_factorable():
    """ some hints are marked as xfail for examples because they missed additional algebraic solution
    which could be found by Factorable hint. Fact_01 raise exception for
    nth_linear_constant_coeff_undetermined_coefficients"""

    y = Dummy('y')
    a0,a1,a2,a3,a4 = symbols('a0, a1, a2, a3, a4')
    return {
            'hint': "factorable",
            'func': f(x),
            'examples':{
    'fact_01': {
        'eq': f(x) + f(x)*f(x).diff(x),
        'sol': [Eq(f(x), 0), Eq(f(x), C1 - x)],
        'XFAIL': ['separable', '1st_exact', '1st_linear', 'Bernoulli', '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_dep_div_indep',
        'lie_group', 'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients',
        'nth_linear_constant_coeff_variation_of_parameters',
        'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters',
        'nth_linear_constant_coeff_undetermined_coefficients']
    },

    'fact_02': {
        'eq': f(x)*(f(x).diff(x)+f(x)*x+2),
        'sol': [Eq(f(x), (C1 - sqrt(2)*sqrt(pi)*erfi(sqrt(2)*x/2))*exp(-x**2/2)), Eq(f(x), 0)],
        'XFAIL': ['Bernoulli', '1st_linear', 'lie_group']
    },

    'fact_03': {
        'eq': (f(x).diff(x)+f(x)*x**2)*(f(x).diff(x, 2) + x*f(x)),
        'sol':  [Eq(f(x), C1*airyai(-x) + C2*airybi(-x)),Eq(f(x), C1*exp(-x**3/3))]
    },

    'fact_04': {
        'eq': (f(x).diff(x)+f(x)*x**2)*(f(x).diff(x, 2) + f(x)),
        'sol': [Eq(f(x), C1*exp(-x**3/3)), Eq(f(x), C1*sin(x) + C2*cos(x))]
    },

    'fact_05': {
        'eq': (f(x).diff(x)**2-1)*(f(x).diff(x)**2-4),
        'sol': [Eq(f(x), C1 - x), Eq(f(x), C1 + x), Eq(f(x), C1 + 2*x), Eq(f(x), C1 - 2*x)]
    },

    'fact_06': {
        'eq': (f(x).diff(x, 2)-exp(f(x)))*f(x).diff(x),
        'sol': [
            Eq(f(x), log(-C1/(cos(sqrt(-C1)*(C2 + x)) + 1))),
            Eq(f(x), log(-C1/(cos(sqrt(-C1)*(C2 - x)) + 1))),
            Eq(f(x), C1)
        ],
        'slow': True,
    },

    'fact_07': {
        'eq': (f(x).diff(x)**2-1)*(f(x)*f(x).diff(x)-1),
        'sol': [Eq(f(x), C1 - x), Eq(f(x), -sqrt(C1 + 2*x)),Eq(f(x), sqrt(C1 + 2*x)), Eq(f(x), C1 + x)]
    },

    'fact_08': {
        'eq': Derivative(f(x), x)**4 - 2*Derivative(f(x), x)**2 + 1,
        'sol': [Eq(f(x), C1 - x), Eq(f(x), C1 + x)]
    },

    'fact_09': {
        'eq': f(x)**2*Derivative(f(x), x)**6 - 2*f(x)**2*Derivative(f(x),
         x)**4 + f(x)**2*Derivative(f(x), x)**2 - 2*f(x)*Derivative(f(x),
         x)**5 + 4*f(x)*Derivative(f(x), x)**3 - 2*f(x)*Derivative(f(x),
         x) + Derivative(f(x), x)**4 - 2*Derivative(f(x), x)**2 + 1,
        'sol': [
            Eq(f(x), C1 - x), Eq(f(x), -sqrt(C1 + 2*x)),
            Eq(f(x), sqrt(C1 + 2*x)), Eq(f(x), C1 + x)
        ]
    },

    'fact_10': {
        'eq': x**4*f(x)**2 + 2*x**4*f(x)*Derivative(f(x), (x, 2)) + x**4*Derivative(f(x),
         (x, 2))**2  + 2*x**3*f(x)*Derivative(f(x), x) + 2*x**3*Derivative(f(x),
         x)*Derivative(f(x), (x, 2)) - 7*x**2*f(x)**2 - 7*x**2*f(x)*Derivative(f(x),
         (x, 2)) + x**2*Derivative(f(x), x)**2 - 7*x*f(x)*Derivative(f(x), x) + 12*f(x)**2,
        'sol': [
            Eq(f(x), C1*besselj(2, x) + C2*bessely(2, x)),
            Eq(f(x), C1*besselj(sqrt(3), x) + C2*bessely(sqrt(3), x))
        ],
        'slow': True,
    },

    'fact_11': {
        'eq': (f(x).diff(x, 2)-exp(f(x)))*(f(x).diff(x, 2)+exp(f(x))),
        'sol': [
            Eq(f(x), log(C1/(cos(C1*sqrt(-1/C1)*(C2 + x)) - 1))),
            Eq(f(x), log(C1/(cos(C1*sqrt(-1/C1)*(C2 - x)) - 1))),
            Eq(f(x), log(C1/(1 - cos(C1*sqrt(-1/C1)*(C2 + x))))),
            Eq(f(x), log(C1/(1 - cos(C1*sqrt(-1/C1)*(C2 - x)))))
        ],
        'dsolve_too_slow': True,
    },

    #Below examples were added for the issue: https://github.com/sympy/sympy/issues/15889
    'fact_12': {
        'eq': exp(f(x).diff(x))-f(x)**2,
        'sol': [Eq(NonElementaryIntegral(1/log(y**2), (y, f(x))), C1 + x)],
        'XFAIL': ['lie_group'] #It shows not implemented error for lie_group.
    },

    'fact_13': {
        'eq': f(x).diff(x)**2 - f(x)**3,
        'sol': [Eq(f(x), 4/(C1**2 - 2*C1*x + x**2))],
        'XFAIL': ['lie_group'] #It shows not implemented error for lie_group.
    },

    'fact_14': {
        'eq': f(x).diff(x)**2 - f(x),
        'sol': [Eq(f(x), C1**2/4 - C1*x/2 + x**2/4)]
    },

    'fact_15': {
        'eq': f(x).diff(x)**2 - f(x)**2,
        'sol': [Eq(f(x), C1*exp(x)), Eq(f(x), C1*exp(-x))]
    },

    'fact_16': {
        'eq': f(x).diff(x)**2 - f(x)**3,
        'sol': [Eq(f(x), 4/(C1**2 - 2*C1*x + x**2))],
    },

    # kamke ode 1.1
    'fact_17': {
        'eq': f(x).diff(x)-(a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)**(-1/2),
        'sol': [Eq(f(x), C1 + Integral(1/sqrt(a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4), x))],
        'slow': True
    },

    # This is from issue: https://github.com/sympy/sympy/issues/9446
    'fact_18':{
        'eq': Eq(f(2 * x), sin(Derivative(f(x)))),
        'sol': [Eq(f(x), C1 + Integral(pi - asin(f(2*x)), x)), Eq(f(x), C1 + Integral(asin(f(2*x)), x))],
        'checkodesol_XFAIL':True
    },

    # This is from issue: https://github.com/sympy/sympy/issues/7093
    'fact_19': {
        'eq': Derivative(f(x), x)**2 - x**3,
        'sol': [Eq(f(x), C1 - 2*x**Rational(5,2)/5), Eq(f(x), C1 + 2*x**Rational(5,2)/5)],
    },

    'fact_20': {
        'eq': x*f(x).diff(x, 2) - x*f(x),
        'sol': [Eq(f(x), C1*exp(-x) + C2*exp(x))],
    },
    }
    }



@_add_example_keys
def _get_examples_ode_sol_almost_linear():
    from sympy.functions.special.error_functions import Ei
    A = Symbol('A', positive=True)
    f = Function('f')
    d = f(x).diff(x)

    return {
            'hint': "almost_linear",
            'func': f(x),
            'examples':{
    'almost_lin_01': {
        'eq': x**2*f(x)**2*d + f(x)**3 + 1,
        'sol': [Eq(f(x), (C1*exp(3/x) - 1)**Rational(1, 3)),
        Eq(f(x), (-1 - sqrt(3)*I)*(C1*exp(3/x) - 1)**Rational(1, 3)/2),
        Eq(f(x), (-1 + sqrt(3)*I)*(C1*exp(3/x) - 1)**Rational(1, 3)/2)],

    },

    'almost_lin_02': {
        'eq': x*f(x)*d + 2*x*f(x)**2 + 1,
        'sol': [Eq(f(x), -sqrt((C1 - 2*Ei(4*x))*exp(-4*x))), Eq(f(x), sqrt((C1 - 2*Ei(4*x))*exp(-4*x)))]
    },

    'almost_lin_03': {
        'eq':  x*d + x*f(x) + 1,
        'sol': [Eq(f(x), (C1 - Ei(x))*exp(-x))]
    },

    'almost_lin_04': {
        'eq': x*exp(f(x))*d + exp(f(x)) + 3*x,
        'sol': [Eq(f(x), log(C1/x - x*Rational(3, 2)))],
    },

    'almost_lin_05': {
        'eq': x + A*(x + diff(f(x), x) + f(x)) + diff(f(x), x) + f(x) + 2,
        'sol': [Eq(f(x), (C1 + Piecewise(
        (x, Eq(A + 1, 0)), ((-A*x + A - x - 1)*exp(x)/(A + 1), True)))*exp(-x))],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_liouville():
    n = Symbol('n')
    _y = Dummy('y')
    return {
            'hint': "Liouville",
            'func': f(x),
            'examples':{
    'liouville_01': {
        'eq': diff(f(x), x)/x + diff(f(x), x, x)/2 - diff(f(x), x)**2/2,
        'sol': [Eq(f(x), log(x/(C1 + C2*x)))],

    },

    'liouville_02': {
        'eq': diff(x*exp(-f(x)), x, x),
        'sol': [Eq(f(x), log(x/(C1 + C2*x)))]
    },

    'liouville_03': {
        'eq':  ((diff(f(x), x)/x + diff(f(x), x, x)/2 - diff(f(x), x)**2/2)*exp(-f(x))/exp(f(x))).expand(),
        'sol': [Eq(f(x), log(x/(C1 + C2*x)))]
    },

    'liouville_04': {
        'eq': diff(f(x), x, x) + 1/f(x)*(diff(f(x), x))**2 + 1/x*diff(f(x), x),
        'sol': [Eq(f(x), -sqrt(C1 + C2*log(x))), Eq(f(x), sqrt(C1 + C2*log(x)))],
    },

    'liouville_05': {
        'eq': x*diff(f(x), x, x) + x/f(x)*diff(f(x), x)**2 + x*diff(f(x), x),
        'sol': [Eq(f(x), -sqrt(C1 + C2*exp(-x))), Eq(f(x), sqrt(C1 + C2*exp(-x)))],
    },

    'liouville_06': {
        'eq': Eq((x*exp(f(x))).diff(x, x), 0),
        'sol': [Eq(f(x), log(C1 + C2/x))],
    },

    'liouville_07': {
        'eq': (diff(f(x), x)/x + diff(f(x), x, x)/2 - diff(f(x), x)**2/2)*exp(-f(x))/exp(f(x)),
        'sol': [Eq(f(x), log(x/(C1 + C2*x)))],
    },

    'liouville_08': {
        'eq': x**2*diff(f(x),x) + (n*f(x) + f(x)**2)*diff(f(x),x)**2 + diff(f(x), (x, 2)),
        'sol': [Eq(C1 + C2*lowergamma(Rational(1,3), x**3/3) + NonElementaryIntegral(exp(_y**3/3)*exp(_y**2*n/2), (_y, f(x))), 0)],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_nth_algebraic():
    M, m, r, t = symbols('M m r t')
    phi = Function('phi')
    k = Symbol('k')
    # This one needs a substitution f' = g.
    # 'algeb_12': {
    #     'eq': -exp(x) + (x*Derivative(f(x), (x, 2)) + Derivative(f(x), x))/x,
    #     'sol': [Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))],
    # },
    return {
            'hint': "nth_algebraic",
            'func': f(x),
            'examples':{
    'algeb_01': {
        'eq': f(x) * f(x).diff(x) * f(x).diff(x, x) * (f(x) - 1) * (f(x).diff(x) - x),
        'sol': [Eq(f(x), C1 + x**2/2), Eq(f(x), C1 + C2*x)]
    },

    'algeb_02': {
        'eq': f(x) * f(x).diff(x) * f(x).diff(x, x) * (f(x) - 1),
        'sol': [Eq(f(x), C1 + C2*x)]
    },

    'algeb_03': {
        'eq': f(x) * f(x).diff(x) * f(x).diff(x, x),
        'sol': [Eq(f(x), C1 + C2*x)]
    },

    'algeb_04': {
        'eq': Eq(-M * phi(t).diff(t),
         Rational(3, 2) * m * r**2 * phi(t).diff(t) * phi(t).diff(t,t)),
        'sol': [Eq(phi(t), C1), Eq(phi(t), C1 + C2*t - M*t**2/(3*m*r**2))],
        'func': phi(t)
    },

    'algeb_05': {
        'eq': (1 - sin(f(x))) * f(x).diff(x),
        'sol': [Eq(f(x), C1)],
        'XFAIL': ['separable']  #It raised exception.
    },

    'algeb_06': {
        'eq': (diff(f(x)) - x)*(diff(f(x)) + x),
        'sol': [Eq(f(x), C1 - x**2/2), Eq(f(x), C1 + x**2/2)]
    },

    'algeb_07': {
        'eq': Eq(Derivative(f(x), x), Derivative(g(x), x)),
        'sol': [Eq(f(x), C1 + g(x))],
    },

    'algeb_08': {
        'eq': f(x).diff(x) - C1,   #this example is from issue 15999
        'sol': [Eq(f(x), C1*x + C2)],
    },

    'algeb_09': {
        'eq': f(x)*f(x).diff(x),
        'sol': [Eq(f(x), C1)],
    },

    'algeb_10': {
        'eq': (diff(f(x)) - x)*(diff(f(x)) + x),
        'sol': [Eq(f(x), C1 - x**2/2), Eq(f(x), C1 + x**2/2)],
    },

    'algeb_11': {
        'eq': f(x) + f(x)*f(x).diff(x),
        'sol': [Eq(f(x), 0), Eq(f(x), C1 - x)],
        'XFAIL': ['separable', '1st_exact', '1st_linear', 'Bernoulli', '1st_homogeneous_coeff_best',
         '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_dep_div_indep',
         'lie_group', 'nth_linear_constant_coeff_undetermined_coefficients',
         'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients',
         'nth_linear_constant_coeff_variation_of_parameters',
         'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters']
         #nth_linear_constant_coeff_undetermined_coefficients raises exception rest all of them misses a solution.
    },

    'algeb_12': {
        'eq': Derivative(x*f(x), x, x, x),
        'sol': [Eq(f(x), (C1 + C2*x + C3*x**2) / x)],
        'XFAIL': ['nth_algebraic']  # It passes only when prep=False is set in dsolve.
    },

    'algeb_13': {
        'eq': Eq(Derivative(x*Derivative(f(x), x), x)/x, exp(x)),
        'sol': [Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))],
        'XFAIL': ['nth_algebraic']  # It passes only when prep=False is set in dsolve.
    },

     # These are simple tests from the old ode module example 14-18
    'algeb_14': {
        'eq': Eq(f(x).diff(x), 0),
        'sol': [Eq(f(x), C1)],
    },

    'algeb_15': {
        'eq': Eq(3*f(x).diff(x) - 5, 0),
        'sol': [Eq(f(x), C1 + x*Rational(5, 3))],
    },

    'algeb_16': {
        'eq': Eq(3*f(x).diff(x), 5),
        'sol': [Eq(f(x), C1 + x*Rational(5, 3))],
    },

    # Type: 2nd order, constant coefficients (two complex roots)
    'algeb_17': {
        'eq': Eq(3*f(x).diff(x) - 1, 0),
        'sol': [Eq(f(x), C1 + x/3)],
    },

    'algeb_18': {
        'eq': Eq(x*f(x).diff(x) - 1, 0),
        'sol': [Eq(f(x), C1 + log(x))],
    },

    # https://github.com/sympy/sympy/issues/6989
    'algeb_19': {
        'eq': f(x).diff(x) - x*exp(-k*x),
        'sol': [Eq(f(x), C1 + Piecewise(((-k*x - 1)*exp(-k*x)/k**2, Ne(k**2, 0)),(x**2/2, True)))],
    },

    'algeb_20': {
        'eq': -f(x).diff(x) + x*exp(-k*x),
        'sol': [Eq(f(x), C1 + Piecewise(((-k*x - 1)*exp(-k*x)/k**2, Ne(k**2, 0)),(x**2/2, True)))],
    },

    # https://github.com/sympy/sympy/issues/10867
    'algeb_21': {
        'eq': Eq(g(x).diff(x).diff(x), (x-2)**2 + (x-3)**3),
        'sol': [Eq(g(x), C1 + C2*x + x**5/20 - 2*x**4/3 + 23*x**3/6 - 23*x**2/2)],
        'func': g(x),
    },

    # https://github.com/sympy/sympy/issues/13691
    'algeb_22': {
        'eq': f(x).diff(x) - C1*g(x).diff(x),
        'sol': [Eq(f(x), C2 + C1*g(x))],
        'func': f(x),
    },

    # https://github.com/sympy/sympy/issues/4838
    'algeb_23': {
        'eq': f(x).diff(x) - 3*C1 - 3*x**2,
        'sol': [Eq(f(x), C2 + 3*C1*x + x**3)],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_nth_order_reducible():
    return {
            'hint': "nth_order_reducible",
            'func': f(x),
            'examples':{
    'reducible_01': {
        'eq': Eq(x*Derivative(f(x), x)**2 + Derivative(f(x), x, 2), 0),
        'sol': [Eq(f(x),C1 - sqrt(-1/C2)*log(-C2*sqrt(-1/C2) + x) +
        sqrt(-1/C2)*log(C2*sqrt(-1/C2) + x))],
        'slow': True,
    },

    'reducible_02': {
        'eq': -exp(x) + (x*Derivative(f(x), (x, 2)) + Derivative(f(x), x))/x,
        'sol': [Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))],
        'slow': True,
    },

    'reducible_03': {
        'eq': Eq(sqrt(2) * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*sin(2**Rational(3, 4)*x/2) + C3*cos(2**Rational(3, 4)*x/2))],
        'slow': True,
    },

    'reducible_04': {
        'eq': f(x).diff(x, 2) + 2*f(x).diff(x),
        'sol': [Eq(f(x), C1 + C2*exp(-2*x))],
    },

    'reducible_05': {
        'eq': f(x).diff(x, 3) + f(x).diff(x, 2) - 6*f(x).diff(x),
        'sol': [Eq(f(x), C1 + C2*exp(-3*x) + C3*exp(2*x))],
        'slow': True,
    },

    'reducible_06': {
        'eq': f(x).diff(x, 4) - f(x).diff(x, 3) - 4*f(x).diff(x, 2) + \
        4*f(x).diff(x),
        'sol': [Eq(f(x), C1 + C2*exp(-2*x) + C3*exp(x) + C4*exp(2*x))],
        'slow': True,
    },

    'reducible_07': {
        'eq': f(x).diff(x, 4) + 3*f(x).diff(x, 3),
        'sol': [Eq(f(x), C1 + C2*x + C3*x**2 + C4*exp(-3*x))],
        'slow': True,
    },

    'reducible_08': {
        'eq': f(x).diff(x, 4) - 2*f(x).diff(x, 2),
        'sol': [Eq(f(x), C1 + C2*x + C3*exp(-sqrt(2)*x) + C4*exp(sqrt(2)*x))],
        'slow': True,
    },

    'reducible_09': {
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 2),
        'sol': [Eq(f(x), C1 + C2*x + C3*sin(2*x) + C4*cos(2*x))],
        'slow': True,
    },

    'reducible_10': {
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x),
        'sol': [Eq(f(x), C1 + C2*x*sin(x) + C2*cos(x) - C3*x*cos(x) + C3*sin(x) + C4*sin(x) + C5*cos(x))],
        'slow': True,
    },

    'reducible_11': {
        'eq': f(x).diff(x, 2) - f(x).diff(x)**3,
        'sol': [Eq(f(x), C1 - sqrt(2)*sqrt(-1/(C2 + x))*(C2 + x)),
        Eq(f(x), C1 + sqrt(2)*sqrt(-1/(C2 + x))*(C2 + x))],
        'slow': True,
    },

    # Needs to be a way to know how to combine derivatives in the expression
    'reducible_12': {
        'eq': Derivative(x*f(x), x, x, x) + Derivative(f(x), x, x, x),
        'sol': [Eq(f(x), C1 + C3/Mul(2, (x**2 + 2*x + 1), evaluate=False) +
        x*(C2 + C3/Mul(2, (x**2 + 2*x + 1), evaluate=False)))], # 2-arg Mul!
        'slow': True,
    },
    }
    }



@_add_example_keys
def _get_examples_ode_sol_nth_linear_undetermined_coefficients():
    # examples 3-27 below are from Ordinary Differential Equations,
    #                     Tenenbaum and Pollard, pg. 231
    g = exp(-x)
    f2 = f(x).diff(x, 2)
    c = 3*f(x).diff(x, 3) + 5*f2 + f(x).diff(x) - f(x) - x
    t = symbols("t")
    u = symbols("u",cls=Function)
    R, L, C, E_0, alpha = symbols("R L C E_0 alpha",positive=True)
    omega = Symbol('omega')
    return {
            'hint': "nth_linear_constant_coeff_undetermined_coefficients",
            'func': f(x),
            'examples':{
    'undet_01': {
        'eq': c - x*g,
        'sol': [Eq(f(x), C3*exp(x/3) - x + (C1 + x*(C2 - x**2/24 - 3*x/32))*exp(-x) - 1)],
        'slow': True,
    },

    'undet_02': {
        'eq': c - g,
        'sol': [Eq(f(x), C3*exp(x/3) - x + (C1 + x*(C2 - x/8))*exp(-x) - 1)],
        'slow': True,
    },

    'undet_03': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - 4,
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 2)],
        'slow': True,
    },

    'undet_04': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - 12*exp(x),
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 2*exp(x))],
        'slow': True,
    },

    'undet_05': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - exp(I*x),
        'sol': [Eq(f(x), (S(3)/10 + I/10)*(C1*exp(-2*x) + C2*exp(-x) - I*exp(I*x)))],
        'slow': True,
    },

    'undet_06': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - sin(x),
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + sin(x)/10 - 3*cos(x)/10)],
        'slow': True,
    },

    'undet_07': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - cos(x),
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 3*sin(x)/10 + cos(x)/10)],
        'slow': True,
    },

    'undet_08': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - (8 + 6*exp(x) + 2*sin(x)),
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + exp(x) + sin(x)/5 - 3*cos(x)/5 + 4)],
        'slow': True,
    },

    'undet_09': {
        'eq': f2 + f(x).diff(x) + f(x) - x**2,
        'sol': [Eq(f(x), -2*x + x**2 + (C1*sin(x*sqrt(3)/2) + C2*cos(x*sqrt(3)/2))*exp(-x/2))],
        'slow': True,
    },

    'undet_10': {
        'eq': f2 - 2*f(x).diff(x) - 8*f(x) - 9*x*exp(x) - 10*exp(-x),
        'sol': [Eq(f(x), -x*exp(x) - 2*exp(-x) + C1*exp(-2*x) + C2*exp(4*x))],
        'slow': True,
    },

    'undet_11': {
        'eq': f2 - 3*f(x).diff(x) - 2*exp(2*x)*sin(x),
        'sol': [Eq(f(x), C1 + C2*exp(3*x) - 3*exp(2*x)*sin(x)/5 - exp(2*x)*cos(x)/5)],
        'slow': True,
    },

    'undet_12': {
        'eq': f(x).diff(x, 4) - 2*f2 + f(x) - x + sin(x),
        'sol': [Eq(f(x), x - sin(x)/4 + (C1 + C2*x)*exp(-x) + (C3 + C4*x)*exp(x))],
        'slow': True,
    },

    'undet_13': {
        'eq': f2 + f(x).diff(x) - x**2 - 2*x,
        'sol': [Eq(f(x), C1 + x**3/3 + C2*exp(-x))],
        'slow': True,
    },

    'undet_14': {
        'eq': f2 + f(x).diff(x) - x - sin(2*x),
        'sol': [Eq(f(x), C1 - x - sin(2*x)/5 - cos(2*x)/10 + x**2/2 + C2*exp(-x))],
        'slow': True,
    },

    'undet_15': {
        'eq': f2 + f(x) - 4*x*sin(x),
        'sol': [Eq(f(x), (C1 - x**2)*cos(x) + (C2 + x)*sin(x))],
        'slow': True,
    },

    'undet_16': {
        'eq': f2 + 4*f(x) - x*sin(2*x),
        'sol': [Eq(f(x), (C1 - x**2/8)*cos(2*x) + (C2 + x/16)*sin(2*x))],
        'slow': True,
    },

    'undet_17': {
        'eq': f2 + 2*f(x).diff(x) + f(x) - x**2*exp(-x),
        'sol': [Eq(f(x), (C1 + x*(C2 + x**3/12))*exp(-x))],
        'slow': True,
    },

    'undet_18': {
        'eq': f(x).diff(x, 3) + 3*f2 + 3*f(x).diff(x) + f(x) - 2*exp(-x) + \
        x**2*exp(-x),
        'sol': [Eq(f(x), (C1 + x*(C2 + x*(C3 - x**3/60 + x/3)))*exp(-x))],
        'slow': True,
    },

    'undet_19': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - exp(-2*x) - x**2,
        'sol': [Eq(f(x), C2*exp(-x) + x**2/2 - x*Rational(3,2) + (C1 - x)*exp(-2*x) + Rational(7,4))],
        'slow': True,
    },

    'undet_20': {
        'eq': f2 - 3*f(x).diff(x) + 2*f(x) - x*exp(-x),
        'sol': [Eq(f(x), C1*exp(x) + C2*exp(2*x) + (6*x + 5)*exp(-x)/36)],
        'slow': True,
    },

    'undet_21': {
        'eq': f2 + f(x).diff(x) - 6*f(x) - x - exp(2*x),
        'sol': [Eq(f(x), Rational(-1, 36) - x/6 + C2*exp(-3*x) + (C1 + x/5)*exp(2*x))],
        'slow': True,
    },

    'undet_22': {
        'eq': f2 + f(x) - sin(x) - exp(-x),
        'sol': [Eq(f(x), C2*sin(x) + (C1 - x/2)*cos(x) + exp(-x)/2)],
        'slow': True,
    },

    'undet_23': {
        'eq': f(x).diff(x, 3) - 3*f2 + 3*f(x).diff(x) - f(x) - exp(x),
        'sol': [Eq(f(x), (C1 + x*(C2 + x*(C3 + x/6)))*exp(x))],
        'slow': True,
    },

    'undet_24': {
        'eq': f2 + f(x) - S.Half - cos(2*x)/2,
        'sol': [Eq(f(x), S.Half - cos(2*x)/6 + C1*sin(x) + C2*cos(x))],
        'slow': True,
    },

    'undet_25': {
        'eq': f(x).diff(x, 3) - f(x).diff(x) - exp(2*x)*(S.Half - cos(2*x)/2),
        'sol': [Eq(f(x), C1 + C2*exp(-x) + C3*exp(x) + (-21*sin(2*x) + 27*cos(2*x) + 130)*exp(2*x)/1560)],
        'slow': True,
    },

    #Note: 'undet_26' is referred in 'undet_37'
    'undet_26': {
        'eq': (f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - 2*x -
        sin(x) - cos(x)),
        'sol': [Eq(f(x), C1 + x**2 + (C2 + x*(C3 - x/8))*sin(x) + (C4 + x*(C5 + x/8))*cos(x))],
        'slow': True,
    },

    'undet_27': {
        'eq': f2 + f(x) - cos(x)/2 + cos(3*x)/2,
        'sol': [Eq(f(x), cos(3*x)/16 + C2*cos(x) + (C1 + x/4)*sin(x))],
        'slow': True,
    },

    'undet_28': {
        'eq': f(x).diff(x) - 1,
        'sol': [Eq(f(x), C1 + x)],
        'slow': True,
    },

    # https://github.com/sympy/sympy/issues/19358
    'undet_29': {
        'eq': f2 + f(x).diff(x) + exp(x-C1),
        'sol': [Eq(f(x), C2 + C3*exp(-x) - exp(-C1 + x)/2)],
        'slow': True,
    },

    # https://github.com/sympy/sympy/issues/18408
    'undet_30': {
        'eq': f(x).diff(x, 3) - f(x).diff(x) - sinh(x),
        'sol': [Eq(f(x), C1 + C2*exp(-x) + C3*exp(x) + x*sinh(x)/2)],
    },

    'undet_31': {
        'eq': f(x).diff(x, 2) - 49*f(x) - sinh(3*x),
        'sol': [Eq(f(x), C1*exp(-7*x) + C2*exp(7*x) - sinh(3*x)/40)],
    },

    'undet_32': {
        'eq': f(x).diff(x, 3) - f(x).diff(x) - sinh(x) - exp(x),
        'sol': [Eq(f(x), C1 + C3*exp(-x) + x*sinh(x)/2 + (C2 + x/2)*exp(x))],
    },

    # https://github.com/sympy/sympy/issues/5096
    'undet_33': {
        'eq': f(x).diff(x, x) + f(x) - x*sin(x - 2),
        'sol': [Eq(f(x), C1*sin(x) + C2*cos(x) - x**2*cos(x - 2)/4 + x*sin(x - 2)/4)],
    },

    'undet_34': {
        'eq': f(x).diff(x, 2) + f(x) - x**4*sin(x-1),
        'sol': [ Eq(f(x), C1*sin(x) + C2*cos(x) - x**5*cos(x - 1)/10 + x**4*sin(x - 1)/4 + x**3*cos(x - 1)/2 - 3*x**2*sin(x - 1)/4 - 3*x*cos(x - 1)/4)],
    },

    'undet_35': {
        'eq': f(x).diff(x, 2) - f(x) - exp(x - 1),
        'sol': [Eq(f(x), C2*exp(-x) + (C1 + x*exp(-1)/2)*exp(x))],
    },

    'undet_36': {
        'eq': f(x).diff(x, 2)+f(x)-(sin(x-2)+1),
        'sol': [Eq(f(x), C1*sin(x) + C2*cos(x) - x*cos(x - 2)/2 + 1)],
    },

    # Equivalent to example_name 'undet_26'.
    # This previously failed because the algorithm for undetermined coefficients
    # didn't know to multiply exp(I*x) by sufficient x because it is linearly
    # dependent on sin(x) and cos(x).
    'undet_37': {
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - 2*x - exp(I*x),
        'sol': [Eq(f(x), C1 + x**2*(I*exp(I*x)/8 + 1) + (C2 + C3*x)*sin(x) + (C4 + C5*x)*cos(x))],
    },

    # https://github.com/sympy/sympy/issues/12623
    'undet_38': {
        'eq': Eq( u(t).diff(t,t) + R /L*u(t).diff(t) + 1/(L*C)*u(t), alpha),
        'sol': [Eq(u(t), C*L*alpha + C2*exp(-t*(R + sqrt(C*R**2 - 4*L)/sqrt(C))/(2*L))
        + C1*exp(t*(-R + sqrt(C*R**2 - 4*L)/sqrt(C))/(2*L)))],
        'func': u(t)
    },

    'undet_39': {
        'eq': Eq( L*C*u(t).diff(t,t) + R*C*u(t).diff(t) + u(t), E_0*exp(I*omega*t) ),
        'sol': [Eq(u(t), C2*exp(-t*(R + sqrt(C*R**2 - 4*L)/sqrt(C))/(2*L))
        + C1*exp(t*(-R + sqrt(C*R**2 - 4*L)/sqrt(C))/(2*L))
        - E_0*exp(I*omega*t)/(C*L*omega**2 - I*C*R*omega - 1))],
        'func': u(t),
    },

    # https://github.com/sympy/sympy/issues/6879
    'undet_40': {
        'eq': Eq(Derivative(f(x), x, 2) - 2*Derivative(f(x), x) + f(x), sin(x)),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(x) + cos(x)/2)],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_separable():
    # test_separable1-5 are from Ordinary Differential Equations, Tenenbaum and
    # Pollard, pg. 55
    t,a = symbols('a,t')
    m = 96
    g = 9.8
    k = .2
    f1 = g * m
    v = Function('v')
    return {
            'hint': "separable",
            'func': f(x),
            'examples':{
    'separable_01': {
        'eq': f(x).diff(x) - f(x),
        'sol': [Eq(f(x), C1*exp(x))],
    },

    'separable_02': {
        'eq': x*f(x).diff(x) - f(x),
        'sol': [Eq(f(x), C1*x)],
    },

    'separable_03': {
        'eq': f(x).diff(x) + sin(x),
        'sol': [Eq(f(x), C1 + cos(x))],
    },

    'separable_04': {
        'eq': f(x)**2 + 1 - (x**2 + 1)*f(x).diff(x),
        'sol': [Eq(f(x), tan(C1 + atan(x)))],
    },

    'separable_05': {
        'eq': f(x).diff(x)/tan(x) - f(x) - 2,
        'sol': [Eq(f(x), C1/cos(x) - 2)],
    },

    'separable_06': {
        'eq': f(x).diff(x) * (1 - sin(f(x))) - 1,
        'sol': [Eq(-x + f(x) + cos(f(x)), C1)],
    },

    'separable_07': {
        'eq': f(x)*x**2*f(x).diff(x) - f(x)**3 - 2*x**2*f(x).diff(x),
        'sol': [Eq(f(x), (-x - sqrt(x*(4*C1*x + x - 4)))/(C1*x - 1)/2),
                Eq(f(x), (-x + sqrt(x*(4*C1*x + x - 4)))/(C1*x - 1)/2)],
        'slow': True,
    },

    'separable_08': {
        'eq': f(x)**2 - 1 - (2*f(x) + x*f(x))*f(x).diff(x),
        'sol': [Eq(f(x), -sqrt(C1*x**2 + 4*C1*x + 4*C1 + 1)),
        Eq(f(x), sqrt(C1*x**2 + 4*C1*x + 4*C1 + 1))],
        'slow': True,
    },

    'separable_09': {
        'eq': x*log(x)*f(x).diff(x) + sqrt(1 + f(x)**2),
        'sol': [Eq(f(x), sinh(C1 - log(log(x))))],  #One more solution is f(x)=I
        'slow': True,
        'checkodesol_XFAIL': True,
    },

    'separable_10': {
        'eq': exp(x + 1)*tan(f(x)) + cos(f(x))*f(x).diff(x),
        'sol': [Eq(E*exp(x) + log(cos(f(x)) - 1)/2 - log(cos(f(x)) + 1)/2 + cos(f(x)), C1)],
        'slow': True,
    },

    'separable_11': {
        'eq': (x*cos(f(x)) + x**2*sin(f(x))*f(x).diff(x) - a**2*sin(f(x))*f(x).diff(x)),
        'sol': [
            Eq(f(x), -acos(C1*sqrt(-a**2 + x**2)) + 2*pi),
            Eq(f(x), acos(C1*sqrt(-a**2 + x**2)))
        ],
        'slow': True,
    },

    'separable_12': {
        'eq': f(x).diff(x) - f(x)*tan(x),
        'sol': [Eq(f(x), C1/cos(x))],
    },

    'separable_13': {
        'eq': (x - 1)*cos(f(x))*f(x).diff(x) - 2*x*sin(f(x)),
        'sol': [
            Eq(f(x), pi - asin(C1*(x**2 - 2*x + 1)*exp(2*x))),
            Eq(f(x), asin(C1*(x**2 - 2*x + 1)*exp(2*x)))
        ],
    },

    'separable_14': {
        'eq': f(x).diff(x) - f(x)*log(f(x))/tan(x),
        'sol': [Eq(f(x), exp(C1*sin(x)))],
    },

    'separable_15': {
        'eq': x*f(x).diff(x) + (1 + f(x)**2)*atan(f(x)),
        'sol': [Eq(f(x), tan(C1/x))],  #Two more solutions are f(x)=0 and f(x)=I
        'slow': True,
        'checkodesol_XFAIL': True,
    },

    'separable_16': {
        'eq': f(x).diff(x) + x*(f(x) + 1),
        'sol': [Eq(f(x), -1 + C1*exp(-x**2/2))],
    },

    'separable_17': {
        'eq': exp(f(x)**2)*(x**2 + 2*x + 1) + (x*f(x) + f(x))*f(x).diff(x),
        'sol': [
            Eq(f(x), -sqrt(log(1/(C1 + x**2 + 2*x)))),
            Eq(f(x), sqrt(log(1/(C1 + x**2 + 2*x))))
        ],
    },

    'separable_18': {
        'eq': f(x).diff(x) + f(x),
        'sol': [Eq(f(x), C1*exp(-x))],
    },

    'separable_19': {
        'eq': sin(x)*cos(2*f(x)) + cos(x)*sin(2*f(x))*f(x).diff(x),
        'sol': [Eq(f(x), pi - acos(C1/cos(x)**2)/2), Eq(f(x), acos(C1/cos(x)**2)/2)],
    },

    'separable_20': {
        'eq': (1 - x)*f(x).diff(x) - x*(f(x) + 1),
        'sol': [Eq(f(x), (C1*exp(-x) - x + 1)/(x - 1))],
    },

    'separable_21': {
        'eq': f(x)*diff(f(x), x) + x - 3*x*f(x)**2,
        'sol': [Eq(f(x), -sqrt(3)*sqrt(C1*exp(3*x**2) + 1)/3),
        Eq(f(x), sqrt(3)*sqrt(C1*exp(3*x**2) + 1)/3)],
    },

    'separable_22': {
        'eq': f(x).diff(x) - exp(x + f(x)),
        'sol': [Eq(f(x), log(-1/(C1 + exp(x))))],
        'XFAIL': ['lie_group'] #It shows 'NoneType' object is not subscriptable for lie_group.
    },

    # https://github.com/sympy/sympy/issues/7081
    'separable_23': {
        'eq': x*(f(x).diff(x)) + 1 - f(x)**2,
        'sol': [Eq(f(x), (-C1 - x**2)/(-C1 + x**2))],
    },

    # https://github.com/sympy/sympy/issues/10379
    'separable_24': {
        'eq': f(t).diff(t)-(1-51.05*y*f(t)),
        'sol': [Eq(f(t), (0.019588638589618023*exp(y*(C1 - 51.049999999999997*t)) + 0.019588638589618023)/y)],
        'func': f(t),
    },

    # https://github.com/sympy/sympy/issues/15999
    'separable_25': {
        'eq': f(x).diff(x) - C1*f(x),
        'sol': [Eq(f(x), C2*exp(C1*x))],
    },

    'separable_26': {
        'eq': f1 - k * (v(t) ** 2) - m * Derivative(v(t)),
        'sol': [Eq(v(t), -68.585712797928991/tanh(C1 - 0.14288690166235204*t))],
        'func': v(t),
        'checkodesol_XFAIL': True,
    },

    #https://github.com/sympy/sympy/issues/22155
    'separable_27': {
        'eq': f(x).diff(x) - exp(f(x) - x),
        'sol': [Eq(f(x), log(-exp(x)/(C1*exp(x) - 1)))],
    }
    }
    }


@_add_example_keys
def _get_examples_ode_sol_1st_exact():
    # Type: Exact differential equation, p(x,f) + q(x,f)*f' == 0,
    # where dp/df == dq/dx
    '''
    Example 7 is an exact equation that fails under the exact engine. It is caught
    by first order homogeneous albeit with a much contorted solution.  The
    exact engine fails because of a poorly simplified integral of q(0,y)dy,
    where q is the function multiplying f'.  The solutions should be
    Eq(sqrt(x**2+f(x)**2)**3+y**3, C1).  The equation below is
    equivalent, but it is so complex that checkodesol fails, and takes a long
    time to do so.
    '''
    return {
            'hint': "1st_exact",
            'func': f(x),
            'examples':{
    '1st_exact_01': {
        'eq': sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x),
        'sol': [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))],
        'slow': True,
    },

    '1st_exact_02': {
        'eq': (2*x*f(x) + 1)/f(x) + (f(x) - x)/f(x)**2*f(x).diff(x),
        'sol': [Eq(f(x), exp(C1 - x**2 + LambertW(-x*exp(-C1 + x**2))))],
        'XFAIL': ['lie_group'], #It shows dsolve raises an exception: List index out of range for lie_group
        'slow': True,
        'checkodesol_XFAIL':True
    },

    '1st_exact_03': {
        'eq': 2*x + f(x)*cos(x) + (2*f(x) + sin(x) - sin(f(x)))*f(x).diff(x),
        'sol': [Eq(f(x)*sin(x) + cos(f(x)) + x**2 + f(x)**2, C1)],
        'XFAIL': ['lie_group'], #It goes into infinite loop for lie_group.
        'slow': True,
    },

    '1st_exact_04': {
        'eq': cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x),
        'sol': [Eq(x*cos(f(x)) + f(x)**3/3, C1)],
        'slow': True,
    },

    '1st_exact_05': {
        'eq': 2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x),
        'sol': [Eq(x**2*f(x) + f(x)**3/3, C1)],
        'slow': True,
        'simplify_flag':False
    },

    # This was from issue: https://github.com/sympy/sympy/issues/11290
    '1st_exact_06': {
        'eq': cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x),
        'sol': [Eq(x*cos(f(x)) + f(x)**3/3, C1)],
        'simplify_flag':False
    },

    '1st_exact_07': {
        'eq': x*sqrt(x**2 + f(x)**2) - (x**2*f(x)/(f(x) - sqrt(x**2 + f(x)**2)))*f(x).diff(x),
        'sol': [Eq(log(x),
        C1 - 9*sqrt(1 + f(x)**2/x**2)*asinh(f(x)/x)/(-27*f(x)/x +
        27*sqrt(1 + f(x)**2/x**2)) - 9*sqrt(1 + f(x)**2/x**2)*
        log(1 - sqrt(1 + f(x)**2/x**2)*f(x)/x + 2*f(x)**2/x**2)/
        (-27*f(x)/x + 27*sqrt(1 + f(x)**2/x**2)) +
        9*asinh(f(x)/x)*f(x)/(x*(-27*f(x)/x + 27*sqrt(1 + f(x)**2/x**2))) +
        9*f(x)*log(1 - sqrt(1 + f(x)**2/x**2)*f(x)/x + 2*f(x)**2/x**2)/
        (x*(-27*f(x)/x + 27*sqrt(1 + f(x)**2/x**2))))],
        'slow': True,
        'dsolve_too_slow':True
    },

    # Type: a(x)f'(x)+b(x)*f(x)+c(x)=0
    '1st_exact_08': {
        'eq': Eq(x**2*f(x).diff(x) + 3*x*f(x) - sin(x)/x, 0),
        'sol': [Eq(f(x), (C1 - cos(x))/x**3)],
    },

    # these examples are from test_exact_enhancement
    '1st_exact_09': {
        'eq': f(x)/x**2 + ((f(x)*x - 1)/x)*f(x).diff(x),
        'sol': [Eq(f(x), (i*sqrt(C1*x**2 + 1) + 1)/x) for i in (-1, 1)],
    },

    '1st_exact_10': {
        'eq': (x*f(x) - 1) + f(x).diff(x)*(x**2 - x*f(x)),
        'sol': [Eq(f(x), x - sqrt(C1 + x**2 - 2*log(x))), Eq(f(x), x + sqrt(C1 + x**2 - 2*log(x)))],
    },

    '1st_exact_11': {
        'eq': (x + 2)*sin(f(x)) + f(x).diff(x)*x*cos(f(x)),
        'sol': [Eq(f(x), -asin(C1*exp(-x)/x**2) + pi), Eq(f(x), asin(C1*exp(-x)/x**2))],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_nth_linear_var_of_parameters():
    g = exp(-x)
    f2 = f(x).diff(x, 2)
    c = 3*f(x).diff(x, 3) + 5*f2 + f(x).diff(x) - f(x) - x
    return {
            'hint': "nth_linear_constant_coeff_variation_of_parameters",
            'func': f(x),
            'examples':{
    'var_of_parameters_01': {
        'eq': c - x*g,
        'sol': [Eq(f(x), C3*exp(x/3) - x + (C1 + x*(C2 - x**2/24 - 3*x/32))*exp(-x) - 1)],
        'slow': True,
    },

    'var_of_parameters_02': {
        'eq': c - g,
        'sol': [Eq(f(x), C3*exp(x/3) - x + (C1 + x*(C2 - x/8))*exp(-x) - 1)],
        'slow': True,
    },

    'var_of_parameters_03': {
        'eq': f(x).diff(x) - 1,
        'sol': [Eq(f(x), C1 + x)],
        'slow': True,
    },

    'var_of_parameters_04': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - 4,
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 2)],
        'slow': True,
    },

    'var_of_parameters_05': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - 12*exp(x),
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 2*exp(x))],
        'slow': True,
    },

    'var_of_parameters_06': {
        'eq': f2 - 2*f(x).diff(x) - 8*f(x) - 9*x*exp(x) - 10*exp(-x),
        'sol': [Eq(f(x), -x*exp(x) - 2*exp(-x) + C1*exp(-2*x) + C2*exp(4*x))],
        'slow': True,
    },

    'var_of_parameters_07': {
        'eq': f2 + 2*f(x).diff(x) + f(x) - x**2*exp(-x),
        'sol': [Eq(f(x), (C1 + x*(C2 + x**3/12))*exp(-x))],
        'slow': True,
    },

    'var_of_parameters_08': {
        'eq': f2 - 3*f(x).diff(x) + 2*f(x) - x*exp(-x),
        'sol': [Eq(f(x), C1*exp(x) + C2*exp(2*x) + (6*x + 5)*exp(-x)/36)],
        'slow': True,
    },

    'var_of_parameters_09': {
        'eq': f(x).diff(x, 3) - 3*f2 + 3*f(x).diff(x) - f(x) - exp(x),
        'sol': [Eq(f(x), (C1 + x*(C2 + x*(C3 + x/6)))*exp(x))],
        'slow': True,
    },

    'var_of_parameters_10': {
        'eq': f2 + 2*f(x).diff(x) + f(x) - exp(-x)/x,
        'sol': [Eq(f(x), (C1 + x*(C2 + log(x)))*exp(-x))],
        'slow': True,
    },

    'var_of_parameters_11': {
        'eq': f2 + f(x) - 1/sin(x)*1/cos(x),
        'sol': [Eq(f(x), (C1 + log(sin(x) - 1)/2 - log(sin(x) + 1)/2
        )*cos(x) + (C2 + log(cos(x) - 1)/2 - log(cos(x) + 1)/2)*sin(x))],
        'slow': True,
    },

    'var_of_parameters_12': {
        'eq': f(x).diff(x, 4) - 1/x,
        'sol': [Eq(f(x), C1 + C2*x + C3*x**2 + x**3*(C4 + log(x)/6))],
        'slow': True,
    },

    # These were from issue: https://github.com/sympy/sympy/issues/15996
    'var_of_parameters_13': {
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - 2*x - exp(I*x),
        'sol': [Eq(f(x), C1 + x**2 + (C2 + x*(C3 - x/8 + 3*exp(I*x)/2 + 3*exp(-I*x)/2) + 5*exp(2*I*x)/16 + 2*I*exp(I*x) - 2*I*exp(-I*x))*sin(x) + (C4 + x*(C5 + I*x/8 + 3*I*exp(I*x)/2 - 3*I*exp(-I*x)/2)
        + 5*I*exp(2*I*x)/16 - 2*exp(I*x) - 2*exp(-I*x))*cos(x) - I*exp(I*x))],
    },

    'var_of_parameters_14': {
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - exp(I*x),
        'sol': [Eq(f(x), C1 + (C2 + x*(C3 - x/8) + 5*exp(2*I*x)/16)*sin(x) + (C4 + x*(C5 + I*x/8) + 5*I*exp(2*I*x)/16)*cos(x) - I*exp(I*x))],
    },

    # https://github.com/sympy/sympy/issues/14395
    'var_of_parameters_15': {
        'eq': Derivative(f(x), x, x) + 9*f(x) - sec(x),
        'sol': [Eq(f(x), (C1 - x/3 + sin(2*x)/3)*sin(3*x) + (C2 + log(cos(x))
        - 2*log(cos(x)**2)/3 + 2*cos(x)**2/3)*cos(3*x))],
        'slow': True,
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_2nd_linear_bessel():
    return {
            'hint': "2nd_linear_bessel",
            'func': f(x),
            'examples':{
    '2nd_lin_bessel_01': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2 - 4)*f(x),
        'sol': [Eq(f(x), C1*besselj(2, x) + C2*bessely(2, x))],
    },

    '2nd_lin_bessel_02': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2 +25)*f(x),
        'sol': [Eq(f(x), C1*besselj(5*I, x) + C2*bessely(5*I, x))],
    },

    '2nd_lin_bessel_03': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2)*f(x),
        'sol': [Eq(f(x), C1*besselj(0, x) + C2*bessely(0, x))],
    },

    '2nd_lin_bessel_04': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (81*x**2 -S(1)/9)*f(x),
        'sol': [Eq(f(x), C1*besselj(S(1)/3, 9*x) + C2*bessely(S(1)/3, 9*x))],
    },

    '2nd_lin_bessel_05': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**4 - 4)*f(x),
        'sol': [Eq(f(x), C1*besselj(1, x**2/2) + C2*bessely(1, x**2/2))],
    },

    '2nd_lin_bessel_06': {
        'eq': x**2*(f(x).diff(x, 2)) + 2*x*(f(x).diff(x)) + (x**4 - 4)*f(x),
        'sol': [Eq(f(x), (C1*besselj(sqrt(17)/4, x**2/2) + C2*bessely(sqrt(17)/4, x**2/2))/sqrt(x))],
    },

    '2nd_lin_bessel_07': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2 - S(1)/4)*f(x),
        'sol': [Eq(f(x), C1*besselj(S(1)/2, x) + C2*bessely(S(1)/2, x))],
    },

    '2nd_lin_bessel_08': {
        'eq': x**2*(f(x).diff(x, 2)) - 3*x*(f(x).diff(x)) + (4*x + 4)*f(x),
        'sol': [Eq(f(x), x**2*(C1*besselj(0, 4*sqrt(x)) + C2*bessely(0, 4*sqrt(x))))],
    },

    '2nd_lin_bessel_09': {
        'eq': x*(f(x).diff(x, 2)) - f(x).diff(x) + 4*x**3*f(x),
        'sol': [Eq(f(x), x*(C1*besselj(S(1)/2, x**2) + C2*bessely(S(1)/2, x**2)))],
    },

    '2nd_lin_bessel_10': {
        'eq': (x-2)**2*(f(x).diff(x, 2)) - (x-2)*f(x).diff(x) + 4*(x-2)**2*f(x),
        'sol': [Eq(f(x), (x - 2)*(C1*besselj(1, 2*x - 4) + C2*bessely(1, 2*x - 4)))],
    },

    # https://github.com/sympy/sympy/issues/4414
    '2nd_lin_bessel_11': {
        'eq': f(x).diff(x, x) + 2/x*f(x).diff(x) + f(x),
        'sol': [Eq(f(x), (C1*besselj(S(1)/2, x) + C2*bessely(S(1)/2, x))/sqrt(x))],
    },
    '2nd_lin_bessel_12': {
        'eq': x**2*f(x).diff(x, 2) + x*f(x).diff(x) + (a**2*x**2/c**2 - b**2)*f(x),
        'sol': [Eq(f(x), C1*besselj(sqrt(b**2), x*sqrt(a**2/c**2)) + C2*bessely(sqrt(b**2), x*sqrt(a**2/c**2)))],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_2nd_2F1_hypergeometric():
    return {
            'hint': "2nd_hypergeometric",
            'func': f(x),
            'examples':{
    '2nd_2F1_hyper_01': {
        'eq': x*(x-1)*f(x).diff(x, 2) + (S(3)/2 -2*x)*f(x).diff(x) + 2*f(x),
        'sol': [Eq(f(x), C1*x**(S(5)/2)*hyper((S(3)/2, S(1)/2), (S(7)/2,), x) + C2*hyper((-1, -2), (-S(3)/2,), x))],
    },

    '2nd_2F1_hyper_02': {
        'eq': x*(x-1)*f(x).diff(x, 2) + (S(7)/2*x)*f(x).diff(x) + f(x),
        'sol': [Eq(f(x), (C1*(1 - x)**(S(5)/2)*hyper((S(1)/2, 2), (S(7)/2,), 1 - x) +
          C2*hyper((-S(1)/2, -2), (-S(3)/2,), 1 - x))/(x - 1)**(S(5)/2))],
    },

    '2nd_2F1_hyper_03': {
        'eq': x*(x-1)*f(x).diff(x, 2) + (S(3)+ S(7)/2*x)*f(x).diff(x) + f(x),
        'sol': [Eq(f(x), (C1*(1 - x)**(S(11)/2)*hyper((S(1)/2, 2), (S(13)/2,), 1 - x) +
          C2*hyper((-S(7)/2, -5), (-S(9)/2,), 1 - x))/(x - 1)**(S(11)/2))],
    },

    '2nd_2F1_hyper_04': {
        'eq': -x**(S(5)/7)*(-416*x**(S(9)/7)/9 - 2385*x**(S(5)/7)/49 + S(298)*x/3)*f(x)/(196*(-x**(S(6)/7) +
         x)**2*(x**(S(6)/7) + x)**2) + Derivative(f(x), (x, 2)),
        'sol': [Eq(f(x), x**(S(45)/98)*(C1*x**(S(4)/49)*hyper((S(1)/3, -S(1)/2), (S(9)/7,), x**(S(2)/7)) +
          C2*hyper((S(1)/21, -S(11)/14), (S(5)/7,), x**(S(2)/7)))/(x**(S(2)/7) - 1)**(S(19)/84))],
        'checkodesol_XFAIL':True,
    },
    }
    }

@_add_example_keys
def _get_examples_ode_sol_2nd_nonlinear_autonomous_conserved():
    return {
            'hint': "2nd_nonlinear_autonomous_conserved",
            'func': f(x),
            'examples': {
    '2nd_nonlinear_autonomous_conserved_01': {
        'eq': f(x).diff(x, 2) + exp(f(x)) + log(f(x)),
        'sol': [
            Eq(Integral(1/sqrt(C1 - 2*_u*log(_u) + 2*_u - 2*exp(_u)), (_u, f(x))), C2 + x),
            Eq(Integral(1/sqrt(C1 - 2*_u*log(_u) + 2*_u - 2*exp(_u)), (_u, f(x))), C2 - x)
        ],
        'simplify_flag': False,
    },
    '2nd_nonlinear_autonomous_conserved_02': {
        'eq': f(x).diff(x, 2) + cbrt(f(x)) + 1/f(x),
        'sol': [
            Eq(sqrt(2)*Integral(1/sqrt(2*C1 - 3*_u**Rational(4, 3) - 4*log(_u)), (_u, f(x))), C2 + x),
            Eq(sqrt(2)*Integral(1/sqrt(2*C1 - 3*_u**Rational(4, 3) - 4*log(_u)), (_u, f(x))), C2 - x)
        ],
        'simplify_flag': False,
    },
    '2nd_nonlinear_autonomous_conserved_03': {
        'eq': f(x).diff(x, 2) + sin(f(x)),
        'sol': [
            Eq(Integral(1/sqrt(C1 + 2*cos(_u)), (_u, f(x))), C2 + x),
            Eq(Integral(1/sqrt(C1 + 2*cos(_u)), (_u, f(x))), C2 - x)
        ],
        'simplify_flag': False,
    },
    '2nd_nonlinear_autonomous_conserved_04': {
        'eq': f(x).diff(x, 2) + cosh(f(x)),
        'sol': [
            Eq(Integral(1/sqrt(C1 - 2*sinh(_u)), (_u, f(x))), C2 + x),
            Eq(Integral(1/sqrt(C1 - 2*sinh(_u)), (_u, f(x))), C2 - x)
        ],
        'simplify_flag': False,
    },
    '2nd_nonlinear_autonomous_conserved_05': {
        'eq': f(x).diff(x, 2) + asin(f(x)),
        'sol': [
            Eq(Integral(1/sqrt(C1 - 2*_u*asin(_u) - 2*sqrt(1 - _u**2)), (_u, f(x))), C2 + x),
            Eq(Integral(1/sqrt(C1 - 2*_u*asin(_u) - 2*sqrt(1 - _u**2)), (_u, f(x))), C2 - x)
        ],
        'simplify_flag': False,
        'XFAIL': ['2nd_nonlinear_autonomous_conserved_Integral']
    }
    }
    }


@_add_example_keys
def _get_examples_ode_sol_separable_reduced():
    df = f(x).diff(x)
    return {
            'hint': "separable_reduced",
            'func': f(x),
            'examples':{
    'separable_reduced_01': {
        'eq': x* df  + f(x)* (1 / (x**2*f(x) - 1)),
        'sol': [Eq(log(x**2*f(x))/3 + log(x**2*f(x) - Rational(3, 2))/6, C1 + log(x))],
        'simplify_flag': False,
        'XFAIL': ['lie_group'], #It hangs.
    },

    #Note: 'separable_reduced_02' is referred in 'separable_reduced_11'
    'separable_reduced_02': {
        'eq': f(x).diff(x) + (f(x) / (x**4*f(x) - x)),
        'sol': [Eq(log(x**3*f(x))/4 + log(x**3*f(x) - Rational(4,3))/12, C1 + log(x))],
        'simplify_flag': False,
        'checkodesol_XFAIL':True, #It hangs for this.
    },

    'separable_reduced_03': {
        'eq': x*df + f(x)*(x**2*f(x)),
        'sol': [Eq(log(x**2*f(x))/2 - log(x**2*f(x) - 2)/2, C1 + log(x))],
        'simplify_flag': False,
    },

    'separable_reduced_04': {
        'eq': Eq(f(x).diff(x) + f(x)/x * (1 + (x**(S(2)/3)*f(x))**2), 0),
        'sol': [Eq(-3*log(x**(S(2)/3)*f(x)) + 3*log(3*x**(S(4)/3)*f(x)**2 + 1)/2, C1 + log(x))],
        'simplify_flag': False,
    },

    'separable_reduced_05': {
        'eq': Eq(f(x).diff(x) + f(x)/x * (1 + (x*f(x))**2), 0),
        'sol': [Eq(f(x), -sqrt(2)*sqrt(1/(C1 + log(x)))/(2*x)),\
                   Eq(f(x), sqrt(2)*sqrt(1/(C1 + log(x)))/(2*x))],
    },

    'separable_reduced_06': {
        'eq': Eq(f(x).diff(x) + (x**4*f(x)**2 + x**2*f(x))*f(x)/(x*(x**6*f(x)**3 + x**4*f(x)**2)), 0),
        'sol': [Eq(f(x), C1 + 1/(2*x**2))],
    },

    'separable_reduced_07': {
        'eq': Eq(f(x).diff(x) + (f(x)**2)*f(x)/(x), 0),
        'sol': [
            Eq(f(x), -sqrt(2)*sqrt(1/(C1 + log(x)))/2),
            Eq(f(x), sqrt(2)*sqrt(1/(C1 + log(x)))/2)
        ],
    },

    'separable_reduced_08': {
        'eq': Eq(f(x).diff(x) + (f(x)+3)*f(x)/(x*(f(x)+2)), 0),
        'sol': [Eq(-log(f(x) + 3)/3 - 2*log(f(x))/3, C1 + log(x))],
        'simplify_flag': False,
        'XFAIL': ['lie_group'], #It hangs.
    },

    'separable_reduced_09': {
        'eq': Eq(f(x).diff(x) + (f(x)+3)*f(x)/x, 0),
        'sol': [Eq(f(x), 3/(C1*x**3 - 1))],
    },

    'separable_reduced_10': {
        'eq': Eq(f(x).diff(x) + (f(x)**2+f(x))*f(x)/(x), 0),
        'sol': [Eq(- log(x) - log(f(x) + 1) + log(f(x)) + 1/f(x), C1)],
        'XFAIL': ['lie_group'],#No algorithms are implemented to solve equation -C1 + x*(_y + 1)*exp(-1/_y)/_y

    },

    # Equivalent to example_name 'separable_reduced_02'. Only difference is testing with simplify=True
    'separable_reduced_11': {
        'eq': f(x).diff(x) + (f(x) / (x**4*f(x) - x)),
        'sol': [Eq(f(x), -sqrt(2)*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
- 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)/6
- sqrt(2)*sqrt(-3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
+ 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 4/x**6
- 4*sqrt(2)/(x**9*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
- 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)))/6 + 1/(3*x**3)),
Eq(f(x), -sqrt(2)*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
- 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)/6
+ sqrt(2)*sqrt(-3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
+ 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 4/x**6
- 4*sqrt(2)/(x**9*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
- 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)))/6 + 1/(3*x**3)),
Eq(f(x), sqrt(2)*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
- 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)/6
- sqrt(2)*sqrt(-3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
+ 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
+ 4/x**6 + 4*sqrt(2)/(x**9*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
- 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)))/6 + 1/(3*x**3)),
Eq(f(x), sqrt(2)*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
- 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)/6
+ sqrt(2)*sqrt(-3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1)
+ x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 4/x**6 + 4*sqrt(2)/(x**9*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1))
- exp(12*C1)/x**6)**Rational(1,3) - 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)))/6 + 1/(3*x**3))],
        'checkodesol_XFAIL':True, #It hangs for this.
        'slow': True,
    },

    #These were from issue: https://github.com/sympy/sympy/issues/6247
    'separable_reduced_12': {
        'eq': x**2*f(x)**2 + x*Derivative(f(x), x),
        'sol': [Eq(f(x), 2*C1/(C1*x**2 - 1))],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_lie_group():
    a, b, c = symbols("a b c")
    return {
            'hint': "lie_group",
            'func': f(x),
            'examples':{
    #Example 1-4 and 19-20 were from issue: https://github.com/sympy/sympy/issues/17322
    'lie_group_01': {
        'eq': x*f(x).diff(x)*(f(x)+4) + (f(x)**2) -2*f(x)-2*x,
        'sol': [],
        'dsolve_too_slow': True,
        'checkodesol_too_slow': True,
    },

    'lie_group_02': {
        'eq': x*f(x).diff(x)*(f(x)+4) + (f(x)**2) -2*f(x)-2*x,
        'sol': [],
        'dsolve_too_slow': True,
    },

    'lie_group_03': {
        'eq': Eq(x**7*Derivative(f(x), x) + 5*x**3*f(x)**2 - (2*x**2 + 2)*f(x)**3, 0),
        'sol': [],
        'dsolve_too_slow': True,
    },

    'lie_group_04': {
        'eq': f(x).diff(x) - (f(x) - x*log(x))**2/x**2 + log(x),
        'sol': [],
        'XFAIL': ['lie_group'],
    },

    'lie_group_05': {
        'eq': f(x).diff(x)**2,
        'sol': [Eq(f(x), C1)],
        'XFAIL': ['factorable'],  #It raises Not Implemented error
    },

    'lie_group_06': {
        'eq': Eq(f(x).diff(x), x**2*f(x)),
        'sol': [Eq(f(x), C1*exp(x**3)**Rational(1, 3))],
    },

    'lie_group_07': {
        'eq': f(x).diff(x) + a*f(x) - c*exp(b*x),
        'sol': [Eq(f(x), Piecewise(((-C1*(a + b) + c*exp(x*(a + b)))*exp(-a*x)/(a + b),\
        Ne(a, -b)), ((-C1 + c*x)*exp(-a*x), True)))],
    },

    'lie_group_08': {
        'eq': f(x).diff(x) + 2*x*f(x) - x*exp(-x**2),
        'sol': [Eq(f(x), (C1 + x**2/2)*exp(-x**2))],
    },

    'lie_group_09': {
        'eq': (1 + 2*x)*(f(x).diff(x)) + 2 - 4*exp(-f(x)),
        'sol': [Eq(f(x), log(C1/(2*x + 1) + 2))],
    },

    'lie_group_10': {
        'eq': x**2*(f(x).diff(x)) - f(x) + x**2*exp(x - (1/x)),
        'sol': [Eq(f(x), (C1 - exp(x))*exp(-1/x))],
        'XFAIL': ['factorable'], #It raises Recursion Error (maixmum depth exceeded)
    },

    'lie_group_11': {
        'eq': x**2*f(x)**2 + x*Derivative(f(x), x),
        'sol': [Eq(f(x), 2/(C1 + x**2))],
    },

    'lie_group_12': {
        'eq': diff(f(x),x) + 2*x*f(x) - x*exp(-x**2),
        'sol': [Eq(f(x), exp(-x**2)*(C1 + x**2/2))],
    },

    'lie_group_13': {
        'eq': diff(f(x),x) + f(x)*cos(x) - exp(2*x),
        'sol': [Eq(f(x), exp(-sin(x))*(C1 + Integral(exp(2*x)*exp(sin(x)), x)))],
    },

    'lie_group_14': {
        'eq': diff(f(x),x) + f(x)*cos(x) - sin(2*x)/2,
        'sol': [Eq(f(x), C1*exp(-sin(x)) + sin(x) - 1)],
    },

    'lie_group_15': {
        'eq': x*diff(f(x),x) + f(x) - x*sin(x),
        'sol': [Eq(f(x), (C1 - x*cos(x) + sin(x))/x)],
    },

    'lie_group_16': {
        'eq': x*diff(f(x),x) - f(x) - x/log(x),
        'sol': [Eq(f(x), x*(C1 + log(log(x))))],
    },

    'lie_group_17': {
        'eq': (f(x).diff(x)-f(x)) * (f(x).diff(x)+f(x)),
        'sol': [Eq(f(x), C1*exp(x)), Eq(f(x), C1*exp(-x))],
    },

    'lie_group_18': {
        'eq': f(x).diff(x) * (f(x).diff(x) - f(x)),
        'sol': [Eq(f(x), C1*exp(x)), Eq(f(x), C1)],
    },

    'lie_group_19': {
        'eq': (f(x).diff(x)-f(x)) * (f(x).diff(x)+f(x)),
        'sol': [Eq(f(x), C1*exp(-x)), Eq(f(x), C1*exp(x))],
    },

    'lie_group_20': {
        'eq': f(x).diff(x)*(f(x).diff(x)+f(x)),
        'sol': [Eq(f(x), C1), Eq(f(x), C1*exp(-x))],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_2nd_linear_airy():
    return {
            'hint': "2nd_linear_airy",
            'func': f(x),
            'examples':{
    '2nd_lin_airy_01': {
        'eq': f(x).diff(x, 2) - x*f(x),
        'sol': [Eq(f(x), C1*airyai(x) + C2*airybi(x))],
    },

    '2nd_lin_airy_02': {
        'eq': f(x).diff(x, 2) + 2*x*f(x),
        'sol': [Eq(f(x), C1*airyai(-2**(S(1)/3)*x) + C2*airybi(-2**(S(1)/3)*x))],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_nth_linear_constant_coeff_homogeneous():
    # From Exercise 20, in Ordinary Differential Equations,
    #                      Tenenbaum and Pollard, pg. 220
    a = Symbol('a', positive=True)
    k = Symbol('k', real=True)
    r1, r2, r3, r4, r5 = [rootof(x**5 + 11*x - 2, n) for n in range(5)]
    r6, r7, r8, r9, r10 = [rootof(x**5 - 3*x + 1, n) for n in range(5)]
    r11, r12, r13, r14, r15 = [rootof(x**5 - 100*x**3 + 1000*x + 1, n) for n in range(5)]
    r16, r17, r18, r19, r20 = [rootof(x**5 - x**4 + 10, n) for n in range(5)]
    r21, r22, r23, r24, r25 = [rootof(x**5 - x + 1, n) for n in range(5)]
    E = exp(1)
    return {
            'hint': "nth_linear_constant_coeff_homogeneous",
            'func': f(x),
            'examples':{
    'lin_const_coeff_hom_01': {
        'eq': f(x).diff(x, 2) + 2*f(x).diff(x),
        'sol': [Eq(f(x), C1 + C2*exp(-2*x))],
    },

    'lin_const_coeff_hom_02': {
        'eq': f(x).diff(x, 2) - 3*f(x).diff(x) + 2*f(x),
        'sol': [Eq(f(x), (C1 + C2*exp(x))*exp(x))],
    },

    'lin_const_coeff_hom_03': {
        'eq': f(x).diff(x, 2) - f(x),
        'sol': [Eq(f(x), C1*exp(-x) + C2*exp(x))],
    },

    'lin_const_coeff_hom_04': {
        'eq': f(x).diff(x, 3) + f(x).diff(x, 2) - 6*f(x).diff(x),
        'sol': [Eq(f(x), C1 + C2*exp(-3*x) + C3*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_05': {
        'eq': 6*f(x).diff(x, 2) - 11*f(x).diff(x) + 4*f(x),
        'sol': [Eq(f(x), C1*exp(x/2) + C2*exp(x*Rational(4, 3)))],
        'slow': True,
    },

    'lin_const_coeff_hom_06': {
        'eq': Eq(f(x).diff(x, 2) + 2*f(x).diff(x) - f(x), 0),
        'sol': [Eq(f(x), C1*exp(x*(-1 + sqrt(2))) + C2*exp(-x*(sqrt(2) + 1)))],
        'slow': True,
    },

    'lin_const_coeff_hom_07': {
        'eq': diff(f(x), x, 3) + diff(f(x), x, 2) - 10*diff(f(x), x) - 6*f(x),
        'sol': [Eq(f(x), C1*exp(3*x) + C3*exp(-x*(2 + sqrt(2))) + C2*exp(x*(-2 + sqrt(2))))],
        'slow': True,
    },

    'lin_const_coeff_hom_08': {
        'eq': f(x).diff(x, 4) - f(x).diff(x, 3) - 4*f(x).diff(x, 2) + \
        4*f(x).diff(x),
        'sol': [Eq(f(x), C1 + C2*exp(-2*x) + C3*exp(x) + C4*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_09': {
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 3) + f(x).diff(x, 2) - \
        4*f(x).diff(x) - 2*f(x),
        'sol': [Eq(f(x), C3*exp(-x) + C4*exp(x) + (C1*exp(-sqrt(2)*x) + C2*exp(sqrt(2)*x))*exp(-2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_10': {
        'eq': f(x).diff(x, 4) - a**2*f(x),
        'sol': [Eq(f(x), C1*exp(-sqrt(a)*x) + C2*exp(sqrt(a)*x) + C3*sin(sqrt(a)*x) + C4*cos(sqrt(a)*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_11': {
        'eq': f(x).diff(x, 2) - 2*k*f(x).diff(x) - 2*f(x),
        'sol': [Eq(f(x), C1*exp(x*(k - sqrt(k**2 + 2))) + C2*exp(x*(k + sqrt(k**2 + 2))))],
        'slow': True,
    },

    'lin_const_coeff_hom_12': {
        'eq': f(x).diff(x, 2) + 4*k*f(x).diff(x) - 12*k**2*f(x),
        'sol': [Eq(f(x), C1*exp(-6*k*x) + C2*exp(2*k*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_13': {
        'eq': f(x).diff(x, 4),
        'sol': [Eq(f(x), C1 + C2*x + C3*x**2 + C4*x**3)],
        'slow': True,
    },

    'lin_const_coeff_hom_14': {
        'eq': f(x).diff(x, 2) + 4*f(x).diff(x) + 4*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(-2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_15': {
        'eq': 3*f(x).diff(x, 3) + 5*f(x).diff(x, 2) + f(x).diff(x) - f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(-x) + C3*exp(x/3))],
        'slow': True,
    },

    'lin_const_coeff_hom_16': {
        'eq': f(x).diff(x, 3) - 6*f(x).diff(x, 2) + 12*f(x).diff(x) - 8*f(x),
        'sol': [Eq(f(x), (C1 + x*(C2 + C3*x))*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_17': {
        'eq': f(x).diff(x, 2) - 2*a*f(x).diff(x) + a**2*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(a*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_18': {
        'eq': f(x).diff(x, 4) + 3*f(x).diff(x, 3),
        'sol': [Eq(f(x), C1 + C2*x + C3*x**2 + C4*exp(-3*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_19': {
        'eq': f(x).diff(x, 4) - 2*f(x).diff(x, 2),
        'sol': [Eq(f(x), C1 + C2*x + C3*exp(-sqrt(2)*x) + C4*exp(sqrt(2)*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_20': {
        'eq': f(x).diff(x, 4) + 2*f(x).diff(x, 3) - 11*f(x).diff(x, 2) - \
        12*f(x).diff(x) + 36*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(-3*x) + (C3 + C4*x)*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_21': {
        'eq': 36*f(x).diff(x, 4) - 37*f(x).diff(x, 2) + 4*f(x).diff(x) + 5*f(x),
        'sol': [Eq(f(x), C1*exp(-x) + C2*exp(-x/3) + C3*exp(x/2) + C4*exp(x*Rational(5, 6)))],
        'slow': True,
    },

    'lin_const_coeff_hom_22': {
        'eq': f(x).diff(x, 4) - 8*f(x).diff(x, 2) + 16*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(-2*x) + (C3 + C4*x)*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_23': {
        'eq': f(x).diff(x, 2) - 2*f(x).diff(x) + 5*f(x),
        'sol': [Eq(f(x), (C1*sin(2*x) + C2*cos(2*x))*exp(x))],
        'slow': True,
    },

    'lin_const_coeff_hom_24': {
        'eq': f(x).diff(x, 2) - f(x).diff(x) + f(x),
        'sol': [Eq(f(x), (C1*sin(x*sqrt(3)/2) + C2*cos(x*sqrt(3)/2))*exp(x/2))],
        'slow': True,
    },

    'lin_const_coeff_hom_25': {
        'eq': f(x).diff(x, 4) + 5*f(x).diff(x, 2) + 6*f(x),
        'sol': [Eq(f(x),
        C1*sin(sqrt(2)*x) + C2*sin(sqrt(3)*x) + C3*cos(sqrt(2)*x) + C4*cos(sqrt(3)*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_26': {
        'eq': f(x).diff(x, 2) - 4*f(x).diff(x) + 20*f(x),
        'sol': [Eq(f(x), (C1*sin(4*x) + C2*cos(4*x))*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_27': {
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 2) + 4*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*sin(x*sqrt(2)) + (C3 + C4*x)*cos(x*sqrt(2)))],
        'slow': True,
    },

    'lin_const_coeff_hom_28': {
        'eq': f(x).diff(x, 3) + 8*f(x),
        'sol': [Eq(f(x), (C1*sin(x*sqrt(3)) + C2*cos(x*sqrt(3)))*exp(x) + C3*exp(-2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_29': {
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 2),
        'sol': [Eq(f(x), C1 + C2*x + C3*sin(2*x) + C4*cos(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_30': {
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x),
        'sol': [Eq(f(x), C1 + (C2 + C3*x)*sin(x) + (C4 + C5*x)*cos(x))],
        'slow': True,
    },

    'lin_const_coeff_hom_31': {
        'eq': f(x).diff(x, 4) + f(x).diff(x, 2) + f(x),
        'sol': [Eq(f(x), (C1*sin(sqrt(3)*x/2) + C2*cos(sqrt(3)*x/2))*exp(-x/2)
        + (C3*sin(sqrt(3)*x/2) + C4*cos(sqrt(3)*x/2))*exp(x/2))],
        'slow': True,
    },

    'lin_const_coeff_hom_32': {
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 2) + f(x),
        'sol': [Eq(f(x), C1*sin(x*sqrt(-sqrt(3) + 2)) + C2*sin(x*sqrt(sqrt(3) + 2))
        + C3*cos(x*sqrt(-sqrt(3) + 2)) + C4*cos(x*sqrt(sqrt(3) + 2)))],
        'slow': True,
    },

    # One real root, two complex conjugate pairs
    'lin_const_coeff_hom_33': {
        'eq': f(x).diff(x, 5) + 11*f(x).diff(x) - 2*f(x),
        'sol': [Eq(f(x),
        C5*exp(r1*x) + exp(re(r2)*x) * (C1*sin(im(r2)*x) + C2*cos(im(r2)*x))
        + exp(re(r4)*x) * (C3*sin(im(r4)*x) + C4*cos(im(r4)*x)))],
        'checkodesol_XFAIL':True,  #It Hangs
    },

    # Three real roots, one complex conjugate pair
    'lin_const_coeff_hom_34': {
        'eq': f(x).diff(x,5) - 3*f(x).diff(x) + f(x),
        'sol': [Eq(f(x),
        C3*exp(r6*x) + C4*exp(r7*x) + C5*exp(r8*x)
        + exp(re(r9)*x) * (C1*sin(im(r9)*x) + C2*cos(im(r9)*x)))],
        'checkodesol_XFAIL':True, #It Hangs
    },

    # Five distinct real roots
    'lin_const_coeff_hom_35': {
        'eq': f(x).diff(x,5) - 100*f(x).diff(x,3) + 1000*f(x).diff(x) + f(x),
        'sol': [Eq(f(x), C1*exp(r11*x) + C2*exp(r12*x) + C3*exp(r13*x) + C4*exp(r14*x) + C5*exp(r15*x))],
        'checkodesol_XFAIL':True, #It Hangs
    },

    # Rational root and unsolvable quintic
    'lin_const_coeff_hom_36': {
        'eq': f(x).diff(x, 6) - 6*f(x).diff(x, 5) + 5*f(x).diff(x, 4) + 10*f(x).diff(x) - 50 * f(x),
        'sol': [Eq(f(x),
        C5*exp(5*x)
        + C6*exp(x*r16)
        + exp(re(r17)*x) * (C1*sin(im(r17)*x) + C2*cos(im(r17)*x))
        + exp(re(r19)*x) * (C3*sin(im(r19)*x) + C4*cos(im(r19)*x)))],
        'checkodesol_XFAIL':True, #It Hangs
    },

    # Five double roots (this is (x**5 - x + 1)**2)
    'lin_const_coeff_hom_37': {
        'eq': f(x).diff(x, 10) - 2*f(x).diff(x, 6) + 2*f(x).diff(x, 5)
        + f(x).diff(x, 2) - 2*f(x).diff(x, 1) + f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(x*r21) + (-((C3 + C4*x)*sin(x*im(r22)))
        + (C5 + C6*x)*cos(x*im(r22)))*exp(x*re(r22)) + (-((C7 + C8*x)*sin(x*im(r24)))
        + (C10*x + C9)*cos(x*im(r24)))*exp(x*re(r24)))],
        'checkodesol_XFAIL':True, #It Hangs
    },

    'lin_const_coeff_hom_38': {
        'eq': Eq(sqrt(2) * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*sin(2**Rational(3, 4)*x/2) + C3*cos(2**Rational(3, 4)*x/2))],
    },

    'lin_const_coeff_hom_39': {
        'eq': Eq(E * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*sin(x/sqrt(E)) + C3*cos(x/sqrt(E)))],
    },

    'lin_const_coeff_hom_40': {
        'eq': Eq(pi * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*sin(x/sqrt(pi)) + C3*cos(x/sqrt(pi)))],
    },

    'lin_const_coeff_hom_41': {
        'eq': Eq(I * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*exp(-sqrt(I)*x) + C3*exp(sqrt(I)*x))],
    },

    'lin_const_coeff_hom_42': {
        'eq': f(x).diff(x, x) + y*f(x),
        'sol': [Eq(f(x), C1*exp(-x*sqrt(-y)) + C2*exp(x*sqrt(-y)))],
    },

    'lin_const_coeff_hom_43': {
        'eq': Eq(9*f(x).diff(x, x) + f(x), 0),
        'sol': [Eq(f(x), C1*sin(x/3) + C2*cos(x/3))],
    },

    'lin_const_coeff_hom_44': {
        'eq': Eq(9*f(x).diff(x, x), f(x)),
        'sol': [Eq(f(x), C1*exp(-x/3) + C2*exp(x/3))],
    },

    'lin_const_coeff_hom_45': {
        'eq': Eq(f(x).diff(x, x) - 3*diff(f(x), x) + 2*f(x), 0),
        'sol': [Eq(f(x), (C1 + C2*exp(x))*exp(x))],
    },

    'lin_const_coeff_hom_46': {
        'eq': Eq(f(x).diff(x, x) - 4*diff(f(x), x) + 4*f(x), 0),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(2*x))],
    },

    # Type: 2nd order, constant coefficients (two real equal roots)
    'lin_const_coeff_hom_47': {
        'eq': Eq(f(x).diff(x, x) + 2*diff(f(x), x) + 3*f(x), 0),
        'sol': [Eq(f(x), (C1*sin(x*sqrt(2)) + C2*cos(x*sqrt(2)))*exp(-x))],
    },

    #These were from issue: https://github.com/sympy/sympy/issues/6247
    'lin_const_coeff_hom_48': {
        'eq': f(x).diff(x, x) + 4*f(x),
        'sol': [Eq(f(x), C1*sin(2*x) + C2*cos(2*x))],
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep():
    return {
            'hint': "1st_homogeneous_coeff_subs_dep_div_indep",
            'func': f(x),
            'examples':{
    'dep_div_indep_01': {
        'eq': f(x)/x*cos(f(x)/x) - (x/f(x)*sin(f(x)/x) + cos(f(x)/x))*f(x).diff(x),
        'sol': [Eq(log(x), C1 - log(f(x)*sin(f(x)/x)/x))],
        'slow': True
    },

    #indep_div_dep actually has a simpler solution for example 2 but it runs too slow.
    'dep_div_indep_02': {
        'eq': x*f(x).diff(x) - f(x) - x*sin(f(x)/x),
        'sol': [Eq(log(x), log(C1) + log(cos(f(x)/x) - 1)/2 - log(cos(f(x)/x) + 1)/2)],
        'simplify_flag':False,
    },

    'dep_div_indep_03': {
        'eq': x*exp(f(x)/x) - f(x)*sin(f(x)/x) + x*sin(f(x)/x)*f(x).diff(x),
        'sol': [Eq(log(x), C1 + exp(-f(x)/x)*sin(f(x)/x)/2 + exp(-f(x)/x)*cos(f(x)/x)/2)],
        'slow': True
    },

    'dep_div_indep_04': {
        'eq': f(x).diff(x) - f(x)/x + 1/sin(f(x)/x),
        'sol': [Eq(f(x), x*(-acos(C1 + log(x)) + 2*pi)), Eq(f(x), x*acos(C1 + log(x)))],
        'slow': True
    },

    # previous code was testing with these other solution:
    # example5_solb = Eq(f(x), log(log(C1/x)**(-x)))
    'dep_div_indep_05': {
        'eq': x*exp(f(x)/x) + f(x) - x*f(x).diff(x),
        'sol': [Eq(f(x), log((1/(C1 - log(x)))**x))],
        'checkodesol_XFAIL':True, #(because of **x?)
    },
    }
    }

@_add_example_keys
def _get_examples_ode_sol_linear_coefficients():
    return {
            'hint': "linear_coefficients",
            'func': f(x),
            'examples':{
    'linear_coeff_01': {
        'eq': f(x).diff(x) + (3 + 2*f(x))/(x + 3),
        'sol': [Eq(f(x), C1/(x**2 + 6*x + 9) - Rational(3, 2))],
    },
    }
    }

@_add_example_keys
def _get_examples_ode_sol_1st_homogeneous_coeff_best():
    return {
            'hint': "1st_homogeneous_coeff_best",
            'func': f(x),
            'examples':{
    # previous code was testing this with other solution:
    # example1_solb = Eq(-f(x)/(1 + log(x/f(x))), C1)
    '1st_homogeneous_coeff_best_01': {
        'eq': f(x) + (x*log(f(x)/x) - 2*x)*diff(f(x), x),
        'sol': [Eq(f(x), -exp(C1)*LambertW(-x*exp(-C1 + 1)))],
        'checkodesol_XFAIL':True, #(because of LambertW?)
    },

    '1st_homogeneous_coeff_best_02': {
        'eq': 2*f(x)*exp(x/f(x)) + f(x)*f(x).diff(x) - 2*x*exp(x/f(x))*f(x).diff(x),
        'sol': [Eq(log(f(x)), C1 - 2*exp(x/f(x)))],
    },

    # previous code was testing this with other solution:
    # example3_solb = Eq(log(C1*x*sqrt(1/x)*sqrt(f(x))) + x**2/(2*f(x)**2), 0)
    '1st_homogeneous_coeff_best_03': {
        'eq': 2*x**2*f(x) + f(x)**3 + (x*f(x)**2 - 2*x**3)*f(x).diff(x),
        'sol': [Eq(f(x), exp(2*C1 + LambertW(-2*x**4*exp(-4*C1))/2)/x)],
        'checkodesol_XFAIL':True,  #(because of LambertW?)
    },

    '1st_homogeneous_coeff_best_04': {
        'eq': (x + sqrt(f(x)**2 - x*f(x)))*f(x).diff(x) - f(x),
        'sol': [Eq(log(f(x)), C1 - 2*sqrt(-x/f(x) + 1))],
        'slow': True,
    },

    '1st_homogeneous_coeff_best_05': {
        'eq': x + f(x) - (x - f(x))*f(x).diff(x),
        'sol': [Eq(log(x), C1 - log(sqrt(1 + f(x)**2/x**2)) + atan(f(x)/x))],
    },

    '1st_homogeneous_coeff_best_06': {
        'eq': x*f(x).diff(x) - f(x) - x*sin(f(x)/x),
        'sol': [Eq(f(x), 2*x*atan(C1*x))],
    },

    '1st_homogeneous_coeff_best_07': {
        'eq': x**2 + f(x)**2 - 2*x*f(x)*f(x).diff(x),
        'sol': [Eq(f(x), -sqrt(x*(C1 + x))), Eq(f(x), sqrt(x*(C1 + x)))],
    },

    '1st_homogeneous_coeff_best_08': {
        'eq': f(x)**2 + (x*sqrt(f(x)**2 - x**2) - x*f(x))*f(x).diff(x),
        'sol': [Eq(f(x), -C1*sqrt(-x/(x - 2*C1))), Eq(f(x), C1*sqrt(-x/(x - 2*C1)))],
        'checkodesol_XFAIL': True  # solutions are valid in a range
    },
    }
    }


def _get_all_examples():
    all_examples = _get_examples_ode_sol_euler_homogeneous + \
    _get_examples_ode_sol_euler_undetermined_coeff + \
    _get_examples_ode_sol_euler_var_para + \
    _get_examples_ode_sol_factorable + \
    _get_examples_ode_sol_bernoulli + \
    _get_examples_ode_sol_nth_algebraic + \
    _get_examples_ode_sol_riccati + \
    _get_examples_ode_sol_1st_linear + \
    _get_examples_ode_sol_1st_exact + \
    _get_examples_ode_sol_almost_linear + \
    _get_examples_ode_sol_nth_order_reducible + \
    _get_examples_ode_sol_nth_linear_undetermined_coefficients + \
    _get_examples_ode_sol_liouville + \
    _get_examples_ode_sol_separable + \
    _get_examples_ode_sol_1st_rational_riccati + \
    _get_examples_ode_sol_nth_linear_var_of_parameters + \
    _get_examples_ode_sol_2nd_linear_bessel + \
    _get_examples_ode_sol_2nd_2F1_hypergeometric + \
    _get_examples_ode_sol_2nd_nonlinear_autonomous_conserved + \
    _get_examples_ode_sol_separable_reduced + \
    _get_examples_ode_sol_lie_group + \
    _get_examples_ode_sol_2nd_linear_airy + \
    _get_examples_ode_sol_nth_linear_constant_coeff_homogeneous +\
    _get_examples_ode_sol_1st_homogeneous_coeff_best +\
    _get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep +\
    _get_examples_ode_sol_linear_coefficients

    return all_examples