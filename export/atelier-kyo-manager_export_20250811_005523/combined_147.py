
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_path.py ===
from __future__ import annotations

import os
import pathlib
import sys
from functools import partial, update_wrapper
from inspect import cleandoc
from typing import IO, TYPE_CHECKING, Any, BinaryIO, ClassVar, TypeVar, overload

from trio._file_io import AsyncIOWrapper, wrap_file
from trio._util import final
from trio.to_thread import run_sync

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from io import BufferedRandom, BufferedReader, BufferedWriter, FileIO, TextIOWrapper

    from _typeshed import (
        OpenBinaryMode,
        OpenBinaryModeReading,
        OpenBinaryModeUpdating,
        OpenBinaryModeWriting,
        OpenTextMode,
    )
    from typing_extensions import Concatenate, Literal, ParamSpec, Self

    P = ParamSpec("P")

    PathT = TypeVar("PathT", bound="Path")
    T = TypeVar("T")


def _wraps_async(  # type: ignore[explicit-any]
    wrapped: Callable[..., object],
) -> Callable[[Callable[P, T]], Callable[P, Awaitable[T]]]:
    def decorator(fn: Callable[P, T]) -> Callable[P, Awaitable[T]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await run_sync(partial(fn, *args, **kwargs))

        update_wrapper(wrapper, wrapped)
        if wrapped.__doc__:
            wrapper.__doc__ = (
                f"Like :meth:`~{wrapped.__module__}.{wrapped.__qualname__}`, but async.\n"
                f"\n"
                f"{cleandoc(wrapped.__doc__)}\n"
            )
        return wrapper

    return decorator


def _wrap_method(
    fn: Callable[Concatenate[pathlib.Path, P], T],
) -> Callable[Concatenate[Path, P], Awaitable[T]]:
    @_wraps_async(fn)
    def wrapper(self: Path, /, *args: P.args, **kwargs: P.kwargs) -> T:
        return fn(self._wrapped_cls(self), *args, **kwargs)

    return wrapper


def _wrap_method_path(
    fn: Callable[Concatenate[pathlib.Path, P], pathlib.Path],
) -> Callable[Concatenate[PathT, P], Awaitable[PathT]]:
    @_wraps_async(fn)
    def wrapper(self: PathT, /, *args: P.args, **kwargs: P.kwargs) -> PathT:
        return self.__class__(fn(self._wrapped_cls(self), *args, **kwargs))

    return wrapper


def _wrap_method_path_iterable(
    fn: Callable[Concatenate[pathlib.Path, P], Iterable[pathlib.Path]],
) -> Callable[Concatenate[PathT, P], Awaitable[Iterable[PathT]]]:
    @_wraps_async(fn)
    def wrapper(self: PathT, /, *args: P.args, **kwargs: P.kwargs) -> Iterable[PathT]:
        return map(self.__class__, [*fn(self._wrapped_cls(self), *args, **kwargs)])

    if wrapper.__doc__:
        wrapper.__doc__ += (
            f"\n"
            f"This is an async method that returns a synchronous iterator, so you\n"
            f"use it like:\n"
            f"\n"
            f".. code:: python\n"
            f"\n"
            f"    for subpath in await mypath.{fn.__name__}():\n"
            f"        ...\n"
            f"\n"
            f".. note::\n"
            f"\n"
            f"    The iterator is loaded into memory immediately during the initial\n"
            f"    call (see `issue #501\n"
            f"    <https://github.com/python-trio/trio/issues/501>`__ for discussion).\n"
        )
    return wrapper


class Path(pathlib.PurePath):
    """An async :class:`pathlib.Path` that executes blocking methods in :meth:`trio.to_thread.run_sync`.

    Instantiating :class:`Path` returns a concrete platform-specific subclass, one of :class:`PosixPath` or
    :class:`WindowsPath`.
    """

    __slots__ = ()

    _wrapped_cls: ClassVar[type[pathlib.Path]]

    def __new__(cls, *args: str | os.PathLike[str]) -> Self:
        if cls is Path:
            cls = WindowsPath if os.name == "nt" else PosixPath  # type: ignore[assignment]
        return super().__new__(cls, *args)

    @classmethod
    @_wraps_async(pathlib.Path.cwd)
    def cwd(cls) -> Self:
        return cls(pathlib.Path.cwd())

    @classmethod
    @_wraps_async(pathlib.Path.home)
    def home(cls) -> Self:
        return cls(pathlib.Path.home())

    @overload
    async def open(
        self,
        mode: OpenTextMode = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> AsyncIOWrapper[TextIOWrapper]: ...

    @overload
    async def open(
        self,
        mode: OpenBinaryMode,
        buffering: Literal[0],
        encoding: None = None,
        errors: None = None,
        newline: None = None,
    ) -> AsyncIOWrapper[FileIO]: ...

    @overload
    async def open(
        self,
        mode: OpenBinaryModeUpdating,
        buffering: Literal[-1, 1] = -1,
        encoding: None = None,
        errors: None = None,
        newline: None = None,
    ) -> AsyncIOWrapper[BufferedRandom]: ...

    @overload
    async def open(
        self,
        mode: OpenBinaryModeWriting,
        buffering: Literal[-1, 1] = -1,
        encoding: None = None,
        errors: None = None,
        newline: None = None,
    ) -> AsyncIOWrapper[BufferedWriter]: ...

    @overload
    async def open(
        self,
        mode: OpenBinaryModeReading,
        buffering: Literal[-1, 1] = -1,
        encoding: None = None,
        errors: None = None,
        newline: None = None,
    ) -> AsyncIOWrapper[BufferedReader]: ...

    @overload
    async def open(
        self,
        mode: OpenBinaryMode,
        buffering: int = -1,
        encoding: None = None,
        errors: None = None,
        newline: None = None,
    ) -> AsyncIOWrapper[BinaryIO]: ...

    @overload
    async def open(  # type: ignore[misc, explicit-any]  # Any usage matches builtins.open().
        self,
        mode: str,
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> AsyncIOWrapper[IO[Any]]: ...

    @_wraps_async(pathlib.Path.open)
    def open(self, *args: Any, **kwargs: Any) -> AsyncIOWrapper[IO[Any]]:  # type: ignore[misc, explicit-any]  # Overload return mismatch.
        return wrap_file(self._wrapped_cls(self).open(*args, **kwargs))

    def __repr__(self) -> str:
        return f"trio.Path({str(self)!r})"

    stat = _wrap_method(pathlib.Path.stat)
    chmod = _wrap_method(pathlib.Path.chmod)
    exists = _wrap_method(pathlib.Path.exists)
    glob = _wrap_method_path_iterable(pathlib.Path.glob)
    rglob = _wrap_method_path_iterable(pathlib.Path.rglob)
    is_dir = _wrap_method(pathlib.Path.is_dir)
    is_file = _wrap_method(pathlib.Path.is_file)
    is_symlink = _wrap_method(pathlib.Path.is_symlink)
    is_socket = _wrap_method(pathlib.Path.is_socket)
    is_fifo = _wrap_method(pathlib.Path.is_fifo)
    is_block_device = _wrap_method(pathlib.Path.is_block_device)
    is_char_device = _wrap_method(pathlib.Path.is_char_device)
    if sys.version_info >= (3, 12):
        is_junction = _wrap_method(pathlib.Path.is_junction)
    iterdir = _wrap_method_path_iterable(pathlib.Path.iterdir)
    lchmod = _wrap_method(pathlib.Path.lchmod)
    lstat = _wrap_method(pathlib.Path.lstat)
    mkdir = _wrap_method(pathlib.Path.mkdir)
    if sys.platform != "win32":
        owner = _wrap_method(pathlib.Path.owner)
        group = _wrap_method(pathlib.Path.group)
    if sys.platform != "win32" or sys.version_info >= (3, 12):
        is_mount = _wrap_method(pathlib.Path.is_mount)
    readlink = _wrap_method_path(pathlib.Path.readlink)
    rename = _wrap_method_path(pathlib.Path.rename)
    replace = _wrap_method_path(pathlib.Path.replace)
    resolve = _wrap_method_path(pathlib.Path.resolve)
    rmdir = _wrap_method(pathlib.Path.rmdir)
    symlink_to = _wrap_method(pathlib.Path.symlink_to)
    if sys.version_info >= (3, 10):
        hardlink_to = _wrap_method(pathlib.Path.hardlink_to)
    touch = _wrap_method(pathlib.Path.touch)
    unlink = _wrap_method(pathlib.Path.unlink)
    absolute = _wrap_method_path(pathlib.Path.absolute)
    expanduser = _wrap_method_path(pathlib.Path.expanduser)
    read_bytes = _wrap_method(pathlib.Path.read_bytes)
    read_text = _wrap_method(pathlib.Path.read_text)
    samefile = _wrap_method(pathlib.Path.samefile)
    write_bytes = _wrap_method(pathlib.Path.write_bytes)
    write_text = _wrap_method(pathlib.Path.write_text)
    if sys.version_info < (3, 12):
        link_to = _wrap_method(pathlib.Path.link_to)
    if sys.version_info >= (3, 13):
        full_match = _wrap_method(pathlib.Path.full_match)

    def as_uri(self) -> str:
        return pathlib.Path.as_uri(self)


@final
class PosixPath(Path, pathlib.PurePosixPath):
    """An async :class:`pathlib.PosixPath` that executes blocking methods in :meth:`trio.to_thread.run_sync`."""

    __slots__ = ()

    _wrapped_cls: ClassVar[type[pathlib.Path]] = pathlib.PosixPath


@final
class WindowsPath(Path, pathlib.PureWindowsPath):
    """An async :class:`pathlib.WindowsPath` that executes blocking methods in :meth:`trio.to_thread.run_sync`."""

    __slots__ = ()

    _wrapped_cls: ClassVar[type[pathlib.Path]] = pathlib.WindowsPath

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\hydrogen.py ===
from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.polynomials import assoc_laguerre
from sympy.functions.special.spherical_harmonics import Ynm


def R_nl(n, l, r, Z=1):
    """
    Returns the Hydrogen radial wavefunction R_{nl}.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    r :
        Radial coordinate.
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)

    Everything is in Hartree atomic units.

    Examples
    ========

    >>> from sympy.physics.hydrogen import R_nl
    >>> from sympy.abc import r, Z
    >>> R_nl(1, 0, r, Z)
    2*sqrt(Z**3)*exp(-Z*r)
    >>> R_nl(2, 0, r, Z)
    sqrt(2)*(-Z*r + 2)*sqrt(Z**3)*exp(-Z*r/2)/4
    >>> R_nl(2, 1, r, Z)
    sqrt(6)*Z*r*sqrt(Z**3)*exp(-Z*r/2)/12

    For Hydrogen atom, you can just use the default value of Z=1:

    >>> R_nl(1, 0, r)
    2*exp(-r)
    >>> R_nl(2, 0, r)
    sqrt(2)*(2 - r)*exp(-r/2)/4
    >>> R_nl(3, 0, r)
    2*sqrt(3)*(2*r**2/9 - 2*r + 3)*exp(-r/3)/27

    For Silver atom, you would use Z=47:

    >>> R_nl(1, 0, r, Z=47)
    94*sqrt(47)*exp(-47*r)
    >>> R_nl(2, 0, r, Z=47)
    47*sqrt(94)*(2 - 47*r)*exp(-47*r/2)/4
    >>> R_nl(3, 0, r, Z=47)
    94*sqrt(141)*(4418*r**2/9 - 94*r + 3)*exp(-47*r/3)/27

    The normalization of the radial wavefunction is:

    >>> from sympy import integrate, oo
    >>> integrate(R_nl(1, 0, r)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 0, r)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 1, r)**2 * r**2, (r, 0, oo))
    1

    It holds for any atomic number:

    >>> integrate(R_nl(1, 0, r, Z=2)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 0, r, Z=3)**2 * r**2, (r, 0, oo))
    1
    >>> integrate(R_nl(2, 1, r, Z=4)**2 * r**2, (r, 0, oo))
    1

    """
    # sympify arguments
    n, l, r, Z = map(S, [n, l, r, Z])
    # radial quantum number
    n_r = n - l - 1
    # rescaled "r"
    a = 1/Z  # Bohr radius
    r0 = 2 * r / (n * a)
    # normalization coefficient
    C = sqrt((S(2)/(n*a))**3 * factorial(n_r) / (2*n*factorial(n + l)))
    # This is an equivalent normalization coefficient, that can be found in
    # some books. Both coefficients seem to be the same fast:
    # C =  S(2)/n**2 * sqrt(1/a**3 * factorial(n_r) / (factorial(n+l)))
    return C * r0**l * assoc_laguerre(n_r, 2*l + 1, r0).expand() * exp(-r0/2)


def Psi_nlm(n, l, m, r, phi, theta, Z=1):
    """
    Returns the Hydrogen wave function psi_{nlm}. It's the product of
    the radial wavefunction R_{nl} and the spherical harmonic Y_{l}^{m}.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    m : integer
        ``m`` is the Magnetic Quantum Number with values
        ranging from ``-l`` to ``l``.
    r :
        radial coordinate
    phi :
        azimuthal angle
    theta :
        polar angle
    Z :
        atomic number (1 for Hydrogen, 2 for Helium, ...)

    Everything is in Hartree atomic units.

    Examples
    ========

    >>> from sympy.physics.hydrogen import Psi_nlm
    >>> from sympy import Symbol
    >>> r=Symbol("r", positive=True)
    >>> phi=Symbol("phi", real=True)
    >>> theta=Symbol("theta", real=True)
    >>> Z=Symbol("Z", positive=True, integer=True, nonzero=True)
    >>> Psi_nlm(1,0,0,r,phi,theta,Z)
    Z**(3/2)*exp(-Z*r)/sqrt(pi)
    >>> Psi_nlm(2,1,1,r,phi,theta,Z)
    -Z**(5/2)*r*exp(I*phi)*exp(-Z*r/2)*sin(theta)/(8*sqrt(pi))

    Integrating the absolute square of a hydrogen wavefunction psi_{nlm}
    over the whole space leads 1.

    The normalization of the hydrogen wavefunctions Psi_nlm is:

    >>> from sympy import integrate, conjugate, pi, oo, sin
    >>> wf=Psi_nlm(2,1,1,r,phi,theta,Z)
    >>> abs_sqrd=wf*conjugate(wf)
    >>> jacobi=r**2*sin(theta)
    >>> integrate(abs_sqrd*jacobi, (r,0,oo), (phi,0,2*pi), (theta,0,pi))
    1
    """

    # sympify arguments
    n, l, m, r, phi, theta, Z = map(S, [n, l, m, r, phi, theta, Z])
    # check if values for n,l,m make physically sense
    if n.is_integer and n < 1:
        raise ValueError("'n' must be positive integer")
    if l.is_integer and not (n > l):
        raise ValueError("'n' must be greater than 'l'")
    if m.is_integer and not (abs(m) <= l):
        raise ValueError("|'m'| must be less or equal 'l'")
    # return the hydrogen wave function
    return R_nl(n, l, r, Z)*Ynm(l, m, theta, phi).expand(func=True)


def E_nl(n, Z=1):
    """
    Returns the energy of the state (n, l) in Hartree atomic units.

    The energy does not depend on "l".

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)

    Examples
    ========

    >>> from sympy.physics.hydrogen import E_nl
    >>> from sympy.abc import n, Z
    >>> E_nl(n, Z)
    -Z**2/(2*n**2)
    >>> E_nl(1)
    -1/2
    >>> E_nl(2)
    -1/8
    >>> E_nl(3)
    -1/18
    >>> E_nl(3, 47)
    -2209/18

    """
    n, Z = S(n), S(Z)
    if n.is_integer and (n < 1):
        raise ValueError("'n' must be positive integer")
    return -Z**2/(2*n**2)


def E_nl_dirac(n, l, spin_up=True, Z=1, c=Float("137.035999037")):
    """
    Returns the relativistic energy of the state (n, l, spin) in Hartree atomic
    units.

    The energy is calculated from the Dirac equation. The rest mass energy is
    *not* included.

    Parameters
    ==========

    n : integer
        Principal Quantum Number which is
        an integer with possible values as 1, 2, 3, 4,...
    l : integer
        ``l`` is the Angular Momentum Quantum Number with
        values ranging from 0 to ``n-1``.
    spin_up :
        True if the electron spin is up (default), otherwise down
    Z :
        Atomic number (1 for Hydrogen, 2 for Helium, ...)
    c :
        Speed of light in atomic units. Default value is 137.035999037,
        taken from https://arxiv.org/abs/1012.3627

    Examples
    ========

    >>> from sympy.physics.hydrogen import E_nl_dirac
    >>> E_nl_dirac(1, 0)
    -0.500006656595360

    >>> E_nl_dirac(2, 0)
    -0.125002080189006
    >>> E_nl_dirac(2, 1)
    -0.125000416028342
    >>> E_nl_dirac(2, 1, False)
    -0.125002080189006

    >>> E_nl_dirac(3, 0)
    -0.0555562951740285
    >>> E_nl_dirac(3, 1)
    -0.0555558020932949
    >>> E_nl_dirac(3, 1, False)
    -0.0555562951740285
    >>> E_nl_dirac(3, 2)
    -0.0555556377366884
    >>> E_nl_dirac(3, 2, False)
    -0.0555558020932949

    """
    n, l, Z, c = map(S, [n, l, Z, c])
    if not (l >= 0):
        raise ValueError("'l' must be positive or zero")
    if not (n > l):
        raise ValueError("'n' must be greater than 'l'")
    if (l == 0 and spin_up is False):
        raise ValueError("Spin must be up for l==0.")
    # skappa is sign*kappa, where sign contains the correct sign
    if spin_up:
        skappa = -l - 1
    else:
        skappa = -l
    beta = sqrt(skappa**2 - Z**2/c**2)
    return c**2/sqrt(1 + Z**2/(n + skappa + beta)**2/c**2) - c**2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\util.py ===
"""
Several methods to simplify expressions involving unit objects.
"""
from functools import reduce
from collections.abc import Iterable
from typing import Optional

from sympy import default_sort_key
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.core.function import Function
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.physics.units.dimensions import Dimension, DimensionSystem
from sympy.physics.units.prefixes import Prefix
from sympy.physics.units.quantities import Quantity
from sympy.physics.units.unitsystem import UnitSystem
from sympy.utilities.iterables import sift


def _get_conversion_matrix_for_expr(expr, target_units, unit_system):
    from sympy.matrices.dense import Matrix

    dimension_system = unit_system.get_dimension_system()

    expr_dim = Dimension(unit_system.get_dimensional_expr(expr))
    dim_dependencies = dimension_system.get_dimensional_dependencies(expr_dim, mark_dimensionless=True)
    target_dims = [Dimension(unit_system.get_dimensional_expr(x)) for x in target_units]
    canon_dim_units = [i for x in target_dims for i in dimension_system.get_dimensional_dependencies(x, mark_dimensionless=True)]
    canon_expr_units = set(dim_dependencies)

    if not canon_expr_units.issubset(set(canon_dim_units)):
        return None

    seen = set()
    canon_dim_units = [i for i in canon_dim_units if not (i in seen or seen.add(i))]

    camat = Matrix([[dimension_system.get_dimensional_dependencies(i, mark_dimensionless=True).get(j, 0) for i in target_dims] for j in canon_dim_units])
    exprmat = Matrix([dim_dependencies.get(k, 0) for k in canon_dim_units])

    try:
        res_exponents = camat.solve(exprmat)
    except NonInvertibleMatrixError:
        return None

    return res_exponents


def convert_to(expr, target_units, unit_system="SI"):
    """
    Convert ``expr`` to the same expression with all of its units and quantities
    represented as factors of ``target_units``, whenever the dimension is compatible.

    ``target_units`` may be a single unit/quantity, or a collection of
    units/quantities.

    Examples
    ========

    >>> from sympy.physics.units import speed_of_light, meter, gram, second, day
    >>> from sympy.physics.units import mile, newton, kilogram, atomic_mass_constant
    >>> from sympy.physics.units import kilometer, centimeter
    >>> from sympy.physics.units import gravitational_constant, hbar
    >>> from sympy.physics.units import convert_to
    >>> convert_to(mile, kilometer)
    25146*kilometer/15625
    >>> convert_to(mile, kilometer).n()
    1.609344*kilometer
    >>> convert_to(speed_of_light, meter/second)
    299792458*meter/second
    >>> convert_to(day, second)
    86400*second
    >>> 3*newton
    3*newton
    >>> convert_to(3*newton, kilogram*meter/second**2)
    3*kilogram*meter/second**2
    >>> convert_to(atomic_mass_constant, gram)
    1.660539060e-24*gram

    Conversion to multiple units:

    >>> convert_to(speed_of_light, [meter, second])
    299792458*meter/second
    >>> convert_to(3*newton, [centimeter, gram, second])
    300000*centimeter*gram/second**2

    Conversion to Planck units:

    >>> convert_to(atomic_mass_constant, [gravitational_constant, speed_of_light, hbar]).n()
    7.62963087839509e-20*hbar**0.5*speed_of_light**0.5/gravitational_constant**0.5

    """
    from sympy.physics.units import UnitSystem
    unit_system = UnitSystem.get_unit_system(unit_system)

    if not isinstance(target_units, (Iterable, Tuple)):
        target_units = [target_units]

    def handle_Adds(expr):
        return Add.fromiter(convert_to(i, target_units, unit_system)
            for i in expr.args)

    if isinstance(expr, Add):
        return handle_Adds(expr)
    elif isinstance(expr, Pow) and isinstance(expr.base, Add):
        return handle_Adds(expr.base) ** expr.exp

    expr = sympify(expr)
    target_units = sympify(target_units)

    if isinstance(expr, Function):
        expr = expr.together()

    if not isinstance(expr, Quantity) and expr.has(Quantity):
        expr = expr.replace(lambda x: isinstance(x, Quantity),
            lambda x: x.convert_to(target_units, unit_system))

    def get_total_scale_factor(expr):
        if isinstance(expr, Mul):
            return reduce(lambda x, y: x * y,
                [get_total_scale_factor(i) for i in expr.args])
        elif isinstance(expr, Pow):
            return get_total_scale_factor(expr.base) ** expr.exp
        elif isinstance(expr, Quantity):
            return unit_system.get_quantity_scale_factor(expr)
        return expr

    depmat = _get_conversion_matrix_for_expr(expr, target_units, unit_system)
    if depmat is None:
        return expr

    expr_scale_factor = get_total_scale_factor(expr)
    return expr_scale_factor * Mul.fromiter(
        (1/get_total_scale_factor(u)*u)**p for u, p in
        zip(target_units, depmat))


def quantity_simplify(expr, across_dimensions: bool=False, unit_system=None):
    """Return an equivalent expression in which prefixes are replaced
    with numerical values and all units of a given dimension are the
    unified in a canonical manner by default. `across_dimensions` allows
    for units of different dimensions to be simplified together.

    `unit_system` must be specified if `across_dimensions` is True.

    Examples
    ========

    >>> from sympy.physics.units.util import quantity_simplify
    >>> from sympy.physics.units.prefixes import kilo
    >>> from sympy.physics.units import foot, inch, joule, coulomb
    >>> quantity_simplify(kilo*foot*inch)
    250*foot**2/3
    >>> quantity_simplify(foot - 6*inch)
    foot/2
    >>> quantity_simplify(5*joule/coulomb, across_dimensions=True, unit_system="SI")
    5*volt
    """

    if expr.is_Atom or not expr.has(Prefix, Quantity):
        return expr

    # replace all prefixes with numerical values
    p = expr.atoms(Prefix)
    expr = expr.xreplace({p: p.scale_factor for p in p})

    # replace all quantities of given dimension with a canonical
    # quantity, chosen from those in the expression
    d = sift(expr.atoms(Quantity), lambda i: i.dimension)
    for k in d:
        if len(d[k]) == 1:
            continue
        v = list(ordered(d[k]))
        ref = v[0]/v[0].scale_factor
        expr = expr.xreplace({vi: ref*vi.scale_factor for vi in v[1:]})

    if across_dimensions:
        # combine quantities of different dimensions into a single
        # quantity that is equivalent to the original expression

        if unit_system is None:
            raise ValueError("unit_system must be specified if across_dimensions is True")

        unit_system = UnitSystem.get_unit_system(unit_system)
        dimension_system: DimensionSystem = unit_system.get_dimension_system()
        dim_expr = unit_system.get_dimensional_expr(expr)
        dim_deps = dimension_system.get_dimensional_dependencies(dim_expr, mark_dimensionless=True)

        target_dimension: Optional[Dimension] = None
        for ds_dim, ds_dim_deps in dimension_system.dimensional_dependencies.items():
            if ds_dim_deps == dim_deps:
                target_dimension = ds_dim
                break

        if target_dimension is None:
            # if we can't find a target dimension, we can't do anything. unsure how to handle this case.
            return expr

        target_unit = unit_system.derived_units.get(target_dimension)
        if target_unit:
            expr = convert_to(expr, target_unit, unit_system)

    return expr


def check_dimensions(expr, unit_system="SI"):
    """Return expr if units in addends have the same
    base dimensions, else raise a ValueError."""
    # the case of adding a number to a dimensional quantity
    # is ignored for the sake of SymPy core routines, so this
    # function will raise an error now if such an addend is
    # found.
    # Also, when doing substitutions, multiplicative constants
    # might be introduced, so remove those now

    from sympy.physics.units import UnitSystem
    unit_system = UnitSystem.get_unit_system(unit_system)

    def addDict(dict1, dict2):
        """Merge dictionaries by adding values of common keys and
        removing keys with value of 0."""
        dict3 = {**dict1, **dict2}
        for key, value in dict3.items():
            if key in dict1 and key in dict2:
                dict3[key] = value + dict1[key]
        return {key:val for key, val in dict3.items() if val != 0}

    adds = expr.atoms(Add)
    DIM_OF = unit_system.get_dimension_system().get_dimensional_dependencies
    for a in adds:
        deset = set()
        for ai in a.args:
            if ai.is_number:
                deset.add(())
                continue
            dims = []
            skip = False
            dimdict = {}
            for i in Mul.make_args(ai):
                if i.has(Quantity):
                    i = Dimension(unit_system.get_dimensional_expr(i))
                if i.has(Dimension):
                    dimdict = addDict(dimdict, DIM_OF(i))
                elif i.free_symbols:
                    skip = True
                    break
            dims.extend(dimdict.items())
            if not skip:
                deset.add(tuple(sorted(dims, key=default_sort_key)))
                if len(deset) > 1:
                    raise ValueError(
                        "addends have incompatible dimensions: {}".format(deset))

    # clear multiplicative constants on Dimensions which may be
    # left after substitution
    reps = {}
    for m in expr.atoms(Mul):
        if any(isinstance(i, Dimension) for i in m.args):
            reps[m] = m.func(*[
                i for i in m.args if not i.is_number])

    return expr.xreplace(reps)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\definitions\__init__.py ===
from .unit_definitions import (
    percent, percents,
    permille,
    rad, radian, radians,
    deg, degree, degrees,
    sr, steradian, steradians,
    mil, angular_mil, angular_mils,
    m, meter, meters,
    kg, kilogram, kilograms,
    s, second, seconds,
    A, ampere, amperes,
    K, kelvin, kelvins,
    mol, mole, moles,
    cd, candela, candelas,
    g, gram, grams,
    mg, milligram, milligrams,
    ug, microgram, micrograms,
    t, tonne, metric_ton,
    newton, newtons, N,
    joule, joules, J,
    watt, watts, W,
    pascal, pascals, Pa, pa,
    hertz, hz, Hz,
    coulomb, coulombs, C,
    volt, volts, v, V,
    ohm, ohms,
    siemens, S, mho, mhos,
    farad, farads, F,
    henry, henrys, H,
    tesla, teslas, T,
    weber, webers, Wb, wb,
    optical_power, dioptre, D,
    lux, lx,
    katal, kat,
    gray, Gy,
    becquerel, Bq,
    km, kilometer, kilometers,
    dm, decimeter, decimeters,
    cm, centimeter, centimeters,
    mm, millimeter, millimeters,
    um, micrometer, micrometers, micron, microns,
    nm, nanometer, nanometers,
    pm, picometer, picometers,
    ft, foot, feet,
    inch, inches,
    yd, yard, yards,
    mi, mile, miles,
    nmi, nautical_mile, nautical_miles,
    ha, hectare,
    l, L, liter, liters,
    dl, dL, deciliter, deciliters,
    cl, cL, centiliter, centiliters,
    ml, mL, milliliter, milliliters,
    ms, millisecond, milliseconds,
    us, microsecond, microseconds,
    ns, nanosecond, nanoseconds,
    ps, picosecond, picoseconds,
    minute, minutes,
    h, hour, hours,
    day, days,
    anomalistic_year, anomalistic_years,
    sidereal_year, sidereal_years,
    tropical_year, tropical_years,
    common_year, common_years,
    julian_year, julian_years,
    draconic_year, draconic_years,
    gaussian_year, gaussian_years,
    full_moon_cycle, full_moon_cycles,
    year, years,
    G, gravitational_constant,
    c, speed_of_light,
    elementary_charge,
    hbar,
    planck,
    eV, electronvolt, electronvolts,
    avogadro_number,
    avogadro, avogadro_constant,
    boltzmann, boltzmann_constant,
    stefan, stefan_boltzmann_constant,
    R, molar_gas_constant,
    faraday_constant,
    josephson_constant,
    von_klitzing_constant,
    Da, dalton, amu, amus, atomic_mass_unit, atomic_mass_constant,
    me, electron_rest_mass,
    gee, gees, acceleration_due_to_gravity,
    u0, magnetic_constant, vacuum_permeability,
    e0, electric_constant, vacuum_permittivity,
    Z0, vacuum_impedance,
    coulomb_constant, coulombs_constant, electric_force_constant,
    atmosphere, atmospheres, atm,
    kPa, kilopascal,
    bar, bars,
    pound, pounds,
    psi,
    dHg0,
    mmHg, torr,
    mmu, mmus, milli_mass_unit,
    quart, quarts,
    angstrom, angstroms,
    ly, lightyear, lightyears,
    au, astronomical_unit, astronomical_units,
    planck_mass,
    planck_time,
    planck_temperature,
    planck_length,
    planck_charge,
    planck_area,
    planck_volume,
    planck_momentum,
    planck_energy,
    planck_force,
    planck_power,
    planck_density,
    planck_energy_density,
    planck_intensity,
    planck_angular_frequency,
    planck_pressure,
    planck_current,
    planck_voltage,
    planck_impedance,
    planck_acceleration,
    bit, bits,
    byte,
    kibibyte, kibibytes,
    mebibyte, mebibytes,
    gibibyte, gibibytes,
    tebibyte, tebibytes,
    pebibyte, pebibytes,
    exbibyte, exbibytes,
    curie, rutherford
)

__all__ = [
    'percent', 'percents',
    'permille',
    'rad', 'radian', 'radians',
    'deg', 'degree', 'degrees',
    'sr', 'steradian', 'steradians',
    'mil', 'angular_mil', 'angular_mils',
    'm', 'meter', 'meters',
    'kg', 'kilogram', 'kilograms',
    's', 'second', 'seconds',
    'A', 'ampere', 'amperes',
    'K', 'kelvin', 'kelvins',
    'mol', 'mole', 'moles',
    'cd', 'candela', 'candelas',
    'g', 'gram', 'grams',
    'mg', 'milligram', 'milligrams',
    'ug', 'microgram', 'micrograms',
    't', 'tonne', 'metric_ton',
    'newton', 'newtons', 'N',
    'joule', 'joules', 'J',
    'watt', 'watts', 'W',
    'pascal', 'pascals', 'Pa', 'pa',
    'hertz', 'hz', 'Hz',
    'coulomb', 'coulombs', 'C',
    'volt', 'volts', 'v', 'V',
    'ohm', 'ohms',
    'siemens', 'S', 'mho', 'mhos',
    'farad', 'farads', 'F',
    'henry', 'henrys', 'H',
    'tesla', 'teslas', 'T',
    'weber', 'webers', 'Wb', 'wb',
    'optical_power', 'dioptre', 'D',
    'lux', 'lx',
    'katal', 'kat',
    'gray', 'Gy',
    'becquerel', 'Bq',
    'km', 'kilometer', 'kilometers',
    'dm', 'decimeter', 'decimeters',
    'cm', 'centimeter', 'centimeters',
    'mm', 'millimeter', 'millimeters',
    'um', 'micrometer', 'micrometers', 'micron', 'microns',
    'nm', 'nanometer', 'nanometers',
    'pm', 'picometer', 'picometers',
    'ft', 'foot', 'feet',
    'inch', 'inches',
    'yd', 'yard', 'yards',
    'mi', 'mile', 'miles',
    'nmi', 'nautical_mile', 'nautical_miles',
    'ha', 'hectare',
    'l', 'L', 'liter', 'liters',
    'dl', 'dL', 'deciliter', 'deciliters',
    'cl', 'cL', 'centiliter', 'centiliters',
    'ml', 'mL', 'milliliter', 'milliliters',
    'ms', 'millisecond', 'milliseconds',
    'us', 'microsecond', 'microseconds',
    'ns', 'nanosecond', 'nanoseconds',
    'ps', 'picosecond', 'picoseconds',
    'minute', 'minutes',
    'h', 'hour', 'hours',
    'day', 'days',
    'anomalistic_year', 'anomalistic_years',
    'sidereal_year', 'sidereal_years',
    'tropical_year', 'tropical_years',
    'common_year', 'common_years',
    'julian_year', 'julian_years',
    'draconic_year', 'draconic_years',
    'gaussian_year', 'gaussian_years',
    'full_moon_cycle', 'full_moon_cycles',
    'year', 'years',
    'G', 'gravitational_constant',
    'c', 'speed_of_light',
    'elementary_charge',
    'hbar',
    'planck',
    'eV', 'electronvolt', 'electronvolts',
    'avogadro_number',
    'avogadro', 'avogadro_constant',
    'boltzmann', 'boltzmann_constant',
    'stefan', 'stefan_boltzmann_constant',
    'R', 'molar_gas_constant',
    'faraday_constant',
    'josephson_constant',
    'von_klitzing_constant',
    'Da', 'dalton', 'amu', 'amus', 'atomic_mass_unit', 'atomic_mass_constant',
    'me', 'electron_rest_mass',
    'gee', 'gees', 'acceleration_due_to_gravity',
    'u0', 'magnetic_constant', 'vacuum_permeability',
    'e0', 'electric_constant', 'vacuum_permittivity',
    'Z0', 'vacuum_impedance',
    'coulomb_constant', 'coulombs_constant', 'electric_force_constant',
    'atmosphere', 'atmospheres', 'atm',
    'kPa', 'kilopascal',
    'bar', 'bars',
    'pound', 'pounds',
    'psi',
    'dHg0',
    'mmHg', 'torr',
    'mmu', 'mmus', 'milli_mass_unit',
    'quart', 'quarts',
    'angstrom', 'angstroms',
    'ly', 'lightyear', 'lightyears',
    'au', 'astronomical_unit', 'astronomical_units',
    'planck_mass',
    'planck_time',
    'planck_temperature',
    'planck_length',
    'planck_charge',
    'planck_area',
    'planck_volume',
    'planck_momentum',
    'planck_energy',
    'planck_force',
    'planck_power',
    'planck_density',
    'planck_energy_density',
    'planck_intensity',
    'planck_angular_frequency',
    'planck_pressure',
    'planck_current',
    'planck_voltage',
    'planck_impedance',
    'planck_acceleration',
    'bit', 'bits',
    'byte',
    'kibibyte', 'kibibytes',
    'mebibyte', 'mebibytes',
    'gibibyte', 'gibibytes',
    'tebibyte', 'tebibytes',
    'pebibyte', 'pebibytes',
    'exbibyte', 'exbibytes',
    'curie', 'rutherford',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\sqlite\dml.py ===
# dialects/sqlite/dml.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from .._typing import _OnConflictIndexElementsT
from .._typing import _OnConflictIndexWhereT
from .._typing import _OnConflictSetT
from .._typing import _OnConflictWhereT
from ... import util
from ...sql import coercions
from ...sql import roles
from ...sql import schema
from ...sql._typing import _DMLTableArgument
from ...sql.base import _exclusive_against
from ...sql.base import _generative
from ...sql.base import ColumnCollection
from ...sql.base import ReadOnlyColumnCollection
from ...sql.dml import Insert as StandardInsert
from ...sql.elements import ClauseElement
from ...sql.elements import ColumnElement
from ...sql.elements import KeyedColumnElement
from ...sql.elements import TextClause
from ...sql.expression import alias
from ...util.typing import Self

__all__ = ("Insert", "insert")


def insert(table: _DMLTableArgument) -> Insert:
    """Construct a sqlite-specific variant :class:`_sqlite.Insert`
    construct.

    .. container:: inherited_member

        The :func:`sqlalchemy.dialects.sqlite.insert` function creates
        a :class:`sqlalchemy.dialects.sqlite.Insert`.  This class is based
        on the dialect-agnostic :class:`_sql.Insert` construct which may
        be constructed using the :func:`_sql.insert` function in
        SQLAlchemy Core.

    The :class:`_sqlite.Insert` construct includes additional methods
    :meth:`_sqlite.Insert.on_conflict_do_update`,
    :meth:`_sqlite.Insert.on_conflict_do_nothing`.

    """
    return Insert(table)


class Insert(StandardInsert):
    """SQLite-specific implementation of INSERT.

    Adds methods for SQLite-specific syntaxes such as ON CONFLICT.

    The :class:`_sqlite.Insert` object is created using the
    :func:`sqlalchemy.dialects.sqlite.insert` function.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`sqlite_on_conflict_insert`

    """

    stringify_dialect = "sqlite"
    inherit_cache = False

    @util.memoized_property
    def excluded(
        self,
    ) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """Provide the ``excluded`` namespace for an ON CONFLICT statement

        SQLite's ON CONFLICT clause allows reference to the row that would
        be inserted, known as ``excluded``.  This attribute provides
        all columns in this row to be referenceable.

        .. tip::  The :attr:`_sqlite.Insert.excluded` attribute is an instance
            of :class:`_expression.ColumnCollection`, which provides an
            interface the same as that of the :attr:`_schema.Table.c`
            collection described at :ref:`metadata_tables_and_columns`.
            With this collection, ordinary names are accessible like attributes
            (e.g. ``stmt.excluded.some_column``), but special names and
            dictionary method names should be accessed using indexed access,
            such as ``stmt.excluded["column name"]`` or
            ``stmt.excluded["values"]``.  See the docstring for
            :class:`_expression.ColumnCollection` for further examples.

        """
        return alias(self.table, name="excluded").columns

    _on_conflict_exclusive = _exclusive_against(
        "_post_values_clause",
        msgs={
            "_post_values_clause": "This Insert construct already has "
            "an ON CONFLICT clause established"
        },
    )

    @_generative
    @_on_conflict_exclusive
    def on_conflict_do_update(
        self,
        index_elements: _OnConflictIndexElementsT = None,
        index_where: _OnConflictIndexWhereT = None,
        set_: _OnConflictSetT = None,
        where: _OnConflictWhereT = None,
    ) -> Self:
        r"""
        Specifies a DO UPDATE SET action for ON CONFLICT clause.

        :param index_elements:
         A sequence consisting of string column names, :class:`_schema.Column`
         objects, or other column expression objects that will be used
         to infer a target index or unique constraint.

        :param index_where:
         Additional WHERE criterion that can be used to infer a
         conditional target index.

        :param set\_:
         A dictionary or other mapping object
         where the keys are either names of columns in the target table,
         or :class:`_schema.Column` objects or other ORM-mapped columns
         matching that of the target table, and expressions or literals
         as values, specifying the ``SET`` actions to take.

         .. versionadded:: 1.4 The
            :paramref:`_sqlite.Insert.on_conflict_do_update.set_`
            parameter supports :class:`_schema.Column` objects from the target
            :class:`_schema.Table` as keys.

         .. warning:: This dictionary does **not** take into account
            Python-specified default UPDATE values or generation functions,
            e.g. those specified using :paramref:`_schema.Column.onupdate`.
            These values will not be exercised for an ON CONFLICT style of
            UPDATE, unless they are manually specified in the
            :paramref:`.Insert.on_conflict_do_update.set_` dictionary.

        :param where:
         Optional argument. An expression object representing a ``WHERE``
         clause that restricts the rows affected by ``DO UPDATE SET``. Rows not
         meeting the ``WHERE`` condition will not be updated (effectively a
         ``DO NOTHING`` for those rows).

        """

        self._post_values_clause = OnConflictDoUpdate(
            index_elements, index_where, set_, where
        )
        return self

    @_generative
    @_on_conflict_exclusive
    def on_conflict_do_nothing(
        self,
        index_elements: _OnConflictIndexElementsT = None,
        index_where: _OnConflictIndexWhereT = None,
    ) -> Self:
        """
        Specifies a DO NOTHING action for ON CONFLICT clause.

        :param index_elements:
         A sequence consisting of string column names, :class:`_schema.Column`
         objects, or other column expression objects that will be used
         to infer a target index or unique constraint.

        :param index_where:
         Additional WHERE criterion that can be used to infer a
         conditional target index.

        """

        self._post_values_clause = OnConflictDoNothing(
            index_elements, index_where
        )
        return self


class OnConflictClause(ClauseElement):
    stringify_dialect = "sqlite"

    inferred_target_elements: Optional[List[Union[str, schema.Column[Any]]]]
    inferred_target_whereclause: Optional[
        Union[ColumnElement[Any], TextClause]
    ]

    def __init__(
        self,
        index_elements: _OnConflictIndexElementsT = None,
        index_where: _OnConflictIndexWhereT = None,
    ):
        if index_elements is not None:
            self.inferred_target_elements = [
                coercions.expect(roles.DDLConstraintColumnRole, column)
                for column in index_elements
            ]
            self.inferred_target_whereclause = (
                coercions.expect(
                    roles.WhereHavingRole,
                    index_where,
                )
                if index_where is not None
                else None
            )
        else:
            self.inferred_target_elements = (
                self.inferred_target_whereclause
            ) = None


class OnConflictDoNothing(OnConflictClause):
    __visit_name__ = "on_conflict_do_nothing"


class OnConflictDoUpdate(OnConflictClause):
    __visit_name__ = "on_conflict_do_update"

    update_values_to_set: List[Tuple[Union[schema.Column[Any], str], Any]]
    update_whereclause: Optional[ColumnElement[Any]]

    def __init__(
        self,
        index_elements: _OnConflictIndexElementsT = None,
        index_where: _OnConflictIndexWhereT = None,
        set_: _OnConflictSetT = None,
        where: _OnConflictWhereT = None,
    ):
        super().__init__(
            index_elements=index_elements,
            index_where=index_where,
        )

        if isinstance(set_, dict):
            if not set_:
                raise ValueError("set parameter dictionary must not be empty")
        elif isinstance(set_, ColumnCollection):
            set_ = dict(set_)
        else:
            raise ValueError(
                "set parameter must be a non-empty dictionary "
                "or a ColumnCollection such as the `.c.` collection "
                "of a Table object"
            )
        self.update_values_to_set = [
            (coercions.expect(roles.DMLColumnRole, key), value)
            for key, value in set_.items()
        ]
        self.update_whereclause = (
            coercions.expect(roles.WhereHavingRole, where)
            if where is not None
            else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\qapply.py ===
"""Logic for applying operators to states.

Todo:
* Sometimes the final result needs to be expanded, we should do this by hand.
"""

from sympy.concrete import Sum
from sympy.core.add import Add
from sympy.core.kind import NumberKind
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sympify import sympify, _sympify

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.state import State, KetBase, BraBase, Wavefunction
from sympy.physics.quantum.tensorproduct import TensorProduct

__all__ = [
    'qapply'
]


#-----------------------------------------------------------------------------
# Main code
#-----------------------------------------------------------------------------


def ip_doit_func(e):
    """Transform the inner products in an expression by calling ``.doit()``."""
    return e.replace(InnerProduct, lambda *args: InnerProduct(*args).doit())


def sum_doit_func(e):
    """Transform the sums in an expression by calling ``.doit()``."""
    return e.replace(Sum, lambda *args: Sum(*args).doit())


def qapply(e, **options):
    """Apply operators to states in a quantum expression.

    Parameters
    ==========

    e : Expr
        The expression containing operators and states. This expression tree
        will be walked to find operators acting on states symbolically.
    options : dict
        A dict of key/value pairs that determine how the operator actions
        are carried out.

        The following options are valid:

        * ``dagger``: try to apply Dagger operators to the left
          (default: False).
        * ``ip_doit``: call ``.doit()`` in inner products when they are
          encountered (default: True).
        * ``sum_doit``: call ``.doit()`` on sums when they are encountered
          (default: False). This is helpful for collapsing sums over Kronecker
          delta's that are created when calling ``qapply``.

    Returns
    =======

    e : Expr
        The original expression, but with the operators applied to states.

    Examples
    ========

        >>> from sympy.physics.quantum import qapply, Ket, Bra
        >>> b = Bra('b')
        >>> k = Ket('k')
        >>> A = k * b
        >>> A
        |k><b|
        >>> qapply(A * b.dual / (b * b.dual))
        |k>
        >>> qapply(k.dual * A / (k.dual * k))
        <b|
    """
    from sympy.physics.quantum.density import Density

    dagger = options.get('dagger', False)
    sum_doit = options.get('sum_doit', False)
    ip_doit = options.get('ip_doit', True)

    e = _sympify(e)

    # Using the kind API here helps us to narrow what types of expressions
    # we call ``ip_doit_func`` on.
    if e.kind == NumberKind:
        return ip_doit_func(e) if ip_doit else e

    # This may be a bit aggressive but ensures that everything gets expanded
    # to its simplest form before trying to apply operators. This includes
    # things like (A+B+C)*|a> and A*(|a>+|b>) and all Commutators and
    # TensorProducts. The only problem with this is that if we can't apply
    # all the Operators, we have just expanded everything.
    # TODO: don't expand the scalars in front of each Mul.
    e = e.expand(commutator=True, tensorproduct=True)

    # If we just have a raw ket, return it.
    if isinstance(e, KetBase):
        return e

    # We have an Add(a, b, c, ...) and compute
    # Add(qapply(a), qapply(b), ...)
    elif isinstance(e, Add):
        result = 0
        for arg in e.args:
            result += qapply(arg, **options)
        return result.expand()

    # For a Density operator call qapply on its state
    elif isinstance(e, Density):
        new_args = [(qapply(state, **options), prob) for (state,
                     prob) in e.args]
        return Density(*new_args)

    # For a raw TensorProduct, call qapply on its args.
    elif isinstance(e, TensorProduct):
        return TensorProduct(*[qapply(t, **options) for t in e.args])

    # For a Sum, call qapply on its function.
    elif isinstance(e, Sum):
        result = Sum(qapply(e.function, **options), *e.limits)
        result = sum_doit_func(result) if sum_doit else result
        return result

    # For a Pow, call qapply on its base.
    elif isinstance(e, Pow):
        return qapply(e.base, **options)**e.exp

    # We have a Mul where there might be actual operators to apply to kets.
    elif isinstance(e, Mul):
        c_part, nc_part = e.args_cnc()
        c_mul = Mul(*c_part)
        nc_mul = Mul(*nc_part)
        if not nc_part: # If we only have a commuting part, just return it.
            result = c_mul
        elif isinstance(nc_mul, Mul):
            result = c_mul*qapply_Mul(nc_mul, **options)
        else:
            result = c_mul*qapply(nc_mul, **options)
        if result == e and dagger:
            result = Dagger(qapply_Mul(Dagger(e), **options))
        result = ip_doit_func(result) if ip_doit else result
        result = sum_doit_func(result) if sum_doit else result
        return result

    # In all other cases (State, Operator, Pow, Commutator, InnerProduct,
    # OuterProduct) we won't ever have operators to apply to kets.
    else:
        return e


def qapply_Mul(e, **options):

    args = list(e.args)
    extra = S.One
    result = None

    # If we only have 0 or 1 args, we have nothing to do and return.
    if len(args) <= 1 or not isinstance(e, Mul):
        return e
    rhs = args.pop()
    lhs = args.pop()

    # Make sure we have two non-commutative objects before proceeding.
    if (not isinstance(rhs, Wavefunction) and sympify(rhs).is_commutative) or \
            (not isinstance(lhs, Wavefunction) and sympify(lhs).is_commutative):
        return e

    # For a Pow with an integer exponent, apply one of them and reduce the
    # exponent by one.
    if isinstance(lhs, Pow) and lhs.exp.is_Integer:
        args.append(lhs.base**(lhs.exp - 1))
        lhs = lhs.base

    # Pull OuterProduct apart
    if isinstance(lhs, OuterProduct):
        args.append(lhs.ket)
        lhs = lhs.bra

    if isinstance(rhs, OuterProduct):
        extra = rhs.bra # Append to the right of the result
        rhs = rhs.ket

    # Call .doit() on Commutator/AntiCommutator.
    if isinstance(lhs, (Commutator, AntiCommutator)):
        comm = lhs.doit()
        if isinstance(comm, Add):
            return qapply(
                e.func(*(args + [comm.args[0], rhs])) +
                e.func(*(args + [comm.args[1], rhs])),
                **options
            )*extra
        else:
            return qapply(e.func(*args)*comm*rhs, **options)*extra

    # Apply tensor products of operators to states
    if isinstance(lhs, TensorProduct) and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in lhs.args) and \
            isinstance(rhs, TensorProduct) and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in rhs.args) and \
            len(lhs.args) == len(rhs.args):
        result = TensorProduct(*[qapply(lhs.args[n]*rhs.args[n], **options) for n in range(len(lhs.args))]).expand(tensorproduct=True)
        return qapply_Mul(e.func(*args), **options)*result*extra

    # For Sums, move the Sum to the right.
    if isinstance(rhs, Sum):
        if isinstance(lhs, Sum):
            if set(lhs.variables).intersection(set(rhs.variables)):
                raise ValueError('Duplicated dummy indices in separate sums in qapply.')
            limits = lhs.limits + rhs.limits
            result = Sum(qapply(lhs.function*rhs.function, **options), *limits)
            return qapply_Mul(e.func(*args)*result, **options)
        else:
            result = Sum(qapply(lhs*rhs.function, **options), *rhs.limits)
            return qapply_Mul(e.func(*args)*result, **options)

    if isinstance(lhs, Sum):
        result = Sum(qapply(lhs.function*rhs, **options), *lhs.limits)
        return qapply_Mul(e.func(*args)*result, **options)

    # Now try to actually apply the operator and build an inner product.
    _apply = getattr(lhs, '_apply_operator', None)
    if _apply is not None:
        try:
            result = _apply(rhs, **options)
        except NotImplementedError:
            result = None
    else:
        result = None

    if result is None:
        _apply_right = getattr(rhs, '_apply_from_right_to', None)
        if _apply_right is not None:
            try:
                result = _apply_right(lhs, **options)
            except NotImplementedError:
                result = None

    if result is None:
        if isinstance(lhs, BraBase) and isinstance(rhs, KetBase):
            result = InnerProduct(lhs, rhs)

    # TODO: I may need to expand before returning the final result.
    if isinstance(result, (int, complex, float)):
        return _sympify(result)
    elif result is None:
        if len(args) == 0:
            # We had two args to begin with so args=[].
            return e
        else:
            return qapply_Mul(e.func(*(args + [lhs])), **options)*rhs*extra
    elif isinstance(result, InnerProduct):
        return result*qapply_Mul(e.func(*args), **options)*extra
    else:  # result is a scalar times a Mul, Add or TensorProduct
        return qapply(e.func(*args)*result, **options)*extra

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\_manylinux.py ===
from __future__ import annotations

import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Generator, Iterator, NamedTuple, Sequence

from ._elffile import EIClass, EIData, ELFFile, EMachine

EF_ARM_ABIMASK = 0xFF000000
EF_ARM_ABI_VER5 = 0x05000000
EF_ARM_ABI_FLOAT_HARD = 0x00000400


# `os.PathLike` not a generic type until Python 3.9, so sticking with `str`
# as the type for `path` until then.
@contextlib.contextmanager
def _parse_elf(path: str) -> Generator[ELFFile | None, None, None]:
    try:
        with open(path, "rb") as f:
            yield ELFFile(f)
    except (OSError, TypeError, ValueError):
        yield None


def _is_linux_armhf(executable: str) -> bool:
    # hard-float ABI can be detected from the ELF header of the running
    # process
    # https://static.docs.arm.com/ihi0044/g/aaelf32.pdf
    with _parse_elf(executable) as f:
        return (
            f is not None
            and f.capacity == EIClass.C32
            and f.encoding == EIData.Lsb
            and f.machine == EMachine.Arm
            and f.flags & EF_ARM_ABIMASK == EF_ARM_ABI_VER5
            and f.flags & EF_ARM_ABI_FLOAT_HARD == EF_ARM_ABI_FLOAT_HARD
        )


def _is_linux_i686(executable: str) -> bool:
    with _parse_elf(executable) as f:
        return (
            f is not None
            and f.capacity == EIClass.C32
            and f.encoding == EIData.Lsb
            and f.machine == EMachine.I386
        )


def _have_compatible_abi(executable: str, archs: Sequence[str]) -> bool:
    if "armv7l" in archs:
        return _is_linux_armhf(executable)
    if "i686" in archs:
        return _is_linux_i686(executable)
    allowed_archs = {
        "x86_64",
        "aarch64",
        "ppc64",
        "ppc64le",
        "s390x",
        "loongarch64",
        "riscv64",
    }
    return any(arch in allowed_archs for arch in archs)


# If glibc ever changes its major version, we need to know what the last
# minor version was, so we can build the complete list of all versions.
# For now, guess what the highest minor version might be, assume it will
# be 50 for testing. Once this actually happens, update the dictionary
# with the actual value.
_LAST_GLIBC_MINOR: dict[int, int] = collections.defaultdict(lambda: 50)


class _GLibCVersion(NamedTuple):
    major: int
    minor: int


def _glibc_version_string_confstr() -> str | None:
    """
    Primary implementation of glibc_version_string using os.confstr.
    """
    # os.confstr is quite a bit faster than ctypes.DLL. It's also less likely
    # to be broken or missing. This strategy is used in the standard library
    # platform module.
    # https://github.com/python/cpython/blob/fcf1d003bf4f0100c/Lib/platform.py#L175-L183
    try:
        # Should be a string like "glibc 2.17".
        version_string: str | None = os.confstr("CS_GNU_LIBC_VERSION")
        assert version_string is not None
        _, version = version_string.rsplit()
    except (AssertionError, AttributeError, OSError, ValueError):
        # os.confstr() or CS_GNU_LIBC_VERSION not available (or a bad value)...
        return None
    return version


def _glibc_version_string_ctypes() -> str | None:
    """
    Fallback implementation of glibc_version_string using ctypes.
    """
    try:
        import ctypes
    except ImportError:
        return None

    # ctypes.CDLL(None) internally calls dlopen(NULL), and as the dlopen
    # manpage says, "If filename is NULL, then the returned handle is for the
    # main program". This way we can let the linker do the work to figure out
    # which libc our process is actually using.
    #
    # We must also handle the special case where the executable is not a
    # dynamically linked executable. This can occur when using musl libc,
    # for example. In this situation, dlopen() will error, leading to an
    # OSError. Interestingly, at least in the case of musl, there is no
    # errno set on the OSError. The single string argument used to construct
    # OSError comes from libc itself and is therefore not portable to
    # hard code here. In any case, failure to call dlopen() means we
    # can proceed, so we bail on our attempt.
    try:
        process_namespace = ctypes.CDLL(None)
    except OSError:
        return None

    try:
        gnu_get_libc_version = process_namespace.gnu_get_libc_version
    except AttributeError:
        # Symbol doesn't exist -> therefore, we are not linked to
        # glibc.
        return None

    # Call gnu_get_libc_version, which returns a string like "2.5"
    gnu_get_libc_version.restype = ctypes.c_char_p
    version_str: str = gnu_get_libc_version()
    # py2 / py3 compatibility:
    if not isinstance(version_str, str):
        version_str = version_str.decode("ascii")

    return version_str


def _glibc_version_string() -> str | None:
    """Returns glibc version string, or None if not using glibc."""
    return _glibc_version_string_confstr() or _glibc_version_string_ctypes()


def _parse_glibc_version(version_str: str) -> tuple[int, int]:
    """Parse glibc version.

    We use a regexp instead of str.split because we want to discard any
    random junk that might come after the minor version -- this might happen
    in patched/forked versions of glibc (e.g. Linaro's version of glibc
    uses version strings like "2.20-2014.11"). See gh-3588.
    """
    m = re.match(r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)", version_str)
    if not m:
        warnings.warn(
            f"Expected glibc version with 2 components major.minor, got: {version_str}",
            RuntimeWarning,
            stacklevel=2,
        )
        return -1, -1
    return int(m.group("major")), int(m.group("minor"))


@functools.lru_cache
def _get_glibc_version() -> tuple[int, int]:
    version_str = _glibc_version_string()
    if version_str is None:
        return (-1, -1)
    return _parse_glibc_version(version_str)


# From PEP 513, PEP 600
def _is_compatible(arch: str, version: _GLibCVersion) -> bool:
    sys_glibc = _get_glibc_version()
    if sys_glibc < version:
        return False
    # Check for presence of _manylinux module.
    try:
        import _manylinux
    except ImportError:
        return True
    if hasattr(_manylinux, "manylinux_compatible"):
        result = _manylinux.manylinux_compatible(version[0], version[1], arch)
        if result is not None:
            return bool(result)
        return True
    if version == _GLibCVersion(2, 5):
        if hasattr(_manylinux, "manylinux1_compatible"):
            return bool(_manylinux.manylinux1_compatible)
    if version == _GLibCVersion(2, 12):
        if hasattr(_manylinux, "manylinux2010_compatible"):
            return bool(_manylinux.manylinux2010_compatible)
    if version == _GLibCVersion(2, 17):
        if hasattr(_manylinux, "manylinux2014_compatible"):
            return bool(_manylinux.manylinux2014_compatible)
    return True


_LEGACY_MANYLINUX_MAP = {
    # CentOS 7 w/ glibc 2.17 (PEP 599)
    (2, 17): "manylinux2014",
    # CentOS 6 w/ glibc 2.12 (PEP 571)
    (2, 12): "manylinux2010",
    # CentOS 5 w/ glibc 2.5 (PEP 513)
    (2, 5): "manylinux1",
}


def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    """Generate manylinux tags compatible to the current platform.

    :param archs: Sequence of compatible architectures.
        The first one shall be the closest to the actual architecture and be the part of
        platform tag after the ``linux_`` prefix, e.g. ``x86_64``.
        The ``linux_`` prefix is assumed as a prerequisite for the current platform to
        be manylinux-compatible.

    :returns: An iterator of compatible manylinux tags.
    """
    if not _have_compatible_abi(sys.executable, archs):
        return
    # Oldest glibc to be supported regardless of architecture is (2, 17).
    too_old_glibc2 = _GLibCVersion(2, 16)
    if set(archs) & {"x86_64", "i686"}:
        # On x86/i686 also oldest glibc to be supported is (2, 5).
        too_old_glibc2 = _GLibCVersion(2, 4)
    current_glibc = _GLibCVersion(*_get_glibc_version())
    glibc_max_list = [current_glibc]
    # We can assume compatibility across glibc major versions.
    # https://sourceware.org/bugzilla/show_bug.cgi?id=24636
    #
    # Build a list of maximum glibc versions so that we can
    # output the canonical list of all glibc from current_glibc
    # down to too_old_glibc2, including all intermediary versions.
    for glibc_major in range(current_glibc.major - 1, 1, -1):
        glibc_minor = _LAST_GLIBC_MINOR[glibc_major]
        glibc_max_list.append(_GLibCVersion(glibc_major, glibc_minor))
    for arch in archs:
        for glibc_max in glibc_max_list:
            if glibc_max.major == too_old_glibc2.major:
                min_minor = too_old_glibc2.minor
            else:
                # For other glibc major versions oldest supported is (x, 0).
                min_minor = -1
            for glibc_minor in range(glibc_max.minor, min_minor, -1):
                glibc_version = _GLibCVersion(glibc_max.major, glibc_minor)
                tag = "manylinux_{}_{}".format(*glibc_version)
                if _is_compatible(arch, glibc_version):
                    yield f"{tag}_{arch}"
                # Handle the legacy manylinux1, manylinux2010, manylinux2014 tags.
                if glibc_version in _LEGACY_MANYLINUX_MAP:
                    legacy_tag = _LEGACY_MANYLINUX_MAP[glibc_version]
                    if _is_compatible(arch, glibc_version):
                        yield f"{legacy_tag}_{arch}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\compat\pickle_compat.py ===
"""
Support pre-0.12 series pickle compatibility.
"""
from __future__ import annotations

import contextlib
import copy
import io
import pickle as pkl
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import BaseOffset

from pandas import Index
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.internals import BlockManager

if TYPE_CHECKING:
    from collections.abc import Generator


def load_reduce(self) -> None:
    stack = self.stack
    args = stack.pop()
    func = stack[-1]

    try:
        stack[-1] = func(*args)
        return
    except TypeError as err:
        # If we have a deprecated function,
        # try to replace and try again.

        msg = "_reconstruct: First argument must be a sub-type of ndarray"

        if msg in str(err):
            try:
                cls = args[0]
                stack[-1] = object.__new__(cls)
                return
            except TypeError:
                pass
        elif args and isinstance(args[0], type) and issubclass(args[0], BaseOffset):
            # TypeError: object.__new__(Day) is not safe, use Day.__new__()
            cls = args[0]
            stack[-1] = cls.__new__(*args)
            return
        elif args and issubclass(args[0], PeriodArray):
            cls = args[0]
            stack[-1] = NDArrayBacked.__new__(*args)
            return

        raise


# If classes are moved, provide compat here.
_class_locations_map = {
    ("pandas.core.sparse.array", "SparseArray"): ("pandas.core.arrays", "SparseArray"),
    # 15477
    ("pandas.core.base", "FrozenNDArray"): ("numpy", "ndarray"),
    # Re-routing unpickle block logic to go through _unpickle_block instead
    # for pandas <= 1.3.5
    ("pandas.core.internals.blocks", "new_block"): (
        "pandas._libs.internals",
        "_unpickle_block",
    ),
    ("pandas.core.indexes.frozen", "FrozenNDArray"): ("numpy", "ndarray"),
    ("pandas.core.base", "FrozenList"): ("pandas.core.indexes.frozen", "FrozenList"),
    # 10890
    ("pandas.core.series", "TimeSeries"): ("pandas.core.series", "Series"),
    ("pandas.sparse.series", "SparseTimeSeries"): (
        "pandas.core.sparse.series",
        "SparseSeries",
    ),
    # 12588, extensions moving
    ("pandas._sparse", "BlockIndex"): ("pandas._libs.sparse", "BlockIndex"),
    ("pandas.tslib", "Timestamp"): ("pandas._libs.tslib", "Timestamp"),
    # 18543 moving period
    ("pandas._period", "Period"): ("pandas._libs.tslibs.period", "Period"),
    ("pandas._libs.period", "Period"): ("pandas._libs.tslibs.period", "Period"),
    # 18014 moved __nat_unpickle from _libs.tslib-->_libs.tslibs.nattype
    ("pandas.tslib", "__nat_unpickle"): (
        "pandas._libs.tslibs.nattype",
        "__nat_unpickle",
    ),
    ("pandas._libs.tslib", "__nat_unpickle"): (
        "pandas._libs.tslibs.nattype",
        "__nat_unpickle",
    ),
    # 15998 top-level dirs moving
    ("pandas.sparse.array", "SparseArray"): (
        "pandas.core.arrays.sparse",
        "SparseArray",
    ),
    ("pandas.indexes.base", "_new_Index"): ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.indexes.base", "Index"): ("pandas.core.indexes.base", "Index"),
    ("pandas.indexes.numeric", "Int64Index"): (
        "pandas.core.indexes.base",
        "Index",  # updated in 50775
    ),
    ("pandas.indexes.range", "RangeIndex"): ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.indexes.multi", "MultiIndex"): ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.tseries.index", "_new_DatetimeIndex"): (
        "pandas.core.indexes.datetimes",
        "_new_DatetimeIndex",
    ),
    ("pandas.tseries.index", "DatetimeIndex"): (
        "pandas.core.indexes.datetimes",
        "DatetimeIndex",
    ),
    ("pandas.tseries.period", "PeriodIndex"): (
        "pandas.core.indexes.period",
        "PeriodIndex",
    ),
    # 19269, arrays moving
    ("pandas.core.categorical", "Categorical"): ("pandas.core.arrays", "Categorical"),
    # 19939, add timedeltaindex, float64index compat from 15998 move
    ("pandas.tseries.tdi", "TimedeltaIndex"): (
        "pandas.core.indexes.timedeltas",
        "TimedeltaIndex",
    ),
    ("pandas.indexes.numeric", "Float64Index"): (
        "pandas.core.indexes.base",
        "Index",  # updated in 50775
    ),
    # 50775, remove Int64Index, UInt64Index & Float64Index from codabase
    ("pandas.core.indexes.numeric", "Int64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.indexes.numeric", "UInt64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.indexes.numeric", "Float64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.arrays.sparse.dtype", "SparseDtype"): (
        "pandas.core.dtypes.dtypes",
        "SparseDtype",
    ),
}


# our Unpickler sub-class to override methods and some dispatcher
# functions for compat and uses a non-public class of the pickle module.


class Unpickler(pkl._Unpickler):
    def find_class(self, module, name):
        # override superclass
        key = (module, name)
        module, name = _class_locations_map.get(key, key)
        return super().find_class(module, name)


Unpickler.dispatch = copy.copy(Unpickler.dispatch)
Unpickler.dispatch[pkl.REDUCE[0]] = load_reduce


def load_newobj(self) -> None:
    args = self.stack.pop()
    cls = self.stack[-1]

    # compat
    if issubclass(cls, Index):
        obj = object.__new__(cls)
    elif issubclass(cls, DatetimeArray) and not args:
        arr = np.array([], dtype="M8[ns]")
        obj = cls.__new__(cls, arr, arr.dtype)
    elif issubclass(cls, TimedeltaArray) and not args:
        arr = np.array([], dtype="m8[ns]")
        obj = cls.__new__(cls, arr, arr.dtype)
    elif cls is BlockManager and not args:
        obj = cls.__new__(cls, (), [], False)
    else:
        obj = cls.__new__(cls, *args)

    self.stack[-1] = obj


Unpickler.dispatch[pkl.NEWOBJ[0]] = load_newobj


def load_newobj_ex(self) -> None:
    kwargs = self.stack.pop()
    args = self.stack.pop()
    cls = self.stack.pop()

    # compat
    if issubclass(cls, Index):
        obj = object.__new__(cls)
    else:
        obj = cls.__new__(cls, *args, **kwargs)
    self.append(obj)


try:
    Unpickler.dispatch[pkl.NEWOBJ_EX[0]] = load_newobj_ex
except (AttributeError, KeyError):
    pass


def load(fh, encoding: str | None = None, is_verbose: bool = False):
    """
    Load a pickle, with a provided encoding,

    Parameters
    ----------
    fh : a filelike object
    encoding : an optional encoding
    is_verbose : show exception output
    """
    try:
        fh.seek(0)
        if encoding is not None:
            up = Unpickler(fh, encoding=encoding)
        else:
            up = Unpickler(fh)
        # "Unpickler" has no attribute "is_verbose"  [attr-defined]
        up.is_verbose = is_verbose  # type: ignore[attr-defined]

        return up.load()
    except (ValueError, TypeError):
        raise


def loads(
    bytes_object: bytes,
    *,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
):
    """
    Analogous to pickle._loads.
    """
    fd = io.BytesIO(bytes_object)
    return Unpickler(
        fd, fix_imports=fix_imports, encoding=encoding, errors=errors
    ).load()


@contextlib.contextmanager
def patch_pickle() -> Generator[None, None, None]:
    """
    Temporarily patch pickle to use our unpickler.
    """
    orig_loads = pkl.loads
    try:
        setattr(pkl, "loads", loads)
        yield
    finally:
        setattr(pkl, "loads", orig_loads)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\_manylinux.py ===
from __future__ import annotations

import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Generator, Iterator, NamedTuple, Sequence

from ._elffile import EIClass, EIData, ELFFile, EMachine

EF_ARM_ABIMASK = 0xFF000000
EF_ARM_ABI_VER5 = 0x05000000
EF_ARM_ABI_FLOAT_HARD = 0x00000400


# `os.PathLike` not a generic type until Python 3.9, so sticking with `str`
# as the type for `path` until then.
@contextlib.contextmanager
def _parse_elf(path: str) -> Generator[ELFFile | None, None, None]:
    try:
        with open(path, "rb") as f:
            yield ELFFile(f)
    except (OSError, TypeError, ValueError):
        yield None


def _is_linux_armhf(executable: str) -> bool:
    # hard-float ABI can be detected from the ELF header of the running
    # process
    # https://static.docs.arm.com/ihi0044/g/aaelf32.pdf
    with _parse_elf(executable) as f:
        return (
            f is not None
            and f.capacity == EIClass.C32
            and f.encoding == EIData.Lsb
            and f.machine == EMachine.Arm
            and f.flags & EF_ARM_ABIMASK == EF_ARM_ABI_VER5
            and f.flags & EF_ARM_ABI_FLOAT_HARD == EF_ARM_ABI_FLOAT_HARD
        )


def _is_linux_i686(executable: str) -> bool:
    with _parse_elf(executable) as f:
        return (
            f is not None
            and f.capacity == EIClass.C32
            and f.encoding == EIData.Lsb
            and f.machine == EMachine.I386
        )


def _have_compatible_abi(executable: str, archs: Sequence[str]) -> bool:
    if "armv7l" in archs:
        return _is_linux_armhf(executable)
    if "i686" in archs:
        return _is_linux_i686(executable)
    allowed_archs = {
        "x86_64",
        "aarch64",
        "ppc64",
        "ppc64le",
        "s390x",
        "loongarch64",
        "riscv64",
    }
    return any(arch in allowed_archs for arch in archs)


# If glibc ever changes its major version, we need to know what the last
# minor version was, so we can build the complete list of all versions.
# For now, guess what the highest minor version might be, assume it will
# be 50 for testing. Once this actually happens, update the dictionary
# with the actual value.
_LAST_GLIBC_MINOR: dict[int, int] = collections.defaultdict(lambda: 50)


class _GLibCVersion(NamedTuple):
    major: int
    minor: int


def _glibc_version_string_confstr() -> str | None:
    """
    Primary implementation of glibc_version_string using os.confstr.
    """
    # os.confstr is quite a bit faster than ctypes.DLL. It's also less likely
    # to be broken or missing. This strategy is used in the standard library
    # platform module.
    # https://github.com/python/cpython/blob/fcf1d003bf4f0100c/Lib/platform.py#L175-L183
    try:
        # Should be a string like "glibc 2.17".
        version_string: str | None = os.confstr("CS_GNU_LIBC_VERSION")
        assert version_string is not None
        _, version = version_string.rsplit()
    except (AssertionError, AttributeError, OSError, ValueError):
        # os.confstr() or CS_GNU_LIBC_VERSION not available (or a bad value)...
        return None
    return version


def _glibc_version_string_ctypes() -> str | None:
    """
    Fallback implementation of glibc_version_string using ctypes.
    """
    try:
        import ctypes
    except ImportError:
        return None

    # ctypes.CDLL(None) internally calls dlopen(NULL), and as the dlopen
    # manpage says, "If filename is NULL, then the returned handle is for the
    # main program". This way we can let the linker do the work to figure out
    # which libc our process is actually using.
    #
    # We must also handle the special case where the executable is not a
    # dynamically linked executable. This can occur when using musl libc,
    # for example. In this situation, dlopen() will error, leading to an
    # OSError. Interestingly, at least in the case of musl, there is no
    # errno set on the OSError. The single string argument used to construct
    # OSError comes from libc itself and is therefore not portable to
    # hard code here. In any case, failure to call dlopen() means we
    # can proceed, so we bail on our attempt.
    try:
        process_namespace = ctypes.CDLL(None)
    except OSError:
        return None

    try:
        gnu_get_libc_version = process_namespace.gnu_get_libc_version
    except AttributeError:
        # Symbol doesn't exist -> therefore, we are not linked to
        # glibc.
        return None

    # Call gnu_get_libc_version, which returns a string like "2.5"
    gnu_get_libc_version.restype = ctypes.c_char_p
    version_str: str = gnu_get_libc_version()
    # py2 / py3 compatibility:
    if not isinstance(version_str, str):
        version_str = version_str.decode("ascii")

    return version_str


def _glibc_version_string() -> str | None:
    """Returns glibc version string, or None if not using glibc."""
    return _glibc_version_string_confstr() or _glibc_version_string_ctypes()


def _parse_glibc_version(version_str: str) -> tuple[int, int]:
    """Parse glibc version.

    We use a regexp instead of str.split because we want to discard any
    random junk that might come after the minor version -- this might happen
    in patched/forked versions of glibc (e.g. Linaro's version of glibc
    uses version strings like "2.20-2014.11"). See gh-3588.
    """
    m = re.match(r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)", version_str)
    if not m:
        warnings.warn(
            f"Expected glibc version with 2 components major.minor, got: {version_str}",
            RuntimeWarning,
            stacklevel=2,
        )
        return -1, -1
    return int(m.group("major")), int(m.group("minor"))


@functools.lru_cache
def _get_glibc_version() -> tuple[int, int]:
    version_str = _glibc_version_string()
    if version_str is None:
        return (-1, -1)
    return _parse_glibc_version(version_str)


# From PEP 513, PEP 600
def _is_compatible(arch: str, version: _GLibCVersion) -> bool:
    sys_glibc = _get_glibc_version()
    if sys_glibc < version:
        return False
    # Check for presence of _manylinux module.
    try:
        import _manylinux
    except ImportError:
        return True
    if hasattr(_manylinux, "manylinux_compatible"):
        result = _manylinux.manylinux_compatible(version[0], version[1], arch)
        if result is not None:
            return bool(result)
        return True
    if version == _GLibCVersion(2, 5):
        if hasattr(_manylinux, "manylinux1_compatible"):
            return bool(_manylinux.manylinux1_compatible)
    if version == _GLibCVersion(2, 12):
        if hasattr(_manylinux, "manylinux2010_compatible"):
            return bool(_manylinux.manylinux2010_compatible)
    if version == _GLibCVersion(2, 17):
        if hasattr(_manylinux, "manylinux2014_compatible"):
            return bool(_manylinux.manylinux2014_compatible)
    return True


_LEGACY_MANYLINUX_MAP = {
    # CentOS 7 w/ glibc 2.17 (PEP 599)
    (2, 17): "manylinux2014",
    # CentOS 6 w/ glibc 2.12 (PEP 571)
    (2, 12): "manylinux2010",
    # CentOS 5 w/ glibc 2.5 (PEP 513)
    (2, 5): "manylinux1",
}


def platform_tags(archs: Sequence[str]) -> Iterator[str]:
    """Generate manylinux tags compatible to the current platform.

    :param archs: Sequence of compatible architectures.
        The first one shall be the closest to the actual architecture and be the part of
        platform tag after the ``linux_`` prefix, e.g. ``x86_64``.
        The ``linux_`` prefix is assumed as a prerequisite for the current platform to
        be manylinux-compatible.

    :returns: An iterator of compatible manylinux tags.
    """
    if not _have_compatible_abi(sys.executable, archs):
        return
    # Oldest glibc to be supported regardless of architecture is (2, 17).
    too_old_glibc2 = _GLibCVersion(2, 16)
    if set(archs) & {"x86_64", "i686"}:
        # On x86/i686 also oldest glibc to be supported is (2, 5).
        too_old_glibc2 = _GLibCVersion(2, 4)
    current_glibc = _GLibCVersion(*_get_glibc_version())
    glibc_max_list = [current_glibc]
    # We can assume compatibility across glibc major versions.
    # https://sourceware.org/bugzilla/show_bug.cgi?id=24636
    #
    # Build a list of maximum glibc versions so that we can
    # output the canonical list of all glibc from current_glibc
    # down to too_old_glibc2, including all intermediary versions.
    for glibc_major in range(current_glibc.major - 1, 1, -1):
        glibc_minor = _LAST_GLIBC_MINOR[glibc_major]
        glibc_max_list.append(_GLibCVersion(glibc_major, glibc_minor))
    for arch in archs:
        for glibc_max in glibc_max_list:
            if glibc_max.major == too_old_glibc2.major:
                min_minor = too_old_glibc2.minor
            else:
                # For other glibc major versions oldest supported is (x, 0).
                min_minor = -1
            for glibc_minor in range(glibc_max.minor, min_minor, -1):
                glibc_version = _GLibCVersion(glibc_max.major, glibc_minor)
                tag = "manylinux_{}_{}".format(*glibc_version)
                if _is_compatible(arch, glibc_version):
                    yield f"{tag}_{arch}"
                # Handle the legacy manylinux1, manylinux2010, manylinux2014 tags.
                if glibc_version in _LEGACY_MANYLINUX_MAP:
                    legacy_tag = _LEGACY_MANYLINUX_MAP[glibc_version]
                    if _is_compatible(arch, glibc_version):
                        yield f"{legacy_tag}_{arch}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\sets\handlers\functions.py ===
from sympy.core.singleton import S
from sympy.sets.sets import Set
from sympy.calculus.singularities import singularities
from sympy.core import Expr, Add
from sympy.core.function import Lambda, FunctionClass, diff, expand_mul
from sympy.core.numbers import Float, oo
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import true
from sympy.multipledispatch import Dispatcher
from sympy.sets import (imageset, Interval, FiniteSet, Union, ImageSet,
    Intersection, Range, Complement)
from sympy.sets.sets import EmptySet, is_function_invertible_in_set
from sympy.sets.fancysets import Integers, Naturals, Reals
from sympy.functions.elementary.exponential import match_real_imag


_x, _y = symbols("x y")

FunctionUnion = (FunctionClass, Lambda)

_set_function = Dispatcher('_set_function')


@_set_function.register(FunctionClass, Set)
def _(f, x):
    return None

@_set_function.register(FunctionUnion, FiniteSet)
def _(f, x):
    return FiniteSet(*map(f, x))

@_set_function.register(Lambda, Interval)
def _(f, x):
    from sympy.solvers.solveset import solveset
    from sympy.series import limit
    # TODO: handle functions with infinitely many solutions (eg, sin, tan)
    # TODO: handle multivariate functions

    expr = f.expr
    if len(expr.free_symbols) > 1 or len(f.variables) != 1:
        return
    var = f.variables[0]
    if not var.is_real:
        if expr.subs(var, Dummy(real=True)).is_real is False:
            return

    if expr.is_Piecewise:
        result = S.EmptySet
        domain_set = x
        for (p_expr, p_cond) in expr.args:
            if p_cond is true:
                intrvl = domain_set
            else:
                intrvl = p_cond.as_set()
                intrvl = Intersection(domain_set, intrvl)

            if p_expr.is_Number:
                image = FiniteSet(p_expr)
            else:
                image = imageset(Lambda(var, p_expr), intrvl)
            result = Union(result, image)

            # remove the part which has been `imaged`
            domain_set = Complement(domain_set, intrvl)
            if domain_set is S.EmptySet:
                break
        return result

    if not x.start.is_comparable or not x.end.is_comparable:
        return

    try:
        from sympy.polys.polyutils import _nsort
        sing = list(singularities(expr, var, x))
        if len(sing) > 1:
            sing = _nsort(sing)
    except NotImplementedError:
        return

    if x.left_open:
        _start = limit(expr, var, x.start, dir="+")
    elif x.start not in sing:
        _start = f(x.start)
    if x.right_open:
        _end = limit(expr, var, x.end, dir="-")
    elif x.end not in sing:
        _end = f(x.end)

    if len(sing) == 0:
        soln_expr = solveset(diff(expr, var), var)
        if not (isinstance(soln_expr, FiniteSet)
                or soln_expr is S.EmptySet):
            return
        solns = list(soln_expr)

        extr = [_start, _end] + [f(i) for i in solns
                                 if i.is_real and i in x]
        start, end = Min(*extr), Max(*extr)

        left_open, right_open = False, False
        if _start <= _end:
            # the minimum or maximum value can occur simultaneously
            # on both the edge of the interval and in some interior
            # point
            if start == _start and start not in solns:
                left_open = x.left_open
            if end == _end and end not in solns:
                right_open = x.right_open
        else:
            if start == _end and start not in solns:
                left_open = x.right_open
            if end == _start and end not in solns:
                right_open = x.left_open

        return Interval(start, end, left_open, right_open)
    else:
        return imageset(f, Interval(x.start, sing[0],
                                    x.left_open, True)) + \
            Union(*[imageset(f, Interval(sing[i], sing[i + 1], True, True))
                    for i in range(0, len(sing) - 1)]) + \
            imageset(f, Interval(sing[-1], x.end, True, x.right_open))

@_set_function.register(FunctionClass, Interval)
def _(f, x):
    if f == exp:
        return Interval(exp(x.start), exp(x.end), x.left_open, x.right_open)
    elif f == log:
        return Interval(log(x.start), log(x.end), x.left_open, x.right_open)
    return ImageSet(Lambda(_x, f(_x)), x)

@_set_function.register(FunctionUnion, Union)
def _(f, x):
    return Union(*(imageset(f, arg) for arg in x.args))

@_set_function.register(FunctionUnion, Intersection)
def _(f, x):
    # If the function is invertible, intersect the maps of the sets.
    if is_function_invertible_in_set(f, x):
        return Intersection(*(imageset(f, arg) for arg in x.args))
    else:
        return ImageSet(Lambda(_x, f(_x)), x)

@_set_function.register(FunctionUnion, EmptySet)
def _(f, x):
    return x

@_set_function.register(FunctionUnion, Set)
def _(f, x):
    return ImageSet(Lambda(_x, f(_x)), x)

@_set_function.register(FunctionUnion, Range)
def _(f, self):
    if not self:
        return S.EmptySet
    if not isinstance(f.expr, Expr):
        return
    if self.size == 1:
        return FiniteSet(f(self[0]))
    if f is S.IdentityFunction:
        return self

    x = f.variables[0]
    expr = f.expr
    # handle f that is linear in f's variable
    if x not in expr.free_symbols or x in expr.diff(x).free_symbols:
        return
    if self.start.is_finite:
        F = f(self.step*x + self.start)  # for i in range(len(self))
    else:
        F = f(-self.step*x + self[-1])
    F = expand_mul(F)
    if F != expr:
        return imageset(x, F, Range(self.size))

@_set_function.register(FunctionUnion, Integers)
def _(f, self):
    expr = f.expr
    if not isinstance(expr, Expr):
        return

    n = f.variables[0]
    if expr == abs(n):
        return S.Naturals0

    # f(x) + c and f(-x) + c cover the same integers
    # so choose the form that has the fewest negatives
    c = f(0)
    fx = f(n) - c
    f_x = f(-n) - c
    neg_count = lambda e: sum(_.could_extract_minus_sign()
        for _ in Add.make_args(e))
    if neg_count(f_x) < neg_count(fx):
        expr = f_x + c

    a = Wild('a', exclude=[n])
    b = Wild('b', exclude=[n])
    match = expr.match(a*n + b)
    if match and match[a] and (
            not match[a].atoms(Float) and
            not match[b].atoms(Float)):
        # canonical shift
        a, b = match[a], match[b]
        if a in [1, -1]:
            # drop integer addends in b
            nonint = []
            for bi in Add.make_args(b):
                if not bi.is_integer:
                    nonint.append(bi)
            b = Add(*nonint)
        if b.is_number and a.is_real:
            # avoid Mod for complex numbers, #11391
            br, bi = match_real_imag(b)
            if br and br.is_comparable and a.is_comparable:
                br %= a
                b = br + S.ImaginaryUnit*bi
        elif b.is_number and a.is_imaginary:
            br, bi = match_real_imag(b)
            ai = a/S.ImaginaryUnit
            if bi and bi.is_comparable and ai.is_comparable:
                bi %= ai
                b = br + S.ImaginaryUnit*bi
        expr = a*n + b

    if expr != f.expr:
        return ImageSet(Lambda(n, expr), S.Integers)


@_set_function.register(FunctionUnion, Naturals)
def _(f, self):
    expr = f.expr
    if not isinstance(expr, Expr):
        return

    x = f.variables[0]
    if not expr.free_symbols - {x}:
        if expr == abs(x):
            if self is S.Naturals:
                return self
            return S.Naturals0
        step = expr.coeff(x)
        c = expr.subs(x, 0)
        if c.is_Integer and step.is_Integer and expr == step*x + c:
            if self is S.Naturals:
                c += step
            if step > 0:
                if step == 1:
                    if c == 0:
                        return S.Naturals0
                    elif c == 1:
                        return S.Naturals
                return Range(c, oo, step)
            return Range(c, -oo, step)


@_set_function.register(FunctionUnion, Reals)
def _(f, self):
    expr = f.expr
    if not isinstance(expr, Expr):
        return
    return _set_function(f, Interval(-oo, oo))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\benchmarks\bench_meijerint.py ===
# conceal the implicit import from the code quality tester
from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.bessel import besseli
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.integrals.transforms import (mellin_transform,
    inverse_fourier_transform, inverse_mellin_transform,
    laplace_transform, inverse_laplace_transform, fourier_transform)

LT = laplace_transform
FT = fourier_transform
MT = mellin_transform
IFT = inverse_fourier_transform
ILT = inverse_laplace_transform
IMT = inverse_mellin_transform

from sympy.abc import x, y
nu, beta, rho = symbols('nu beta rho')

apos, bpos, cpos, dpos, posk, p = symbols('a b c d k p', positive=True)
k = Symbol('k', real=True)
negk = Symbol('k', negative=True)

mu1, mu2 = symbols('mu1 mu2', real=True, nonzero=True, finite=True)
sigma1, sigma2 = symbols('sigma1 sigma2', real=True, nonzero=True,
                         finite=True, positive=True)
rate = Symbol('lambda', positive=True)


def normal(x, mu, sigma):
    return 1/sqrt(2*pi*sigma**2)*exp(-(x - mu)**2/2/sigma**2)


def exponential(x, rate):
    return rate*exp(-rate*x)
alpha, beta = symbols('alpha beta', positive=True)
betadist = x**(alpha - 1)*(1 + x)**(-alpha - beta)*gamma(alpha + beta) \
    /gamma(alpha)/gamma(beta)
kint = Symbol('k', integer=True, positive=True)
chi = 2**(1 - kint/2)*x**(kint - 1)*exp(-x**2/2)/gamma(kint/2)
chisquared = 2**(-k/2)/gamma(k/2)*x**(k/2 - 1)*exp(-x/2)
dagum = apos*p/x*(x/bpos)**(apos*p)/(1 + x**apos/bpos**apos)**(p + 1)
d1, d2 = symbols('d1 d2', positive=True)
f = sqrt(((d1*x)**d1 * d2**d2)/(d1*x + d2)**(d1 + d2))/x \
    /gamma(d1/2)/gamma(d2/2)*gamma((d1 + d2)/2)
nupos, sigmapos = symbols('nu sigma', positive=True)
rice = x/sigmapos**2*exp(-(x**2 + nupos**2)/2/sigmapos**2)*besseli(0, x*
                         nupos/sigmapos**2)
mu = Symbol('mu', real=True)
laplace = exp(-abs(x - mu)/bpos)/2/bpos

u = Symbol('u', polar=True)
tpos = Symbol('t', positive=True)


def E(expr):
    integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                     (x, 0, oo), (y, -oo, oo), meijerg=True)
    integrate(expr*exponential(x, rate)*normal(y, mu1, sigma1),
                     (y, -oo, oo), (x, 0, oo), meijerg=True)

bench = [
    'MT(x**nu*Heaviside(x - 1), x, s)',
    'MT(x**nu*Heaviside(1 - x), x, s)',
    'MT((1-x)**(beta - 1)*Heaviside(1-x), x, s)',
    'MT((x-1)**(beta - 1)*Heaviside(x-1), x, s)',
    'MT((1+x)**(-rho), x, s)',
    'MT(abs(1-x)**(-rho), x, s)',
    'MT((1-x)**(beta-1)*Heaviside(1-x) + a*(x-1)**(beta-1)*Heaviside(x-1), x, s)',
    'MT((x**a-b**a)/(x-b), x, s)',
    'MT((x**a-bpos**a)/(x-bpos), x, s)',
    'MT(exp(-x), x, s)',
    'MT(exp(-1/x), x, s)',
    'MT(log(x)**4*Heaviside(1-x), x, s)',
    'MT(log(x)**3*Heaviside(x-1), x, s)',
    'MT(log(x + 1), x, s)',
    'MT(log(1/x + 1), x, s)',
    'MT(log(abs(1 - x)), x, s)',
    'MT(log(abs(1 - 1/x)), x, s)',
    'MT(log(x)/(x+1), x, s)',
    'MT(log(x)**2/(x+1), x, s)',
    'MT(log(x)/(x+1)**2, x, s)',
    'MT(erf(sqrt(x)), x, s)',

    'MT(besselj(a, 2*sqrt(x)), x, s)',
    'MT(sin(sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(cos(sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))**2, x, s)',
    'MT(besselj(a, sqrt(x))*besselj(-a, sqrt(x)), x, s)',
    'MT(besselj(a - 1, sqrt(x))*besselj(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*besselj(b, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))**2 + besselj(-a, sqrt(x))**2, x, s)',
    'MT(bessely(a, 2*sqrt(x)), x, s)',
    'MT(sin(sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(cos(sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*bessely(a, sqrt(x)), x, s)',
    'MT(besselj(a, sqrt(x))*bessely(b, sqrt(x)), x, s)',
    'MT(bessely(a, sqrt(x))**2, x, s)',

    'MT(besselk(a, 2*sqrt(x)), x, s)',
    'MT(besselj(a, 2*sqrt(2*sqrt(x)))*besselk(a, 2*sqrt(2*sqrt(x))), x, s)',
    'MT(besseli(a, sqrt(x))*besselk(a, sqrt(x)), x, s)',
    'MT(besseli(b, sqrt(x))*besselk(a, sqrt(x)), x, s)',
    'MT(exp(-x/2)*besselk(a, x/2), x, s)',

    # later: ILT, IMT

    'LT((t-apos)**bpos*exp(-cpos*(t-apos))*Heaviside(t-apos), t, s)',
    'LT(t**apos, t, s)',
    'LT(Heaviside(t), t, s)',
    'LT(Heaviside(t - apos), t, s)',
    'LT(1 - exp(-apos*t), t, s)',
    'LT((exp(2*t)-1)*exp(-bpos - t)*Heaviside(t)/2, t, s, noconds=True)',
    'LT(exp(t), t, s)',
    'LT(exp(2*t), t, s)',
    'LT(exp(apos*t), t, s)',
    'LT(log(t/apos), t, s)',
    'LT(erf(t), t, s)',
    'LT(sin(apos*t), t, s)',
    'LT(cos(apos*t), t, s)',
    'LT(exp(-apos*t)*sin(bpos*t), t, s)',
    'LT(exp(-apos*t)*cos(bpos*t), t, s)',
    'LT(besselj(0, t), t, s, noconds=True)',
    'LT(besselj(1, t), t, s, noconds=True)',

    'FT(Heaviside(1 - abs(2*apos*x)), x, k)',
    'FT(Heaviside(1-abs(apos*x))*(1-abs(apos*x)), x, k)',
    'FT(exp(-apos*x)*Heaviside(x), x, k)',
    'IFT(1/(apos + 2*pi*I*x), x, posk, noconds=False)',
    'IFT(1/(apos + 2*pi*I*x), x, -posk, noconds=False)',
    'IFT(1/(apos + 2*pi*I*x), x, negk)',
    'FT(x*exp(-apos*x)*Heaviside(x), x, k)',
    'FT(exp(-apos*x)*sin(bpos*x)*Heaviside(x), x, k)',
    'FT(exp(-apos*x**2), x, k)',
    'IFT(sqrt(pi/apos)*exp(-(pi*k)**2/apos), k, x)',
    'FT(exp(-apos*abs(x)), x, k)',

    'integrate(normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x**2*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(x**3*normal(x, mu1, sigma1), (x, -oo, oo), meijerg=True)',
    'integrate(normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(x*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(x*y*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate((x+y+1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate((x+y-1)*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '                   (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(x**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '                (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(y**2*normal(x, mu1, sigma1)*normal(y, mu2, sigma2),'
    '          (x, -oo, oo), (y, -oo, oo), meijerg=True)',
    'integrate(exponential(x, rate), (x, 0, oo), meijerg=True)',
    'integrate(x*exponential(x, rate), (x, 0, oo), meijerg=True)',
    'integrate(x**2*exponential(x, rate), (x, 0, oo), meijerg=True)',
    'E(1)',
    'E(x*y)',
    'E(x*y**2)',
    'E((x+y+1)**2)',
    'E(x+y+1)',
    'E((x+y-1)**2)',
    'integrate(betadist, (x, 0, oo), meijerg=True)',
    'integrate(x*betadist, (x, 0, oo), meijerg=True)',
    'integrate(x**2*betadist, (x, 0, oo), meijerg=True)',
    'integrate(chi, (x, 0, oo), meijerg=True)',
    'integrate(x*chi, (x, 0, oo), meijerg=True)',
    'integrate(x**2*chi, (x, 0, oo), meijerg=True)',
    'integrate(chisquared, (x, 0, oo), meijerg=True)',
    'integrate(x*chisquared, (x, 0, oo), meijerg=True)',
    'integrate(x**2*chisquared, (x, 0, oo), meijerg=True)',
    'integrate(((x-k)/sqrt(2*k))**3*chisquared, (x, 0, oo), meijerg=True)',
    'integrate(dagum, (x, 0, oo), meijerg=True)',
    'integrate(x*dagum, (x, 0, oo), meijerg=True)',
    'integrate(x**2*dagum, (x, 0, oo), meijerg=True)',
    'integrate(f, (x, 0, oo), meijerg=True)',
    'integrate(x*f, (x, 0, oo), meijerg=True)',
    'integrate(x**2*f, (x, 0, oo), meijerg=True)',
    'integrate(rice, (x, 0, oo), meijerg=True)',
    'integrate(laplace, (x, -oo, oo), meijerg=True)',
    'integrate(x*laplace, (x, -oo, oo), meijerg=True)',
    'integrate(x**2*laplace, (x, -oo, oo), meijerg=True)',
    'integrate(log(x) * x**(k-1) * exp(-x) / gamma(k), (x, 0, oo))',

    'integrate(sin(z*x)*(x**2-1)**(-(y+S(1)/2)), (x, 1, oo), meijerg=True)',
    'integrate(besselj(0,x)*besselj(1,x)*exp(-x**2), (x, 0, oo), meijerg=True)',
    'integrate(besselj(0,x)*besselj(1,x)*besselk(0,x), (x, 0, oo), meijerg=True)',
    'integrate(besselj(0,x)*besselj(1,x)*exp(-x**2), (x, 0, oo), meijerg=True)',
    'integrate(besselj(a,x)*besselj(b,x)/x, (x,0,oo), meijerg=True)',

    'hyperexpand(meijerg((-s - a/2 + 1, -s + a/2 + 1), (-a/2 - S(1)/2, -s + a/2 + S(3)/2), (a/2, -a/2), (-a/2 - S(1)/2, -s + a/2 + S(3)/2), 1))',
    "gammasimp(S('2**(2*s)*(-pi*gamma(-a + 1)*gamma(a + 1)*gamma(-a - s + 1)*gamma(-a + s - 1/2)*gamma(a - s + 3/2)*gamma(a + s + 1)/(a*(a + s)) - gamma(-a - 1/2)*gamma(-a + 1)*gamma(a + 1)*gamma(a + 3/2)*gamma(-s + 3/2)*gamma(s - 1/2)*gamma(-a + s + 1)*gamma(a - s + 1)/(a*(-a + s)))*gamma(-2*s + 1)*gamma(s + 1)/(pi*s*gamma(-a - 1/2)*gamma(a + 3/2)*gamma(-s + 1)*gamma(-s + 3/2)*gamma(s - 1/2)*gamma(-a - s + 1)*gamma(-a + s - 1/2)*gamma(a - s + 1)*gamma(a - s + 3/2))'))",

    'mellin_transform(E1(x), x, s)',
    'inverse_mellin_transform(gamma(s)/s, s, x, (0, oo))',
    'mellin_transform(expint(a, x), x, s)',
    'mellin_transform(Si(x), x, s)',
    'inverse_mellin_transform(-2**s*sqrt(pi)*gamma((s + 1)/2)/(2*s*gamma(-s/2 + 1)), s, x, (-1, 0))',
    'mellin_transform(Ci(sqrt(x)), x, s)',
    'inverse_mellin_transform(-4**s*sqrt(pi)*gamma(s)/(2*s*gamma(-s + S(1)/2)),s, u, (0, 1))',
    'laplace_transform(Ci(x), x, s)',
    'laplace_transform(expint(a, x), x, s)',
    'laplace_transform(expint(1, x), x, s)',
    'laplace_transform(expint(2, x), x, s)',
    'inverse_laplace_transform(-log(1 + s**2)/2/s, s, u)',
    'inverse_laplace_transform(log(s + 1)/s, s, x)',
    'inverse_laplace_transform((s - log(s + 1))/s**2, s, x)',
    'laplace_transform(Chi(x), x, s)',
    'laplace_transform(Shi(x), x, s)',

    'integrate(exp(-z*x)/x, (x, 1, oo), meijerg=True, conds="none")',
    'integrate(exp(-z*x)/x**2, (x, 1, oo), meijerg=True, conds="none")',
    'integrate(exp(-z*x)/x**3, (x, 1, oo), meijerg=True,conds="none")',
    'integrate(-cos(x)/x, (x, tpos, oo), meijerg=True)',
    'integrate(-sin(x)/x, (x, tpos, oo), meijerg=True)',
    'integrate(sin(x)/x, (x, 0, z), meijerg=True)',
    'integrate(sinh(x)/x, (x, 0, z), meijerg=True)',
    'integrate(exp(-x)/x, x, meijerg=True)',
    'integrate(exp(-x)/x**2, x, meijerg=True)',
    'integrate(cos(u)/u, u, meijerg=True)',
    'integrate(cosh(u)/u, u, meijerg=True)',
    'integrate(expint(1, x), x, meijerg=True)',
    'integrate(expint(2, x), x, meijerg=True)',
    'integrate(Si(x), x, meijerg=True)',
    'integrate(Ci(u), u, meijerg=True)',
    'integrate(Shi(x), x, meijerg=True)',
    'integrate(Chi(u), u, meijerg=True)',
    'integrate(Si(x)*exp(-x), (x, 0, oo), meijerg=True)',
    'integrate(expint(1, x)*sin(x), (x, 0, oo), meijerg=True)'
]

from time import time
from sympy.core.cache import clear_cache
import sys

timings = []

if __name__ == '__main__':
    for n, string in enumerate(bench):
        clear_cache()
        _t = time()
        exec(string)
        _t = time() - _t
        timings += [(_t, string)]
        sys.stdout.write('.')
        sys.stdout.flush()
        if n % (len(bench) // 10) == 0:
            sys.stdout.write('%s' % (10*n // len(bench)))
    print()

    timings.sort(key=lambda x: -x[0])

    for ti, string in timings:
        print('%.2fs %s' % (ti, string))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\_trigonometric_special.py ===
r"""A module for special angle formulas for trigonometric functions

TODO
====

This module should be developed in the future to contain direct square root
representation of

.. math
    F(\frac{n}{m} \pi)

for every

- $m \in \{ 3, 5, 17, 257, 65537 \}$
- $n \in \mathbb{N}$, $0 \le n < m$
- $F \in \{\sin, \cos, \tan, \csc, \sec, \cot\}$

Without multi-step rewrites
(e.g. $\tan \to \cos/\sin \to \cos/\sqrt \to \ sqrt$)
or using chebyshev identities
(e.g. $\cos \to \cos + \cos^2 + \cdots \to \sqrt{} + \sqrt{}^2 + \cdots $),
which are trivial to implement in sympy,
and had used to give overly complicated expressions.

The reference can be found below, if anyone may need help implementing them.

References
==========

.. [*] Gottlieb, Christian. (1999). The Simple and straightforward construction
   of the regular 257-gon. The Mathematical Intelligencer. 21. 31-37.
   10.1007/BF03024829.
.. [*] https://resources.wolframcloud.com/FunctionRepository/resources/Cos2PiOverFermatPrime
"""
from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.intfunc import igcdex
from sympy.core.numbers import Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit


def migcdex(*x: int) -> tuple[tuple[int, ...], int]:
    r"""Compute extended gcd for multiple integers.

    Explanation
    ===========

    Given the integers $x_1, \cdots, x_n$ and
    an extended gcd for multiple arguments are defined as a solution
    $(y_1, \cdots, y_n), g$ for the diophantine equation
    $x_1 y_1 + \cdots + x_n y_n = g$ such that
    $g = \gcd(x_1, \cdots, x_n)$.

    Examples
    ========

    >>> from sympy.functions.elementary._trigonometric_special import migcdex
    >>> migcdex()
    ((), 0)
    >>> migcdex(4)
    ((1,), 4)
    >>> migcdex(4, 6)
    ((-1, 1), 2)
    >>> migcdex(6, 10, 15)
    ((1, 1, -1), 1)
    """
    if not x:
        return (), 0

    if len(x) == 1:
        return (1,), x[0]

    if len(x) == 2:
        u, v, h = igcdex(x[0], x[1])
        return (u, v), h

    y, g = migcdex(*x[1:])
    u, v, h = igcdex(x[0], g)
    return (u, *(v * i for i in y)), h


def ipartfrac(*denoms: int) -> tuple[int, ...]:
    r"""Compute the partial fraction decomposition.

    Explanation
    ===========

    Given a rational number $\frac{1}{q_1 \cdots q_n}$ where all
    $q_1, \cdots, q_n$ are pairwise coprime,

    A partial fraction decomposition is defined as

    .. math::
        \frac{1}{q_1 \cdots q_n} = \frac{p_1}{q_1} + \cdots + \frac{p_n}{q_n}

    And it can be derived from solving the following diophantine equation for
    the $p_1, \cdots, p_n$

    .. math::
        1 = p_1 \prod_{i \ne 1}q_i + \cdots + p_n \prod_{i \ne n}q_i

    Where $q_1, \cdots, q_n$ being pairwise coprime implies
    $\gcd(\prod_{i \ne 1}q_i, \cdots, \prod_{i \ne n}q_i) = 1$,
    which guarantees the existence of the solution.

    It is sufficient to compute partial fraction decomposition only
    for numerator $1$ because partial fraction decomposition for any
    $\frac{n}{q_1 \cdots q_n}$ can be easily computed by multiplying
    the result by $n$ afterwards.

    Parameters
    ==========

    denoms : int
        The pairwise coprime integer denominators $q_i$ which defines the
        rational number $\frac{1}{q_1 \cdots q_n}$

    Returns
    =======

    tuple[int, ...]
        The list of numerators which semantically corresponds to $p_i$ of the
        partial fraction decomposition
        $\frac{1}{q_1 \cdots q_n} = \frac{p_1}{q_1} + \cdots + \frac{p_n}{q_n}$

    Examples
    ========

    >>> from sympy import Rational, Mul
    >>> from sympy.functions.elementary._trigonometric_special import ipartfrac

    >>> denoms = 2, 3, 5
    >>> numers = ipartfrac(2, 3, 5)
    >>> numers
    (1, 7, -14)

    >>> Rational(1, Mul(*denoms))
    1/30
    >>> out = 0
    >>> for n, d in zip(numers, denoms):
    ...    out += Rational(n, d)
    >>> out
    1/30
    """
    if not denoms:
        return ()

    def mul(x: int, y: int) -> int:
        return x * y

    denom = reduce(mul, denoms)
    a = [denom // x for x in denoms]
    h, _ = migcdex(*a)
    return h


def fermat_coords(n: int) -> list[int] | None:
    """If n can be factored in terms of Fermat primes with
    multiplicity of each being 1, return those primes, else
    None
    """
    primes = []
    for p in [3, 5, 17, 257, 65537]:
        quotient, remainder = divmod(n, p)
        if remainder == 0:
            n = quotient
            primes.append(p)
            if n == 1:
                return primes
    return None


@cacheit
def cos_3() -> Expr:
    r"""Computes $\cos \frac{\pi}{3}$ in square roots"""
    return S.Half


@cacheit
def cos_5() -> Expr:
    r"""Computes $\cos \frac{\pi}{5}$ in square roots"""
    return (sqrt(5) + 1) / 4


@cacheit
def cos_17() -> Expr:
    r"""Computes $\cos \frac{\pi}{17}$ in square roots"""
    return sqrt(
        (15 + sqrt(17)) / 32 + sqrt(2) * (sqrt(17 - sqrt(17)) +
        sqrt(sqrt(2) * (-8 * sqrt(17 + sqrt(17)) - (1 - sqrt(17))
        * sqrt(17 - sqrt(17))) + 6 * sqrt(17) + 34)) / 32)


@cacheit
def cos_257() -> Expr:
    r"""Computes $\cos \frac{\pi}{257}$ in square roots

    References
    ==========

    .. [*] https://math.stackexchange.com/questions/516142/how-does-cos2-pi-257-look-like-in-real-radicals
    .. [*] https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    """
    def f1(a: Expr, b: Expr) -> tuple[Expr, Expr]:
        return (a + sqrt(a**2 + b)) / 2, (a - sqrt(a**2 + b)) / 2

    def f2(a: Expr, b: Expr) -> Expr:
        return (a - sqrt(a**2 + b))/2

    t1, t2 = f1(S.NegativeOne, Integer(256))
    z1, z3 = f1(t1, Integer(64))
    z2, z4 = f1(t2, Integer(64))
    y1, y5 = f1(z1, 4*(5 + t1 + 2*z1))
    y6, y2 = f1(z2, 4*(5 + t2 + 2*z2))
    y3, y7 = f1(z3, 4*(5 + t1 + 2*z3))
    y8, y4 = f1(z4, 4*(5 + t2 + 2*z4))
    x1, x9 = f1(y1, -4*(t1 + y1 + y3 + 2*y6))
    x2, x10 = f1(y2, -4*(t2 + y2 + y4 + 2*y7))
    x3, x11 = f1(y3, -4*(t1 + y3 + y5 + 2*y8))
    x4, x12 = f1(y4, -4*(t2 + y4 + y6 + 2*y1))
    x5, x13 = f1(y5, -4*(t1 + y5 + y7 + 2*y2))
    x6, x14 = f1(y6, -4*(t2 + y6 + y8 + 2*y3))
    x15, x7 = f1(y7, -4*(t1 + y7 + y1 + 2*y4))
    x8, x16 = f1(y8, -4*(t2 + y8 + y2 + 2*y5))
    v1 = f2(x1, -4*(x1 + x2 + x3 + x6))
    v2 = f2(x2, -4*(x2 + x3 + x4 + x7))
    v3 = f2(x8, -4*(x8 + x9 + x10 + x13))
    v4 = f2(x9, -4*(x9 + x10 + x11 + x14))
    v5 = f2(x10, -4*(x10 + x11 + x12 + x15))
    v6 = f2(x16, -4*(x16 + x1 + x2 + x5))
    u1 = -f2(-v1, -4*(v2 + v3))
    u2 = -f2(-v4, -4*(v5 + v6))
    w1 = -2*f2(-u1, -4*u2)
    return sqrt(sqrt(2)*sqrt(w1 + 4)/8 + S.Half)


def cos_table() -> dict[int, Callable[[], Expr]]:
    r"""Lazily evaluated table for $\cos \frac{\pi}{n}$ in square roots for
    $n \in \{3, 5, 17, 257, 65537\}$.

    Notes
    =====

    65537 is the only other known Fermat prime and it is nearly impossible to
    build in the current SymPy due to performance issues.

    References
    ==========

    https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    """
    return {
        3: cos_3,
        5: cos_5,
        17: cos_17,
        257: cos_257
    }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\routing\converters.py ===
from __future__ import annotations

import re
import typing as t
import uuid
from urllib.parse import quote

if t.TYPE_CHECKING:
    from .map import Map


class ValidationError(ValueError):
    """Validation error.  If a rule converter raises this exception the rule
    does not match the current URL and the next URL is tried.
    """


class BaseConverter:
    """Base class for all converters.

    .. versionchanged:: 2.3
        ``part_isolating`` defaults to ``False`` if ``regex`` contains a ``/``.
    """

    regex = "[^/]+"
    weight = 100
    part_isolating = True

    def __init_subclass__(cls, **kwargs: t.Any) -> None:
        super().__init_subclass__(**kwargs)

        # If the converter isn't inheriting its regex, disable part_isolating by default
        # if the regex contains a / character.
        if "regex" in cls.__dict__ and "part_isolating" not in cls.__dict__:
            cls.part_isolating = "/" not in cls.regex

    def __init__(self, map: Map, *args: t.Any, **kwargs: t.Any) -> None:
        self.map = map

    def to_python(self, value: str) -> t.Any:
        return value

    def to_url(self, value: t.Any) -> str:
        # safe = https://url.spec.whatwg.org/#url-path-segment-string
        return quote(str(value), safe="!$&'()*+,/:;=@")


class UnicodeConverter(BaseConverter):
    """This converter is the default converter and accepts any string but
    only one path segment.  Thus the string can not include a slash.

    This is the default validator.

    Example::

        Rule('/pages/<page>'),
        Rule('/<string(length=2):lang_code>')

    :param map: the :class:`Map`.
    :param minlength: the minimum length of the string.  Must be greater
                      or equal 1.
    :param maxlength: the maximum length of the string.
    :param length: the exact length of the string.
    """

    def __init__(
        self,
        map: Map,
        minlength: int = 1,
        maxlength: int | None = None,
        length: int | None = None,
    ) -> None:
        super().__init__(map)
        if length is not None:
            length_regex = f"{{{int(length)}}}"
        else:
            if maxlength is None:
                maxlength_value = ""
            else:
                maxlength_value = str(int(maxlength))
            length_regex = f"{{{int(minlength)},{maxlength_value}}}"
        self.regex = f"[^/]{length_regex}"


class AnyConverter(BaseConverter):
    """Matches one of the items provided.  Items can either be Python
    identifiers or strings::

        Rule('/<any(about, help, imprint, class, "foo,bar"):page_name>')

    :param map: the :class:`Map`.
    :param items: this function accepts the possible items as positional
                  arguments.

    .. versionchanged:: 2.2
        Value is validated when building a URL.
    """

    def __init__(self, map: Map, *items: str) -> None:
        super().__init__(map)
        self.items = set(items)
        self.regex = f"(?:{'|'.join([re.escape(x) for x in items])})"

    def to_url(self, value: t.Any) -> str:
        if value in self.items:
            return str(value)

        valid_values = ", ".join(f"'{item}'" for item in sorted(self.items))
        raise ValueError(f"'{value}' is not one of {valid_values}")


class PathConverter(BaseConverter):
    """Like the default :class:`UnicodeConverter`, but it also matches
    slashes.  This is useful for wikis and similar applications::

        Rule('/<path:wikipage>')
        Rule('/<path:wikipage>/edit')

    :param map: the :class:`Map`.
    """

    part_isolating = False
    regex = "[^/].*?"
    weight = 200


class NumberConverter(BaseConverter):
    """Baseclass for `IntegerConverter` and `FloatConverter`.

    :internal:
    """

    weight = 50
    num_convert: t.Callable[[t.Any], t.Any] = int

    def __init__(
        self,
        map: Map,
        fixed_digits: int = 0,
        min: int | None = None,
        max: int | None = None,
        signed: bool = False,
    ) -> None:
        if signed:
            self.regex = self.signed_regex
        super().__init__(map)
        self.fixed_digits = fixed_digits
        self.min = min
        self.max = max
        self.signed = signed

    def to_python(self, value: str) -> t.Any:
        if self.fixed_digits and len(value) != self.fixed_digits:
            raise ValidationError()
        value_num = self.num_convert(value)
        if (self.min is not None and value_num < self.min) or (
            self.max is not None and value_num > self.max
        ):
            raise ValidationError()
        return value_num

    def to_url(self, value: t.Any) -> str:
        value_str = str(self.num_convert(value))
        if self.fixed_digits:
            value_str = value_str.zfill(self.fixed_digits)
        return value_str

    @property
    def signed_regex(self) -> str:
        return f"-?{self.regex}"


class IntegerConverter(NumberConverter):
    """This converter only accepts integer values::

        Rule("/page/<int:page>")

    By default it only accepts unsigned, positive values. The ``signed``
    parameter will enable signed, negative values. ::

        Rule("/page/<int(signed=True):page>")

    :param map: The :class:`Map`.
    :param fixed_digits: The number of fixed digits in the URL. If you
        set this to ``4`` for example, the rule will only match if the
        URL looks like ``/0001/``. The default is variable length.
    :param min: The minimal value.
    :param max: The maximal value.
    :param signed: Allow signed (negative) values.

    .. versionadded:: 0.15
        The ``signed`` parameter.
    """

    regex = r"\d+"


class FloatConverter(NumberConverter):
    """This converter only accepts floating point values::

        Rule("/probability/<float:probability>")

    By default it only accepts unsigned, positive values. The ``signed``
    parameter will enable signed, negative values. ::

        Rule("/offset/<float(signed=True):offset>")

    :param map: The :class:`Map`.
    :param min: The minimal value.
    :param max: The maximal value.
    :param signed: Allow signed (negative) values.

    .. versionadded:: 0.15
        The ``signed`` parameter.
    """

    regex = r"\d+\.\d+"
    num_convert = float

    def __init__(
        self,
        map: Map,
        min: float | None = None,
        max: float | None = None,
        signed: bool = False,
    ) -> None:
        super().__init__(map, min=min, max=max, signed=signed)  # type: ignore


class UUIDConverter(BaseConverter):
    """This converter only accepts UUID strings::

        Rule('/object/<uuid:identifier>')

    .. versionadded:: 0.10

    :param map: the :class:`Map`.
    """

    regex = (
        r"[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-"
        r"[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}"
    )

    def to_python(self, value: str) -> uuid.UUID:
        return uuid.UUID(value)

    def to_url(self, value: uuid.UUID) -> str:
        return str(value)


#: the default converter mapping for the map.
DEFAULT_CONVERTERS: t.Mapping[str, type[BaseConverter]] = {
    "default": UnicodeConverter,
    "string": UnicodeConverter,
    "any": AnyConverter,
    "path": PathConverter,
    "int": IntegerConverter,
    "float": FloatConverter,
    "uuid": UUIDConverter,
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\operators\conv.py ===
import numpy as np
import onnx
from onnx import onnx_pb as onnx_proto

from ..quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizedValue,
    QuantizedValueType,
    attribute_to_kwarg,
    find_by_name,
    get_mul_node,
)
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase


class ConvInteger(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def add_bias(self, nodes, scaled_output):
        """
        Given a node, this function handles bias add by adding a "reshape" node on bias and an "add" node
            parameter nodes: new nodes would be appended into nodes
            parameter node: current node (Conv)
            parameter scaled_output: output of quant conv without bias
            parameter output: output of Conv
            parameter bias_name: bias of Conv
            return: the name of output
        """
        node = self.node
        model = self.quantizer.model
        # Add tensors for the shape to be reshaped to
        weight = find_by_name(node.input[1], model.initializer())
        if weight is None:
            raise ValueError(f"Expected {node.input[1]} to be an initializer")

        # Add reshape for correct broadcase
        output = node.output[0]
        reshape_input_data = node.input[2]  # bias of Conv
        reshape_input_shape = output + "_bias_reshape_shape"
        reshape_output = output + "_bias_reshape_output"

        shape = np.ones((len(weight.dims)), dtype=np.int64)
        shape[1] = -1
        init_shape = onnx.helper.make_tensor(
            reshape_input_shape, onnx_proto.TensorProto.INT64, [len(weight.dims)], shape
        )
        model.add_initializer(init_shape)

        reshape_node = onnx.helper.make_node("Reshape", [reshape_input_data, reshape_input_shape], [reshape_output])
        nodes.append(reshape_node)

        # Add an Add operation for bias
        add_node = onnx.helper.make_node("Add", [scaled_output, reshape_output], [output], output + "_bias_add")
        nodes.append(add_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Conv"
        #  Get Quantized from both activation(input[0]) and weight(input[1])
        (
            quantized_input_names,
            zero_point_names,
            scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [0])

        (
            quantized_input_names_weight,
            zero_point_names_weight,
            scale_names_weight,
            nodes_weight,
        ) = self.quantizer.quantize_weight(node, [1], reduce_range=self.quantizer.reduce_range)
        quantized_input_names.extend(quantized_input_names_weight)
        zero_point_names.extend(zero_point_names_weight)
        scale_names.extend(scale_names_weight)
        nodes.extend(nodes_weight)

        conv_integer_output = node.output[0] + "_output_quantized"
        conv_integer_name = node.name + "_quant" if node.name else ""

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        conv_integer_node = onnx.helper.make_node(
            "ConvInteger", quantized_input_names + zero_point_names, [conv_integer_output], conv_integer_name, **kwargs
        )
        nodes.append(conv_integer_node)

        # Add cast operation to cast convInteger output to float.
        onnx_type = self.quantizer.get_tensor_type(node.output[0], mandatory=True)
        cast_op_output = conv_integer_output + "_cast_output"
        cast_node = onnx.helper.make_node(
            "Cast",
            [conv_integer_output],
            [cast_op_output],
            conv_integer_output + "_cast",
            to=onnx_type,  # TODO: FLOAT ot FLOAT16
        )
        nodes.append(cast_node)

        # Add mul operation to multiply scales of two inputs.
        assert len(scale_names) == 2
        if conv_integer_name:
            scales_mul_op = conv_integer_name + "_scales_mul"
        else:
            scales_mul_op = scale_names[0] + "_" + scale_names[1] + "_mul"

        scales_mul_node = find_by_name(scales_mul_op, self.quantizer.new_nodes)
        if scales_mul_node is None:
            scales_mul_node = get_mul_node(scale_names, scales_mul_op + ":0", scales_mul_op)
            nodes.append(scales_mul_node)

        scales_mul_op_output = scales_mul_node.output[0]

        has_bias = len(node.input) == 3
        scaled_output_name = node.output[0] if not has_bias else node.output[0] + "quant_scaled_output"

        # Add mul operation to multiply mul_scales_op result with output of ConvInteger
        # and make the output of this node the same as output of original conv node.
        output_scale_mul_op = conv_integer_name + "_output_scale_mul" if conv_integer_name else ""
        nodes.append(
            get_mul_node(
                [cast_op_output, scales_mul_op_output],
                scaled_output_name,
                output_scale_mul_op,
            )
        )

        if has_bias:
            self.add_bias(nodes, scaled_output_name)

        self.quantizer.new_nodes += nodes


class QLinearConv(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Conv"

        (
            data_found,
            output_scale_name,
            output_zp_name,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0])

        if self.quantizer.is_input_a_initializer(node.input[1]) and self.quantizer.is_per_channel():
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_activation(node, [0])
            quant_weight_tuple = self.quantizer.quantize_weight_per_channel(
                node.input[1],
                onnx_proto.TensorProto.INT8,
                0,  # self.quantizer.weight_qType?
            )
            quantized_input_names.append(quant_weight_tuple[0])
            zero_point_names.append(quant_weight_tuple[1])
            scale_names.append(quant_weight_tuple[2])
        else:
            (
                quantized_input_names,
                zero_point_names,
                scale_names,
                nodes,
            ) = self.quantizer.quantize_activation(node, [0])

            (
                quantized_input_names_weight,
                zero_point_names_weight,
                scale_names_weight,
                nodes_weight,
            ) = self.quantizer.quantize_weight(node, [1], reduce_range=self.quantizer.reduce_range)
            quantized_input_names.extend(quantized_input_names_weight)
            zero_point_names.extend(zero_point_names_weight)
            scale_names.extend(scale_names_weight)
            nodes.extend(nodes_weight)

        if not data_found or quantized_input_names is None:
            return super().quantize()

        quantized_bias_name = ""
        bias_present = False
        if len(node.input) == 3:
            if self.quantizer.weight_qType == onnx_proto.TensorProto.FLOAT8E4M3FN:
                raise RuntimeError("Quantization to FLOAT8E4M3FN for operator Conv is not supported.")
            quantized_bias_name = self.quantizer.quantize_bias_static(node.input[2], node.input[0], node.input[1])
            bias_present = True

        qlinear_conv_output = node.output[0] + TENSOR_NAME_QUANT_SUFFIX
        qlinear_conv_name = node.name + "_quant" if node.name else ""

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        qlinear_conv_inputs = []
        # Input 0
        qlinear_conv_inputs.append(quantized_input_names[0])
        qlinear_conv_inputs.append(scale_names[0])
        qlinear_conv_inputs.append(zero_point_names[0])
        # Input 1
        qlinear_conv_inputs.append(quantized_input_names[1])
        qlinear_conv_inputs.append(scale_names[1])
        qlinear_conv_inputs.append(zero_point_names[1])

        # Output
        qlinear_conv_inputs.append(output_scale_name)
        qlinear_conv_inputs.append(output_zp_name)

        if bias_present:
            qlinear_conv_inputs.append(quantized_bias_name)

        qlinear_conv_node = onnx.helper.make_node(
            "QLinearConv", qlinear_conv_inputs, [qlinear_conv_output], qlinear_conv_name, **kwargs
        )
        nodes.append(qlinear_conv_node)

        # Create an entry for this quantized value
        q_output = QuantizedValue(
            node.output[0],
            qlinear_conv_output,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = q_output

        self.quantizer.new_nodes += nodes


class QDQConv(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert node.op_type == "Conv" or node.op_type == "ConvTranspose"

        self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])

        is_weight_per_channel, weight_axis = self.quantizer.is_tensor_per_channel(
            node.input[1], default_axis=0 if node.op_type == "Conv" else 1
        )
        if is_weight_per_channel:
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], weight_axis)
        else:
            self.quantizer.quantize_weight_tensor(node.input[1])

        if len(node.input) == 3:
            self.quantizer.quantize_bias_tensor(node.name, node.input[2], node.input[0], node.input[1])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_gpt_attention_no_past.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionGptAttentionNoPast(Fusion):
    """
    Fuse GPT-2 Attention without past state into one Attention node.
    This does not support attention_mask graph input right now.
    """

    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, "Attention", ["LayerNormalization", "SkipLayerNormalization"], "without past")
        # TODO: detect num_heads from graph like FusionAttention
        self.num_heads = num_heads
        self.mask_filter_value = None

    def create_attention_node(self, gemm, gemm_qkv, input, output):
        attention_node_name = self.model.create_node_name("Attention")
        attention_node = helper.make_node(
            "Attention",
            inputs=[input, gemm.input[1], gemm.input[2]],
            outputs=[attention_node_name + "_output"],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [
                helper.make_attribute("num_heads", self.num_heads),
                helper.make_attribute("unidirectional", 1),
            ]
        )
        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        matmul_node = helper.make_node(
            "MatMul",
            inputs=[attention_node_name + "_output", gemm_qkv.input[1]],
            outputs=[attention_node_name + "_matmul_output"],
            name=attention_node_name + "_matmul",
        )

        add_node = helper.make_node(
            "Add",
            inputs=[attention_node_name + "_matmul_output", gemm_qkv.input[2]],
            outputs=[output],
            name=attention_node_name + "_add",
        )

        self.nodes_to_add.extend([attention_node, matmul_node, add_node])
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name
        self.node_name_to_graph_name[add_node.name] = self.this_graph_name

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # (TODO) hasesh/tlwu: Investigate what fixes the following logic needs in order
        # to fuse the Attention sub-graph. With some changes to other fusions, this stopped
        # working.
        return_indice = []

        is_normalize_node_skiplayernorm = normalize_node.op_type == "SkipLayerNormalization"
        qkv_nodes = None

        if not is_normalize_node_skiplayernorm:
            qkv_nodes = self.model.match_parent_path(
                normalize_node,
                ["Add", "Reshape", "Gemm", "Reshape", "Reshape", "Transpose", "MatMul"],
                [0, None, 0, 0, 0, 0, 0],
                output_name_to_node=output_name_to_node,
                return_indice=return_indice,
            )
        else:
            qkv_nodes = self.model.match_parent_path(
                normalize_node,
                ["Reshape", "Gemm", "Reshape", "Reshape", "Transpose", "MatMul"],
                [None, 0, 0, 0, 0, 0],
                output_name_to_node=output_name_to_node,
                return_indice=return_indice,
            )

        if qkv_nodes is None:
            return

        another_input = None
        if not is_normalize_node_skiplayernorm:
            (
                add_qkv,
                reshape_qkv,
                gemm_qkv,
                reshape_1,
                reshape_2,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes

            another_input = add_qkv.input[1 - return_indice[0]]
        else:
            (
                reshape_qkv,
                gemm_qkv,
                reshape_1,
                reshape_2,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes

        v_nodes = self.model.match_parent_path(
            matmul_qkv,
            ["Transpose", "Reshape", "Split", "Reshape", "Gemm", "Reshape"],
            [1, 0, 0, 0, 0, 0],
        )
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (
            transpose_v,
            reshape_v,
            split_v,
            reshape_after_gemm,
            gemm,
            reshape_before_gemm,
        ) = v_nodes

        layernorm_before_attention = self.model.get_parent(reshape_before_gemm, 0, output_name_to_node)
        if layernorm_before_attention is None or (
            layernorm_before_attention.op_type != "LayerNormalization"
            and layernorm_before_attention.op_type != "SkipLayerNormalization"
        ):
            if layernorm_before_attention.op_type != "Add":
                logger.debug(f"failed to get (skip)layernorm before gemm. Got {layernorm_before_attention.op_type}")
                return

        # `another_input` will be non-None only if
        # (1) SkipLayerNorm fusion wasn't turned ON
        # (2) SkipLayerNorm fusion was turned ON but upstream layer's LayerNorm + Add was not
        # fused into a SkipLayerNorm. This can happen if the shapes to the Add node are different.
        # So, keep the following check if SkipLayerNorm fusion is turned ON or OFF.
        if another_input is not None:
            if another_input not in layernorm_before_attention.input:
                # match openai-gpt
                if another_input not in layernorm_before_attention.output:
                    logger.debug("Add and (Skip)LayerNormalization shall have one same input")
                    return

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Sub", "Mul", "Div", "MatMul"], [0, 0, 0, 0, 0])
        if qk_nodes is not None:
            (softmax_qk, sub_qk, mul_qk, div_qk, matmul_qk) = qk_nodes
            mask_nodes = self.model.match_parent_path(
                sub_qk,
                [
                    "Mul",
                    "Sub",
                    "Slice",
                    "Slice",
                    "Unsqueeze",
                    "Sub",
                    "Squeeze",
                    "Slice",
                    "Shape",
                    "Div",
                ],
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            )
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match mask path")
                return
            div_mask = mask_nodes[-1]

            if div_qk != div_mask:
                logger.debug("fuse_attention: skip since div_qk != div_mask")
                return
            if len(mask_nodes) > 1 and mask_nodes[0].op_type == "Mul":
                _, mul_val = self.model.get_constant_input(mask_nodes[0])
                if mul_val != -10000:
                    self.mask_filter_value = mul_val

        else:
            # New pattern for gpt2 from PyTorch 1.5.0 and Transformers 2.9.0.
            qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Where", "Div", "MatMul"], [0, 0, 1, 0])
            if qk_nodes is not None:
                (softmax_qk, where_qk, div_qk, matmul_qk) = qk_nodes
                mask_nodes = self.model.match_parent_path(
                    where_qk,
                    [
                        "Cast",
                        "Slice",
                        "Slice",
                        "Unsqueeze",
                        "Sub",
                        "Squeeze",
                        "Slice",
                        "Shape",
                        "Div",
                    ],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                )
                if mask_nodes is None:
                    logger.debug("fuse_attention: failed to match mask path")
                    return
                div_mask = mask_nodes[-1]

                if div_qk != div_mask:
                    logger.debug("fuse_attention: skip since div_qk != div_mask")
                    return
            else:
                # match openai-gpt
                qk_nodes = self.model.match_parent_path(
                    matmul_qkv,
                    ["Softmax", "Add", "Mul", "Div", "MatMul"],
                    [0, 0, 0, 0, 0],
                )
                if qk_nodes is None:
                    logger.debug("fuse_attention: failed to match qk path")
                    return
                (softmax_qk, add_qk, mul_qk, div_qk, matmul_qk) = qk_nodes
                mask_nodes = self.model.match_parent_path(
                    mul_qk,
                    ["Slice", "Slice", "Unsqueeze", "Squeeze", "Slice", "Shape", "Div"],
                    [1, 0, 2, 0, 0, 0, 0],
                )
                if mask_nodes is None:
                    logger.debug("fuse_attention: failed to match mask path")
                    return
                div_mask = mask_nodes[-1]

                if div_qk != div_mask:
                    logger.debug("fuse_attention: skip since div_qk != div_mask")
                    return

        q_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Split"], [0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (transpose_q, reshape_q, split_q) = q_nodes
        if split_v != split_q:
            logger.debug("fuse_attention: skip since split_v != split_q")
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Split"], [1, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return
        (transpose_k, reshape_k, split_k) = k_nodes
        if split_v != split_k:
            logger.debug("fuse_attention: skip since split_v != split_k")
            return

        self.create_attention_node(gemm, gemm_qkv, layernorm_before_attention.output[0], reshape_qkv.output[0])

        # we rely on prune_graph() to clean old subgraph nodes:
        # qk_nodes + q_nodes + k_nodes + v_nodes + mask_nodes + [reshape_qkv, transpose_qkv, matmul_qkv]
        self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\pwa.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: PWA (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target


@dataclass
class FileHandlerAccept:
    '''
    The following types are the replica of
    https://crsrc.org/c/chrome/browser/web_applications/proto/web_app_os_integration_state.proto;drc=9910d3be894c8f142c977ba1023f30a656bc13fc;l=67
    '''
    #: New name of the mimetype according to
    #: https://www.iana.org/assignments/media-types/media-types.xhtml
    media_type: str

    file_extensions: typing.List[str]

    def to_json(self):
        json = dict()
        json['mediaType'] = self.media_type
        json['fileExtensions'] = [i for i in self.file_extensions]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            media_type=str(json['mediaType']),
            file_extensions=[str(i) for i in json['fileExtensions']],
        )


@dataclass
class FileHandler:
    action: str

    accepts: typing.List[FileHandlerAccept]

    display_name: str

    def to_json(self):
        json = dict()
        json['action'] = self.action
        json['accepts'] = [i.to_json() for i in self.accepts]
        json['displayName'] = self.display_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            action=str(json['action']),
            accepts=[FileHandlerAccept.from_json(i) for i in json['accepts']],
            display_name=str(json['displayName']),
        )


class DisplayMode(enum.Enum):
    '''
    If user prefers opening the app in browser or an app window.
    '''
    STANDALONE = "standalone"
    BROWSER = "browser"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def get_os_app_state(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[int, typing.List[FileHandler]]]:
    '''
    Returns the following OS state for the given manifest id.

    :param manifest_id: The id from the webapp's manifest file, commonly it's the url of the site installing the webapp. See https://web.dev/learn/pwa/web-app-manifest.
    :returns: A tuple with the following items:

        0. **badgeCount** - 
        1. **fileHandlers** - 
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.getOsAppState',
        'params': params,
    }
    json = yield cmd_dict
    return (
        int(json['badgeCount']),
        [FileHandler.from_json(i) for i in json['fileHandlers']]
    )


def install(
        manifest_id: str,
        install_url_or_bundle_url: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Installs the given manifest identity, optionally using the given install_url
    or IWA bundle location.

    TODO(crbug.com/337872319) Support IWA to meet the following specific
    requirement.
    IWA-specific install description: If the manifest_id is isolated-app://,
    install_url_or_bundle_url is required, and can be either an http(s) URL or
    file:// URL pointing to a signed web bundle (.swbn). The .swbn file's
    signing key must correspond to manifest_id. If Chrome is not in IWA dev
    mode, the installation will fail, regardless of the state of the allowlist.

    :param manifest_id:
    :param install_url_or_bundle_url: *(Optional)* The location of the app or bundle overriding the one derived from the manifestId.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if install_url_or_bundle_url is not None:
        params['installUrlOrBundleUrl'] = install_url_or_bundle_url
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.install',
        'params': params,
    }
    json = yield cmd_dict


def uninstall(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Uninstalls the given manifest_id and closes any opened app windows.

    :param manifest_id:
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.uninstall',
        'params': params,
    }
    json = yield cmd_dict


def launch(
        manifest_id: str,
        url: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,target.TargetID]:
    '''
    Launches the installed web app, or an url in the same web app instead of the
    default start url if it is provided. Returns a page Target.TargetID which
    can be used to attach to via Target.attachToTarget or similar APIs.

    :param manifest_id:
    :param url: *(Optional)*
    :returns: ID of the tab target created as a result.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if url is not None:
        params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.launch',
        'params': params,
    }
    json = yield cmd_dict
    return target.TargetID.from_json(json['targetId'])


def launch_files_in_app(
        manifest_id: str,
        files: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[target.TargetID]]:
    '''
    Opens one or more local files from an installed web app identified by its
    manifestId. The web app needs to have file handlers registered to process
    the files. The API returns one or more page Target.TargetIDs which can be
    used to attach to via Target.attachToTarget or similar APIs.
    If some files in the parameters cannot be handled by the web app, they will
    be ignored. If none of the files can be handled, this API returns an error.
    If no files are provided as the parameter, this API also returns an error.

    According to the definition of the file handlers in the manifest file, one
    Target.TargetID may represent a page handling one or more files. The order
    of the returned Target.TargetIDs is not guaranteed.

    TODO(crbug.com/339454034): Check the existences of the input files.

    :param manifest_id:
    :param files:
    :returns: IDs of the tab targets created as the result.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    params['files'] = [i for i in files]
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.launchFilesInApp',
        'params': params,
    }
    json = yield cmd_dict
    return [target.TargetID.from_json(i) for i in json['targetIds']]


def open_current_page_in_app(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Opens the current page in its web app identified by the manifest id, needs
    to be called on a page target. This function returns immediately without
    waiting for the app to finish loading.

    :param manifest_id:
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.openCurrentPageInApp',
        'params': params,
    }
    json = yield cmd_dict


def change_app_user_settings(
        manifest_id: str,
        link_capturing: typing.Optional[bool] = None,
        display_mode: typing.Optional[DisplayMode] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes user settings of the web app identified by its manifestId. If the
    app was not installed, this command returns an error. Unset parameters will
    be ignored; unrecognized values will cause an error.

    Unlike the ones defined in the manifest files of the web apps, these
    settings are provided by the browser and controlled by the users, they
    impact the way the browser handling the web apps.

    See the comment of each parameter.

    :param manifest_id:
    :param link_capturing: *(Optional)* If user allows the links clicked on by the user in the app's scope, or extended scope if the manifest has scope extensions and the flags ```DesktopPWAsLinkCapturingWithScopeExtensions```` and ````WebAppEnableScopeExtensions``` are enabled.  Note, the API does not support resetting the linkCapturing to the initial value, uninstalling and installing the web app again will reset it.  TODO(crbug.com/339453269): Setting this value on ChromeOS is not supported yet.
    :param display_mode: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if link_capturing is not None:
        params['linkCapturing'] = link_capturing
    if display_mode is not None:
        params['displayMode'] = display_mode.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.changeAppUserSettings',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\pwa.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: PWA (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target


@dataclass
class FileHandlerAccept:
    '''
    The following types are the replica of
    https://crsrc.org/c/chrome/browser/web_applications/proto/web_app_os_integration_state.proto;drc=9910d3be894c8f142c977ba1023f30a656bc13fc;l=67
    '''
    #: New name of the mimetype according to
    #: https://www.iana.org/assignments/media-types/media-types.xhtml
    media_type: str

    file_extensions: typing.List[str]

    def to_json(self):
        json = dict()
        json['mediaType'] = self.media_type
        json['fileExtensions'] = [i for i in self.file_extensions]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            media_type=str(json['mediaType']),
            file_extensions=[str(i) for i in json['fileExtensions']],
        )


@dataclass
class FileHandler:
    action: str

    accepts: typing.List[FileHandlerAccept]

    display_name: str

    def to_json(self):
        json = dict()
        json['action'] = self.action
        json['accepts'] = [i.to_json() for i in self.accepts]
        json['displayName'] = self.display_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            action=str(json['action']),
            accepts=[FileHandlerAccept.from_json(i) for i in json['accepts']],
            display_name=str(json['displayName']),
        )


class DisplayMode(enum.Enum):
    '''
    If user prefers opening the app in browser or an app window.
    '''
    STANDALONE = "standalone"
    BROWSER = "browser"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def get_os_app_state(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[int, typing.List[FileHandler]]]:
    '''
    Returns the following OS state for the given manifest id.

    :param manifest_id: The id from the webapp's manifest file, commonly it's the url of the site installing the webapp. See https://web.dev/learn/pwa/web-app-manifest.
    :returns: A tuple with the following items:

        0. **badgeCount** - 
        1. **fileHandlers** - 
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.getOsAppState',
        'params': params,
    }
    json = yield cmd_dict
    return (
        int(json['badgeCount']),
        [FileHandler.from_json(i) for i in json['fileHandlers']]
    )


def install(
        manifest_id: str,
        install_url_or_bundle_url: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Installs the given manifest identity, optionally using the given install_url
    or IWA bundle location.

    TODO(crbug.com/337872319) Support IWA to meet the following specific
    requirement.
    IWA-specific install description: If the manifest_id is isolated-app://,
    install_url_or_bundle_url is required, and can be either an http(s) URL or
    file:// URL pointing to a signed web bundle (.swbn). The .swbn file's
    signing key must correspond to manifest_id. If Chrome is not in IWA dev
    mode, the installation will fail, regardless of the state of the allowlist.

    :param manifest_id:
    :param install_url_or_bundle_url: *(Optional)* The location of the app or bundle overriding the one derived from the manifestId.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if install_url_or_bundle_url is not None:
        params['installUrlOrBundleUrl'] = install_url_or_bundle_url
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.install',
        'params': params,
    }
    json = yield cmd_dict


def uninstall(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Uninstalls the given manifest_id and closes any opened app windows.

    :param manifest_id:
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.uninstall',
        'params': params,
    }
    json = yield cmd_dict


def launch(
        manifest_id: str,
        url: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,target.TargetID]:
    '''
    Launches the installed web app, or an url in the same web app instead of the
    default start url if it is provided. Returns a page Target.TargetID which
    can be used to attach to via Target.attachToTarget or similar APIs.

    :param manifest_id:
    :param url: *(Optional)*
    :returns: ID of the tab target created as a result.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if url is not None:
        params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.launch',
        'params': params,
    }
    json = yield cmd_dict
    return target.TargetID.from_json(json['targetId'])


def launch_files_in_app(
        manifest_id: str,
        files: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[target.TargetID]]:
    '''
    Opens one or more local files from an installed web app identified by its
    manifestId. The web app needs to have file handlers registered to process
    the files. The API returns one or more page Target.TargetIDs which can be
    used to attach to via Target.attachToTarget or similar APIs.
    If some files in the parameters cannot be handled by the web app, they will
    be ignored. If none of the files can be handled, this API returns an error.
    If no files are provided as the parameter, this API also returns an error.

    According to the definition of the file handlers in the manifest file, one
    Target.TargetID may represent a page handling one or more files. The order
    of the returned Target.TargetIDs is not guaranteed.

    TODO(crbug.com/339454034): Check the existences of the input files.

    :param manifest_id:
    :param files:
    :returns: IDs of the tab targets created as the result.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    params['files'] = [i for i in files]
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.launchFilesInApp',
        'params': params,
    }
    json = yield cmd_dict
    return [target.TargetID.from_json(i) for i in json['targetIds']]


def open_current_page_in_app(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Opens the current page in its web app identified by the manifest id, needs
    to be called on a page target. This function returns immediately without
    waiting for the app to finish loading.

    :param manifest_id:
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.openCurrentPageInApp',
        'params': params,
    }
    json = yield cmd_dict


def change_app_user_settings(
        manifest_id: str,
        link_capturing: typing.Optional[bool] = None,
        display_mode: typing.Optional[DisplayMode] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes user settings of the web app identified by its manifestId. If the
    app was not installed, this command returns an error. Unset parameters will
    be ignored; unrecognized values will cause an error.

    Unlike the ones defined in the manifest files of the web apps, these
    settings are provided by the browser and controlled by the users, they
    impact the way the browser handling the web apps.

    See the comment of each parameter.

    :param manifest_id:
    :param link_capturing: *(Optional)* If user allows the links clicked on by the user in the app's scope, or extended scope if the manifest has scope extensions and the flags ```DesktopPWAsLinkCapturingWithScopeExtensions```` and ````WebAppEnableScopeExtensions``` are enabled.  Note, the API does not support resetting the linkCapturing to the initial value, uninstalling and installing the web app again will reset it.  TODO(crbug.com/339453269): Setting this value on ChromeOS is not supported yet.
    :param display_mode: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if link_capturing is not None:
        params['linkCapturing'] = link_capturing
    if display_mode is not None:
        params['displayMode'] = display_mode.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.changeAppUserSettings',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\pwa.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: PWA (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target


@dataclass
class FileHandlerAccept:
    '''
    The following types are the replica of
    https://crsrc.org/c/chrome/browser/web_applications/proto/web_app_os_integration_state.proto;drc=9910d3be894c8f142c977ba1023f30a656bc13fc;l=67
    '''
    #: New name of the mimetype according to
    #: https://www.iana.org/assignments/media-types/media-types.xhtml
    media_type: str

    file_extensions: typing.List[str]

    def to_json(self):
        json = dict()
        json['mediaType'] = self.media_type
        json['fileExtensions'] = [i for i in self.file_extensions]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            media_type=str(json['mediaType']),
            file_extensions=[str(i) for i in json['fileExtensions']],
        )


@dataclass
class FileHandler:
    action: str

    accepts: typing.List[FileHandlerAccept]

    display_name: str

    def to_json(self):
        json = dict()
        json['action'] = self.action
        json['accepts'] = [i.to_json() for i in self.accepts]
        json['displayName'] = self.display_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            action=str(json['action']),
            accepts=[FileHandlerAccept.from_json(i) for i in json['accepts']],
            display_name=str(json['displayName']),
        )


class DisplayMode(enum.Enum):
    '''
    If user prefers opening the app in browser or an app window.
    '''
    STANDALONE = "standalone"
    BROWSER = "browser"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def get_os_app_state(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[int, typing.List[FileHandler]]]:
    '''
    Returns the following OS state for the given manifest id.

    :param manifest_id: The id from the webapp's manifest file, commonly it's the url of the site installing the webapp. See https://web.dev/learn/pwa/web-app-manifest.
    :returns: A tuple with the following items:

        0. **badgeCount** - 
        1. **fileHandlers** - 
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.getOsAppState',
        'params': params,
    }
    json = yield cmd_dict
    return (
        int(json['badgeCount']),
        [FileHandler.from_json(i) for i in json['fileHandlers']]
    )


def install(
        manifest_id: str,
        install_url_or_bundle_url: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Installs the given manifest identity, optionally using the given install_url
    or IWA bundle location.

    TODO(crbug.com/337872319) Support IWA to meet the following specific
    requirement.
    IWA-specific install description: If the manifest_id is isolated-app://,
    install_url_or_bundle_url is required, and can be either an http(s) URL or
    file:// URL pointing to a signed web bundle (.swbn). The .swbn file's
    signing key must correspond to manifest_id. If Chrome is not in IWA dev
    mode, the installation will fail, regardless of the state of the allowlist.

    :param manifest_id:
    :param install_url_or_bundle_url: *(Optional)* The location of the app or bundle overriding the one derived from the manifestId.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if install_url_or_bundle_url is not None:
        params['installUrlOrBundleUrl'] = install_url_or_bundle_url
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.install',
        'params': params,
    }
    json = yield cmd_dict


def uninstall(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Uninstalls the given manifest_id and closes any opened app windows.

    :param manifest_id:
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.uninstall',
        'params': params,
    }
    json = yield cmd_dict


def launch(
        manifest_id: str,
        url: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,target.TargetID]:
    '''
    Launches the installed web app, or an url in the same web app instead of the
    default start url if it is provided. Returns a page Target.TargetID which
    can be used to attach to via Target.attachToTarget or similar APIs.

    :param manifest_id:
    :param url: *(Optional)*
    :returns: ID of the tab target created as a result.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if url is not None:
        params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.launch',
        'params': params,
    }
    json = yield cmd_dict
    return target.TargetID.from_json(json['targetId'])


def launch_files_in_app(
        manifest_id: str,
        files: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[target.TargetID]]:
    '''
    Opens one or more local files from an installed web app identified by its
    manifestId. The web app needs to have file handlers registered to process
    the files. The API returns one or more page Target.TargetIDs which can be
    used to attach to via Target.attachToTarget or similar APIs.
    If some files in the parameters cannot be handled by the web app, they will
    be ignored. If none of the files can be handled, this API returns an error.
    If no files are provided as the parameter, this API also returns an error.

    According to the definition of the file handlers in the manifest file, one
    Target.TargetID may represent a page handling one or more files. The order
    of the returned Target.TargetIDs is not guaranteed.

    TODO(crbug.com/339454034): Check the existences of the input files.

    :param manifest_id:
    :param files:
    :returns: IDs of the tab targets created as the result.
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    params['files'] = [i for i in files]
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.launchFilesInApp',
        'params': params,
    }
    json = yield cmd_dict
    return [target.TargetID.from_json(i) for i in json['targetIds']]


def open_current_page_in_app(
        manifest_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Opens the current page in its web app identified by the manifest id, needs
    to be called on a page target. This function returns immediately without
    waiting for the app to finish loading.

    :param manifest_id:
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.openCurrentPageInApp',
        'params': params,
    }
    json = yield cmd_dict


def change_app_user_settings(
        manifest_id: str,
        link_capturing: typing.Optional[bool] = None,
        display_mode: typing.Optional[DisplayMode] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Changes user settings of the web app identified by its manifestId. If the
    app was not installed, this command returns an error. Unset parameters will
    be ignored; unrecognized values will cause an error.

    Unlike the ones defined in the manifest files of the web apps, these
    settings are provided by the browser and controlled by the users, they
    impact the way the browser handling the web apps.

    See the comment of each parameter.

    :param manifest_id:
    :param link_capturing: *(Optional)* If user allows the links clicked on by the user in the app's scope, or extended scope if the manifest has scope extensions and the flags ```DesktopPWAsLinkCapturingWithScopeExtensions```` and ````WebAppEnableScopeExtensions``` are enabled.  Note, the API does not support resetting the linkCapturing to the initial value, uninstalling and installing the web app again will reset it.  TODO(crbug.com/339453269): Setting this value on ChromeOS is not supported yet.
    :param display_mode: *(Optional)*
    '''
    params: T_JSON_DICT = dict()
    params['manifestId'] = manifest_id
    if link_capturing is not None:
        params['linkCapturing'] = link_capturing
    if display_mode is not None:
        params['displayMode'] = display_mode.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'PWA.changeAppUserSettings',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\mod.py ===
from .add import Add
from .exprtools import gcd_terms
from .function import DefinedFunction
from .kind import NumberKind
from .logic import fuzzy_and, fuzzy_not
from .mul import Mul
from .numbers import equal_valued
from .relational import is_le, is_lt, is_ge, is_gt
from .singleton import S


class Mod(DefinedFunction):
    """Represents a modulo operation on symbolic expressions.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    The convention used is the same as Python's: the remainder always has the
    same sign as the divisor.

    Many objects can be evaluated modulo ``n`` much faster than they can be
    evaluated directly (or at all).  For this, ``evaluate=False`` is
    necessary to prevent eager evaluation:

    >>> from sympy import binomial, factorial, Mod, Pow
    >>> Mod(Pow(2, 10**16, evaluate=False), 97)
    61
    >>> Mod(factorial(10**9, evaluate=False), 10**9 + 9)
    712524808
    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> x**2 % y
    Mod(x**2, y)
    >>> _.subs({x: 5, y: 6})
    1

    """

    kind = NumberKind

    @classmethod
    def eval(cls, p, q):
        def number_eval(p, q):
            """Try to return p % q if both are numbers or +/-p is known
            to be less than or equal q.
            """

            if q.is_zero:
                raise ZeroDivisionError("Modulo by zero")
            if p is S.NaN or q is S.NaN or p.is_finite is False or q.is_finite is False:
                return S.NaN
            if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
                return S.Zero

            if q.is_Number:
                if p.is_Number:
                    return p%q
                if q == 2:
                    if p.is_even:
                        return S.Zero
                    elif p.is_odd:
                        return S.One

            if hasattr(p, '_eval_Mod'):
                rv = getattr(p, '_eval_Mod')(q)
                if rv is not None:
                    return rv

            # by ratio
            r = p/q
            if r.is_integer:
                return S.Zero
            try:
                d = int(r)
            except TypeError:
                pass
            else:
                if isinstance(d, int):
                    rv = p - d*q
                    if (rv*q < 0) == True:
                        rv += q
                    return rv

            # by difference
            # -2|q| < p < 2|q|
            if q.is_positive:
                comp1, comp2 = is_le, is_lt
            elif q.is_negative:
                comp1, comp2 = is_ge, is_gt
            else:
                return
            ls = -2*q
            r = p - q
            for _ in range(4):
                if not comp1(ls, p):
                    return
                if comp2(r, ls):
                    return p - ls
                ls += q

        rv = number_eval(p, q)
        if rv is not None:
            return rv

        # denest
        if isinstance(p, cls):
            qinner = p.args[1]
            if qinner % q == 0:
                return cls(p.args[0], q)
            elif (qinner*(q - qinner)).is_nonnegative:
                # |qinner| < |q| and have same sign
                return p
        elif isinstance(-p, cls):
            qinner = (-p).args[1]
            if qinner % q == 0:
                return cls(-(-p).args[0], q)
            elif (qinner*(q + qinner)).is_nonpositive:
                # |qinner| < |q| and have different sign
                return p
        elif isinstance(p, Add):
            # separating into modulus and non modulus
            both_l = non_mod_l, mod_l = [], []
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)
            # if q same for all
            if mod_l and all(inner.args[1] == q for inner in mod_l):
                net = Add(*non_mod_l) + Add(*[i.args[0] for i in mod_l])
                return cls(net, q)

        elif isinstance(p, Mul):
            # separating into modulus and non modulus
            both_l = non_mod_l, mod_l = [], []
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)

            if mod_l and all(inner.args[1] == q for inner in mod_l) and all(t.is_integer for t in p.args) and q.is_integer:
                # finding distributive term
                non_mod_l = [cls(x, q) for x in non_mod_l]
                mod = []
                non_mod = []
                for j in non_mod_l:
                    if isinstance(j, cls):
                        mod.append(j.args[0])
                    else:
                        non_mod.append(j)
                prod_mod = Mul(*mod)
                prod_non_mod = Mul(*non_mod)
                prod_mod1 = Mul(*[i.args[0] for i in mod_l])
                net = prod_mod1*prod_mod
                return prod_non_mod*cls(net, q)

            if q.is_Integer and q is not S.One:
                if all(t.is_integer for t in p.args):
                    non_mod_l = [i % q if i.is_Integer else i for i in p.args]
                    if any(iq is S.Zero for iq in non_mod_l):
                        return S.Zero

            p = Mul(*(non_mod_l + mod_l))

        # XXX other possibilities?

        from sympy.polys.polyerrors import PolynomialError
        from sympy.polys.polytools import gcd

        # extract gcd; any further simplification should be done by the user
        try:
            G = gcd(p, q)
            if not equal_valued(G, 1):
                p, q = [gcd_terms(i/G, clear=False, fraction=False)
                        for i in (p, q)]
        except PolynomialError:  # issue 21373
            G = S.One
        pwas, qwas = p, q

        # simplify terms
        # (x + y + 2) % x -> Mod(y + 2, x)
        if p.is_Add:
            args = []
            for i in p.args:
                a = cls(i, q)
                if a.count(cls) > i.count(cls):
                    args.append(i)
                else:
                    args.append(a)
            if args != list(p.args):
                p = Add(*args)

        else:
            # handle coefficients if they are not Rational
            # since those are not handled by factor_terms
            # e.g. Mod(.6*x, .3*y) -> 0.3*Mod(2*x, y)
            cp, p = p.as_coeff_Mul()
            cq, q = q.as_coeff_Mul()
            ok = False
            if not cp.is_Rational or not cq.is_Rational:
                r = cp % cq
                if equal_valued(r, 0):
                    G *= cq
                    p *= int(cp/cq)
                    ok = True
            if not ok:
                p = cp*p
                q = cq*q

        # simple -1 extraction
        if p.could_extract_minus_sign() and q.could_extract_minus_sign():
            G, p, q = [-i for i in (G, p, q)]

        # check again to see if p and q can now be handled as numbers
        rv = number_eval(p, q)
        if rv is not None:
            return rv*G

        # put 1.0 from G on inside
        if G.is_Float and equal_valued(G, 1):
            p *= G
            return cls(p, q, evaluate=False)
        elif G.is_Mul and G.args[0].is_Float and equal_valued(G.args[0], 1):
            p = G.args[0]*p
            G = Mul._from_args(G.args[1:])
        return G*cls(p, q, evaluate=(p, q) != (pwas, qwas))

    def _eval_is_integer(self):
        p, q = self.args
        if fuzzy_and([p.is_integer, q.is_integer, fuzzy_not(q.is_zero)]):
            return True

    def _eval_is_nonnegative(self):
        if self.args[1].is_positive:
            return True

    def _eval_is_nonpositive(self):
        if self.args[1].is_negative:
            return True

    def _eval_rewrite_as_floor(self, a, b, **kwargs):
        from sympy.functions.elementary.integers import floor
        return a - b*floor(a/b)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.functions.elementary.integers import floor
        return self.rewrite(floor)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.functions.elementary.integers import floor
        return self.rewrite(floor)._eval_nseries(x, n, logx=logx, cdir=cdir)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\conftest.py ===
"""
Pytest configuration and fixtures for the Numpy test suite.
"""
import os
import string
import sys
import tempfile
import warnings
from contextlib import contextmanager

import hypothesis
import pytest

import numpy
import numpy as np
from numpy._core._multiarray_tests import get_fpu_mode
from numpy._core.tests._natype import get_stringdtype_dtype, pd_NA
from numpy.testing._private.utils import NOGIL_BUILD

try:
    from scipy_doctest.conftest import dt_config
    HAVE_SCPDT = True
except ModuleNotFoundError:
    HAVE_SCPDT = False


_old_fpu_mode = None
_collect_results = {}

# Use a known and persistent tmpdir for hypothesis' caches, which
# can be automatically cleared by the OS or user.
hypothesis.configuration.set_hypothesis_home_dir(
    os.path.join(tempfile.gettempdir(), ".hypothesis")
)

# We register two custom profiles for Numpy - for details see
# https://hypothesis.readthedocs.io/en/latest/settings.html
# The first is designed for our own CI runs; the latter also
# forces determinism and is designed for use via np.test()
hypothesis.settings.register_profile(
    name="numpy-profile", deadline=None, print_blob=True,
)
hypothesis.settings.register_profile(
    name="np.test() profile",
    deadline=None, print_blob=True, database=None, derandomize=True,
    suppress_health_check=list(hypothesis.HealthCheck),
)
# Note that the default profile is chosen based on the presence
# of pytest.ini, but can be overridden by passing the
# --hypothesis-profile=NAME argument to pytest.
_pytest_ini = os.path.join(os.path.dirname(__file__), "..", "pytest.ini")
hypothesis.settings.load_profile(
    "numpy-profile" if os.path.isfile(_pytest_ini) else "np.test() profile"
)

# The experimentalAPI is used in _umath_tests
os.environ["NUMPY_EXPERIMENTAL_DTYPE_API"] = "1"

def pytest_configure(config):
    config.addinivalue_line("markers",
        "valgrind_error: Tests that are known to error under valgrind.")
    config.addinivalue_line("markers",
        "leaks_references: Tests that are known to leak references.")
    config.addinivalue_line("markers",
        "slow: Tests that are very slow.")
    config.addinivalue_line("markers",
        "slow_pypy: Tests that are very slow on pypy.")


def pytest_addoption(parser):
    parser.addoption("--available-memory", action="store", default=None,
                     help=("Set amount of memory available for running the "
                           "test suite. This can result to tests requiring "
                           "especially large amounts of memory to be skipped. "
                           "Equivalent to setting environment variable "
                           "NPY_AVAILABLE_MEM. Default: determined"
                           "automatically."))


gil_enabled_at_start = True
if NOGIL_BUILD:
    gil_enabled_at_start = sys._is_gil_enabled()


def pytest_sessionstart(session):
    available_mem = session.config.getoption('available_memory')
    if available_mem is not None:
        os.environ['NPY_AVAILABLE_MEM'] = available_mem


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if NOGIL_BUILD and not gil_enabled_at_start and sys._is_gil_enabled():
        tr = terminalreporter
        tr.ensure_newline()
        tr.section("GIL re-enabled", sep="=", red=True, bold=True)
        tr.line("The GIL was re-enabled at runtime during the tests.")
        tr.line("This can happen with no test failures if the RuntimeWarning")
        tr.line("raised by Python when this happens is filtered by a test.")
        tr.line("")
        tr.line("Please ensure all new C modules declare support for running")
        tr.line("without the GIL. Any new tests that intentionally imports ")
        tr.line("code that re-enables the GIL should do so in a subprocess.")
        pytest.exit("GIL re-enabled during tests", returncode=1)

# FIXME when yield tests are gone.
@pytest.hookimpl()
def pytest_itemcollected(item):
    """
    Check FPU precision mode was not changed during test collection.

    The clumsy way we do it here is mainly necessary because numpy
    still uses yield tests, which can execute code at test collection
    time.
    """
    global _old_fpu_mode

    mode = get_fpu_mode()

    if _old_fpu_mode is None:
        _old_fpu_mode = mode
    elif mode != _old_fpu_mode:
        _collect_results[item] = (_old_fpu_mode, mode)
        _old_fpu_mode = mode


@pytest.fixture(scope="function", autouse=True)
def check_fpu_mode(request):
    """
    Check FPU precision mode was not changed during the test.
    """
    old_mode = get_fpu_mode()
    yield
    new_mode = get_fpu_mode()

    if old_mode != new_mode:
        raise AssertionError(f"FPU precision mode changed from {old_mode:#x} to "
                             f"{new_mode:#x} during the test")

    collect_result = _collect_results.get(request.node)
    if collect_result is not None:
        old_mode, new_mode = collect_result
        raise AssertionError(f"FPU precision mode changed from {old_mode:#x} to "
                             f"{new_mode:#x} when collecting the test")


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy

@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '0')


if HAVE_SCPDT:

    @contextmanager
    def warnings_errors_and_rng(test=None):
        """Filter out the wall of DeprecationWarnings.
        """
        msgs = ["The numpy.linalg.linalg",
                "The numpy.fft.helper",
                "dep_util",
                "pkg_resources",
                "numpy.core.umath",
                "msvccompiler",
                "Deprecated call",
                "numpy.core",
                "Importing from numpy.matlib",
                "This function is deprecated.",    # random_integers
                "Data type alias 'a'",     # numpy.rec.fromfile
                "Arrays of 2-dimensional vectors",   # matlib.cross
                "`in1d` is deprecated", ]
        msg = "|".join(msgs)

        msgs_r = [
            "invalid value encountered",
            "divide by zero encountered"
        ]
        msg_r = "|".join(msgs_r)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=DeprecationWarning, message=msg
            )
            warnings.filterwarnings(
                'ignore', category=RuntimeWarning, message=msg_r
            )
            yield

    # find and check doctests under this context manager
    dt_config.user_context_mgr = warnings_errors_and_rng

    # numpy specific tweaks from refguide-check
    dt_config.rndm_markers.add('#uninitialized')
    dt_config.rndm_markers.add('# uninitialized')

    # make the checker pick on mismatched dtypes
    dt_config.strict_check = True

    import doctest
    dt_config.optionflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

    # recognize the StringDType repr
    dt_config.check_namespace['StringDType'] = numpy.dtypes.StringDType

    # temporary skips
    dt_config.skiplist = {
        'numpy.savez',    # unclosed file
        'numpy.matlib.savez',
        'numpy.__array_namespace_info__',
        'numpy.matlib.__array_namespace_info__',
    }

    # xfail problematic tutorials
    dt_config.pytest_extra_xfail = {
        'how-to-verify-bug.rst': '',
        'c-info.ufunc-tutorial.rst': '',
        'basics.interoperability.rst': 'needs pandas',
        'basics.dispatch.rst': 'errors out in /testing/overrides.py',
        'basics.subclassing.rst': '.. testcode:: admonitions not understood',
        'misc.rst': 'manipulates warnings',
    }

    # ignores are for things fail doctest collection (optionals etc)
    dt_config.pytest_extra_ignore = [
        'numpy/distutils',
        'numpy/_core/cversions.py',
        'numpy/_pyinstaller',
        'numpy/random/_examples',
        'numpy/f2py/_backends/_distutils.py',
    ]


@pytest.fixture
def random_string_list():
    chars = list(string.ascii_letters + string.digits)
    chars = np.array(chars, dtype="U1")
    ret = np.random.choice(chars, size=100 * 10, replace=True)
    return ret.view("U100")


@pytest.fixture(params=[True, False])
def coerce(request):
    return request.param


@pytest.fixture(
    params=["unset", None, pd_NA, np.nan, float("nan"), "__nan__"],
    ids=["unset", "None", "pandas.NA", "np.nan", "float('nan')", "string nan"],
)
def na_object(request):
    return request.param


@pytest.fixture()
def dtype(na_object, coerce):
    return get_stringdtype_dtype(na_object, coerce)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_gelu.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionGelu(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "Gelu", "Erf")

    def fuse(self, erf_node, input_name_to_nodes: dict, output_name_to_node: dict):
        if self.fuse_1(erf_node, input_name_to_nodes, output_name_to_node):
            return
        if self.fuse_2(erf_node, input_name_to_nodes, output_name_to_node):
            return
        self.fuse_3(erf_node, input_name_to_nodes, output_name_to_node)

    def fuse_1(self, erf_node, input_name_to_nodes: dict, output_name_to_node: dict) -> bool | None:
        """
        This pattern is from PyTorch model
        Fuse Gelu with Erf into one node:
        Pattern 1:
                       +-------Mul(0.5)---------------------+
                       |                                    |
                       |                                    v
                    [root] --> Div -----> Erf  --> Add --> Mul -->
                              (B=1.4142...)       (1)

        Pattern 2:
                       +------------------------------------+
                       |                                    |
                       |                                    v
                    [root] --> Div -----> Erf  --> Add --> Mul -->Mul -->
                              (B=1.4142...)       (1)            (0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if erf_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_erf = children[0]

        if not self.model.has_constant_input(add_after_erf, 1):
            return

        if add_after_erf.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_after_erf = children[0]

        div = self.model.match_parent(erf_node, "Div", 0, output_name_to_node)
        if div is None:
            return

        if self.model.find_constant_input(div, 1.4142, delta=0.001) != 1:
            return

        subgraph_input = div.input[0]

        another = 1 if mul_after_erf.input[0] == add_after_erf.output[0] else 0
        if subgraph_input == mul_after_erf.input[another]:  # pattern 2
            children = input_name_to_nodes[mul_after_erf.output[0]]
            if len(children) != 1 or children[0].op_type != "Mul":
                return
            mul_half = children[0]
            if not self.model.has_constant_input(mul_half, 0.5):
                return
            subgraph_output = mul_half.output[0]
        else:  # pattern 1
            mul_half = self.model.match_parent(mul_after_erf, "Mul", another, output_name_to_node)
            if mul_half is None:
                return

            if not self.model.has_constant_input(mul_half, 0.5):
                return

            if subgraph_input not in mul_half.input:
                return

            subgraph_output = mul_after_erf.output[0]

        subgraph_nodes = [div, erf_node, add_after_erf, mul_after_erf, mul_half]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes, [subgraph_output], input_name_to_nodes, output_name_to_node
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "Gelu", inputs=[subgraph_input], outputs=[subgraph_output], name=self.model.create_node_name("Gelu")
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.increase_counter("Gelu")
        return True

    def fuse_2(self, erf_node, input_name_to_nodes: dict, output_name_to_node: dict) -> bool | None:
        """
        This pattern is from Keras model
        Fuse Gelu with Erf into one node:
                       +------------------------------------------+
                       |                                          |
                       |                                          v
                    [root] --> Div -----> Erf  --> Add --> Mul -->Mul
                              (B=1.4142...)       (A=1)   (A=0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if erf_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_erf = children[0]

        if not self.model.has_constant_input(add_after_erf, 1):
            return

        if add_after_erf.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_after_erf = children[0]

        if not self.model.has_constant_input(mul_after_erf, 0.5):
            return

        if mul_after_erf.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[mul_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul = children[0]

        div = self.model.match_parent(erf_node, "Div", 0, output_name_to_node)
        if div is None:
            return

        sqrt_node = None
        if self.model.find_constant_input(div, 1.4142, delta=0.001) != 1:
            sqrt_node = self.model.match_parent(div, "Sqrt", 1, output_name_to_node)
            if sqrt_node is None:
                return
            if not self.model.has_constant_input(sqrt_node, 2.0):
                return

        root_node = self.model.get_parent(div, 0, output_name_to_node)
        if root_node is None:
            return

        if root_node.output[0] not in mul.input:
            return

        subgraph_nodes = [div, erf_node, add_after_erf, mul_after_erf, mul]
        if sqrt_node:
            subgraph_nodes.append(sqrt_node)

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes, [mul.output[0]], input_name_to_nodes, output_name_to_node
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "Gelu", inputs=[root_node.output[0]], outputs=[mul.output[0]], name=self.model.create_node_name("Gelu")
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.increase_counter("Gelu")
        return True

    def fuse_3(self, erf_node, input_name_to_nodes: dict, output_name_to_node: dict) -> bool | None:
        """
        This pattern is from TensorFlow model
        Fuse Gelu with Erf into one node:
                       +----------------------------------------------+
                       |                                              |
                       |                                              v
                    [root] --> Mul -----> Erf    -->   Add --> Mul -->Mul
                               (A=0.7071067690849304)  (B=1)  (B=0.5)

        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """

        if erf_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[erf_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_erf = children[0]

        if not self.model.has_constant_input(add_after_erf, 1):
            return

        if add_after_erf.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_erf.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_half = children[0]

        if not self.model.has_constant_input(mul_half, 0.5):
            return

        first_mul = self.model.match_parent(erf_node, "Mul", 0, output_name_to_node)
        if first_mul is None:
            return

        i = self.model.find_constant_input(first_mul, 0.7071067690849304, delta=0.001)
        if i < 0:
            return

        root_node = self.model.get_parent(first_mul, 0 if i == 1 else 1, output_name_to_node)
        if root_node is None:
            return

        if mul_half.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[mul_half.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        last_mul = children[0]

        if not (last_mul.input[0] == root_node.output[0] or last_mul.input[1] == root_node.output[0]):
            return

        subgraph_nodes = [first_mul, erf_node, add_after_erf, mul_half, last_mul]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [last_mul.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "Gelu", inputs=[root_node.output[0]], outputs=[last_mul.output[0]], name=self.model.create_node_name("Gelu")
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.increase_counter("Gelu")
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_unet.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

from fusion_attention_unet import FusionAttentionUnet
from fusion_bias_add import FusionBiasAdd
from fusion_biassplitgelu import FusionBiasSplitGelu
from fusion_group_norm import FusionGroupNorm
from fusion_nhwc_conv import FusionNhwcConv
from fusion_options import FusionOptions
from fusion_skip_group_norm import FusionSkipGroupNorm
from fusion_transpose import FusionInsertTranspose, FusionTranspose
from import_utils import is_installed
from onnx import ModelProto
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel

logger = logging.getLogger(__name__)


class UnetOnnxModel(BertOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize UNet ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)

        super().__init__(model, num_heads=num_heads, hidden_size=hidden_size)

    def preprocess(self):
        self.remove_useless_div()

    def postprocess(self):
        self.prune_graph()
        self.remove_unused_constant()

    def remove_useless_div(self):
        """Remove Div by 1"""
        div_nodes = [node for node in self.nodes() if node.op_type == "Div"]

        nodes_to_remove = []
        for div in div_nodes:
            if self.find_constant_input(div, 1.0) == 1:
                nodes_to_remove.append(div)

        for node in nodes_to_remove:
            self.replace_input_of_all_nodes(node.output[0], node.input[0])

        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)
            logger.info("Removed %d Div nodes", len(nodes_to_remove))

    def convert_conv_to_nhwc(self):
        # Transpose weights in offline might help since ORT does not apply constant-folding on Transpose nodes.
        conv_to_nhwc_conv = FusionNhwcConv(self, update_weight=True)
        conv_to_nhwc_conv.apply()

    def merge_adjacent_transpose(self):
        fusion_transpose = FusionTranspose(self)
        fusion_transpose.apply()

        remove_count = 0
        nodes = self.get_nodes_by_op_type("Transpose")
        for node in nodes:
            permutation = OnnxModel.get_node_attribute(node, "perm")
            assert isinstance(permutation, list)
            if permutation != list(range(len(permutation))):
                continue
            assert not (
                self.find_graph_output(node.output[0])
                or self.find_graph_input(node.input[0])
                or self.find_graph_output(node.input[0])
            )

            # Let all children nodes skip current Transpose node and link to its parent
            # Note that we cannot update parent node output since parent node might have more than one children.
            self.replace_input_of_all_nodes(node.output[0], node.input[0])

            self.remove_node(node)
            remove_count += 1

        total = len(fusion_transpose.nodes_to_remove) + remove_count
        if total:
            logger.info("Removed %d Transpose nodes", total)

    def fuse_multi_head_attention(self, options: FusionOptions | None = None):
        # Self Attention
        enable_packed_qkv = (options is None) or options.enable_packed_qkv
        self_attention_fusion = FusionAttentionUnet(
            self,
            self.hidden_size,
            self.num_heads,
            is_cross_attention=False,
            enable_packed_qkv=enable_packed_qkv,
            enable_packed_kv=False,
        )
        self_attention_fusion.apply()

        # Cross Attention
        enable_packed_kv = (options is None) or options.enable_packed_kv
        cross_attention_fusion = FusionAttentionUnet(
            self,
            self.hidden_size,
            self.num_heads,
            is_cross_attention=True,
            enable_packed_qkv=False,
            enable_packed_kv=enable_packed_kv,
        )
        cross_attention_fusion.apply()

    def fuse_bias_add(self):
        fusion = FusionBiasAdd(self)
        fusion.apply()

    def optimize(self, options: FusionOptions | None = None):
        if is_installed("tqdm"):
            import tqdm
            from tqdm.contrib.logging import logging_redirect_tqdm

            with logging_redirect_tqdm():
                steps = 18
                progress_bar = tqdm.tqdm(range(steps), initial=0, desc="fusion")
                self._optimize(options, progress_bar)
        else:
            logger.info("tqdm is not installed. Run optimization without progress bar")
            self._optimize(options, None)

    def _optimize(self, options: FusionOptions | None = None, progress_bar=None):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()
        if progress_bar:
            progress_bar.update(1)

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()
        if progress_bar:
            progress_bar.update(1)

        self.preprocess()
        if progress_bar:
            progress_bar.update(1)

        self.fuse_reshape()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_group_norm:
            channels_last = (options is None) or options.group_norm_channels_last
            group_norm_fusion = FusionGroupNorm(self, channels_last)
            group_norm_fusion.apply()

            insert_transpose_fusion = FusionInsertTranspose(self)
            insert_transpose_fusion.apply()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_bias_splitgelu:
            bias_split_gelu_fusion = FusionBiasSplitGelu(self)
            bias_split_gelu_fusion.apply()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_attention:
            # self.save_model_to_file("before_mha.onnx")
            self.fuse_multi_head_attention(options)
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()
        if progress_bar:
            progress_bar.update(1)

        self.fuse_shape()
        if progress_bar:
            progress_bar.update(1)

        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        self.utils.remove_useless_reshape_nodes()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_skip_group_norm:
            skip_group_norm_fusion = FusionSkipGroupNorm(self)
            skip_group_norm_fusion.apply()
        if progress_bar:
            progress_bar.update(1)

        if (options is None) or options.enable_bias_skip_layer_norm:
            # Fuse SkipLayerNormalization and Add Bias before it.
            self.fuse_add_bias_skip_layer_norm()
        if progress_bar:
            progress_bar.update(1)

        if options is not None and options.enable_gelu_approximation:
            self.gelu_approximation()
        if progress_bar:
            progress_bar.update(1)

        if options is None or options.enable_nhwc_conv:
            self.convert_conv_to_nhwc()
            self.merge_adjacent_transpose()
        if progress_bar:
            progress_bar.update(1)

        if options is not None and options.enable_bias_add:
            self.fuse_bias_add()
        if progress_bar:
            progress_bar.update(1)

        self.postprocess()
        if progress_bar:
            progress_bar.update(1)

        logger.info(f"opset version: {self.get_opset_version()}")

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "Attention",
            "MultiHeadAttention",
            "LayerNormalization",
            "SkipLayerNormalization",
            "BiasSplitGelu",
            "GroupNorm",
            "SkipGroupNorm",
            "NhwcConv",
            "BiasAdd",
        ]

        for op in ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)

        logger.info(f"Optimized operators:{op_count}")
        return op_count

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\_testing\contexts.py ===
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
)
import uuid

from pandas._config import using_copy_on_write

from pandas.compat import PYPY
from pandas.errors import ChainedAssignmentError

from pandas import set_option

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from collections.abc import Generator

    from pandas._typing import (
        BaseBuffer,
        CompressionOptions,
        FilePath,
    )


@contextmanager
def decompress_file(
    path: FilePath | BaseBuffer, compression: CompressionOptions
) -> Generator[IO[bytes], None, None]:
    """
    Open a compressed file and return a file object.

    Parameters
    ----------
    path : str
        The path where the file is read from.

    compression : {'gzip', 'bz2', 'zip', 'xz', 'zstd', None}
        Name of the decompression to use

    Returns
    -------
    file object
    """
    with get_handle(path, "rb", compression=compression, is_text=False) as handle:
        yield handle.handle


@contextmanager
def set_timezone(tz: str) -> Generator[None, None, None]:
    """
    Context manager for temporarily setting a timezone.

    Parameters
    ----------
    tz : str
        A string representing a valid timezone.

    Examples
    --------
    >>> from datetime import datetime
    >>> from dateutil.tz import tzlocal
    >>> tzlocal().tzname(datetime(2021, 1, 1))  # doctest: +SKIP
    'IST'

    >>> with set_timezone('US/Eastern'):
    ...     tzlocal().tzname(datetime(2021, 1, 1))
    ...
    'EST'
    """
    import time

    def setTZ(tz) -> None:
        if hasattr(time, "tzset"):
            if tz is None:
                try:
                    del os.environ["TZ"]
                except KeyError:
                    pass
            else:
                os.environ["TZ"] = tz
                time.tzset()

    orig_tz = os.environ.get("TZ")
    setTZ(tz)
    try:
        yield
    finally:
        setTZ(orig_tz)


@contextmanager
def ensure_clean(
    filename=None, return_filelike: bool = False, **kwargs: Any
) -> Generator[Any, None, None]:
    """
    Gets a temporary path and agrees to remove on close.

    This implementation does not use tempfile.mkstemp to avoid having a file handle.
    If the code using the returned path wants to delete the file itself, windows
    requires that no program has a file handle to it.

    Parameters
    ----------
    filename : str (optional)
        suffix of the created file.
    return_filelike : bool (default False)
        if True, returns a file-like which is *always* cleaned. Necessary for
        savefig and other functions which want to append extensions.
    **kwargs
        Additional keywords are passed to open().

    """
    folder = Path(tempfile.gettempdir())

    if filename is None:
        filename = ""
    filename = str(uuid.uuid4()) + filename
    path = folder / filename

    path.touch()

    handle_or_str: str | IO = str(path)
    encoding = kwargs.pop("encoding", None)
    if return_filelike:
        kwargs.setdefault("mode", "w+b")
        if encoding is None and "b" not in kwargs["mode"]:
            encoding = "utf-8"
        handle_or_str = open(path, encoding=encoding, **kwargs)

    try:
        yield handle_or_str
    finally:
        if not isinstance(handle_or_str, str):
            handle_or_str.close()
        if path.is_file():
            path.unlink()


@contextmanager
def with_csv_dialect(name: str, **kwargs) -> Generator[None, None, None]:
    """
    Context manager to temporarily register a CSV dialect for parsing CSV.

    Parameters
    ----------
    name : str
        The name of the dialect.
    kwargs : mapping
        The parameters for the dialect.

    Raises
    ------
    ValueError : the name of the dialect conflicts with a builtin one.

    See Also
    --------
    csv : Python's CSV library.
    """
    import csv

    _BUILTIN_DIALECTS = {"excel", "excel-tab", "unix"}

    if name in _BUILTIN_DIALECTS:
        raise ValueError("Cannot override builtin dialect.")

    csv.register_dialect(name, **kwargs)
    try:
        yield
    finally:
        csv.unregister_dialect(name)


@contextmanager
def use_numexpr(use, min_elements=None) -> Generator[None, None, None]:
    from pandas.core.computation import expressions as expr

    if min_elements is None:
        min_elements = expr._MIN_ELEMENTS

    olduse = expr.USE_NUMEXPR
    oldmin = expr._MIN_ELEMENTS
    set_option("compute.use_numexpr", use)
    expr._MIN_ELEMENTS = min_elements
    try:
        yield
    finally:
        expr._MIN_ELEMENTS = oldmin
        set_option("compute.use_numexpr", olduse)


def raises_chained_assignment_error(warn=True, extra_warnings=(), extra_match=()):
    from pandas._testing import assert_produces_warning

    if not warn:
        from contextlib import nullcontext

        return nullcontext()

    if PYPY and not extra_warnings:
        from contextlib import nullcontext

        return nullcontext()
    elif PYPY and extra_warnings:
        return assert_produces_warning(
            extra_warnings,
            match="|".join(extra_match),
        )
    else:
        if using_copy_on_write():
            warning = ChainedAssignmentError
            match = (
                "A value is trying to be set on a copy of a DataFrame or Series "
                "through chained assignment"
            )
        else:
            warning = FutureWarning  # type: ignore[assignment]
            # TODO update match
            match = "ChainedAssignmentError"
        if extra_warnings:
            warning = (warning, *extra_warnings)  # type: ignore[assignment]
        return assert_produces_warning(
            warning,
            match="|".join((match, *extra_match)),
        )


def assert_cow_warning(warn=True, match=None, **kwargs):
    """
    Assert that a warning is raised in the CoW warning mode.

    Parameters
    ----------
    warn : bool, default True
        By default, check that a warning is raised. Can be turned off by passing False.
    match : str
        The warning message to match against, if different from the default.
    kwargs
        Passed through to assert_produces_warning
    """
    from pandas._testing import assert_produces_warning

    if not warn:
        from contextlib import nullcontext

        return nullcontext()

    if not match:
        match = "Setting a value on a view"

    return assert_produces_warning(FutureWarning, match=match, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\support\select.py ===
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

from typing import List

from selenium.common.exceptions import NoSuchElementException, UnexpectedTagNameException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement


class Select:
    def __init__(self, webelement: WebElement) -> None:
        """Constructor. A check is made that the given element is, indeed, a
        SELECT tag. If it is not, then an UnexpectedTagNameException is thrown.

        :Args:
         - webelement - SELECT element to wrap

        Example:
            from selenium.webdriver.support.ui import Select \n
            Select(driver.find_element(By.TAG_NAME, "select")).select_by_index(2)
        """
        if webelement.tag_name.lower() != "select":
            raise UnexpectedTagNameException(f"Select only works on <select> elements, not on {webelement.tag_name}")
        self._el = webelement
        multi = self._el.get_dom_attribute("multiple")
        self.is_multiple = multi and multi != "false"

    @property
    def options(self) -> List[WebElement]:
        """Returns a list of all options belonging to this select tag."""
        return self._el.find_elements(By.TAG_NAME, "option")

    @property
    def all_selected_options(self) -> List[WebElement]:
        """Returns a list of all selected options belonging to this select
        tag."""
        return [opt for opt in self.options if opt.is_selected()]

    @property
    def first_selected_option(self) -> WebElement:
        """The first selected option in this select tag (or the currently
        selected option in a normal select)"""
        for opt in self.options:
            if opt.is_selected():
                return opt
        raise NoSuchElementException("No options are selected")

    def select_by_value(self, value: str) -> None:
        """Select all options that have a value matching the argument. That is,
        when given "foo" this would select an option like:

        <option value="foo">Bar</option>

        :Args:
         - value - The value to match against

        throws NoSuchElementException If there is no option with specified value in SELECT
        """
        css = f"option[value ={self._escape_string(value)}]"
        opts = self._el.find_elements(By.CSS_SELECTOR, css)
        matched = False
        for opt in opts:
            self._set_selected(opt)
            if not self.is_multiple:
                return
            matched = True
        if not matched:
            raise NoSuchElementException(f"Cannot locate option with value: {value}")

    def select_by_index(self, index: int) -> None:
        """Select the option at the given index. This is done by examining the
        "index" attribute of an element, and not merely by counting.

        :Args:
         - index - The option at this index will be selected

        throws NoSuchElementException If there is no option with specified index in SELECT
        """
        match = str(index)
        for opt in self.options:
            if opt.get_attribute("index") == match:
                self._set_selected(opt)
                return
        raise NoSuchElementException(f"Could not locate element with index {index}")

    def select_by_visible_text(self, text: str) -> None:
        """Select all options that display text matching the argument. That is,
        when given "Bar" this would select an option like:

         <option value="foo">Bar</option>

        :Args:
         - text - The visible text to match against

         throws NoSuchElementException If there is no option with specified text in SELECT
        """
        xpath = f".//option[normalize-space(.) = {self._escape_string(text)}]"
        opts = self._el.find_elements(By.XPATH, xpath)
        matched = False
        for opt in opts:
            if not self._has_css_property_and_visible(opt):
                raise NoSuchElementException(f"Invisible option with text: {text}")
            self._set_selected(opt)
            if not self.is_multiple:
                return
            matched = True

        if len(opts) == 0 and " " in text:
            sub_string_without_space = self._get_longest_token(text)
            if sub_string_without_space == "":
                candidates = self.options
            else:
                xpath = f".//option[contains(.,{self._escape_string(sub_string_without_space)})]"
                candidates = self._el.find_elements(By.XPATH, xpath)
            for candidate in candidates:
                if text == candidate.text:
                    if not self._has_css_property_and_visible(candidate):
                        raise NoSuchElementException(f"Invisible option with text: {text}")
                    self._set_selected(candidate)
                    if not self.is_multiple:
                        return
                    matched = True

        if not matched:
            raise NoSuchElementException(f"Could not locate element with visible text: {text}")

    def deselect_all(self) -> None:
        """Clear all selected entries.

        This is only valid when the SELECT supports multiple selections.
        throws NotImplementedError If the SELECT does not support
        multiple selections
        """
        if not self.is_multiple:
            raise NotImplementedError("You may only deselect all options of a multi-select")
        for opt in self.options:
            self._unset_selected(opt)

    def deselect_by_value(self, value: str) -> None:
        """Deselect all options that have a value matching the argument. That
        is, when given "foo" this would deselect an option like:

         <option value="foo">Bar</option>

        :Args:
         - value - The value to match against

         throws NoSuchElementException If there is no option with specified value in SELECT
        """
        if not self.is_multiple:
            raise NotImplementedError("You may only deselect options of a multi-select")
        matched = False
        css = f"option[value = {self._escape_string(value)}]"
        opts = self._el.find_elements(By.CSS_SELECTOR, css)
        for opt in opts:
            self._unset_selected(opt)
            matched = True
        if not matched:
            raise NoSuchElementException(f"Could not locate element with value: {value}")

    def deselect_by_index(self, index: int) -> None:
        """Deselect the option at the given index. This is done by examining
        the "index" attribute of an element, and not merely by counting.

        :Args:
         - index - The option at this index will be deselected

         throws NoSuchElementException If there is no option with specified index in SELECT
        """
        if not self.is_multiple:
            raise NotImplementedError("You may only deselect options of a multi-select")
        for opt in self.options:
            if opt.get_attribute("index") == str(index):
                self._unset_selected(opt)
                return
        raise NoSuchElementException(f"Could not locate element with index {index}")

    def deselect_by_visible_text(self, text: str) -> None:
        """Deselect all options that display text matching the argument. That
        is, when given "Bar" this would deselect an option like:

        <option value="foo">Bar</option>

        :Args:
         - text - The visible text to match against
        """
        if not self.is_multiple:
            raise NotImplementedError("You may only deselect options of a multi-select")
        matched = False
        xpath = f".//option[normalize-space(.) = {self._escape_string(text)}]"
        opts = self._el.find_elements(By.XPATH, xpath)
        for opt in opts:
            if not self._has_css_property_and_visible(opt):
                raise NoSuchElementException(f"Invisible option with text: {text}")
            self._unset_selected(opt)
            matched = True
        if not matched:
            raise NoSuchElementException(f"Could not locate element with visible text: {text}")

    def _set_selected(self, option) -> None:
        if not option.is_selected():
            if not option.is_enabled():
                raise NotImplementedError("You may not select a disabled option")
            option.click()

    def _unset_selected(self, option) -> None:
        if option.is_selected():
            option.click()

    def _escape_string(self, value: str) -> str:
        if '"' in value and "'" in value:
            substrings = value.split('"')
            result = ["concat("]
            for substring in substrings:
                result.append(f'"{substring}"')
                result.append(", '\"', ")
            result = result[0:-1]
            if value.endswith('"'):
                result.append(", '\"'")
            return "".join(result) + ")"

        if '"' in value:
            return f"'{value}'"

        return f'"{value}"'

    def _get_longest_token(self, value: str) -> str:
        items = value.split(" ")
        longest = ""
        for item in items:
            if len(item) > len(longest):
                longest = item
        return longest

    def _has_css_property_and_visible(self, option) -> bool:
        css_value_candidates = ["hidden", "none", "0", "0.0"]
        css_property_candidates = ["visibility", "display", "opacity"]

        for property in css_property_candidates:
            css_value = option.value_of_css_property(property)
            if css_value in css_value_candidates:
                return False
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_rowcount.py ===
# testing/suite/test_rowcount.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from sqlalchemy import bindparam
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import testing
from sqlalchemy import text
from sqlalchemy.testing import eq_
from sqlalchemy.testing import fixtures


class RowCountTest(fixtures.TablesTest):
    """test rowcount functionality"""

    __requires__ = ("sane_rowcount",)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "employees",
            metadata,
            Column(
                "employee_id",
                Integer,
                autoincrement=False,
                primary_key=True,
            ),
            Column("name", String(50)),
            Column("department", String(1)),
        )

    @classmethod
    def insert_data(cls, connection):
        cls.data = data = [
            ("Angela", "A"),
            ("Andrew", "A"),
            ("Anand", "A"),
            ("Bob", "B"),
            ("Bobette", "B"),
            ("Buffy", "B"),
            ("Charlie", "C"),
            ("Cynthia", "C"),
            ("Chris", "C"),
        ]

        employees_table = cls.tables.employees
        connection.execute(
            employees_table.insert(),
            [
                {"employee_id": i, "name": n, "department": d}
                for i, (n, d) in enumerate(data)
            ],
        )

    def test_basic(self, connection):
        employees_table = self.tables.employees
        s = select(
            employees_table.c.name, employees_table.c.department
        ).order_by(employees_table.c.employee_id)
        rows = connection.execute(s).fetchall()

        eq_(rows, self.data)

    @testing.variation("statement", ["update", "delete", "insert", "select"])
    @testing.variation("close_first", [True, False])
    def test_non_rowcount_scenarios_no_raise(
        self, connection, statement, close_first
    ):
        employees_table = self.tables.employees

        # WHERE matches 3, 3 rows changed
        department = employees_table.c.department

        if statement.update:
            r = connection.execute(
                employees_table.update().where(department == "C"),
                {"department": "Z"},
            )
        elif statement.delete:
            r = connection.execute(
                employees_table.delete().where(department == "C"),
                {"department": "Z"},
            )
        elif statement.insert:
            r = connection.execute(
                employees_table.insert(),
                [
                    {"employee_id": 25, "name": "none 1", "department": "X"},
                    {"employee_id": 26, "name": "none 2", "department": "Z"},
                    {"employee_id": 27, "name": "none 3", "department": "Z"},
                ],
            )
        elif statement.select:
            s = select(
                employees_table.c.name, employees_table.c.department
            ).where(employees_table.c.department == "C")
            r = connection.execute(s)
            r.all()
        else:
            statement.fail()

        if close_first:
            r.close()

        assert r.rowcount in (-1, 3)

    def test_update_rowcount1(self, connection):
        employees_table = self.tables.employees

        # WHERE matches 3, 3 rows changed
        department = employees_table.c.department
        r = connection.execute(
            employees_table.update().where(department == "C"),
            {"department": "Z"},
        )
        assert r.rowcount == 3

    def test_update_rowcount2(self, connection):
        employees_table = self.tables.employees

        # WHERE matches 3, 0 rows changed
        department = employees_table.c.department

        r = connection.execute(
            employees_table.update().where(department == "C"),
            {"department": "C"},
        )
        eq_(r.rowcount, 3)

    @testing.variation("implicit_returning", [True, False])
    @testing.variation(
        "dml",
        [
            ("update", testing.requires.update_returning),
            ("delete", testing.requires.delete_returning),
        ],
    )
    def test_update_delete_rowcount_return_defaults(
        self, connection, implicit_returning, dml
    ):
        """note this test should succeed for all RETURNING backends
        as of 2.0.  In
        Idf28379f8705e403a3c6a937f6a798a042ef2540 we changed rowcount to use
        len(rows) when we have implicit returning

        """

        if implicit_returning:
            employees_table = self.tables.employees
        else:
            employees_table = Table(
                "employees",
                MetaData(),
                Column(
                    "employee_id",
                    Integer,
                    autoincrement=False,
                    primary_key=True,
                ),
                Column("name", String(50)),
                Column("department", String(1)),
                implicit_returning=False,
            )

        department = employees_table.c.department

        if dml.update:
            stmt = (
                employees_table.update()
                .where(department == "C")
                .values(name=employees_table.c.department + "Z")
                .return_defaults()
            )
        elif dml.delete:
            stmt = (
                employees_table.delete()
                .where(department == "C")
                .return_defaults()
            )
        else:
            dml.fail()

        r = connection.execute(stmt)
        eq_(r.rowcount, 3)

    def test_raw_sql_rowcount(self, connection):
        # test issue #3622, make sure eager rowcount is called for text
        result = connection.exec_driver_sql(
            "update employees set department='Z' where department='C'"
        )
        eq_(result.rowcount, 3)

    def test_text_rowcount(self, connection):
        # test issue #3622, make sure eager rowcount is called for text
        result = connection.execute(
            text("update employees set department='Z' where department='C'")
        )
        eq_(result.rowcount, 3)

    def test_delete_rowcount(self, connection):
        employees_table = self.tables.employees

        # WHERE matches 3, 3 rows deleted
        department = employees_table.c.department
        r = connection.execute(
            employees_table.delete().where(department == "C")
        )
        eq_(r.rowcount, 3)

    @testing.requires.sane_multi_rowcount
    def test_multi_update_rowcount(self, connection):
        employees_table = self.tables.employees
        stmt = (
            employees_table.update()
            .where(employees_table.c.name == bindparam("emp_name"))
            .values(department="C")
        )

        r = connection.execute(
            stmt,
            [
                {"emp_name": "Bob"},
                {"emp_name": "Cynthia"},
                {"emp_name": "nonexistent"},
            ],
        )

        eq_(r.rowcount, 2)

    @testing.requires.sane_multi_rowcount
    def test_multi_delete_rowcount(self, connection):
        employees_table = self.tables.employees

        stmt = employees_table.delete().where(
            employees_table.c.name == bindparam("emp_name")
        )

        r = connection.execute(
            stmt,
            [
                {"emp_name": "Bob"},
                {"emp_name": "Cynthia"},
                {"emp_name": "nonexistent"},
            ],
        )

        eq_(r.rowcount, 2)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\request.py ===
from __future__ import annotations

import io
import typing
from base64 import b64encode
from enum import Enum

from ..exceptions import UnrewindableBodyError
from .util import to_bytes

if typing.TYPE_CHECKING:
    from typing import Final

# Pass as a value within ``headers`` to skip
# emitting some HTTP headers that are added automatically.
# The only headers that are supported are ``Accept-Encoding``,
# ``Host``, and ``User-Agent``.
SKIP_HEADER = "@@@SKIP_HEADER@@@"
SKIPPABLE_HEADERS = frozenset(["accept-encoding", "host", "user-agent"])

ACCEPT_ENCODING = "gzip,deflate"
try:
    try:
        import brotlicffi as _unused_module_brotli  # type: ignore[import-not-found] # noqa: F401
    except ImportError:
        import brotli as _unused_module_brotli  # type: ignore[import-not-found] # noqa: F401
except ImportError:
    pass
else:
    ACCEPT_ENCODING += ",br"
try:
    import zstandard as _unused_module_zstd  # noqa: F401
except ImportError:
    pass
else:
    ACCEPT_ENCODING += ",zstd"


class _TYPE_FAILEDTELL(Enum):
    token = 0


_FAILEDTELL: Final[_TYPE_FAILEDTELL] = _TYPE_FAILEDTELL.token

_TYPE_BODY_POSITION = typing.Union[int, _TYPE_FAILEDTELL]

# When sending a request with these methods we aren't expecting
# a body so don't need to set an explicit 'Content-Length: 0'
# The reason we do this in the negative instead of tracking methods
# which 'should' have a body is because unknown methods should be
# treated as if they were 'POST' which *does* expect a body.
_METHODS_NOT_EXPECTING_BODY = {"GET", "HEAD", "DELETE", "TRACE", "OPTIONS", "CONNECT"}


def make_headers(
    keep_alive: bool | None = None,
    accept_encoding: bool | list[str] | str | None = None,
    user_agent: str | None = None,
    basic_auth: str | None = None,
    proxy_basic_auth: str | None = None,
    disable_cache: bool | None = None,
) -> dict[str, str]:
    """
    Shortcuts for generating request headers.

    :param keep_alive:
        If ``True``, adds 'connection: keep-alive' header.

    :param accept_encoding:
        Can be a boolean, list, or string.
        ``True`` translates to 'gzip,deflate'.  If the dependencies for
        Brotli (either the ``brotli`` or ``brotlicffi`` package) and/or Zstandard
        (the ``zstandard`` package) algorithms are installed, then their encodings are
        included in the string ('br' and 'zstd', respectively).
        List will get joined by comma.
        String will be used as provided.

    :param user_agent:
        String representing the user-agent you want, such as
        "python-urllib3/0.6"

    :param basic_auth:
        Colon-separated username:password string for 'authorization: basic ...'
        auth header.

    :param proxy_basic_auth:
        Colon-separated username:password string for 'proxy-authorization: basic ...'
        auth header.

    :param disable_cache:
        If ``True``, adds 'cache-control: no-cache' header.

    Example:

    .. code-block:: python

        import urllib3

        print(urllib3.util.make_headers(keep_alive=True, user_agent="Batman/1.0"))
        # {'connection': 'keep-alive', 'user-agent': 'Batman/1.0'}
        print(urllib3.util.make_headers(accept_encoding=True))
        # {'accept-encoding': 'gzip,deflate'}
    """
    headers: dict[str, str] = {}
    if accept_encoding:
        if isinstance(accept_encoding, str):
            pass
        elif isinstance(accept_encoding, list):
            accept_encoding = ",".join(accept_encoding)
        else:
            accept_encoding = ACCEPT_ENCODING
        headers["accept-encoding"] = accept_encoding

    if user_agent:
        headers["user-agent"] = user_agent

    if keep_alive:
        headers["connection"] = "keep-alive"

    if basic_auth:
        headers["authorization"] = (
            f"Basic {b64encode(basic_auth.encode('latin-1')).decode()}"
        )

    if proxy_basic_auth:
        headers["proxy-authorization"] = (
            f"Basic {b64encode(proxy_basic_auth.encode('latin-1')).decode()}"
        )

    if disable_cache:
        headers["cache-control"] = "no-cache"

    return headers


def set_file_position(
    body: typing.Any, pos: _TYPE_BODY_POSITION | None
) -> _TYPE_BODY_POSITION | None:
    """
    If a position is provided, move file to that point.
    Otherwise, we'll attempt to record a position for future use.
    """
    if pos is not None:
        rewind_body(body, pos)
    elif getattr(body, "tell", None) is not None:
        try:
            pos = body.tell()
        except OSError:
            # This differentiates from None, allowing us to catch
            # a failed `tell()` later when trying to rewind the body.
            pos = _FAILEDTELL

    return pos


def rewind_body(body: typing.IO[typing.AnyStr], body_pos: _TYPE_BODY_POSITION) -> None:
    """
    Attempt to rewind body to a certain position.
    Primarily used for request redirects and retries.

    :param body:
        File-like object that supports seek.

    :param int pos:
        Position to seek to in file.
    """
    body_seek = getattr(body, "seek", None)
    if body_seek is not None and isinstance(body_pos, int):
        try:
            body_seek(body_pos)
        except OSError as e:
            raise UnrewindableBodyError(
                "An error occurred when rewinding request body for redirect/retry."
            ) from e
    elif body_pos is _FAILEDTELL:
        raise UnrewindableBodyError(
            "Unable to record file position for rewinding "
            "request body during a redirect/retry."
        )
    else:
        raise ValueError(
            f"body_pos must be of type integer, instead it was {type(body_pos)}."
        )


class ChunksAndContentLength(typing.NamedTuple):
    chunks: typing.Iterable[bytes] | None
    content_length: int | None


def body_to_chunks(
    body: typing.Any | None, method: str, blocksize: int
) -> ChunksAndContentLength:
    """Takes the HTTP request method, body, and blocksize and
    transforms them into an iterable of chunks to pass to
    socket.sendall() and an optional 'Content-Length' header.

    A 'Content-Length' of 'None' indicates the length of the body
    can't be determined so should use 'Transfer-Encoding: chunked'
    for framing instead.
    """

    chunks: typing.Iterable[bytes] | None
    content_length: int | None

    # No body, we need to make a recommendation on 'Content-Length'
    # based on whether that request method is expected to have
    # a body or not.
    if body is None:
        chunks = None
        if method.upper() not in _METHODS_NOT_EXPECTING_BODY:
            content_length = 0
        else:
            content_length = None

    # Bytes or strings become bytes
    elif isinstance(body, (str, bytes)):
        chunks = (to_bytes(body),)
        content_length = len(chunks[0])

    # File-like object, TODO: use seek() and tell() for length?
    elif hasattr(body, "read"):

        def chunk_readable() -> typing.Iterable[bytes]:
            nonlocal body, blocksize
            encode = isinstance(body, io.TextIOBase)
            while True:
                datablock = body.read(blocksize)
                if not datablock:
                    break
                if encode:
                    datablock = datablock.encode("utf-8")
                yield datablock

        chunks = chunk_readable()
        content_length = None

    # Otherwise we need to start checking via duck-typing.
    else:
        try:
            # Check if the body implements the buffer API.
            mv = memoryview(body)
        except TypeError:
            try:
                # Check if the body is an iterable
                chunks = iter(body)
                content_length = None
            except TypeError:
                raise TypeError(
                    f"'body' must be a bytes-like object, file-like "
                    f"object, or iterable. Instead was {body!r}"
                ) from None
        else:
            # Since it implements the buffer API can be passed directly to socket.sendall()
            chunks = (body,)
            content_length = mv.nbytes

    return ChunksAndContentLength(chunks=chunks, content_length=content_length)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\wrappers.py ===
from __future__ import annotations

import typing as t

from werkzeug.exceptions import BadRequest
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Request as RequestBase
from werkzeug.wrappers import Response as ResponseBase

from . import json
from .globals import current_app
from .helpers import _split_blueprint_path

if t.TYPE_CHECKING:  # pragma: no cover
    from werkzeug.routing import Rule


class Request(RequestBase):
    """The request object used by default in Flask.  Remembers the
    matched endpoint and view arguments.

    It is what ends up as :class:`~flask.request`.  If you want to replace
    the request object used you can subclass this and set
    :attr:`~flask.Flask.request_class` to your subclass.

    The request object is a :class:`~werkzeug.wrappers.Request` subclass and
    provides all of the attributes Werkzeug defines plus a few Flask
    specific ones.
    """

    json_module: t.Any = json

    #: The internal URL rule that matched the request.  This can be
    #: useful to inspect which methods are allowed for the URL from
    #: a before/after handler (``request.url_rule.methods``) etc.
    #: Though if the request's method was invalid for the URL rule,
    #: the valid list is available in ``routing_exception.valid_methods``
    #: instead (an attribute of the Werkzeug exception
    #: :exc:`~werkzeug.exceptions.MethodNotAllowed`)
    #: because the request was never internally bound.
    #:
    #: .. versionadded:: 0.6
    url_rule: Rule | None = None

    #: A dict of view arguments that matched the request.  If an exception
    #: happened when matching, this will be ``None``.
    view_args: dict[str, t.Any] | None = None

    #: If matching the URL failed, this is the exception that will be
    #: raised / was raised as part of the request handling.  This is
    #: usually a :exc:`~werkzeug.exceptions.NotFound` exception or
    #: something similar.
    routing_exception: HTTPException | None = None

    _max_content_length: int | None = None
    _max_form_memory_size: int | None = None
    _max_form_parts: int | None = None

    @property
    def max_content_length(self) -> int | None:
        """The maximum number of bytes that will be read during this request. If
        this limit is exceeded, a 413 :exc:`~werkzeug.exceptions.RequestEntityTooLarge`
        error is raised. If it is set to ``None``, no limit is enforced at the
        Flask application level. However, if it is ``None`` and the request has
        no ``Content-Length`` header and the WSGI server does not indicate that
        it terminates the stream, then no data is read to avoid an infinite
        stream.

        Each request defaults to the :data:`MAX_CONTENT_LENGTH` config, which
        defaults to ``None``. It can be set on a specific ``request`` to apply
        the limit to that specific view. This should be set appropriately based
        on an application's or view's specific needs.

        .. versionchanged:: 3.1
            This can be set per-request.

        .. versionchanged:: 0.6
            This is configurable through Flask config.
        """
        if self._max_content_length is not None:
            return self._max_content_length

        if not current_app:
            return super().max_content_length

        return current_app.config["MAX_CONTENT_LENGTH"]  # type: ignore[no-any-return]

    @max_content_length.setter
    def max_content_length(self, value: int | None) -> None:
        self._max_content_length = value

    @property
    def max_form_memory_size(self) -> int | None:
        """The maximum size in bytes any non-file form field may be in a
        ``multipart/form-data`` body. If this limit is exceeded, a 413
        :exc:`~werkzeug.exceptions.RequestEntityTooLarge` error is raised. If it
        is set to ``None``, no limit is enforced at the Flask application level.

        Each request defaults to the :data:`MAX_FORM_MEMORY_SIZE` config, which
        defaults to ``500_000``. It can be set on a specific ``request`` to
        apply the limit to that specific view. This should be set appropriately
        based on an application's or view's specific needs.

        .. versionchanged:: 3.1
            This is configurable through Flask config.
        """
        if self._max_form_memory_size is not None:
            return self._max_form_memory_size

        if not current_app:
            return super().max_form_memory_size

        return current_app.config["MAX_FORM_MEMORY_SIZE"]  # type: ignore[no-any-return]

    @max_form_memory_size.setter
    def max_form_memory_size(self, value: int | None) -> None:
        self._max_form_memory_size = value

    @property  # type: ignore[override]
    def max_form_parts(self) -> int | None:
        """The maximum number of fields that may be present in a
        ``multipart/form-data`` body. If this limit is exceeded, a 413
        :exc:`~werkzeug.exceptions.RequestEntityTooLarge` error is raised. If it
        is set to ``None``, no limit is enforced at the Flask application level.

        Each request defaults to the :data:`MAX_FORM_PARTS` config, which
        defaults to ``1_000``. It can be set on a specific ``request`` to apply
        the limit to that specific view. This should be set appropriately based
        on an application's or view's specific needs.

        .. versionchanged:: 3.1
            This is configurable through Flask config.
        """
        if self._max_form_parts is not None:
            return self._max_form_parts

        if not current_app:
            return super().max_form_parts

        return current_app.config["MAX_FORM_PARTS"]  # type: ignore[no-any-return]

    @max_form_parts.setter
    def max_form_parts(self, value: int | None) -> None:
        self._max_form_parts = value

    @property
    def endpoint(self) -> str | None:
        """The endpoint that matched the request URL.

        This will be ``None`` if matching failed or has not been
        performed yet.

        This in combination with :attr:`view_args` can be used to
        reconstruct the same URL or a modified URL.
        """
        if self.url_rule is not None:
            return self.url_rule.endpoint  # type: ignore[no-any-return]

        return None

    @property
    def blueprint(self) -> str | None:
        """The registered name of the current blueprint.

        This will be ``None`` if the endpoint is not part of a
        blueprint, or if URL matching failed or has not been performed
        yet.

        This does not necessarily match the name the blueprint was
        created with. It may have been nested, or registered with a
        different name.
        """
        endpoint = self.endpoint

        if endpoint is not None and "." in endpoint:
            return endpoint.rpartition(".")[0]

        return None

    @property
    def blueprints(self) -> list[str]:
        """The registered names of the current blueprint upwards through
        parent blueprints.

        This will be an empty list if there is no current blueprint, or
        if URL matching failed.

        .. versionadded:: 2.0.1
        """
        name = self.blueprint

        if name is None:
            return []

        return _split_blueprint_path(name)

    def _load_form_data(self) -> None:
        super()._load_form_data()

        # In debug mode we're replacing the files multidict with an ad-hoc
        # subclass that raises a different error for key errors.
        if (
            current_app
            and current_app.debug
            and self.mimetype != "multipart/form-data"
            and not self.files
        ):
            from .debughelpers import attach_enctype_error_multidict

            attach_enctype_error_multidict(self)

    def on_json_loading_failed(self, e: ValueError | None) -> t.Any:
        try:
            return super().on_json_loading_failed(e)
        except BadRequest as ebr:
            if current_app and current_app.debug:
                raise

            raise BadRequest() from ebr


class Response(ResponseBase):
    """The response object that is used by default in Flask.  Works like the
    response object from Werkzeug but is set to have an HTML mimetype by
    default.  Quite often you don't have to create this object yourself because
    :meth:`~flask.Flask.make_response` will take care of that for you.

    If you want to replace the response object used you can subclass this and
    set :attr:`~flask.Flask.response_class` to your subclass.

    .. versionchanged:: 1.0
        JSON support is added to the response, like the request. This is useful
        when testing to get the test client response data as JSON.

    .. versionchanged:: 1.0

        Added :attr:`max_cookie_size`.
    """

    default_mimetype: str | None = "text/html"

    json_module = json

    autocorrect_location_header = False

    @property
    def max_cookie_size(self) -> int:  # type: ignore
        """Read-only view of the :data:`MAX_COOKIE_SIZE` config key.

        See :attr:`~werkzeug.wrappers.Response.max_cookie_size` in
        Werkzeug's docs.
        """
        if current_app:
            return current_app.config["MAX_COOKIE_SIZE"]  # type: ignore[no-any-return]

        # return Werkzeug's default when not in an app context
        return super().max_cookie_size

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\tree.py ===
from typing import Iterator, List, Optional, Tuple

from ._loop import loop_first, loop_last
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .jupyter import JupyterMixin
from .measure import Measurement
from .segment import Segment
from .style import Style, StyleStack, StyleType
from .styled import Styled

GuideType = Tuple[str, str, str, str]


class Tree(JupyterMixin):
    """A renderable for a tree structure.

    Attributes:
        ASCII_GUIDES (GuideType): Guide lines used when Console.ascii_only is True.
        TREE_GUIDES (List[GuideType, GuideType, GuideType]): Default guide lines.

    Args:
        label (RenderableType): The renderable or str for the tree label.
        style (StyleType, optional): Style of this tree. Defaults to "tree".
        guide_style (StyleType, optional): Style of the guide lines. Defaults to "tree.line".
        expanded (bool, optional): Also display children. Defaults to True.
        highlight (bool, optional): Highlight renderable (if str). Defaults to False.
        hide_root (bool, optional): Hide the root node. Defaults to False.
    """

    ASCII_GUIDES = ("    ", "|   ", "+-- ", "`-- ")
    TREE_GUIDES = [
        ("    ", "│   ", "├── ", "└── "),
        ("    ", "┃   ", "┣━━ ", "┗━━ "),
        ("    ", "║   ", "╠══ ", "╚══ "),
    ]

    def __init__(
        self,
        label: RenderableType,
        *,
        style: StyleType = "tree",
        guide_style: StyleType = "tree.line",
        expanded: bool = True,
        highlight: bool = False,
        hide_root: bool = False,
    ) -> None:
        self.label = label
        self.style = style
        self.guide_style = guide_style
        self.children: List[Tree] = []
        self.expanded = expanded
        self.highlight = highlight
        self.hide_root = hide_root

    def add(
        self,
        label: RenderableType,
        *,
        style: Optional[StyleType] = None,
        guide_style: Optional[StyleType] = None,
        expanded: bool = True,
        highlight: Optional[bool] = False,
    ) -> "Tree":
        """Add a child tree.

        Args:
            label (RenderableType): The renderable or str for the tree label.
            style (StyleType, optional): Style of this tree. Defaults to "tree".
            guide_style (StyleType, optional): Style of the guide lines. Defaults to "tree.line".
            expanded (bool, optional): Also display children. Defaults to True.
            highlight (Optional[bool], optional): Highlight renderable (if str). Defaults to False.

        Returns:
            Tree: A new child Tree, which may be further modified.
        """
        node = Tree(
            label,
            style=self.style if style is None else style,
            guide_style=self.guide_style if guide_style is None else guide_style,
            expanded=expanded,
            highlight=self.highlight if highlight is None else highlight,
        )
        self.children.append(node)
        return node

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "RenderResult":
        stack: List[Iterator[Tuple[bool, Tree]]] = []
        pop = stack.pop
        push = stack.append
        new_line = Segment.line()

        get_style = console.get_style
        null_style = Style.null()
        guide_style = get_style(self.guide_style, default="") or null_style
        SPACE, CONTINUE, FORK, END = range(4)

        _Segment = Segment

        def make_guide(index: int, style: Style) -> Segment:
            """Make a Segment for a level of the guide lines."""
            if options.ascii_only:
                line = self.ASCII_GUIDES[index]
            else:
                guide = 1 if style.bold else (2 if style.underline2 else 0)
                line = self.TREE_GUIDES[0 if options.legacy_windows else guide][index]
            return _Segment(line, style)

        levels: List[Segment] = [make_guide(CONTINUE, guide_style)]
        push(iter(loop_last([self])))

        guide_style_stack = StyleStack(get_style(self.guide_style))
        style_stack = StyleStack(get_style(self.style))
        remove_guide_styles = Style(bold=False, underline2=False)

        depth = 0

        while stack:
            stack_node = pop()
            try:
                last, node = next(stack_node)
            except StopIteration:
                levels.pop()
                if levels:
                    guide_style = levels[-1].style or null_style
                    levels[-1] = make_guide(FORK, guide_style)
                    guide_style_stack.pop()
                    style_stack.pop()
                continue
            push(stack_node)
            if last:
                levels[-1] = make_guide(END, levels[-1].style or null_style)

            guide_style = guide_style_stack.current + get_style(node.guide_style)
            style = style_stack.current + get_style(node.style)
            prefix = levels[(2 if self.hide_root else 1) :]
            renderable_lines = console.render_lines(
                Styled(node.label, style),
                options.update(
                    width=options.max_width
                    - sum(level.cell_length for level in prefix),
                    highlight=self.highlight,
                    height=None,
                ),
                pad=options.justify is not None,
            )

            if not (depth == 0 and self.hide_root):
                for first, line in loop_first(renderable_lines):
                    if prefix:
                        yield from _Segment.apply_style(
                            prefix,
                            style.background_style,
                            post_style=remove_guide_styles,
                        )
                    yield from line
                    yield new_line
                    if first and prefix:
                        prefix[-1] = make_guide(
                            SPACE if last else CONTINUE, prefix[-1].style or null_style
                        )

            if node.expanded and node.children:
                levels[-1] = make_guide(
                    SPACE if last else CONTINUE, levels[-1].style or null_style
                )
                levels.append(
                    make_guide(END if len(node.children) == 1 else FORK, guide_style)
                )
                style_stack.push(get_style(node.style))
                guide_style_stack.push(get_style(node.guide_style))
                push(iter(loop_last(node.children)))
                depth += 1

    def __rich_measure__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> "Measurement":
        stack: List[Iterator[Tree]] = [iter([self])]
        pop = stack.pop
        push = stack.append
        minimum = 0
        maximum = 0
        measure = Measurement.get
        level = 0
        while stack:
            iter_tree = pop()
            try:
                tree = next(iter_tree)
            except StopIteration:
                level -= 1
                continue
            push(iter_tree)
            min_measure, max_measure = measure(console, options, tree.label)
            indent = level * 4
            minimum = max(min_measure + indent, minimum)
            maximum = max(max_measure + indent, maximum)
            if tree.expanded and tree.children:
                push(iter(tree.children))
                level += 1
        return Measurement(minimum, maximum)


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich.console import Group
    from pip._vendor.rich.markdown import Markdown
    from pip._vendor.rich.panel import Panel
    from pip._vendor.rich.syntax import Syntax
    from pip._vendor.rich.table import Table

    table = Table(row_styles=["", "dim"])

    table.add_column("Released", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Box Office", justify="right", style="green")

    table.add_row("Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$952,110,690")
    table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
    table.add_row("Dec 15, 2017", "Star Wars Ep. V111: The Last Jedi", "$1,332,539,889")
    table.add_row("Dec 16, 2016", "Rogue One: A Star Wars Story", "$1,332,439,889")

    code = """\
class Segment(NamedTuple):
    text: str = ""
    style: Optional[Style] = None
    is_control: bool = False
"""
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)

    markdown = Markdown(
        """\
### example.md
> Hello, World!
>
> Markdown _all_ the things
"""
    )

    root = Tree("🌲 [b green]Rich Tree", highlight=True, hide_root=True)

    node = root.add(":file_folder: Renderables", guide_style="red")
    simple_node = node.add(":file_folder: [bold yellow]Atomic", guide_style="uu green")
    simple_node.add(Group("📄 Syntax", syntax))
    simple_node.add(Group("📄 Markdown", Panel(markdown, border_style="green")))

    containers_node = node.add(
        ":file_folder: [bold magenta]Containers", guide_style="bold magenta"
    )
    containers_node.expanded = True
    panel = Panel.fit("Just a panel", border_style="red")
    containers_node.add(Group("📄 Panels", panel))

    containers_node.add(Group("📄 [b magenta]Table", table))

    console = Console()

    console.print(root)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\series\limitseq.py ===
"""Limits of sequences"""

from sympy.calculus.accumulationbounds import AccumulationBounds
from sympy.core.add import Add
from sympy.core.function import PoleError
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.combinatorial.factorials import factorial, subfactorial
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series.limits import Limit


def difference_delta(expr, n=None, step=1):
    """Difference Operator.

    Explanation
    ===========

    Discrete analog of differential operator. Given a sequence x[n],
    returns the sequence x[n + step] - x[n].

    Examples
    ========

    >>> from sympy import difference_delta as dd
    >>> from sympy.abc import n
    >>> dd(n*(n + 1), n)
    2*n + 2
    >>> dd(n*(n + 1), n, 2)
    4*n + 6

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/DifferenceDelta.html
    """
    expr = sympify(expr)

    if n is None:
        f = expr.free_symbols
        if len(f) == 1:
            n = f.pop()
        elif len(f) == 0:
            return S.Zero
        else:
            raise ValueError("Since there is more than one variable in the"
                             " expression, a variable must be supplied to"
                             " take the difference of %s" % expr)
    step = sympify(step)
    if step.is_number is False or step.is_finite is False:
        raise ValueError("Step should be a finite number.")

    if hasattr(expr, '_eval_difference_delta'):
        result = expr._eval_difference_delta(n, step)
        if result:
            return result

    return expr.subs(n, n + step) - expr


def dominant(expr, n):
    """Finds the dominant term in a sum, that is a term that dominates
    every other term.

    Explanation
    ===========

    If limit(a/b, n, oo) is oo then a dominates b.
    If limit(a/b, n, oo) is 0 then b dominates a.
    Otherwise, a and b are comparable.

    If there is no unique dominant term, then returns ``None``.

    Examples
    ========

    >>> from sympy import Sum
    >>> from sympy.series.limitseq import dominant
    >>> from sympy.abc import n, k
    >>> dominant(5*n**3 + 4*n**2 + n + 1, n)
    5*n**3
    >>> dominant(2**n + Sum(k, (k, 0, n)), n)
    2**n

    See Also
    ========

    sympy.series.limitseq.dominant
    """
    terms = Add.make_args(expr.expand(func=True))
    term0 = terms[-1]
    comp = [term0]  # comparable terms
    for t in terms[:-1]:
        r = term0/t
        e = r.gammasimp()
        if e == r:
            e = r.factor()
        l = limit_seq(e, n)
        if l is None:
            return None
        elif l.is_zero:
            term0 = t
            comp = [term0]
        elif l not in [S.Infinity, S.NegativeInfinity]:
            comp.append(t)
    if len(comp) > 1:
        return None
    return term0


def _limit_inf(expr, n):
    try:
        return Limit(expr, n, S.Infinity).doit(deep=False)
    except (NotImplementedError, PoleError):
        return None


def _limit_seq(expr, n, trials):
    from sympy.concrete.summations import Sum

    for i in range(trials):
        if not expr.has(Sum):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()
        if not den.has(n) or not num.has(n):
            result = _limit_inf(expr.doit(), n)
            if result is not None:
                return result
            return None

        num, den = (difference_delta(t.expand(), n) for t in [num, den])
        expr = (num / den).gammasimp()

        if not expr.has(Sum):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()

        num = dominant(num, n)
        if num is None:
            return None

        den = dominant(den, n)
        if den is None:
            return None

        expr = (num / den).gammasimp()


def limit_seq(expr, n=None, trials=5):
    """Finds the limit of a sequence as index ``n`` tends to infinity.

    Parameters
    ==========

    expr : Expr
        SymPy expression for the ``n-th`` term of the sequence
    n : Symbol, optional
        The index of the sequence, an integer that tends to positive
        infinity. If None, inferred from the expression unless it has
        multiple symbols.
    trials: int, optional
        The algorithm is highly recursive. ``trials`` is a safeguard from
        infinite recursion in case the limit is not easily computed by the
        algorithm. Try increasing ``trials`` if the algorithm returns ``None``.

    Admissible Terms
    ================

    The algorithm is designed for sequences built from rational functions,
    indefinite sums, and indefinite products over an indeterminate n. Terms of
    alternating sign are also allowed, but more complex oscillatory behavior is
    not supported.

    Examples
    ========

    >>> from sympy import limit_seq, Sum, binomial
    >>> from sympy.abc import n, k, m
    >>> limit_seq((5*n**3 + 3*n**2 + 4) / (3*n**3 + 4*n - 5), n)
    5/3
    >>> limit_seq(binomial(2*n, n) / Sum(binomial(2*k, k), (k, 1, n)), n)
    3/4
    >>> limit_seq(Sum(k**2 * Sum(2**m/m, (m, 1, k)), (k, 1, n)) / (2**n*n), n)
    4

    See Also
    ========

    sympy.series.limitseq.dominant

    References
    ==========

    .. [1] Computing Limits of Sequences - Manuel Kauers
    """

    from sympy.concrete.summations import Sum
    if n is None:
        free = expr.free_symbols
        if len(free) == 1:
            n = free.pop()
        elif not free:
            return expr
        else:
            raise ValueError("Expression has more than one variable. "
                             "Please specify a variable.")
    elif n not in expr.free_symbols:
        return expr

    expr = expr.rewrite(fibonacci, S.GoldenRatio)
    expr = expr.rewrite(factorial, subfactorial, gamma)
    n_ = Dummy("n", integer=True, positive=True)
    n1 = Dummy("n", odd=True, positive=True)
    n2 = Dummy("n", even=True, positive=True)

    # If there is a negative term raised to a power involving n, or a
    # trigonometric function, then consider even and odd n separately.
    powers = (p.as_base_exp() for p in expr.atoms(Pow))
    if (any(b.is_negative and e.has(n) for b, e in powers) or
            expr.has(cos, sin)):
        L1 = _limit_seq(expr.xreplace({n: n1}), n1, trials)
        if L1 is not None:
            L2 = _limit_seq(expr.xreplace({n: n2}), n2, trials)
            if L1 != L2:
                if L1.is_comparable and L2.is_comparable:
                    return AccumulationBounds(Min(L1, L2), Max(L1, L2))
                else:
                    return None
    else:
        L1 = _limit_seq(expr.xreplace({n: n_}), n_, trials)
    if L1 is not None:
        return L1
    else:
        if expr.is_Add:
            limits = [limit_seq(term, n, trials) for term in expr.args]
            if any(result is None for result in limits):
                return None
            else:
                return Add(*limits)
        # Maybe the absolute value is easier to deal with (though not if
        # it has a Sum). If it tends to 0, the limit is 0.
        elif not expr.has(Sum):
            lim = _limit_seq(Abs(expr.xreplace({n: n_})), n_, trials)
            if lim is not None and lim.is_zero:
                return S.Zero

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\expressions\from_indexed_to_array.py ===
from collections import defaultdict

from sympy import Function
from sympy.combinatorics.permutations import _af_invert
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.tensor.array.expressions import ArrayElementwiseApplyFunc
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.combinatorics import Permutation
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, \
    get_shape, ArrayElement, _array_tensor_product, _array_diagonal, _array_contraction, _array_add, \
    _permute_dims, OneArray, ArrayAdd
from sympy.tensor.array.expressions.utils import _get_argindex, _get_diagonal_indices


def convert_indexed_to_array(expr, first_indices=None):
    r"""
    Parse indexed expression into a form useful for code generation.

    Examples
    ========

    >>> from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    >>> from sympy import MatrixSymbol, Sum, symbols

    >>> i, j, k, d = symbols("i j k d")
    >>> M = MatrixSymbol("M", d, d)
    >>> N = MatrixSymbol("N", d, d)

    Recognize the trace in summation form:

    >>> expr = Sum(M[i, i], (i, 0, d-1))
    >>> convert_indexed_to_array(expr)
    ArrayContraction(M, (0, 1))

    Recognize the extraction of the diagonal by using the same index `i` on
    both axes of the matrix:

    >>> expr = M[i, i]
    >>> convert_indexed_to_array(expr)
    ArrayDiagonal(M, (0, 1))

    This function can help perform the transformation expressed in two
    different mathematical notations as:

    `\sum_{j=0}^{N-1} A_{i,j} B_{j,k} \Longrightarrow \mathbf{A}\cdot \mathbf{B}`

    Recognize the matrix multiplication in summation form:

    >>> expr = Sum(M[i, j]*N[j, k], (j, 0, d-1))
    >>> convert_indexed_to_array(expr)
    ArrayContraction(ArrayTensorProduct(M, N), (1, 2))

    Specify that ``k`` has to be the starting index:

    >>> convert_indexed_to_array(expr, first_indices=[k])
    ArrayContraction(ArrayTensorProduct(N, M), (0, 3))
    """

    result, indices = _convert_indexed_to_array(expr)

    if any(isinstance(i, (int, Integer)) for i in indices):
        result = ArrayElement(result, indices)
        indices = []

    if not first_indices:
        return result

    def _check_is_in(elem, indices):
        if elem in indices:
            return True
        if any(elem in i for i in indices if isinstance(i, frozenset)):
            return True
        return False

    repl = {j: i for i in indices if isinstance(i, frozenset) for j in i}
    first_indices = [repl.get(i, i) for i in first_indices]
    for i in first_indices:
        if not _check_is_in(i, indices):
            first_indices.remove(i)
    first_indices.extend([i for i in indices if not _check_is_in(i, first_indices)])

    def _get_pos(elem, indices):
        if elem in indices:
            return indices.index(elem)
        for i, e in enumerate(indices):
            if not isinstance(e, frozenset):
                continue
            if elem in e:
                return i
        raise ValueError("not found")

    permutation = _af_invert([_get_pos(i, first_indices) for i in indices])
    if isinstance(result, ArrayAdd):
        return _array_add(*[_permute_dims(arg, permutation) for arg in result.args])
    else:
        return _permute_dims(result, permutation)


def _convert_indexed_to_array(expr):
    if isinstance(expr, Sum):
        function = expr.function
        summation_indices = expr.variables
        subexpr, subindices = _convert_indexed_to_array(function)
        subindicessets = {j: i for i in subindices if isinstance(i, frozenset) for j in i}
        summation_indices = sorted({subindicessets.get(i, i) for i in summation_indices}, key=default_sort_key)
        # TODO: check that Kronecker delta is only contracted to one other element:
        kronecker_indices = set()
        if isinstance(function, Mul):
            for arg in function.args:
                if not isinstance(arg, KroneckerDelta):
                    continue
                arg_indices = sorted(set(arg.indices), key=default_sort_key)
                if len(arg_indices) == 2:
                    kronecker_indices.update(arg_indices)
        kronecker_indices = sorted(kronecker_indices, key=default_sort_key)
        # Check dimensional consistency:
        shape = get_shape(subexpr)
        if shape:
            for ind, istart, iend in expr.limits:
                i = _get_argindex(subindices, ind)
                if istart != 0 or iend+1 != shape[i]:
                    raise ValueError("summation index and array dimension mismatch: %s" % ind)
        contraction_indices = []
        subindices = list(subindices)
        if isinstance(subexpr, ArrayDiagonal):
            diagonal_indices = list(subexpr.diagonal_indices)
            dindices = subindices[-len(diagonal_indices):]
            subindices = subindices[:-len(diagonal_indices)]
            for index in summation_indices:
                if index in dindices:
                    position = dindices.index(index)
                    contraction_indices.append(diagonal_indices[position])
                    diagonal_indices[position] = None
            diagonal_indices = [i for i in diagonal_indices if i is not None]
            for i, ind in enumerate(subindices):
                if ind in summation_indices:
                    pass
            if diagonal_indices:
                subexpr = _array_diagonal(subexpr.expr, *diagonal_indices)
            else:
                subexpr = subexpr.expr

        axes_contraction = defaultdict(list)
        for i, ind in enumerate(subindices):
            include = all(j not in kronecker_indices for j in ind) if isinstance(ind, frozenset) else ind not in kronecker_indices
            if ind in summation_indices and include:
                axes_contraction[ind].append(i)
                subindices[i] = None
        for k, v in axes_contraction.items():
            if any(i in kronecker_indices for i in k) if isinstance(k, frozenset) else k in kronecker_indices:
                continue
            contraction_indices.append(tuple(v))
        free_indices = [i for i in subindices if i is not None]
        indices_ret = list(free_indices)
        indices_ret.sort(key=lambda x: free_indices.index(x))
        return _array_contraction(
                subexpr,
                *contraction_indices,
                free_indices=free_indices
            ), tuple(indices_ret)
    if isinstance(expr, Mul):
        args, indices = zip(*[_convert_indexed_to_array(arg) for arg in expr.args])
        # Check if there are KroneckerDelta objects:
        kronecker_delta_repl = {}
        for arg in args:
            if not isinstance(arg, KroneckerDelta):
                continue
            # Diagonalize two indices:
            i, j = arg.indices
            kindices = set(arg.indices)
            if i in kronecker_delta_repl:
                kindices.update(kronecker_delta_repl[i])
            if j in kronecker_delta_repl:
                kindices.update(kronecker_delta_repl[j])
            kindices = frozenset(kindices)
            for index in kindices:
                kronecker_delta_repl[index] = kindices
        # Remove KroneckerDelta objects, their relations should be handled by
        # ArrayDiagonal:
        newargs = []
        newindices = []
        for arg, loc_indices in zip(args, indices):
            if isinstance(arg, KroneckerDelta):
                continue
            newargs.append(arg)
            newindices.append(loc_indices)
        flattened_indices = [kronecker_delta_repl.get(j, j) for i in newindices for j in i]
        diagonal_indices, ret_indices = _get_diagonal_indices(flattened_indices)
        tp = _array_tensor_product(*newargs)
        if diagonal_indices:
            return _array_diagonal(tp, *diagonal_indices), ret_indices
        else:
            return tp, ret_indices
    if isinstance(expr, MatrixElement):
        indices = expr.args[1:]
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return _array_diagonal(expr.args[0], *diagonal_indices), ret_indices
        else:
            return expr.args[0], ret_indices
    if isinstance(expr, ArrayElement):
        indices = expr.indices
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return _array_diagonal(expr.name, *diagonal_indices), ret_indices
        else:
            return expr.name, ret_indices
    if isinstance(expr, Indexed):
        indices = expr.indices
        diagonal_indices, ret_indices = _get_diagonal_indices(indices)
        if diagonal_indices:
            return _array_diagonal(expr.base, *diagonal_indices), ret_indices
        else:
            return expr.args[0], ret_indices
    if isinstance(expr, IndexedBase):
        raise NotImplementedError
    if isinstance(expr, KroneckerDelta):
        return expr, expr.indices
    if isinstance(expr, Add):
        args, indices = zip(*[_convert_indexed_to_array(arg) for arg in expr.args])
        args = list(args)
        # Check if all indices are compatible. Otherwise expand the dimensions:
        index0 = []
        shape0 = []
        for arg, arg_indices in zip(args, indices):
            arg_indices_set = set(arg_indices)
            arg_indices_missing = arg_indices_set.difference(index0)
            index0.extend([i for i in arg_indices if i in arg_indices_missing])
            arg_shape = get_shape(arg)
            shape0.extend([arg_shape[i] for i, e in enumerate(arg_indices) if e in arg_indices_missing])
        for i, (arg, arg_indices) in enumerate(zip(args, indices)):
            if len(arg_indices) < len(index0):
                missing_indices_pos = [i for i, e in enumerate(index0) if e not in arg_indices]
                missing_shape = [shape0[i] for i in missing_indices_pos]
                arg_indices = tuple(index0[j] for j in missing_indices_pos) + arg_indices
                args[i] = _array_tensor_product(OneArray(*missing_shape), args[i])
            permutation = Permutation([arg_indices.index(j) for j in index0])
            # Perform index permutations:
            args[i] = _permute_dims(args[i], permutation)
        return _array_add(*args), tuple(index0)
    if isinstance(expr, Pow):
        subexpr, subindices = _convert_indexed_to_array(expr.base)
        if isinstance(expr.exp, (int, Integer)):
            diags = zip(*[(2*i, 2*i + 1) for i in range(expr.exp)])
            arr = _array_diagonal(_array_tensor_product(*[subexpr for i in range(expr.exp)]), *diags)
            return arr, subindices
    if isinstance(expr, Function):
        subexpr, subindices = _convert_indexed_to_array(expr.args[0])
        return ArrayElementwiseApplyFunc(type(expr), subexpr), subindices
    return expr, ()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\jinja2\tests.py ===
"""Built-in template tests used with the ``is`` operator."""

import operator
import typing as t
from collections import abc
from numbers import Number

from .runtime import Undefined
from .utils import pass_environment

if t.TYPE_CHECKING:
    from .environment import Environment


def test_odd(value: int) -> bool:
    """Return true if the variable is odd."""
    return value % 2 == 1


def test_even(value: int) -> bool:
    """Return true if the variable is even."""
    return value % 2 == 0


def test_divisibleby(value: int, num: int) -> bool:
    """Check if a variable is divisible by a number."""
    return value % num == 0


def test_defined(value: t.Any) -> bool:
    """Return true if the variable is defined:

    .. sourcecode:: jinja

        {% if variable is defined %}
            value of variable: {{ variable }}
        {% else %}
            variable is not defined
        {% endif %}

    See the :func:`default` filter for a simple way to set undefined
    variables.
    """
    return not isinstance(value, Undefined)


def test_undefined(value: t.Any) -> bool:
    """Like :func:`defined` but the other way round."""
    return isinstance(value, Undefined)


@pass_environment
def test_filter(env: "Environment", value: str) -> bool:
    """Check if a filter exists by name. Useful if a filter may be
    optionally available.

    .. code-block:: jinja

        {% if 'markdown' is filter %}
            {{ value | markdown }}
        {% else %}
            {{ value }}
        {% endif %}

    .. versionadded:: 3.0
    """
    return value in env.filters


@pass_environment
def test_test(env: "Environment", value: str) -> bool:
    """Check if a test exists by name. Useful if a test may be
    optionally available.

    .. code-block:: jinja

        {% if 'loud' is test %}
            {% if value is loud %}
                {{ value|upper }}
            {% else %}
                {{ value|lower }}
            {% endif %}
        {% else %}
            {{ value }}
        {% endif %}

    .. versionadded:: 3.0
    """
    return value in env.tests


def test_none(value: t.Any) -> bool:
    """Return true if the variable is none."""
    return value is None


def test_boolean(value: t.Any) -> bool:
    """Return true if the object is a boolean value.

    .. versionadded:: 2.11
    """
    return value is True or value is False


def test_false(value: t.Any) -> bool:
    """Return true if the object is False.

    .. versionadded:: 2.11
    """
    return value is False


def test_true(value: t.Any) -> bool:
    """Return true if the object is True.

    .. versionadded:: 2.11
    """
    return value is True


# NOTE: The existing 'number' test matches booleans and floats
def test_integer(value: t.Any) -> bool:
    """Return true if the object is an integer.

    .. versionadded:: 2.11
    """
    return isinstance(value, int) and value is not True and value is not False


# NOTE: The existing 'number' test matches booleans and integers
def test_float(value: t.Any) -> bool:
    """Return true if the object is a float.

    .. versionadded:: 2.11
    """
    return isinstance(value, float)


def test_lower(value: str) -> bool:
    """Return true if the variable is lowercased."""
    return str(value).islower()


def test_upper(value: str) -> bool:
    """Return true if the variable is uppercased."""
    return str(value).isupper()


def test_string(value: t.Any) -> bool:
    """Return true if the object is a string."""
    return isinstance(value, str)


def test_mapping(value: t.Any) -> bool:
    """Return true if the object is a mapping (dict etc.).

    .. versionadded:: 2.6
    """
    return isinstance(value, abc.Mapping)


def test_number(value: t.Any) -> bool:
    """Return true if the variable is a number."""
    return isinstance(value, Number)


def test_sequence(value: t.Any) -> bool:
    """Return true if the variable is a sequence. Sequences are variables
    that are iterable.
    """
    try:
        len(value)
        value.__getitem__  # noqa B018
    except Exception:
        return False

    return True


def test_sameas(value: t.Any, other: t.Any) -> bool:
    """Check if an object points to the same memory address than another
    object:

    .. sourcecode:: jinja

        {% if foo.attribute is sameas false %}
            the foo attribute really is the `False` singleton
        {% endif %}
    """
    return value is other


def test_iterable(value: t.Any) -> bool:
    """Check if it's possible to iterate over an object."""
    try:
        iter(value)
    except TypeError:
        return False

    return True


def test_escaped(value: t.Any) -> bool:
    """Check if the value is escaped."""
    return hasattr(value, "__html__")


def test_in(value: t.Any, seq: t.Container[t.Any]) -> bool:
    """Check if value is in seq.

    .. versionadded:: 2.10
    """
    return value in seq


TESTS = {
    "odd": test_odd,
    "even": test_even,
    "divisibleby": test_divisibleby,
    "defined": test_defined,
    "undefined": test_undefined,
    "filter": test_filter,
    "test": test_test,
    "none": test_none,
    "boolean": test_boolean,
    "false": test_false,
    "true": test_true,
    "integer": test_integer,
    "float": test_float,
    "lower": test_lower,
    "upper": test_upper,
    "string": test_string,
    "mapping": test_mapping,
    "number": test_number,
    "sequence": test_sequence,
    "iterable": test_iterable,
    "callable": callable,
    "sameas": test_sameas,
    "escaped": test_escaped,
    "in": test_in,
    "==": operator.eq,
    "eq": operator.eq,
    "equalto": operator.eq,
    "!=": operator.ne,
    "ne": operator.ne,
    ">": operator.gt,
    "gt": operator.gt,
    "greaterthan": operator.gt,
    "ge": operator.ge,
    ">=": operator.ge,
    "<": operator.lt,
    "lt": operator.lt,
    "lessthan": operator.lt,
    "<=": operator.le,
    "le": operator.le,
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\static_quantize_runner.py ===
import argparse
import json
import os

import numpy as np
import onnx

import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, StaticQuantConfig, quantize
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod


class OnnxModelCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path):
        self.model_dir = os.path.dirname(model_path)
        data_dirs = [
            os.path.join(self.model_dir, a) for a in os.listdir(self.model_dir) if a.startswith("test_data_set_")
        ]
        model_inputs = onnxruntime.InferenceSession(model_path).get_inputs()
        name2tensors = []
        for data_dir in data_dirs:
            name2tensor = {}
            data_paths = [os.path.join(data_dir, a) for a in sorted(os.listdir(data_dir))]
            data_ndarrays = [self.read_onnx_pb_data(data_path) for data_path in data_paths]
            for model_input, data_ndarray in zip(model_inputs, data_ndarrays, strict=False):
                name2tensor[model_input.name] = data_ndarray
            name2tensors.append(name2tensor)
        assert len(name2tensors) == len(data_dirs)
        assert len(name2tensors[0]) == len(model_inputs)

        self.calibration_data = iter(name2tensors)

    def get_next(self) -> dict:
        """generate the input data dict for ONNXinferenceSession run"""
        return next(self.calibration_data, None)

    def read_onnx_pb_data(self, file_pb):
        tensor = onnx.TensorProto()
        with open(file_pb, "rb") as f:
            tensor.ParseFromString(f.read())
        ret = onnx.numpy_helper.to_array(tensor)
        return ret


def parse_arguments():
    parser = argparse.ArgumentParser(description="The arguments for static quantization")
    parser.add_argument("-i", "--input_model_path", required=True, help="Path to the input onnx model")
    parser.add_argument(
        "-o", "--output_quantized_model_path", required=True, help="Path to the output quantized onnx model"
    )
    parser.add_argument(
        "--activation_type",
        choices=["qint8", "quint8", "qint16", "quint16", "qint4", "quint4", "qfloat8e4m3fn"],
        default="quint8",
        help="Activation quantization type used",
    )
    parser.add_argument(
        "--weight_type",
        choices=["qint8", "quint8", "qint16", "quint16", "qint4", "quint4", "qfloat8e4m3fn"],
        default="qint8",
        help="Weight quantization type used",
    )
    parser.add_argument("--enable_subgraph", action="store_true", help="If set, subgraph will be quantized.")
    parser.add_argument(
        "--force_quantize_no_input_check",
        action="store_true",
        help="By default, some latent operators like maxpool, transpose, do not quantize if their input is not"
        " quantized already. Setting to True to force such operator always quantize input and so generate"
        " quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.",
    )
    parser.add_argument(
        "--matmul_const_b_only",
        action="store_true",
        help="If set, only MatMul with const B will be quantized.",
    )
    parser.add_argument(
        "--add_qdq_pair_to_weight",
        action="store_true",
        help="If set, it remains floating-point weight and inserts both QuantizeLinear/DeQuantizeLinear"
        " nodes to weight.",
    )
    parser.add_argument(
        "--dedicated_qdq_pair",
        action="store_true",
        help="If set, it will create identical and dedicated QDQ pair for each node.",
    )
    parser.add_argument(
        "--op_types_to_exclude_output_quantization",
        nargs="+",
        default=[],
        help="If any op type is specified, it won't quantize the output of ops with this specific op types.",
    )
    parser.add_argument(
        "--calibration_method",
        default="minmax",
        choices=["minmax", "entropy", "percentile", "distribution"],
        help="Calibration method used",
    )
    parser.add_argument("--quant_format", default="qdq", choices=["qdq", "qoperator"], help="Quantization format used")
    parser.add_argument(
        "--calib_tensor_range_symmetric",
        action="store_true",
        help="If enabled, the final range of tensor during calibration will be explicitly"
        " set to symmetric to central point 0",
    )
    # TODO: --calib_strided_minmax"
    # TODO: --calib_moving_average_constant"
    # TODO: --calib_max_intermediate_outputs"
    parser.add_argument(
        "--calib_moving_average",
        action="store_true",
        help="If enabled, the moving average of"
        " the minimum and maximum values will be computed when the calibration method selected is MinMax.",
    )
    parser.add_argument(
        "--disable_quantize_bias",
        action="store_true",
        help="Whether to quantize floating-point biases by solely inserting a DeQuantizeLinear node"
        " If not set, it remains floating-point bias and does not insert any quantization nodes"
        " associated with biases.",
    )

    # TODO: Add arguments related to Smooth Quant

    parser.add_argument(
        "--use_qdq_contrib_ops",
        action="store_true",
        help="If set, the inserted QuantizeLinear and DequantizeLinear ops will have the com.microsoft domain,"
        " which forces use of ONNX Runtime's QuantizeLinear and DequantizeLinear contrib op implementations.",
    )
    parser.add_argument(
        "--minimum_real_range",
        type=float,
        default=0.0001,
        help="If set to a floating-point value, the calculation of the quantization parameters"
        " (i.e., scale and zero point) will enforce a minimum range between rmin and rmax. If (rmax-rmin)"
        " is less than the specified minimum range, rmax will be set to rmin + MinimumRealRange. This is"
        " necessary for EPs like QNN that require a minimum floating-point range when determining "
        " quantization parameters.",
    )
    parser.add_argument(
        "--qdq_keep_removable_activations",
        action="store_true",
        help="If set, removable activations (e.g., Clip or Relu) will not be removed,"
        " and will be explicitly represented in the QDQ model.",
    )
    parser.add_argument(
        "--qdq_disable_weight_adjust_for_int32_bias",
        action="store_true",
        help="If set, QDQ quantizer will not adjust the weight's scale when the bias"
        " has a scale (input_scale * weight_scale) that is too small.",
    )
    parser.add_argument("--per_channel", action="store_true", help="Whether using per-channel quantization")
    parser.add_argument(
        "--nodes_to_quantize",
        nargs="+",
        default=None,
        help="List of nodes names to quantize. When this list is not None only the nodes in this list are quantized.",
    )
    parser.add_argument(
        "--nodes_to_exclude",
        nargs="+",
        default=None,
        help="List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.",
    )
    parser.add_argument(
        "--op_per_channel_axis",
        nargs=2,
        action="append",
        metavar=("OP_TYPE", "PER_CHANNEL_AXIS"),
        default=[],
        help="Set channel axis for specific op type, for example: --op_per_channel_axis MatMul 1, and it's"
        " effective only when per channel quantization is supported and per_channel is True. If specific"
        " op type supports per channel quantization but not explicitly specified with channel axis,"
        " default channel axis will be used.",
    )
    parser.add_argument("--tensor_quant_overrides", help="Set the json file for tensor quantization overrides.")
    return parser.parse_args()


def get_tensor_quant_overrides(file):
    # TODO: Enhance the function to handle more real cases of json file
    if not file:
        return {}
    with open(file) as f:
        quant_override_dict = json.load(f)
    for tensor in quant_override_dict:
        for enc_dict in quant_override_dict[tensor]:
            enc_dict["scale"] = np.array(enc_dict["scale"], dtype=np.float32)
            enc_dict["zero_point"] = np.array(enc_dict["zero_point"])
    return quant_override_dict


def main():
    args = parse_arguments()
    data_reader = OnnxModelCalibrationDataReader(model_path=args.input_model_path)
    arg2quant_type = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
        "qint16": QuantType.QInt16,
        "quint16": QuantType.QUInt16,
        "qint4": QuantType.QInt4,
        "quint4": QuantType.QUInt4,
        "qfloat8e4m3fn": QuantType.QFLOAT8E4M3FN,
    }
    activation_type = arg2quant_type[args.activation_type]
    weight_type = arg2quant_type[args.weight_type]
    qdq_op_type_per_channel_support_to_axis = dict(args.op_per_channel_axis)
    extra_options = {
        "EnableSubgraph": args.enable_subgraph,
        "ForceQuantizeNoInputCheck": args.force_quantize_no_input_check,
        "MatMulConstBOnly": args.matmul_const_b_only,
        "AddQDQPairToWeight": args.add_qdq_pair_to_weight,
        "OpTypesToExcludeOutputQuantization": args.op_types_to_exclude_output_quantization,
        "DedicatedQDQPair": args.dedicated_qdq_pair,
        "QDQOpTypePerChannelSupportToAxis": qdq_op_type_per_channel_support_to_axis,
        "CalibTensorRangeSymmetric": args.calib_tensor_range_symmetric,
        "CalibMovingAverage": args.calib_moving_average,
        "QuantizeBias": not args.disable_quantize_bias,
        "UseQDQContribOps": args.use_qdq_contrib_ops,
        "MinimumRealRange": args.minimum_real_range,
        "QDQKeepRemovableActivations": args.qdq_keep_removable_activations,
        "QDQDisableWeightAdjustForInt32Bias": args.qdq_disable_weight_adjust_for_int32_bias,
        # Load json file for encoding override
        "TensorQuantOverrides": get_tensor_quant_overrides(args.tensor_quant_overrides),
    }
    arg2calib_method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    arg2quant_format = {
        "qdq": QuantFormat.QDQ,
        "qoperator": QuantFormat.QOperator,
    }
    sqc = StaticQuantConfig(
        calibration_data_reader=data_reader,
        calibrate_method=arg2calib_method[args.calibration_method],
        quant_format=arg2quant_format[args.quant_format],
        activation_type=activation_type,
        weight_type=weight_type,
        op_types_to_quantize=None,
        nodes_to_quantize=args.nodes_to_quantize,
        nodes_to_exclude=args.nodes_to_exclude,
        per_channel=args.per_channel,
        reduce_range=False,
        use_external_data_format=False,
        calibration_providers=None,  # Use CPUExecutionProvider
        extra_options=extra_options,
    )
    quantize(model_input=args.input_model_path, model_output=args.output_quantized_model_path, quant_config=sqc)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\compare_bert_results.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# It is a tool to compare the inference results of the original model and optimized model.

import argparse
import statistics
from pathlib import Path

import numpy as np
import psutil
from bert_perf_test import create_session, onnxruntime_inference
from bert_test_data import generate_test_data, get_bert_inputs, output_test_data


def run_model(model_path, all_inputs, use_gpu, disable_optimization):
    import onnxruntime

    graph_optimization_level = None
    if disable_optimization:
        graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    intra_op_num_threads = psutil.cpu_count(logical=False)

    session = create_session(
        model_path, use_gpu, "cuda" if use_gpu else "cpu", intra_op_num_threads, graph_optimization_level
    )

    output_names = [output.name for output in session.get_outputs()]
    results, latency_list = onnxruntime_inference(session, all_inputs, output_names)
    return results, latency_list, output_names


def compare(baseline_results, treatment_results, verbose, rtol=1e-1, atol=1e-3):
    # Validate the output of baseline and treatment, to make sure the results are similar.
    diff_count = 0
    max_abs_diff = 0
    max_diff_percentage = 0
    case_passed = True
    for test_case_id, results in enumerate(baseline_results):
        for i in range(len(results)):
            treatment_output = treatment_results[test_case_id][i]
            abs_diff_tensor = np.abs(treatment_output - results[i])
            abs_diff = np.amax(abs_diff_tensor)
            if verbose and abs_diff > atol:
                print("abs_diff", abs_diff)
                print("treatment", treatment_output)
                print("baseline", results[i])

            count_exceeding = np.sum(abs_diff_tensor > atol)
            total_elements = abs_diff_tensor.size
            percentage_exceeding = (count_exceeding / total_elements) * 100
            max_diff_percentage = max(max_diff_percentage, percentage_exceeding)

            max_abs_diff = max(max_abs_diff, abs_diff)
            if not np.allclose(results[i].tolist(), treatment_output.tolist(), rtol=rtol, atol=atol):
                if case_passed:
                    case_passed = False
                    diff_count += 1

                    if verbose:
                        print(f"case {test_case_id} output {i}")
                        print(f"baseline={results[i].tolist()}\ntreatment={treatment_output}")
                        print(f"abs_diff={abs_diff}")

    if diff_count == 0:
        print(f"100% passed for {len(baseline_results)} random inputs given thresholds (rtol={rtol}, atol={atol}).")
    else:
        print(
            f"WARNING: {diff_count} out of {len(baseline_results)} results NOT passed for thresholds (rtol={rtol}, atol={atol})."
        )

    print(f"maximum absolute difference={max_abs_diff}")
    print(f"maximum percentage of elements that exceeds atol={atol} is {max_diff_percentage:.3f}%")
    return max_abs_diff, case_passed


def run_test(
    baseline_model,
    optimized_model,
    output_dir,
    batch_size,
    sequence_length,
    use_gpu,
    test_cases,
    seed,
    verbose,
    rtol,
    atol,
    input_ids_name,
    segment_ids_name,
    input_mask_name,
    mask_type,
    dictionary_size: int = 1024,
):
    # Try deduce input names from optimized model.
    input_ids, segment_ids, input_mask = get_bert_inputs(
        optimized_model, input_ids_name, segment_ids_name, input_mask_name
    )

    # Use random mask length for accuracy test. It might introduce slight inflation in latency reported in this script.
    average_sequence_length = int(sequence_length / 2) if sequence_length >= 2 else sequence_length
    all_inputs = generate_test_data(
        batch_size,
        sequence_length,
        test_cases,
        seed,
        verbose,
        input_ids,
        segment_ids,
        input_mask,
        average_sequence_length,
        True,  # random sequence length
        mask_type,
        dictionary_size=dictionary_size,
    )

    baseline_results, baseline_latency, output_names = run_model(
        baseline_model, all_inputs, use_gpu, disable_optimization=True
    )
    if verbose:
        print(f"baseline average latency (all optimizations disabled): {statistics.mean(baseline_latency) * 1000} ms")

    if output_dir is not None:
        for i, inputs in enumerate(all_inputs):
            output_test_data(output_dir, i, inputs)

    treatment_results, treatment_latency, treatment_output_names = run_model(
        optimized_model, all_inputs, use_gpu, disable_optimization=False
    )
    if verbose:
        print(f"treatment average latency: {statistics.mean(treatment_latency) * 1000} ms")

    # Validate the output of baseline and treatment, to make sure the results are similar.
    return compare(baseline_results, treatment_results, verbose, rtol, atol)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", required=True, type=str, help="baseline onnx model path.")

    parser.add_argument(
        "--optimized_model",
        required=True,
        type=str,
        default=None,
        help="path of the optimized model. It shall have same inputs as the baseline model.",
    )

    parser.add_argument(
        "--output_dir",
        required=False,
        type=str,
        default=None,
        help="output test data path. If not specified, test data will not be saved.",
    )

    parser.add_argument("--batch_size", required=True, type=int, help="batch size of input")

    parser.add_argument(
        "--sequence_length",
        required=True,
        type=int,
        help="maximum sequence length of input",
    )

    parser.add_argument("--rtol", required=False, type=float, default=1e-3, help="relative tolerance")

    parser.add_argument("--atol", required=False, type=float, default=1e-4, help="absolute tolerance")

    parser.add_argument(
        "--samples",
        required=False,
        type=int,
        default=100,
        help="number of test cases to be generated",
    )

    parser.add_argument("--seed", required=False, type=int, default=3, help="random seed")

    parser.add_argument("--use_gpu", required=False, action="store_true", help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="print verbose information",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--input_ids",
        required=False,
        type=str,
        default=None,
        help="input name for input ids",
    )
    parser.add_argument(
        "--segment_ids",
        required=False,
        type=str,
        default=None,
        help="input name for segment ids",
    )
    parser.add_argument(
        "--input_mask",
        required=False,
        type=str,
        default=None,
        help="input name for attention mask",
    )

    parser.add_argument(
        "--mask_type",
        required=False,
        type=int,
        default=2,
        help="mask type: (1: mask index or sequence length, 2: raw 2D mask, 3: key len, cumulated lengths of query and key)",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.output_dir is not None:
        # create the output directory if not existed
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=True)

    run_test(
        args.baseline_model,
        args.optimized_model,
        args.output_dir,
        args.batch_size,
        args.sequence_length,
        args.use_gpu,
        args.samples,
        args.seed,
        args.verbose,
        args.rtol,
        args.atol,
        args.input_ids,
        args.segment_ids,
        args.input_mask,
        args.mask_type,
    )


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\freeze.py ===
import collections
import logging
import os
from dataclasses import dataclass, field
from typing import Container, Dict, Generator, Iterable, List, NamedTuple, Optional, Set

from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import InvalidVersion

from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.req.constructors import (
    install_req_from_editable,
    install_req_from_line,
)
from pip._internal.req.req_file import COMMENT_RE
from pip._internal.utils.direct_url_helpers import direct_url_as_pep440_direct_reference

logger = logging.getLogger(__name__)


class _EditableInfo(NamedTuple):
    requirement: str
    comments: List[str]


def freeze(
    requirement: Optional[List[str]] = None,
    local_only: bool = False,
    user_only: bool = False,
    paths: Optional[List[str]] = None,
    isolated: bool = False,
    exclude_editable: bool = False,
    skip: Container[str] = (),
) -> Generator[str, None, None]:
    installations: Dict[str, FrozenRequirement] = {}

    dists = get_environment(paths).iter_installed_distributions(
        local_only=local_only,
        skip=(),
        user_only=user_only,
    )
    for dist in dists:
        req = FrozenRequirement.from_dist(dist)
        if exclude_editable and req.editable:
            continue
        installations[req.canonical_name] = req

    if requirement:
        # the options that don't get turned into an InstallRequirement
        # should only be emitted once, even if the same option is in multiple
        # requirements files, so we need to keep track of what has been emitted
        # so that we don't emit it again if it's seen again
        emitted_options: Set[str] = set()
        # keep track of which files a requirement is in so that we can
        # give an accurate warning if a requirement appears multiple times.
        req_files: Dict[str, List[str]] = collections.defaultdict(list)
        for req_file_path in requirement:
            with open(req_file_path) as req_file:
                for line in req_file:
                    if (
                        not line.strip()
                        or line.strip().startswith("#")
                        or line.startswith(
                            (
                                "-r",
                                "--requirement",
                                "-f",
                                "--find-links",
                                "-i",
                                "--index-url",
                                "--pre",
                                "--trusted-host",
                                "--process-dependency-links",
                                "--extra-index-url",
                                "--use-feature",
                            )
                        )
                    ):
                        line = line.rstrip()
                        if line not in emitted_options:
                            emitted_options.add(line)
                            yield line
                        continue

                    if line.startswith("-e") or line.startswith("--editable"):
                        if line.startswith("-e"):
                            line = line[2:].strip()
                        else:
                            line = line[len("--editable") :].strip().lstrip("=")
                        line_req = install_req_from_editable(
                            line,
                            isolated=isolated,
                        )
                    else:
                        line_req = install_req_from_line(
                            COMMENT_RE.sub("", line).strip(),
                            isolated=isolated,
                        )

                    if not line_req.name:
                        logger.info(
                            "Skipping line in requirement file [%s] because "
                            "it's not clear what it would install: %s",
                            req_file_path,
                            line.strip(),
                        )
                        logger.info(
                            "  (add #egg=PackageName to the URL to avoid"
                            " this warning)"
                        )
                    else:
                        line_req_canonical_name = canonicalize_name(line_req.name)
                        if line_req_canonical_name not in installations:
                            # either it's not installed, or it is installed
                            # but has been processed already
                            if not req_files[line_req.name]:
                                logger.warning(
                                    "Requirement file [%s] contains %s, but "
                                    "package %r is not installed",
                                    req_file_path,
                                    COMMENT_RE.sub("", line).strip(),
                                    line_req.name,
                                )
                            else:
                                req_files[line_req.name].append(req_file_path)
                        else:
                            yield str(installations[line_req_canonical_name]).rstrip()
                            del installations[line_req_canonical_name]
                            req_files[line_req.name].append(req_file_path)

        # Warn about requirements that were included multiple times (in a
        # single requirements file or in different requirements files).
        for name, files in req_files.items():
            if len(files) > 1:
                logger.warning(
                    "Requirement %s included multiple times [%s]",
                    name,
                    ", ".join(sorted(set(files))),
                )

        yield ("## The following requirements were added by pip freeze:")
    for installation in sorted(installations.values(), key=lambda x: x.name.lower()):
        if installation.canonical_name not in skip:
            yield str(installation).rstrip()


def _format_as_name_version(dist: BaseDistribution) -> str:
    try:
        dist_version = dist.version
    except InvalidVersion:
        # legacy version
        return f"{dist.raw_name}==={dist.raw_version}"
    else:
        return f"{dist.raw_name}=={dist_version}"


def _get_editable_info(dist: BaseDistribution) -> _EditableInfo:
    """
    Compute and return values (req, comments) for use in
    FrozenRequirement.from_dist().
    """
    editable_project_location = dist.editable_project_location
    assert editable_project_location
    location = os.path.normcase(os.path.abspath(editable_project_location))

    from pip._internal.vcs import RemoteNotFoundError, RemoteNotValidError, vcs

    vcs_backend = vcs.get_backend_for_dir(location)

    if vcs_backend is None:
        display = _format_as_name_version(dist)
        logger.debug(
            'No VCS found for editable requirement "%s" in: %r',
            display,
            location,
        )
        return _EditableInfo(
            requirement=location,
            comments=[f"# Editable install with no version control ({display})"],
        )

    vcs_name = type(vcs_backend).__name__

    try:
        req = vcs_backend.get_src_requirement(location, dist.raw_name)
    except RemoteNotFoundError:
        display = _format_as_name_version(dist)
        return _EditableInfo(
            requirement=location,
            comments=[f"# Editable {vcs_name} install with no remote ({display})"],
        )
    except RemoteNotValidError as ex:
        display = _format_as_name_version(dist)
        return _EditableInfo(
            requirement=location,
            comments=[
                f"# Editable {vcs_name} install ({display}) with either a deleted "
                f"local remote or invalid URI:",
                f"# '{ex.url}'",
            ],
        )
    except BadCommand:
        logger.warning(
            "cannot determine version of editable source in %s "
            "(%s command not found in path)",
            location,
            vcs_backend.name,
        )
        return _EditableInfo(requirement=location, comments=[])
    except InstallationError as exc:
        logger.warning("Error when trying to get requirement for VCS system %s", exc)
    else:
        return _EditableInfo(requirement=req, comments=[])

    logger.warning("Could not determine repository location of %s", location)

    return _EditableInfo(
        requirement=location,
        comments=["## !! Could not determine repository location"],
    )


@dataclass(frozen=True)
class FrozenRequirement:
    name: str
    req: str
    editable: bool
    comments: Iterable[str] = field(default_factory=tuple)

    @property
    def canonical_name(self) -> NormalizedName:
        return canonicalize_name(self.name)

    @classmethod
    def from_dist(cls, dist: BaseDistribution) -> "FrozenRequirement":
        editable = dist.editable
        if editable:
            req, comments = _get_editable_info(dist)
        else:
            comments = []
            direct_url = dist.direct_url
            if direct_url:
                # if PEP 610 metadata is present, use it
                req = direct_url_as_pep440_direct_reference(direct_url, dist.raw_name)
            else:
                # name==version requirement
                req = _format_as_name_version(dist)

        return cls(dist.raw_name, req, editable, comments=comments)

    def __str__(self) -> str:
        req = self.req
        if self.editable:
            req = f"-e {req}"
        return "\n".join(list(self.comments) + [str(req)]) + "\n"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\commutator.py ===
"""The commutator: [A,B] = A*B - B*A."""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.kind import KindDispatcher
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.kind import _OperatorKind, OperatorKind


__all__ = [
    'Commutator'
]

#-----------------------------------------------------------------------------
# Commutator
#-----------------------------------------------------------------------------


class Commutator(Expr):
    """The standard commutator, in an unevaluated state.

    Explanation
    ===========

    Evaluating a commutator is defined [1]_ as: ``[A, B] = A*B - B*A``. This
    class returns the commutator in an unevaluated form. To evaluate the
    commutator, use the ``.doit()`` method.

    Canonical ordering of a commutator is ``[A, B]`` for ``A < B``. The
    arguments of the commutator are put into canonical order using ``__cmp__``.
    If ``B < A``, then ``[B, A]`` is returned as ``-[A, B]``.

    Parameters
    ==========

    A : Expr
        The first argument of the commutator [A,B].
    B : Expr
        The second argument of the commutator [A,B].

    Examples
    ========

    >>> from sympy.physics.quantum import Commutator, Dagger, Operator
    >>> from sympy.abc import x, y
    >>> A = Operator('A')
    >>> B = Operator('B')
    >>> C = Operator('C')

    Create a commutator and use ``.doit()`` to evaluate it:

    >>> comm = Commutator(A, B)
    >>> comm
    [A,B]
    >>> comm.doit()
    A*B - B*A

    The commutator orders it arguments in canonical order:

    >>> comm = Commutator(B, A); comm
    -[A,B]

    Commutative constants are factored out:

    >>> Commutator(3*x*A, x*y*B)
    3*x**2*y*[A,B]

    Using ``.expand(commutator=True)``, the standard commutator expansion rules
    can be applied:

    >>> Commutator(A+B, C).expand(commutator=True)
    [A,C] + [B,C]
    >>> Commutator(A, B+C).expand(commutator=True)
    [A,B] + [A,C]
    >>> Commutator(A*B, C).expand(commutator=True)
    [A,C]*B + A*[B,C]
    >>> Commutator(A, B*C).expand(commutator=True)
    [A,B]*C + B*[A,C]

    Adjoint operations applied to the commutator are properly applied to the
    arguments:

    >>> Dagger(Commutator(A, B))
    -[Dagger(A),Dagger(B)]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Commutator
    """
    is_commutative = False

    _kind_dispatcher = KindDispatcher("Commutator_kind_dispatcher", commutative=True)

    @property
    def kind(self):
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    def __new__(cls, A, B):
        r = cls.eval(A, B)
        if r is not None:
            return r
        obj = Expr.__new__(cls, A, B)
        return obj

    @classmethod
    def eval(cls, a, b):
        if not (a and b):
            return S.Zero
        if a == b:
            return S.Zero
        if a.is_commutative or b.is_commutative:
            return S.Zero

        # [xA,yB]  ->  xy*[A,B]
        ca, nca = a.args_cnc()
        cb, ncb = b.args_cnc()
        c_part = ca + cb
        if c_part:
            return Mul(Mul(*c_part), cls(Mul._from_args(nca), Mul._from_args(ncb)))

        # Canonical ordering of arguments
        # The Commutator [A, B] is in canonical form if A < B.
        if a.compare(b) == 1:
            return S.NegativeOne*cls(b, a)

    def _expand_pow(self, A, B, sign):
        exp = A.exp
        if not exp.is_integer or not exp.is_constant() or abs(exp) <= 1:
            # nothing to do
            return self
        base = A.base
        if exp.is_negative:
            base = A.base**-1
            exp = -exp
        comm = Commutator(base, B).expand(commutator=True)

        result = base**(exp - 1) * comm
        for i in range(1, exp):
            result += base**(exp - 1 - i) * comm * base**i
        return sign*result.expand()

    def _eval_expand_commutator(self, **hints):
        A = self.args[0]
        B = self.args[1]

        if isinstance(A, Add):
            # [A + B, C]  ->  [A, C] + [B, C]
            sargs = []
            for term in A.args:
                comm = Commutator(term, B)
                if isinstance(comm, Commutator):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            return Add(*sargs)
        elif isinstance(B, Add):
            # [A, B + C]  ->  [A, B] + [A, C]
            sargs = []
            for term in B.args:
                comm = Commutator(A, term)
                if isinstance(comm, Commutator):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            return Add(*sargs)
        elif isinstance(A, Mul):
            # [A*B, C] -> A*[B, C] + [A, C]*B
            a = A.args[0]
            b = Mul(*A.args[1:])
            c = B
            comm1 = Commutator(b, c)
            comm2 = Commutator(a, c)
            if isinstance(comm1, Commutator):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, Commutator):
                comm2 = comm2._eval_expand_commutator()
            first = Mul(a, comm1)
            second = Mul(comm2, b)
            return Add(first, second)
        elif isinstance(B, Mul):
            # [A, B*C] -> [A, B]*C + B*[A, C]
            a = A
            b = B.args[0]
            c = Mul(*B.args[1:])
            comm1 = Commutator(a, b)
            comm2 = Commutator(a, c)
            if isinstance(comm1, Commutator):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, Commutator):
                comm2 = comm2._eval_expand_commutator()
            first = Mul(comm1, c)
            second = Mul(b, comm2)
            return Add(first, second)
        elif isinstance(A, Pow):
            # [A**n, C] -> A**(n - 1)*[A, C] + A**(n - 2)*[A, C]*A + ... + [A, C]*A**(n-1)
            return self._expand_pow(A, B, 1)
        elif isinstance(B, Pow):
            # [A, C**n] -> C**(n - 1)*[C, A] + C**(n - 2)*[C, A]*C + ... + [C, A]*C**(n-1)
            return self._expand_pow(B, A, -1)

        # No changes, so return self
        return self

    def doit(self, **hints):
        """ Evaluate commutator """
        # Keep the import of Operator here to avoid problems with
        # circular imports.
        from sympy.physics.quantum.operator import Operator
        A = self.args[0]
        B = self.args[1]
        if isinstance(A, Operator) and isinstance(B, Operator):
            try:
                comm = A._eval_commutator(B, **hints)
            except NotImplementedError:
                try:
                    comm = -1*B._eval_commutator(A, **hints)
                except NotImplementedError:
                    comm = None
            if comm is not None:
                return comm.doit(**hints)
        return (A*B - B*A).doit(**hints)

    def _eval_adjoint(self):
        return Commutator(Dagger(self.args[1]), Dagger(self.args[0]))

    def _sympyrepr(self, printer, *args):
        return "%s(%s,%s)" % (
            self.__class__.__name__, printer._print(
                self.args[0]), printer._print(self.args[1])
        )

    def _sympystr(self, printer, *args):
        return "[%s,%s]" % (
            printer._print(self.args[0]), printer._print(self.args[1]))

    def _pretty(self, printer, *args):
        pform = printer._print(self.args[0], *args)
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(printer._print(self.args[1], *args)))
        pform = prettyForm(*pform.parens(left='[', right=']'))
        return pform

    def _latex(self, printer, *args):
        return "\\left[%s,%s\\right]" % tuple([
            printer._print(arg, *args) for arg in self.args])


@Commutator._kind_dispatcher.register(_OperatorKind, _OperatorKind)
def find_op_kind(e1, e2):
    """Find the kind of an anticommutator of two OperatorKinds."""
    return OperatorKind

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\toperators.py ===
from sympy import permutedims
from sympy.core.numbers import Number
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.tensor.tensor import Tensor, TensExpr, TensAdd, TensMul


class PartialDerivative(TensExpr):
    """
    Partial derivative for tensor expressions.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead
    >>> from sympy.tensor.toperators import PartialDerivative
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> A = TensorHead("A", [L])
    >>> B = TensorHead("B", [L])
    >>> i, j, k = symbols("i j k")

    >>> expr = PartialDerivative(A(i), A(j))
    >>> expr
    PartialDerivative(A(i), A(j))

    The ``PartialDerivative`` object behaves like a tensorial expression:

    >>> expr.get_indices()
    [i, -j]

    Notice that the deriving variables have opposite valence than the
    printed one: ``A(j)`` is printed as covariant, but the index of the
    derivative is actually contravariant, i.e. ``-j``.

    Indices can be contracted:

    >>> expr = PartialDerivative(A(i), A(i))
    >>> expr
    PartialDerivative(A(L_0), A(L_0))
    >>> expr.get_indices()
    [L_0, -L_0]

    The method ``.get_indices()`` always returns all indices (even the
    contracted ones). If only uncontracted indices are needed, call
    ``.get_free_indices()``:

    >>> expr.get_free_indices()
    []

    Nested partial derivatives are flattened:

    >>> expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    >>> expr
    PartialDerivative(A(i), A(j), A(k))
    >>> expr.get_indices()
    [i, -j, -k]

    Replace a derivative with array values:

    >>> from sympy.abc import x, y
    >>> from sympy import sin, log
    >>> compA = [sin(x), log(x)*y**3]
    >>> compB = [x, y]
    >>> expr = PartialDerivative(A(i), B(j))
    >>> expr.replace_with_arrays({A(i): compA, B(i): compB})
    [[cos(x), 0], [y**3/x, 3*y**2*log(x)]]

    The returned array is indexed by `(i, -j)`.

    Be careful that other SymPy modules put the indices of the deriving
    variables before the indices of the derivand in the derivative result.
    For example:

    >>> expr.get_free_indices()
    [i, -j]

    >>> from sympy import Matrix, Array
    >>> Matrix(compA).diff(Matrix(compB)).reshape(2, 2)
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    >>> Array(compA).diff(Array(compB))
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]

    These are the transpose of the result of ``PartialDerivative``,
    as the matrix and the array modules put the index `-j` before `i` in the
    derivative result. An array read with index order `(-j, i)` is indeed the
    transpose of the same array read with index order `(i, -j)`. By specifying
    the index order to ``.replace_with_arrays`` one can get a compatible
    expression:

    >>> expr.replace_with_arrays({A(i): compA, B(i): compB}, [-j, i])
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    """

    def __new__(cls, expr, *variables):

        # Flatten:
        if isinstance(expr, PartialDerivative):
            variables = expr.variables + variables
            expr = expr.expr

        args, indices, free, dum = cls._contract_indices_for_derivative(
            S(expr), variables)

        obj = TensExpr.__new__(cls, *args)

        obj._indices = indices
        obj._free = free
        obj._dum = dum
        return obj

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    @classmethod
    def _contract_indices_for_derivative(cls, expr, variables):
        variables_opposite_valence = []

        for i in variables:
            if isinstance(i, Tensor):
                i_free_indices = i.get_free_indices()
                variables_opposite_valence.append(
                        i.xreplace({k: -k for k in i_free_indices}))
            elif isinstance(i, Symbol):
                variables_opposite_valence.append(i)

        args, indices, free, dum = TensMul._tensMul_contract_indices(
            [expr] + variables_opposite_valence, replace_indices=True)

        for i in range(1, len(args)):
            args_i = args[i]
            if isinstance(args_i, Tensor):
                i_indices = args[i].get_free_indices()
                args[i] = args[i].xreplace({k: -k for k in i_indices})

        return args, indices, free, dum

    def doit(self, **hints):
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        obj = self.func(*args)
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        return obj

    def _expand_partial_derivative(self):
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        obj = self.func(*args)
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        result = obj

        if not args[0].free_symbols:
            return S.Zero
        elif isinstance(obj.expr, TensAdd):
            # take care of sums of multi PDs
            result = obj.expr.func(*[
                    self.func(a, *obj.variables)._expand_partial_derivative()
                    for a in result.expr.args])
        elif isinstance(obj.expr, TensMul):
            # take care of products of multi PDs
            if len(obj.variables) == 1:
                # derivative with respect to single variable
                terms = []
                mulargs = list(obj.expr.args)
                for ind in range(len(mulargs)):
                    if not isinstance(sympify(mulargs[ind]), Number):
                        # a number coefficient is not considered for
                        # expansion of PartialDerivative
                        d = self.func(mulargs[ind], *obj.variables)._expand_partial_derivative()
                        terms.append(TensMul(*(mulargs[:ind]
                                               + [d]
                                               + mulargs[(ind + 1):])))
                result = TensAdd.fromiter(terms)
            else:
                # derivative with respect to multiple variables
                # decompose:
                # partial(expr, (u, v))
                # = partial(partial(expr, u).doit(), v).doit()
                result = obj.expr  # init with expr
                for v in obj.variables:
                    result = self.func(result, v)._expand_partial_derivative()
                    # then throw PD on it

        return result

    def _perform_derivative(self):
        result = self.expr
        for v in self.variables:
            if isinstance(result, TensExpr):
                result = result._eval_partial_derivative(v)
            else:
                if v._diff_wrt:
                    result = result._eval_derivative(v)
                else:
                    result = S.Zero
        return result

    def get_indices(self):
        return self._indices

    def get_free_indices(self):
        free = sorted(self._free, key=lambda x: x[1])
        return [i[0] for i in free]

    def _replace_indices(self, repl):
        expr = self.expr.xreplace(repl)
        mirrored = {-k: -v for k, v in repl.items()}
        variables = [i.xreplace(mirrored) for i in self.variables]
        return self.func(expr, *variables)

    @property
    def expr(self):
        return self.args[0]

    @property
    def variables(self):
        return self.args[1:]

    def _extract_data(self, replacement_dict):
        from .array import derive_by_array, tensorcontraction
        indices, array = self.expr._extract_data(replacement_dict)
        for variable in self.variables:
            var_indices, var_array = variable._extract_data(replacement_dict)
            var_indices = [-i for i in var_indices]
            coeff_array, var_array = zip(*[i.as_coeff_Mul() for i in var_array])
            dim_before = len(array.shape)
            array = derive_by_array(array, var_array)
            dim_after = len(array.shape)
            dim_increase = dim_after - dim_before
            array = permutedims(array, [i + dim_increase for i in range(dim_before)] + list(range(dim_increase)))
            array = array.as_mutable()
            varindex = var_indices[0]
            # Remove coefficients of base vector:
            coeff_index = [0] + [slice(None) for i in range(len(indices))]
            for i, coeff in enumerate(coeff_array):
                coeff_index[0] = i
                array[tuple(coeff_index)] /= coeff
            if -varindex in indices:
                pos = indices.index(-varindex)
                array = tensorcontraction(array, (0, pos+1))
                indices.pop(pos)
            else:
                indices.append(varindex)
        return indices, array

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\array_utils.py ===
from ._array_utils_impl import (  # noqa: F401
    __all__,
    __doc__,
    byte_bounds,
    normalize_axis_index,
    normalize_axis_tuple,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\_methods.py ===
"""
Array methods which are called by both the C-code for the method
and the Python code for the NumPy-namespace function

"""
import os
import pickle
import warnings
from contextlib import nullcontext

import numpy as np
from numpy._core import multiarray as mu
from numpy._core import numerictypes as nt
from numpy._core import umath as um
from numpy._core.multiarray import asanyarray
from numpy._globals import _NoValue

# save those O(100) nanoseconds!
bool_dt = mu.dtype("bool")
umr_maximum = um.maximum.reduce
umr_minimum = um.minimum.reduce
umr_sum = um.add.reduce
umr_prod = um.multiply.reduce
umr_bitwise_count = um.bitwise_count
umr_any = um.logical_or.reduce
umr_all = um.logical_and.reduce

# Complex types to -> (2,)float view for fast-path computation in _var()
_complex_to_float = {
    nt.dtype(nt.csingle): nt.dtype(nt.single),
    nt.dtype(nt.cdouble): nt.dtype(nt.double),
}
# Special case for windows: ensure double takes precedence
if nt.dtype(nt.longdouble) != nt.dtype(nt.double):
    _complex_to_float.update({
        nt.dtype(nt.clongdouble): nt.dtype(nt.longdouble),
    })

# avoid keyword arguments to speed up parsing, saves about 15%-20% for very
# small reductions
def _amax(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_maximum(a, axis, None, out, keepdims, initial, where)

def _amin(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_minimum(a, axis, None, out, keepdims, initial, where)

def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
         initial=_NoValue, where=True):
    return umr_sum(a, axis, dtype, out, keepdims, initial, where)

def _prod(a, axis=None, dtype=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_prod(a, axis, dtype, out, keepdims, initial, where)

def _any(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # By default, return a boolean for any and all
    if dtype is None:
        dtype = bool_dt
    # Parsing keyword arguments is currently fairly slow, so avoid it for now
    if where is True:
        return umr_any(a, axis, dtype, out, keepdims)
    return umr_any(a, axis, dtype, out, keepdims, where=where)

def _all(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # By default, return a boolean for any and all
    if dtype is None:
        dtype = bool_dt
    # Parsing keyword arguments is currently fairly slow, so avoid it for now
    if where is True:
        return umr_all(a, axis, dtype, out, keepdims)
    return umr_all(a, axis, dtype, out, keepdims, where=where)

def _count_reduce_items(arr, axis, keepdims=False, where=True):
    # fast-path for the default case
    if where is True:
        # no boolean mask given, calculate items according to axis
        if axis is None:
            axis = tuple(range(arr.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
        items = nt.intp(items)
    else:
        # TODO: Optimize case when `where` is broadcast along a non-reduction
        # axis and full sum is more excessive than needed.

        # guarded to protect circular imports
        from numpy.lib._stride_tricks_impl import broadcast_to
        # count True values in (potentially broadcasted) boolean mask
        items = umr_sum(broadcast_to(where, arr.shape), axis, nt.intp, None,
                        keepdims)
    return items

def _clip(a, min=None, max=None, out=None, **kwargs):
    if a.dtype.kind in "iu":
        # If min/max is a Python integer, deal with out-of-bound values here.
        # (This enforces NEP 50 rules as no value based promotion is done.)
        if type(min) is int and min <= np.iinfo(a.dtype).min:
            min = None
        if type(max) is int and max >= np.iinfo(a.dtype).max:
            max = None

    if min is None and max is None:
        # return identity
        return um.positive(a, out=out, **kwargs)
    elif min is None:
        return um.minimum(a, max, out=out, **kwargs)
    elif max is None:
        return um.maximum(a, min, out=out, **kwargs)
    else:
        return um.clip(a, min, max, out=out, **kwargs)

def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    arr = asanyarray(a)

    is_float16_result = False

    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    if rcount == 0 if where is True else umr_any(rcount == 0, axis=None):
        warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None:
        if issubclass(arr.dtype.type, (nt.integer, nt.bool)):
            dtype = mu.dtype('f8')
        elif issubclass(arr.dtype.type, nt.float16):
            dtype = mu.dtype('f4')
            is_float16_result = True

    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
        if is_float16_result and out is None:
            ret = arr.dtype.type(ret)
    elif hasattr(ret, 'dtype'):
        if is_float16_result:
            ret = arr.dtype.type(ret / rcount)
        else:
            ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount

    return ret

def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True, mean=None):
    arr = asanyarray(a)

    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    # Make this warning show up on top.
    if ddof >= rcount if where is True else umr_any(ddof >= rcount, axis=None):
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                      stacklevel=2)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool)):
        dtype = mu.dtype('f8')

    if mean is not None:
        arrmean = mean
    else:
        # Compute the mean.
        # Note that if dtype is not of inexact type then arraymean will
        # not be either.
        arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where)
        # The shape of rcount has to match arrmean to not change the shape of
        # out in broadcasting. Otherwise, it cannot be stored back to arrmean.
        if rcount.ndim == 0:
            # fast-path for default case when where is True
            div = rcount
        else:
            # matching rcount to arrmean when where is specified as array
            div = rcount.reshape(arrmean.shape)
        if isinstance(arrmean, mu.ndarray):
            arrmean = um.true_divide(arrmean, div, out=arrmean,
                                     casting='unsafe', subok=False)
        elif hasattr(arrmean, "dtype"):
            arrmean = arrmean.dtype.type(arrmean / rcount)
        else:
            arrmean = arrmean / rcount

    # Compute sum of squared deviations from mean
    # Note that x may not be inexact and that we need it to be an array,
    # not a scalar.
    x = asanyarray(arr - arrmean)

    if issubclass(arr.dtype.type, (nt.floating, nt.integer)):
        x = um.multiply(x, x, out=x)
    # Fast-paths for built-in complex types
    elif x.dtype in _complex_to_float:
        xv = x.view(dtype=(_complex_to_float[x.dtype], (2,)))
        um.multiply(xv, xv, out=xv)
        x = um.add(xv[..., 0], xv[..., 1], out=x.real).real
    # Most general case; includes handling object arrays containing imaginary
    # numbers and complex types with non-native byteorder
    else:
        x = um.multiply(x, um.conjugate(x), out=x).real

    ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where)

    # Compute degrees of freedom and make sure it is not negative.
    rcount = um.maximum(rcount - ddof, 0)

    # divide by degrees of freedom
    if isinstance(ret, mu.ndarray):
        ret = um.true_divide(
                ret, rcount, out=ret, casting='unsafe', subok=False)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount

    return ret

def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True, mean=None):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
               keepdims=keepdims, where=where, mean=mean)

    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        ret = um.sqrt(ret)

    return ret

def _ptp(a, axis=None, out=None, keepdims=False):
    return um.subtract(
        umr_maximum(a, axis, None, out, keepdims),
        umr_minimum(a, axis, None, None, keepdims),
        out
    )

def _dump(self, file, protocol=2):
    if hasattr(file, 'write'):
        ctx = nullcontext(file)
    else:
        ctx = open(os.fspath(file), "wb")
    with ctx as f:
        pickle.dump(self, f, protocol=protocol)

def _dumps(self, protocol=2):
    return pickle.dumps(self, protocol=protocol)

def _bitwise_count(a, out=None, *, where=True, casting='same_kind',
          order='K', dtype=None, subok=True):
    return umr_bitwise_count(a, out, where=where, casting=casting,
            order=order, dtype=dtype, subok=subok)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\gbq.py ===
""" Google BigQuery support """
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)
import warnings

from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

    from pandas import DataFrame


def _try_import():
    # since pandas is a dependency of pandas-gbq
    # we need to import on first use
    msg = (
        "pandas-gbq is required to load data from Google BigQuery. "
        "See the docs: https://pandas-gbq.readthedocs.io."
    )
    pandas_gbq = import_optional_dependency("pandas_gbq", extra=msg)
    return pandas_gbq


def read_gbq(
    query: str,
    project_id: str | None = None,
    index_col: str | None = None,
    col_order: list[str] | None = None,
    reauth: bool = False,
    auth_local_webserver: bool = True,
    dialect: str | None = None,
    location: str | None = None,
    configuration: dict[str, Any] | None = None,
    credentials: Credentials | None = None,
    use_bqstorage_api: bool | None = None,
    max_results: int | None = None,
    progress_bar_type: str | None = None,
) -> DataFrame:
    """
    Load data from Google BigQuery.

    .. deprecated:: 2.2.0

       Please use ``pandas_gbq.read_gbq`` instead.

    This function requires the `pandas-gbq package
    <https://pandas-gbq.readthedocs.io>`__.

    See the `How to authenticate with Google BigQuery
    <https://pandas-gbq.readthedocs.io/en/latest/howto/authentication.html>`__
    guide for authentication instructions.

    Parameters
    ----------
    query : str
        SQL-Like Query to return data values.
    project_id : str, optional
        Google BigQuery Account project ID. Optional when available from
        the environment.
    index_col : str, optional
        Name of result column to use for index in results DataFrame.
    col_order : list(str), optional
        List of BigQuery column names in the desired order for results
        DataFrame.
    reauth : bool, default False
        Force Google BigQuery to re-authenticate the user. This is useful
        if multiple accounts are used.
    auth_local_webserver : bool, default True
        Use the `local webserver flow`_ instead of the `console flow`_
        when getting user credentials.

        .. _local webserver flow:
            https://google-auth-oauthlib.readthedocs.io/en/latest/reference/google_auth_oauthlib.flow.html#google_auth_oauthlib.flow.InstalledAppFlow.run_local_server
        .. _console flow:
            https://google-auth-oauthlib.readthedocs.io/en/latest/reference/google_auth_oauthlib.flow.html#google_auth_oauthlib.flow.InstalledAppFlow.run_console

        *New in version 0.2.0 of pandas-gbq*.

        .. versionchanged:: 1.5.0
           Default value is changed to ``True``. Google has deprecated the
           ``auth_local_webserver = False`` `"out of band" (copy-paste)
           flow
           <https://developers.googleblog.com/2022/02/making-oauth-flows-safer.html?m=1#disallowed-oob>`_.
    dialect : str, default 'legacy'
        Note: The default value is changing to 'standard' in a future version.

        SQL syntax dialect to use. Value can be one of:

        ``'legacy'``
            Use BigQuery's legacy SQL dialect. For more information see
            `BigQuery Legacy SQL Reference
            <https://cloud.google.com/bigquery/docs/reference/legacy-sql>`__.
        ``'standard'``
            Use BigQuery's standard SQL, which is
            compliant with the SQL 2011 standard. For more information
            see `BigQuery Standard SQL Reference
            <https://cloud.google.com/bigquery/docs/reference/standard-sql/>`__.
    location : str, optional
        Location where the query job should run. See the `BigQuery locations
        documentation
        <https://cloud.google.com/bigquery/docs/dataset-locations>`__ for a
        list of available locations. The location must match that of any
        datasets used in the query.

        *New in version 0.5.0 of pandas-gbq*.
    configuration : dict, optional
        Query config parameters for job processing.
        For example:

            configuration = {'query': {'useQueryCache': False}}

        For more information see `BigQuery REST API Reference
        <https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.query>`__.
    credentials : google.auth.credentials.Credentials, optional
        Credentials for accessing Google APIs. Use this parameter to override
        default credentials, such as to use Compute Engine
        :class:`google.auth.compute_engine.Credentials` or Service Account
        :class:`google.oauth2.service_account.Credentials` directly.

        *New in version 0.8.0 of pandas-gbq*.
    use_bqstorage_api : bool, default False
        Use the `BigQuery Storage API
        <https://cloud.google.com/bigquery/docs/reference/storage/>`__ to
        download query results quickly, but at an increased cost. To use this
        API, first `enable it in the Cloud Console
        <https://console.cloud.google.com/apis/library/bigquerystorage.googleapis.com>`__.
        You must also have the `bigquery.readsessions.create
        <https://cloud.google.com/bigquery/docs/access-control#roles>`__
        permission on the project you are billing queries to.

        This feature requires version 0.10.0 or later of the ``pandas-gbq``
        package. It also requires the ``google-cloud-bigquery-storage`` and
        ``fastavro`` packages.

    max_results : int, optional
        If set, limit the maximum number of rows to fetch from the query
        results.

    progress_bar_type : Optional, str
        If set, use the `tqdm <https://tqdm.github.io/>`__ library to
        display a progress bar while the data downloads. Install the
        ``tqdm`` package to use this feature.

        Possible values of ``progress_bar_type`` include:

        ``None``
            No progress bar.
        ``'tqdm'``
            Use the :func:`tqdm.tqdm` function to print a progress bar
            to :data:`sys.stderr`.
        ``'tqdm_notebook'``
            Use the :func:`tqdm.tqdm_notebook` function to display a
            progress bar as a Jupyter notebook widget.
        ``'tqdm_gui'``
            Use the :func:`tqdm.tqdm_gui` function to display a
            progress bar as a graphical dialog box.

    Returns
    -------
    df: DataFrame
        DataFrame representing results of query.

    See Also
    --------
    pandas_gbq.read_gbq : This function in the pandas-gbq library.
    DataFrame.to_gbq : Write a DataFrame to Google BigQuery.

    Examples
    --------
    Example taken from `Google BigQuery documentation
    <https://cloud.google.com/bigquery/docs/pandas-gbq-migration>`_

    >>> sql = "SELECT name FROM table_name WHERE state = 'TX' LIMIT 100;"
    >>> df = pd.read_gbq(sql, dialect="standard")  # doctest: +SKIP
    >>> project_id = "your-project-id"  # doctest: +SKIP
    >>> df = pd.read_gbq(sql,
    ...                  project_id=project_id,
    ...                  dialect="standard"
    ...                  )  # doctest: +SKIP
    """
    warnings.warn(
        "read_gbq is deprecated and will be removed in a future version. "
        "Please use pandas_gbq.read_gbq instead: "
        "https://pandas-gbq.readthedocs.io/en/latest/api.html#pandas_gbq.read_gbq",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    pandas_gbq = _try_import()

    kwargs: dict[str, str | bool | int | None] = {}

    # START: new kwargs.  Don't populate unless explicitly set.
    if use_bqstorage_api is not None:
        kwargs["use_bqstorage_api"] = use_bqstorage_api
    if max_results is not None:
        kwargs["max_results"] = max_results

    kwargs["progress_bar_type"] = progress_bar_type
    # END: new kwargs

    return pandas_gbq.read_gbq(
        query,
        project_id=project_id,
        index_col=index_col,
        col_order=col_order,
        reauth=reauth,
        auth_local_webserver=auth_local_webserver,
        dialect=dialect,
        location=location,
        configuration=configuration,
        credentials=credentials,
        **kwargs,
    )


def to_gbq(
    dataframe: DataFrame,
    destination_table: str,
    project_id: str | None = None,
    chunksize: int | None = None,
    reauth: bool = False,
    if_exists: str = "fail",
    auth_local_webserver: bool = True,
    table_schema: list[dict[str, str]] | None = None,
    location: str | None = None,
    progress_bar: bool = True,
    credentials: Credentials | None = None,
) -> None:
    warnings.warn(
        "to_gbq is deprecated and will be removed in a future version. "
        "Please use pandas_gbq.to_gbq instead: "
        "https://pandas-gbq.readthedocs.io/en/latest/api.html#pandas_gbq.to_gbq",
        FutureWarning,
        stacklevel=find_stack_level(),
    )
    pandas_gbq = _try_import()
    pandas_gbq.to_gbq(
        dataframe,
        destination_table,
        project_id=project_id,
        chunksize=chunksize,
        reauth=reauth,
        if_exists=if_exists,
        auth_local_webserver=auth_local_webserver,
        table_schema=table_schema,
        location=location,
        progress_bar=progress_bar,
        credentials=credentials,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\contrib\emscripten\connection.py ===
from __future__ import annotations

import os
import typing

# use http.client.HTTPException for consistency with non-emscripten
from http.client import HTTPException as HTTPException  # noqa: F401
from http.client import ResponseNotReady

from ..._base_connection import _TYPE_BODY
from ...connection import HTTPConnection, ProxyConfig, port_by_scheme
from ...exceptions import TimeoutError
from ...response import BaseHTTPResponse
from ...util.connection import _TYPE_SOCKET_OPTIONS
from ...util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT
from ...util.url import Url
from .fetch import _RequestError, _TimeoutError, send_request, send_streaming_request
from .request import EmscriptenRequest
from .response import EmscriptenHttpResponseWrapper, EmscriptenResponse

if typing.TYPE_CHECKING:
    from ..._base_connection import BaseHTTPConnection, BaseHTTPSConnection


class EmscriptenHTTPConnection:
    default_port: typing.ClassVar[int] = port_by_scheme["http"]
    default_socket_options: typing.ClassVar[_TYPE_SOCKET_OPTIONS]

    timeout: None | (float)

    host: str
    port: int
    blocksize: int
    source_address: tuple[str, int] | None
    socket_options: _TYPE_SOCKET_OPTIONS | None

    proxy: Url | None
    proxy_config: ProxyConfig | None

    is_verified: bool = False
    proxy_is_verified: bool | None = None

    _response: EmscriptenResponse | None

    def __init__(
        self,
        host: str,
        port: int = 0,
        *,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 8192,
        socket_options: _TYPE_SOCKET_OPTIONS | None = None,
        proxy: Url | None = None,
        proxy_config: ProxyConfig | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout if isinstance(timeout, float) else 0.0
        self.scheme = "http"
        self._closed = True
        self._response = None
        # ignore these things because we don't
        # have control over that stuff
        self.proxy = None
        self.proxy_config = None
        self.blocksize = blocksize
        self.source_address = None
        self.socket_options = None
        self.is_verified = False

    def set_tunnel(
        self,
        host: str,
        port: int | None = 0,
        headers: typing.Mapping[str, str] | None = None,
        scheme: str = "http",
    ) -> None:
        pass

    def connect(self) -> None:
        pass

    def request(
        self,
        method: str,
        url: str,
        body: _TYPE_BODY | None = None,
        headers: typing.Mapping[str, str] | None = None,
        # We know *at least* botocore is depending on the order of the
        # first 3 parameters so to be safe we only mark the later ones
        # as keyword-only to ensure we have space to extend.
        *,
        chunked: bool = False,
        preload_content: bool = True,
        decode_content: bool = True,
        enforce_content_length: bool = True,
    ) -> None:
        self._closed = False
        if url.startswith("/"):
            # no scheme / host / port included, make a full url
            url = f"{self.scheme}://{self.host}:{self.port}" + url
        request = EmscriptenRequest(
            url=url,
            method=method,
            timeout=self.timeout if self.timeout else 0,
            decode_content=decode_content,
        )
        request.set_body(body)
        if headers:
            for k, v in headers.items():
                request.set_header(k, v)
        self._response = None
        try:
            if not preload_content:
                self._response = send_streaming_request(request)
            if self._response is None:
                self._response = send_request(request)
        except _TimeoutError as e:
            raise TimeoutError(e.message) from e
        except _RequestError as e:
            raise HTTPException(e.message) from e

    def getresponse(self) -> BaseHTTPResponse:
        if self._response is not None:
            return EmscriptenHttpResponseWrapper(
                internal_response=self._response,
                url=self._response.request.url,
                connection=self,
            )
        else:
            raise ResponseNotReady()

    def close(self) -> None:
        self._closed = True
        self._response = None

    @property
    def is_closed(self) -> bool:
        """Whether the connection either is brand new or has been previously closed.
        If this property is True then both ``is_connected`` and ``has_connected_to_proxy``
        properties must be False.
        """
        return self._closed

    @property
    def is_connected(self) -> bool:
        """Whether the connection is actively connected to any origin (proxy or target)"""
        return True

    @property
    def has_connected_to_proxy(self) -> bool:
        """Whether the connection has successfully connected to its proxy.
        This returns False if no proxy is in use. Used to determine whether
        errors are coming from the proxy layer or from tunnelling to the target origin.
        """
        return False


class EmscriptenHTTPSConnection(EmscriptenHTTPConnection):
    default_port = port_by_scheme["https"]
    # all this is basically ignored, as browser handles https
    cert_reqs: int | str | None = None
    ca_certs: str | None = None
    ca_cert_dir: str | None = None
    ca_cert_data: None | str | bytes = None
    cert_file: str | None
    key_file: str | None
    key_password: str | None
    ssl_context: typing.Any | None
    ssl_version: int | str | None = None
    ssl_minimum_version: int | None = None
    ssl_maximum_version: int | None = None
    assert_hostname: None | str | typing.Literal[False]
    assert_fingerprint: str | None = None

    def __init__(
        self,
        host: str,
        port: int = 0,
        *,
        timeout: _TYPE_TIMEOUT = _DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 16384,
        socket_options: (
            None | _TYPE_SOCKET_OPTIONS
        ) = HTTPConnection.default_socket_options,
        proxy: Url | None = None,
        proxy_config: ProxyConfig | None = None,
        cert_reqs: int | str | None = None,
        assert_hostname: None | str | typing.Literal[False] = None,
        assert_fingerprint: str | None = None,
        server_hostname: str | None = None,
        ssl_context: typing.Any | None = None,
        ca_certs: str | None = None,
        ca_cert_dir: str | None = None,
        ca_cert_data: None | str | bytes = None,
        ssl_minimum_version: int | None = None,
        ssl_maximum_version: int | None = None,
        ssl_version: int | str | None = None,  # Deprecated
        cert_file: str | None = None,
        key_file: str | None = None,
        key_password: str | None = None,
    ) -> None:
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            source_address=source_address,
            blocksize=blocksize,
            socket_options=socket_options,
            proxy=proxy,
            proxy_config=proxy_config,
        )
        self.scheme = "https"

        self.key_file = key_file
        self.cert_file = cert_file
        self.key_password = key_password
        self.ssl_context = ssl_context
        self.server_hostname = server_hostname
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint
        self.ssl_version = ssl_version
        self.ssl_minimum_version = ssl_minimum_version
        self.ssl_maximum_version = ssl_maximum_version
        self.ca_certs = ca_certs and os.path.expanduser(ca_certs)
        self.ca_cert_dir = ca_cert_dir and os.path.expanduser(ca_cert_dir)
        self.ca_cert_data = ca_cert_data

        self.cert_reqs = None

        # The browser will automatically verify all requests.
        # We have no control over that setting.
        self.is_verified = True

    def set_cert(
        self,
        key_file: str | None = None,
        cert_file: str | None = None,
        cert_reqs: int | str | None = None,
        key_password: str | None = None,
        ca_certs: str | None = None,
        assert_hostname: None | str | typing.Literal[False] = None,
        assert_fingerprint: str | None = None,
        ca_cert_dir: str | None = None,
        ca_cert_data: None | str | bytes = None,
    ) -> None:
        pass


# verify that this class implements BaseHTTP(s) connection correctly
if typing.TYPE_CHECKING:
    _supports_http_protocol: BaseHTTPConnection = EmscriptenHTTPConnection("", 0)
    _supports_https_protocol: BaseHTTPSConnection = EmscriptenHTTPSConnection("", 0)