
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\terminal_theme.py ===
from typing import List, Optional, Tuple

from .color_triplet import ColorTriplet
from .palette import Palette

_ColorTuple = Tuple[int, int, int]


class TerminalTheme:
    """A color theme used when exporting console content.

    Args:
        background (Tuple[int, int, int]): The background color.
        foreground (Tuple[int, int, int]): The foreground (text) color.
        normal (List[Tuple[int, int, int]]): A list of 8 normal intensity colors.
        bright (List[Tuple[int, int, int]], optional): A list of 8 bright colors, or None
            to repeat normal intensity. Defaults to None.
    """

    def __init__(
        self,
        background: _ColorTuple,
        foreground: _ColorTuple,
        normal: List[_ColorTuple],
        bright: Optional[List[_ColorTuple]] = None,
    ) -> None:
        self.background_color = ColorTriplet(*background)
        self.foreground_color = ColorTriplet(*foreground)
        self.ansi_colors = Palette(normal + (bright or normal))


DEFAULT_TERMINAL_THEME = TerminalTheme(
    (255, 255, 255),
    (0, 0, 0),
    [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (192, 192, 192),
    ],
    [
        (128, 128, 128),
        (255, 0, 0),
        (0, 255, 0),
        (255, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ],
)

MONOKAI = TerminalTheme(
    (12, 12, 12),
    (217, 217, 217),
    [
        (26, 26, 26),
        (244, 0, 95),
        (152, 224, 36),
        (253, 151, 31),
        (157, 101, 255),
        (244, 0, 95),
        (88, 209, 235),
        (196, 197, 181),
        (98, 94, 76),
    ],
    [
        (244, 0, 95),
        (152, 224, 36),
        (224, 213, 97),
        (157, 101, 255),
        (244, 0, 95),
        (88, 209, 235),
        (246, 246, 239),
    ],
)
DIMMED_MONOKAI = TerminalTheme(
    (25, 25, 25),
    (185, 188, 186),
    [
        (58, 61, 67),
        (190, 63, 72),
        (135, 154, 59),
        (197, 166, 53),
        (79, 118, 161),
        (133, 92, 141),
        (87, 143, 164),
        (185, 188, 186),
        (136, 137, 135),
    ],
    [
        (251, 0, 31),
        (15, 114, 47),
        (196, 112, 51),
        (24, 109, 227),
        (251, 0, 103),
        (46, 112, 109),
        (253, 255, 185),
    ],
)
NIGHT_OWLISH = TerminalTheme(
    (255, 255, 255),
    (64, 63, 83),
    [
        (1, 22, 39),
        (211, 66, 62),
        (42, 162, 152),
        (218, 170, 1),
        (72, 118, 214),
        (64, 63, 83),
        (8, 145, 106),
        (122, 129, 129),
        (122, 129, 129),
    ],
    [
        (247, 110, 110),
        (73, 208, 197),
        (218, 194, 107),
        (92, 167, 228),
        (105, 112, 152),
        (0, 201, 144),
        (152, 159, 177),
    ],
)

SVG_EXPORT_THEME = TerminalTheme(
    (41, 41, 41),
    (197, 200, 198),
    [
        (75, 78, 85),
        (204, 85, 90),
        (152, 168, 75),
        (208, 179, 68),
        (96, 138, 177),
        (152, 114, 159),
        (104, 160, 179),
        (197, 200, 198),
        (154, 155, 153),
    ],
    [
        (255, 38, 39),
        (0, 130, 61),
        (208, 132, 66),
        (25, 132, 233),
        (255, 44, 122),
        (57, 130, 128),
        (253, 253, 197),
    ],
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_deprecations.py ===
# testing/suite/test_deprecations.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from .. import fixtures
from ..assertions import eq_
from ..schema import Column
from ..schema import Table
from ... import Integer
from ... import select
from ... import testing
from ... import union


class DeprecatedCompoundSelectTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("x", Integer),
            Column("y", Integer),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "x": 1, "y": 2},
                {"id": 2, "x": 2, "y": 3},
                {"id": 3, "x": 3, "y": 4},
                {"id": 4, "x": 4, "y": 5},
            ],
        )

    def _assert_result(self, conn, select, result, params=()):
        eq_(conn.execute(select, params).fetchall(), result)

    def test_plain_union(self, connection):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2)
        s2 = select(table).where(table.c.id == 3)

        u1 = union(s1, s2)
        with testing.expect_deprecated(
            "The SelectBase.c and SelectBase.columns "
            "attributes are deprecated"
        ):
            self._assert_result(
                connection, u1.order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)]
            )

    # note we've had to remove one use case entirely, which is this
    # one.   the Select gets its FROMS from the WHERE clause and the
    # columns clause, but not the ORDER BY, which means the old ".c" system
    # allowed you to "order_by(s.c.foo)" to get an unnamed column in the
    # ORDER BY without adding the SELECT into the FROM and breaking the
    # query.  Users will have to adjust for this use case if they were doing
    # it before.
    def _dont_test_select_from_plain_union(self, connection):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2)
        s2 = select(table).where(table.c.id == 3)

        u1 = union(s1, s2).alias().select()
        with testing.expect_deprecated(
            "The SelectBase.c and SelectBase.columns "
            "attributes are deprecated"
        ):
            self._assert_result(
                connection, u1.order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)]
            )

    @testing.requires.order_by_col_from_union
    @testing.requires.parens_in_union_contained_select_w_limit_offset
    def test_limit_offset_selectable_in_unions(self, connection):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).limit(1).order_by(table.c.id)
        s2 = select(table).where(table.c.id == 3).limit(1).order_by(table.c.id)

        u1 = union(s1, s2).limit(2)
        with testing.expect_deprecated(
            "The SelectBase.c and SelectBase.columns "
            "attributes are deprecated"
        ):
            self._assert_result(
                connection, u1.order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)]
            )

    @testing.requires.parens_in_union_contained_select_wo_limit_offset
    def test_order_by_selectable_in_unions(self, connection):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).order_by(table.c.id)
        s2 = select(table).where(table.c.id == 3).order_by(table.c.id)

        u1 = union(s1, s2).limit(2)
        with testing.expect_deprecated(
            "The SelectBase.c and SelectBase.columns "
            "attributes are deprecated"
        ):
            self._assert_result(
                connection, u1.order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)]
            )

    def test_distinct_selectable_in_unions(self, connection):
        table = self.tables.some_table
        s1 = select(table).where(table.c.id == 2).distinct()
        s2 = select(table).where(table.c.id == 3).distinct()

        u1 = union(s1, s2).limit(2)
        with testing.expect_deprecated(
            "The SelectBase.c and SelectBase.columns "
            "attributes are deprecated"
        ):
            self._assert_result(
                connection, u1.order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)]
            )

    def test_limit_offset_aliased_selectable_in_unions(self, connection):
        table = self.tables.some_table
        s1 = (
            select(table)
            .where(table.c.id == 2)
            .limit(1)
            .order_by(table.c.id)
            .alias()
            .select()
        )
        s2 = (
            select(table)
            .where(table.c.id == 3)
            .limit(1)
            .order_by(table.c.id)
            .alias()
            .select()
        )

        u1 = union(s1, s2).limit(2)
        with testing.expect_deprecated(
            "The SelectBase.c and SelectBase.columns "
            "attributes are deprecated"
        ):
            self._assert_result(
                connection, u1.order_by(u1.c.id), [(2, 2, 3), (3, 3, 4)]
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\fglmtools.py ===
"""Implementation of matrix FGLM Groebner basis conversion algorithm. """


from sympy.polys.monomials import monomial_mul, monomial_div

def matrix_fglm(F, ring, O_to):
    """
    Converts the reduced Groebner basis ``F`` of a zero-dimensional
    ideal w.r.t. ``O_from`` to a reduced Groebner basis
    w.r.t. ``O_to``.

    References
    ==========

    .. [1] J.C. Faugere, P. Gianni, D. Lazard, T. Mora (1994). Efficient
           Computation of Zero-dimensional Groebner Bases by Change of
           Ordering
    """
    domain = ring.domain
    ngens = ring.ngens

    ring_to = ring.clone(order=O_to)

    old_basis = _basis(F, ring)
    M = _representing_matrices(old_basis, F, ring)

    # V contains the normalforms (wrt O_from) of S
    S = [ring.zero_monom]
    V = [[domain.one] + [domain.zero] * (len(old_basis) - 1)]
    G = []

    L = [(i, 0) for i in range(ngens)]  # (i, j) corresponds to x_i * S[j]
    L.sort(key=lambda k_l: O_to(_incr_k(S[k_l[1]], k_l[0])), reverse=True)
    t = L.pop()

    P = _identity_matrix(len(old_basis), domain)

    while True:
        s = len(S)
        v = _matrix_mul(M[t[0]], V[t[1]])
        _lambda = _matrix_mul(P, v)

        if all(_lambda[i] == domain.zero for i in range(s, len(old_basis))):
            # there is a linear combination of v by V
            lt = ring.term_new(_incr_k(S[t[1]], t[0]), domain.one)
            rest = ring.from_dict({S[i]: _lambda[i] for i in range(s)})

            g = (lt - rest).set_ring(ring_to)
            if g:
                G.append(g)
        else:
            # v is linearly independent from V
            P = _update(s, _lambda, P)
            S.append(_incr_k(S[t[1]], t[0]))
            V.append(v)

            L.extend([(i, s) for i in range(ngens)])
            L = list(set(L))
            L.sort(key=lambda k_l: O_to(_incr_k(S[k_l[1]], k_l[0])), reverse=True)

        L = [(k, l) for (k, l) in L if all(monomial_div(_incr_k(S[l], k), g.LM) is None for g in G)]

        if not L:
            G = [ g.monic() for g in G ]
            return sorted(G, key=lambda g: O_to(g.LM), reverse=True)

        t = L.pop()


def _incr_k(m, k):
    return tuple(list(m[:k]) + [m[k] + 1] + list(m[k + 1:]))


def _identity_matrix(n, domain):
    M = [[domain.zero]*n for _ in range(n)]

    for i in range(n):
        M[i][i] = domain.one

    return M


def _matrix_mul(M, v):
    return [sum(row[i] * v[i] for i in range(len(v))) for row in M]


def _update(s, _lambda, P):
    """
    Update ``P`` such that for the updated `P'` `P' v = e_{s}`.
    """
    k = min(j for j in range(s, len(_lambda)) if _lambda[j] != 0)

    for r in range(len(_lambda)):
        if r != k:
            P[r] = [P[r][j] - (P[k][j] * _lambda[r]) / _lambda[k] for j in range(len(P[r]))]

    P[k] = [P[k][j] / _lambda[k] for j in range(len(P[k]))]
    P[k], P[s] = P[s], P[k]

    return P


def _representing_matrices(basis, G, ring):
    r"""
    Compute the matrices corresponding to the linear maps `m \mapsto
    x_i m` for all variables `x_i`.
    """
    domain = ring.domain
    u = ring.ngens-1

    def var(i):
        return tuple([0] * i + [1] + [0] * (u - i))

    def representing_matrix(m):
        M = [[domain.zero] * len(basis) for _ in range(len(basis))]

        for i, v in enumerate(basis):
            r = ring.term_new(monomial_mul(m, v), domain.one).rem(G)

            for monom, coeff in r.terms():
                j = basis.index(monom)
                M[j][i] = coeff

        return M

    return [representing_matrix(var(i)) for i in range(u + 1)]


def _basis(G, ring):
    r"""
    Computes a list of monomials which are not divisible by the leading
    monomials wrt to ``O`` of ``G``. These monomials are a basis of
    `K[X_1, \ldots, X_n]/(G)`.
    """
    order = ring.order

    leading_monomials = [g.LM for g in G]
    candidates = [ring.zero_monom]
    basis = []

    while candidates:
        t = candidates.pop()
        basis.append(t)

        new_candidates = [_incr_k(t, k) for k in range(ring.ngens)
            if all(monomial_div(_incr_k(t, k), lmg) is None
            for lmg in leading_monomials)]
        candidates.extend(new_candidates)
        candidates.sort(key=order, reverse=True)

    basis = list(set(basis))

    return sorted(basis, key=order)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_generated_io_kqueue.py ===
# ***********************************************************
# ******* WARNING: AUTOGENERATED! ALL EDITS WILL BE LOST ******
# *************************************************************
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ._ki import enable_ki_protection
from ._run import GLOBAL_RUN_CONTEXT

if TYPE_CHECKING:
    import select
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from .. import _core
    from .._file_io import _HasFileNo
    from ._traps import Abort, RaiseCancelT

assert not TYPE_CHECKING or sys.platform == "darwin"


__all__ = [
    "current_kqueue",
    "monitor_kevent",
    "notify_closing",
    "wait_kevent",
    "wait_readable",
    "wait_writable",
]


@enable_ki_protection
def current_kqueue() -> select.kqueue:
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__.
    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.current_kqueue()
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def monitor_kevent(
    ident: int, filter: int
) -> AbstractContextManager[_core.UnboundedQueue[select.kevent]]:
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__.
    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.monitor_kevent(ident, filter)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def wait_kevent(
    ident: int, filter: int, abort_func: Callable[[RaiseCancelT], Abort]
) -> Abort:
    """TODO: these are implemented, but are currently more of a sketch than
    anything real. See `#26
    <https://github.com/python-trio/trio/issues/26>`__.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_kevent(
            ident, filter, abort_func
        )
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def wait_readable(fd: int | _HasFileNo) -> None:
    """Block until the kernel reports that the given object is readable.

    On Unix systems, ``fd`` must either be an integer file descriptor,
    or else an object with a ``.fileno()`` method which returns an
    integer file descriptor. Any kind of file descriptor can be passed,
    though the exact semantics will depend on your kernel. For example,
    this probably won't do anything useful for on-disk files.

    On Windows systems, ``fd`` must either be an integer ``SOCKET``
    handle, or else an object with a ``.fileno()`` method which returns
    an integer ``SOCKET`` handle. File descriptors aren't supported,
    and neither are handles that refer to anything besides a
    ``SOCKET``.

    :raises trio.BusyResourceError:
        if another task is already waiting for the given socket to
        become readable.
    :raises trio.ClosedResourceError:
        if another task calls :func:`notify_closing` while this
        function is still working.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_readable(fd)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
async def wait_writable(fd: int | _HasFileNo) -> None:
    """Block until the kernel reports that the given object is writable.

    See `wait_readable` for the definition of ``fd``.

    :raises trio.BusyResourceError:
        if another task is already waiting for the given socket to
        become writable.
    :raises trio.ClosedResourceError:
        if another task calls :func:`notify_closing` while this
        function is still working.
    """
    try:
        return await GLOBAL_RUN_CONTEXT.runner.io_manager.wait_writable(fd)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


@enable_ki_protection
def notify_closing(fd: int | _HasFileNo) -> None:
    """Notify waiters of the given object that it will be closed.

    Call this before closing a file descriptor (on Unix) or socket (on
    Windows). This will cause any `wait_readable` or `wait_writable`
    calls on the given object to immediately wake up and raise
    `~trio.ClosedResourceError`.

    This doesn't actually close the object – you still have to do that
    yourself afterwards. Also, you want to be careful to make sure no
    new tasks start waiting on the object in between when you call this
    and when it's actually closed. So to close something properly, you
    usually want to do these steps in order:

    1. Explicitly mark the object as closed, so that any new attempts
       to use it will abort before they start.
    2. Call `notify_closing` to wake up any already-existing users.
    3. Actually close the object.

    It's also possible to do them in a different order if that's more
    convenient, *but only if* you make sure not to have any checkpoints in
    between the steps. This way they all happen in a single atomic
    step, so other tasks won't be able to tell what order they happened
    in anyway.
    """
    try:
        return GLOBAL_RUN_CONTEXT.runner.io_manager.notify_closing(fd)
    except AttributeError:
        raise RuntimeError("must be called from async context") from None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\tests\test_umath.py ===
import fnmatch
import itertools
import operator
import platform
import sys
import warnings
from collections import namedtuple
from fractions import Fraction
from functools import reduce

import pytest

import numpy as np
import numpy._core.umath as ncu
from numpy._core import _umath_tests as ncu_tests
from numpy._core import sctypes
from numpy.testing import (
    HAS_REFCOUNT,
    IS_MUSL,
    IS_PYPY,
    IS_WASM,
    _gen_alignment_data,
    assert_,
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_almost_equal_nulp,
    assert_array_equal,
    assert_array_max_ulp,
    assert_equal,
    assert_no_warnings,
    assert_raises,
    assert_raises_regex,
    suppress_warnings,
)
from numpy.testing._private.utils import _glibc_older_than

UFUNCS = [obj for obj in np._core.umath.__dict__.values()
         if isinstance(obj, np.ufunc)]

UFUNCS_UNARY = [
    uf for uf in UFUNCS if uf.nin == 1
]
UFUNCS_UNARY_FP = [
    uf for uf in UFUNCS_UNARY if 'f->f' in uf.types
]

UFUNCS_BINARY = [
    uf for uf in UFUNCS if uf.nin == 2
]
UFUNCS_BINARY_ACC = [
    uf for uf in UFUNCS_BINARY if hasattr(uf, "accumulate") and uf.nout == 1
]

def interesting_binop_operands(val1, val2, dtype):
    """
    Helper to create "interesting" operands to cover common code paths:
    * scalar inputs
    * only first "values" is an array (e.g. scalar division fast-paths)
    * Longer array (SIMD) placing the value of interest at different positions
    * Oddly strided arrays which may not be SIMD compatible

    It does not attempt to cover unaligned access or mixed dtypes.
    These are normally handled by the casting/buffering machinery.

    This is not a fixture (currently), since I believe a fixture normally
    only yields once?
    """
    fill_value = 1  # could be a parameter, but maybe not an optional one?

    arr1 = np.full(10003, dtype=dtype, fill_value=fill_value)
    arr2 = np.full(10003, dtype=dtype, fill_value=fill_value)

    arr1[0] = val1
    arr2[0] = val2

    extractor = lambda res: res
    yield arr1[0], arr2[0], extractor, "scalars"

    extractor = lambda res: res
    yield arr1[0, ...], arr2[0, ...], extractor, "scalar-arrays"

    # reset array values to fill_value:
    arr1[0] = fill_value
    arr2[0] = fill_value

    for pos in [0, 1, 2, 3, 4, 5, -1, -2, -3, -4]:
        arr1[pos] = val1
        arr2[pos] = val2

        extractor = lambda res: res[pos]
        yield arr1, arr2, extractor, f"off-{pos}"
        yield arr1, arr2[pos], extractor, f"off-{pos}-with-scalar"

        arr1[pos] = fill_value
        arr2[pos] = fill_value

    for stride in [-1, 113]:
        op1 = arr1[::stride]
        op2 = arr2[::stride]
        op1[10] = val1
        op2[10] = val2

        extractor = lambda res: res[10]
        yield op1, op2, extractor, f"stride-{stride}"

        op1[10] = fill_value
        op2[10] = fill_value


def on_powerpc():
    """ True if we are running on a Power PC platform."""
    return platform.processor() == 'powerpc' or \
           platform.machine().startswith('ppc')


def bad_arcsinh():
    """The blocklisted trig functions are not accurate on aarch64/PPC for
    complex256. Rather than dig through the actual problem skip the
    test. This should be fixed when we can move past glibc2.17
    which is the version in manylinux2014
    """
    if platform.machine() == 'aarch64':
        x = 1.78e-10
    elif on_powerpc():
        x = 2.16e-10
    else:
        return False
    v1 = np.arcsinh(np.float128(x))
    v2 = np.arcsinh(np.complex256(x)).real
    # The eps for float128 is 1-e33, so this is way bigger
    return abs((v1 / v2) - 1.0) > 1e-23


class _FilterInvalids:
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)


class TestConstants:
    def test_pi(self):
        assert_allclose(ncu.pi, 3.141592653589793, 1e-15)

    def test_e(self):
        assert_allclose(ncu.e, 2.718281828459045, 1e-15)

    def test_euler_gamma(self):
        assert_allclose(ncu.euler_gamma, 0.5772156649015329, 1e-15)


class TestOut:
    def test_out_subok(self):
        for subok in (True, False):
            a = np.array(0.5)
            o = np.empty(())

            r = np.add(a, 2, o, subok=subok)
            assert_(r is o)
            r = np.add(a, 2, out=o, subok=subok)
            assert_(r is o)
            r = np.add(a, 2, out=(o,), subok=subok)
            assert_(r is o)

            d = np.array(5.7)
            o1 = np.empty(())
            o2 = np.empty((), dtype=np.int32)

            r1, r2 = np.frexp(d, o1, None, subok=subok)
            assert_(r1 is o1)
            r1, r2 = np.frexp(d, None, o2, subok=subok)
            assert_(r2 is o2)
            r1, r2 = np.frexp(d, o1, o2, subok=subok)
            assert_(r1 is o1)
            assert_(r2 is o2)

            r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
            assert_(r1 is o1)
            r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
            assert_(r2 is o2)
            r1, r2 = np.frexp(d, out=(o1, o2), subok=subok)
            assert_(r1 is o1)
            assert_(r2 is o2)

            with assert_raises(TypeError):
                # Out argument must be tuple, since there are multiple outputs.
                r1, r2 = np.frexp(d, out=o1, subok=subok)

            assert_raises(TypeError, np.add, a, 2, o, o, subok=subok)
            assert_raises(TypeError, np.add, a, 2, o, out=o, subok=subok)
            assert_raises(TypeError, np.add, a, 2, None, out=o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(o, o), subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(), subok=subok)
            assert_raises(TypeError, np.add, a, 2, [], subok=subok)
            assert_raises(TypeError, np.add, a, 2, out=[], subok=subok)
            assert_raises(TypeError, np.add, a, 2, out=([],), subok=subok)
            o.flags.writeable = False
            assert_raises(ValueError, np.add, a, 2, o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(o,), subok=subok)

    def test_out_wrap_subok(self):
        class ArrayWrap(np.ndarray):
            __array_priority__ = 10

            def __new__(cls, arr):
                return np.asarray(arr).view(cls).copy()

            def __array_wrap__(self, arr, context=None, return_scalar=False):
                return arr.view(type(self))

        for subok in (True, False):
            a = ArrayWrap([0.5])

            r = np.add(a, 2, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            r = np.add(a, 2, None, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            r = np.add(a, 2, out=None, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            r = np.add(a, 2, out=(None,), subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            d = ArrayWrap([5.7])
            o1 = np.empty((1,))
            o2 = np.empty((1,), dtype=np.int32)

            r1, r2 = np.frexp(d, o1, subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            r1, r2 = np.frexp(d, o1, None, subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            r1, r2 = np.frexp(d, None, o2, subok=subok)
            if subok:
                assert_(isinstance(r1, ArrayWrap))
            else:
                assert_(type(r1) == np.ndarray)

            r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
            if subok:
                assert_(isinstance(r1, ArrayWrap))
            else:
                assert_(type(r1) == np.ndarray)

            with assert_raises(TypeError):
                # Out argument must be tuple, since there are multiple outputs.
                r1, r2 = np.frexp(d, out=o1, subok=subok)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_out_wrap_no_leak(self):
        # Regression test for gh-26545
        class ArrSubclass(np.ndarray):
            pass

        arr = np.arange(10).view(ArrSubclass)
        orig_refcount = sys.getrefcount(arr)
        arr *= 1
        assert sys.getrefcount(arr) == orig_refcount


class TestComparisons:
    import operator

    @pytest.mark.parametrize('dtype', sctypes['uint'] + sctypes['int'] +
                             sctypes['float'] + [np.bool])
    @pytest.mark.parametrize('py_comp,np_comp', [
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal)
    ])
    def test_comparison_functions(self, dtype, py_comp, np_comp):
        # Initialize input arrays
        if dtype == np.bool:
            a = np.random.choice(a=[False, True], size=1000)
            b = np.random.choice(a=[False, True], size=1000)
            scalar = True
        else:
            a = np.random.randint(low=1, high=10, size=1000).astype(dtype)
            b = np.random.randint(low=1, high=10, size=1000).astype(dtype)
            scalar = 5
        np_scalar = np.dtype(dtype).type(scalar)
        a_lst = a.tolist()
        b_lst = b.tolist()

        # (Binary) Comparison (x1=array, x2=array)
        comp_b = np_comp(a, b).view(np.uint8)
        comp_b_list = [int(py_comp(x, y)) for x, y in zip(a_lst, b_lst)]

        # (Scalar1) Comparison (x1=scalar, x2=array)
        comp_s1 = np_comp(np_scalar, b).view(np.uint8)
        comp_s1_list = [int(py_comp(scalar, x)) for x in b_lst]

        # (Scalar2) Comparison (x1=array, x2=scalar)
        comp_s2 = np_comp(a, np_scalar).view(np.uint8)
        comp_s2_list = [int(py_comp(x, scalar)) for x in a_lst]

        # Sequence: Binary, Scalar1 and Scalar2
        assert_(comp_b.tolist() == comp_b_list,
            f"Failed comparison ({py_comp.__name__})")
        assert_(comp_s1.tolist() == comp_s1_list,
            f"Failed comparison ({py_comp.__name__})")
        assert_(comp_s2.tolist() == comp_s2_list,
            f"Failed comparison ({py_comp.__name__})")

    def test_ignore_object_identity_in_equal(self):
        # Check comparing identical objects whose comparison
        # is not a simple boolean, e.g., arrays that are compared elementwise.
        a = np.array([np.array([1, 2, 3]), None], dtype=object)
        assert_raises(ValueError, np.equal, a, a)

        # Check error raised when comparing identical non-comparable objects.
        class FunkyType:
            def __eq__(self, other):
                raise TypeError("I won't compare")

        a = np.array([FunkyType()])
        assert_raises(TypeError, np.equal, a, a)

        # Check identity doesn't override comparison mismatch.
        a = np.array([np.nan], dtype=object)
        assert_equal(np.equal(a, a), [False])

    def test_ignore_object_identity_in_not_equal(self):
        # Check comparing identical objects whose comparison
        # is not a simple boolean, e.g., arrays that are compared elementwise.
        a = np.array([np.array([1, 2, 3]), None], dtype=object)
        assert_raises(ValueError, np.not_equal, a, a)

        # Check error raised when comparing identical non-comparable objects.
        class FunkyType:
            def __ne__(self, other):
                raise TypeError("I won't compare")

        a = np.array([FunkyType()])
        assert_raises(TypeError, np.not_equal, a, a)

        # Check identity doesn't override comparison mismatch.
        a = np.array([np.nan], dtype=object)
        assert_equal(np.not_equal(a, a), [True])

    def test_error_in_equal_reduce(self):
        # gh-20929
        # make sure np.equal.reduce raises a TypeError if an array is passed
        # without specifying the dtype
        a = np.array([0, 0])
        assert_equal(np.equal.reduce(a, dtype=bool), True)
        assert_raises(TypeError, np.equal.reduce, a)

    def test_object_dtype(self):
        assert np.equal(1, [1], dtype=object).dtype == object
        assert np.equal(1, [1], signature=(None, None, "O")).dtype == object

    def test_object_nonbool_dtype_error(self):
        # bool output dtype is fine of course:
        assert np.equal(1, [1], dtype=bool).dtype == bool

        # but the following are examples do not have a loop:
        with pytest.raises(TypeError, match="No loop matching"):
            np.equal(1, 1, dtype=np.int64)

        with pytest.raises(TypeError, match="No loop matching"):
            np.equal(1, 1, sig=(None, None, "l"))

    @pytest.mark.parametrize("dtypes", ["qQ", "Qq"])
    @pytest.mark.parametrize('py_comp, np_comp', [
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal)
    ])
    @pytest.mark.parametrize("vals", [(2**60, 2**60 + 1), (2**60 + 1, 2**60)])
    def test_large_integer_direct_comparison(
            self, dtypes, py_comp, np_comp, vals):
        # Note that float(2**60) + 1 == float(2**60).
        a1 = np.array([2**60], dtype=dtypes[0])
        a2 = np.array([2**60 + 1], dtype=dtypes[1])
        expected = py_comp(2**60, 2**60 + 1)

        assert py_comp(a1, a2) == expected
        assert np_comp(a1, a2) == expected
        # Also check the scalars:
        s1 = a1[0]
        s2 = a2[0]
        assert isinstance(s1, np.integer)
        assert isinstance(s2, np.integer)
        # The Python operator here is mainly interesting:
        assert py_comp(s1, s2) == expected
        assert np_comp(s1, s2) == expected

    @pytest.mark.parametrize("dtype", np.typecodes['UnsignedInteger'])
    @pytest.mark.parametrize('py_comp_func, np_comp_func', [
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal)
    ])
    @pytest.mark.parametrize("flip", [True, False])
    def test_unsigned_signed_direct_comparison(
            self, dtype, py_comp_func, np_comp_func, flip):
        if flip:
            py_comp = lambda x, y: py_comp_func(y, x)
            np_comp = lambda x, y: np_comp_func(y, x)
        else:
            py_comp = py_comp_func
            np_comp = np_comp_func

        arr = np.array([np.iinfo(dtype).max], dtype=dtype)
        expected = py_comp(int(arr[0]), -1)

        assert py_comp(arr, -1) == expected
        assert np_comp(arr, -1) == expected

        scalar = arr[0]
        assert isinstance(scalar, np.integer)
        # The Python operator here is mainly interesting:
        assert py_comp(scalar, -1) == expected
        assert np_comp(scalar, -1) == expected


class TestAdd:
    def test_reduce_alignment(self):
        # gh-9876
        # make sure arrays with weird strides work with the optimizations in
        # pairwise_sum_@TYPE@. On x86, the 'b' field will count as aligned at a
        # 4 byte offset, even though its itemsize is 8.
        a = np.zeros(2, dtype=[('a', np.int32), ('b', np.float64)])
        a['a'] = -1
        assert_equal(a['b'].sum(), 0)


class TestDivision:
    def test_division_int(self):
        # int division should follow Python
        x = np.array([5, 10, 90, 100, -5, -10, -90, -100, -120])
        if 5 / 10 == 0.5:
            assert_equal(x / 100, [0.05, 0.1, 0.9, 1,
                                   -0.05, -0.1, -0.9, -1, -1.2])
        else:
            assert_equal(x / 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x // 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x % 100, [5, 10, 90, 0, 95, 90, 10, 0, 80])

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype,ex_val", itertools.product(
        sctypes['int'] + sctypes['uint'], (
            (
                # dividend
                "np.array(range(fo.max-lsize, fo.max)).astype(dtype),"
                # divisors
                "np.arange(lsize).astype(dtype),"
                # scalar divisors
                "range(15)"
            ),
            (
                # dividend
                "np.arange(fo.min, fo.min+lsize).astype(dtype),"
                # divisors
                "np.arange(lsize//-2, lsize//2).astype(dtype),"
                # scalar divisors
                "range(fo.min, fo.min + 15)"
            ), (
                # dividend
                "np.array(range(fo.max-lsize, fo.max)).astype(dtype),"
                # divisors
                "np.arange(lsize).astype(dtype),"
                # scalar divisors
                "[1,3,9,13,neg, fo.min+1, fo.min//2, fo.max//3, fo.max//4]"
            )
        )
    ))
    def test_division_int_boundary(self, dtype, ex_val):
        fo = np.iinfo(dtype)
        neg = -1 if fo.min < 0 else 1
        # Large enough to test SIMD loops and remainder elements
        lsize = 512 + 7
        a, b, divisors = eval(ex_val)
        a_lst, b_lst = a.tolist(), b.tolist()

        c_div = lambda n, d: (
            0 if d == 0 else (
                fo.min if (n and n == fo.min and d == -1) else n // d
            )
        )
        with np.errstate(divide='ignore'):
            ac = a.copy()
            ac //= b
            div_ab = a // b
        div_lst = [c_div(x, y) for x, y in zip(a_lst, b_lst)]

        msg = "Integer arrays floor division check (//)"
        assert all(div_ab == div_lst), msg
        msg_eq = "Integer arrays floor division check (//=)"
        assert all(ac == div_lst), msg_eq

        for divisor in divisors:
            ac = a.copy()
            with np.errstate(divide='ignore', over='ignore'):
                div_a = a // divisor
                ac //= divisor
            div_lst = [c_div(i, divisor) for i in a_lst]

            assert all(div_a == div_lst), msg
            assert all(ac == div_lst), msg_eq

        with np.errstate(divide='raise', over='raise'):
            if 0 in b:
                # Verify overflow case
                with pytest.raises(FloatingPointError,
                        match="divide by zero encountered in floor_divide"):
                    a // b
            else:
                a // b
            if fo.min and fo.min in a:
                with pytest.raises(FloatingPointError,
                        match='overflow encountered in floor_divide'):
                    a // -1
            elif fo.min:
                a // -1
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in floor_divide"):
                a // 0
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in floor_divide"):
                ac = a.copy()
                ac //= 0

            np.array([], dtype=dtype) // 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype,ex_val", itertools.product(
        sctypes['int'] + sctypes['uint'], (
            "np.array([fo.max, 1, 2, 1, 1, 2, 3], dtype=dtype)",
            "np.array([fo.min, 1, -2, 1, 1, 2, -3]).astype(dtype)",
            "np.arange(fo.min, fo.min+(100*10), 10, dtype=dtype)",
            "np.array(range(fo.max-(100*7), fo.max, 7)).astype(dtype)",
        )
    ))
    def test_division_int_reduce(self, dtype, ex_val):
        fo = np.iinfo(dtype)
        a = eval(ex_val)
        lst = a.tolist()
        c_div = lambda n, d: (
            0 if d == 0 or (n and n == fo.min and d == -1) else n // d
        )

        with np.errstate(divide='ignore'):
            div_a = np.floor_divide.reduce(a)
        div_lst = reduce(c_div, lst)
        msg = "Reduce floor integer division check"
        assert div_a == div_lst, msg

        with np.errstate(divide='raise', over='raise'):
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in reduce"):
                np.floor_divide.reduce(np.arange(-100, 100).astype(dtype))
            if fo.min:
                with pytest.raises(FloatingPointError,
                        match='overflow encountered in reduce'):
                    np.floor_divide.reduce(
                        np.array([fo.min, 1, -1], dtype=dtype)
                    )

    @pytest.mark.parametrize(
            "dividend,divisor,quotient",
            [(np.timedelta64(2, 'Y'), np.timedelta64(2, 'M'), 12),
             (np.timedelta64(2, 'Y'), np.timedelta64(-2, 'M'), -12),
             (np.timedelta64(-2, 'Y'), np.timedelta64(2, 'M'), -12),
             (np.timedelta64(-2, 'Y'), np.timedelta64(-2, 'M'), 12),
             (np.timedelta64(2, 'M'), np.timedelta64(-2, 'Y'), -1),
             (np.timedelta64(2, 'Y'), np.timedelta64(0, 'M'), 0),
             (np.timedelta64(2, 'Y'), 2, np.timedelta64(1, 'Y')),
             (np.timedelta64(2, 'Y'), -2, np.timedelta64(-1, 'Y')),
             (np.timedelta64(-2, 'Y'), 2, np.timedelta64(-1, 'Y')),
             (np.timedelta64(-2, 'Y'), -2, np.timedelta64(1, 'Y')),
             (np.timedelta64(-2, 'Y'), -2, np.timedelta64(1, 'Y')),
             (np.timedelta64(-2, 'Y'), -3, np.timedelta64(0, 'Y')),
             (np.timedelta64(-2, 'Y'), 0, np.timedelta64('Nat', 'Y')),
            ])
    def test_division_int_timedelta(self, dividend, divisor, quotient):
        # If either divisor is 0 or quotient is Nat, check for division by 0
        if divisor and (isinstance(quotient, int) or not np.isnat(quotient)):
            msg = "Timedelta floor division check"
            assert dividend // divisor == quotient, msg

            # Test for arrays as well
            msg = "Timedelta arrays floor division check"
            dividend_array = np.array([dividend] * 5)
            quotient_array = np.array([quotient] * 5)
            assert all(dividend_array // divisor == quotient_array), msg
        else:
            if IS_WASM:
                pytest.skip("fp errors don't work in wasm")
            with np.errstate(divide='raise', invalid='raise'):
                with pytest.raises(FloatingPointError):
                    dividend // divisor

    def test_division_complex(self):
        # check that implementation is correct
        msg = "Complex division implementation check"
        x = np.array([1. + 1. * 1j, 1. + .5 * 1j, 1. + 2. * 1j], dtype=np.complex128)
        assert_almost_equal(x**2 / x, x, err_msg=msg)
        # check overflow, underflow
        msg = "Complex division overflow/underflow check"
        x = np.array([1.e+110, 1.e-110], dtype=np.complex128)
        y = x**2 / x
        assert_almost_equal(y / x, [1, 1], err_msg=msg)

    def test_zero_division_complex(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.array([0.0], dtype=np.complex128)
            y = 1.0 / x
            assert_(np.isinf(y)[0])
            y = complex(np.inf, np.nan) / x
            assert_(np.isinf(y)[0])
            y = complex(np.nan, np.inf) / x
            assert_(np.isinf(y)[0])
            y = complex(np.inf, np.inf) / x
            assert_(np.isinf(y)[0])
            y = 0.0 / x
            assert_(np.isnan(y)[0])

    def test_floor_division_complex(self):
        # check that floor division, divmod and remainder raises type errors
        x = np.array([.9 + 1j, -.1 + 1j, .9 + .5 * 1j, .9 + 2. * 1j], dtype=np.complex128)
        with pytest.raises(TypeError):
            x // 7
        with pytest.raises(TypeError):
            np.divmod(x, 7)
        with pytest.raises(TypeError):
            np.remainder(x, 7)

    def test_floor_division_signed_zero(self):
        # Check that the sign bit is correctly set when dividing positive and
        # negative zero by one.
        x = np.zeros(10)
        assert_equal(np.signbit(x // 1), 0)
        assert_equal(np.signbit((-x) // 1), 1)

    @pytest.mark.skipif(hasattr(np.__config__, "blas_ssl2_info"),
            reason="gh-22982")
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_floor_division_errors(self, dtype):
        fnan = np.array(np.nan, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        fzer = np.array(0.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        # divide by zero error check
        with np.errstate(divide='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.floor_divide, fone, fzer)
        with np.errstate(divide='ignore', invalid='raise'):
            np.floor_divide(fone, fzer)

        # The following already contain a NaN and should not warn
        with np.errstate(all='raise'):
            np.floor_divide(fnan, fone)
            np.floor_divide(fone, fnan)
            np.floor_divide(fnan, fzer)
            np.floor_divide(fzer, fnan)

    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_floor_division_corner_cases(self, dtype):
        # test corner cases like 1.0//0.0 for errors and return vals
        x = np.zeros(10, dtype=dtype)
        y = np.ones(10, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        fzer = np.array(0.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in floor_divide")
            div = np.floor_divide(fnan, fone)
            assert np.isnan(div), f"div: {div}"
            div = np.floor_divide(fone, fnan)
            assert np.isnan(div), f"div: {div}"
            div = np.floor_divide(fnan, fzer)
            assert np.isnan(div), f"div: {div}"
        # verify 1.0//0.0 computations return inf
        with np.errstate(divide='ignore'):
            z = np.floor_divide(y, x)
            assert_(np.isinf(z).all())

def floor_divide_and_remainder(x, y):
    return (np.floor_divide(x, y), np.remainder(x, y))


def _signs(dt):
    if dt in np.typecodes['UnsignedInteger']:
        return (+1,)
    else:
        return (+1, -1)


class TestRemainder:

    def test_remainder_basic(self):
        dt = np.typecodes['AllInteger'] + np.typecodes['Float']
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 71, dtype=dt1)
                    b = np.array(sg2 * 19, dtype=dt2)
                    div, rem = op(a, b)
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_remainder_exact(self):
        # test that float results are exact for small integers. This also
        # holds for the same integers scaled by powers of two.
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = [divmod(*t) for t in arg]

        a, b = np.array(arg, dtype=int).T
        # convert exact integer results from Python to float so that
        # signed zero can be used, it is checked.
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        for op in [floor_divide_and_remainder, np.divmod]:
            for dt in np.typecodes['Float']:
                msg = f'op: {op.__name__}, dtype: {dt}'
                fa = a.astype(dt)
                fb = b.astype(dt)
                div, rem = op(fa, fb)
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)

    def test_float_remainder_roundoff(self):
        # gh-6127
        dt = np.typecodes['Float']
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1 * 78 * 6e-8, dtype=dt1)
                    b = np.array(sg2 * 6e-8, dtype=dt2)
                    div, rem = op(a, b)
                    # Equal assertion should hold when fmod is used
                    assert_equal(div * b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith("darwin"),
            reason="MacOS seems to not give the correct 'invalid' warning for "
                   "`fmod`.  Hopefully, others always do.")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_float_divmod_errors(self, dtype):
        # Check valid errors raised for divmod and remainder
        fzero = np.array(0.0, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)
        # since divmod is combination of both remainder and divide
        # ops it will set both dividebyzero and invalid flags
        with np.errstate(divide='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.divmod, fone, fzero)
        with np.errstate(divide='ignore', invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, fone, fzero)
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, fzero, fzero)
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, finf, finf)
        with np.errstate(divide='ignore', invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, finf, fzero)
        with np.errstate(divide='raise', invalid='ignore'):
            # inf / 0 does not set any flags, only the modulo creates a NaN
            np.divmod(finf, fzero)

    @pytest.mark.skipif(hasattr(np.__config__, "blas_ssl2_info"),
            reason="gh-22982")
    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith("darwin"),
           reason="MacOS seems to not give the correct 'invalid' warning for "
                  "`fmod`.  Hopefully, others always do.")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    @pytest.mark.parametrize('fn', [np.fmod, np.remainder])
    def test_float_remainder_errors(self, dtype, fn):
        fzero = np.array(0.0, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)

        # The following already contain a NaN and should not warn.
        with np.errstate(all='raise'):
            with pytest.raises(FloatingPointError,
                    match="invalid value"):
                fn(fone, fzero)
            fn(fnan, fzero)
            fn(fzero, fnan)
            fn(fone, fnan)
            fn(fnan, fone)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_float_remainder_overflow(self):
        a = np.finfo(np.float64).tiny
        with np.errstate(over='ignore', invalid='ignore'):
            div, mod = np.divmod(4, a)
            np.isinf(div)
            assert_(mod == 0)
        with np.errstate(over='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.divmod, 4, a)
        with np.errstate(invalid='raise', over='ignore'):
            assert_raises(FloatingPointError, np.divmod, 4, a)

    def test_float_divmod_corner_cases(self):
        # check nan cases
        for dt in np.typecodes['Float']:
            fnan = np.array(np.nan, dtype=dt)
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "invalid value encountered in divmod")
                sup.filter(RuntimeWarning, "divide by zero encountered in divmod")
                div, rem = np.divmod(fone, fzer)
                assert np.isinf(div), f'dt: {dt}, div: {rem}'
                assert np.isnan(rem), f'dt: {dt}, rem: {rem}'
                div, rem = np.divmod(fzer, fzer)
                assert np.isnan(rem), f'dt: {dt}, rem: {rem}'
                assert_(np.isnan(div)), f'dt: {dt}, rem: {rem}'
                div, rem = np.divmod(finf, finf)
                assert np.isnan(div), f'dt: {dt}, rem: {rem}'
                assert np.isnan(rem), f'dt: {dt}, rem: {rem}'
                div, rem = np.divmod(finf, fzer)
                assert np.isinf(div), f'dt: {dt}, rem: {rem}'
                assert np.isnan(rem), f'dt: {dt}, rem: {rem}'
                div, rem = np.divmod(fnan, fone)
                assert np.isnan(rem), f"dt: {dt}, rem: {rem}"
                assert np.isnan(div), f"dt: {dt}, rem: {rem}"
                div, rem = np.divmod(fone, fnan)
                assert np.isnan(rem), f"dt: {dt}, rem: {rem}"
                assert np.isnan(div), f"dt: {dt}, rem: {rem}"
                div, rem = np.divmod(fnan, fzer)
                assert np.isnan(rem), f"dt: {dt}, rem: {rem}"
                assert np.isnan(div), f"dt: {dt}, rem: {rem}"

    def test_float_remainder_corner_cases(self):
        # Check remainder magnitude.
        for dt in np.typecodes['Float']:
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            fnan = np.array(np.nan, dtype=dt)
            b = np.array(1.0, dtype=dt)
            a = np.nextafter(np.array(0.0, dtype=dt), -b)
            rem = np.remainder(a, b)
            assert_(rem <= b, f'dt: {dt}')
            rem = np.remainder(-a, -b)
            assert_(rem >= -b, f'dt: {dt}')

        # Check nans, inf
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in remainder")
            sup.filter(RuntimeWarning, "invalid value encountered in fmod")
            for dt in np.typecodes['Float']:
                fone = np.array(1.0, dtype=dt)
                fzer = np.array(0.0, dtype=dt)
                finf = np.array(np.inf, dtype=dt)
                fnan = np.array(np.nan, dtype=dt)
                rem = np.remainder(fone, fzer)
                assert_(np.isnan(rem), f'dt: {dt}, rem: {rem}')
                # MSVC 2008 returns NaN here, so disable the check.
                #rem = np.remainder(fone, finf)
                #assert_(rem == fone, 'dt: %s, rem: %s' % (dt, rem))
                rem = np.remainder(finf, fone)
                fmod = np.fmod(finf, fone)
                assert_(np.isnan(fmod), f'dt: {dt}, fmod: {fmod}')
                assert_(np.isnan(rem), f'dt: {dt}, rem: {rem}')
                rem = np.remainder(finf, finf)
                fmod = np.fmod(finf, fone)
                assert_(np.isnan(rem), f'dt: {dt}, rem: {rem}')
                assert_(np.isnan(fmod), f'dt: {dt}, fmod: {fmod}')
                rem = np.remainder(finf, fzer)
                fmod = np.fmod(finf, fzer)
                assert_(np.isnan(rem), f'dt: {dt}, rem: {rem}')
                assert_(np.isnan(fmod), f'dt: {dt}, fmod: {fmod}')
                rem = np.remainder(fone, fnan)
                fmod = np.fmod(fone, fnan)
                assert_(np.isnan(rem), f'dt: {dt}, rem: {rem}')
                assert_(np.isnan(fmod), f'dt: {dt}, fmod: {fmod}')
                rem = np.remainder(fnan, fzer)
                fmod = np.fmod(fnan, fzer)
                assert_(np.isnan(rem), f'dt: {dt}, rem: {rem}')
                assert_(np.isnan(fmod), f'dt: {dt}, fmod: {rem}')
                rem = np.remainder(fnan, fone)
                fmod = np.fmod(fnan, fone)
                assert_(np.isnan(rem), f'dt: {dt}, rem: {rem}')
                assert_(np.isnan(fmod), f'dt: {dt}, fmod: {rem}')


class TestDivisionIntegerOverflowsAndDivideByZero:
    result_type = namedtuple('result_type',
            ['nocast', 'casted'])
    helper_lambdas = {
        'zero': lambda dtype: 0,
        'min': lambda dtype: np.iinfo(dtype).min,
        'neg_min': lambda dtype: -np.iinfo(dtype).min,
        'min-zero': lambda dtype: (np.iinfo(dtype).min, 0),
        'neg_min-zero': lambda dtype: (-np.iinfo(dtype).min, 0),
    }
    overflow_results = {
        np.remainder: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        np.fmod: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        operator.mod: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        operator.floordiv: result_type(
            helper_lambdas['min'], helper_lambdas['neg_min']),
        np.floor_divide: result_type(
            helper_lambdas['min'], helper_lambdas['neg_min']),
        np.divmod: result_type(
            helper_lambdas['min-zero'], helper_lambdas['neg_min-zero'])
    }

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype", np.typecodes["Integer"])
    def test_signed_division_overflow(self, dtype):
        to_check = interesting_binop_operands(np.iinfo(dtype).min, -1, dtype)
        for op1, op2, extractor, operand_identifier in to_check:
            with pytest.warns(RuntimeWarning, match="overflow encountered"):
                res = op1 // op2

            assert res.dtype == op1.dtype
            assert extractor(res) == np.iinfo(op1.dtype).min

            # Remainder is well defined though, and does not warn:
            res = op1 % op2
            assert res.dtype == op1.dtype
            assert extractor(res) == 0
            # Check fmod as well:
            res = np.fmod(op1, op2)
            assert extractor(res) == 0

            # Divmod warns for the division part:
            with pytest.warns(RuntimeWarning, match="overflow encountered"):
                res1, res2 = np.divmod(op1, op2)

            assert res1.dtype == res2.dtype == op1.dtype
            assert extractor(res1) == np.iinfo(op1.dtype).min
            assert extractor(res2) == 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_divide_by_zero(self, dtype):
        # Note that the return value cannot be well defined here, but NumPy
        # currently uses 0 consistently.  This could be changed.
        to_check = interesting_binop_operands(1, 0, dtype)
        for op1, op2, extractor, operand_identifier in to_check:
            with pytest.warns(RuntimeWarning, match="divide by zero"):
                res = op1 // op2

            assert res.dtype == op1.dtype
            assert extractor(res) == 0

            with pytest.warns(RuntimeWarning, match="divide by zero"):
                res1, res2 = np.divmod(op1, op2)

            assert res1.dtype == res2.dtype == op1.dtype
            assert extractor(res1) == 0
            assert extractor(res2) == 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dividend_dtype", sctypes['int'])
    @pytest.mark.parametrize("divisor_dtype", sctypes['int'])
    @pytest.mark.parametrize("operation",
            [np.remainder, np.fmod, np.divmod, np.floor_divide,
             operator.mod, operator.floordiv])
    @np.errstate(divide='warn', over='warn')
    def test_overflows(self, dividend_dtype, divisor_dtype, operation):
        # SIMD tries to perform the operation on as many elements as possible
        # that is a multiple of the register's size. We resort to the
        # default implementation for the leftover elements.
        # We try to cover all paths here.
        arrays = [np.array([np.iinfo(dividend_dtype).min] * i,
                           dtype=dividend_dtype) for i in range(1, 129)]
        divisor = np.array([-1], dtype=divisor_dtype)
        # If dividend is a larger type than the divisor (`else` case),
        # then, result will be a larger type than dividend and will not
        # result in an overflow for `divmod` and `floor_divide`.
        if np.dtype(dividend_dtype).itemsize >= np.dtype(
                divisor_dtype).itemsize and operation in (
                        np.divmod, np.floor_divide, operator.floordiv):
            with pytest.warns(
                    RuntimeWarning,
                    match="overflow encountered in"):
                result = operation(
                            dividend_dtype(np.iinfo(dividend_dtype).min),
                            divisor_dtype(-1)
                        )
                assert result == self.overflow_results[operation].nocast(
                        dividend_dtype)

            # Arrays
            for a in arrays:
                # In case of divmod, we need to flatten the result
                # column first as we get a column vector of quotient and
                # remainder and a normal flatten of the expected result.
                with pytest.warns(
                        RuntimeWarning,
                        match="overflow encountered in"):
                    result = np.array(operation(a, divisor)).flatten('f')
                    expected_array = np.array(
                            [self.overflow_results[operation].nocast(
                                dividend_dtype)] * len(a)).flatten()
                    assert_array_equal(result, expected_array)
        else:
            # Scalars
            result = operation(
                        dividend_dtype(np.iinfo(dividend_dtype).min),
                        divisor_dtype(-1)
                    )
            assert result == self.overflow_results[operation].casted(
                    dividend_dtype)

            # Arrays
            for a in arrays:
                # See above comment on flatten
                result = np.array(operation(a, divisor)).flatten('f')
                expected_array = np.array(
                        [self.overflow_results[operation].casted(
                            dividend_dtype)] * len(a)).flatten()
                assert_array_equal(result, expected_array)


class TestCbrt:
    def test_cbrt_scalar(self):
        assert_almost_equal((np.cbrt(np.float32(-2.5)**3)), -2.5)

    def test_cbrt(self):
        x = np.array([1., 2., -3., np.inf, -np.inf])
        assert_almost_equal(np.cbrt(x**3), x)

        assert_(np.isnan(np.cbrt(np.nan)))
        assert_equal(np.cbrt(np.inf), np.inf)
        assert_equal(np.cbrt(-np.inf), -np.inf)


class TestPower:
    def test_power_float(self):
        x = np.array([1., 2., 3.])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_equal(x**2, [1., 4., 9.])
        y = x.copy()
        y **= 2
        assert_equal(y, [1., 4., 9.])
        assert_almost_equal(x**(-1), [1., 0.5, 1. / 3])
        assert_almost_equal(x**(0.5), [1., ncu.sqrt(2), ncu.sqrt(3)])

        for out, inp, msg in _gen_alignment_data(dtype=np.float32,
                                                 type='unary',
                                                 max_size=11):
            exp = [ncu.sqrt(i) for i in inp]
            assert_almost_equal(inp**(0.5), exp, err_msg=msg)
            np.sqrt(inp, out=out)
            assert_equal(out, exp, err_msg=msg)

        for out, inp, msg in _gen_alignment_data(dtype=np.float64,
                                                 type='unary',
                                                 max_size=7):
            exp = [ncu.sqrt(i) for i in inp]
            assert_almost_equal(inp**(0.5), exp, err_msg=msg)
            np.sqrt(inp, out=out)
            assert_equal(out, exp, err_msg=msg)

    def test_power_complex(self):
        x = np.array([1 + 2j, 2 + 3j, 3 + 4j])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_almost_equal(x**2, [-3 + 4j, -5 + 12j, -7 + 24j])
        assert_almost_equal(x**3, [(1 + 2j)**3, (2 + 3j)**3, (3 + 4j)**3])
        assert_almost_equal(x**4, [(1 + 2j)**4, (2 + 3j)**4, (3 + 4j)**4])
        assert_almost_equal(x**(-1), [1 / (1 + 2j), 1 / (2 + 3j), 1 / (3 + 4j)])
        assert_almost_equal(x**(-2), [1 / (1 + 2j)**2, 1 / (2 + 3j)**2, 1 / (3 + 4j)**2])
        assert_almost_equal(x**(-3), [(-11 + 2j) / 125, (-46 - 9j) / 2197,
                                      (-117 - 44j) / 15625])
        assert_almost_equal(x**(0.5), [ncu.sqrt(1 + 2j), ncu.sqrt(2 + 3j),
                                       ncu.sqrt(3 + 4j)])
        norm = 1. / ((x**14)[0])
        assert_almost_equal(x**14 * norm,
                [i * norm for i in [-76443 + 16124j, 23161315 + 58317492j,
                                    5583548873 + 2465133864j]])

        # Ticket #836
        def assert_complex_equal(x, y):
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        for z in [complex(0, np.inf), complex(1, np.inf)]:
            z = np.array([z], dtype=np.complex128)
            with np.errstate(invalid="ignore"):
                assert_complex_equal(z**1, z)
                assert_complex_equal(z**2, z * z)
                assert_complex_equal(z**3, z * z * z)

    def test_power_zero(self):
        # ticket #1271
        zero = np.array([0j])
        one = np.array([1 + 0j])
        cnan = np.array([complex(np.nan, np.nan)])
        # FIXME cinf not tested.
        #cinf = np.array([complex(np.inf, 0)])

        def assert_complex_equal(x, y):
            x, y = np.asarray(x), np.asarray(y)
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        # positive powers
        for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
            assert_complex_equal(np.power(zero, p), zero)

        # zero power
        assert_complex_equal(np.power(zero, 0), one)
        with np.errstate(invalid="ignore"):
            assert_complex_equal(np.power(zero, 0 + 1j), cnan)

            # negative power
            for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
                assert_complex_equal(np.power(zero, -p), cnan)
            assert_complex_equal(np.power(zero, -1 + 0.2j), cnan)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_zero_power_nonzero(self):
        # Testing 0^{Non-zero} issue 18378
        zero = np.array([0.0 + 0.0j])
        cnan = np.array([complex(np.nan, np.nan)])

        def assert_complex_equal(x, y):
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        # Complex powers with positive real part will not generate a warning
        assert_complex_equal(np.power(zero, 1 + 4j), zero)
        assert_complex_equal(np.power(zero, 2 - 3j), zero)
        # Testing zero values when real part is greater than zero
        assert_complex_equal(np.power(zero, 1 + 1j), zero)
        assert_complex_equal(np.power(zero, 1 + 0j), zero)
        assert_complex_equal(np.power(zero, 1 - 1j), zero)
        # Complex powers will negative real part or 0 (provided imaginary
        # part is not zero) will generate a NAN and hence a RUNTIME warning
        with pytest.warns(expected_warning=RuntimeWarning) as r:
            assert_complex_equal(np.power(zero, -1 + 1j), cnan)
            assert_complex_equal(np.power(zero, -2 - 3j), cnan)
            assert_complex_equal(np.power(zero, -7 + 0j), cnan)
            assert_complex_equal(np.power(zero, 0 + 1j), cnan)
            assert_complex_equal(np.power(zero, 0 - 1j), cnan)
        assert len(r) == 5

    def test_fast_power(self):
        x = np.array([1, 2, 3], np.int16)
        res = x**2.0
        assert_((x**2.00001).dtype is res.dtype)
        assert_array_equal(res, [1, 4, 9])
        # check the inplace operation on the casted copy doesn't mess with x
        assert_(not np.may_share_memory(res, x))
        assert_array_equal(x, [1, 2, 3])

        # Check that the fast path ignores 1-element not 0-d arrays
        res = x ** np.array([[[2]]])
        assert_equal(res.shape, (1, 1, 3))

    def test_integer_power(self):
        a = np.array([15, 15], 'i8')
        b = np.power(a, a)
        assert_equal(b, [437893890380859375, 437893890380859375])

    def test_integer_power_with_integer_zero_exponent(self):
        dtypes = np.typecodes['Integer']
        for dt in dtypes:
            arr = np.arange(-10, 10, dtype=dt)
            assert_equal(np.power(arr, 0), np.ones_like(arr))

        dtypes = np.typecodes['UnsignedInteger']
        for dt in dtypes:
            arr = np.arange(10, dtype=dt)
            assert_equal(np.power(arr, 0), np.ones_like(arr))

    def test_integer_power_of_1(self):
        dtypes = np.typecodes['AllInteger']
        for dt in dtypes:
            arr = np.arange(10, dtype=dt)
            assert_equal(np.power(1, arr), np.ones_like(arr))

    def test_integer_power_of_zero(self):
        dtypes = np.typecodes['AllInteger']
        for dt in dtypes:
            arr = np.arange(1, 10, dtype=dt)
            assert_equal(np.power(0, arr), np.zeros_like(arr))

    def test_integer_to_negative_power(self):
        dtypes = np.typecodes['Integer']
        for dt in dtypes:
            a = np.array([0, 1, 2, 3], dtype=dt)
            b = np.array([0, 1, 2, -3], dtype=dt)
            one = np.array(1, dtype=dt)
            minusone = np.array(-1, dtype=dt)
            assert_raises(ValueError, np.power, a, b)
            assert_raises(ValueError, np.power, a, minusone)
            assert_raises(ValueError, np.power, one, b)
            assert_raises(ValueError, np.power, one, minusone)

    def test_float_to_inf_power(self):
        for dt in [np.float32, np.float64]:
            a = np.array([1, 1, 2, 2, -2, -2, np.inf, -np.inf], dt)
            b = np.array([np.inf, -np.inf, np.inf, -np.inf,
                                np.inf, -np.inf, np.inf, -np.inf], dt)
            r = np.array([1, 1, np.inf, 0, np.inf, 0, np.inf, 0], dt)
            assert_equal(np.power(a, b), r)

    def test_power_fast_paths(self):
        # gh-26055
        for dt in [np.float32, np.float64]:
            a = np.array([0, 1.1, 2, 12e12, -10., np.inf, -np.inf], dt)
            expected = np.array([0.0, 1.21, 4., 1.44e+26, 100, np.inf, np.inf])
            result = np.power(a, 2.)
            assert_array_max_ulp(result, expected.astype(dt), maxulp=1)

            a = np.array([0, 1.1, 2, 12e12], dt)
            expected = np.sqrt(a).astype(dt)
            result = np.power(a, 0.5)
            assert_array_max_ulp(result, expected, maxulp=1)


class TestFloat_power:
    def test_type_conversion(self):
        arg_type = '?bhilBHILefdgFDG'
        res_type = 'ddddddddddddgDDG'
        for dtin, dtout in zip(arg_type, res_type):
            msg = f"dtin: {dtin}, dtout: {dtout}"
            arg = np.ones(1, dtype=dtin)
            res = np.float_power(arg, arg)
            assert_(res.dtype.name == np.dtype(dtout).name, msg)


class TestLog2:
    @pytest.mark.parametrize('dt', ['f', 'd', 'g'])
    def test_log2_values(self, dt):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        xf = np.array(x, dtype=dt)
        yf = np.array(y, dtype=dt)
        assert_almost_equal(np.log2(xf), yf)

    @pytest.mark.parametrize("i", range(1, 65))
    def test_log2_ints(self, i):
        # a good log2 implementation should provide this,
        # might fail on OS with bad libm
        v = np.log2(2.**i)
        assert_equal(v, float(i), err_msg='at exponent %d' % i)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_log2_special(self):
        assert_equal(np.log2(1.), 0.)
        assert_equal(np.log2(np.inf), np.inf)
        assert_(np.isnan(np.log2(np.nan)))

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.log2(-1.)))
            assert_(np.isnan(np.log2(-np.inf)))
            assert_equal(np.log2(0.), -np.inf)
            assert_(w[0].category is RuntimeWarning)
            assert_(w[1].category is RuntimeWarning)
            assert_(w[2].category is RuntimeWarning)


class TestExp2:
    def test_exp2_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for dt in ['f', 'd', 'g']:
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_almost_equal(np.exp2(yf), xf)


class TestLogAddExp2(_FilterInvalids):
    # Need test for intermediate precisions
    def test_logaddexp2_values(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for dt, dec_ in zip(['f', 'd', 'g'], [6, 15, 15]):
            xf = np.log2(np.array(x, dtype=dt))
            yf = np.log2(np.array(y, dtype=dt))
            zf = np.log2(np.array(z, dtype=dt))
            assert_almost_equal(np.logaddexp2(xf, yf), zf, decimal=dec_)

    def test_logaddexp2_range(self):
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for dt in ['f', 'd', 'g']:
            logxf = np.array(x, dtype=dt)
            logyf = np.array(y, dtype=dt)
            logzf = np.array(z, dtype=dt)
            assert_almost_equal(np.logaddexp2(logxf, logyf), logzf)

    def test_inf(self):
        inf = np.inf
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]    # noqa: E221
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]  # noqa: E221
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        with np.errstate(invalid='raise'):
            for dt in ['f', 'd', 'g']:
                logxf = np.array(x, dtype=dt)
                logyf = np.array(y, dtype=dt)
                logzf = np.array(z, dtype=dt)
                assert_equal(np.logaddexp2(logxf, logyf), logzf)

    def test_nan(self):
        assert_(np.isnan(np.logaddexp2(np.nan, np.inf)))
        assert_(np.isnan(np.logaddexp2(np.inf, np.nan)))
        assert_(np.isnan(np.logaddexp2(np.nan, 0)))
        assert_(np.isnan(np.logaddexp2(0, np.nan)))
        assert_(np.isnan(np.logaddexp2(np.nan, np.nan)))

    def test_reduce(self):
        assert_equal(np.logaddexp2.identity, -np.inf)
        assert_equal(np.logaddexp2.reduce([]), -np.inf)
        assert_equal(np.logaddexp2.reduce([-np.inf]), -np.inf)
        assert_equal(np.logaddexp2.reduce([-np.inf, 0]), 0)


class TestLog:
    def test_log_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for dt in ['f', 'd', 'g']:
            log2_ = 0.69314718055994530943
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt) * log2_
            assert_almost_equal(np.log(xf), yf)

        # test aliasing(issue #17761)
        x = np.array([2, 0.937500, 3, 0.947500, 1.054697])
        xf = np.log(x)
        assert_almost_equal(np.log(x, out=x), xf)

    def test_log_values_maxofdtype(self):
        # test log() of max for dtype does not raise
        dtypes = [np.float32, np.float64]
        # This is failing at least on linux aarch64 (see gh-25460), and on most
        # other non x86-64 platforms checking `longdouble` isn't too useful as
        # it's an alias for float64.
        if platform.machine() == 'x86_64':
            dtypes += [np.longdouble]

        for dt in dtypes:
            with np.errstate(all='raise'):
                x = np.finfo(dt).max
                np.log(x)

    def test_log_strides(self):
        np.random.seed(42)
        strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        sizes = np.arange(2, 100)
        for ii in sizes:
            x_f64 = np.float64(np.random.uniform(low=0.01, high=100.0, size=ii))
            x_special = x_f64.copy()
            x_special[3:-1:4] = 1.0
            y_true = np.log(x_f64)
            y_special = np.log(x_special)
            for jj in strides:
                assert_array_almost_equal_nulp(np.log(x_f64[::jj]), y_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.log(x_special[::jj]), y_special[::jj], nulp=2)

    # Reference values were computed with mpmath, with mp.dps = 200.
    @pytest.mark.parametrize(
        'z, wref',
        [(1 + 1e-12j, 5e-25 + 1e-12j),
         (1.000000000000001 + 3e-08j,
          1.5602230246251546e-15 + 2.999999999999996e-08j),
         (0.9999995000000417 + 0.0009999998333333417j,
          7.831475869017683e-18 + 0.001j),
         (0.9999999999999996 + 2.999999999999999e-08j,
          5.9107901499372034e-18 + 3e-08j),
         (0.99995000042 - 0.009999833j,
          -7.015159763822903e-15 - 0.009999999665816696j)],
    )
    def test_log_precision_float64(self, z, wref):
        w = np.log(z)
        assert_allclose(w, wref, rtol=1e-15)

    # Reference values were computed with mpmath, with mp.dps = 200.
    @pytest.mark.parametrize(
        'z, wref',
        [(np.complex64(1.0 + 3e-6j), np.complex64(4.5e-12 + 3e-06j)),
         (np.complex64(1.0 - 2e-5j), np.complex64(1.9999999e-10 - 2e-5j)),
         (np.complex64(0.9999999 + 1e-06j),
          np.complex64(-1.192088e-07 + 1.0000001e-06j))],
    )
    def test_log_precision_float32(self, z, wref):
        w = np.log(z)
        assert_allclose(w, wref, rtol=1e-6)


class TestExp:
    def test_exp_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for dt in ['f', 'd', 'g']:
            log2_ = 0.69314718055994530943
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt) * log2_
            assert_almost_equal(np.exp(yf), xf)

    def test_exp_strides(self):
        np.random.seed(42)
        strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        sizes = np.arange(2, 100)
        for ii in sizes:
            x_f64 = np.float64(np.random.uniform(low=0.01, high=709.1, size=ii))
            y_true = np.exp(x_f64)
            for jj in strides:
                assert_array_almost_equal_nulp(np.exp(x_f64[::jj]), y_true[::jj], nulp=2)

class TestSpecialFloats:
    def test_exp_values(self):
        with np.errstate(under='raise', over='raise'):
            x = [np.nan,  np.nan, np.inf, 0.]
            y = [np.nan, -np.nan, np.inf, -np.inf]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.exp(yf), xf)

    # See: https://github.com/numpy/numpy/issues/19192
    @pytest.mark.xfail(
        _glibc_older_than("2.17"),
        reason="Older glibc versions may not raise appropriate FP exceptions"
    )
    def test_exp_exceptions(self):
        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.exp, np.float16(11.0899))
            assert_raises(FloatingPointError, np.exp, np.float32(100.))
            assert_raises(FloatingPointError, np.exp, np.float32(1E19))
            assert_raises(FloatingPointError, np.exp, np.float64(800.))
            assert_raises(FloatingPointError, np.exp, np.float64(1E19))

        with np.errstate(under='raise'):
            assert_raises(FloatingPointError, np.exp, np.float16(-17.5))
            assert_raises(FloatingPointError, np.exp, np.float32(-1000.))
            assert_raises(FloatingPointError, np.exp, np.float32(-1E19))
            assert_raises(FloatingPointError, np.exp, np.float64(-1000.))
            assert_raises(FloatingPointError, np.exp, np.float64(-1E19))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_log_values(self):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.inf, np.nan, -np.inf, np.nan]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -1.0]
            y1p = [np.nan, -np.nan, np.inf, -np.inf, -1.0, -2.0]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                yf1p = np.array(y1p, dtype=dt)
                assert_equal(np.log(yf), xf)
                assert_equal(np.log2(yf), xf)
                assert_equal(np.log10(yf), xf)
                assert_equal(np.log1p(yf1p), xf)

        with np.errstate(divide='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.log,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-1.0, dtype=dt))

        with np.errstate(invalid='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.log,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-2.0, dtype=dt))

        # See https://github.com/numpy/numpy/issues/18005
        with assert_no_warnings():
            a = np.array(1e9, dtype='float32')
            np.log(a)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype', ['e', 'f', 'd', 'g'])
    def test_sincos_values(self, dtype):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.nan, np.nan]
            y = [np.nan, -np.nan, np.inf, -np.inf]
            xf = np.array(x, dtype=dtype)
            yf = np.array(y, dtype=dtype)
            assert_equal(np.sin(yf), xf)
            assert_equal(np.cos(yf), xf)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(
        sys.platform.startswith("darwin"),
        reason="underflow is triggered for scalar 'sin'"
    )
    def test_sincos_underflow(self):
        with np.errstate(under='raise'):
            underflow_trigger = np.array(
                float.fromhex("0x1.f37f47a03f82ap-511"),
                dtype=np.float64
            )
            np.sin(underflow_trigger)
            np.cos(underflow_trigger)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('callable', [np.sin, np.cos])
    @pytest.mark.parametrize('dtype', ['e', 'f', 'd'])
    @pytest.mark.parametrize('value', [np.inf, -np.inf])
    def test_sincos_errors(self, callable, dtype, value):
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, callable,
                np.array([value], dtype=dtype))

    @pytest.mark.parametrize('callable', [np.sin, np.cos])
    @pytest.mark.parametrize('dtype', ['f', 'd'])
    @pytest.mark.parametrize('stride', [-1, 1, 2, 4, 5])
    def test_sincos_overlaps(self, callable, dtype, stride):
        N = 100
        M = N // abs(stride)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N, dtype)
        y = callable(x[::stride])
        callable(x[::stride], out=x[:M])
        assert_equal(x[:M], y)

    @pytest.mark.parametrize('dt', ['e', 'f', 'd', 'g'])
    def test_sqrt_values(self, dt):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.inf, np.nan, 0.]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.]
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_equal(np.sqrt(yf), xf)

        # with np.errstate(invalid='raise'):
        #     assert_raises(
        #         FloatingPointError, np.sqrt, np.array(-100., dtype=dt)
        #     )

    def test_abs_values(self):
        x = [np.nan,  np.nan, np.inf, np.inf, 0., 0., 1.0, 1.0]
        y = [np.nan, -np.nan, np.inf, -np.inf, 0., -0., -1.0, 1.0]
        for dt in ['e', 'f', 'd', 'g']:
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_equal(np.abs(yf), xf)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_square_values(self):
        x = [np.nan,  np.nan, np.inf, np.inf]
        y = [np.nan, -np.nan, np.inf, -np.inf]
        with np.errstate(all='ignore'):
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.square(yf), xf)

        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.square,
                          np.array(1E3, dtype='e'))
            assert_raises(FloatingPointError, np.square,
                          np.array(1E32, dtype='f'))
            assert_raises(FloatingPointError, np.square,
                          np.array(1E200, dtype='d'))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_reciprocal_values(self):
        with np.errstate(all='ignore'):
            x = [np.nan,  np.nan, 0.0, -0.0, np.inf, -np.inf]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0., -0.]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.reciprocal(yf), xf)

        with np.errstate(divide='raise'):
            for dt in ['e', 'f', 'd', 'g']:
                assert_raises(FloatingPointError, np.reciprocal,
                              np.array(-0.0, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_tan(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, 0.0, -0.0, np.inf, -np.inf]
            out = [np.nan, np.nan, 0.0, -0.0, np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.tan(in_arr), out_arr)

        with np.errstate(invalid='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.tan,
                              np.array(np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.tan,
                              np.array(-np.inf, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arcsincos(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arcsin(in_arr), out_arr)
                assert_equal(np.arccos(in_arr), out_arr)

        for callable in [np.arcsin, np.arccos]:
            for value in [np.inf, -np.inf, 2.0, -2.0]:
                for dt in ['e', 'f', 'd']:
                    with np.errstate(invalid='raise'):
                        assert_raises(FloatingPointError, callable,
                                      np.array(value, dtype=dt))

    def test_arctan(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan]
            out = [np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arctan(in_arr), out_arr)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_sinh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, -np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.sinh(in_arr), out_arr)

        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.sinh,
                          np.array(12.0, dtype='e'))
            assert_raises(FloatingPointError, np.sinh,
                          np.array(120.0, dtype='f'))
            assert_raises(FloatingPointError, np.sinh,
                          np.array(1200.0, dtype='d'))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif('bsd' in sys.platform,
            reason="fallback implementation may not raise, see gh-2487")
    def test_cosh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.cosh(in_arr), out_arr)

        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.cosh,
                          np.array(12.0, dtype='e'))
            assert_raises(FloatingPointError, np.cosh,
                          np.array(120.0, dtype='f'))
            assert_raises(FloatingPointError, np.cosh,
                          np.array(1200.0, dtype='d'))

    def test_tanh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, 1.0, -1.0]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_array_max_ulp(np.tanh(in_arr), out_arr, 3)

    def test_arcsinh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, -np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.arcsinh(in_arr), out_arr)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arccosh(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf, 1.0, 0.0]
            out = [np.nan, np.nan, np.inf, np.nan, 0.0, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arccosh(in_arr), out_arr)

        for value in [0.0, -np.inf]:
            with np.errstate(invalid='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.arccosh,
                                  np.array(value, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arctanh(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf, 1.0, -1.0, 2.0]
            out = [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arctanh(in_arr), out_arr)

        for value in [1.01, np.inf, -np.inf, 1.0, -1.0]:
            with np.errstate(invalid='raise', divide='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.arctanh,
                                  np.array(value, dtype=dt))

        # Make sure glibc < 2.18 atanh is not used, issue 25087
        assert np.signbit(np.arctanh(-1j).real)

    # See: https://github.com/numpy/numpy/issues/20448
    @pytest.mark.xfail(
        _glibc_older_than("2.17"),
        reason="Older glibc versions may not raise appropriate FP exceptions"
    )
    def test_exp2(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.inf, 0.0]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.exp2(in_arr), out_arr)

        for value in [2000.0, -2000.0]:
            with np.errstate(over='raise', under='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.exp2,
                                  np.array(value, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_expm1(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.inf, -1.0]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.expm1(in_arr), out_arr)

        for value in [200.0, 2000.0]:
            with np.errstate(over='raise'):
                for dt in ['e', 'f']:
                    assert_raises(FloatingPointError, np.expm1,
                                  np.array(value, dtype=dt))

    # test to ensure no spurious FP exceptions are raised due to SIMD
    INF_INVALID_ERR = [
        np.cos, np.sin, np.tan, np.arccos, np.arcsin, np.spacing, np.arctanh
    ]
    NEG_INVALID_ERR = [
        np.log, np.log2, np.log10, np.log1p, np.sqrt, np.arccosh,
        np.arctanh
    ]
    ONE_INVALID_ERR = [
        np.arctanh,
    ]
    LTONE_INVALID_ERR = [
        np.arccosh,
    ]
    BYZERO_ERR = [
        np.log, np.log2, np.log10, np.reciprocal, np.arccosh
    ]

    @pytest.mark.parametrize("ufunc", UFUNCS_UNARY_FP)
    @pytest.mark.parametrize("dtype", ('e', 'f', 'd'))
    @pytest.mark.parametrize("data, escape", (
        ([0.03], LTONE_INVALID_ERR),
        ([0.03] * 32, LTONE_INVALID_ERR),
        # neg
        ([-1.0], NEG_INVALID_ERR),
        ([-1.0] * 32, NEG_INVALID_ERR),
        # flat
        ([1.0], ONE_INVALID_ERR),
        ([1.0] * 32, ONE_INVALID_ERR),
        # zero
        ([0.0], BYZERO_ERR),
        ([0.0] * 32, BYZERO_ERR),
        ([-0.0], BYZERO_ERR),
        ([-0.0] * 32, BYZERO_ERR),
        # nan
        ([0.5, 0.5, 0.5, np.nan], LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, np.nan] * 32, LTONE_INVALID_ERR),
        ([np.nan, 1.0, 1.0, 1.0], ONE_INVALID_ERR),
        ([np.nan, 1.0, 1.0, 1.0] * 32, ONE_INVALID_ERR),
        ([np.nan], []),
        ([np.nan] * 32, []),
        # inf
        ([0.5, 0.5, 0.5, np.inf], INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, np.inf] * 32, INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([np.inf, 1.0, 1.0, 1.0], INF_INVALID_ERR),
        ([np.inf, 1.0, 1.0, 1.0] * 32, INF_INVALID_ERR),
        ([np.inf], INF_INVALID_ERR),
        ([np.inf] * 32, INF_INVALID_ERR),
        # ninf
        ([0.5, 0.5, 0.5, -np.inf],
         NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, -np.inf] * 32,
         NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([-np.inf, 1.0, 1.0, 1.0], NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf, 1.0, 1.0, 1.0] * 32, NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf], NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf] * 32, NEG_INVALID_ERR + INF_INVALID_ERR),
    ))
    def test_unary_spurious_fpexception(self, ufunc, dtype, data, escape):
        if escape and ufunc in escape:
            return
        # FIXME: NAN raises FP invalid exception:
        #  - ceil/float16 on MSVC:32-bit
        #  - spacing/float16 on almost all platforms
        #  - spacing all floats on MSVC vs2022
        if ufunc == np.spacing:
            return
        if ufunc == np.ceil and dtype == 'e':
            return
        array = np.array(data, dtype=dtype)
        with assert_no_warnings():
            ufunc(array)

    @pytest.mark.parametrize("dtype", ('e', 'f', 'd'))
    def test_divide_spurious_fpexception(self, dtype):
        dt = np.dtype(dtype)
        dt_info = np.finfo(dt)
        subnorm = dt_info.smallest_subnormal
        # Verify a bug fix caused due to filling the remaining lanes of the
        # partially loaded dividend SIMD vector with ones, which leads to
        # raising an overflow warning when the divisor is denormal.
        # see https://github.com/numpy/numpy/issues/25097
        with assert_no_warnings():
            np.zeros(128 + 1, dtype=dt) / subnorm

class TestFPClass:
    @pytest.mark.parametrize("stride", [-5, -4, -3, -2, -1, 1,
                                2, 4, 5, 6, 7, 8, 9, 10])
    def test_fpclass(self, stride):
        arr_f64 = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.2251e-308], dtype='d')
        arr_f32 = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 1.4013e-045, -1.4013e-045], dtype='f')
        nan     = np.array([True, True, False, False, False, False, False, False, False, False])  # noqa: E221
        inf     = np.array([False, False, True, True, False, False, False, False, False, False])  # noqa: E221
        sign    = np.array([False, True, False, True, True, False, True, False, False, True])     # noqa: E221
        finite  = np.array([False, False, False, False, True, True, True, True, True, True])      # noqa: E221
        assert_equal(np.isnan(arr_f32[::stride]), nan[::stride])
        assert_equal(np.isnan(arr_f64[::stride]), nan[::stride])
        assert_equal(np.isinf(arr_f32[::stride]), inf[::stride])
        assert_equal(np.isinf(arr_f64[::stride]), inf[::stride])
        if platform.machine() == 'riscv64':
            # On RISC-V, many operations that produce NaNs, such as converting
            # a -NaN from f64 to f32, return a canonical NaN.  The canonical
            # NaNs are always positive.  See section 11.3 NaN Generation and
            # Propagation of the RISC-V Unprivileged ISA for more details.
            # We disable the sign test on riscv64 for -np.nan as we
            # cannot assume that its sign will be honoured in these tests.
            arr_f64_rv = np.copy(arr_f64)
            arr_f32_rv = np.copy(arr_f32)
            arr_f64_rv[1] = -1.0
            arr_f32_rv[1] = -1.0
            assert_equal(np.signbit(arr_f32_rv[::stride]), sign[::stride])
            assert_equal(np.signbit(arr_f64_rv[::stride]), sign[::stride])
        else:
            assert_equal(np.signbit(arr_f32[::stride]), sign[::stride])
            assert_equal(np.signbit(arr_f64[::stride]), sign[::stride])
        assert_equal(np.isfinite(arr_f32[::stride]), finite[::stride])
        assert_equal(np.isfinite(arr_f64[::stride]), finite[::stride])

    @pytest.mark.parametrize("dtype", ['d', 'f'])
    def test_fp_noncontiguous(self, dtype):
        data = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0,
                            1.0, -0.0, 0.0, 2.2251e-308,
                            -2.2251e-308], dtype=dtype)
        nan = np.array([True, True, False, False, False, False,
                            False, False, False, False])
        inf = np.array([False, False, True, True, False, False,
                            False, False, False, False])
        sign = np.array([False, True, False, True, True, False,
                            True, False, False, True])
        finite = np.array([False, False, False, False, True, True,
                            True, True, True, True])
        out = np.ndarray(data.shape, dtype='bool')
        ncontig_in = data[1::3]
        ncontig_out = out[1::3]
        contig_in = np.array(ncontig_in)

        if platform.machine() == 'riscv64':
            # Disable the -np.nan signbit tests on riscv64.  See comments in
            # test_fpclass for more details.
            data_rv = np.copy(data)
            data_rv[1] = -1.0
            ncontig_sign_in = data_rv[1::3]
            contig_sign_in = np.array(ncontig_sign_in)
        else:
            ncontig_sign_in = ncontig_in
            contig_sign_in = contig_in

        assert_equal(ncontig_in.flags.c_contiguous, False)
        assert_equal(ncontig_out.flags.c_contiguous, False)
        assert_equal(contig_in.flags.c_contiguous, True)
        assert_equal(ncontig_sign_in.flags.c_contiguous, False)
        assert_equal(contig_sign_in.flags.c_contiguous, True)
        # ncontig in, ncontig out
        assert_equal(np.isnan(ncontig_in, out=ncontig_out), nan[1::3])
        assert_equal(np.isinf(ncontig_in, out=ncontig_out), inf[1::3])
        assert_equal(np.signbit(ncontig_sign_in, out=ncontig_out), sign[1::3])
        assert_equal(np.isfinite(ncontig_in, out=ncontig_out), finite[1::3])
        # contig in, ncontig out
        assert_equal(np.isnan(contig_in, out=ncontig_out), nan[1::3])
        assert_equal(np.isinf(contig_in, out=ncontig_out), inf[1::3])
        assert_equal(np.signbit(contig_sign_in, out=ncontig_out), sign[1::3])
        assert_equal(np.isfinite(contig_in, out=ncontig_out), finite[1::3])
        # ncontig in, contig out
        assert_equal(np.isnan(ncontig_in), nan[1::3])
        assert_equal(np.isinf(ncontig_in), inf[1::3])
        assert_equal(np.signbit(ncontig_sign_in), sign[1::3])
        assert_equal(np.isfinite(ncontig_in), finite[1::3])
        # contig in, contig out, nd stride
        data_split = np.array(np.array_split(data, 2))
        nan_split = np.array(np.array_split(nan, 2))
        inf_split = np.array(np.array_split(inf, 2))
        sign_split = np.array(np.array_split(sign, 2))
        finite_split = np.array(np.array_split(finite, 2))
        assert_equal(np.isnan(data_split), nan_split)
        assert_equal(np.isinf(data_split), inf_split)
        if platform.machine() == 'riscv64':
            data_split_rv = np.array(np.array_split(data_rv, 2))
            assert_equal(np.signbit(data_split_rv), sign_split)
        else:
            assert_equal(np.signbit(data_split), sign_split)
        assert_equal(np.isfinite(data_split), finite_split)

class TestLDExp:
    @pytest.mark.parametrize("stride", [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize("dtype", ['f', 'd'])
    def test_ldexp(self, dtype, stride):
        mant = np.array([0.125, 0.25, 0.5, 1., 1., 2., 4., 8.], dtype=dtype)
        exp = np.array([3, 2, 1, 0, 0, -1, -2, -3], dtype='i')
        out = np.zeros(8, dtype=dtype)
        assert_equal(np.ldexp(mant[::stride], exp[::stride], out=out[::stride]), np.ones(8, dtype=dtype)[::stride])
        assert_equal(out[::stride], np.ones(8, dtype=dtype)[::stride])

class TestFRExp:
    @pytest.mark.parametrize("stride", [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize("dtype", ['f', 'd'])
    @pytest.mark.skipif(not sys.platform.startswith('linux'),
                        reason="np.frexp gives different answers for NAN/INF on windows and linux")
    @pytest.mark.xfail(IS_MUSL, reason="gh23049")
    def test_frexp(self, dtype, stride):
        arr = np.array([np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0], dtype=dtype)
        mant_true = np.array([np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0, 0.5, -0.5], dtype=dtype)
        exp_true = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype='i')
        out_mant = np.ones(8, dtype=dtype)
        out_exp = 2 * np.ones(8, dtype='i')
        mant, exp = np.frexp(arr[::stride], out=(out_mant[::stride], out_exp[::stride]))
        assert_equal(mant_true[::stride], mant)
        assert_equal(exp_true[::stride], exp)
        assert_equal(out_mant[::stride], mant_true[::stride])
        assert_equal(out_exp[::stride], exp_true[::stride])


# func : [maxulperror, low, high]
avx_ufuncs = {'sqrt'        : [1,  0.,   100.],   # noqa: E203
              'absolute'    : [0, -100., 100.],   # noqa: E203
              'reciprocal'  : [1,  1.,   100.],   # noqa: E203
              'square'      : [1, -100., 100.],   # noqa: E203
              'rint'        : [0, -100., 100.],   # noqa: E203
              'floor'       : [0, -100., 100.],   # noqa: E203
              'ceil'        : [0, -100., 100.],   # noqa: E203
              'trunc'       : [0, -100., 100.]}   # noqa: E203

class TestAVXUfuncs:
    def test_avx_based_ufunc(self):
        strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        np.random.seed(42)
        for func, prop in avx_ufuncs.items():
            maxulperr = prop[0]
            minval = prop[1]
            maxval = prop[2]
            # various array sizes to ensure masking in AVX is tested
            for size in range(1, 32):
                myfunc = getattr(np, func)
                x_f32 = np.random.uniform(low=minval, high=maxval,
                                          size=size).astype(np.float32)
                x_f64 = x_f32.astype(np.float64)
                x_f128 = x_f32.astype(np.longdouble)
                y_true128 = myfunc(x_f128)
                if maxulperr == 0:
                    assert_equal(myfunc(x_f32), y_true128.astype(np.float32))
                    assert_equal(myfunc(x_f64), y_true128.astype(np.float64))
                else:
                    assert_array_max_ulp(myfunc(x_f32),
                                         y_true128.astype(np.float32),
                                         maxulp=maxulperr)
                    assert_array_max_ulp(myfunc(x_f64),
                                         y_true128.astype(np.float64),
                                         maxulp=maxulperr)
                # various strides to test gather instruction
                if size > 1:
                    y_true32 = myfunc(x_f32)
                    y_true64 = myfunc(x_f64)
                    for jj in strides:
                        assert_equal(myfunc(x_f64[::jj]), y_true64[::jj])
                        assert_equal(myfunc(x_f32[::jj]), y_true32[::jj])

class TestAVXFloat32Transcendental:
    def test_exp_float32(self):
        np.random.seed(42)
        x_f32 = np.float32(np.random.uniform(low=0.0, high=88.1, size=1000000))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.exp(x_f32), np.float32(np.exp(x_f64)), maxulp=3)

    def test_log_float32(self):
        np.random.seed(42)
        x_f32 = np.float32(np.random.uniform(low=0.0, high=1000, size=1000000))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.log(x_f32), np.float32(np.log(x_f64)), maxulp=4)

    def test_sincos_float32(self):
        np.random.seed(42)
        N = 1000000
        M = np.int_(N / 20)
        index = np.random.randint(low=0, high=N, size=M)
        x_f32 = np.float32(np.random.uniform(low=-100., high=100., size=N))
        if not _glibc_older_than("2.17"):
            # test coverage for elements > 117435.992f for which glibc is used
            # this is known to be problematic on old glibc, so skip it there
            x_f32[index] = np.float32(10E+10 * np.random.rand(M))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.sin(x_f32), np.float32(np.sin(x_f64)), maxulp=2)
        assert_array_max_ulp(np.cos(x_f32), np.float32(np.cos(x_f64)), maxulp=2)
        # test aliasing(issue #17761)
        tx_f32 = x_f32.copy()
        assert_array_max_ulp(np.sin(x_f32, out=x_f32), np.float32(np.sin(x_f64)), maxulp=2)
        assert_array_max_ulp(np.cos(tx_f32, out=tx_f32), np.float32(np.cos(x_f64)), maxulp=2)

    def test_strided_float32(self):
        np.random.seed(42)
        strides = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        sizes = np.arange(2, 100)
        for ii in sizes:
            x_f32 = np.float32(np.random.uniform(low=0.01, high=88.1, size=ii))
            x_f32_large = x_f32.copy()
            x_f32_large[3:-1:4] = 120000.0
            exp_true = np.exp(x_f32)
            log_true = np.log(x_f32)
            sin_true = np.sin(x_f32_large)
            cos_true = np.cos(x_f32_large)
            for jj in strides:
                assert_array_almost_equal_nulp(np.exp(x_f32[::jj]), exp_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.log(x_f32[::jj]), log_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.sin(x_f32_large[::jj]), sin_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.cos(x_f32_large[::jj]), cos_true[::jj], nulp=2)

class TestLogAddExp(_FilterInvalids):
    def test_logaddexp_values(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for dt, dec_ in zip(['f', 'd', 'g'], [6, 15, 15]):
            xf = np.log(np.array(x, dtype=dt))
            yf = np.log(np.array(y, dtype=dt))
            zf = np.log(np.array(z, dtype=dt))
            assert_almost_equal(np.logaddexp(xf, yf), zf, decimal=dec_)

    def test_logaddexp_range(self):
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for dt in ['f', 'd', 'g']:
            logxf = np.array(x, dtype=dt)
            logyf = np.array(y, dtype=dt)
            logzf = np.array(z, dtype=dt)
            assert_almost_equal(np.logaddexp(logxf, logyf), logzf)

    def test_inf(self):
        inf = np.inf
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]    # noqa: E221
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]  # noqa: E221
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        with np.errstate(invalid='raise'):
            for dt in ['f', 'd', 'g']:
                logxf = np.array(x, dtype=dt)
                logyf = np.array(y, dtype=dt)
                logzf = np.array(z, dtype=dt)
                assert_equal(np.logaddexp(logxf, logyf), logzf)

    def test_nan(self):
        assert_(np.isnan(np.logaddexp(np.nan, np.inf)))
        assert_(np.isnan(np.logaddexp(np.inf, np.nan)))
        assert_(np.isnan(np.logaddexp(np.nan, 0)))
        assert_(np.isnan(np.logaddexp(0, np.nan)))
        assert_(np.isnan(np.logaddexp(np.nan, np.nan)))

    def test_reduce(self):
        assert_equal(np.logaddexp.identity, -np.inf)
        assert_equal(np.logaddexp.reduce([]), -np.inf)


class TestLog1p:
    def test_log1p(self):
        assert_almost_equal(ncu.log1p(0.2), ncu.log(1.2))
        assert_almost_equal(ncu.log1p(1e-6), ncu.log(1 + 1e-6))

    def test_special(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            assert_equal(ncu.log1p(np.nan), np.nan)
            assert_equal(ncu.log1p(np.inf), np.inf)
            assert_equal(ncu.log1p(-1.), -np.inf)
            assert_equal(ncu.log1p(-2.), np.nan)
            assert_equal(ncu.log1p(-np.inf), np.nan)


class TestExpm1:
    def test_expm1(self):
        assert_almost_equal(ncu.expm1(0.2), ncu.exp(0.2) - 1)
        assert_almost_equal(ncu.expm1(1e-6), ncu.exp(1e-6) - 1)

    def test_special(self):
        assert_equal(ncu.expm1(np.inf), np.inf)
        assert_equal(ncu.expm1(0.), 0.)
        assert_equal(ncu.expm1(-0.), -0.)
        assert_equal(ncu.expm1(np.inf), np.inf)
        assert_equal(ncu.expm1(-np.inf), -1.)

    def test_complex(self):
        x = np.asarray(1e-12)
        assert_allclose(x, ncu.expm1(x))
        x = x.astype(np.complex128)
        assert_allclose(x, ncu.expm1(x))


class TestHypot:
    def test_simple(self):
        assert_almost_equal(ncu.hypot(1, 1), ncu.sqrt(2))
        assert_almost_equal(ncu.hypot(0, 0), 0)

    def test_reduce(self):
        assert_almost_equal(ncu.hypot.reduce([3.0, 4.0]), 5.0)
        assert_almost_equal(ncu.hypot.reduce([3.0, 4.0, 0]), 5.0)
        assert_almost_equal(ncu.hypot.reduce([9.0, 12.0, 20.0]), 25.0)
        assert_equal(ncu.hypot.reduce([]), 0.0)


def assert_hypot_isnan(x, y):
    with np.errstate(invalid='ignore'):
        assert_(np.isnan(ncu.hypot(x, y)),
                f"hypot({x}, {y}) is {ncu.hypot(x, y)}, not nan")


def assert_hypot_isinf(x, y):
    with np.errstate(invalid='ignore'):
        assert_(np.isinf(ncu.hypot(x, y)),
                f"hypot({x}, {y}) is {ncu.hypot(x, y)}, not inf")


class TestHypotSpecialValues:
    def test_nan_outputs(self):
        assert_hypot_isnan(np.nan, np.nan)
        assert_hypot_isnan(np.nan, 1)

    def test_nan_outputs2(self):
        assert_hypot_isinf(np.nan, np.inf)
        assert_hypot_isinf(np.inf, np.nan)
        assert_hypot_isinf(np.inf, 0)
        assert_hypot_isinf(0, np.inf)
        assert_hypot_isinf(np.inf, np.inf)
        assert_hypot_isinf(np.inf, 23.0)

    def test_no_fpe(self):
        assert_no_warnings(ncu.hypot, np.inf, 0)


def assert_arctan2_isnan(x, y):
    assert_(np.isnan(ncu.arctan2(x, y)), f"arctan({x}, {y}) is {ncu.arctan2(x, y)}, not nan")


def assert_arctan2_ispinf(x, y):
    assert_((np.isinf(ncu.arctan2(x, y)) and ncu.arctan2(x, y) > 0), f"arctan({x}, {y}) is {ncu.arctan2(x, y)}, not +inf")


def assert_arctan2_isninf(x, y):
    assert_((np.isinf(ncu.arctan2(x, y)) and ncu.arctan2(x, y) < 0), f"arctan({x}, {y}) is {ncu.arctan2(x, y)}, not -inf")


def assert_arctan2_ispzero(x, y):
    assert_((ncu.arctan2(x, y) == 0 and not np.signbit(ncu.arctan2(x, y))), f"arctan({x}, {y}) is {ncu.arctan2(x, y)}, not +0")


def assert_arctan2_isnzero(x, y):
    assert_((ncu.arctan2(x, y) == 0 and np.signbit(ncu.arctan2(x, y))), f"arctan({x}, {y}) is {ncu.arctan2(x, y)}, not -0")


class TestArctan2SpecialValues:
    def test_one_one(self):
        # atan2(1, 1) returns pi/4.
        assert_almost_equal(ncu.arctan2(1, 1), 0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(-1, 1), -0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(1, -1), 0.75 * np.pi)

    def test_zero_nzero(self):
        # atan2(+-0, -0) returns +-pi.
        assert_almost_equal(ncu.arctan2(ncu.PZERO, ncu.NZERO), np.pi)
        assert_almost_equal(ncu.arctan2(ncu.NZERO, ncu.NZERO), -np.pi)

    def test_zero_pzero(self):
        # atan2(+-0, +0) returns +-0.
        assert_arctan2_ispzero(ncu.PZERO, ncu.PZERO)
        assert_arctan2_isnzero(ncu.NZERO, ncu.PZERO)

    def test_zero_negative(self):
        # atan2(+-0, x) returns +-pi for x < 0.
        assert_almost_equal(ncu.arctan2(ncu.PZERO, -1), np.pi)
        assert_almost_equal(ncu.arctan2(ncu.NZERO, -1), -np.pi)

    def test_zero_positive(self):
        # atan2(+-0, x) returns +-0 for x > 0.
        assert_arctan2_ispzero(ncu.PZERO, 1)
        assert_arctan2_isnzero(ncu.NZERO, 1)

    def test_positive_zero(self):
        # atan2(y, +-0) returns +pi/2 for y > 0.
        assert_almost_equal(ncu.arctan2(1, ncu.PZERO), 0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(1, ncu.NZERO), 0.5 * np.pi)

    def test_negative_zero(self):
        # atan2(y, +-0) returns -pi/2 for y < 0.
        assert_almost_equal(ncu.arctan2(-1, ncu.PZERO), -0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(-1, ncu.NZERO), -0.5 * np.pi)

    def test_any_ninf(self):
        # atan2(+-y, -infinity) returns +-pi for finite y > 0.
        assert_almost_equal(ncu.arctan2(1, -np.inf),  np.pi)
        assert_almost_equal(ncu.arctan2(-1, -np.inf), -np.pi)

    def test_any_pinf(self):
        # atan2(+-y, +infinity) returns +-0 for finite y > 0.
        assert_arctan2_ispzero(1, np.inf)
        assert_arctan2_isnzero(-1, np.inf)

    def test_inf_any(self):
        # atan2(+-infinity, x) returns +-pi/2 for finite x.
        assert_almost_equal(ncu.arctan2( np.inf, 1),  0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, 1), -0.5 * np.pi)

    def test_inf_ninf(self):
        # atan2(+-infinity, -infinity) returns +-3*pi/4.
        assert_almost_equal(ncu.arctan2( np.inf, -np.inf),  0.75 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, -np.inf), -0.75 * np.pi)

    def test_inf_pinf(self):
        # atan2(+-infinity, +infinity) returns +-pi/4.
        assert_almost_equal(ncu.arctan2( np.inf, np.inf),  0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, np.inf), -0.25 * np.pi)

    def test_nan_any(self):
        # atan2(nan, x) returns nan for any x, including inf
        assert_arctan2_isnan(np.nan, np.inf)
        assert_arctan2_isnan(np.inf, np.nan)
        assert_arctan2_isnan(np.nan, np.nan)


class TestLdexp:
    def _check_ldexp(self, tp):
        assert_almost_equal(ncu.ldexp(np.array(2., np.float32),
                                      np.array(3, tp)), 16.)
        assert_almost_equal(ncu.ldexp(np.array(2., np.float64),
                                      np.array(3, tp)), 16.)
        assert_almost_equal(ncu.ldexp(np.array(2., np.longdouble),
                                      np.array(3, tp)), 16.)

    def test_ldexp(self):
        # The default Python int type should work
        assert_almost_equal(ncu.ldexp(2., 3),  16.)
        # The following int types should all be accepted
        self._check_ldexp(np.int8)
        self._check_ldexp(np.int16)
        self._check_ldexp(np.int32)
        self._check_ldexp('i')
        self._check_ldexp('l')

    def test_ldexp_overflow(self):
        # silence warning emitted on overflow
        with np.errstate(over="ignore"):
            imax = np.iinfo(np.dtype('l')).max
            imin = np.iinfo(np.dtype('l')).min
            assert_equal(ncu.ldexp(2., imax), np.inf)
            assert_equal(ncu.ldexp(2., imin), 0)


class TestMaximum(_FilterInvalids):
    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.maximum.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), np.nan)
            assert_equal(func(tmp2), np.nan)

    def test_reduce_complex(self):
        assert_equal(np.maximum.reduce([1, 2j]), 1)
        assert_equal(np.maximum.reduce([1 + 3j, 2j]), 1 + 3j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([nan, nan, nan])
        assert_equal(np.maximum(arg1, arg2), out)

    def test_object_nans(self):
        # Multiple checks to give this a chance to
        # fail if cmp is used instead of rich compare.
        # Failure cannot be guaranteed.
        for i in range(1):
            x = np.array(float('nan'), object)
            y = 1.0
            z = np.array(float('nan'), object)
            assert_(np.maximum(x, y) == 1.0)
            assert_(np.maximum(z, y) == 1.0)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([nan, nan, nan], dtype=complex)
            assert_equal(np.maximum(arg1, arg2), out)

    def test_object_array(self):
        arg1 = np.arange(5, dtype=object)
        arg2 = arg1 + 1
        assert_equal(np.maximum(arg1, arg2), arg2)

    def test_strided_array(self):
        arr1 = np.array([-4.0,  1.0, 10.0,   0.0, np.nan, -np.nan, np.inf, -np.inf])
        arr2 = np.array([-2.0, -1.0, np.nan, 1.0, 0.0,     np.nan, 1.0,    -3.0])  # noqa: E221
        maxtrue = np.array([-2.0, 1.0, np.nan, 1.0, np.nan, np.nan, np.inf, -3.0])
        out = np.ones(8)
        out_maxtrue = np.array([-2.0, 1.0, 1.0, 10.0, 1.0, 1.0, np.nan, 1.0])
        assert_equal(np.maximum(arr1, arr2), maxtrue)
        assert_equal(np.maximum(arr1[::2], arr2[::2]), maxtrue[::2])
        assert_equal(np.maximum(arr1[:4:], arr2[::2]), np.array([-2.0, np.nan, 10.0, 1.0]))
        assert_equal(np.maximum(arr1[::3], arr2[:3:]), np.array([-2.0, 0.0, np.nan]))
        assert_equal(np.maximum(arr1[:6:2], arr2[::3], out=out[::3]), np.array([-2.0, 10., np.nan]))
        assert_equal(out, out_maxtrue)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            test_cases = [
                # v1    v2          expected
                (dtmin, -np.inf,    dtmin),
                (dtmax, -np.inf,    dtmax),
                (d1,    d1_next,    d1_next),
                (dtmax, np.nan,     np.nan),
            ]

            for v1, v2, expected in test_cases:
                assert_equal(np.maximum([v1], [v2]), [expected])
                assert_equal(np.maximum.reduce([v1, v2]), expected)


class TestMinimum(_FilterInvalids):
    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.minimum.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), np.nan)
            assert_equal(func(tmp2), np.nan)

    def test_reduce_complex(self):
        assert_equal(np.minimum.reduce([1, 2j]), 2j)
        assert_equal(np.minimum.reduce([1 + 3j, 2j]), 2j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([nan, nan, nan])
        assert_equal(np.minimum(arg1, arg2), out)

    def test_object_nans(self):
        # Multiple checks to give this a chance to
        # fail if cmp is used instead of rich compare.
        # Failure cannot be guaranteed.
        for i in range(1):
            x = np.array(float('nan'), object)
            y = 1.0
            z = np.array(float('nan'), object)
            assert_(np.minimum(x, y) == 1.0)
            assert_(np.minimum(z, y) == 1.0)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([nan, nan, nan], dtype=complex)
            assert_equal(np.minimum(arg1, arg2), out)

    def test_object_array(self):
        arg1 = np.arange(5, dtype=object)
        arg2 = arg1 + 1
        assert_equal(np.minimum(arg1, arg2), arg1)

    def test_strided_array(self):
        arr1 = np.array([-4.0, 1.0, 10.0,  0.0, np.nan, -np.nan, np.inf, -np.inf])
        arr2 = np.array([-2.0, -1.0, np.nan, 1.0, 0.0,    np.nan, 1.0, -3.0])
        mintrue = np.array([-4.0, -1.0, np.nan, 0.0, np.nan, np.nan, 1.0, -np.inf])
        out = np.ones(8)
        out_mintrue = np.array([-4.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0])
        assert_equal(np.minimum(arr1, arr2), mintrue)
        assert_equal(np.minimum(arr1[::2], arr2[::2]), mintrue[::2])
        assert_equal(np.minimum(arr1[:4:], arr2[::2]), np.array([-4.0, np.nan, 0.0, 0.0]))
        assert_equal(np.minimum(arr1[::3], arr2[:3:]), np.array([-4.0, -1.0, np.nan]))
        assert_equal(np.minimum(arr1[:6:2], arr2[::3], out=out[::3]), np.array([-4.0, 1.0, np.nan]))
        assert_equal(out, out_mintrue)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            test_cases = [
                # v1    v2          expected
                (dtmin, np.inf,     dtmin),
                (dtmax, np.inf,     dtmax),
                (d1,    d1_next,    d1),
                (dtmin, np.nan,     np.nan),
            ]

            for v1, v2, expected in test_cases:
                assert_equal(np.minimum([v1], [v2]), [expected])
                assert_equal(np.minimum.reduce([v1, v2]), expected)


class TestFmax(_FilterInvalids):
    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.fmax.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), 9)
            assert_equal(func(tmp2), 9)

    def test_reduce_complex(self):
        assert_equal(np.fmax.reduce([1, 2j]), 1)
        assert_equal(np.fmax.reduce([1 + 3j, 2j]), 1 + 3j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([0,   0,   nan])
        assert_equal(np.fmax(arg1, arg2), out)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([0,    0, nan], dtype=complex)
            assert_equal(np.fmax(arg1, arg2), out)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            test_cases = [
                # v1    v2          expected
                (dtmin, -np.inf,    dtmin),
                (dtmax, -np.inf,    dtmax),
                (d1,    d1_next,    d1_next),
                (dtmax, np.nan,     dtmax),
            ]

            for v1, v2, expected in test_cases:
                assert_equal(np.fmax([v1], [v2]), [expected])
                assert_equal(np.fmax.reduce([v1, v2]), expected)


class TestFmin(_FilterInvalids):
    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.fmin.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), 1)
            assert_equal(func(tmp2), 1)

    def test_reduce_complex(self):
        assert_equal(np.fmin.reduce([1, 2j]), 2j)
        assert_equal(np.fmin.reduce([1 + 3j, 2j]), 2j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([0,   0,   nan])
        assert_equal(np.fmin(arg1, arg2), out)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([0,    0, nan], dtype=complex)
            assert_equal(np.fmin(arg1, arg2), out)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            test_cases = [
                # v1    v2          expected
                (dtmin, np.inf,     dtmin),
                (dtmax, np.inf,     dtmax),
                (d1,    d1_next,    d1),
                (dtmin, np.nan,     dtmin),
            ]

            for v1, v2, expected in test_cases:
                assert_equal(np.fmin([v1], [v2]), [expected])
                assert_equal(np.fmin.reduce([v1, v2]), expected)


class TestBool:
    def test_exceptions(self):
        a = np.ones(1, dtype=np.bool)
        assert_raises(TypeError, np.negative, a)
        assert_raises(TypeError, np.positive, a)
        assert_raises(TypeError, np.subtract, a, a)

    def test_truth_table_logical(self):
        # 2, 3 and 4 serves as true values
        input1 = [0, 0, 3, 2]
        input2 = [0, 4, 0, 2]

        typecodes = (np.typecodes['AllFloat']
                     + np.typecodes['AllInteger']
                     + '?')     # boolean
        for dtype in map(np.dtype, typecodes):
            arg1 = np.asarray(input1, dtype=dtype)
            arg2 = np.asarray(input2, dtype=dtype)

            # OR
            out = [False, True, True, True]
            for func in (np.logical_or, np.maximum):
                assert_equal(func(arg1, arg2).astype(bool), out)
            # AND
            out = [False, False, False, True]
            for func in (np.logical_and, np.minimum):
                assert_equal(func(arg1, arg2).astype(bool), out)
            # XOR
            out = [False, True, True, False]
            for func in (np.logical_xor, np.not_equal):
                assert_equal(func(arg1, arg2).astype(bool), out)

    def test_truth_table_bitwise(self):
        arg1 = [False, False, True, True]
        arg2 = [False, True, False, True]

        out = [False, True, True, True]
        assert_equal(np.bitwise_or(arg1, arg2), out)

        out = [False, False, False, True]
        assert_equal(np.bitwise_and(arg1, arg2), out)

        out = [False, True, True, False]
        assert_equal(np.bitwise_xor(arg1, arg2), out)

    def test_reduce(self):
        none = np.array([0, 0, 0, 0], bool)
        some = np.array([1, 0, 1, 1], bool)
        every = np.array([1, 1, 1, 1], bool)
        empty = np.array([], bool)

        arrs = [none, some, every, empty]

        for arr in arrs:
            assert_equal(np.logical_and.reduce(arr), all(arr))

        for arr in arrs:
            assert_equal(np.logical_or.reduce(arr), any(arr))

        for arr in arrs:
            assert_equal(np.logical_xor.reduce(arr), arr.sum() % 2 == 1)


class TestBitwiseUFuncs:

    _all_ints_bits = [
        np.dtype(c).itemsize * 8 for c in np.typecodes["AllInteger"]]
    bitwise_types = [
        np.dtype(c) for c in '?' + np.typecodes["AllInteger"] + 'O']
    bitwise_bits = [
        2,  # boolean type
        *_all_ints_bits,  # All integers
        max(_all_ints_bits) + 1,  # Object_ type
    ]

    def test_values(self):
        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            msg = f"dt = '{dt.char}'"

            assert_equal(np.bitwise_not(zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_not(ones), zeros, err_msg=msg)

            assert_equal(np.bitwise_or(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_or(zeros, ones), ones, err_msg=msg)
            assert_equal(np.bitwise_or(ones, zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_or(ones, ones), ones, err_msg=msg)

            assert_equal(np.bitwise_xor(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_xor(zeros, ones), ones, err_msg=msg)
            assert_equal(np.bitwise_xor(ones, zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_xor(ones, ones), zeros, err_msg=msg)

            assert_equal(np.bitwise_and(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(zeros, ones), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(ones, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(ones, ones), ones, err_msg=msg)

    def test_types(self):
        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            msg = f"dt = '{dt.char}'"

            assert_(np.bitwise_not(zeros).dtype == dt, msg)
            assert_(np.bitwise_or(zeros, zeros).dtype == dt, msg)
            assert_(np.bitwise_xor(zeros, zeros).dtype == dt, msg)
            assert_(np.bitwise_and(zeros, zeros).dtype == dt, msg)

    def test_identity(self):
        assert_(np.bitwise_or.identity == 0, 'bitwise_or')
        assert_(np.bitwise_xor.identity == 0, 'bitwise_xor')
        assert_(np.bitwise_and.identity == -1, 'bitwise_and')

    def test_reduction(self):
        binary_funcs = (np.bitwise_or, np.bitwise_xor, np.bitwise_and)

        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            for f in binary_funcs:
                msg = f"dt: '{dt}', f: '{f}'"
                assert_equal(f.reduce(zeros), zeros, err_msg=msg)
                assert_equal(f.reduce(ones), ones, err_msg=msg)

        # Test empty reduction, no object dtype
        for dt in self.bitwise_types[:-1]:
            # No object array types
            empty = np.array([], dtype=dt)
            for f in binary_funcs:
                msg = f"dt: '{dt}', f: '{f}'"
                tgt = np.array(f.identity).astype(dt)
                res = f.reduce(empty)
                assert_equal(res, tgt, err_msg=msg)
                assert_(res.dtype == tgt.dtype, msg)

        # Empty object arrays use the identity.  Note that the types may
        # differ, the actual type used is determined by the assign_identity
        # function and is not the same as the type returned by the identity
        # method.
        for f in binary_funcs:
            msg = f"dt: '{f}'"
            empty = np.array([], dtype=object)
            tgt = f.identity
            res = f.reduce(empty)
            assert_equal(res, tgt, err_msg=msg)

        # Non-empty object arrays do not use the identity
        for f in binary_funcs:
            msg = f"dt: '{f}'"
            btype = np.array([True], dtype=object)
            assert_(type(f.reduce(btype)) is bool, msg)

    @pytest.mark.parametrize("input_dtype_obj, bitsize",
            zip(bitwise_types, bitwise_bits))
    def test_bitwise_count(self, input_dtype_obj, bitsize):
        input_dtype = input_dtype_obj.type

        for i in range(1, bitsize):
            num = 2**i - 1
            msg = f"bitwise_count for {num}"
            assert i == np.bitwise_count(input_dtype(num)), msg
            if np.issubdtype(
                input_dtype, np.signedinteger) or input_dtype == np.object_:
                assert i == np.bitwise_count(input_dtype(-num)), msg

        a = np.array([2**i - 1 for i in range(1, bitsize)], dtype=input_dtype)
        bitwise_count_a = np.bitwise_count(a)
        expected = np.arange(1, bitsize, dtype=input_dtype)

        msg = f"array bitwise_count for {input_dtype}"
        assert all(bitwise_count_a == expected), msg


class TestInt:
    def test_logical_not(self):
        x = np.ones(10, dtype=np.int16)
        o = np.ones(10 * 2, dtype=bool)
        tgt = o.copy()
        tgt[::2] = False
        os = o[::2]
        assert_array_equal(np.logical_not(x, out=os), False)
        assert_array_equal(o, tgt)


class TestFloatingPoint:
    def test_floating_point(self):
        assert_equal(ncu.FLOATING_POINT_SUPPORT, 1)


class TestDegrees:
    def test_degrees(self):
        assert_almost_equal(ncu.degrees(np.pi), 180.0)
        assert_almost_equal(ncu.degrees(-0.5 * np.pi), -90.0)


class TestRadians:
    def test_radians(self):
        assert_almost_equal(ncu.radians(180.0), np.pi)
        assert_almost_equal(ncu.radians(-90.0), -0.5 * np.pi)


class TestHeavside:
    def test_heaviside(self):
        x = np.array([[-30.0, -0.1, 0.0, 0.2], [7.5, np.nan, np.inf, -np.inf]])
        expectedhalf = np.array([[0.0, 0.0, 0.5, 1.0], [1.0, np.nan, 1.0, 0.0]])
        expected1 = expectedhalf.copy()
        expected1[0, 2] = 1

        h = ncu.heaviside(x, 0.5)
        assert_equal(h, expectedhalf)

        h = ncu.heaviside(x, 1.0)
        assert_equal(h, expected1)

        x = x.astype(np.float32)

        h = ncu.heaviside(x, np.float32(0.5))
        assert_equal(h, expectedhalf.astype(np.float32))

        h = ncu.heaviside(x, np.float32(1.0))
        assert_equal(h, expected1.astype(np.float32))


class TestSign:
    def test_sign(self):
        a = np.array([np.inf, -np.inf, np.nan, 0.0, 3.0, -3.0])
        out = np.zeros(a.shape)
        tgt = np.array([1., -1., np.nan, 0.0, 1.0, -1.0])

        with np.errstate(invalid='ignore'):
            res = ncu.sign(a)
            assert_equal(res, tgt)
            res = ncu.sign(a, out)
            assert_equal(res, tgt)
            assert_equal(out, tgt)

    def test_sign_complex(self):
        a = np.array([
            np.inf, -np.inf, complex(0, np.inf), complex(0, -np.inf),
            complex(np.inf, np.inf), complex(np.inf, -np.inf),  # nan
            np.nan, complex(0, np.nan), complex(np.nan, np.nan),  # nan
            0.0,  # 0.
            3.0, -3.0, -2j, 3.0 + 4.0j, -8.0 + 6.0j
        ])
        out = np.zeros(a.shape, a.dtype)
        tgt = np.array([
            1., -1., 1j, -1j,
            ] + [complex(np.nan, np.nan)] * 5 + [
            0.0,
            1.0, -1.0, -1j, 0.6 + 0.8j, -0.8 + 0.6j])

        with np.errstate(invalid='ignore'):
            res = ncu.sign(a)
            assert_equal(res, tgt)
            res = ncu.sign(a, out)
            assert_(res is out)
            assert_equal(res, tgt)

    def test_sign_dtype_object(self):
        # In reference to github issue #6229

        foo = np.array([-.1, 0, .1])
        a = np.sign(foo.astype(object))
        b = np.sign(foo)

        assert_array_equal(a, b)

    def test_sign_dtype_nan_object(self):
        # In reference to github issue #6229
        def test_nan():
            foo = np.array([np.nan])
            # FIXME: a not used
            a = np.sign(foo.astype(object))

        assert_raises(TypeError, test_nan)

class TestMinMax:
    def test_minmax_blocked(self):
        # simd tests on max/min, test all alignments, slow but important
        # for 2 * vz + 2 * (vs - 1) + 1 (unrolled once)
        for dt, sz in [(np.float32, 15), (np.float64, 7)]:
            for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary',
                                                     max_size=sz):
                for i in range(inp.size):
                    inp[:] = np.arange(inp.size, dtype=dt)
                    inp[i] = np.nan
                    emsg = lambda: f'{inp!r}\n{msg}'
                    with suppress_warnings() as sup:
                        sup.filter(RuntimeWarning,
                                   "invalid value encountered in reduce")
                        assert_(np.isnan(inp.max()), msg=emsg)
                        assert_(np.isnan(inp.min()), msg=emsg)

                    inp[i] = 1e10
                    assert_equal(inp.max(), 1e10, err_msg=msg)
                    inp[i] = -1e10
                    assert_equal(inp.min(), -1e10, err_msg=msg)

    def test_lower_align(self):
        # check data that is not aligned to element size
        # i.e doubles are aligned to 4 bytes on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_equal(d.max(), d[0])
        assert_equal(d.min(), d[0])

    def test_reduce_reorder(self):
        # gh 10370, 11029 Some compilers reorder the call to npy_getfloatstatus
        # and put it before the call to an intrinsic function that causes
        # invalid status to be set. Also make sure warnings are not emitted
        for n in (2, 4, 8, 16, 32):
            for dt in (np.float32, np.float16, np.complex64):
                for r in np.diagflat(np.array([np.nan] * n, dtype=dt)):
                    assert_equal(np.min(r), np.nan)

    def test_minimize_no_warns(self):
        a = np.minimum(np.nan, 1)
        assert_equal(a, np.nan)


class TestAbsoluteNegative:
    def test_abs_neg_blocked(self):
        # simd tests on abs, test all alignments for vz + 2 * (vs - 1) + 1
        for dt, sz in [(np.float32, 11), (np.float64, 5)]:
            for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary',
                                                     max_size=sz):
                tgt = [ncu.absolute(i) for i in inp]
                np.absolute(inp, out=out)
                assert_equal(out, tgt, err_msg=msg)
                assert_((out >= 0).all())

                tgt = [-1 * (i) for i in inp]
                np.negative(inp, out=out)
                assert_equal(out, tgt, err_msg=msg)

                for v in [np.nan, -np.inf, np.inf]:
                    for i in range(inp.size):
                        d = np.arange(inp.size, dtype=dt)
                        inp[:] = -d
                        inp[i] = v
                        d[i] = -v if v == -np.inf else v
                        assert_array_equal(np.abs(inp), d, err_msg=msg)
                        np.abs(inp, out=out)
                        assert_array_equal(out, d, err_msg=msg)

                        assert_array_equal(-inp, -1 * inp, err_msg=msg)
                        d = -1 * inp
                        np.negative(inp, out=out)
                        assert_array_equal(out, d, err_msg=msg)

    def test_lower_align(self):
        # check data that is not aligned to element size
        # i.e doubles are aligned to 4 bytes on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_equal(np.abs(d), d)
        assert_equal(np.negative(d), -d)
        np.negative(d, out=d)
        np.negative(np.ones_like(d), out=d)
        np.abs(d, out=d)
        np.abs(np.ones_like(d), out=d)

    @pytest.mark.parametrize("dtype", ['d', 'f', 'int32', 'int64'])
    @pytest.mark.parametrize("big", [True, False])
    def test_noncontiguous(self, dtype, big):
        data = np.array([-1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.5, 2.5, -6,
                            6, -2.2251e-308, -8, 10], dtype=dtype)
        expect = np.array([1.0, -1.0, 0.0, -0.0, -2.2251e-308, 2.5, -2.5, 6,
                            -6, 2.2251e-308, 8, -10], dtype=dtype)
        if big:
            data = np.repeat(data, 10)
            expect = np.repeat(expect, 10)
        out = np.ndarray(data.shape, dtype=dtype)
        ncontig_in = data[1::2]
        ncontig_out = out[1::2]
        contig_in = np.array(ncontig_in)
        # contig in, contig out
        assert_array_equal(np.negative(contig_in), expect[1::2])
        # contig in, ncontig out
        assert_array_equal(np.negative(contig_in, out=ncontig_out),
                                expect[1::2])
        # ncontig in, contig out
        assert_array_equal(np.negative(ncontig_in), expect[1::2])
        # ncontig in, ncontig out
        assert_array_equal(np.negative(ncontig_in, out=ncontig_out),
                                expect[1::2])
        # contig in, contig out, nd stride
        data_split = np.array(np.array_split(data, 2))
        expect_split = np.array(np.array_split(expect, 2))
        assert_equal(np.negative(data_split), expect_split)


class TestPositive:
    def test_valid(self):
        valid_dtypes = [int, float, complex, object]
        for dtype in valid_dtypes:
            x = np.arange(5, dtype=dtype)
            result = np.positive(x)
            assert_equal(x, result, err_msg=str(dtype))

    def test_invalid(self):
        with assert_raises(TypeError):
            np.positive(True)
        with assert_raises(TypeError):
            np.positive(np.datetime64('2000-01-01'))
        with assert_raises(TypeError):
            np.positive(np.array(['foo'], dtype=str))
        with assert_raises(TypeError):
            np.positive(np.array(['bar'], dtype=object))


class TestSpecialMethods:
    def test_wrap(self):

        class with_wrap:
            def __array__(self, dtype=None, copy=None):
                return np.zeros(1)

            def __array_wrap__(self, arr, context, return_scalar):
                r = with_wrap()
                r.arr = arr
                r.context = context
                return r

        a = with_wrap()
        x = ncu.minimum(a, a)
        assert_equal(x.arr, np.zeros(1))
        func, args, i = x.context
        assert_(func is ncu.minimum)
        assert_equal(len(args), 2)
        assert_equal(args[0], a)
        assert_equal(args[1], a)
        assert_equal(i, 0)

    def test_wrap_out(self):
        # Calling convention for out should not affect how special methods are
        # called

        class StoreArrayPrepareWrap(np.ndarray):
            _wrap_args = None
            _prepare_args = None

            def __new__(cls):
                return np.zeros(()).view(cls)

            def __array_wrap__(self, obj, context, return_scalar):
                self._wrap_args = context[1]
                return obj

            @property
            def args(self):
                # We need to ensure these are fetched at the same time, before
                # any other ufuncs are called by the assertions
                return self._wrap_args

            def __repr__(self):
                return "a"  # for short test output

        def do_test(f_call, f_expected):
            a = StoreArrayPrepareWrap()

            f_call(a)

            w = a.args
            expected = f_expected(a)
            try:
                assert w == expected
            except AssertionError as e:
                # assert_equal produces truly useless error messages
                raise AssertionError("\n".join([
                    "Bad arguments passed in ufunc call",
                    f" expected:              {expected}",
                    f" __array_wrap__ got:    {w}"
                ]))

        # method not on the out argument
        do_test(lambda a: np.add(a, 0), lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, None), lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, out=None), lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, out=(None,)), lambda a: (a, 0))

        # method on the out argument
        do_test(lambda a: np.add(0, 0, a), lambda a: (0, 0, a))
        do_test(lambda a: np.add(0, 0, out=a), lambda a: (0, 0, a))
        do_test(lambda a: np.add(0, 0, out=(a,)), lambda a: (0, 0, a))

        # Also check the where mask handling:
        do_test(lambda a: np.add(a, 0, where=False), lambda a: (a, 0))
        do_test(lambda a: np.add(0, 0, a, where=False), lambda a: (0, 0, a))

    def test_wrap_with_iterable(self):
        # test fix for bug #1026:

        class with_wrap(np.ndarray):
            __array_priority__ = 10

            def __new__(cls):
                return np.asarray(1).view(cls).copy()

            def __array_wrap__(self, arr, context, return_scalar):
                return arr.view(type(self))

        a = with_wrap()
        x = ncu.multiply(a, (1, 2, 3))
        assert_(isinstance(x, with_wrap))
        assert_array_equal(x, np.array((1, 2, 3)))

    def test_priority_with_scalar(self):
        # test fix for bug #826:

        class A(np.ndarray):
            __array_priority__ = 10

            def __new__(cls):
                return np.asarray(1.0, 'float64').view(cls).copy()

        a = A()
        x = np.float64(1) * a
        assert_(isinstance(x, A))
        assert_array_equal(x, np.array(1))

    def test_priority(self):

        class A:
            def __array__(self, dtype=None, copy=None):
                return np.zeros(1)

            def __array_wrap__(self, arr, context, return_scalar):
                r = type(self)()
                r.arr = arr
                r.context = context
                return r

        class B(A):
            __array_priority__ = 20.

        class C(A):
            __array_priority__ = 40.

        x = np.zeros(1)
        a = A()
        b = B()
        c = C()
        f = ncu.minimum
        assert_(type(f(x, x)) is np.ndarray)
        assert_(type(f(x, a)) is A)
        assert_(type(f(x, b)) is B)
        assert_(type(f(x, c)) is C)
        assert_(type(f(a, x)) is A)
        assert_(type(f(b, x)) is B)
        assert_(type(f(c, x)) is C)

        assert_(type(f(a, a)) is A)
        assert_(type(f(a, b)) is B)
        assert_(type(f(b, a)) is B)
        assert_(type(f(b, b)) is B)
        assert_(type(f(b, c)) is C)
        assert_(type(f(c, b)) is C)
        assert_(type(f(c, c)) is C)

        assert_(type(ncu.exp(a) is A))
        assert_(type(ncu.exp(b) is B))
        assert_(type(ncu.exp(c) is C))

    def test_failing_wrap(self):

        class A:
            def __array__(self, dtype=None, copy=None):
                return np.zeros(2)

            def __array_wrap__(self, arr, context, return_scalar):
                raise RuntimeError

        a = A()
        assert_raises(RuntimeError, ncu.maximum, a, a)
        assert_raises(RuntimeError, ncu.maximum.reduce, a)

    def test_failing_out_wrap(self):

        singleton = np.array([1.0])

        class Ok(np.ndarray):
            def __array_wrap__(self, obj, context, return_scalar):
                return singleton

        class Bad(np.ndarray):
            def __array_wrap__(self, obj, context, return_scalar):
                raise RuntimeError

        ok = np.empty(1).view(Ok)
        bad = np.empty(1).view(Bad)
        # double-free (segfault) of "ok" if "bad" raises an exception
        for i in range(10):
            assert_raises(RuntimeError, ncu.frexp, 1, ok, bad)

    def test_none_wrap(self):
        # Tests that issue #8507 is resolved. Previously, this would segfault

        class A:
            def __array__(self, dtype=None, copy=None):
                return np.zeros(1)

            def __array_wrap__(self, arr, context=None, return_scalar=False):
                return None

        a = A()
        assert_equal(ncu.maximum(a, a), None)

    def test_default_prepare(self):

        class with_wrap:
            __array_priority__ = 10

            def __array__(self, dtype=None, copy=None):
                return np.zeros(1)

            def __array_wrap__(self, arr, context, return_scalar):
                return arr

        a = with_wrap()
        x = ncu.minimum(a, a)
        assert_equal(x, np.zeros(1))
        assert_equal(type(x), np.ndarray)

    def test_array_too_many_args(self):

        class A:
            def __array__(self, dtype, context, copy=None):
                return np.zeros(1)

        a = A()
        assert_raises_regex(TypeError, '2 required positional', np.sum, a)

    def test_ufunc_override(self):
        # check override works even with instance with high priority.
        class A:
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return self, func, method, inputs, kwargs

        class MyNDArray(np.ndarray):
            __array_priority__ = 100

        a = A()
        b = np.array([1]).view(MyNDArray)
        res0 = np.multiply(a, b)
        res1 = np.multiply(b, b, out=a)

        # self
        assert_equal(res0[0], a)
        assert_equal(res1[0], a)
        assert_equal(res0[1], np.multiply)
        assert_equal(res1[1], np.multiply)
        assert_equal(res0[2], '__call__')
        assert_equal(res1[2], '__call__')
        assert_equal(res0[3], (a, b))
        assert_equal(res1[3], (b, b))
        assert_equal(res0[4], {})
        assert_equal(res1[4], {'out': (a,)})

    def test_ufunc_override_mro(self):

        # Some multi arg functions for testing.
        def tres_mul(a, b, c):
            return a * b * c

        def quatro_mul(a, b, c, d):
            return a * b * c * d

        # Make these into ufuncs.
        three_mul_ufunc = np.frompyfunc(tres_mul, 3, 1)
        four_mul_ufunc = np.frompyfunc(quatro_mul, 4, 1)

        class A:
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return "A"

        class ASub(A):
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return "ASub"

        class B:
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return "B"

        class C:
            def __init__(self):
                self.count = 0

            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                self.count += 1
                return NotImplemented

        class CSub(C):
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                self.count += 1
                return NotImplemented

        a = A()
        a_sub = ASub()
        b = B()
        c = C()

        # Standard
        res = np.multiply(a, a_sub)
        assert_equal(res, "ASub")
        res = np.multiply(a_sub, b)
        assert_equal(res, "ASub")

        # With 1 NotImplemented
        res = np.multiply(c, a)
        assert_equal(res, "A")
        assert_equal(c.count, 1)
        # Check our counter works, so we can trust tests below.
        res = np.multiply(c, a)
        assert_equal(c.count, 2)

        # Both NotImplemented.
        c = C()
        c_sub = CSub()
        assert_raises(TypeError, np.multiply, c, c_sub)
        assert_equal(c.count, 1)
        assert_equal(c_sub.count, 1)
        c.count = c_sub.count = 0
        assert_raises(TypeError, np.multiply, c_sub, c)
        assert_equal(c.count, 1)
        assert_equal(c_sub.count, 1)
        c.count = 0
        assert_raises(TypeError, np.multiply, c, c)
        assert_equal(c.count, 1)
        c.count = 0
        assert_raises(TypeError, np.multiply, 2, c)
        assert_equal(c.count, 1)

        # Ternary testing.
        assert_equal(three_mul_ufunc(a, 1, 2), "A")
        assert_equal(three_mul_ufunc(1, a, 2), "A")
        assert_equal(three_mul_ufunc(1, 2, a), "A")

        assert_equal(three_mul_ufunc(a, a, 6), "A")
        assert_equal(three_mul_ufunc(a, 2, a), "A")
        assert_equal(three_mul_ufunc(a, 2, b), "A")
        assert_equal(three_mul_ufunc(a, 2, a_sub), "ASub")
        assert_equal(three_mul_ufunc(a, a_sub, 3), "ASub")
        c.count = 0
        assert_equal(three_mul_ufunc(c, a_sub, 3), "ASub")
        assert_equal(c.count, 1)
        c.count = 0
        assert_equal(three_mul_ufunc(1, a_sub, c), "ASub")
        assert_equal(c.count, 0)

        c.count = 0
        assert_equal(three_mul_ufunc(a, b, c), "A")
        assert_equal(c.count, 0)
        c_sub.count = 0
        assert_equal(three_mul_ufunc(a, b, c_sub), "A")
        assert_equal(c_sub.count, 0)
        assert_equal(three_mul_ufunc(1, 2, b), "B")

        assert_raises(TypeError, three_mul_ufunc, 1, 2, c)
        assert_raises(TypeError, three_mul_ufunc, c_sub, 2, c)
        assert_raises(TypeError, three_mul_ufunc, c_sub, 2, 3)

        # Quaternary testing.
        assert_equal(four_mul_ufunc(a, 1, 2, 3), "A")
        assert_equal(four_mul_ufunc(1, a, 2, 3), "A")
        assert_equal(four_mul_ufunc(1, 1, a, 3), "A")
        assert_equal(four_mul_ufunc(1, 1, 2, a), "A")

        assert_equal(four_mul_ufunc(a, b, 2, 3), "A")
        assert_equal(four_mul_ufunc(1, a, 2, b), "A")
        assert_equal(four_mul_ufunc(b, 1, a, 3), "B")
        assert_equal(four_mul_ufunc(a_sub, 1, 2, a), "ASub")
        assert_equal(four_mul_ufunc(a, 1, 2, a_sub), "ASub")

        c = C()
        c_sub = CSub()
        assert_raises(TypeError, four_mul_ufunc, 1, 2, 3, c)
        assert_equal(c.count, 1)
        c.count = 0
        assert_raises(TypeError, four_mul_ufunc, 1, 2, c_sub, c)
        assert_equal(c_sub.count, 1)
        assert_equal(c.count, 1)
        c2 = C()
        c.count = c_sub.count = 0
        assert_raises(TypeError, four_mul_ufunc, 1, c, c_sub, c2)
        assert_equal(c_sub.count, 1)
        assert_equal(c.count, 1)
        assert_equal(c2.count, 0)
        c.count = c2.count = c_sub.count = 0
        assert_raises(TypeError, four_mul_ufunc, c2, c, c_sub, c)
        assert_equal(c_sub.count, 1)
        assert_equal(c.count, 0)
        assert_equal(c2.count, 1)

    def test_ufunc_override_methods(self):

        class A:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return self, ufunc, method, inputs, kwargs

        # __call__
        a = A()
        with assert_raises(TypeError):
            np.multiply.__call__(1, a, foo='bar', answer=42)
        res = np.multiply.__call__(1, a, subok='bar', where=42)
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], '__call__')
        assert_equal(res[3], (1, a))
        assert_equal(res[4], {'subok': 'bar', 'where': 42})

        # __call__, wrong args
        assert_raises(TypeError, np.multiply, a)
        assert_raises(TypeError, np.multiply, a, a, a, a)
        assert_raises(TypeError, np.multiply, a, a, sig='a', signature='a')
        assert_raises(TypeError, ncu_tests.inner1d, a, a, axis=0, axes=[0, 0])

        # reduce, positional args
        res = np.multiply.reduce(a, 'axis0', 'dtype0', 'out0', 'keep0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'reduce')
        assert_equal(res[3], (a,))
        assert_equal(res[4], {'dtype': 'dtype0',
                              'out': ('out0',),
                              'keepdims': 'keep0',
                              'axis': 'axis0'})

        # reduce, kwargs
        res = np.multiply.reduce(a, axis='axis0', dtype='dtype0', out='out0',
                                 keepdims='keep0', initial='init0',
                                 where='where0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'reduce')
        assert_equal(res[3], (a,))
        assert_equal(res[4], {'dtype': 'dtype0',
                              'out': ('out0',),
                              'keepdims': 'keep0',
                              'axis': 'axis0',
                              'initial': 'init0',
                              'where': 'where0'})

        # reduce, output equal to None removed, but not other explicit ones,
        # even if they are at their default value.
        res = np.multiply.reduce(a, 0, None, None, False)
        assert_equal(res[4], {'axis': 0, 'dtype': None, 'keepdims': False})
        res = np.multiply.reduce(a, out=None, axis=0, keepdims=True)
        assert_equal(res[4], {'axis': 0, 'keepdims': True})
        res = np.multiply.reduce(a, None, out=(None,), dtype=None)
        assert_equal(res[4], {'axis': None, 'dtype': None})
        res = np.multiply.reduce(a, 0, None, None, False, 2, True)
        assert_equal(res[4], {'axis': 0, 'dtype': None, 'keepdims': False,
                              'initial': 2, 'where': True})
        # np._NoValue ignored for initial
        res = np.multiply.reduce(a, 0, None, None, False,
                                 np._NoValue, True)
        assert_equal(res[4], {'axis': 0, 'dtype': None, 'keepdims': False,
                              'where': True})
        # None kept for initial, True for where.
        res = np.multiply.reduce(a, 0, None, None, False, None, True)
        assert_equal(res[4], {'axis': 0, 'dtype': None, 'keepdims': False,
                              'initial': None, 'where': True})

        # reduce, wrong args
        assert_raises(ValueError, np.multiply.reduce, a, out=())
        assert_raises(ValueError, np.multiply.reduce, a, out=('out0', 'out1'))
        assert_raises(TypeError, np.multiply.reduce, a, 'axis0', axis='axis0')

        # accumulate, pos args
        res = np.multiply.accumulate(a, 'axis0', 'dtype0', 'out0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'accumulate')
        assert_equal(res[3], (a,))
        assert_equal(res[4], {'dtype': 'dtype0',
                              'out': ('out0',),
                              'axis': 'axis0'})

        # accumulate, kwargs
        res = np.multiply.accumulate(a, axis='axis0', dtype='dtype0',
                                     out='out0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'accumulate')
        assert_equal(res[3], (a,))
        assert_equal(res[4], {'dtype': 'dtype0',
                              'out': ('out0',),
                              'axis': 'axis0'})

        # accumulate, output equal to None removed.
        res = np.multiply.accumulate(a, 0, None, None)
        assert_equal(res[4], {'axis': 0, 'dtype': None})
        res = np.multiply.accumulate(a, out=None, axis=0, dtype='dtype1')
        assert_equal(res[4], {'axis': 0, 'dtype': 'dtype1'})
        res = np.multiply.accumulate(a, None, out=(None,), dtype=None)
        assert_equal(res[4], {'axis': None, 'dtype': None})

        # accumulate, wrong args
        assert_raises(ValueError, np.multiply.accumulate, a, out=())
        assert_raises(ValueError, np.multiply.accumulate, a,
                      out=('out0', 'out1'))
        assert_raises(TypeError, np.multiply.accumulate, a,
                      'axis0', axis='axis0')

        # reduceat, pos args
        res = np.multiply.reduceat(a, [4, 2], 'axis0', 'dtype0', 'out0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'reduceat')
        assert_equal(res[3], (a, [4, 2]))
        assert_equal(res[4], {'dtype': 'dtype0',
                              'out': ('out0',),
                              'axis': 'axis0'})

        # reduceat, kwargs
        res = np.multiply.reduceat(a, [4, 2], axis='axis0', dtype='dtype0',
                                   out='out0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'reduceat')
        assert_equal(res[3], (a, [4, 2]))
        assert_equal(res[4], {'dtype': 'dtype0',
                              'out': ('out0',),
                              'axis': 'axis0'})

        # reduceat, output equal to None removed.
        res = np.multiply.reduceat(a, [4, 2], 0, None, None)
        assert_equal(res[4], {'axis': 0, 'dtype': None})
        res = np.multiply.reduceat(a, [4, 2], axis=None, out=None, dtype='dt')
        assert_equal(res[4], {'axis': None, 'dtype': 'dt'})
        res = np.multiply.reduceat(a, [4, 2], None, None, out=(None,))
        assert_equal(res[4], {'axis': None, 'dtype': None})

        # reduceat, wrong args
        assert_raises(ValueError, np.multiply.reduce, a, [4, 2], out=())
        assert_raises(ValueError, np.multiply.reduce, a, [4, 2],
                      out=('out0', 'out1'))
        assert_raises(TypeError, np.multiply.reduce, a, [4, 2],
                      'axis0', axis='axis0')

        # outer
        res = np.multiply.outer(a, 42)
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'outer')
        assert_equal(res[3], (a, 42))
        assert_equal(res[4], {})

        # outer, wrong args
        assert_raises(TypeError, np.multiply.outer, a)
        assert_raises(TypeError, np.multiply.outer, a, a, a, a)
        assert_raises(TypeError, np.multiply.outer, a, a, sig='a', signature='a')

        # at
        res = np.multiply.at(a, [4, 2], 'b0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'at')
        assert_equal(res[3], (a, [4, 2], 'b0'))

        # at, wrong args
        assert_raises(TypeError, np.multiply.at, a)
        assert_raises(TypeError, np.multiply.at, a, a, a, a)

    def test_ufunc_override_out(self):

        class A:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return kwargs

        class B:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return kwargs

        a = A()
        b = B()
        res0 = np.multiply(a, b, 'out_arg')
        res1 = np.multiply(a, b, out='out_arg')
        res2 = np.multiply(2, b, 'out_arg')
        res3 = np.multiply(3, b, out='out_arg')
        res4 = np.multiply(a, 4, 'out_arg')
        res5 = np.multiply(a, 5, out='out_arg')

        assert_equal(res0['out'][0], 'out_arg')
        assert_equal(res1['out'][0], 'out_arg')
        assert_equal(res2['out'][0], 'out_arg')
        assert_equal(res3['out'][0], 'out_arg')
        assert_equal(res4['out'][0], 'out_arg')
        assert_equal(res5['out'][0], 'out_arg')

        # ufuncs with multiple output modf and frexp.
        res6 = np.modf(a, 'out0', 'out1')
        res7 = np.frexp(a, 'out0', 'out1')
        assert_equal(res6['out'][0], 'out0')
        assert_equal(res6['out'][1], 'out1')
        assert_equal(res7['out'][0], 'out0')
        assert_equal(res7['out'][1], 'out1')

        # While we're at it, check that default output is never passed on.
        assert_(np.sin(a, None) == {})
        assert_(np.sin(a, out=None) == {})
        assert_(np.sin(a, out=(None,)) == {})
        assert_(np.modf(a, None) == {})
        assert_(np.modf(a, None, None) == {})
        assert_(np.modf(a, out=(None, None)) == {})
        with assert_raises(TypeError):
            # Out argument must be tuple, since there are multiple outputs.
            np.modf(a, out=None)

        # don't give positional and output argument, or too many arguments.
        # wrong number of arguments in the tuple is an error too.
        assert_raises(TypeError, np.multiply, a, b, 'one', out='two')
        assert_raises(TypeError, np.multiply, a, b, 'one', 'two')
        assert_raises(ValueError, np.multiply, a, b, out=('one', 'two'))
        assert_raises(TypeError, np.multiply, a, out=())
        assert_raises(TypeError, np.modf, a, 'one', out=('two', 'three'))
        assert_raises(TypeError, np.modf, a, 'one', 'two', 'three')
        assert_raises(ValueError, np.modf, a, out=('one', 'two', 'three'))
        assert_raises(ValueError, np.modf, a, out=('one',))

    def test_ufunc_override_where(self):

        class OverriddenArrayOld(np.ndarray):

            def _unwrap(self, objs):
                cls = type(self)
                result = []
                for obj in objs:
                    if isinstance(obj, cls):
                        obj = np.array(obj)
                    elif type(obj) != np.ndarray:
                        return NotImplemented
                    result.append(obj)
                return result

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

                inputs = self._unwrap(inputs)
                if inputs is NotImplemented:
                    return NotImplemented

                kwargs = kwargs.copy()
                if "out" in kwargs:
                    kwargs["out"] = self._unwrap(kwargs["out"])
                    if kwargs["out"] is NotImplemented:
                        return NotImplemented

                r = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
                if r is not NotImplemented:
                    r = r.view(type(self))

                return r

        class OverriddenArrayNew(OverriddenArrayOld):
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

                kwargs = kwargs.copy()
                if "where" in kwargs:
                    kwargs["where"] = self._unwrap((kwargs["where"], ))
                    if kwargs["where"] is NotImplemented:
                        return NotImplemented
                    else:
                        kwargs["where"] = kwargs["where"][0]

                r = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
                if r is not NotImplemented:
                    r = r.view(type(self))

                return r

        ufunc = np.negative

        array = np.array([1, 2, 3])
        where = np.array([True, False, True])
        expected = ufunc(array, where=where)

        with pytest.raises(TypeError):
            ufunc(array, where=where.view(OverriddenArrayOld))

        result_1 = ufunc(
            array,
            where=where.view(OverriddenArrayNew)
        )
        assert isinstance(result_1, OverriddenArrayNew)
        assert np.all(np.array(result_1) == expected, where=where)

        result_2 = ufunc(
            array.view(OverriddenArrayNew),
            where=where.view(OverriddenArrayNew)
        )
        assert isinstance(result_2, OverriddenArrayNew)
        assert np.all(np.array(result_2) == expected, where=where)

    def test_ufunc_override_exception(self):

        class A:
            def __array_ufunc__(self, *a, **kwargs):
                raise ValueError("oops")

        a = A()
        assert_raises(ValueError, np.negative, 1, out=a)
        assert_raises(ValueError, np.negative, a)
        assert_raises(ValueError, np.divide, 1., a)

    def test_ufunc_override_not_implemented(self):

        class A:
            def __array_ufunc__(self, *args, **kwargs):
                return NotImplemented

        msg = ("operand type(s) all returned NotImplemented from "
               "__array_ufunc__(<ufunc 'negative'>, '__call__', <*>): 'A'")
        with assert_raises_regex(TypeError, fnmatch.translate(msg)):
            np.negative(A())

        msg = ("operand type(s) all returned NotImplemented from "
               "__array_ufunc__(<ufunc 'add'>, '__call__', <*>, <object *>, "
               "out=(1,)): 'A', 'object', 'int'")
        with assert_raises_regex(TypeError, fnmatch.translate(msg)):
            np.add(A(), object(), out=1)

    def test_ufunc_override_disabled(self):

        class OptOut:
            __array_ufunc__ = None

        opt_out = OptOut()

        # ufuncs always raise
        msg = "operand 'OptOut' does not support ufuncs"
        with assert_raises_regex(TypeError, msg):
            np.add(opt_out, 1)
        with assert_raises_regex(TypeError, msg):
            np.add(1, opt_out)
        with assert_raises_regex(TypeError, msg):
            np.negative(opt_out)

        # opt-outs still hold even when other arguments have pathological
        # __array_ufunc__ implementations

        class GreedyArray:
            def __array_ufunc__(self, *args, **kwargs):
                return self

        greedy = GreedyArray()
        assert_(np.negative(greedy) is greedy)
        with assert_raises_regex(TypeError, msg):
            np.add(greedy, opt_out)
        with assert_raises_regex(TypeError, msg):
            np.add(greedy, 1, out=opt_out)

    def test_gufunc_override(self):
        # gufunc are just ufunc instances, but follow a different path,
        # so check __array_ufunc__ overrides them properly.
        class A:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return self, ufunc, method, inputs, kwargs

        inner1d = ncu_tests.inner1d
        a = A()
        res = inner1d(a, a)
        assert_equal(res[0], a)
        assert_equal(res[1], inner1d)
        assert_equal(res[2], '__call__')
        assert_equal(res[3], (a, a))
        assert_equal(res[4], {})

        res = inner1d(1, 1, out=a)
        assert_equal(res[0], a)
        assert_equal(res[1], inner1d)
        assert_equal(res[2], '__call__')
        assert_equal(res[3], (1, 1))
        assert_equal(res[4], {'out': (a,)})

        # wrong number of arguments in the tuple is an error too.
        assert_raises(TypeError, inner1d, a, out='two')
        assert_raises(TypeError, inner1d, a, a, 'one', out='two')
        assert_raises(TypeError, inner1d, a, a, 'one', 'two')
        assert_raises(ValueError, inner1d, a, a, out=('one', 'two'))
        assert_raises(ValueError, inner1d, a, a, out=())

    def test_ufunc_override_with_super(self):
        # NOTE: this class is used in doc/source/user/basics.subclassing.rst
        # if you make any changes here, do update it there too.
        class A(np.ndarray):
            def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
                args = []
                in_no = []
                for i, input_ in enumerate(inputs):
                    if isinstance(input_, A):
                        in_no.append(i)
                        args.append(input_.view(np.ndarray))
                    else:
                        args.append(input_)

                outputs = out
                out_no = []
                if outputs:
                    out_args = []
                    for j, output in enumerate(outputs):
                        if isinstance(output, A):
                            out_no.append(j)
                            out_args.append(output.view(np.ndarray))
                        else:
                            out_args.append(output)
                    kwargs['out'] = tuple(out_args)
                else:
                    outputs = (None,) * ufunc.nout

                info = {}
                if in_no:
                    info['inputs'] = in_no
                if out_no:
                    info['outputs'] = out_no

                results = super().__array_ufunc__(ufunc, method,
                                                  *args, **kwargs)
                if results is NotImplemented:
                    return NotImplemented

                if method == 'at':
                    if isinstance(inputs[0], A):
                        inputs[0].info = info
                    return

                if ufunc.nout == 1:
                    results = (results,)

                results = tuple((np.asarray(result).view(A)
                                 if output is None else output)
                                for result, output in zip(results, outputs))
                if results and isinstance(results[0], A):
                    results[0].info = info

                return results[0] if len(results) == 1 else results

        class B:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                if any(isinstance(input_, A) for input_ in inputs):
                    return "A!"
                else:
                    return NotImplemented

        d = np.arange(5.)
        # 1 input, 1 output
        a = np.arange(5.).view(A)
        b = np.sin(a)
        check = np.sin(d)
        assert_(np.all(check == b))
        assert_equal(b.info, {'inputs': [0]})
        b = np.sin(d, out=(a,))
        assert_(np.all(check == b))
        assert_equal(b.info, {'outputs': [0]})
        assert_(b is a)
        a = np.arange(5.).view(A)
        b = np.sin(a, out=a)
        assert_(np.all(check == b))
        assert_equal(b.info, {'inputs': [0], 'outputs': [0]})

        # 1 input, 2 outputs
        a = np.arange(5.).view(A)
        b1, b2 = np.modf(a)
        assert_equal(b1.info, {'inputs': [0]})
        b1, b2 = np.modf(d, out=(None, a))
        assert_(b2 is a)
        assert_equal(b1.info, {'outputs': [1]})
        a = np.arange(5.).view(A)
        b = np.arange(5.).view(A)
        c1, c2 = np.modf(a, out=(a, b))
        assert_(c1 is a)
        assert_(c2 is b)
        assert_equal(c1.info, {'inputs': [0], 'outputs': [0, 1]})

        # 2 input, 1 output
        a = np.arange(5.).view(A)
        b = np.arange(5.).view(A)
        c = np.add(a, b, out=a)
        assert_(c is a)
        assert_equal(c.info, {'inputs': [0, 1], 'outputs': [0]})
        # some tests with a non-ndarray subclass
        a = np.arange(5.)
        b = B()
        assert_(a.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
        assert_(b.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
        assert_raises(TypeError, np.add, a, b)
        a = a.view(A)
        assert_(a.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
        assert_(b.__array_ufunc__(np.add, '__call__', a, b) == "A!")
        assert_(np.add(a, b) == "A!")
        # regression check for gh-9102 -- tests ufunc.reduce implicitly.
        d = np.array([[1, 2, 3], [1, 2, 3]])
        a = d.view(A)
        c = a.any()
        check = d.any()
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        c = a.max()
        check = d.max()
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        b = np.array(0).view(A)
        c = a.max(out=b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        check = a.max(axis=0)
        b = np.zeros_like(check).view(A)
        c = a.max(axis=0, out=b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        # simple explicit tests of reduce, accumulate, reduceat
        check = np.add.reduce(d, axis=1)
        c = np.add.reduce(a, axis=1)
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        b = np.zeros_like(c)
        c = np.add.reduce(a, 1, None, b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        check = np.add.accumulate(d, axis=0)
        c = np.add.accumulate(a, axis=0)
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        b = np.zeros_like(c)
        c = np.add.accumulate(a, 0, None, b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        indices = [0, 2, 1]
        check = np.add.reduceat(d, indices, axis=1)
        c = np.add.reduceat(a, indices, axis=1)
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        b = np.zeros_like(c)
        c = np.add.reduceat(a, indices, 1, None, b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        # and a few tests for at
        d = np.array([[1, 2, 3], [1, 2, 3]])
        check = d.copy()
        a = d.copy().view(A)
        np.add.at(check, ([0, 1], [0, 2]), 1.)
        np.add.at(a, ([0, 1], [0, 2]), 1.)
        assert_equal(a, check)
        assert_(a.info, {'inputs': [0]})
        b = np.array(1.).view(A)
        a = d.copy().view(A)
        np.add.at(a, ([0, 1], [0, 2]), b)
        assert_equal(a, check)
        assert_(a.info, {'inputs': [0, 2]})

    def test_array_ufunc_direct_call(self):
        # This is mainly a regression test for gh-24023 (shouldn't segfault)
        a = np.array(1)
        with pytest.raises(TypeError):
            a.__array_ufunc__()

        # No kwargs means kwargs may be NULL on the C-level
        with pytest.raises(TypeError):
            a.__array_ufunc__(1, 2)

        # And the same with a valid call:
        res = a.__array_ufunc__(np.add, "__call__", a, a)
        assert_array_equal(res, a + a)

    def test_ufunc_docstring(self):
        original_doc = np.add.__doc__
        new_doc = "new docs"
        expected_dict = (
            {} if IS_PYPY else {"__module__": "numpy", "__qualname__": "add"}
        )

        np.add.__doc__ = new_doc
        assert np.add.__doc__ == new_doc
        assert np.add.__dict__["__doc__"] == new_doc

        del np.add.__doc__
        assert np.add.__doc__ == original_doc
        assert np.add.__dict__ == expected_dict

        np.add.__dict__["other"] = 1
        np.add.__dict__["__doc__"] = new_doc
        assert np.add.__doc__ == new_doc

        del np.add.__dict__["__doc__"]
        assert np.add.__doc__ == original_doc
        del np.add.__dict__["other"]
        assert np.add.__dict__ == expected_dict


class TestChoose:
    def test_mixed(self):
        c = np.array([True, True])
        a = np.array([True, True])
        assert_equal(np.choose(c, (a, 1)), np.array([1, 1]))


class TestRationalFunctions:
    def test_lcm(self):
        self._test_lcm_inner(np.int16)
        self._test_lcm_inner(np.uint16)

    def test_lcm_object(self):
        self._test_lcm_inner(np.object_)

    def test_gcd(self):
        self._test_gcd_inner(np.int16)
        self._test_lcm_inner(np.uint16)

    def test_gcd_object(self):
        self._test_gcd_inner(np.object_)

    def _test_lcm_inner(self, dtype):
        # basic use
        a = np.array([12, 120], dtype=dtype)
        b = np.array([20, 200], dtype=dtype)
        assert_equal(np.lcm(a, b), [60, 600])

        if not issubclass(dtype, np.unsignedinteger):
            # negatives are ignored
            a = np.array([12, -12,  12, -12], dtype=dtype)
            b = np.array([20,  20, -20, -20], dtype=dtype)
            assert_equal(np.lcm(a, b), [60] * 4)

        # reduce
        a = np.array([3, 12, 20], dtype=dtype)
        assert_equal(np.lcm.reduce([3, 12, 20]), 60)

        # broadcasting, and a test including 0
        a = np.arange(6).astype(dtype)
        b = 20
        assert_equal(np.lcm(a, b), [0, 20, 20, 60, 20, 20])

    def _test_gcd_inner(self, dtype):
        # basic use
        a = np.array([12, 120], dtype=dtype)
        b = np.array([20, 200], dtype=dtype)
        assert_equal(np.gcd(a, b), [4, 40])

        if not issubclass(dtype, np.unsignedinteger):
            # negatives are ignored
            a = np.array([12, -12,  12, -12], dtype=dtype)
            b = np.array([20,  20, -20, -20], dtype=dtype)
            assert_equal(np.gcd(a, b), [4] * 4)

        # reduce
        a = np.array([15, 25, 35], dtype=dtype)
        assert_equal(np.gcd.reduce(a), 5)

        # broadcasting, and a test including 0
        a = np.arange(6).astype(dtype)
        b = 20
        assert_equal(np.gcd(a, b), [20,  1,  2,  1,  4,  5])

    def test_lcm_overflow(self):
        # verify that we don't overflow when a*b does overflow
        big = np.int32(np.iinfo(np.int32).max // 11)
        a = 2 * big
        b = 5 * big
        assert_equal(np.lcm(a, b), 10 * big)

    def test_gcd_overflow(self):
        for dtype in (np.int32, np.int64):
            # verify that we don't overflow when taking abs(x)
            # not relevant for lcm, where the result is unrepresentable anyway
            a = dtype(np.iinfo(dtype).min)  # negative power of two
            q = -(a // 4)
            assert_equal(np.gcd(a,  q * 3), q)
            assert_equal(np.gcd(a, -q * 3), q)

    def test_decimal(self):
        from decimal import Decimal
        a = np.array([1,  1, -1, -1]) * Decimal('0.20')
        b = np.array([1, -1,  1, -1]) * Decimal('0.12')

        assert_equal(np.gcd(a, b), 4 * [Decimal('0.04')])
        assert_equal(np.lcm(a, b), 4 * [Decimal('0.60')])

    def test_float(self):
        # not well-defined on float due to rounding errors
        assert_raises(TypeError, np.gcd, 0.3, 0.4)
        assert_raises(TypeError, np.lcm, 0.3, 0.4)

    def test_huge_integers(self):
        # Converting to an array first is a bit different as it means we
        # have an explicit object dtype:
        assert_equal(np.array(2**200), 2**200)
        # Special promotion rules should ensure that this also works for
        # two Python integers (even if slow).
        # (We do this for comparisons, as the result is always bool and
        # we also special case array comparisons with Python integers)
        np.equal(2**200, 2**200)

        # But, we cannot do this when it would affect the result dtype:
        with pytest.raises(OverflowError):
            np.gcd(2**100, 3**100)

        # Asking for `object` explicitly is fine, though:
        assert np.gcd(2**100, 3**100, dtype=object) == 1

        # As of now, the below work, because it is using arrays (which
        # will be object arrays)
        a = np.array(2**100 * 3**5)
        b = np.array([2**100 * 5**7, 2**50 * 3**10])
        assert_equal(np.gcd(a, b), [2**100,               2**50 * 3**5])
        assert_equal(np.lcm(a, b), [2**100 * 3**5 * 5**7, 2**100 * 3**10])

    def test_inf_and_nan(self):
        inf = np.array([np.inf], dtype=np.object_)
        assert_raises(ValueError, np.gcd, inf, 1)
        assert_raises(ValueError, np.gcd, 1, inf)
        assert_raises(ValueError, np.gcd, np.nan, inf)
        assert_raises(TypeError, np.gcd, 4, float(np.inf))


class TestRoundingFunctions:

    def test_object_direct(self):
        """ test direct implementation of these magic methods """
        class C:
            def __floor__(self):
                return 1

            def __ceil__(self):
                return 2

            def __trunc__(self):
                return 3

        arr = np.array([C(), C()])
        assert_equal(np.floor(arr), [1, 1])
        assert_equal(np.ceil(arr),  [2, 2])
        assert_equal(np.trunc(arr), [3, 3])

    def test_object_indirect(self):
        """ test implementations via __float__ """
        class C:
            def __float__(self):
                return -2.5

        arr = np.array([C(), C()])
        assert_equal(np.floor(arr), [-3, -3])
        assert_equal(np.ceil(arr),  [-2, -2])
        with pytest.raises(TypeError):
            np.trunc(arr)  # consistent with math.trunc

    def test_fraction(self):
        f = Fraction(-4, 3)
        assert_equal(np.floor(f), -2)
        assert_equal(np.ceil(f), -1)
        assert_equal(np.trunc(f), -1)

    @pytest.mark.parametrize('func', [np.floor, np.ceil, np.trunc])
    @pytest.mark.parametrize('dtype', [np.bool, np.float64, np.float32,
                                       np.int64, np.uint32])
    def test_output_dtype(self, func, dtype):
        arr = np.array([-2, 0, 4, 8]).astype(dtype)
        result = func(arr)
        assert_equal(arr, result)
        assert result.dtype == dtype


class TestComplexFunctions:
    funcs = [np.arcsin,  np.arccos,  np.arctan, np.arcsinh, np.arccosh,
             np.arctanh, np.sin,     np.cos,    np.tan,     np.exp,
             np.exp2,    np.log,     np.sqrt,   np.log10,   np.log2,
             np.log1p]

    def test_it(self):
        for f in self.funcs:
            if f is np.arccosh:
                x = 1.5
            else:
                x = .5
            fr = f(x)
            fz = f(complex(x))
            assert_almost_equal(fz.real, fr, err_msg=f'real part {f}')
            assert_almost_equal(fz.imag, 0., err_msg=f'imag part {f}')

    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_precisions_consistent(self):
        z = 1 + 1j
        for f in self.funcs:
            fcf = f(np.csingle(z))
            fcd = f(np.cdouble(z))
            fcl = f(np.clongdouble(z))
            assert_almost_equal(fcf, fcd, decimal=6, err_msg=f'fch-fcd {f}')
            assert_almost_equal(fcl, fcd, decimal=15, err_msg=f'fch-fcl {f}')

    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_branch_cuts(self):
        # check branch cuts and continuity on them
        _check_branch_cut(np.log,   -0.5, 1j, 1, -1, True)  # noqa: E221
        _check_branch_cut(np.log2,  -0.5, 1j, 1, -1, True)  # noqa: E221
        _check_branch_cut(np.log10, -0.5, 1j, 1, -1, True)
        _check_branch_cut(np.log1p, -1.5, 1j, 1, -1, True)
        _check_branch_cut(np.sqrt,  -0.5, 1j, 1, -1, True)  # noqa: E221

        _check_branch_cut(np.arcsin, [ -2, 2],   [1j, 1j], 1, -1, True)
        _check_branch_cut(np.arccos, [ -2, 2],   [1j, 1j], 1, -1, True)
        _check_branch_cut(np.arctan, [0 - 2j, 2j],  [1,  1], -1, 1, True)

        _check_branch_cut(np.arcsinh, [0 - 2j,  2j], [1,   1], -1, 1, True)
        _check_branch_cut(np.arccosh, [ -1, 0.5], [1j,  1j], 1, -1, True)
        _check_branch_cut(np.arctanh, [ -2,   2], [1j, 1j], 1, -1, True)

        # check against bogus branch cuts: assert continuity between quadrants
        _check_branch_cut(np.arcsin, [0 - 2j, 2j], [ 1,  1], 1, 1)
        _check_branch_cut(np.arccos, [0 - 2j, 2j], [ 1,  1], 1, 1)
        _check_branch_cut(np.arctan, [ -2,  2], [1j, 1j], 1, 1)

        _check_branch_cut(np.arcsinh, [ -2,  2, 0], [1j, 1j, 1], 1, 1)
        _check_branch_cut(np.arccosh, [0 - 2j, 2j, 2], [1,  1,  1j], 1, 1)
        _check_branch_cut(np.arctanh, [0 - 2j, 2j, 0], [1,  1,  1j], 1, 1)

    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_branch_cuts_complex64(self):
        # check branch cuts and continuity on them
        _check_branch_cut(np.log,   -0.5, 1j, 1, -1, True, np.complex64)  # noqa: E221
        _check_branch_cut(np.log2,  -0.5, 1j, 1, -1, True, np.complex64)  # noqa: E221
        _check_branch_cut(np.log10, -0.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.log1p, -1.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.sqrt,  -0.5, 1j, 1, -1, True, np.complex64)  # noqa: E221

        _check_branch_cut(np.arcsin, [ -2, 2],   [1j, 1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arccos, [ -2, 2],   [1j, 1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arctan, [0 - 2j, 2j],  [1,  1], -1, 1, True, np.complex64)

        _check_branch_cut(np.arcsinh, [0 - 2j,  2j], [1,   1], -1, 1, True, np.complex64)
        _check_branch_cut(np.arccosh, [ -1, 0.5], [1j,  1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arctanh, [ -2,   2], [1j, 1j], 1, -1, True, np.complex64)

        # check against bogus branch cuts: assert continuity between quadrants
        _check_branch_cut(np.arcsin, [0 - 2j, 2j], [ 1,  1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arccos, [0 - 2j, 2j], [ 1,  1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arctan, [ -2,  2], [1j, 1j], 1, 1, False, np.complex64)

        _check_branch_cut(np.arcsinh, [ -2,  2, 0], [1j, 1j, 1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arccosh, [0 - 2j, 2j, 2], [1,  1,  1j], 1, 1, False, np.complex64)
        _check_branch_cut(np.arctanh, [0 - 2j, 2j, 0], [1,  1,  1j], 1, 1, False, np.complex64)

    def test_against_cmath(self):
        import cmath

        points = [-1 - 1j, -1 + 1j, +1 - 1j, +1 + 1j]
        name_map = {'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
                    'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh'}
        atol = 4 * np.finfo(complex).eps
        for func in self.funcs:
            fname = func.__name__.split('.')[-1]
            cname = name_map.get(fname, fname)
            try:
                cfunc = getattr(cmath, cname)
            except AttributeError:
                continue
            for p in points:
                a = complex(func(np.complex128(p)))
                b = cfunc(p)
                assert_(
                    abs(a - b) < atol,
                    f"{fname} {p}: {a}; cmath: {b}"
                )

    @pytest.mark.xfail(
        # manylinux2014 uses glibc2.17
        _glibc_older_than("2.18"),
        reason="Older glibc versions are imprecise (maybe passes with SIMD?)"
    )
    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    @pytest.mark.parametrize('dtype', [
        np.complex64, np.complex128, np.clongdouble
    ])
    def test_loss_of_precision(self, dtype):
        """Check loss of precision in complex arc* functions"""
        if dtype is np.clongdouble and platform.machine() != 'x86_64':
            # Failures on musllinux, aarch64, s390x, ppc64le (see gh-17554)
            pytest.skip('Only works reliably for x86-64 and recent glibc')

        # Check against known-good functions

        info = np.finfo(dtype)
        real_dtype = dtype(0.).real.dtype
        eps = info.eps

        def check(x, rtol):
            x = x.astype(real_dtype)

            z = x.astype(dtype)
            d = np.absolute(np.arcsinh(x) / np.arcsinh(z).real - 1)
            assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(),
                                      'arcsinh'))

            z = (1j * x).astype(dtype)
            d = np.absolute(np.arcsinh(x) / np.arcsin(z).imag - 1)
            assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(),
                                      'arcsin'))

            z = x.astype(dtype)
            d = np.absolute(np.arctanh(x) / np.arctanh(z).real - 1)
            assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(),
                                      'arctanh'))

            z = (1j * x).astype(dtype)
            d = np.absolute(np.arctanh(x) / np.arctan(z).imag - 1)
            assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(),
                                      'arctan'))

        # The switchover was chosen as 1e-3; hence there can be up to
        # ~eps/1e-3 of relative cancellation error before it

        x_series = np.logspace(-20, -3.001, 200)
        x_basic = np.logspace(-2.999, 0, 10, endpoint=False)

        if dtype is np.clongdouble:
            if bad_arcsinh():
                pytest.skip("Trig functions of np.clongdouble values known "
                            "to be inaccurate on aarch64 and PPC for some "
                            "compilation configurations.")
            # It's not guaranteed that the system-provided arc functions
            # are accurate down to a few epsilons. (Eg. on Linux 64-bit)
            # So, give more leeway for long complex tests here:
            check(x_series, 50.0 * eps)
        else:
            check(x_series, 2.1 * eps)
        check(x_basic, 2.0 * eps / 1e-3)

        # Check a few points

        z = np.array([1e-5 * (1 + 1j)], dtype=dtype)
        p = 9.999999999333333333e-6 + 1.000000000066666666e-5j
        d = np.absolute(1 - np.arctanh(z) / p)
        assert_(np.all(d < 1e-15))

        p = 1.0000000000333333333e-5 + 9.999999999666666667e-6j
        d = np.absolute(1 - np.arcsinh(z) / p)
        assert_(np.all(d < 1e-15))

        p = 9.999999999333333333e-6j + 1.000000000066666666e-5
        d = np.absolute(1 - np.arctan(z) / p)
        assert_(np.all(d < 1e-15))

        p = 1.0000000000333333333e-5j + 9.999999999666666667e-6
        d = np.absolute(1 - np.arcsin(z) / p)
        assert_(np.all(d < 1e-15))

        # Check continuity across switchover points

        def check(func, z0, d=1):
            z0 = np.asarray(z0, dtype=dtype)
            zp = z0 + abs(z0) * d * eps * 2
            zm = z0 - abs(z0) * d * eps * 2
            assert_(np.all(zp != zm), (zp, zm))

            # NB: the cancellation error at the switchover is at least eps
            good = (abs(func(zp) - func(zm)) < 2 * eps)
            assert_(np.all(good), (func, z0[~good]))

        for func in (np.arcsinh, np.arcsinh, np.arcsin, np.arctanh, np.arctan):
            pts = [rp + 1j * ip for rp in (-1e-3, 0, 1e-3) for ip in (-1e-3, 0, 1e-3)
                   if rp != 0 or ip != 0]
            check(func, pts, 1)
            check(func, pts, 1j)
            check(func, pts, 1 + 1j)

    @np.errstate(all="ignore")
    def test_promotion_corner_cases(self):
        for func in self.funcs:
            assert func(np.float16(1)).dtype == np.float16
            # Integer to low precision float promotion is a dubious choice:
            assert func(np.uint8(1)).dtype == np.float16
            assert func(np.int16(1)).dtype == np.float32


class TestAttributes:
    def test_attributes(self):
        add = ncu.add
        assert_equal(add.__name__, 'add')
        assert_(add.ntypes >= 18)  # don't fail if types added
        assert_('ii->i' in add.types)
        assert_equal(add.nin, 2)
        assert_equal(add.nout, 1)
        assert_equal(add.identity, 0)

    def test_doc(self):
        # don't bother checking the long list of kwargs, which are likely to
        # change
        assert_(ncu.add.__doc__.startswith(
            "add(x1, x2, /, out=None, *, where=True"))
        assert_(ncu.frexp.__doc__.startswith(
            "frexp(x[, out1, out2], / [, out=(None, None)], *, where=True"))


class TestSubclass:

    def test_subclass_op(self):

        class simple(np.ndarray):
            def __new__(subtype, shape):
                self = np.ndarray.__new__(subtype, shape, dtype=object)
                self.fill(0)
                return self

        a = simple((3, 4))
        assert_equal(a + a, a)


class TestFrompyfunc:

    def test_identity(self):
        def mul(a, b):
            return a * b

        # with identity=value
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1, identity=1)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_equal(mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)), 1)
        assert_equal(mul_ufunc.reduce([]), 1)

        # with identity=None (reorderable)
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1, identity=None)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_equal(mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)), 1)
        assert_raises(ValueError, lambda: mul_ufunc.reduce([]))

        # with no identity (not reorderable)
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_raises(ValueError, lambda: mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)))
        assert_raises(ValueError, lambda: mul_ufunc.reduce([]))


def _check_branch_cut(f, x0, dx, re_sign=1, im_sign=-1, sig_zero_ok=False,
                      dtype=complex):
    """
    Check for a branch cut in a function.

    Assert that `x0` lies on a branch cut of function `f` and `f` is
    continuous from the direction `dx`.

    Parameters
    ----------
    f : func
        Function to check
    x0 : array-like
        Point on branch cut
    dx : array-like
        Direction to check continuity in
    re_sign, im_sign : {1, -1}
        Change of sign of the real or imaginary part expected
    sig_zero_ok : bool
        Whether to check if the branch cut respects signed zero (if applicable)
    dtype : dtype
        Dtype to check (should be complex)

    """
    x0 = np.atleast_1d(x0).astype(dtype)
    dx = np.atleast_1d(dx).astype(dtype)

    if np.dtype(dtype).char == 'F':
        scale = np.finfo(dtype).eps * 1e2
        atol = np.float32(1e-2)
    else:
        scale = np.finfo(dtype).eps * 1e3
        atol = 1e-4

    y0 = f(x0)
    yp = f(x0 + dx * scale * np.absolute(x0) / np.absolute(dx))
    ym = f(x0 - dx * scale * np.absolute(x0) / np.absolute(dx))

    assert_(np.all(np.absolute(y0.real - yp.real) < atol), (y0, yp))
    assert_(np.all(np.absolute(y0.imag - yp.imag) < atol), (y0, yp))
    assert_(np.all(np.absolute(y0.real - ym.real * re_sign) < atol), (y0, ym))
    assert_(np.all(np.absolute(y0.imag - ym.imag * im_sign) < atol), (y0, ym))

    if sig_zero_ok:
        # check that signed zeros also work as a displacement
        jr = (x0.real == 0) & (dx.real != 0)
        ji = (x0.imag == 0) & (dx.imag != 0)
        if np.any(jr):
            x = x0[jr]
            x.real = ncu.NZERO
            ym = f(x)
            assert_(np.all(np.absolute(y0[jr].real - ym.real * re_sign) < atol), (y0[jr], ym))
            assert_(np.all(np.absolute(y0[jr].imag - ym.imag * im_sign) < atol), (y0[jr], ym))

        if np.any(ji):
            x = x0[ji]
            x.imag = ncu.NZERO
            ym = f(x)
            assert_(np.all(np.absolute(y0[ji].real - ym.real * re_sign) < atol), (y0[ji], ym))
            assert_(np.all(np.absolute(y0[ji].imag - ym.imag * im_sign) < atol), (y0[ji], ym))

def test_copysign():
    assert_(np.copysign(1, -1) == -1)
    with np.errstate(divide="ignore"):
        assert_(1 / np.copysign(0, -1) < 0)
        assert_(1 / np.copysign(0, 1) > 0)
    assert_(np.signbit(np.copysign(np.nan, -1)))
    assert_(not np.signbit(np.copysign(np.nan, 1)))

def _test_nextafter(t):
    one = t(1)
    two = t(2)
    zero = t(0)
    eps = np.finfo(t).eps
    assert_(np.nextafter(one, two) - one == eps)
    assert_(np.nextafter(one, zero) - one < 0)
    assert_(np.isnan(np.nextafter(np.nan, one)))
    assert_(np.isnan(np.nextafter(one, np.nan)))
    assert_(np.nextafter(one, one) == one)

def test_nextafter():
    return _test_nextafter(np.float64)


def test_nextafterf():
    return _test_nextafter(np.float32)


@pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double")
@pytest.mark.xfail(condition=platform.machine().startswith("ppc64"),
                    reason="IBM double double")
def test_nextafterl():
    return _test_nextafter(np.longdouble)


def test_nextafter_0():
    for t, direction in itertools.product(np._core.sctypes['float'], (1, -1)):
        # The value of tiny for double double is NaN, so we need to pass the
        # assert
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            if not np.isnan(np.finfo(t).tiny):
                tiny = np.finfo(t).tiny
                assert_(
                    0. < direction * np.nextafter(t(0), t(direction)) < tiny)
        assert_equal(np.nextafter(t(0), t(direction)) / t(2.1), direction * 0.0)

def _test_spacing(t):
    one = t(1)
    eps = np.finfo(t).eps
    nan = t(np.nan)
    inf = t(np.inf)
    with np.errstate(invalid='ignore'):
        assert_equal(np.spacing(one), eps)
        assert_(np.isnan(np.spacing(nan)))
        assert_(np.isnan(np.spacing(inf)))
        assert_(np.isnan(np.spacing(-inf)))
        assert_(np.spacing(t(1e30)) != 0)

def test_spacing():
    return _test_spacing(np.float64)

def test_spacingf():
    return _test_spacing(np.float32)


@pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double")
@pytest.mark.xfail(condition=platform.machine().startswith("ppc64"),
                    reason="IBM double double")
def test_spacingl():
    return _test_spacing(np.longdouble)

def test_spacing_gfortran():
    # Reference from this fortran file, built with gfortran 4.3.3 on linux
    # 32bits:
    #       PROGRAM test_spacing
    #        INTEGER, PARAMETER :: SGL = SELECTED_REAL_KIND(p=6, r=37)
    #        INTEGER, PARAMETER :: DBL = SELECTED_REAL_KIND(p=13, r=200)
    #
    #        WRITE(*,*) spacing(0.00001_DBL)
    #        WRITE(*,*) spacing(1.0_DBL)
    #        WRITE(*,*) spacing(1000._DBL)
    #        WRITE(*,*) spacing(10500._DBL)
    #
    #        WRITE(*,*) spacing(0.00001_SGL)
    #        WRITE(*,*) spacing(1.0_SGL)
    #        WRITE(*,*) spacing(1000._SGL)
    #        WRITE(*,*) spacing(10500._SGL)
    #       END PROGRAM
    ref = {np.float64: [1.69406589450860068E-021,
                        2.22044604925031308E-016,
                        1.13686837721616030E-013,
                        1.81898940354585648E-012],
           np.float32: [9.09494702E-13,
                        1.19209290E-07,
                        6.10351563E-05,
                        9.76562500E-04]}

    for dt, dec_ in zip([np.float32, np.float64], (10, 20)):
        x = np.array([1e-5, 1, 1000, 10500], dtype=dt)
        assert_array_almost_equal(np.spacing(x), ref[dt], decimal=dec_)

def test_nextafter_vs_spacing():
    # XXX: spacing does not handle long double yet
    for t in [np.float32, np.float64]:
        for _f in [1, 1e-5, 1000]:
            f = t(_f)
            f1 = t(_f + 1)
            assert_(np.nextafter(f, f1) - f == np.spacing(f))

def test_pos_nan():
    """Check np.nan is a positive nan."""
    assert_(np.signbit(np.nan) == 0)

def test_reduceat():
    """Test bug in reduceat when structured arrays are not copied."""
    db = np.dtype([('name', 'S11'), ('time', np.int64), ('value', np.float32)])
    a = np.empty([100], dtype=db)
    a['name'] = 'Simple'
    a['time'] = 10
    a['value'] = 100
    indx = [0, 7, 15, 25]

    h2 = []
    val1 = indx[0]
    for val2 in indx[1:]:
        h2.append(np.add.reduce(a['value'][val1:val2]))
        val1 = val2
    h2.append(np.add.reduce(a['value'][val1:]))
    h2 = np.array(h2)

    # test buffered -- this should work
    h1 = np.add.reduceat(a['value'], indx)
    assert_array_almost_equal(h1, h2)

    # This is when the error occurs.
    # test no buffer
    np.setbufsize(32)
    h1 = np.add.reduceat(a['value'], indx)
    np.setbufsize(ncu.UFUNC_BUFSIZE_DEFAULT)
    assert_array_almost_equal(h1, h2)

def test_reduceat_empty():
    """Reduceat should work with empty arrays"""
    indices = np.array([], 'i4')
    x = np.array([], 'f8')
    result = np.add.reduceat(x, indices)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (0,))
    # Another case with a slightly different zero-sized shape
    x = np.ones((5, 2))
    result = np.add.reduceat(x, [], axis=0)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (0, 2))
    result = np.add.reduceat(x, [], axis=1)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (5, 0))

def test_complex_nan_comparisons():
    nans = [complex(np.nan, 0), complex(0, np.nan), complex(np.nan, np.nan)]
    fins = [complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1),
            complex(1, 1), complex(-1, -1), complex(0, 0)]

    with np.errstate(invalid='ignore'):
        for x in nans + fins:
            x = np.array([x])
            for y in nans + fins:
                y = np.array([y])

                if np.isfinite(x) and np.isfinite(y):
                    continue

                assert_equal(x < y, False, err_msg=f"{x!r} < {y!r}")
                assert_equal(x > y, False, err_msg=f"{x!r} > {y!r}")
                assert_equal(x <= y, False, err_msg=f"{x!r} <= {y!r}")
                assert_equal(x >= y, False, err_msg=f"{x!r} >= {y!r}")
                assert_equal(x == y, False, err_msg=f"{x!r} == {y!r}")


def test_rint_big_int():
    # np.rint bug for large integer values on Windows 32-bit and MKL
    # https://github.com/numpy/numpy/issues/6685
    val = 4607998452777363968
    # This is exactly representable in floating point
    assert_equal(val, int(float(val)))
    # Rint should not change the value
    assert_equal(val, np.rint(val))


@pytest.mark.parametrize('ftype', [np.float32, np.float64])
def test_memoverlap_accumulate(ftype):
    # Reproduces bug https://github.com/numpy/numpy/issues/15597
    arr = np.array([0.61, 0.60, 0.77, 0.41, 0.19], dtype=ftype)
    out_max = np.array([0.61, 0.61, 0.77, 0.77, 0.77], dtype=ftype)
    out_min = np.array([0.61, 0.60, 0.60, 0.41, 0.19], dtype=ftype)
    assert_equal(np.maximum.accumulate(arr), out_max)
    assert_equal(np.minimum.accumulate(arr), out_min)

@pytest.mark.parametrize("ufunc, dtype", [
    (ufunc, t[0])
    for ufunc in UFUNCS_BINARY_ACC
    for t in ufunc.types
    if t[-1] == '?' and t[0] not in 'DFGMmO'
])
def test_memoverlap_accumulate_cmp(ufunc, dtype):
    if ufunc.signature:
        pytest.skip('For generic signatures only')
    for size in (2, 8, 32, 64, 128, 256):
        arr = np.array([0, 1, 1] * size, dtype=dtype)
        acc = ufunc.accumulate(arr, dtype='?')
        acc_u8 = acc.view(np.uint8)
        exp = np.array(list(itertools.accumulate(arr, ufunc)), dtype=np.uint8)
        assert_equal(exp, acc_u8)

@pytest.mark.parametrize("ufunc, dtype", [
    (ufunc, t[0])
    for ufunc in UFUNCS_BINARY_ACC
    for t in ufunc.types
    if t[0] == t[1] and t[0] == t[-1] and t[0] not in 'DFGMmO?'
])
def test_memoverlap_accumulate_symmetric(ufunc, dtype):
    if ufunc.signature:
        pytest.skip('For generic signatures only')
    with np.errstate(all='ignore'):
        for size in (2, 8, 32, 64, 128, 256):
            arr = np.array([0, 1, 2] * size).astype(dtype)
            acc = ufunc.accumulate(arr, dtype=dtype)
            exp = np.array(list(itertools.accumulate(arr, ufunc)), dtype=dtype)
            assert_equal(exp, acc)

def test_signaling_nan_exceptions():
    with assert_no_warnings():
        a = np.ndarray(shape=(), dtype='float32', buffer=b'\x00\xe0\xbf\xff')
        np.isnan(a)

@pytest.mark.parametrize("arr", [
    np.arange(2),
    np.matrix([0, 1]),
    np.matrix([[0, 1], [2, 5]]),
    ])
def test_outer_subclass_preserve(arr):
    # for gh-8661
    class foo(np.ndarray):
        pass
    actual = np.multiply.outer(arr.view(foo), arr.view(foo))
    assert actual.__class__.__name__ == 'foo'

def test_outer_bad_subclass():
    class BadArr1(np.ndarray):
        def __array_finalize__(self, obj):
            # The outer call reshapes to 3 dims, try to do a bad reshape.
            if self.ndim == 3:
                self.shape = self.shape + (1,)

    class BadArr2(np.ndarray):
        def __array_finalize__(self, obj):
            if isinstance(obj, BadArr2):
                # outer inserts 1-sized dims. In that case disturb them.
                if self.shape[-1] == 1:
                    self.shape = self.shape[::-1]

    for cls in [BadArr1, BadArr2]:
        arr = np.ones((2, 3)).view(cls)
        with assert_raises(TypeError) as a:
            # The first array gets reshaped (not the second one)
            np.add.outer(arr, [1, 2])

        # This actually works, since we only see the reshaping error:
        arr = np.ones((2, 3)).view(cls)
        assert type(np.add.outer([1, 2], arr)) is cls

def test_outer_exceeds_maxdims():
    deep = np.ones((1,) * 33)
    with assert_raises(ValueError):
        np.add.outer(deep, deep)

def test_bad_legacy_ufunc_silent_errors():
    # legacy ufuncs can't report errors and NumPy can't check if the GIL
    # is released.  So NumPy has to check after the GIL is released just to
    # cover all bases.  `np.power` uses/used to use this.
    arr = np.arange(3).astype(np.float64)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error(arr, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # not contiguous means the fast-path cannot be taken
        non_contig = arr.repeat(20).reshape(-1, 6)[:, ::2]
        ncu_tests.always_error(non_contig, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.outer(arr, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.reduce(arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.reduceat(arr, [0, 1])

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.accumulate(arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.at(arr, [0, 1, 2], arr)


@pytest.mark.parametrize('x1', [np.arange(3.0), [0.0, 1.0, 2.0]])
def test_bad_legacy_gufunc_silent_errors(x1):
    # Verify that an exception raised in a gufunc loop propagates correctly.
    # The signature of always_error_gufunc is '(i),()->()'.
    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error_gufunc(x1, 0.0)


class TestAddDocstring:
    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    @pytest.mark.skipif(IS_PYPY, reason="PyPy does not modify tp_doc")
    def test_add_same_docstring(self):
        # test for attributes (which are C-level defined)
        ncu.add_docstring(np.ndarray.flat, np.ndarray.flat.__doc__)

        # And typical functions:
        def func():
            """docstring"""
            return

        ncu.add_docstring(func, func.__doc__)

    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    def test_different_docstring_fails(self):
        # test for attributes (which are C-level defined)
        with assert_raises(RuntimeError):
            ncu.add_docstring(np.ndarray.flat, "different docstring")

        # And typical functions:
        def func():
            """docstring"""
            return

        with assert_raises(RuntimeError):
            ncu.add_docstring(func, "different docstring")


class TestAdd_newdoc_ufunc:
    @pytest.mark.filterwarnings("ignore:_add_newdoc_ufunc:DeprecationWarning")
    def test_ufunc_arg(self):
        assert_raises(TypeError, ncu._add_newdoc_ufunc, 2, "blah")
        assert_raises(ValueError, ncu._add_newdoc_ufunc, np.add, "blah")

    @pytest.mark.filterwarnings("ignore:_add_newdoc_ufunc:DeprecationWarning")
    def test_string_arg(self):
        assert_raises(TypeError, ncu._add_newdoc_ufunc, np.add, 3)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\PropertyBag.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class PropertyBag(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = PropertyBag()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsPropertyBag(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def PropertyBagBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x44\x54\x43", size_prefixed=size_prefixed)

    # PropertyBag
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # PropertyBag
    def Ints(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.IntProperty import IntProperty
            obj = IntProperty()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # PropertyBag
    def IntsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # PropertyBag
    def IntsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # PropertyBag
    def Floats(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.FloatProperty import FloatProperty
            obj = FloatProperty()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # PropertyBag
    def FloatsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # PropertyBag
    def FloatsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # PropertyBag
    def Strings(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.StringProperty import StringProperty
            obj = StringProperty()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # PropertyBag
    def StringsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # PropertyBag
    def StringsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

def PropertyBagStart(builder):
    builder.StartObject(3)

def Start(builder):
    PropertyBagStart(builder)

def PropertyBagAddInts(builder, ints):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(ints), 0)

def AddInts(builder, ints):
    PropertyBagAddInts(builder, ints)

def PropertyBagStartIntsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartIntsVector(builder, numElems: int) -> int:
    return PropertyBagStartIntsVector(builder, numElems)

def PropertyBagAddFloats(builder, floats):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(floats), 0)

def AddFloats(builder, floats):
    PropertyBagAddFloats(builder, floats)

def PropertyBagStartFloatsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartFloatsVector(builder, numElems: int) -> int:
    return PropertyBagStartFloatsVector(builder, numElems)

def PropertyBagAddStrings(builder, strings):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(strings), 0)

def AddStrings(builder, strings):
    PropertyBagAddStrings(builder, strings)

def PropertyBagStartStringsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartStringsVector(builder, numElems: int) -> int:
    return PropertyBagStartStringsVector(builder, numElems)

def PropertyBagEnd(builder):
    return builder.EndObject()

def End(builder):
    return PropertyBagEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\wait.py ===
import errno
import select
import sys
from functools import partial

try:
    from time import monotonic
except ImportError:
    from time import time as monotonic

__all__ = ["NoWayToWaitForSocketError", "wait_for_read", "wait_for_write"]


class NoWayToWaitForSocketError(Exception):
    pass


# How should we wait on sockets?
#
# There are two types of APIs you can use for waiting on sockets: the fancy
# modern stateful APIs like epoll/kqueue, and the older stateless APIs like
# select/poll. The stateful APIs are more efficient when you have a lots of
# sockets to keep track of, because you can set them up once and then use them
# lots of times. But we only ever want to wait on a single socket at a time
# and don't want to keep track of state, so the stateless APIs are actually
# more efficient. So we want to use select() or poll().
#
# Now, how do we choose between select() and poll()? On traditional Unixes,
# select() has a strange calling convention that makes it slow, or fail
# altogether, for high-numbered file descriptors. The point of poll() is to fix
# that, so on Unixes, we prefer poll().
#
# On Windows, there is no poll() (or at least Python doesn't provide a wrapper
# for it), but that's OK, because on Windows, select() doesn't have this
# strange calling convention; plain select() works fine.
#
# So: on Windows we use select(), and everywhere else we use poll(). We also
# fall back to select() in case poll() is somehow broken or missing.

if sys.version_info >= (3, 5):
    # Modern Python, that retries syscalls by default
    def _retry_on_intr(fn, timeout):
        return fn(timeout)

else:
    # Old and broken Pythons.
    def _retry_on_intr(fn, timeout):
        if timeout is None:
            deadline = float("inf")
        else:
            deadline = monotonic() + timeout

        while True:
            try:
                return fn(timeout)
            # OSError for 3 <= pyver < 3.5, select.error for pyver <= 2.7
            except (OSError, select.error) as e:
                # 'e.args[0]' incantation works for both OSError and select.error
                if e.args[0] != errno.EINTR:
                    raise
                else:
                    timeout = deadline - monotonic()
                    if timeout < 0:
                        timeout = 0
                    if timeout == float("inf"):
                        timeout = None
                    continue


def select_wait_for_socket(sock, read=False, write=False, timeout=None):
    if not read and not write:
        raise RuntimeError("must specify at least one of read=True, write=True")
    rcheck = []
    wcheck = []
    if read:
        rcheck.append(sock)
    if write:
        wcheck.append(sock)
    # When doing a non-blocking connect, most systems signal success by
    # marking the socket writable. Windows, though, signals success by marked
    # it as "exceptional". We paper over the difference by checking the write
    # sockets for both conditions. (The stdlib selectors module does the same
    # thing.)
    fn = partial(select.select, rcheck, wcheck, wcheck)
    rready, wready, xready = _retry_on_intr(fn, timeout)
    return bool(rready or wready or xready)


def poll_wait_for_socket(sock, read=False, write=False, timeout=None):
    if not read and not write:
        raise RuntimeError("must specify at least one of read=True, write=True")
    mask = 0
    if read:
        mask |= select.POLLIN
    if write:
        mask |= select.POLLOUT
    poll_obj = select.poll()
    poll_obj.register(sock, mask)

    # For some reason, poll() takes timeout in milliseconds
    def do_poll(t):
        if t is not None:
            t *= 1000
        return poll_obj.poll(t)

    return bool(_retry_on_intr(do_poll, timeout))


def null_wait_for_socket(*args, **kwargs):
    raise NoWayToWaitForSocketError("no select-equivalent available")


def _have_working_poll():
    # Apparently some systems have a select.poll that fails as soon as you try
    # to use it, either due to strange configuration or broken monkeypatching
    # from libraries like eventlet/greenlet.
    try:
        poll_obj = select.poll()
        _retry_on_intr(poll_obj.poll, 0)
    except (AttributeError, OSError):
        return False
    else:
        return True


def wait_for_socket(*args, **kwargs):
    # We delay choosing which implementation to use until the first time we're
    # called. We could do it at import time, but then we might make the wrong
    # decision if someone goes wild with monkeypatching select.poll after
    # we're imported.
    global wait_for_socket
    if _have_working_poll():
        wait_for_socket = poll_wait_for_socket
    elif hasattr(select, "select"):
        wait_for_socket = select_wait_for_socket
    else:  # Platform-specific: Appengine.
        wait_for_socket = null_wait_for_socket
    return wait_for_socket(*args, **kwargs)


def wait_for_read(sock, timeout=None):
    """Waits for reading to be available on a given socket.
    Returns True if the socket is readable, or False if the timeout expired.
    """
    return wait_for_socket(sock, read=True, timeout=timeout)


def wait_for_write(sock, timeout=None):
    """Waits for writing to be available on a given socket.
    Returns True if the socket is readable, or False if the timeout expired.
    """
    return wait_for_socket(sock, write=True, timeout=timeout)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\client_config.py ===
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

import base64
import os
import socket
from enum import Enum
from typing import Optional
from urllib import parse

import certifi

from selenium.webdriver.common.proxy import Proxy, ProxyType


class AuthType(Enum):
    BASIC = "Basic"
    BEARER = "Bearer"
    X_API_KEY = "X-API-Key"


class _ClientConfigDescriptor:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, cls):
        return obj.__dict__[self.name]

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.name] = value


class ClientConfig:
    remote_server_addr = _ClientConfigDescriptor("_remote_server_addr")
    """Gets and Sets Remote Server."""
    keep_alive = _ClientConfigDescriptor("_keep_alive")
    """Gets and Sets Keep Alive value."""
    proxy = _ClientConfigDescriptor("_proxy")
    """Gets and Sets the proxy used for communicating to the driver/server."""
    ignore_certificates = _ClientConfigDescriptor("_ignore_certificates")
    """Gets and Sets the ignore certificate check value."""
    init_args_for_pool_manager = _ClientConfigDescriptor("_init_args_for_pool_manager")
    """Gets and Sets the ignore certificate check."""
    timeout = _ClientConfigDescriptor("_timeout")
    """Gets and Sets the timeout (in seconds) used for communicating to the
    driver/server."""
    ca_certs = _ClientConfigDescriptor("_ca_certs")
    """Gets and Sets the path to bundle of CA certificates."""
    username = _ClientConfigDescriptor("_username")
    """Gets and Sets the username used for basic authentication to the
    remote."""
    password = _ClientConfigDescriptor("_password")
    """Gets and Sets the password used for basic authentication to the
    remote."""
    auth_type = _ClientConfigDescriptor("_auth_type")
    """Gets and Sets the type of authentication to the remote server."""
    token = _ClientConfigDescriptor("_token")
    """Gets and Sets the token used for authentication to the remote server."""
    user_agent = _ClientConfigDescriptor("_user_agent")
    """Gets and Sets user agent to be added to the request headers."""
    extra_headers = _ClientConfigDescriptor("_extra_headers")
    """Gets and Sets extra headers to be added to the request."""

    def __init__(
        self,
        remote_server_addr: str,
        keep_alive: Optional[bool] = True,
        proxy: Optional[Proxy] = Proxy(raw={"proxyType": ProxyType.SYSTEM}),
        ignore_certificates: Optional[bool] = False,
        init_args_for_pool_manager: Optional[dict] = None,
        timeout: Optional[int] = None,
        ca_certs: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_type: Optional[AuthType] = AuthType.BASIC,
        token: Optional[str] = None,
        user_agent: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ) -> None:
        self.remote_server_addr = remote_server_addr
        self.keep_alive = keep_alive
        self.proxy = proxy
        self.ignore_certificates = ignore_certificates
        self.init_args_for_pool_manager = init_args_for_pool_manager or {}
        self.timeout = socket.getdefaulttimeout() if timeout is None else timeout
        self.username = username
        self.password = password
        self.auth_type = auth_type
        self.token = token
        self.user_agent = user_agent
        self.extra_headers = extra_headers

        self.ca_certs = (
            (os.getenv("REQUESTS_CA_BUNDLE") if "REQUESTS_CA_BUNDLE" in os.environ else certifi.where())
            if ca_certs is None
            else ca_certs
        )

    def reset_timeout(self) -> None:
        """Resets the timeout to the default value of socket."""
        self._timeout = socket.getdefaulttimeout()

    def get_proxy_url(self) -> Optional[str]:
        """Returns the proxy URL to use for the connection."""
        proxy_type = self.proxy.proxy_type
        remote_add = parse.urlparse(self.remote_server_addr)
        if proxy_type is ProxyType.DIRECT:
            return None
        if proxy_type is ProxyType.SYSTEM:
            _no_proxy = os.environ.get("no_proxy", os.environ.get("NO_PROXY"))
            if _no_proxy:
                for entry in map(str.strip, _no_proxy.split(",")):
                    if entry == "*":
                        return None
                    n_url = parse.urlparse(entry)
                    if n_url.netloc and remote_add.netloc == n_url.netloc:
                        return None
                    if n_url.path in remote_add.netloc:
                        return None
            return os.environ.get(
                "https_proxy" if self.remote_server_addr.startswith("https://") else "http_proxy",
                os.environ.get("HTTPS_PROXY" if self.remote_server_addr.startswith("https://") else "HTTP_PROXY"),
            )
        if proxy_type is ProxyType.MANUAL:
            return self.proxy.sslProxy if self.remote_server_addr.startswith("https://") else self.proxy.http_proxy
        return None

    def get_auth_header(self) -> Optional[dict]:
        """Returns the authorization to add to the request headers."""
        if self.auth_type is AuthType.BASIC and self.username and self.password:
            credentials = f"{self.username}:{self.password}"
            encoded_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
            return {"Authorization": f"{AuthType.BASIC.value} {encoded_credentials}"}
        if self.auth_type is AuthType.BEARER and self.token:
            return {"Authorization": f"{AuthType.BEARER.value} {self.token}"}
        if self.auth_type is AuthType.X_API_KEY and self.token:
            return {f"{AuthType.X_API_KEY.value}": f"{self.token}"}
        return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\quantities.py ===
"""
Physical quantities.
"""

from sympy.core.expr import AtomicExpr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.physics.units.dimensions import _QuantityMapper
from sympy.physics.units.prefixes import Prefix


class Quantity(AtomicExpr):
    """
    Physical quantity: can be a unit of measure, a constant or a generic quantity.
    """

    is_commutative = True
    is_real = True
    is_number = False
    is_nonzero = True
    is_physical_constant = False
    _diff_wrt = True

    def __new__(cls, name, abbrev=None,
                latex_repr=None, pretty_unicode_repr=None,
                pretty_ascii_repr=None, mathml_presentation_repr=None,
                is_prefixed=False,
                **assumptions):

        if not isinstance(name, Symbol):
            name = Symbol(name)

        if abbrev is None:
            abbrev = name
        elif isinstance(abbrev, str):
            abbrev = Symbol(abbrev)

        # HACK: These are here purely for type checking. They actually get assigned below.
        cls._is_prefixed = is_prefixed

        obj = AtomicExpr.__new__(cls, name, abbrev)
        obj._name = name
        obj._abbrev = abbrev
        obj._latex_repr = latex_repr
        obj._unicode_repr = pretty_unicode_repr
        obj._ascii_repr = pretty_ascii_repr
        obj._mathml_repr = mathml_presentation_repr
        obj._is_prefixed = is_prefixed
        return obj

    def set_global_dimension(self, dimension):
        _QuantityMapper._quantity_dimension_global[self] = dimension

    def set_global_relative_scale_factor(self, scale_factor, reference_quantity):
        """
        Setting a scale factor that is valid across all unit system.
        """
        from sympy.physics.units import UnitSystem
        scale_factor = sympify(scale_factor)
        if isinstance(scale_factor, Prefix):
            self._is_prefixed = True
        # replace all prefixes by their ratio to canonical units:
        scale_factor = scale_factor.replace(
            lambda x: isinstance(x, Prefix),
            lambda x: x.scale_factor
        )
        scale_factor = sympify(scale_factor)
        UnitSystem._quantity_scale_factors_global[self] = (scale_factor, reference_quantity)
        UnitSystem._quantity_dimensional_equivalence_map_global[self] = reference_quantity

    @property
    def name(self):
        return self._name

    @property
    def dimension(self):
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_default_unit_system()
        return unit_system.get_quantity_dimension(self)

    @property
    def abbrev(self):
        """
        Symbol representing the unit name.

        Prepend the abbreviation with the prefix symbol if it is defines.
        """
        return self._abbrev

    @property
    def scale_factor(self):
        """
        Overall magnitude of the quantity as compared to the canonical units.
        """
        from sympy.physics.units import UnitSystem
        unit_system = UnitSystem.get_default_unit_system()
        return unit_system.get_quantity_scale_factor(self)

    def _eval_is_positive(self):
        return True

    def _eval_is_constant(self):
        return True

    def _eval_Abs(self):
        return self

    def _eval_subs(self, old, new):
        if isinstance(new, Quantity) and self != old:
            return self

    def _latex(self, printer):
        if self._latex_repr:
            return self._latex_repr
        else:
            return r'\text{{{}}}'.format(self.args[1] \
                          if len(self.args) >= 2 else self.args[0])

    def convert_to(self, other, unit_system="SI"):
        """
        Convert the quantity to another quantity of same dimensions.

        Examples
        ========

        >>> from sympy.physics.units import speed_of_light, meter, second
        >>> speed_of_light
        speed_of_light
        >>> speed_of_light.convert_to(meter/second)
        299792458*meter/second

        >>> from sympy.physics.units import liter
        >>> liter.convert_to(meter**3)
        meter**3/1000
        """
        from .util import convert_to
        return convert_to(self, other, unit_system)

    @property
    def free_symbols(self):
        """Return free symbols from quantity."""
        return set()

    @property
    def is_prefixed(self):
        """Whether or not the quantity is prefixed. Eg. `kilogram` is prefixed, but `gram` is not."""
        return self._is_prefixed

class PhysicalConstant(Quantity):
    """Represents a physical constant, eg. `speed_of_light` or `avogadro_constant`."""

    is_physical_constant = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\routing\exceptions.py ===
from __future__ import annotations

import difflib
import typing as t

from ..exceptions import BadRequest
from ..exceptions import HTTPException
from ..utils import cached_property
from ..utils import redirect

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment

    from ..wrappers.request import Request
    from ..wrappers.response import Response
    from .map import MapAdapter
    from .rules import Rule


class RoutingException(Exception):
    """Special exceptions that require the application to redirect, notifying
    about missing urls, etc.

    :internal:
    """


class RequestRedirect(HTTPException, RoutingException):
    """Raise if the map requests a redirect. This is for example the case if
    `strict_slashes` are activated and an url that requires a trailing slash.

    The attribute `new_url` contains the absolute destination url.
    """

    code = 308

    def __init__(self, new_url: str) -> None:
        super().__init__(new_url)
        self.new_url = new_url

    def get_response(
        self,
        environ: WSGIEnvironment | Request | None = None,
        scope: dict[str, t.Any] | None = None,
    ) -> Response:
        return redirect(self.new_url, self.code)


class RequestPath(RoutingException):
    """Internal exception."""

    __slots__ = ("path_info",)

    def __init__(self, path_info: str) -> None:
        super().__init__()
        self.path_info = path_info


class RequestAliasRedirect(RoutingException):  # noqa: B903
    """This rule is an alias and wants to redirect to the canonical URL."""

    def __init__(self, matched_values: t.Mapping[str, t.Any], endpoint: t.Any) -> None:
        super().__init__()
        self.matched_values = matched_values
        self.endpoint = endpoint


class BuildError(RoutingException, LookupError):
    """Raised if the build system cannot find a URL for an endpoint with the
    values provided.
    """

    def __init__(
        self,
        endpoint: t.Any,
        values: t.Mapping[str, t.Any],
        method: str | None,
        adapter: MapAdapter | None = None,
    ) -> None:
        super().__init__(endpoint, values, method)
        self.endpoint = endpoint
        self.values = values
        self.method = method
        self.adapter = adapter

    @cached_property
    def suggested(self) -> Rule | None:
        return self.closest_rule(self.adapter)

    def closest_rule(self, adapter: MapAdapter | None) -> Rule | None:
        def _score_rule(rule: Rule) -> float:
            return sum(
                [
                    0.98
                    * difflib.SequenceMatcher(
                        # endpoints can be any type, compare as strings
                        None,
                        str(rule.endpoint),
                        str(self.endpoint),
                    ).ratio(),
                    0.01 * bool(set(self.values or ()).issubset(rule.arguments)),
                    0.01 * bool(rule.methods and self.method in rule.methods),
                ]
            )

        if adapter and adapter.map._rules:
            return max(adapter.map._rules, key=_score_rule)

        return None

    def __str__(self) -> str:
        message = [f"Could not build url for endpoint {self.endpoint!r}"]
        if self.method:
            message.append(f" ({self.method!r})")
        if self.values:
            message.append(f" with values {sorted(self.values)!r}")
        message.append(".")
        if self.suggested:
            if self.endpoint == self.suggested.endpoint:
                if (
                    self.method
                    and self.suggested.methods is not None
                    and self.method not in self.suggested.methods
                ):
                    message.append(
                        " Did you mean to use methods"
                        f" {sorted(self.suggested.methods)!r}?"
                    )
                missing_values = self.suggested.arguments.union(
                    set(self.suggested.defaults or ())
                ) - set(self.values.keys())
                if missing_values:
                    message.append(
                        f" Did you forget to specify values {sorted(missing_values)!r}?"
                    )
            else:
                message.append(f" Did you mean {self.suggested.endpoint!r} instead?")
        return "".join(message)


class WebsocketMismatch(BadRequest):
    """The only matched rule is either a WebSocket and the request is
    HTTP, or the rule is HTTP and the request is a WebSocket.
    """


class NoMatch(Exception):
    __slots__ = ("have_match_for", "websocket_mismatch")

    def __init__(self, have_match_for: set[str], websocket_mismatch: bool) -> None:
        self.have_match_for = have_match_for
        self.websocket_mismatch = websocket_mismatch

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\styles\styleable.py ===
# Copyright (c) 2010-2024 openpyxl

from copy import copy

from .numbers import (
    BUILTIN_FORMATS,
    BUILTIN_FORMATS_MAX_SIZE,
    BUILTIN_FORMATS_REVERSE,
)
from .proxy import StyleProxy
from .cell_style import StyleArray
from .named_styles import NamedStyle
from .builtins import styles


class StyleDescriptor:

    def __init__(self, collection, key):
        self.collection = collection
        self.key = key

    def __set__(self, instance, value):
        coll = getattr(instance.parent.parent, self.collection)
        if not getattr(instance, "_style"):
            instance._style = StyleArray()
        setattr(instance._style, self.key, coll.add(value))


    def __get__(self, instance, cls):
        coll = getattr(instance.parent.parent, self.collection)
        if not getattr(instance, "_style"):
            instance._style = StyleArray()
        idx =  getattr(instance._style, self.key)
        return StyleProxy(coll[idx])


class NumberFormatDescriptor:

    key = "numFmtId"
    collection = '_number_formats'

    def __set__(self, instance, value):
        coll = getattr(instance.parent.parent, self.collection)
        if value in BUILTIN_FORMATS_REVERSE:
            idx = BUILTIN_FORMATS_REVERSE[value]
        else:
            idx = coll.add(value) + BUILTIN_FORMATS_MAX_SIZE

        if not getattr(instance, "_style"):
            instance._style = StyleArray()
        setattr(instance._style, self.key, idx)


    def __get__(self, instance, cls):
        if not getattr(instance, "_style"):
            instance._style = StyleArray()
        idx = getattr(instance._style, self.key)
        if idx < BUILTIN_FORMATS_MAX_SIZE:
            return BUILTIN_FORMATS.get(idx, "General")
        coll = getattr(instance.parent.parent, self.collection)
        return coll[idx - BUILTIN_FORMATS_MAX_SIZE]


class NamedStyleDescriptor:

    key = "xfId"
    collection = "_named_styles"


    def __set__(self, instance, value):
        if not getattr(instance, "_style"):
            instance._style = StyleArray()
        coll = getattr(instance.parent.parent, self.collection)
        if isinstance(value, NamedStyle):
            style = value
            if style not in coll:
                instance.parent.parent.add_named_style(style)
        elif value not in coll.names:
            if value in styles: # is it builtin?
                style = styles[value]
                if style not in coll:
                    instance.parent.parent.add_named_style(style)
            else:
                raise ValueError("{0} is not a known style".format(value))
        else:
            style = coll[value]
        instance._style = copy(style.as_tuple())


    def __get__(self, instance, cls):
        if not getattr(instance, "_style"):
            instance._style = StyleArray()
        idx = getattr(instance._style, self.key)
        coll = getattr(instance.parent.parent, self.collection)
        return coll.names[idx]


class StyleArrayDescriptor:

    def __init__(self, key):
        self.key = key

    def __set__(self, instance, value):
        if instance._style is None:
            instance._style = StyleArray()
        setattr(instance._style, self.key, value)


    def __get__(self, instance, cls):
        if instance._style is None:
            return False
        return bool(getattr(instance._style, self.key))


class StyleableObject:
    """
    Base class for styleble objects implementing proxy and lookup functions
    """

    font = StyleDescriptor('_fonts', "fontId")
    fill = StyleDescriptor('_fills', "fillId")
    border = StyleDescriptor('_borders', "borderId")
    number_format = NumberFormatDescriptor()
    protection = StyleDescriptor('_protections', "protectionId")
    alignment = StyleDescriptor('_alignments', "alignmentId")
    style = NamedStyleDescriptor()
    quotePrefix = StyleArrayDescriptor('quotePrefix')
    pivotButton = StyleArrayDescriptor('pivotButton')

    __slots__ = ('parent', '_style')

    def __init__(self, sheet, style_array=None):
        self.parent = sheet
        if style_array is not None:
            style_array = StyleArray(style_array)
        self._style = style_array


    @property
    def style_id(self):
        if self._style is None:
            self._style = StyleArray()
        return self.parent.parent._cell_styles.add(self._style)


    @property
    def has_style(self):
        if self._style is None:
            return False
        return any(self._style)


# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\properties.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    String,
    Float,
    Integer,
    Bool,
    NoneSet,
    Set,
)

from openpyxl.descriptors.excel import Guid


class WorkbookProperties(Serialisable):

    tagname = "workbookPr"

    date1904 = Bool(allow_none=True)
    dateCompatibility = Bool(allow_none=True)
    showObjects = NoneSet(values=(['all', 'placeholders']))
    showBorderUnselectedTables = Bool(allow_none=True)
    filterPrivacy = Bool(allow_none=True)
    promptedSolutions = Bool(allow_none=True)
    showInkAnnotation = Bool(allow_none=True)
    backupFile = Bool(allow_none=True)
    saveExternalLinkValues = Bool(allow_none=True)
    updateLinks = NoneSet(values=(['userSet', 'never', 'always']))
    codeName = String(allow_none=True)
    hidePivotFieldList = Bool(allow_none=True)
    showPivotChartFilter = Bool(allow_none=True)
    allowRefreshQuery = Bool(allow_none=True)
    publishItems = Bool(allow_none=True)
    checkCompatibility = Bool(allow_none=True)
    autoCompressPictures = Bool(allow_none=True)
    refreshAllConnections = Bool(allow_none=True)
    defaultThemeVersion = Integer(allow_none=True)

    def __init__(self,
                 date1904=None,
                 dateCompatibility=None,
                 showObjects=None,
                 showBorderUnselectedTables=None,
                 filterPrivacy=None,
                 promptedSolutions=None,
                 showInkAnnotation=None,
                 backupFile=None,
                 saveExternalLinkValues=None,
                 updateLinks=None,
                 codeName=None,
                 hidePivotFieldList=None,
                 showPivotChartFilter=None,
                 allowRefreshQuery=None,
                 publishItems=None,
                 checkCompatibility=None,
                 autoCompressPictures=None,
                 refreshAllConnections=None,
                 defaultThemeVersion=None,
                ):
        self.date1904 = date1904
        self.dateCompatibility = dateCompatibility
        self.showObjects = showObjects
        self.showBorderUnselectedTables = showBorderUnselectedTables
        self.filterPrivacy = filterPrivacy
        self.promptedSolutions = promptedSolutions
        self.showInkAnnotation = showInkAnnotation
        self.backupFile = backupFile
        self.saveExternalLinkValues = saveExternalLinkValues
        self.updateLinks = updateLinks
        self.codeName = codeName
        self.hidePivotFieldList = hidePivotFieldList
        self.showPivotChartFilter = showPivotChartFilter
        self.allowRefreshQuery = allowRefreshQuery
        self.publishItems = publishItems
        self.checkCompatibility = checkCompatibility
        self.autoCompressPictures = autoCompressPictures
        self.refreshAllConnections = refreshAllConnections
        self.defaultThemeVersion = defaultThemeVersion


class CalcProperties(Serialisable):

    tagname = "calcPr"

    calcId = Integer()
    calcMode = NoneSet(values=(['manual', 'auto', 'autoNoTable']))
    fullCalcOnLoad = Bool(allow_none=True)
    refMode = NoneSet(values=(['A1', 'R1C1']))
    iterate = Bool(allow_none=True)
    iterateCount = Integer(allow_none=True)
    iterateDelta = Float(allow_none=True)
    fullPrecision = Bool(allow_none=True)
    calcCompleted = Bool(allow_none=True)
    calcOnSave = Bool(allow_none=True)
    concurrentCalc = Bool(allow_none=True)
    concurrentManualCount = Integer(allow_none=True)
    forceFullCalc = Bool(allow_none=True)

    def __init__(self,
                 calcId=124519,
                 calcMode=None,
                 fullCalcOnLoad=True,
                 refMode=None,
                 iterate=None,
                 iterateCount=None,
                 iterateDelta=None,
                 fullPrecision=None,
                 calcCompleted=None,
                 calcOnSave=None,
                 concurrentCalc=None,
                 concurrentManualCount=None,
                 forceFullCalc=None,
                ):
        self.calcId = calcId
        self.calcMode = calcMode
        self.fullCalcOnLoad = fullCalcOnLoad
        self.refMode = refMode
        self.iterate = iterate
        self.iterateCount = iterateCount
        self.iterateDelta = iterateDelta
        self.fullPrecision = fullPrecision
        self.calcCompleted = calcCompleted
        self.calcOnSave = calcOnSave
        self.concurrentCalc = concurrentCalc
        self.concurrentManualCount = concurrentManualCount
        self.forceFullCalc = forceFullCalc


class FileVersion(Serialisable):

    tagname = "fileVersion"

    appName = String(allow_none=True)
    lastEdited = String(allow_none=True)
    lowestEdited = String(allow_none=True)
    rupBuild = String(allow_none=True)
    codeName = Guid(allow_none=True)

    def __init__(self,
                 appName=None,
                 lastEdited=None,
                 lowestEdited=None,
                 rupBuild=None,
                 codeName=None,
                ):
        self.appName = appName
        self.lastEdited = lastEdited
        self.lowestEdited = lowestEdited
        self.rupBuild = rupBuild
        self.codeName = codeName

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\exceptions.py ===
"""
requests.exceptions
~~~~~~~~~~~~~~~~~~~

This module contains the set of Requests' exceptions.
"""
from pip._vendor.urllib3.exceptions import HTTPError as BaseHTTPError

from .compat import JSONDecodeError as CompatJSONDecodeError


class RequestException(IOError):
    """There was an ambiguous exception that occurred while handling your
    request.
    """

    def __init__(self, *args, **kwargs):
        """Initialize RequestException with `request` and `response` objects."""
        response = kwargs.pop("response", None)
        self.response = response
        self.request = kwargs.pop("request", None)
        if response is not None and not self.request and hasattr(response, "request"):
            self.request = self.response.request
        super().__init__(*args, **kwargs)


class InvalidJSONError(RequestException):
    """A JSON error occurred."""


class JSONDecodeError(InvalidJSONError, CompatJSONDecodeError):
    """Couldn't decode the text into json"""

    def __init__(self, *args, **kwargs):
        """
        Construct the JSONDecodeError instance first with all
        args. Then use it's args to construct the IOError so that
        the json specific args aren't used as IOError specific args
        and the error message from JSONDecodeError is preserved.
        """
        CompatJSONDecodeError.__init__(self, *args)
        InvalidJSONError.__init__(self, *self.args, **kwargs)

    def __reduce__(self):
        """
        The __reduce__ method called when pickling the object must
        be the one from the JSONDecodeError (be it json/simplejson)
        as it expects all the arguments for instantiation, not just
        one like the IOError, and the MRO would by default call the
        __reduce__ method from the IOError due to the inheritance order.
        """
        return CompatJSONDecodeError.__reduce__(self)


class HTTPError(RequestException):
    """An HTTP error occurred."""


class ConnectionError(RequestException):
    """A Connection error occurred."""


class ProxyError(ConnectionError):
    """A proxy error occurred."""


class SSLError(ConnectionError):
    """An SSL error occurred."""


class Timeout(RequestException):
    """The request timed out.

    Catching this error will catch both
    :exc:`~requests.exceptions.ConnectTimeout` and
    :exc:`~requests.exceptions.ReadTimeout` errors.
    """


class ConnectTimeout(ConnectionError, Timeout):
    """The request timed out while trying to connect to the remote server.

    Requests that produced this error are safe to retry.
    """


class ReadTimeout(Timeout):
    """The server did not send any data in the allotted amount of time."""


class URLRequired(RequestException):
    """A valid URL is required to make a request."""


class TooManyRedirects(RequestException):
    """Too many redirects."""


class MissingSchema(RequestException, ValueError):
    """The URL scheme (e.g. http or https) is missing."""


class InvalidSchema(RequestException, ValueError):
    """The URL scheme provided is either invalid or unsupported."""


class InvalidURL(RequestException, ValueError):
    """The URL provided was somehow invalid."""


class InvalidHeader(RequestException, ValueError):
    """The header value provided was somehow invalid."""


class InvalidProxyURL(InvalidURL):
    """The proxy URL provided is invalid."""


class ChunkedEncodingError(RequestException):
    """The server declared chunked encoding but sent an invalid chunk."""


class ContentDecodingError(RequestException, BaseHTTPError):
    """Failed to decode response content."""


class StreamConsumedError(RequestException, TypeError):
    """The content for this response was already consumed."""


class RetryError(RequestException):
    """Custom retries logic failed"""


class UnrewindableBodyError(RequestException):
    """Requests encountered an error when trying to rewind a body."""


# Warnings


class RequestsWarning(Warning):
    """Base warning for Requests."""


class FileModeWarning(RequestsWarning, DeprecationWarning):
    """A file was opened in text mode, but Requests determined its binary length."""


class RequestsDependencyWarning(RequestsWarning):
    """An imported dependency doesn't match the expected version range."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\measure.py ===
from operator import itemgetter
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Sequence

from . import errors
from .protocol import is_renderable, rich_cast

if TYPE_CHECKING:
    from .console import Console, ConsoleOptions, RenderableType


class Measurement(NamedTuple):
    """Stores the minimum and maximum widths (in characters) required to render an object."""

    minimum: int
    """Minimum number of cells required to render."""
    maximum: int
    """Maximum number of cells required to render."""

    @property
    def span(self) -> int:
        """Get difference between maximum and minimum."""
        return self.maximum - self.minimum

    def normalize(self) -> "Measurement":
        """Get measurement that ensures that minimum <= maximum and minimum >= 0

        Returns:
            Measurement: A normalized measurement.
        """
        minimum, maximum = self
        minimum = min(max(0, minimum), maximum)
        return Measurement(max(0, minimum), max(0, max(minimum, maximum)))

    def with_maximum(self, width: int) -> "Measurement":
        """Get a RenderableWith where the widths are <= width.

        Args:
            width (int): Maximum desired width.

        Returns:
            Measurement: New Measurement object.
        """
        minimum, maximum = self
        return Measurement(min(minimum, width), min(maximum, width))

    def with_minimum(self, width: int) -> "Measurement":
        """Get a RenderableWith where the widths are >= width.

        Args:
            width (int): Minimum desired width.

        Returns:
            Measurement: New Measurement object.
        """
        minimum, maximum = self
        width = max(0, width)
        return Measurement(max(minimum, width), max(maximum, width))

    def clamp(
        self, min_width: Optional[int] = None, max_width: Optional[int] = None
    ) -> "Measurement":
        """Clamp a measurement within the specified range.

        Args:
            min_width (int): Minimum desired width, or ``None`` for no minimum. Defaults to None.
            max_width (int): Maximum desired width, or ``None`` for no maximum. Defaults to None.

        Returns:
            Measurement: New Measurement object.
        """
        measurement = self
        if min_width is not None:
            measurement = measurement.with_minimum(min_width)
        if max_width is not None:
            measurement = measurement.with_maximum(max_width)
        return measurement

    @classmethod
    def get(
        cls, console: "Console", options: "ConsoleOptions", renderable: "RenderableType"
    ) -> "Measurement":
        """Get a measurement for a renderable.

        Args:
            console (~rich.console.Console): Console instance.
            options (~rich.console.ConsoleOptions): Console options.
            renderable (RenderableType): An object that may be rendered with Rich.

        Raises:
            errors.NotRenderableError: If the object is not renderable.

        Returns:
            Measurement: Measurement object containing range of character widths required to render the object.
        """
        _max_width = options.max_width
        if _max_width < 1:
            return Measurement(0, 0)
        if isinstance(renderable, str):
            renderable = console.render_str(
                renderable, markup=options.markup, highlight=False
            )
        renderable = rich_cast(renderable)
        if is_renderable(renderable):
            get_console_width: Optional[
                Callable[["Console", "ConsoleOptions"], "Measurement"]
            ] = getattr(renderable, "__rich_measure__", None)
            if get_console_width is not None:
                render_width = (
                    get_console_width(console, options)
                    .normalize()
                    .with_maximum(_max_width)
                )
                if render_width.maximum < 1:
                    return Measurement(0, 0)
                return render_width.normalize()
            else:
                return Measurement(0, _max_width)
        else:
            raise errors.NotRenderableError(
                f"Unable to get render width for {renderable!r}; "
                "a str, Segment, or object with __rich_console__ method is required"
            )


def measure_renderables(
    console: "Console",
    options: "ConsoleOptions",
    renderables: Sequence["RenderableType"],
) -> "Measurement":
    """Get a measurement that would fit a number of renderables.

    Args:
        console (~rich.console.Console): Console instance.
        options (~rich.console.ConsoleOptions): Console options.
        renderables (Iterable[RenderableType]): One or more renderable objects.

    Returns:
        Measurement: Measurement object containing range of character widths required to
            contain all given renderables.
    """
    if not renderables:
        return Measurement(0, 0)
    get_measurement = Measurement.get
    measurements = [
        get_measurement(console, options, renderable) for renderable in renderables
    ]
    measured_width = Measurement(
        max(measurements, key=itemgetter(0)).minimum,
        max(measurements, key=itemgetter(1)).maximum,
    )
    return measured_width

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\exceptions.py ===
"""
requests.exceptions
~~~~~~~~~~~~~~~~~~~

This module contains the set of Requests' exceptions.
"""
from urllib3.exceptions import HTTPError as BaseHTTPError

from .compat import JSONDecodeError as CompatJSONDecodeError


class RequestException(IOError):
    """There was an ambiguous exception that occurred while handling your
    request.
    """

    def __init__(self, *args, **kwargs):
        """Initialize RequestException with `request` and `response` objects."""
        response = kwargs.pop("response", None)
        self.response = response
        self.request = kwargs.pop("request", None)
        if response is not None and not self.request and hasattr(response, "request"):
            self.request = self.response.request
        super().__init__(*args, **kwargs)


class InvalidJSONError(RequestException):
    """A JSON error occurred."""


class JSONDecodeError(InvalidJSONError, CompatJSONDecodeError):
    """Couldn't decode the text into json"""

    def __init__(self, *args, **kwargs):
        """
        Construct the JSONDecodeError instance first with all
        args. Then use it's args to construct the IOError so that
        the json specific args aren't used as IOError specific args
        and the error message from JSONDecodeError is preserved.
        """
        CompatJSONDecodeError.__init__(self, *args)
        InvalidJSONError.__init__(self, *self.args, **kwargs)

    def __reduce__(self):
        """
        The __reduce__ method called when pickling the object must
        be the one from the JSONDecodeError (be it json/simplejson)
        as it expects all the arguments for instantiation, not just
        one like the IOError, and the MRO would by default call the
        __reduce__ method from the IOError due to the inheritance order.
        """
        return CompatJSONDecodeError.__reduce__(self)


class HTTPError(RequestException):
    """An HTTP error occurred."""


class ConnectionError(RequestException):
    """A Connection error occurred."""


class ProxyError(ConnectionError):
    """A proxy error occurred."""


class SSLError(ConnectionError):
    """An SSL error occurred."""


class Timeout(RequestException):
    """The request timed out.

    Catching this error will catch both
    :exc:`~requests.exceptions.ConnectTimeout` and
    :exc:`~requests.exceptions.ReadTimeout` errors.
    """


class ConnectTimeout(ConnectionError, Timeout):
    """The request timed out while trying to connect to the remote server.

    Requests that produced this error are safe to retry.
    """


class ReadTimeout(Timeout):
    """The server did not send any data in the allotted amount of time."""


class URLRequired(RequestException):
    """A valid URL is required to make a request."""


class TooManyRedirects(RequestException):
    """Too many redirects."""


class MissingSchema(RequestException, ValueError):
    """The URL scheme (e.g. http or https) is missing."""


class InvalidSchema(RequestException, ValueError):
    """The URL scheme provided is either invalid or unsupported."""


class InvalidURL(RequestException, ValueError):
    """The URL provided was somehow invalid."""


class InvalidHeader(RequestException, ValueError):
    """The header value provided was somehow invalid."""


class InvalidProxyURL(InvalidURL):
    """The proxy URL provided is invalid."""


class ChunkedEncodingError(RequestException):
    """The server declared chunked encoding but sent an invalid chunk."""


class ContentDecodingError(RequestException, BaseHTTPError):
    """Failed to decode response content."""


class StreamConsumedError(RequestException, TypeError):
    """The content for this response was already consumed."""


class RetryError(RequestException):
    """Custom retries logic failed"""


class UnrewindableBodyError(RequestException):
    """Requests encountered an error when trying to rewind a body."""


# Warnings


class RequestsWarning(Warning):
    """Base warning for Requests."""


class FileModeWarning(RequestsWarning, DeprecationWarning):
    """A file was opened in text mode, but Requests determined its binary length."""


class RequestsDependencyWarning(RequestsWarning):
    """An imported dependency doesn't match the expected version range."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\strategies\core.py ===
""" Generic SymPy-Independent Strategies """
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import TypeVar
from sys import stdout


_S = TypeVar('_S')
_T = TypeVar('_T')


def identity(x: _T) -> _T:
    return x


def exhaust(rule: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """ Apply a rule repeatedly until it has no effect """
    def exhaustive_rl(expr: _T) -> _T:
        new, old = rule(expr), expr
        while new != old:
            new, old = rule(new), new
        return new
    return exhaustive_rl


def memoize(rule: Callable[[_S], _T]) -> Callable[[_S], _T]:
    """Memoized version of a rule

    Notes
    =====

    This cache can grow infinitely, so it is not recommended to use this
    than ``functools.lru_cache`` unless you need very heavy computation.
    """
    cache: dict[_S, _T] = {}

    def memoized_rl(expr: _S) -> _T:
        if expr in cache:
            return cache[expr]
        else:
            result = rule(expr)
            cache[expr] = result
            return result
    return memoized_rl


def condition(
    cond: Callable[[_T], bool], rule: Callable[[_T], _T]
) -> Callable[[_T], _T]:
    """ Only apply rule if condition is true """
    def conditioned_rl(expr: _T) -> _T:
        if cond(expr):
            return rule(expr)
        return expr
    return conditioned_rl


def chain(*rules: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """
    Compose a sequence of rules so that they apply to the expr sequentially
    """
    def chain_rl(expr: _T) -> _T:
        for rule in rules:
            expr = rule(expr)
        return expr
    return chain_rl


def debug(rule, file=None):
    """ Print out before and after expressions each time rule is used """
    if file is None:
        file = stdout

    def debug_rl(*args, **kwargs):
        expr = args[0]
        result = rule(*args, **kwargs)
        if result != expr:
            file.write("Rule: %s\n" % rule.__name__)
            file.write("In:   %s\nOut:  %s\n\n" % (expr, result))
        return result
    return debug_rl


def null_safe(rule: Callable[[_T], _T | None]) -> Callable[[_T], _T]:
    """ Return original expr if rule returns None """
    def null_safe_rl(expr: _T) -> _T:
        result = rule(expr)
        if result is None:
            return expr
        return result
    return null_safe_rl


def tryit(rule: Callable[[_T], _T], exception) -> Callable[[_T], _T]:
    """ Return original expr if rule raises exception """
    def try_rl(expr: _T) -> _T:
        try:
            return rule(expr)
        except exception:
            return expr
    return try_rl


def do_one(*rules: Callable[[_T], _T]) -> Callable[[_T], _T]:
    """ Try each of the rules until one works. Then stop. """
    def do_one_rl(expr: _T) -> _T:
        for rl in rules:
            result = rl(expr)
            if result != expr:
                return result
        return expr
    return do_one_rl


def switch(
    key: Callable[[_S], _T],
    ruledict: Mapping[_T, Callable[[_S], _S]]
) -> Callable[[_S], _S]:
    """ Select a rule based on the result of key called on the function """
    def switch_rl(expr: _S) -> _S:
        rl = ruledict.get(key(expr), identity)
        return rl(expr)
    return switch_rl


# XXX Untyped default argument for minimize function
# where python requires SupportsRichComparison type
def _identity(x):
    return x


def minimize(
    *rules: Callable[[_S], _T],
    objective=_identity
) -> Callable[[_S], _T]:
    """ Select result of rules that minimizes objective

    >>> from sympy.strategies import minimize
    >>> inc = lambda x: x + 1
    >>> dec = lambda x: x - 1
    >>> rl = minimize(inc, dec)
    >>> rl(4)
    3

    >>> rl = minimize(inc, dec, objective=lambda x: -x)  # maximize
    >>> rl(4)
    5
    """
    def minrule(expr: _S) -> _T:
        return min([rule(expr) for rule in rules], key=objective)
    return minrule

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\preloaded.py ===
# util/preloaded.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""supplies the "preloaded" registry to resolve circular module imports at
runtime.

"""
from __future__ import annotations

import sys
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
from typing import TypeVar

_FN = TypeVar("_FN", bound=Callable[..., Any])


if TYPE_CHECKING:
    from sqlalchemy import dialects as _dialects
    from sqlalchemy import orm as _orm
    from sqlalchemy.engine import cursor as _engine_cursor
    from sqlalchemy.engine import default as _engine_default
    from sqlalchemy.engine import reflection as _engine_reflection
    from sqlalchemy.engine import result as _engine_result
    from sqlalchemy.engine import url as _engine_url
    from sqlalchemy.orm import attributes as _orm_attributes
    from sqlalchemy.orm import base as _orm_base
    from sqlalchemy.orm import clsregistry as _orm_clsregistry
    from sqlalchemy.orm import decl_api as _orm_decl_api
    from sqlalchemy.orm import decl_base as _orm_decl_base
    from sqlalchemy.orm import dependency as _orm_dependency
    from sqlalchemy.orm import descriptor_props as _orm_descriptor_props
    from sqlalchemy.orm import mapperlib as _orm_mapper
    from sqlalchemy.orm import properties as _orm_properties
    from sqlalchemy.orm import relationships as _orm_relationships
    from sqlalchemy.orm import session as _orm_session
    from sqlalchemy.orm import state as _orm_state
    from sqlalchemy.orm import strategies as _orm_strategies
    from sqlalchemy.orm import strategy_options as _orm_strategy_options
    from sqlalchemy.orm import util as _orm_util
    from sqlalchemy.sql import default_comparator as _sql_default_comparator
    from sqlalchemy.sql import dml as _sql_dml
    from sqlalchemy.sql import elements as _sql_elements
    from sqlalchemy.sql import functions as _sql_functions
    from sqlalchemy.sql import naming as _sql_naming
    from sqlalchemy.sql import schema as _sql_schema
    from sqlalchemy.sql import selectable as _sql_selectable
    from sqlalchemy.sql import sqltypes as _sql_sqltypes
    from sqlalchemy.sql import traversals as _sql_traversals
    from sqlalchemy.sql import util as _sql_util

    # sigh, appease mypy 0.971 which does not accept imports as instance
    # variables of a module
    dialects = _dialects
    engine_cursor = _engine_cursor
    engine_default = _engine_default
    engine_reflection = _engine_reflection
    engine_result = _engine_result
    engine_url = _engine_url
    orm_clsregistry = _orm_clsregistry
    orm_base = _orm_base
    orm = _orm
    orm_attributes = _orm_attributes
    orm_decl_api = _orm_decl_api
    orm_decl_base = _orm_decl_base
    orm_descriptor_props = _orm_descriptor_props
    orm_dependency = _orm_dependency
    orm_mapper = _orm_mapper
    orm_properties = _orm_properties
    orm_relationships = _orm_relationships
    orm_session = _orm_session
    orm_strategies = _orm_strategies
    orm_strategy_options = _orm_strategy_options
    orm_state = _orm_state
    orm_util = _orm_util
    sql_default_comparator = _sql_default_comparator
    sql_dml = _sql_dml
    sql_elements = _sql_elements
    sql_functions = _sql_functions
    sql_naming = _sql_naming
    sql_selectable = _sql_selectable
    sql_traversals = _sql_traversals
    sql_schema = _sql_schema
    sql_sqltypes = _sql_sqltypes
    sql_util = _sql_util


class _ModuleRegistry:
    """Registry of modules to load in a package init file.

    To avoid potential thread safety issues for imports that are deferred
    in a function, like https://bugs.python.org/issue38884, these modules
    are added to the system module cache by importing them after the packages
    has finished initialization.

    A global instance is provided under the name :attr:`.preloaded`. Use
    the function :func:`.preload_module` to register modules to load and
    :meth:`.import_prefix` to load all the modules that start with the
    given path.

    While the modules are loaded in the global module cache, it's advisable
    to access them using :attr:`.preloaded` to ensure that it was actually
    registered. Each registered module is added to the instance ``__dict__``
    in the form `<package>_<module>`, omitting ``sqlalchemy`` from the package
    name. Example: ``sqlalchemy.sql.util`` becomes ``preloaded.sql_util``.
    """

    def __init__(self, prefix="sqlalchemy."):
        self.module_registry = set()
        self.prefix = prefix

    def preload_module(self, *deps: str) -> Callable[[_FN], _FN]:
        """Adds the specified modules to the list to load.

        This method can be used both as a normal function and as a decorator.
        No change is performed to the decorated object.
        """
        self.module_registry.update(deps)
        return lambda fn: fn

    def import_prefix(self, path: str) -> None:
        """Resolve all the modules in the registry that start with the
        specified path.
        """
        for module in self.module_registry:
            if self.prefix:
                key = module.split(self.prefix)[-1].replace(".", "_")
            else:
                key = module
            if (
                not path or module.startswith(path)
            ) and key not in self.__dict__:
                __import__(module, globals(), locals())
                self.__dict__[key] = globals()[key] = sys.modules[module]


_reg = _ModuleRegistry()
preload_module = _reg.preload_module
import_prefix = _reg.import_prefix

# this appears to do absolutely nothing for any version of mypy
# if TYPE_CHECKING:
#    def __getattr__(key: str) -> ModuleType:
#        ...

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\matpow.py ===
from .matexpr import MatrixExpr
from .special import Identity
from sympy.core import S
from sympy.core.expr import ExprBuilder
from sympy.core.cache import cacheit
from sympy.core.power import Pow
from sympy.core.sympify import _sympify
from sympy.matrices import MatrixBase
from sympy.matrices.exceptions import NonSquareMatrixError


class MatPow(MatrixExpr):
    def __new__(cls, base, exp, evaluate=False, **options):
        base = _sympify(base)
        if not base.is_Matrix:
            raise TypeError("MatPow base should be a matrix")

        if base.is_square is False:
            raise NonSquareMatrixError("Power of non-square matrix %s" % base)

        exp = _sympify(exp)
        obj = super().__new__(cls, base, exp)

        if evaluate:
            obj = obj.doit(deep=False)

        return obj

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

    @property
    def shape(self):
        return self.base.shape

    @cacheit
    def _get_explicit_matrix(self):
        return self.base.as_explicit()**self.exp

    def _entry(self, i, j, **kwargs):
        from sympy.matrices.expressions import MatMul
        A = self.doit()
        if isinstance(A, MatPow):
            # We still have a MatPow, make an explicit MatMul out of it.
            if A.exp.is_Integer and A.exp.is_positive:
                A = MatMul(*[A.base for k in range(A.exp)])
            elif not self._is_shape_symbolic():
                return A._get_explicit_matrix()[i, j]
            else:
                # Leave the expression unevaluated:
                from sympy.matrices.expressions.matexpr import MatrixElement
                return MatrixElement(self, i, j)
        return A[i, j]

    def doit(self, **hints):
        if hints.get('deep', True):
            base, exp = (arg.doit(**hints) for arg in self.args)
        else:
            base, exp = self.args

        # combine all powers, e.g. (A ** 2) ** 3 -> A ** 6
        while isinstance(base, MatPow):
            exp *= base.args[1]
            base = base.args[0]

        if isinstance(base, MatrixBase):
            # Delegate
            return base ** exp

        # Handle simple cases so that _eval_power() in MatrixExpr sub-classes can ignore them
        if exp == S.One:
            return base
        if exp == S.Zero:
            return Identity(base.rows)
        if exp == S.NegativeOne:
            from sympy.matrices.expressions import Inverse
            return Inverse(base).doit(**hints)

        eval_power = getattr(base, '_eval_power', None)
        if eval_power is not None:
            return eval_power(exp)

        return MatPow(base, exp)

    def _eval_transpose(self):
        base, exp = self.args
        return MatPow(base.transpose(), exp)

    def _eval_adjoint(self):
        base, exp = self.args
        return MatPow(base.adjoint(), exp)

    def _eval_conjugate(self):
        base, exp = self.args
        return MatPow(base.conjugate(), exp)

    def _eval_derivative(self, x):
        return Pow._eval_derivative(self, x)

    def _eval_derivative_matrix_lines(self, x):
        from sympy.tensor.array.expressions.array_expressions import ArrayContraction
        from ...tensor.array.expressions.array_expressions import ArrayTensorProduct
        from .matmul import MatMul
        from .inverse import Inverse
        exp = self.exp
        if self.base.shape == (1, 1) and not exp.has(x):
            lr = self.base._eval_derivative_matrix_lines(x)
            for i in lr:
                subexpr = ExprBuilder(
                    ArrayContraction,
                    [
                        ExprBuilder(
                            ArrayTensorProduct,
                            [
                                Identity(1),
                                i._lines[0],
                                exp*self.base**(exp-1),
                                i._lines[1],
                                Identity(1),
                            ]
                        ),
                        (0, 3, 4), (5, 7, 8)
                    ],
                    validator=ArrayContraction._validate
                )
                i._first_pointer_parent = subexpr.args[0].args
                i._first_pointer_index = 0
                i._second_pointer_parent = subexpr.args[0].args
                i._second_pointer_index = 4
                i._lines = [subexpr]
            return lr
        if (exp > 0) == True:
            newexpr = MatMul.fromiter([self.base for i in range(exp)])
        elif (exp == -1) == True:
            return Inverse(self.base)._eval_derivative_matrix_lines(x)
        elif (exp < 0) == True:
            newexpr = MatMul.fromiter([Inverse(self.base) for i in range(-exp)])
        elif (exp == 0) == True:
            return self.doit()._eval_derivative_matrix_lines(x)
        else:
            raise NotImplementedError("cannot evaluate %s derived by %s" % (self, x))
        return newexpr._eval_derivative_matrix_lines(x)

    def _eval_inverse(self):
        return MatPow(self.base, -self.exp)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\digits.py ===
from collections import defaultdict

from sympy.utilities.iterables import multiset, is_palindromic as _palindromic
from sympy.utilities.misc import as_int


def digits(n, b=10, digits=None):
    """
    Return a list of the digits of ``n`` in base ``b``. The first
    element in the list is ``b`` (or ``-b`` if ``n`` is negative).

    Examples
    ========

    >>> from sympy.ntheory.digits import digits
    >>> digits(35)
    [10, 3, 5]

    If the number is negative, the negative sign will be placed on the
    base (which is the first element in the returned list):

    >>> digits(-35)
    [-10, 3, 5]

    Bases other than 10 (and greater than 1) can be selected with ``b``:

    >>> digits(27, b=2)
    [2, 1, 1, 0, 1, 1]

    Use the ``digits`` keyword if a certain number of digits is desired:

    >>> digits(35, digits=4)
    [10, 0, 0, 3, 5]

    Parameters
    ==========

    n: integer
        The number whose digits are returned.

    b: integer
        The base in which digits are computed.

    digits: integer (or None for all digits)
        The number of digits to be returned (padded with zeros, if
        necessary).

    See Also
    ========
    sympy.core.intfunc.num_digits, count_digits
    """

    b = as_int(b)
    n = as_int(n)
    if b < 2:
        raise ValueError("b must be greater than 1")
    else:
        x, y = abs(n), []
        while x >= b:
            x, r = divmod(x, b)
            y.append(r)
        y.append(x)
        y.append(-b if n < 0 else b)
        y.reverse()
        ndig = len(y) - 1
        if digits is not None:
            if ndig > digits:
                raise ValueError(
                    "For %s, at least %s digits are needed." % (n, ndig))
            elif ndig < digits:
                y[1:1] = [0]*(digits - ndig)
        return y


def count_digits(n, b=10):
    """
    Return a dictionary whose keys are the digits of ``n`` in the
    given base, ``b``, with keys indicating the digits appearing in the
    number and values indicating how many times that digit appeared.

    Examples
    ========

    >>> from sympy.ntheory import count_digits

    >>> count_digits(1111339)
    {1: 4, 3: 2, 9: 1}

    The digits returned are always represented in base-10
    but the number itself can be entered in any format that is
    understood by Python; the base of the number can also be
    given if it is different than 10:

    >>> n = 0xFA; n
    250
    >>> count_digits(_)
    {0: 1, 2: 1, 5: 1}
    >>> count_digits(n, 16)
    {10: 1, 15: 1}

    The default dictionary will return a 0 for any digit that did
    not appear in the number. For example, which digits appear 7
    times in ``77!``:

    >>> from sympy import factorial
    >>> c77 = count_digits(factorial(77))
    >>> [i for i in range(10) if c77[i] == 7]
    [1, 3, 7, 9]

    See Also
    ========
    sympy.core.intfunc.num_digits, digits
    """
    rv = defaultdict(int, multiset(digits(n, b)).items())
    rv.pop(b) if b in rv else rv.pop(-b)  # b or -b is there
    return rv


def is_palindromic(n, b=10):
    """return True if ``n`` is the same when read from left to right
    or right to left in the given base, ``b``.

    Examples
    ========

    >>> from sympy.ntheory import is_palindromic

    >>> all(is_palindromic(i) for i in (-11, 1, 22, 121))
    True

    The second argument allows you to test numbers in other
    bases. For example, 88 is palindromic in base-10 but not
    in base-8:

    >>> is_palindromic(88, 8)
    False

    On the other hand, a number can be palindromic in base-8 but
    not in base-10:

    >>> 0o121, is_palindromic(0o121)
    (81, False)

    Or it might be palindromic in both bases:

    >>> oct(121), is_palindromic(121, 8) and is_palindromic(121)
    ('0o171', True)

    """
    return _palindromic(digits(n, b), 1)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\f2py\diagnose.py ===
#!/usr/bin/env python3
import os
import sys
import tempfile


def run():
    _path = os.getcwd()
    os.chdir(tempfile.gettempdir())
    print('------')
    print(f'os.name={os.name!r}')
    print('------')
    print(f'sys.platform={sys.platform!r}')
    print('------')
    print('sys.version:')
    print(sys.version)
    print('------')
    print('sys.prefix:')
    print(sys.prefix)
    print('------')
    print(f"sys.path={':'.join(sys.path)!r}")
    print('------')

    try:
        import numpy
        has_newnumpy = 1
    except ImportError as e:
        print('Failed to import new numpy:', e)
        has_newnumpy = 0

    try:
        from numpy.f2py import f2py2e
        has_f2py2e = 1
    except ImportError as e:
        print('Failed to import f2py2e:', e)
        has_f2py2e = 0

    try:
        import numpy.distutils
        has_numpy_distutils = 2
    except ImportError:
        try:
            import numpy_distutils
            has_numpy_distutils = 1
        except ImportError as e:
            print('Failed to import numpy_distutils:', e)
            has_numpy_distutils = 0

    if has_newnumpy:
        try:
            print(f'Found new numpy version {numpy.__version__!r} in {numpy.__file__}')
        except Exception as msg:
            print('error:', msg)
            print('------')

    if has_f2py2e:
        try:
            print('Found f2py2e version %r in %s' %
                  (f2py2e.__version__.version, f2py2e.__file__))
        except Exception as msg:
            print('error:', msg)
            print('------')

    if has_numpy_distutils:
        try:
            if has_numpy_distutils == 2:
                print('Found numpy.distutils version %r in %r' % (
                    numpy.distutils.__version__,
                    numpy.distutils.__file__))
            else:
                print('Found numpy_distutils version %r in %r' % (
                    numpy_distutils.numpy_distutils_version.numpy_distutils_version,
                    numpy_distutils.__file__))
            print('------')
        except Exception as msg:
            print('error:', msg)
            print('------')
        try:
            if has_numpy_distutils == 1:
                print(
                    'Importing numpy_distutils.command.build_flib ...', end=' ')
                import numpy_distutils.command.build_flib as build_flib
                print('ok')
                print('------')
                try:
                    print(
                        'Checking availability of supported Fortran compilers:')
                    for compiler_class in build_flib.all_compilers:
                        compiler_class(verbose=1).is_available()
                        print('------')
                except Exception as msg:
                    print('error:', msg)
                    print('------')
        except Exception as msg:
            print(
                'error:', msg, '(ignore it, build_flib is obsolete for numpy.distutils 0.2.2 and up)')
            print('------')
        try:
            if has_numpy_distutils == 2:
                print('Importing numpy.distutils.fcompiler ...', end=' ')
                import numpy.distutils.fcompiler as fcompiler
            else:
                print('Importing numpy_distutils.fcompiler ...', end=' ')
                import numpy_distutils.fcompiler as fcompiler
            print('ok')
            print('------')
            try:
                print('Checking availability of supported Fortran compilers:')
                fcompiler.show_fcompilers()
                print('------')
            except Exception as msg:
                print('error:', msg)
                print('------')
        except Exception as msg:
            print('error:', msg)
            print('------')
        try:
            if has_numpy_distutils == 2:
                print('Importing numpy.distutils.cpuinfo ...', end=' ')
                from numpy.distutils.cpuinfo import cpuinfo
                print('ok')
                print('------')
            else:
                try:
                    print(
                        'Importing numpy_distutils.command.cpuinfo ...', end=' ')
                    from numpy_distutils.command.cpuinfo import cpuinfo
                    print('ok')
                    print('------')
                except Exception as msg:
                    print('error:', msg, '(ignore it)')
                    print('Importing numpy_distutils.cpuinfo ...', end=' ')
                    from numpy_distutils.cpuinfo import cpuinfo
                    print('ok')
                    print('------')
            cpu = cpuinfo()
            print('CPU information:', end=' ')
            for name in dir(cpuinfo):
                if name[0] == '_' and name[1] != '_' and getattr(cpu, name[1:])():
                    print(name[1:], end=' ')
            print('------')
        except Exception as msg:
            print('error:', msg)
            print('------')
    os.chdir(_path)


if __name__ == "__main__":
    run()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\past_helper.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging

import torch

logger = logging.getLogger(__name__)


class PastKeyValuesHelper:
    """Helper functions to process past key values for encoder-decoder model"""

    @staticmethod
    def get_past_names(num_layers, present: bool = False):
        past_self_names = []
        past_cross_names = []
        for i in range(num_layers):
            past_self_names.extend(
                [f"present_key_self_{i}", f"present_value_self_{i}"]
                if present
                else [f"past_key_self_{i}", f"past_value_self_{i}"]
            )
            past_cross_names.extend(
                [f"present_key_cross_{i}", f"present_value_cross_{i}"]
                if present
                else [f"past_key_cross_{i}", f"past_value_cross_{i}"]
            )
        return past_self_names + past_cross_names

    @staticmethod
    def group_by_self_or_cross(present_key_values):
        """Split present state from grouped by layer to grouped by self/cross attention.
        Before: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0), (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1), ...
        After: (past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...), (past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...)

        """
        present_self = []
        present_cross = []
        for _i, present_layer_i in enumerate(present_key_values):
            assert len(present_layer_i) == 4, f"Expected to have four items. Got {len(present_layer_i)}"
            (
                present_key_self,
                present_value_self,
                present_key_cross,
                present_value_cross,
            ) = present_layer_i
            present_self.extend([present_key_self, present_value_self])
            present_cross.extend([present_key_cross, present_value_cross])
        return present_self, present_cross

    @staticmethod
    def group_by_layer(past, num_layers):
        """Reorder past state from grouped by self/cross attention to grouped by layer.
        Before: past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ..., past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...
        After: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0), (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1),
        """
        assert len(past) == 4 * num_layers
        return tuple(
            [
                past[2 * i],
                past[2 * i + 1],
                past[2 * num_layers + 2 * i],
                past[2 * num_layers + 2 * i + 1],
            ]
            for i in range(num_layers)
        )

    @staticmethod
    def back_group_by_layer(past_key_values: tuple[tuple[torch.Tensor]]):
        """Categorize present_key_values from self and cross attention to layer by layer.

        Reorder past state from grouped by self/cross attention to grouped by layer.
        Before: past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...,
                past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...
        After: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
                (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1),

        Args:
            present_key_values: From past_key_values of a model (group by self and cross attention)

        Returns:
            past_tuples: present key and values grouped by layer.
        """
        past_tuples = ()
        half_idx = len(past_key_values) // 2
        for i in range(len(past_key_values) // 4):
            idx = 2 * i
            past_tuples += (
                (
                    past_key_values[idx],
                    past_key_values[idx + 1],
                    past_key_values[half_idx + idx],
                    past_key_values[half_idx + idx + 1],
                ),
            )
        return past_tuples

    @staticmethod
    def group_by_self_and_cross(present_key_values: tuple[torch.Tensor], concat: bool = False):
        """Categorize present_key_values into self and cross attention.

        Split present state from grouped by layer to grouped by self/cross attention.
        Before: (past_key_self_0, past_value_self_0, past_key_cross_0, past_value_cross_0),
                (past_key_self_1, past_value_self_1, past_key_cross_1, past_value_cross_1), ...
        After: (past_key_self_0, past_value_self_0, past_key_self_1, past_value_self_1, ...),
                (past_key_cross_0, past_value_cross_0, past_key_cross_1, past_value_cross_1, ...)

        Args:
            present_key_values: From past_key_values of a model (group by layer)
            concat: If concat self attention with cross attention key/value to return

        Returns:
            present_self (Tuple[torch.Tensor]): present key and values from self attention
            present_cross (Tuple[torch.Tensor]): present key and values from cross attention
        """
        present_self: list[torch.Tensor] = []
        present_cross: list[torch.Tensor] = []
        for _, present_layer_i in enumerate(present_key_values):
            assert len(present_layer_i) == 4, f"Expected to have four items. Got {len(present_layer_i)}"
            present_key_self, present_value_self, present_key_cross, present_value_cross = present_layer_i
            present_self.extend([present_key_self, present_value_self])
            present_cross.extend([present_key_cross, present_value_cross])
        if concat:
            return present_self + present_cross
        else:
            return present_self, present_cross

    @staticmethod
    def get_input_names(past_key_values: tuple[tuple[torch.Tensor]], encoder=True):
        """Process input names of model wrapper.

        Args:
            past_key_values: Consider `self` and `cross` past_key_values

        Returns:
            names (List[string]): input names
        """
        names = []
        num_layers = len(past_key_values) // 4 if encoder else len(past_key_values)
        prefix = "past_" if not encoder else "present_"
        for i in range(num_layers):
            names.extend([prefix + s for s in [f"key_self_{i}", f"value_self_{i}"]])
        for i in range(num_layers):
            names.extend([prefix + s for s in [f"key_cross_{i}", f"value_cross_{i}"]])
        return names

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\array_algos\putmask.py ===
"""
EA-compatible analogue to np.putmask
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import is_list_like

from pandas.core.arrays import ExtensionArray

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        npt,
    )

    from pandas import MultiIndex


def putmask_inplace(values: ArrayLike, mask: npt.NDArray[np.bool_], value: Any) -> None:
    """
    ExtensionArray-compatible implementation of np.putmask.  The main
    difference is we do not handle repeating or truncating like numpy.

    Parameters
    ----------
    values: np.ndarray or ExtensionArray
    mask : np.ndarray[bool]
        We assume extract_bool_array has already been called.
    value : Any
    """

    if (
        not isinstance(values, np.ndarray)
        or (values.dtype == object and not lib.is_scalar(value))
        # GH#43424: np.putmask raises TypeError if we cannot cast between types with
        # rule = "safe", a stricter guarantee we may not have here
        or (
            isinstance(value, np.ndarray) and not np.can_cast(value.dtype, values.dtype)
        )
    ):
        # GH#19266 using np.putmask gives unexpected results with listlike value
        #  along with object dtype
        if is_list_like(value) and len(value) == len(values):
            values[mask] = value[mask]
        else:
            values[mask] = value
    else:
        # GH#37833 np.putmask is more performant than __setitem__
        np.putmask(values, mask, value)


def putmask_without_repeat(
    values: np.ndarray, mask: npt.NDArray[np.bool_], new: Any
) -> None:
    """
    np.putmask will truncate or repeat if `new` is a listlike with
    len(new) != len(values).  We require an exact match.

    Parameters
    ----------
    values : np.ndarray
    mask : np.ndarray[bool]
    new : Any
    """
    if getattr(new, "ndim", 0) >= 1:
        new = new.astype(values.dtype, copy=False)

    # TODO: this prob needs some better checking for 2D cases
    nlocs = mask.sum()
    if nlocs > 0 and is_list_like(new) and getattr(new, "ndim", 1) == 1:
        shape = np.shape(new)
        # np.shape compat for if setitem_datetimelike_compat
        #  changed arraylike to list e.g. test_where_dt64_2d
        if nlocs == shape[-1]:
            # GH#30567
            # If length of ``new`` is less than the length of ``values``,
            # `np.putmask` would first repeat the ``new`` array and then
            # assign the masked values hence produces incorrect result.
            # `np.place` on the other hand uses the ``new`` values at it is
            # to place in the masked locations of ``values``
            np.place(values, mask, new)
            # i.e. values[mask] = new
        elif mask.shape[-1] == shape[-1] or shape[-1] == 1:
            np.putmask(values, mask, new)
        else:
            raise ValueError("cannot assign mismatch length to masked array")
    else:
        np.putmask(values, mask, new)


def validate_putmask(
    values: ArrayLike | MultiIndex, mask: np.ndarray
) -> tuple[npt.NDArray[np.bool_], bool]:
    """
    Validate mask and check if this putmask operation is a no-op.
    """
    mask = extract_bool_array(mask)
    if mask.shape != values.shape:
        raise ValueError("putmask: mask and data must be the same size")

    noop = not mask.any()
    return mask, noop


def extract_bool_array(mask: ArrayLike) -> npt.NDArray[np.bool_]:
    """
    If we have a SparseArray or BooleanArray, convert it to ndarray[bool].
    """
    if isinstance(mask, ExtensionArray):
        # We could have BooleanArray, Sparse[bool], ...
        #  Except for BooleanArray, this is equivalent to just
        #  np.asarray(mask, dtype=bool)
        mask = mask.to_numpy(dtype=bool, na_value=False)

    mask = np.asarray(mask, dtype=bool)
    return mask


def setitem_datetimelike_compat(values: np.ndarray, num_set: int, other):
    """
    Parameters
    ----------
    values : np.ndarray
    num_set : int
        For putmask, this is mask.sum()
    other : Any
    """
    if values.dtype == object:
        dtype, _ = infer_dtype_from(other)

        if lib.is_np_dtype(dtype, "mM"):
            # https://github.com/numpy/numpy/issues/12550
            #  timedelta64 will incorrectly cast to int
            if not is_list_like(other):
                other = [other] * num_set
            else:
                other = list(other)

    return other

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\filesystem.py ===
import fnmatch
import os
import os.path
import random
import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Generator, List, Union, cast

from pip._internal.utils.compat import get_path_uid
from pip._internal.utils.misc import format_size
from pip._internal.utils.retry import retry


def check_path_owner(path: str) -> bool:
    # If we don't have a way to check the effective uid of this process, then
    # we'll just assume that we own the directory.
    if sys.platform == "win32" or not hasattr(os, "geteuid"):
        return True

    assert os.path.isabs(path)

    previous = None
    while path != previous:
        if os.path.lexists(path):
            # Check if path is writable by current user.
            if os.geteuid() == 0:
                # Special handling for root user in order to handle properly
                # cases where users use sudo without -H flag.
                try:
                    path_uid = get_path_uid(path)
                except OSError:
                    return False
                return path_uid == 0
            else:
                return os.access(path, os.W_OK)
        else:
            previous, path = path, os.path.dirname(path)
    return False  # assume we don't own the path


@contextmanager
def adjacent_tmp_file(path: str, **kwargs: Any) -> Generator[BinaryIO, None, None]:
    """Return a file-like object pointing to a tmp file next to path.

    The file is created securely and is ensured to be written to disk
    after the context reaches its end.

    kwargs will be passed to tempfile.NamedTemporaryFile to control
    the way the temporary file will be opened.
    """
    with NamedTemporaryFile(
        delete=False,
        dir=os.path.dirname(path),
        prefix=os.path.basename(path),
        suffix=".tmp",
        **kwargs,
    ) as f:
        result = cast(BinaryIO, f)
        try:
            yield result
        finally:
            result.flush()
            os.fsync(result.fileno())


replace = retry(stop_after_delay=1, wait=0.25)(os.replace)


# test_writable_dir and _test_writable_dir_win are copied from Flit,
# with the author's agreement to also place them under pip's license.
def test_writable_dir(path: str) -> bool:
    """Check if a directory is writable.

    Uses os.access() on POSIX, tries creating files on Windows.
    """
    # If the directory doesn't exist, find the closest parent that does.
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break  # Should never get here, but infinite loops are bad
        path = parent

    if os.name == "posix":
        return os.access(path, os.W_OK)

    return _test_writable_dir_win(path)


def _test_writable_dir_win(path: str) -> bool:
    # os.access doesn't work on Windows: http://bugs.python.org/issue2528
    # and we can't use tempfile: http://bugs.python.org/issue22107
    basename = "accesstest_deleteme_fishfingers_custard_"
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    for _ in range(10):
        name = basename + "".join(random.choice(alphabet) for _ in range(6))
        file = os.path.join(path, name)
        try:
            fd = os.open(file, os.O_RDWR | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            pass
        except PermissionError:
            # This could be because there's a directory with the same name.
            # But it's highly unlikely there's a directory called that,
            # so we'll assume it's because the parent dir is not writable.
            # This could as well be because the parent dir is not readable,
            # due to non-privileged user access.
            return False
        else:
            os.close(fd)
            os.unlink(file)
            return True

    # This should never be reached
    raise OSError("Unexpected condition testing for writable directory")


def find_files(path: str, pattern: str) -> List[str]:
    """Returns a list of absolute paths of files beneath path, recursively,
    with filenames which match the UNIX-style shell glob pattern."""
    result: List[str] = []
    for root, _, files in os.walk(path):
        matches = fnmatch.filter(files, pattern)
        result.extend(os.path.join(root, f) for f in matches)
    return result


def file_size(path: str) -> Union[int, float]:
    # If it's a symlink, return 0.
    if os.path.islink(path):
        return 0
    return os.path.getsize(path)


def format_file_size(path: str) -> str:
    return format_size(file_size(path))


def directory_size(path: str) -> Union[int, float]:
    size = 0.0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            size += file_size(file_path)
    return size


def format_directory_size(path: str) -> str:
    return format_size(directory_size(path))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\repr.py ===
import inspect
from functools import partial
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")


Result = Iterable[Union[Any, Tuple[Any], Tuple[str, Any], Tuple[str, Any, Any]]]
RichReprResult = Result


class ReprError(Exception):
    """An error occurred when attempting to build a repr."""


@overload
def auto(cls: Optional[Type[T]]) -> Type[T]:
    ...


@overload
def auto(*, angular: bool = False) -> Callable[[Type[T]], Type[T]]:
    ...


def auto(
    cls: Optional[Type[T]] = None, *, angular: Optional[bool] = None
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Class decorator to create __repr__ from __rich_repr__"""

    def do_replace(cls: Type[T], angular: Optional[bool] = None) -> Type[T]:
        def auto_repr(self: T) -> str:
            """Create repr string from __rich_repr__"""
            repr_str: List[str] = []
            append = repr_str.append

            angular: bool = getattr(self.__rich_repr__, "angular", False)  # type: ignore[attr-defined]
            for arg in self.__rich_repr__():  # type: ignore[attr-defined]
                if isinstance(arg, tuple):
                    if len(arg) == 1:
                        append(repr(arg[0]))
                    else:
                        key, value, *default = arg
                        if key is None:
                            append(repr(value))
                        else:
                            if default and default[0] == value:
                                continue
                            append(f"{key}={value!r}")
                else:
                    append(repr(arg))
            if angular:
                return f"<{self.__class__.__name__} {' '.join(repr_str)}>"
            else:
                return f"{self.__class__.__name__}({', '.join(repr_str)})"

        def auto_rich_repr(self: Type[T]) -> Result:
            """Auto generate __rich_rep__ from signature of __init__"""
            try:
                signature = inspect.signature(self.__init__)
                for name, param in signature.parameters.items():
                    if param.kind == param.POSITIONAL_ONLY:
                        yield getattr(self, name)
                    elif param.kind in (
                        param.POSITIONAL_OR_KEYWORD,
                        param.KEYWORD_ONLY,
                    ):
                        if param.default is param.empty:
                            yield getattr(self, param.name)
                        else:
                            yield param.name, getattr(self, param.name), param.default
            except Exception as error:
                raise ReprError(
                    f"Failed to auto generate __rich_repr__; {error}"
                ) from None

        if not hasattr(cls, "__rich_repr__"):
            auto_rich_repr.__doc__ = "Build a rich repr"
            cls.__rich_repr__ = auto_rich_repr  # type: ignore[attr-defined]

        auto_repr.__doc__ = "Return repr(self)"
        cls.__repr__ = auto_repr  # type: ignore[assignment]
        if angular is not None:
            cls.__rich_repr__.angular = angular  # type: ignore[attr-defined]
        return cls

    if cls is None:
        return partial(do_replace, angular=angular)
    else:
        return do_replace(cls, angular=angular)


@overload
def rich_repr(cls: Optional[Type[T]]) -> Type[T]:
    ...


@overload
def rich_repr(*, angular: bool = False) -> Callable[[Type[T]], Type[T]]:
    ...


def rich_repr(
    cls: Optional[Type[T]] = None, *, angular: bool = False
) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    if cls is None:
        return auto(angular=angular)
    else:
        return auto(cls)


if __name__ == "__main__":

    @auto
    class Foo:
        def __rich_repr__(self) -> Result:
            yield "foo"
            yield "bar", {"shopping": ["eggs", "ham", "pineapple"]}
            yield "buy", "hand sanitizer"

    foo = Foo()
    from pip._vendor.rich.console import Console

    console = Console()

    console.rule("Standard repr")
    console.print(foo)

    console.print(foo, width=60)
    console.print(foo, width=30)

    console.rule("Angular repr")
    Foo.__rich_repr__.angular = True  # type: ignore[attr-defined]

    console.print(foo)

    console.print(foo, width=60)
    console.print(foo, width=30)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\connection.py ===
from __future__ import absolute_import

import socket

from ..contrib import _appengine_environ
from ..exceptions import LocationParseError
from ..packages import six
from .wait import NoWayToWaitForSocketError, wait_for_read


def is_connection_dropped(conn):  # Platform-specific
    """
    Returns True if the connection is dropped and should be closed.

    :param conn:
        :class:`http.client.HTTPConnection` object.

    Note: For platforms like AppEngine, this will always return ``False`` to
    let the platform handle connection recycling transparently for us.
    """
    sock = getattr(conn, "sock", False)
    if sock is False:  # Platform-specific: AppEngine
        return False
    if sock is None:  # Connection already closed (such as by httplib).
        return True
    try:
        # Returns True if readable, which here means it's been dropped
        return wait_for_read(sock, timeout=0.0)
    except NoWayToWaitForSocketError:  # Platform-specific: AppEngine
        return False


# This function is copied from socket.py in the Python 2.7 standard
# library test suite. Added to its signature is only `socket_options`.
# One additional modification is that we avoid binding to IPv6 servers
# discovered in DNS if the system doesn't have IPv6 functionality.
def create_connection(
    address,
    timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address=None,
    socket_options=None,
):
    """Connect to *address* and return the socket object.

    Convenience function.  Connect to *address* (a 2-tuple ``(host,
    port)``) and return the socket object.  Passing the optional
    *timeout* parameter will set the timeout on the socket instance
    before attempting to connect.  If no *timeout* is supplied, the
    global default timeout setting returned by :func:`socket.getdefaulttimeout`
    is used.  If *source_address* is set it must be a tuple of (host, port)
    for the socket to bind as a source address before making the connection.
    An host of '' or port 0 tells the OS to use the default.
    """

    host, port = address
    if host.startswith("["):
        host = host.strip("[]")
    err = None

    # Using the value from allowed_gai_family() in the context of getaddrinfo lets
    # us select whether to work with IPv4 DNS records, IPv6 records, or both.
    # The original create_connection function always returns all records.
    family = allowed_gai_family()

    try:
        host.encode("idna")
    except UnicodeError:
        return six.raise_from(
            LocationParseError(u"'%s', label empty or too long" % host), None
        )

    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
        af, socktype, proto, canonname, sa = res
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)

            # If provided, set socket level options before connecting.
            _set_socket_options(sock, socket_options)

            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sa)
            return sock

        except socket.error as e:
            err = e
            if sock is not None:
                sock.close()
                sock = None

    if err is not None:
        raise err

    raise socket.error("getaddrinfo returns an empty list")


def _set_socket_options(sock, options):
    if options is None:
        return

    for opt in options:
        sock.setsockopt(*opt)


def allowed_gai_family():
    """This function is designed to work in the context of
    getaddrinfo, where family=socket.AF_UNSPEC is the default and
    will perform a DNS search for both IPv6 and IPv4 records."""

    family = socket.AF_INET
    if HAS_IPV6:
        family = socket.AF_UNSPEC
    return family


def _has_ipv6(host):
    """Returns True if the system can bind an IPv6 address."""
    sock = None
    has_ipv6 = False

    # App Engine doesn't support IPV6 sockets and actually has a quota on the
    # number of sockets that can be used, so just early out here instead of
    # creating a socket needlessly.
    # See https://github.com/urllib3/urllib3/issues/1446
    if _appengine_environ.is_appengine_sandbox():
        return False

    if socket.has_ipv6:
        # has_ipv6 returns true if cPython was compiled with IPv6 support.
        # It does not tell us if the system has IPv6 support enabled. To
        # determine that we must bind to an IPv6 address.
        # https://github.com/urllib3/urllib3/pull/611
        # https://bugs.python.org/issue658327
        try:
            sock = socket.socket(socket.AF_INET6)
            sock.bind((host, 0))
            has_ipv6 = True
        except Exception:
            pass

    if sock:
        sock.close()
    return has_ipv6


HAS_IPV6 = _has_ipv6("::1")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\chromium\options.py ===
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

import base64
import os
from typing import BinaryIO, Dict, List, Optional, Union

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions


class ChromiumOptions(ArgOptions):
    KEY = "goog:chromeOptions"

    def __init__(self) -> None:
        super().__init__()
        self._binary_location: str = ""
        self._extension_files: List[str] = []
        self._extensions: List[str] = []
        self._experimental_options: Dict[str, Union[str, int, dict, List[str]]] = {}
        self._debugger_address: Optional[str] = None

    @property
    def binary_location(self) -> str:
        """:Returns: The location of the binary, otherwise an empty string."""
        return self._binary_location

    @binary_location.setter
    def binary_location(self, value: str) -> None:
        """Allows you to set where the chromium binary lives.

        :Args:
         - value: path to the Chromium binary
        """
        if not isinstance(value, str):
            raise TypeError(self.BINARY_LOCATION_ERROR)
        self._binary_location = value

    @property
    def debugger_address(self) -> Optional[str]:
        """:Returns: The address of the remote devtools instance."""
        return self._debugger_address

    @debugger_address.setter
    def debugger_address(self, value: str) -> None:
        """Allows you to set the address of the remote devtools instance that
        the ChromeDriver instance will try to connect to during an active wait.

        :Args:
         - value: address of remote devtools instance if any (hostname[:port])
        """
        if not isinstance(value, str):
            raise TypeError("Debugger Address must be a string")
        self._debugger_address = value

    @property
    def extensions(self) -> List[str]:
        """:Returns: A list of encoded extensions that will be loaded."""

        def _decode(file_data: BinaryIO) -> str:
            # Should not use base64.encodestring() which inserts newlines every
            # 76 characters (per RFC 1521).  Chromedriver has to remove those
            # unnecessary newlines before decoding, causing performance hit.
            return base64.b64encode(file_data.read()).decode("utf-8")

        encoded_extensions = []
        for extension in self._extension_files:
            with open(extension, "rb") as f:
                encoded_extensions.append(_decode(f))

        return encoded_extensions + self._extensions

    def add_extension(self, extension: str) -> None:
        """Adds the path to the extension to a list that will be used to
        extract it to the ChromeDriver.

        :Args:
         - extension: path to the \\*.crx file
        """
        if extension:
            extension_to_add = os.path.abspath(os.path.expanduser(extension))
            if os.path.exists(extension_to_add):
                self._extension_files.append(extension_to_add)
            else:
                raise OSError("Path to the extension doesn't exist")
        else:
            raise ValueError("argument can not be null")

    def add_encoded_extension(self, extension: str) -> None:
        """Adds Base64 encoded string with extension data to a list that will
        be used to extract it to the ChromeDriver.

        :Args:
         - extension: Base64 encoded string with extension data
        """
        if extension:
            self._extensions.append(extension)
        else:
            raise ValueError("argument can not be null")

    @property
    def experimental_options(self) -> dict:
        """:Returns: A dictionary of experimental options for chromium."""
        return self._experimental_options

    def add_experimental_option(self, name: str, value: Union[str, int, dict, List[str]]) -> None:
        """Adds an experimental option which is passed to chromium.

        :Args:
          name: The experimental option name.
          value: The option value.
        """
        self._experimental_options[name] = value

    def to_capabilities(self) -> dict:
        """Creates a capabilities with all the options that have been set
        :Returns: A dictionary with everything."""
        caps = self._caps
        chrome_options = self.experimental_options.copy()
        if self.mobile_options:
            chrome_options.update(self.mobile_options)
        chrome_options["extensions"] = self.extensions
        if self.binary_location:
            chrome_options["binary"] = self.binary_location
        chrome_options["args"] = self._arguments
        if self.debugger_address:
            chrome_options["debuggerAddress"] = self.debugger_address

        caps[self.KEY] = chrome_options

        return caps

    @property
    def default_capabilities(self) -> dict:
        return DesiredCapabilities.CHROME.copy()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\log.py ===
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

import json
import pkgutil
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any, AsyncGenerator, Dict, Optional

from selenium.webdriver.common.by import By

cdp = None


def import_cdp():
    global cdp
    if not cdp:
        cdp = import_module("selenium.webdriver.common.bidi.cdp")


class Log:
    """This class allows access to logging APIs that use the new WebDriver Bidi
    protocol.

    This class is not to be used directly and should be used from the
    webdriver base classes.
    """

    def __init__(self, driver, bidi_session) -> None:
        self.driver = driver
        self.session = bidi_session.session
        self.cdp = bidi_session.cdp
        self.devtools = bidi_session.devtools
        _pkg = ".".join(__name__.split(".")[:-1])
        # Ensure _mutation_listener_js is not None before decoding
        _mutation_listener_js_bytes: Optional[bytes] = pkgutil.get_data(_pkg, "mutation-listener.js")
        if _mutation_listener_js_bytes is None:
            raise ValueError("Failed to load mutation-listener.js")
        self._mutation_listener_js = _mutation_listener_js_bytes.decode("utf8").strip()

    @asynccontextmanager
    async def mutation_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Listen for mutation events and emit them as they are found.

        :Usage:
             ::

               async with driver.log.mutation_events() as event:
                    pages.load("dynamic.html")
                    driver.find_element(By.ID, "reveal").click()
                    WebDriverWait(driver, 5)\
                        .until(EC.visibility_of(driver.find_element(By.ID, "revealed")))

                assert event["attribute_name"] == "style"
                assert event["current_value"] == ""
                assert event["old_value"] == "display:none;"
        """

        page = self.cdp.get_session_context("page.enable")
        await page.execute(self.devtools.page.enable())
        runtime = self.cdp.get_session_context("runtime.enable")
        await runtime.execute(self.devtools.runtime.enable())
        await runtime.execute(self.devtools.runtime.add_binding("__webdriver_attribute"))
        self.driver.pin_script(self._mutation_listener_js)
        script_key = await page.execute(
            self.devtools.page.add_script_to_evaluate_on_new_document(self._mutation_listener_js)
        )
        self.driver.pin_script(self._mutation_listener_js, script_key)
        self.driver.execute_script(f"return {self._mutation_listener_js}")

        event: Dict[str, Any] = {}
        async with runtime.wait_for(self.devtools.runtime.BindingCalled) as evnt:
            yield event

        payload = json.loads(evnt.value.payload)
        elements: list = self.driver.find_elements(By.CSS_SELECTOR, f'*[data-__webdriver_id="{payload["target"]}"]')
        if not elements:
            elements.append(None)
        event["element"] = elements[0]
        event["attribute_name"] = payload["name"]
        event["current_value"] = payload["value"]
        event["old_value"] = payload["oldValue"]

    @asynccontextmanager
    async def add_js_error_listener(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Listen for JS errors and when the contextmanager exits check if
        there were JS Errors.

        :Usage:
             ::

                async with driver.log.add_js_error_listener() as error:
                    driver.find_element(By.ID, "throwing-mouseover").click()
                assert bool(error)
                assert error.exception_details.stack_trace.call_frames[0].function_name == "onmouseover"
        """

        session = self.cdp.get_session_context("page.enable")
        await session.execute(self.devtools.page.enable())
        session = self.cdp.get_session_context("runtime.enable")
        await session.execute(self.devtools.runtime.enable())
        js_exception = self.devtools.runtime.ExceptionThrown(None, None)
        async with session.wait_for(self.devtools.runtime.ExceptionThrown) as exception:
            yield js_exception
        js_exception.timestamp = exception.value.timestamp
        js_exception.exception_details = exception.value.exception_details

    @asynccontextmanager
    async def add_listener(self, event_type) -> AsyncGenerator[Dict[str, Any], None]:
        """Listen for certain events that are passed in.

        :Args:
         - event_type: The type of event that we want to look at.

        :Usage:
             ::

                async with driver.log.add_listener(Console.log) as messages:
                    driver.execute_script("console.log('I like cheese')")
                assert messages["message"] == "I love cheese"
        """

        from selenium.webdriver.common.bidi.console import Console

        session = self.cdp.get_session_context("page.enable")
        await session.execute(self.devtools.page.enable())
        session = self.cdp.get_session_context("runtime.enable")
        await session.execute(self.devtools.runtime.enable())
        console: Dict[str, Any] = {"message": None, "level": None}
        async with session.wait_for(self.devtools.runtime.ConsoleAPICalled) as messages:
            yield console

        if event_type == Console.ALL or event_type.value == messages.value.type_:
            console["message"] = messages.value.args[0].value
            console["level"] = messages.value.args[0].type_

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\switch_to.py ===
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

from typing import Optional, Union

from selenium.common.exceptions import NoSuchElementException, NoSuchFrameException, NoSuchWindowException
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

from .command import Command


class SwitchTo:
    def __init__(self, driver) -> None:
        import weakref

        self._driver = weakref.proxy(driver)

    @property
    def active_element(self) -> WebElement:
        """Returns the element with focus, or BODY if nothing has focus.

        :Usage:
            ::

                element = driver.switch_to.active_element
        """
        return self._driver.execute(Command.W3C_GET_ACTIVE_ELEMENT)["value"]

    @property
    def alert(self) -> Alert:
        """Switches focus to an alert on the page.

        :Usage:
            ::

                alert = driver.switch_to.alert
        """
        alert = Alert(self._driver)
        _ = alert.text
        return alert

    def default_content(self) -> None:
        """Switch focus to the default frame.

        :Usage:
            ::

                driver.switch_to.default_content()
        """
        self._driver.execute(Command.SWITCH_TO_FRAME, {"id": None})

    def frame(self, frame_reference: Union[str, int, WebElement]) -> None:
        """Switches focus to the specified frame, by index, name, or
        webelement.

        :Args:
         - frame_reference: The name of the window to switch to, an integer representing the index,
                            or a webelement that is an (i)frame to switch to.

        :Usage:
            ::

                driver.switch_to.frame("frame_name")
                driver.switch_to.frame(1)
                driver.switch_to.frame(driver.find_elements(By.TAG_NAME, "iframe")[0])
        """
        if isinstance(frame_reference, str):
            try:
                frame_reference = self._driver.find_element(By.ID, frame_reference)
            except NoSuchElementException:
                try:
                    frame_reference = self._driver.find_element(By.NAME, frame_reference)
                except NoSuchElementException as exc:
                    raise NoSuchFrameException(frame_reference) from exc

        self._driver.execute(Command.SWITCH_TO_FRAME, {"id": frame_reference})

    def new_window(self, type_hint: Optional[str] = None) -> None:
        """Switches to a new top-level browsing context.

        The type hint can be one of "tab" or "window". If not specified the
        browser will automatically select it.

        :Usage:
            ::

                driver.switch_to.new_window("tab")
        """
        value = self._driver.execute(Command.NEW_WINDOW, {"type": type_hint})["value"]
        self._w3c_window(value["handle"])

    def parent_frame(self) -> None:
        """Switches focus to the parent context. If the current context is the
        top level browsing context, the context remains unchanged.

        :Usage:
            ::

                driver.switch_to.parent_frame()
        """
        self._driver.execute(Command.SWITCH_TO_PARENT_FRAME)

    def window(self, window_name: str) -> None:
        """Switches focus to the specified window.

        :Args:
         - window_name: The name or window handle of the window to switch to.

        :Usage:
            ::

                driver.switch_to.window("main")
        """
        self._w3c_window(window_name)

    def _w3c_window(self, window_name: str) -> None:
        def send_handle(h):
            self._driver.execute(Command.SWITCH_TO_WINDOW, {"handle": h})

        try:
            # Try using it as a handle first.
            send_handle(window_name)
        except NoSuchWindowException:
            # Check every window to try to find the given window name.
            original_handle = self._driver.current_window_handle
            handles = self._driver.window_handles
            for handle in handles:
                send_handle(handle)
                current_name = self._driver.execute_script("return window.name")
                if window_name == current_name:
                    return
            send_handle(original_handle)
            raise

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\heuristicgcd.py ===
"""Heuristic polynomial GCD algorithm (HEUGCD). """

from .polyerrors import HeuristicGCDFailed

HEU_GCD_MAX = 6

def heugcd(f, g):
    """
    Heuristic polynomial GCD in ``Z[X]``.

    Given univariate polynomials ``f`` and ``g`` in ``Z[X]``, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    ``f`` and ``g`` at certain points and computing (fast) integer GCD
    of those evaluations. The polynomial GCD is recovered from the integer
    image by interpolation. The evaluation process reduces f and g variable
    by variable into a large integer. The final step is to verify if the
    interpolated polynomial is the correct GCD. This gives cofactors of
    the input polynomials as a side effect.

    Examples
    ========

    >>> from sympy.polys.heuristicgcd import heugcd
    >>> from sympy.polys import ring, ZZ

    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> h, cff, cfg = heugcd(f, g)
    >>> h, cff, cfg
    (x + y, x + y, x)

    >>> cff*h == f
    True
    >>> cfg*h == g
    True

    References
    ==========

    .. [1] [Liao95]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    ring = f.ring
    x0 = ring.gens[0]
    domain = ring.domain

    gcd, f, g = f.extract_ground(g)

    f_norm = f.max_norm()
    g_norm = g.max_norm()

    B = domain(2*min(f_norm, g_norm) + 29)

    x = max(min(B, 99*domain.sqrt(B)),
            2*min(f_norm // abs(f.LC),
                  g_norm // abs(g.LC)) + 4)

    for i in range(0, HEU_GCD_MAX):
        ff = f.evaluate(x0, x)
        gg = g.evaluate(x0, x)

        if ff and gg:
            if ring.ngens == 1:
                h, cff, cfg = domain.cofactors(ff, gg)
            else:
                h, cff, cfg = heugcd(ff, gg)

            h = _gcd_interpolate(h, x, ring)
            h = h.primitive()[1]

            cff_, r = f.div(h)

            if not r:
                cfg_, r = g.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff_, cfg_

            cff = _gcd_interpolate(cff, x, ring)

            h, r = f.div(cff)

            if not r:
                cfg_, r = g.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff, cfg_

            cfg = _gcd_interpolate(cfg, x, ring)

            h, r = g.div(cfg)

            if not r:
                cff_, r = f.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff_, cfg

        x = 73794*x * domain.sqrt(domain.sqrt(x)) // 27011

    raise HeuristicGCDFailed('no luck')

def _gcd_interpolate(h, x, ring):
    """Interpolate polynomial GCD from integer GCD. """
    f, i = ring.zero, 0

    # TODO: don't expose poly repr implementation details
    if ring.ngens == 1:
        while h:
            g = h % x
            if g > x // 2: g -= x
            h = (h - g) // x

            # f += X**i*g
            if g:
                f[(i,)] = g
            i += 1
    else:
        while h:
            g = h.trunc_ground(x)
            h = (h - g).quo_ground(x)

            # f += X**i*g
            if g:
                for monom, coeff in g.iterterms():
                    f[(i,) + monom] = coeff
            i += 1

    if f.LC < 0:
        return -f
    else:
        return  f

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\__init__.py ===
"""Private counterpart of ``numpy.typing``."""

from ._array_like import ArrayLike as ArrayLike
from ._array_like import NDArray as NDArray
from ._array_like import _ArrayLike as _ArrayLike
from ._array_like import _ArrayLikeAnyString_co as _ArrayLikeAnyString_co
from ._array_like import _ArrayLikeBool_co as _ArrayLikeBool_co
from ._array_like import _ArrayLikeBytes_co as _ArrayLikeBytes_co
from ._array_like import _ArrayLikeComplex128_co as _ArrayLikeComplex128_co
from ._array_like import _ArrayLikeComplex_co as _ArrayLikeComplex_co
from ._array_like import _ArrayLikeDT64_co as _ArrayLikeDT64_co
from ._array_like import _ArrayLikeFloat64_co as _ArrayLikeFloat64_co
from ._array_like import _ArrayLikeFloat_co as _ArrayLikeFloat_co
from ._array_like import _ArrayLikeInt as _ArrayLikeInt
from ._array_like import _ArrayLikeInt_co as _ArrayLikeInt_co
from ._array_like import _ArrayLikeNumber_co as _ArrayLikeNumber_co
from ._array_like import _ArrayLikeObject_co as _ArrayLikeObject_co
from ._array_like import _ArrayLikeStr_co as _ArrayLikeStr_co
from ._array_like import _ArrayLikeString_co as _ArrayLikeString_co
from ._array_like import _ArrayLikeTD64_co as _ArrayLikeTD64_co
from ._array_like import _ArrayLikeUInt_co as _ArrayLikeUInt_co
from ._array_like import _ArrayLikeVoid_co as _ArrayLikeVoid_co
from ._array_like import _FiniteNestedSequence as _FiniteNestedSequence
from ._array_like import _SupportsArray as _SupportsArray
from ._array_like import _SupportsArrayFunc as _SupportsArrayFunc

#
from ._char_codes import _BoolCodes as _BoolCodes
from ._char_codes import _ByteCodes as _ByteCodes
from ._char_codes import _BytesCodes as _BytesCodes
from ._char_codes import _CDoubleCodes as _CDoubleCodes
from ._char_codes import _CharacterCodes as _CharacterCodes
from ._char_codes import _CLongDoubleCodes as _CLongDoubleCodes
from ._char_codes import _Complex64Codes as _Complex64Codes
from ._char_codes import _Complex128Codes as _Complex128Codes
from ._char_codes import _ComplexFloatingCodes as _ComplexFloatingCodes
from ._char_codes import _CSingleCodes as _CSingleCodes
from ._char_codes import _DoubleCodes as _DoubleCodes
from ._char_codes import _DT64Codes as _DT64Codes
from ._char_codes import _FlexibleCodes as _FlexibleCodes
from ._char_codes import _Float16Codes as _Float16Codes
from ._char_codes import _Float32Codes as _Float32Codes
from ._char_codes import _Float64Codes as _Float64Codes
from ._char_codes import _FloatingCodes as _FloatingCodes
from ._char_codes import _GenericCodes as _GenericCodes
from ._char_codes import _HalfCodes as _HalfCodes
from ._char_codes import _InexactCodes as _InexactCodes
from ._char_codes import _Int8Codes as _Int8Codes
from ._char_codes import _Int16Codes as _Int16Codes
from ._char_codes import _Int32Codes as _Int32Codes
from ._char_codes import _Int64Codes as _Int64Codes
from ._char_codes import _IntCCodes as _IntCCodes
from ._char_codes import _IntCodes as _IntCodes
from ._char_codes import _IntegerCodes as _IntegerCodes
from ._char_codes import _IntPCodes as _IntPCodes
from ._char_codes import _LongCodes as _LongCodes
from ._char_codes import _LongDoubleCodes as _LongDoubleCodes
from ._char_codes import _LongLongCodes as _LongLongCodes
from ._char_codes import _NumberCodes as _NumberCodes
from ._char_codes import _ObjectCodes as _ObjectCodes
from ._char_codes import _ShortCodes as _ShortCodes
from ._char_codes import _SignedIntegerCodes as _SignedIntegerCodes
from ._char_codes import _SingleCodes as _SingleCodes
from ._char_codes import _StrCodes as _StrCodes
from ._char_codes import _StringCodes as _StringCodes
from ._char_codes import _TD64Codes as _TD64Codes
from ._char_codes import _UByteCodes as _UByteCodes
from ._char_codes import _UInt8Codes as _UInt8Codes
from ._char_codes import _UInt16Codes as _UInt16Codes
from ._char_codes import _UInt32Codes as _UInt32Codes
from ._char_codes import _UInt64Codes as _UInt64Codes
from ._char_codes import _UIntCCodes as _UIntCCodes
from ._char_codes import _UIntCodes as _UIntCodes
from ._char_codes import _UIntPCodes as _UIntPCodes
from ._char_codes import _ULongCodes as _ULongCodes
from ._char_codes import _ULongLongCodes as _ULongLongCodes
from ._char_codes import _UnsignedIntegerCodes as _UnsignedIntegerCodes
from ._char_codes import _UShortCodes as _UShortCodes
from ._char_codes import _VoidCodes as _VoidCodes

#
from ._dtype_like import DTypeLike as DTypeLike
from ._dtype_like import _DTypeLike as _DTypeLike
from ._dtype_like import _DTypeLikeBool as _DTypeLikeBool
from ._dtype_like import _DTypeLikeBytes as _DTypeLikeBytes
from ._dtype_like import _DTypeLikeComplex as _DTypeLikeComplex
from ._dtype_like import _DTypeLikeComplex_co as _DTypeLikeComplex_co
from ._dtype_like import _DTypeLikeDT64 as _DTypeLikeDT64
from ._dtype_like import _DTypeLikeFloat as _DTypeLikeFloat
from ._dtype_like import _DTypeLikeInt as _DTypeLikeInt
from ._dtype_like import _DTypeLikeObject as _DTypeLikeObject
from ._dtype_like import _DTypeLikeStr as _DTypeLikeStr
from ._dtype_like import _DTypeLikeTD64 as _DTypeLikeTD64
from ._dtype_like import _DTypeLikeUInt as _DTypeLikeUInt
from ._dtype_like import _DTypeLikeVoid as _DTypeLikeVoid
from ._dtype_like import _SupportsDType as _SupportsDType
from ._dtype_like import _VoidDTypeLike as _VoidDTypeLike

#
from ._nbit import _NBitByte as _NBitByte
from ._nbit import _NBitDouble as _NBitDouble
from ._nbit import _NBitHalf as _NBitHalf
from ._nbit import _NBitInt as _NBitInt
from ._nbit import _NBitIntC as _NBitIntC
from ._nbit import _NBitIntP as _NBitIntP
from ._nbit import _NBitLong as _NBitLong
from ._nbit import _NBitLongDouble as _NBitLongDouble
from ._nbit import _NBitLongLong as _NBitLongLong
from ._nbit import _NBitShort as _NBitShort
from ._nbit import _NBitSingle as _NBitSingle

#
from ._nbit_base import (
    NBitBase as NBitBase,  # type: ignore[deprecated]  # pyright: ignore[reportDeprecated]
)
from ._nbit_base import _8Bit as _8Bit
from ._nbit_base import _16Bit as _16Bit
from ._nbit_base import _32Bit as _32Bit
from ._nbit_base import _64Bit as _64Bit
from ._nbit_base import _96Bit as _96Bit
from ._nbit_base import _128Bit as _128Bit

#
from ._nested_sequence import _NestedSequence as _NestedSequence

#
from ._scalars import _BoolLike_co as _BoolLike_co
from ._scalars import _CharLike_co as _CharLike_co
from ._scalars import _ComplexLike_co as _ComplexLike_co
from ._scalars import _FloatLike_co as _FloatLike_co
from ._scalars import _IntLike_co as _IntLike_co
from ._scalars import _NumberLike_co as _NumberLike_co
from ._scalars import _ScalarLike_co as _ScalarLike_co
from ._scalars import _TD64Like_co as _TD64Like_co
from ._scalars import _UIntLike_co as _UIntLike_co
from ._scalars import _VoidLike_co as _VoidLike_co

#
from ._shape import _AnyShape as _AnyShape
from ._shape import _Shape as _Shape
from ._shape import _ShapeLike as _ShapeLike

#
from ._ufunc import _GUFunc_Nin2_Nout1 as _GUFunc_Nin2_Nout1
from ._ufunc import _UFunc_Nin1_Nout1 as _UFunc_Nin1_Nout1
from ._ufunc import _UFunc_Nin1_Nout2 as _UFunc_Nin1_Nout2
from ._ufunc import _UFunc_Nin2_Nout1 as _UFunc_Nin2_Nout1
from ._ufunc import _UFunc_Nin2_Nout2 as _UFunc_Nin2_Nout2

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\determinant.py ===
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.matrixbase import MatrixBase


class Determinant(Expr):
    """Matrix Determinant

    Represents the determinant of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Determinant, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Determinant(A)
    Determinant(A)
    >>> Determinant(eye(3)).doit()
    1
    """
    is_commutative = True

    def __new__(cls, mat):
        mat = sympify(mat)
        if not mat.is_Matrix:
            raise TypeError("Input to Determinant, %s, not a matrix" % str(mat))

        if mat.is_square is False:
            raise NonSquareMatrixError("Det of a non-square matrix")

        return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    @property
    def kind(self):
        return self.arg.kind.element_kind

    def doit(self, **hints):
        arg = self.arg
        if hints.get('deep', True):
            arg = arg.doit(**hints)

        result = arg._eval_determinant()
        if result is not None:
            return result

        return self


def det(matexpr):
    """ Matrix Determinant

    Examples
    ========

    >>> from sympy import MatrixSymbol, det, eye
    >>> A = MatrixSymbol('A', 3, 3)
    >>> det(A)
    Determinant(A)
    >>> det(eye(3))
    1
    """

    return Determinant(matexpr).doit()

class Permanent(Expr):
    """Matrix Permanent

    Represents the permanent of a matrix expression.

    Examples
    ========

    >>> from sympy import MatrixSymbol, Permanent, ones
    >>> A = MatrixSymbol('A', 3, 3)
    >>> Permanent(A)
    Permanent(A)
    >>> Permanent(ones(3, 3)).doit()
    6
    """

    def __new__(cls, mat):
        mat = sympify(mat)
        if not mat.is_Matrix:
            raise TypeError("Input to Permanent, %s, not a matrix" % str(mat))

        return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    def doit(self, expand=False, **hints):
        if isinstance(self.arg, MatrixBase):
            return self.arg.per()
        else:
            return self

def per(matexpr):
    """ Matrix Permanent

    Examples
    ========

    >>> from sympy import MatrixSymbol, Matrix, per, ones
    >>> A = MatrixSymbol('A', 3, 3)
    >>> per(A)
    Permanent(A)
    >>> per(ones(5, 5))
    120
    >>> M = Matrix([1, 2, 5])
    >>> per(M)
    8
    """

    return Permanent(matexpr).doit()

from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict


def refine_Determinant(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine, det
    >>> X = MatrixSymbol('X', 2, 2)
    >>> det(X)
    Determinant(X)
    >>> with assuming(Q.orthogonal(X)):
    ...     print(refine(det(X)))
    1
    """
    if ask(Q.orthogonal(expr.arg), assumptions):
        return S.One
    elif ask(Q.singular(expr.arg), assumptions):
        return S.Zero
    elif ask(Q.unit_triangular(expr.arg), assumptions):
        return S.One

    return expr


handlers_dict['Determinant'] = refine_Determinant

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\point.py ===
from sympy.core.basic import Basic
from sympy.core.symbol import Str
from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import _path
from sympy.core.cache import cacheit


class Point(Basic):
    """
    Represents a point in 3-D space.
    """

    def __new__(cls, name, position=Vector.zero, parent_point=None):
        name = str(name)
        # Check the args first
        if not isinstance(position, Vector):
            raise TypeError(
                "position should be an instance of Vector, not %s" % type(
                    position))
        if (not isinstance(parent_point, Point) and
                parent_point is not None):
            raise TypeError(
                "parent_point should be an instance of Point, not %s" % type(
                    parent_point))
        # Super class construction
        if parent_point is None:
            obj = super().__new__(cls, Str(name), position)
        else:
            obj = super().__new__(cls, Str(name), position, parent_point)
        # Decide the object parameters
        obj._name = name
        obj._pos = position
        if parent_point is None:
            obj._parent = None
            obj._root = obj
        else:
            obj._parent = parent_point
            obj._root = parent_point._root
        # Return object
        return obj

    @cacheit
    def position_wrt(self, other):
        """
        Returns the position vector of this Point with respect to
        another Point/CoordSys3D.

        Parameters
        ==========

        other : Point/CoordSys3D
            If other is a Point, the position of this Point wrt it is
            returned. If its an instance of CoordSyRect, the position
            wrt its origin is returned.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> N.origin.position_wrt(p1)
        (-10)*N.i

        """

        if (not isinstance(other, Point) and
                not isinstance(other, CoordSys3D)):
            raise TypeError(str(other) +
                            "is not a Point or CoordSys3D")
        if isinstance(other, CoordSys3D):
            other = other.origin
        # Handle special cases
        if other == self:
            return Vector.zero
        elif other == self._parent:
            return self._pos
        elif other._parent == self:
            return -1 * other._pos
        # Else, use point tree to calculate position
        rootindex, path = _path(self, other)
        result = Vector.zero
        for i in range(rootindex):
            result += path[i]._pos
        for i in range(rootindex + 1, len(path)):
            result -= path[i]._pos
        return result

    def locate_new(self, name, position):
        """
        Returns a new Point located at the given position wrt this
        Point.
        Thus, the position vector of the new Point wrt this one will
        be equal to the given 'position' parameter.

        Parameters
        ==========

        name : str
            Name of the new point

        position : Vector
            The position vector of the new Point wrt this one

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p1.position_wrt(N.origin)
        10*N.i

        """
        return Point(name, position, self)

    def express_coordinates(self, coordinate_system):
        """
        Returns the Cartesian/rectangular coordinates of this point
        wrt the origin of the given CoordSys3D instance.

        Parameters
        ==========

        coordinate_system : CoordSys3D
            The coordinate system to express the coordinates of this
            Point in.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> N = CoordSys3D('N')
        >>> p1 = N.origin.locate_new('p1', 10 * N.i)
        >>> p2 = p1.locate_new('p2', 5 * N.j)
        >>> p2.express_coordinates(N)
        (10, 5, 0)

        """

        # Determine the position vector
        pos_vect = self.position_wrt(coordinate_system.origin)
        # Express it in the given coordinate system
        return tuple(pos_vect.to_matrix(coordinate_system))

    def _sympystr(self, printer):
        return self._name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_highlevel_serve_listeners.py ===
from __future__ import annotations

import errno
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, NoReturn, TypeVar

import trio

# Errors that accept(2) can return, and which indicate that the system is
# overloaded
ACCEPT_CAPACITY_ERRNOS = {
    errno.EMFILE,
    errno.ENFILE,
    errno.ENOMEM,
    errno.ENOBUFS,
}

# How long to sleep when we get one of those errors
SLEEP_TIME = 0.100

# The logger we use to complain when this happens
LOGGER = logging.getLogger("trio.serve_listeners")


StreamT = TypeVar("StreamT", bound=trio.abc.AsyncResource)
ListenerT = TypeVar("ListenerT", bound=trio.abc.Listener[Any])  # type: ignore[explicit-any]
Handler = Callable[[StreamT], Awaitable[object]]


async def _run_handler(stream: StreamT, handler: Handler[StreamT]) -> None:
    try:
        await handler(stream)
    finally:
        await trio.aclose_forcefully(stream)


async def _serve_one_listener(
    listener: trio.abc.Listener[StreamT],
    handler_nursery: trio.Nursery,
    handler: Handler[StreamT],
) -> NoReturn:
    async with listener:
        while True:
            try:
                stream = await listener.accept()
            except OSError as exc:
                if exc.errno in ACCEPT_CAPACITY_ERRNOS:
                    LOGGER.error(
                        "accept returned %s (%s); retrying in %s seconds",
                        errno.errorcode[exc.errno],
                        os.strerror(exc.errno),
                        SLEEP_TIME,
                        exc_info=True,
                    )
                    await trio.sleep(SLEEP_TIME)
                else:
                    raise
            else:
                handler_nursery.start_soon(_run_handler, stream, handler)


# This cannot be typed correctly, we need generic typevar bounds / HKT to indicate the
# relationship between StreamT & ListenerT.
# https://github.com/python/typing/issues/1226
# https://github.com/python/typing/issues/548


async def serve_listeners(  # type: ignore[explicit-any]
    handler: Handler[StreamT],
    listeners: list[ListenerT],
    *,
    handler_nursery: trio.Nursery | None = None,
    task_status: trio.TaskStatus[list[ListenerT]] = trio.TASK_STATUS_IGNORED,
) -> NoReturn:
    r"""Listen for incoming connections on ``listeners``, and for each one
    start a task running ``handler(stream)``.

    .. warning::

       If ``handler`` raises an exception, then this function doesn't do
       anything special to catch it – so by default the exception will
       propagate out and crash your server. If you don't want this, then catch
       exceptions inside your ``handler``, or use a ``handler_nursery`` object
       that responds to exceptions in some other way.

    Args:

      handler: An async callable, that will be invoked like
          ``handler_nursery.start_soon(handler, stream)`` for each incoming
          connection.

      listeners: A list of :class:`~trio.abc.Listener` objects.
          :func:`serve_listeners` takes responsibility for closing them.

      handler_nursery: The nursery used to start handlers, or any object with
          a ``start_soon`` method. If ``None`` (the default), then
          :func:`serve_listeners` will create a new nursery internally and use
          that.

      task_status: This function can be used with ``nursery.start``, which
          will return ``listeners``.

    Returns:

      This function never returns unless cancelled.

    Resource handling:

      If ``handler`` neglects to close the ``stream``, then it will be closed
      using :func:`trio.aclose_forcefully`.

    Error handling:

      Most errors coming from :meth:`~trio.abc.Listener.accept` are allowed to
      propagate out (crashing the server in the process). However, some errors –
      those which indicate that the server is temporarily overloaded – are
      handled specially. These are :class:`OSError`\s with one of the following
      errnos:

      * ``EMFILE``: process is out of file descriptors
      * ``ENFILE``: system is out of file descriptors
      * ``ENOBUFS``, ``ENOMEM``: the kernel hit some sort of memory limitation
        when trying to create a socket object

      When :func:`serve_listeners` gets one of these errors, then it:

      * Logs the error to the standard library logger ``trio.serve_listeners``
        (level = ERROR, with exception information included). By default this
        causes it to be printed to stderr.
      * Waits 100 ms before calling ``accept`` again, in hopes that the
        system will recover.

    """
    async with trio.open_nursery() as nursery:
        if handler_nursery is None:
            handler_nursery = nursery
        for listener in listeners:
            nursery.start_soon(_serve_one_listener, listener, handler_nursery, handler)
        # The listeners are already queueing connections when we're called,
        # but we wait until the end to call started() just in case we get an
        # error or whatever.
        task_status.started(listeners)

    raise AssertionError(
        "_serve_one_listener should never complete",
    )  # pragma: no cover

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\sam2\sam2_utils.py ===
# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import sys
from collections.abc import Mapping

import torch
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

logger = logging.getLogger(__name__)


def _get_model_cfg(model_type) -> str:
    assert model_type in ["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"]
    if model_type == "sam2_hiera_tiny":
        model_cfg = "sam2_hiera_t.yaml"
    elif model_type == "sam2_hiera_small":
        model_cfg = "sam2_hiera_s.yaml"
    elif model_type == "sam2_hiera_base_plus":
        model_cfg = "sam2_hiera_b+.yaml"
    else:
        model_cfg = "sam2_hiera_l.yaml"
    return model_cfg


def load_sam2_model(sam2_dir, model_type, device: str | torch.device = "cpu") -> SAM2Base:
    checkpoints_dir = os.path.join(sam2_dir, "checkpoints")
    sam2_config_dir = os.path.join(sam2_dir, "sam2_configs")
    if not os.path.exists(sam2_dir):
        raise FileNotFoundError(f"{sam2_dir} does not exist. Please specify --sam2_dir correctly.")

    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"{checkpoints_dir} does not exist. Please specify --sam2_dir correctly.")

    if not os.path.exists(sam2_config_dir):
        raise FileNotFoundError(f"{sam2_config_dir} does not exist. Please specify --sam2_dir correctly.")

    checkpoint_path = os.path.join(checkpoints_dir, f"{model_type}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} does not exist. Please download checkpoints under the directory.")

    if sam2_dir not in sys.path:
        sys.path.append(sam2_dir)

    model_cfg = _get_model_cfg(model_type)
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    return sam2_model


def sam2_onnx_path(output_dir, model_type, component, multimask_output=False, suffix=""):
    if component == "image_encoder":
        return os.path.join(output_dir, f"{model_type}_image_encoder{suffix}.onnx")
    elif component == "mask_decoder":
        return os.path.join(output_dir, f"{model_type}_mask_decoder{suffix}.onnx")
    elif component == "prompt_encoder":
        return os.path.join(output_dir, f"{model_type}_prompt_encoder{suffix}.onnx")
    else:
        assert component == "image_decoder"
        return os.path.join(
            output_dir, f"{model_type}_image_decoder" + ("_multi" if multimask_output else "") + f"{suffix}.onnx"
        )


def encoder_shape_dict(batch_size: int, height: int, width: int) -> Mapping[str, list[int]]:
    assert height == 1024 and width == 1024, "Only 1024x1024 images are supported."
    return {
        "image": [batch_size, 3, height, width],
        "image_features_0": [batch_size, 32, height // 4, width // 4],
        "image_features_1": [batch_size, 64, height // 8, width // 8],
        "image_embeddings": [batch_size, 256, height // 16, width // 16],
    }


def decoder_shape_dict(
    original_image_height: int,
    original_image_width: int,
    num_labels: int = 1,
    max_points: int = 16,
    num_masks: int = 1,
) -> dict:
    height: int = 1024
    width: int = 1024
    return {
        "image_features_0": [1, 32, height // 4, width // 4],
        "image_features_1": [1, 64, height // 8, width // 8],
        "image_embeddings": [1, 256, height // 16, width // 16],
        "point_coords": [num_labels, max_points, 2],
        "point_labels": [num_labels, max_points],
        "input_masks": [num_labels, 1, height // 4, width // 4],
        "has_input_masks": [num_labels],
        "original_image_size": [2],
        "masks": [num_labels, num_masks, original_image_height, original_image_width],
        "iou_predictions": [num_labels, num_masks],
        "low_res_masks": [num_labels, num_masks, height // 4, width // 4],
    }


def compare_tensors_with_tolerance(
    name: str,
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol=5e-3,
    rtol=1e-4,
    mismatch_percentage_tolerance=0.1,
) -> bool:
    assert tensor1.shape == tensor2.shape
    a = tensor1.clone().float()
    b = tensor2.clone().float()

    differences = torch.abs(a - b)
    mismatch_count = (differences > (rtol * torch.max(torch.abs(a), torch.abs(b)) + atol)).sum().item()

    total_elements = a.numel()
    mismatch_percentage = (mismatch_count / total_elements) * 100

    passed = mismatch_percentage < mismatch_percentage_tolerance

    log_func = logger.error if not passed else logger.info
    log_func(
        "%s: mismatched elements percentage %.2f (%d/%d). Verification %s (threshold=%.2f).",
        name,
        mismatch_percentage,
        mismatch_count,
        total_elements,
        "passed" if passed else "failed",
        mismatch_percentage_tolerance,
    )

    return passed


def random_sam2_input_image(batch_size=1, image_height=1024, image_width=1024) -> torch.Tensor:
    image = torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32).cpu()
    return image


def setup_logger(verbose=True):
    if verbose:
        logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s")
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.basicConfig(format="[%(message)s")
        logging.getLogger().setLevel(logging.WARNING)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\dtypes\generic.py ===
""" define generic base classes for pandas objects """
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Type,
    cast,
)

if TYPE_CHECKING:
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
    from pandas.core.arrays import (
        DatetimeArray,
        ExtensionArray,
        NumpyExtensionArray,
        PeriodArray,
        TimedeltaArray,
    )
    from pandas.core.generic import NDFrame


# define abstract base classes to enable isinstance type checking on our
# objects
def create_pandas_abc_type(name, attr, comp):
    def _check(inst) -> bool:
        return getattr(inst, attr, "_typ") in comp

    # https://github.com/python/mypy/issues/1006
    # error: 'classmethod' used with a non-method
    @classmethod  # type: ignore[misc]
    def _instancecheck(cls, inst) -> bool:
        return _check(inst) and not isinstance(inst, type)

    @classmethod  # type: ignore[misc]
    def _subclasscheck(cls, inst) -> bool:
        # Raise instead of returning False
        # This is consistent with default __subclasscheck__ behavior
        if not isinstance(inst, type):
            raise TypeError("issubclass() arg 1 must be a class")

        return _check(inst)

    dct = {"__instancecheck__": _instancecheck, "__subclasscheck__": _subclasscheck}
    meta = type("ABCBase", (type,), dct)
    return meta(name, (), dct)


ABCRangeIndex = cast(
    "Type[RangeIndex]",
    create_pandas_abc_type("ABCRangeIndex", "_typ", ("rangeindex",)),
)
ABCMultiIndex = cast(
    "Type[MultiIndex]",
    create_pandas_abc_type("ABCMultiIndex", "_typ", ("multiindex",)),
)
ABCDatetimeIndex = cast(
    "Type[DatetimeIndex]",
    create_pandas_abc_type("ABCDatetimeIndex", "_typ", ("datetimeindex",)),
)
ABCTimedeltaIndex = cast(
    "Type[TimedeltaIndex]",
    create_pandas_abc_type("ABCTimedeltaIndex", "_typ", ("timedeltaindex",)),
)
ABCPeriodIndex = cast(
    "Type[PeriodIndex]",
    create_pandas_abc_type("ABCPeriodIndex", "_typ", ("periodindex",)),
)
ABCCategoricalIndex = cast(
    "Type[CategoricalIndex]",
    create_pandas_abc_type("ABCCategoricalIndex", "_typ", ("categoricalindex",)),
)
ABCIntervalIndex = cast(
    "Type[IntervalIndex]",
    create_pandas_abc_type("ABCIntervalIndex", "_typ", ("intervalindex",)),
)
ABCIndex = cast(
    "Type[Index]",
    create_pandas_abc_type(
        "ABCIndex",
        "_typ",
        {
            "index",
            "rangeindex",
            "multiindex",
            "datetimeindex",
            "timedeltaindex",
            "periodindex",
            "categoricalindex",
            "intervalindex",
        },
    ),
)


ABCNDFrame = cast(
    "Type[NDFrame]",
    create_pandas_abc_type("ABCNDFrame", "_typ", ("series", "dataframe")),
)
ABCSeries = cast(
    "Type[Series]",
    create_pandas_abc_type("ABCSeries", "_typ", ("series",)),
)
ABCDataFrame = cast(
    "Type[DataFrame]", create_pandas_abc_type("ABCDataFrame", "_typ", ("dataframe",))
)

ABCCategorical = cast(
    "Type[Categorical]",
    create_pandas_abc_type("ABCCategorical", "_typ", ("categorical")),
)
ABCDatetimeArray = cast(
    "Type[DatetimeArray]",
    create_pandas_abc_type("ABCDatetimeArray", "_typ", ("datetimearray")),
)
ABCTimedeltaArray = cast(
    "Type[TimedeltaArray]",
    create_pandas_abc_type("ABCTimedeltaArray", "_typ", ("timedeltaarray")),
)
ABCPeriodArray = cast(
    "Type[PeriodArray]",
    create_pandas_abc_type("ABCPeriodArray", "_typ", ("periodarray",)),
)
ABCExtensionArray = cast(
    "Type[ExtensionArray]",
    create_pandas_abc_type(
        "ABCExtensionArray",
        "_typ",
        # Note: IntervalArray and SparseArray are included bc they have _typ="extension"
        {"extension", "categorical", "periodarray", "datetimearray", "timedeltaarray"},
    ),
)
ABCNumpyExtensionArray = cast(
    "Type[NumpyExtensionArray]",
    create_pandas_abc_type("ABCNumpyExtensionArray", "_typ", ("npy_extension",)),
)