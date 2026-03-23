
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_insert.py ===
# testing/suite/test_insert.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from decimal import Decimal
import uuid

from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid


class LastrowidTest(fixtures.TablesTest):
    run_deletes = "each"

    __backend__ = True

    __requires__ = "implements_get_lastrowid", "autoincrement_insert"

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "autoinc_pk",
            metadata,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("data", String(50)),
            implicit_returning=False,
        )

        Table(
            "manual_pk",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("data", String(50)),
            implicit_returning=False,
        )

    def _assert_round_trip(self, table, conn):
        row = conn.execute(table.select()).first()
        eq_(
            row,
            (
                conn.dialect.default_sequence_base,
                "some data",
            ),
        )

    def test_autoincrement_on_insert(self, connection):
        connection.execute(
            self.tables.autoinc_pk.insert(), dict(data="some data")
        )
        self._assert_round_trip(self.tables.autoinc_pk, connection)

    def test_last_inserted_id(self, connection):
        r = connection.execute(
            self.tables.autoinc_pk.insert(), dict(data="some data")
        )
        pk = connection.scalar(select(self.tables.autoinc_pk.c.id))
        eq_(r.inserted_primary_key, (pk,))

    @requirements.dbapi_lastrowid
    def test_native_lastrowid_autoinc(self, connection):
        r = connection.execute(
            self.tables.autoinc_pk.insert(), dict(data="some data")
        )
        lastrowid = r.lastrowid
        pk = connection.scalar(select(self.tables.autoinc_pk.c.id))
        eq_(lastrowid, pk)


class InsertBehaviorTest(fixtures.TablesTest):
    run_deletes = "each"
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "autoinc_pk",
            metadata,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("data", String(50)),
        )
        Table(
            "manual_pk",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("data", String(50)),
        )
        Table(
            "no_implicit_returning",
            metadata,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("data", String(50)),
            implicit_returning=False,
        )
        Table(
            "includes_defaults",
            metadata,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("data", String(50)),
            Column("x", Integer, default=5),
            Column(
                "y",
                Integer,
                default=literal_column("2", type_=Integer) + literal(2),
            ),
        )

    @testing.variation("style", ["plain", "return_defaults"])
    @testing.variation("executemany", [True, False])
    def test_no_results_for_non_returning_insert(
        self, connection, style, executemany
    ):
        """test another INSERT issue found during #10453"""

        table = self.tables.no_implicit_returning

        stmt = table.insert()
        if style.return_defaults:
            stmt = stmt.return_defaults()

        if executemany:
            data = [
                {"data": "d1"},
                {"data": "d2"},
                {"data": "d3"},
                {"data": "d4"},
                {"data": "d5"},
            ]
        else:
            data = {"data": "d1"}

        r = connection.execute(stmt, data)
        assert not r.returns_rows

    @requirements.autoincrement_insert
    def test_autoclose_on_insert(self, connection):
        r = connection.execute(
            self.tables.autoinc_pk.insert(), dict(data="some data")
        )
        assert r._soft_closed
        assert not r.closed
        assert r.is_insert

        # new as of I8091919d45421e3f53029b8660427f844fee0228; for the moment
        # an insert where the PK was taken from a row that the dialect
        # selected, as is the case for mssql/pyodbc, will still report
        # returns_rows as true because there's a cursor description.  in that
        # case, the row had to have been consumed at least.
        assert not r.returns_rows or r.fetchone() is None

    @requirements.insert_returning
    def test_autoclose_on_insert_implicit_returning(self, connection):
        r = connection.execute(
            # return_defaults() ensures RETURNING will be used,
            # new in 2.0 as sqlite/mariadb offer both RETURNING and
            # cursor.lastrowid
            self.tables.autoinc_pk.insert().return_defaults(),
            dict(data="some data"),
        )
        assert r._soft_closed
        assert not r.closed
        assert r.is_insert

        # note we are experimenting with having this be True
        # as of I8091919d45421e3f53029b8660427f844fee0228 .
        # implicit returning has fetched the row, but it still is a
        # "returns rows"
        assert r.returns_rows

        # and we should be able to fetchone() on it, we just get no row
        eq_(r.fetchone(), None)

        # and the keys, etc.
        eq_(r.keys(), ["id"])

        # but the dialect took in the row already.   not really sure
        # what the best behavior is.

    @requirements.empty_inserts
    def test_empty_insert(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert())
        assert r._soft_closed
        assert not r.closed

        r = connection.execute(
            self.tables.autoinc_pk.select().where(
                self.tables.autoinc_pk.c.id != None
            )
        )
        eq_(len(r.all()), 1)

    @requirements.empty_inserts_executemany
    def test_empty_insert_multiple(self, connection):
        r = connection.execute(self.tables.autoinc_pk.insert(), [{}, {}, {}])
        assert r._soft_closed
        assert not r.closed

        r = connection.execute(
            self.tables.autoinc_pk.select().where(
                self.tables.autoinc_pk.c.id != None
            )
        )

        eq_(len(r.all()), 3)

    @requirements.insert_from_select
    def test_insert_from_select_autoinc(self, connection):
        src_table = self.tables.manual_pk
        dest_table = self.tables.autoinc_pk
        connection.execute(
            src_table.insert(),
            [
                dict(id=1, data="data1"),
                dict(id=2, data="data2"),
                dict(id=3, data="data3"),
            ],
        )

        result = connection.execute(
            dest_table.insert().from_select(
                ("data",),
                select(src_table.c.data).where(
                    src_table.c.data.in_(["data2", "data3"])
                ),
            )
        )

        eq_(result.inserted_primary_key, (None,))

        result = connection.execute(
            select(dest_table.c.data).order_by(dest_table.c.data)
        )
        eq_(result.fetchall(), [("data2",), ("data3",)])

    @requirements.insert_from_select
    def test_insert_from_select_autoinc_no_rows(self, connection):
        src_table = self.tables.manual_pk
        dest_table = self.tables.autoinc_pk

        result = connection.execute(
            dest_table.insert().from_select(
                ("data",),
                select(src_table.c.data).where(
                    src_table.c.data.in_(["data2", "data3"])
                ),
            )
        )
        eq_(result.inserted_primary_key, (None,))

        result = connection.execute(
            select(dest_table.c.data).order_by(dest_table.c.data)
        )

        eq_(result.fetchall(), [])

    @requirements.insert_from_select
    def test_insert_from_select(self, connection):
        table = self.tables.manual_pk
        connection.execute(
            table.insert(),
            [
                dict(id=1, data="data1"),
                dict(id=2, data="data2"),
                dict(id=3, data="data3"),
            ],
        )

        connection.execute(
            table.insert()
            .inline()
            .from_select(
                ("id", "data"),
                select(table.c.id + 5, table.c.data).where(
                    table.c.data.in_(["data2", "data3"])
                ),
            )
        )

        eq_(
            connection.execute(
                select(table.c.data).order_by(table.c.data)
            ).fetchall(),
            [("data1",), ("data2",), ("data2",), ("data3",), ("data3",)],
        )

    @requirements.insert_from_select
    def test_insert_from_select_with_defaults(self, connection):
        table = self.tables.includes_defaults
        connection.execute(
            table.insert(),
            [
                dict(id=1, data="data1"),
                dict(id=2, data="data2"),
                dict(id=3, data="data3"),
            ],
        )

        connection.execute(
            table.insert()
            .inline()
            .from_select(
                ("id", "data"),
                select(table.c.id + 5, table.c.data).where(
                    table.c.data.in_(["data2", "data3"])
                ),
            )
        )

        eq_(
            connection.execute(
                select(table).order_by(table.c.data, table.c.id)
            ).fetchall(),
            [
                (1, "data1", 5, 4),
                (2, "data2", 5, 4),
                (7, "data2", 5, 4),
                (3, "data3", 5, 4),
                (8, "data3", 5, 4),
            ],
        )


class ReturningTest(fixtures.TablesTest):
    run_create_tables = "each"
    __requires__ = "insert_returning", "autoincrement_insert"
    __backend__ = True

    def _assert_round_trip(self, table, conn):
        row = conn.execute(table.select()).first()
        eq_(
            row,
            (
                conn.dialect.default_sequence_base,
                "some data",
            ),
        )

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "autoinc_pk",
            metadata,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("data", String(50)),
        )

    @requirements.fetch_rows_post_commit
    def test_explicit_returning_pk_autocommit(self, connection):
        table = self.tables.autoinc_pk
        r = connection.execute(
            table.insert().returning(table.c.id), dict(data="some data")
        )
        pk = r.first()[0]
        fetched_pk = connection.scalar(select(table.c.id))
        eq_(fetched_pk, pk)

    def test_explicit_returning_pk_no_autocommit(self, connection):
        table = self.tables.autoinc_pk
        r = connection.execute(
            table.insert().returning(table.c.id), dict(data="some data")
        )

        pk = r.first()[0]
        fetched_pk = connection.scalar(select(table.c.id))
        eq_(fetched_pk, pk)

    def test_autoincrement_on_insert_implicit_returning(self, connection):
        connection.execute(
            self.tables.autoinc_pk.insert(), dict(data="some data")
        )
        self._assert_round_trip(self.tables.autoinc_pk, connection)

    def test_last_inserted_id_implicit_returning(self, connection):
        r = connection.execute(
            self.tables.autoinc_pk.insert(), dict(data="some data")
        )
        pk = connection.scalar(select(self.tables.autoinc_pk.c.id))
        eq_(r.inserted_primary_key, (pk,))

    @requirements.insert_executemany_returning
    def test_insertmanyvalues_returning(self, connection):
        r = connection.execute(
            self.tables.autoinc_pk.insert().returning(
                self.tables.autoinc_pk.c.id
            ),
            [
                {"data": "d1"},
                {"data": "d2"},
                {"data": "d3"},
                {"data": "d4"},
                {"data": "d5"},
            ],
        )
        rall = r.all()

        pks = connection.execute(select(self.tables.autoinc_pk.c.id))

        eq_(rall, pks.all())

    @testing.combinations(
        (Double(), 8.5514716, True),
        (
            Double(53),
            8.5514716,
            True,
            testing.requires.float_or_double_precision_behaves_generically,
        ),
        (Float(), 8.5514, True),
        (
            Float(8),
            8.5514,
            True,
            testing.requires.float_or_double_precision_behaves_generically,
        ),
        (
            Numeric(precision=15, scale=12, asdecimal=False),
            8.5514716,
            True,
            testing.requires.literal_float_coercion,
        ),
        (
            Numeric(precision=15, scale=12, asdecimal=True),
            Decimal("8.5514716"),
            False,
        ),
        argnames="type_,value,do_rounding",
    )
    @testing.variation("sort_by_parameter_order", [True, False])
    @testing.variation("multiple_rows", [True, False])
    def test_insert_w_floats(
        self,
        connection,
        metadata,
        sort_by_parameter_order,
        type_,
        value,
        do_rounding,
        multiple_rows,
    ):
        """test #9701.

        this tests insertmanyvalues as well as decimal / floating point
        RETURNING types

        """

        t = Table(
            # Oracle backends seems to be getting confused if
            # this table is named the same as the one
            # in test_imv_returning_datatypes.  use a different name
            "f_t",
            metadata,
            Column("id", Integer, Identity(), primary_key=True),
            Column("value", type_),
        )

        t.create(connection)

        result = connection.execute(
            t.insert().returning(
                t.c.id,
                t.c.value,
                sort_by_parameter_order=bool(sort_by_parameter_order),
            ),
            (
                [{"value": value} for i in range(10)]
                if multiple_rows
                else {"value": value}
            ),
        )

        if multiple_rows:
            i_range = range(1, 11)
        else:
            i_range = range(1, 2)

        # we want to test only that we are getting floating points back
        # with some degree of the original value maintained, that it is not
        # being truncated to an integer.  there's too much variation in how
        # drivers return floats, which should not be relied upon to be
        # exact, for us to just compare as is (works for PG drivers but not
        # others) so we use rounding here.  There's precedent for this
        # in suite/test_types.py::NumericTest as well

        if do_rounding:
            eq_(
                {(id_, round(val_, 5)) for id_, val_ in result},
                {(id_, round(value, 5)) for id_ in i_range},
            )

            eq_(
                {
                    round(val_, 5)
                    for val_ in connection.scalars(select(t.c.value))
                },
                {round(value, 5)},
            )
        else:
            eq_(
                set(result),
                {(id_, value) for id_ in i_range},
            )

            eq_(
                set(connection.scalars(select(t.c.value))),
                {value},
            )

    @testing.combinations(
        (
            "non_native_uuid",
            Uuid(native_uuid=False),
            uuid.uuid4(),
        ),
        (
            "non_native_uuid_str",
            Uuid(as_uuid=False, native_uuid=False),
            str(uuid.uuid4()),
        ),
        (
            "generic_native_uuid",
            Uuid(native_uuid=True),
            uuid.uuid4(),
            testing.requires.uuid_data_type,
        ),
        (
            "generic_native_uuid_str",
            Uuid(as_uuid=False, native_uuid=True),
            str(uuid.uuid4()),
            testing.requires.uuid_data_type,
        ),
        ("UUID", UUID(), uuid.uuid4(), testing.requires.uuid_data_type),
        (
            "LargeBinary1",
            LargeBinary(),
            b"this is binary",
        ),
        ("LargeBinary2", LargeBinary(), b"7\xe7\x9f"),
        argnames="type_,value",
        id_="iaa",
    )
    @testing.variation("sort_by_parameter_order", [True, False])
    @testing.variation("multiple_rows", [True, False])
    @testing.requires.insert_returning
    def test_imv_returning_datatypes(
        self,
        connection,
        metadata,
        sort_by_parameter_order,
        type_,
        value,
        multiple_rows,
    ):
        """test #9739, #9808 (similar to #9701).

        this tests insertmanyvalues in conjunction with various datatypes.

        These tests are particularly for the asyncpg driver which needs
        most types to be explicitly cast for the new IMV format

        """
        t = Table(
            "d_t",
            metadata,
            Column("id", Integer, Identity(), primary_key=True),
            Column("value", type_),
        )

        t.create(connection)

        result = connection.execute(
            t.insert().returning(
                t.c.id,
                t.c.value,
                sort_by_parameter_order=bool(sort_by_parameter_order),
            ),
            (
                [{"value": value} for i in range(10)]
                if multiple_rows
                else {"value": value}
            ),
        )

        if multiple_rows:
            i_range = range(1, 11)
        else:
            i_range = range(1, 2)

        eq_(
            set(result),
            {(id_, value) for id_ in i_range},
        )

        eq_(
            set(connection.scalars(select(t.c.value))),
            {value},
        )


__all__ = ("LastrowidTest", "InsertBehaviorTest", "ReturningTest")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_run.py ===
from __future__ import annotations

import enum
import functools
import gc
import itertools
import random
import select
import sys
import warnings
from collections import deque
from contextlib import AbstractAsyncContextManager, contextmanager, suppress
from contextvars import copy_context
from heapq import heapify, heappop, heappush
from math import inf, isnan
from time import perf_counter
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    NoReturn,
    Protocol,
    cast,
    overload,
)

import attrs
from outcome import Error, Outcome, Value, capture
from sniffio import thread_local as sniffio_library
from sortedcontainers import SortedDict

from .. import _core
from .._abc import Clock, Instrument
from .._deprecate import warn_deprecated
from .._util import NoPublicConstructor, coroutine_or_error, final
from ._asyncgens import AsyncGenerators
from ._concat_tb import concat_tb
from ._entry_queue import EntryQueue, TrioToken
from ._exceptions import Cancelled, RunFinishedError, TrioInternalError
from ._instrumentation import Instruments
from ._ki import KIManager, enable_ki_protection
from ._parking_lot import GLOBAL_PARKING_LOT_BREAKER
from ._run_context import GLOBAL_RUN_CONTEXT as GLOBAL_RUN_CONTEXT
from ._thread_cache import start_thread_soon
from ._traps import (
    Abort,
    CancelShieldedCheckpoint,
    PermanentlyDetachCoroutineObject,
    WaitTaskRescheduled,
    cancel_shielded_checkpoint,
    wait_task_rescheduled,
)

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup


if TYPE_CHECKING:
    import contextvars
    import types
    from collections.abc import (
        Awaitable,
        Callable,
        Generator,
        Iterator,
        Sequence,
    )
    from types import TracebackType

    # for some strange reason Sphinx works with outcome.Outcome, but not Outcome, in
    # start_guest_run. Same with types.FrameType in iter_await_frames
    import outcome
    from typing_extensions import Self, TypeVar, TypeVarTuple, Unpack

    PosArgT = TypeVarTuple("PosArgT")
    StatusT = TypeVar("StatusT", default=None)
    StatusT_contra = TypeVar("StatusT_contra", contravariant=True, default=None)
    BaseExcT = TypeVar("BaseExcT", bound=BaseException)
else:
    from typing import TypeVar

    StatusT = TypeVar("StatusT")
    StatusT_contra = TypeVar("StatusT_contra", contravariant=True)

RetT = TypeVar("RetT")


DEADLINE_HEAP_MIN_PRUNE_THRESHOLD: Final = 1000

# Passed as a sentinel
_NO_SEND: Final[Outcome[object]] = cast("Outcome[object]", object())

# Used to track if an exceptiongroup can be collapsed
NONSTRICT_EXCEPTIONGROUP_NOTE = 'This is a "loose" ExceptionGroup, and may be collapsed by Trio if it only contains one exception - typically after `Cancelled` has been stripped from it. Note this has consequences for exception handling, and strict_exception_groups=True is recommended.'


@final
class _NoStatus(metaclass=NoPublicConstructor):
    """Sentinel for unset TaskStatus._value."""


# Decorator to mark methods public. This does nothing by itself, but
# trio/_tools/gen_exports.py looks for it.
def _public(fn: RetT) -> RetT:
    return fn


# When running under Hypothesis, we want examples to be reproducible and
# shrinkable.  We therefore register `_hypothesis_plugin_setup()` as a
# plugin, so that importing *Hypothesis* will make Trio's task
# scheduling loop deterministic.  We have a test for that, of course.
# Before Hypothesis supported entry-point plugins this integration was
# handled by pytest-trio, but we want it to work in e.g. unittest too.
_ALLOW_DETERMINISTIC_SCHEDULING: Final = False
_r = random.Random()


# no cover because we don't check the hypothesis plugin works with hypothesis
def _hypothesis_plugin_setup() -> None:  # pragma: no cover
    from hypothesis import register_random

    global _ALLOW_DETERMINISTIC_SCHEDULING
    _ALLOW_DETERMINISTIC_SCHEDULING = True  # type: ignore
    register_random(_r)

    # monkeypatch repr_callable to make repr's way better
    # requires importing hypothesis (in the test file or in conftest.py)
    try:
        from hypothesis.internal.reflection import get_pretty_function_description

        import trio.testing._raises_group

        def repr_callable(fun: Callable[[BaseExcT], bool]) -> str:
            # add quotes around the signature
            return repr(get_pretty_function_description(fun))

        trio.testing._raises_group.repr_callable = repr_callable
    except ImportError:
        pass


def _count_context_run_tb_frames() -> int:
    """Count implementation dependent traceback frames from Context.run()

    On CPython, Context.run() is implemented in C and doesn't show up in
    tracebacks. On PyPy, it is implemented in Python and adds 1 frame to
    tracebacks.

    Returns:
        int: Traceback frame count

    """

    def function_with_unique_name_xyzzy() -> NoReturn:
        try:
            1 / 0  # noqa: B018  # We need a ZeroDivisionError to fire
        except ZeroDivisionError:
            raise
        else:  # pragma: no cover
            raise TrioInternalError(
                "A ZeroDivisionError should have been raised, but it wasn't.",
            )

    ctx = copy_context()
    try:
        ctx.run(function_with_unique_name_xyzzy)
    except ZeroDivisionError as exc:
        tb = exc.__traceback__
        # Skip the frame where we caught it
        tb = tb.tb_next  # type: ignore[union-attr]
        count = 0
        while tb.tb_frame.f_code.co_name != "function_with_unique_name_xyzzy":  # type: ignore[union-attr]
            tb = tb.tb_next  # type: ignore[union-attr]
            count += 1
        return count
    else:  # pragma: no cover
        raise TrioInternalError(
            f"The purpose of {function_with_unique_name_xyzzy.__name__} is "
            "to raise a ZeroDivisionError, but it didn't.",
        )


CONTEXT_RUN_TB_FRAMES: Final = _count_context_run_tb_frames()


@attrs.frozen
class SystemClock(Clock):
    # Add a large random offset to our clock to ensure that if people
    # accidentally call time.perf_counter() directly or start comparing clocks
    # between different runs, then they'll notice the bug quickly:
    offset: float = attrs.Factory(lambda: _r.uniform(10000, 200000))

    def start_clock(self) -> None:
        pass

    # In cPython 3, on every platform except Windows, perf_counter is
    # exactly the same as time.monotonic; and on Windows, it uses
    # QueryPerformanceCounter instead of GetTickCount64.
    def current_time(self) -> float:
        return self.offset + perf_counter()

    def deadline_to_sleep_time(self, deadline: float) -> float:
        return deadline - self.current_time()


class IdlePrimedTypes(enum.Enum):
    WAITING_FOR_IDLE = 1
    AUTOJUMP_CLOCK = 2


################################################################
# CancelScope and friends
################################################################


def collapse_exception_group(
    excgroup: BaseExceptionGroup[BaseException],
) -> BaseException:
    """Recursively collapse any single-exception groups into that single contained
    exception.

    """
    exceptions = list(excgroup.exceptions)
    modified = False
    for i, exc in enumerate(exceptions):
        if isinstance(exc, BaseExceptionGroup):
            new_exc = collapse_exception_group(exc)
            if new_exc is not exc:
                modified = True
                exceptions[i] = new_exc

    if (
        len(exceptions) == 1
        and isinstance(excgroup, BaseExceptionGroup)
        and NONSTRICT_EXCEPTIONGROUP_NOTE in getattr(excgroup, "__notes__", ())
    ):
        exceptions[0].__traceback__ = concat_tb(
            excgroup.__traceback__,
            exceptions[0].__traceback__,
        )
        return exceptions[0]
    elif modified:
        return excgroup.derive(exceptions)
    else:
        return excgroup


@attrs.define(eq=False)
class Deadlines:
    """A container of deadlined cancel scopes.

    Only contains scopes with non-infinite deadlines that are currently
    attached to at least one task.

    """

    # Heap of (deadline, id(CancelScope), CancelScope)
    _heap: list[tuple[float, int, CancelScope]] = attrs.Factory(list)
    # Count of active deadlines (those that haven't been changed)
    _active: int = 0

    def add(self, deadline: float, cancel_scope: CancelScope) -> None:
        heappush(self._heap, (deadline, id(cancel_scope), cancel_scope))
        self._active += 1

    def remove(self, deadline: float, cancel_scope: CancelScope) -> None:
        self._active -= 1

    def next_deadline(self) -> float:
        while self._heap:
            deadline, _, cancel_scope = self._heap[0]
            if deadline == cancel_scope._registered_deadline:
                return deadline
            else:
                # This entry is stale; discard it and try again
                heappop(self._heap)
        return inf

    def _prune(self) -> None:
        # In principle, it's possible for a cancel scope to toggle back and
        # forth repeatedly between the same two deadlines, and end up with
        # lots of stale entries that *look* like they're still active, because
        # their deadline is correct, but in fact are redundant. So when
        # pruning we have to eliminate entries with the wrong deadline, *and*
        # eliminate duplicates.
        seen = set()
        pruned_heap = []
        for deadline, tiebreaker, cancel_scope in self._heap:
            if deadline == cancel_scope._registered_deadline:
                if cancel_scope in seen:
                    continue
                seen.add(cancel_scope)
                pruned_heap.append((deadline, tiebreaker, cancel_scope))
        # See test_cancel_scope_deadline_duplicates for a test that exercises
        # this assert:
        assert len(pruned_heap) == self._active
        heapify(pruned_heap)
        self._heap = pruned_heap

    def expire(self, now: float) -> bool:
        did_something = False
        while self._heap and self._heap[0][0] <= now:
            deadline, _, cancel_scope = heappop(self._heap)
            if deadline == cancel_scope._registered_deadline:
                did_something = True
                # This implicitly calls self.remove(), so we don't need to
                # decrement _active here
                cancel_scope.cancel()
        # If we've accumulated too many stale entries, then prune the heap to
        # keep it under control. (We only do this occasionally in a batch, to
        # keep the amortized cost down)
        if len(self._heap) > self._active * 2 + DEADLINE_HEAP_MIN_PRUNE_THRESHOLD:
            self._prune()
        return did_something


@attrs.define(eq=False)
class CancelStatus:
    """Tracks the cancellation status for a contiguous extent
    of code that will become cancelled, or not, as a unit.

    Each task has at all times a single "active" CancelStatus whose
    cancellation state determines whether checkpoints executed in that
    task raise Cancelled. Each 'with CancelScope(...)' context is
    associated with a particular CancelStatus.  When a task enters
    such a context, a CancelStatus is created which becomes the active
    CancelStatus for that task; when the 'with' block is exited, the
    active CancelStatus for that task goes back to whatever it was
    before.

    CancelStatus objects are arranged in a tree whose structure
    mirrors the lexical nesting of the cancel scope contexts.  When a
    CancelStatus becomes cancelled, it notifies all of its direct
    children, who become cancelled in turn (and continue propagating
    the cancellation down the tree) unless they are shielded. (There
    will be at most one such child except in the case of a
    CancelStatus that immediately encloses a nursery.) At the leaves
    of this tree are the tasks themselves, which get woken up to deliver
    an abort when their direct parent CancelStatus becomes cancelled.

    You can think of CancelStatus as being responsible for the
    "plumbing" of cancellations as oppposed to CancelScope which is
    responsible for the origination of them.

    """

    # Our associated cancel scope. Can be any object with attributes
    # `deadline`, `shield`, and `cancel_called`, but in current usage
    # is always a CancelScope object. Must not be None.
    _scope: CancelScope = attrs.field(alias="scope")

    # True iff the tasks in self._tasks should receive cancellations
    # when they checkpoint. Always True when scope.cancel_called is True;
    # may also be True due to a cancellation propagated from our
    # parent.  Unlike scope.cancel_called, this does not necessarily stay
    # true once it becomes true. For example, we might become
    # effectively cancelled due to the cancel scope two levels out
    # becoming cancelled, but then the cancel scope one level out
    # becomes shielded so we're not effectively cancelled anymore.
    effectively_cancelled: bool = False

    # The CancelStatus whose cancellations can propagate to us; we
    # become effectively cancelled when they do, unless scope.shield
    # is True.  May be None (for the outermost CancelStatus in a call
    # to trio.run(), briefly during TaskStatus.started(), or during
    # recovery from misnesting of cancel scopes).
    _parent: CancelStatus | None = attrs.field(default=None, repr=False, alias="parent")

    # All of the CancelStatuses that have this CancelStatus as their parent.
    _children: set[CancelStatus] = attrs.field(factory=set, init=False, repr=False)

    # Tasks whose cancellation state is currently tied directly to
    # the cancellation state of this CancelStatus object. Don't modify
    # this directly; instead, use Task._activate_cancel_status().
    # Invariant: all(task._cancel_status is self for task in self._tasks)
    _tasks: set[Task] = attrs.field(factory=set, init=False, repr=False)

    # Set to True on still-active cancel statuses that are children
    # of a cancel status that's been closed. This is used to permit
    # recovery from misnested cancel scopes (well, at least enough
    # recovery to show a useful traceback).
    abandoned_by_misnesting: bool = attrs.field(default=False, init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if self._parent is not None:
            self._parent._children.add(self)
            self.recalculate()

    # parent/children/tasks accessors are used by TaskStatus.started()

    @property
    def parent(self) -> CancelStatus | None:
        return self._parent

    @parent.setter
    def parent(self, parent: CancelStatus | None) -> None:
        if self._parent is not None:
            self._parent._children.remove(self)
        self._parent = parent
        if self._parent is not None:
            self._parent._children.add(self)
            self.recalculate()

    @property
    def children(self) -> frozenset[CancelStatus]:
        return frozenset(self._children)

    @property
    def tasks(self) -> frozenset[Task]:
        return frozenset(self._tasks)

    def encloses(self, other: CancelStatus | None) -> bool:
        """Returns true if this cancel status is a direct or indirect
        parent of cancel status *other*, or if *other* is *self*.
        """
        while other is not None:
            if other is self:
                return True
            other = other.parent
        return False

    def close(self) -> None:
        self.parent = None  # now we're not a child of self.parent anymore
        if self._tasks or self._children:
            # Cancel scopes weren't exited in opposite order of being
            # entered. CancelScope._close() deals with raising an error
            # if appropriate; our job is to leave things in a reasonable
            # state for unwinding our dangling children. We choose to leave
            # this part of the CancelStatus tree unlinked from everyone
            # else, cancelled, and marked so that exiting a CancelScope
            # within the abandoned subtree doesn't affect the active
            # CancelStatus. Note that it's possible for us to get here
            # without CancelScope._close() raising an error, if a
            # nursery's cancel scope is closed within the nursery's
            # nested child and no other cancel scopes are involved,
            # but in that case task_exited() will deal with raising
            # the error.
            self._mark_abandoned()

            # Since our CancelScope is about to forget about us, and we
            # have no parent anymore, there's nothing left to call
            # recalculate(). So, we can stay cancelled by setting
            # effectively_cancelled and updating our children.
            self.effectively_cancelled = True
            for task in self._tasks:
                task._attempt_delivery_of_any_pending_cancel()
            for child in self._children:
                child.recalculate()

    @property
    def parent_cancellation_is_visible_to_us(self) -> bool:
        return (
            self._parent is not None
            and not self._scope.shield
            and self._parent.effectively_cancelled
        )

    def recalculate(self) -> None:
        # This does a depth-first traversal over this and descendent cancel
        # statuses, to ensure their state is up-to-date. It's basically a
        # recursive algorithm, but we use an explicit stack to avoid any
        # issues with stack overflow.
        todo = [self]
        while todo:
            current = todo.pop()
            new_state = (
                current._scope.cancel_called
                or current.parent_cancellation_is_visible_to_us
            )
            if new_state != current.effectively_cancelled:
                current.effectively_cancelled = new_state
                if new_state:
                    for task in current._tasks:
                        task._attempt_delivery_of_any_pending_cancel()
                todo.extend(current._children)

    def _mark_abandoned(self) -> None:
        self.abandoned_by_misnesting = True
        for child in self._children:
            child._mark_abandoned()

    def effective_deadline(self) -> float:
        if self.effectively_cancelled:
            return -inf
        if self._parent is None or self._scope.shield:
            return self._scope.deadline
        return min(self._scope.deadline, self._parent.effective_deadline())


MISNESTING_ADVICE = """
This is probably a bug in your code, that has caused Trio's internal state to
become corrupted. We'll do our best to recover, but from now on there are
no guarantees.

Typically this is caused by one of the following:
  - yielding within a generator or async generator that's opened a cancel
    scope or nursery (unless the generator is a @contextmanager or
    @asynccontextmanager); see https://github.com/python-trio/trio/issues/638
  - manually calling __enter__ or __exit__ on a trio.CancelScope, or
    __aenter__ or __aexit__ on the object returned by trio.open_nursery();
    doing so correctly is difficult and you should use @[async]contextmanager
    instead, or maybe [Async]ExitStack
  - using [Async]ExitStack to interleave the entries/exits of cancel scopes
    and/or nurseries in a way that couldn't be achieved by some nesting of
    'with' and 'async with' blocks
  - using the low-level coroutine object protocol to execute some parts of
    an async function in a different cancel scope/nursery context than
    other parts
If you don't believe you're doing any of these things, please file a bug:
https://github.com/python-trio/trio/issues/new
"""


@final
@attrs.define(eq=False, repr=False)
class CancelScope:
    """A *cancellation scope*: the link between a unit of cancellable
    work and Trio's cancellation system.

    A :class:`CancelScope` becomes associated with some cancellable work
    when it is used as a context manager surrounding that work::

        cancel_scope = trio.CancelScope()
        ...
        with cancel_scope:
            await long_running_operation()

    Inside the ``with`` block, a cancellation of ``cancel_scope`` (via
    a call to its :meth:`cancel` method or via the expiry of its
    :attr:`deadline`) will immediately interrupt the
    ``long_running_operation()`` by raising :exc:`Cancelled` at its
    next :ref:`checkpoint <checkpoints>`.

    The context manager ``__enter__`` returns the :class:`CancelScope`
    object itself, so you can also write ``with trio.CancelScope() as
    cancel_scope:``.

    If a cancel scope becomes cancelled before entering its ``with`` block,
    the :exc:`Cancelled` exception will be raised at the first
    checkpoint inside the ``with`` block. This allows a
    :class:`CancelScope` to be created in one :ref:`task <tasks>` and
    passed to another, so that the first task can later cancel some work
    inside the second.

    Cancel scopes are not reusable or reentrant; that is, each cancel
    scope can be used for at most one ``with`` block.  (You'll get a
    :exc:`RuntimeError` if you violate this rule.)

    The :class:`CancelScope` constructor takes initial values for the
    cancel scope's :attr:`deadline` and :attr:`shield` attributes; these
    may be freely modified after construction, whether or not the scope
    has been entered yet, and changes take immediate effect.
    """

    _cancel_status: CancelStatus | None = attrs.field(default=None, init=False)
    _has_been_entered: bool = attrs.field(default=False, init=False)
    _registered_deadline: float = attrs.field(default=inf, init=False)
    _cancel_called: bool = attrs.field(default=False, init=False)
    cancelled_caught: bool = attrs.field(default=False, init=False)

    # Constructor arguments:
    _relative_deadline: float = attrs.field(
        default=inf,
        kw_only=True,
        alias="relative_deadline",
    )
    _deadline: float = attrs.field(default=inf, kw_only=True, alias="deadline")
    _shield: bool = attrs.field(default=False, kw_only=True, alias="shield")

    def __attrs_post_init__(self) -> None:
        if isnan(self._deadline):
            raise ValueError("deadline must not be NaN")
        if isnan(self._relative_deadline):
            raise ValueError("relative deadline must not be NaN")
        if self._relative_deadline < 0:
            raise ValueError("timeout must be non-negative")
        if self._relative_deadline != inf and self._deadline != inf:
            raise ValueError(
                "Cannot specify both a deadline and a relative deadline",
            )

    @enable_ki_protection
    def __enter__(self) -> Self:
        task = _core.current_task()
        if self._has_been_entered:
            raise RuntimeError(
                "Each CancelScope may only be used for a single 'with' block",
            )
        self._has_been_entered = True

        if self._relative_deadline != inf:
            assert self._deadline == inf
            self._deadline = current_time() + self._relative_deadline
            self._relative_deadline = inf

        if current_time() >= self._deadline:
            self.cancel()
        with self._might_change_registered_deadline():
            self._cancel_status = CancelStatus(scope=self, parent=task._cancel_status)
            task._activate_cancel_status(self._cancel_status)
        return self

    def _close(self, exc: BaseException | None) -> BaseException | None:
        if self._cancel_status is None:
            new_exc = RuntimeError(
                f"Cancel scope stack corrupted: attempted to exit {self!r} "
                "which had already been exited",
            )
            new_exc.__context__ = exc
            return new_exc
        scope_task = current_task()
        if scope_task._cancel_status is not self._cancel_status:
            # Cancel scope misnesting: this cancel scope isn't the most
            # recently opened by this task (that's still open). That is,
            # our assumptions about context managers forming a stack
            # have been violated. Try and make the best of it.
            if self._cancel_status.abandoned_by_misnesting:
                # We are an inner cancel scope that was still active when
                # some outer scope was closed. The closure of that outer
                # scope threw an error, so we don't need to throw another
                # one; it would just confuse the traceback.
                pass
            elif not self._cancel_status.encloses(scope_task._cancel_status):
                # This task isn't even indirectly contained within the
                # cancel scope it's trying to close. Raise an error
                # without changing any state.
                new_exc = RuntimeError(
                    f"Cancel scope stack corrupted: attempted to exit {self!r} "
                    f"from unrelated {scope_task!r}\n{MISNESTING_ADVICE}",
                )
                new_exc.__context__ = exc
                return new_exc
            else:
                # Otherwise, there's some inner cancel scope(s) that
                # we're abandoning by closing this outer one.
                # CancelStatus.close() will take care of the plumbing;
                # we just need to make sure we don't let the error
                # pass silently.
                new_exc = RuntimeError(
                    f"Cancel scope stack corrupted: attempted to exit {self!r} "
                    f"in {scope_task!r} that's still within its child {scope_task._cancel_status._scope!r}\n{MISNESTING_ADVICE}",
                )
                new_exc.__context__ = exc
                exc = new_exc
                scope_task._activate_cancel_status(self._cancel_status.parent)
        else:
            scope_task._activate_cancel_status(self._cancel_status.parent)
        if (
            exc is not None
            and self._cancel_status.effectively_cancelled
            and not self._cancel_status.parent_cancellation_is_visible_to_us
        ):
            if isinstance(exc, Cancelled):
                self.cancelled_caught = True
                exc = None
            elif isinstance(exc, BaseExceptionGroup):
                matched, exc = exc.split(Cancelled)
                if matched:
                    self.cancelled_caught = True

                if exc:
                    exc = collapse_exception_group(exc)

        self._cancel_status.close()
        with self._might_change_registered_deadline():
            self._cancel_status = None
        return exc

    @enable_ki_protection
    def __exit__(
        self,
        etype: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        # NB: NurseryManager calls _close() directly rather than __exit__(),
        # so __exit__() must be just _close() plus this logic for adapting
        # the exception-filtering result to the context manager API.

        # Tracebacks show the 'raise' line below out of context, so let's give
        # this variable a name that makes sense out of context.
        remaining_error_after_cancel_scope = self._close(exc)
        if remaining_error_after_cancel_scope is None:
            return True
        elif remaining_error_after_cancel_scope is exc:
            return False
        else:
            # Copied verbatim from the old MultiErrorCatcher.  Python doesn't
            # allow us to encapsulate this __context__ fixup.
            old_context = remaining_error_after_cancel_scope.__context__
            try:
                raise remaining_error_after_cancel_scope
            finally:
                _, value, _ = sys.exc_info()
                assert value is remaining_error_after_cancel_scope
                value.__context__ = old_context
                # delete references from locals to avoid creating cycles
                # see test_cancel_scope_exit_doesnt_create_cyclic_garbage
                # Note: still relevant
                del remaining_error_after_cancel_scope, value, _, exc

    def __repr__(self) -> str:
        if self._cancel_status is not None:
            binding = "active"
        elif self._has_been_entered:
            binding = "exited"
        else:
            binding = "unbound"

        if self._cancel_called:
            state = ", cancelled"
        elif self._deadline == inf:
            state = ""
        else:
            try:
                now = current_time()
            except RuntimeError:  # must be called from async context
                state = ""
            else:
                state = ", deadline is {:.2f} seconds {}".format(
                    abs(self._deadline - now),
                    "from now" if self._deadline >= now else "ago",
                )

        return f"<trio.CancelScope at {id(self):#x}, {binding}{state}>"

    @contextmanager
    @enable_ki_protection
    def _might_change_registered_deadline(self) -> Iterator[None]:
        try:
            yield
        finally:
            old = self._registered_deadline
            if self._cancel_status is None or self._cancel_called:
                new = inf
            else:
                new = self._deadline
            if old != new:
                self._registered_deadline = new
                runner = GLOBAL_RUN_CONTEXT.runner
                if runner.is_guest:
                    old_next_deadline = runner.deadlines.next_deadline()
                if old != inf:
                    runner.deadlines.remove(old, self)
                if new != inf:
                    runner.deadlines.add(new, self)
                if runner.is_guest:
                    new_next_deadline = runner.deadlines.next_deadline()
                    if old_next_deadline != new_next_deadline:
                        runner.force_guest_tick_asap()

    @property
    def deadline(self) -> float:
        """Read-write, :class:`float`. An absolute time on the current
        run's clock at which this scope will automatically become
        cancelled. You can adjust the deadline by modifying this
        attribute, e.g.::

           # I need a little more time!
           cancel_scope.deadline += 30

        Note that for efficiency, the core run loop only checks for
        expired deadlines every once in a while. This means that in
        certain cases there may be a short delay between when the clock
        says the deadline should have expired, and when checkpoints
        start raising :exc:`~trio.Cancelled`. This is a very obscure
        corner case that you're unlikely to notice, but we document it
        for completeness. (If this *does* cause problems for you, of
        course, then `we want to know!
        <https://github.com/python-trio/trio/issues>`__)

        Defaults to :data:`math.inf`, which means "no deadline", though
        this can be overridden by the ``deadline=`` argument to
        the :class:`~trio.CancelScope` constructor.
        """
        if self._relative_deadline != inf:
            assert self._deadline == inf
            warnings.warn(
                DeprecationWarning(
                    "unentered relative cancel scope does not have an absolute deadline. Use `.relative_deadline`",
                ),
                stacklevel=2,
            )
            return current_time() + self._relative_deadline
        return self._deadline

    @deadline.setter
    def deadline(self, new_deadline: float) -> None:
        if isnan(new_deadline):
            raise ValueError("deadline must not be NaN")
        if self._relative_deadline != inf:
            assert self._deadline == inf
            warnings.warn(
                DeprecationWarning(
                    "unentered relative cancel scope does not have an absolute deadline. Transforming into an absolute cancel scope. First set `.relative_deadline = math.inf` if you do want an absolute cancel scope.",
                ),
                stacklevel=2,
            )
            self._relative_deadline = inf
        with self._might_change_registered_deadline():
            self._deadline = float(new_deadline)

    @property
    def relative_deadline(self) -> float:
        """Read-write, :class:`float`. The number of seconds remaining until this
        scope's deadline, relative to the current time.

        Defaults to :data:`math.inf` ("no deadline"). Must be non-negative.

        When modified
        Before entering: sets the deadline relative to when the scope enters.
        After entering: sets a new deadline relative to the current time.

        Raises:
          RuntimeError: if trying to read or modify an unentered scope with an absolute deadline, i.e. when :attr:`is_relative` is ``False``.
        """
        if self._has_been_entered:
            return self._deadline - current_time()
        elif self._deadline != inf:
            assert self._relative_deadline == inf
            raise RuntimeError(
                "unentered non-relative cancel scope does not have a relative deadline",
            )
        return self._relative_deadline

    @relative_deadline.setter
    def relative_deadline(self, new_relative_deadline: float) -> None:
        if isnan(new_relative_deadline):
            raise ValueError("relative deadline must not be NaN")
        if new_relative_deadline < 0:
            raise ValueError("relative deadline must be non-negative")
        if self._has_been_entered:
            with self._might_change_registered_deadline():
                self._deadline = current_time() + float(new_relative_deadline)
        elif self._deadline != inf:
            assert self._relative_deadline == inf
            raise RuntimeError(
                "unentered non-relative cancel scope does not have a relative deadline",
            )
        else:
            self._relative_deadline = new_relative_deadline

    @property
    def is_relative(self) -> bool | None:
        """Returns None after entering. Returns False if both deadline and
        relative_deadline are inf."""
        assert not (self._deadline != inf and self._relative_deadline != inf)
        if self._has_been_entered:
            return None
        return self._relative_deadline != inf

    @property
    def shield(self) -> bool:
        """Read-write, :class:`bool`, default :data:`False`. So long as
        this is set to :data:`True`, then the code inside this scope
        will not receive :exc:`~trio.Cancelled` exceptions from scopes
        that are outside this scope. They can still receive
        :exc:`~trio.Cancelled` exceptions from (1) this scope, or (2)
        scopes inside this scope. You can modify this attribute::

           with trio.CancelScope() as cancel_scope:
               cancel_scope.shield = True
               # This cannot be interrupted by any means short of
               # killing the process:
               await sleep(10)

               cancel_scope.shield = False
               # Now this can be cancelled normally:
               await sleep(10)

        Defaults to :data:`False`, though this can be overridden by the
        ``shield=`` argument to the :class:`~trio.CancelScope` constructor.
        """
        return self._shield

    @shield.setter
    @enable_ki_protection
    def shield(self, new_value: bool) -> None:
        if not isinstance(new_value, bool):
            raise TypeError("shield must be a bool")
        self._shield = new_value
        if self._cancel_status is not None:
            self._cancel_status.recalculate()

    @enable_ki_protection
    def cancel(self) -> None:
        """Cancels this scope immediately.

        This method is idempotent, i.e., if the scope was already
        cancelled then this method silently does nothing.
        """
        if self._cancel_called:
            return
        with self._might_change_registered_deadline():
            self._cancel_called = True
        if self._cancel_status is not None:
            self._cancel_status.recalculate()

    @property
    def cancel_called(self) -> bool:
        """Readonly :class:`bool`. Records whether cancellation has been
        requested for this scope, either by an explicit call to
        :meth:`cancel` or by the deadline expiring.

        This attribute being True does *not* necessarily mean that the
        code within the scope has been, or will be, affected by the
        cancellation. For example, if :meth:`cancel` was called after
        the last checkpoint in the ``with`` block, when it's too late to
        deliver a :exc:`~trio.Cancelled` exception, then this attribute
        will still be True.

        This attribute is mostly useful for debugging and introspection.
        If you want to know whether or not a chunk of code was actually
        cancelled, then :attr:`cancelled_caught` is usually more
        appropriate.
        """
        if (  # noqa: SIM102  # collapsible-if but this way is nicer
            self._cancel_status is not None or not self._has_been_entered
        ):
            # Scope is active or not yet entered: make sure cancel_called
            # is true if the deadline has passed. This shouldn't
            # be able to actually change behavior, since we check for
            # deadline expiry on scope entry and at every checkpoint,
            # but it makes the value returned by cancel_called more
            # closely match expectations.
            if not self._cancel_called and current_time() >= self._deadline:
                self.cancel()
        return self._cancel_called


################################################################
# Nursery and friends
################################################################


class TaskStatus(Protocol[StatusT_contra]):
    """The interface provided by :meth:`Nursery.start()` to the spawned task.

    This is provided via the ``task_status`` keyword-only parameter.
    """

    @overload
    def started(self: TaskStatus[None]) -> None: ...

    @overload
    def started(self, value: StatusT_contra) -> None: ...

    def started(self, value: StatusT_contra | None = None) -> None:
        """Tasks call this method to indicate that they have initialized.

        See `nursery.start() <trio.Nursery.start>` for more information.
        """


# This code needs to be read alongside the code from Nursery.start to make
# sense.
@attrs.define(eq=False, repr=False, slots=False)
class _TaskStatus(TaskStatus[StatusT]):
    _old_nursery: Nursery
    _new_nursery: Nursery
    # NoStatus is a sentinel.
    _value: StatusT | type[_NoStatus] = _NoStatus

    def __repr__(self) -> str:
        return f"<Task status object at {id(self):#x}>"

    @overload
    def started(self: _TaskStatus[None]) -> None: ...

    @overload
    def started(self: _TaskStatus[StatusT], value: StatusT) -> None: ...

    def started(self, value: StatusT | None = None) -> None:
        if self._value is not _NoStatus:
            raise RuntimeError("called 'started' twice on the same task status")
        self._value = cast("StatusT", value)  # If None, StatusT == None

        # If the old nursery is cancelled, then quietly quit now; the child
        # will eventually exit on its own, and we don't want to risk moving
        # children that might have propagating Cancelled exceptions into
        # a place with no cancelled cancel scopes to catch them.
        assert self._old_nursery._cancel_status is not None
        if self._old_nursery._cancel_status.effectively_cancelled:
            return

        # Can't be closed, b/c we checked in start() and then _pending_starts
        # should keep it open.
        assert not self._new_nursery._closed

        # Move tasks from the old nursery to the new
        tasks = self._old_nursery._children
        self._old_nursery._children = set()
        for task in tasks:
            task._parent_nursery = self._new_nursery
            task._eventual_parent_nursery = None
            self._new_nursery._children.add(task)

        # Move all children of the old nursery's cancel status object
        # to be underneath the new nursery instead. This includes both
        # tasks and child cancel status objects.
        # NB: If the new nursery is cancelled, reparenting a cancel
        # status to be underneath it can invoke an abort_fn, which might
        # do something evil like cancel the old nursery. We thus break
        # everything off from the old nursery before we start attaching
        # anything to the new.
        cancel_status_children = self._old_nursery._cancel_status.children
        cancel_status_tasks = set(self._old_nursery._cancel_status.tasks)
        cancel_status_tasks.discard(self._old_nursery._parent_task)
        for cancel_status in cancel_status_children:
            cancel_status.parent = None
        for task in cancel_status_tasks:
            task._activate_cancel_status(None)
        for cancel_status in cancel_status_children:
            cancel_status.parent = self._new_nursery._cancel_status
        for task in cancel_status_tasks:
            task._activate_cancel_status(self._new_nursery._cancel_status)

        # That should have removed all the children from the old nursery
        assert not self._old_nursery._children

        # And finally, poke the old nursery so it notices that all its
        # children have disappeared and can exit.
        self._old_nursery._check_nursery_closed()


@attrs.define(slots=False)
class NurseryManager:
    """Nursery context manager.

    Note we explicitly avoid @asynccontextmanager and @async_generator
    since they add a lot of extraneous stack frames to exceptions, as
    well as cause problematic behavior with handling of StopIteration
    and StopAsyncIteration.

    """

    strict_exception_groups: bool = True

    @enable_ki_protection
    async def __aenter__(self) -> Nursery:
        self._scope = CancelScope()
        self._scope.__enter__()
        self._nursery = Nursery._create(
            current_task(),
            self._scope,
            self.strict_exception_groups,
        )
        return self._nursery

    @enable_ki_protection
    async def __aexit__(
        self,
        etype: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        new_exc = await self._nursery._nested_child_finished(exc)
        # Tracebacks show the 'raise' line below out of context, so let's give
        # this variable a name that makes sense out of context.
        combined_error_from_nursery = self._scope._close(new_exc)
        if combined_error_from_nursery is None:
            return True
        elif combined_error_from_nursery is exc:
            return False
        else:
            # Copied verbatim from the old MultiErrorCatcher.  Python doesn't
            # allow us to encapsulate this __context__ fixup.
            old_context = combined_error_from_nursery.__context__
            try:
                raise combined_error_from_nursery
            finally:
                _, value, _ = sys.exc_info()
                assert value is combined_error_from_nursery
                value.__context__ = old_context
                # delete references from locals to avoid creating cycles
                # see test_cancel_scope_exit_doesnt_create_cyclic_garbage
                del _, combined_error_from_nursery, value, new_exc

    # make sure these raise errors in static analysis if called
    if not TYPE_CHECKING:

        def __enter__(self) -> NoReturn:
            raise RuntimeError(
                "use 'async with open_nursery(...)', not 'with open_nursery(...)'",
            )

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> NoReturn:  # pragma: no cover
            raise AssertionError("Never called, but should be defined")


def open_nursery(
    strict_exception_groups: bool | None = None,
) -> AbstractAsyncContextManager[Nursery]:
    """Returns an async context manager which must be used to create a
    new `Nursery`.

    It does not block on entry; on exit it blocks until all child tasks
    have exited. If no child tasks are running on exit, it will insert a
    schedule point (but no cancellation point) - equivalent to
    :func:`trio.lowlevel.cancel_shielded_checkpoint`. This means a nursery
    is never the source of a cancellation exception, it only propagates it
    from sub-tasks.

    Args:
      strict_exception_groups (bool): Unless set to False, even a single raised exception
          will be wrapped in an exception group. If not specified, uses the value passed
          to :func:`run`, which defaults to true. Setting it to False will be deprecated
          and ultimately removed in a future version of Trio.

    """
    # only warn if explicitly set to falsy, not if we get it from the global context.
    if strict_exception_groups is not None and not strict_exception_groups:
        warn_deprecated(
            "open_nursery(strict_exception_groups=False)",
            version="0.25.0",
            issue=2929,
            instead=(
                "the default value of True and rewrite exception handlers to handle ExceptionGroups. "
                "See https://trio.readthedocs.io/en/stable/reference-core.html#designing-for-multiple-errors"
            ),
            use_triodeprecationwarning=True,
        )

    if strict_exception_groups is None:
        strict_exception_groups = GLOBAL_RUN_CONTEXT.runner.strict_exception_groups

    return NurseryManager(strict_exception_groups=strict_exception_groups)


@final
class Nursery(metaclass=NoPublicConstructor):
    """A context which may be used to spawn (or cancel) child tasks.

    Not constructed directly, use `open_nursery` instead.

    The nursery will remain open until all child tasks have completed,
    or until it is cancelled, at which point it will cancel all its
    remaining child tasks and close.

    Nurseries ensure the absence of orphaned Tasks, since all running
    tasks will belong to an open Nursery.

    Attributes:
        cancel_scope:
            Creating a nursery also implicitly creates a cancellation scope,
            which is exposed as the :attr:`cancel_scope` attribute. This is
            used internally to implement the logic where if an error occurs
            then ``__aexit__`` cancels all children, but you can use it for
            other things, e.g. if you want to explicitly cancel all children
            in response to some external event.
    """

    def __init__(
        self,
        parent_task: Task,
        cancel_scope: CancelScope,
        strict_exception_groups: bool,
    ) -> None:
        self._parent_task = parent_task
        self._strict_exception_groups = strict_exception_groups
        parent_task._child_nurseries.append(self)
        # the cancel status that children inherit - we take a snapshot, so it
        # won't be affected by any changes in the parent.
        self._cancel_status = parent_task._cancel_status
        # the cancel scope that directly surrounds us; used for cancelling all
        # children.
        self.cancel_scope = cancel_scope
        assert self.cancel_scope._cancel_status is self._cancel_status
        self._children: set[Task] = set()
        self._pending_excs: list[BaseException] = []
        # The "nested child" is how this code refers to the contents of the
        # nursery's 'async with' block, which acts like a child Task in all
        # the ways we can make it.
        self._nested_child_running = True
        self._parent_waiting_in_aexit = False
        self._pending_starts = 0
        self._closed = False

    @property
    def child_tasks(self) -> frozenset[Task]:
        """(`frozenset`): Contains all the child :class:`~trio.lowlevel.Task`
        objects which are still running."""
        return frozenset(self._children)

    @property
    def parent_task(self) -> Task:
        "(`~trio.lowlevel.Task`):  The Task that opened this nursery."
        return self._parent_task

    def _add_exc(self, exc: BaseException) -> None:
        self._pending_excs.append(exc)
        self.cancel_scope.cancel()

    def _check_nursery_closed(self) -> None:
        if not any([self._nested_child_running, self._children, self._pending_starts]):
            self._closed = True
            if self._parent_waiting_in_aexit:
                self._parent_waiting_in_aexit = False
                GLOBAL_RUN_CONTEXT.runner.reschedule(self._parent_task)

    def _child_finished(
        self,
        task: Task,
        outcome: Outcome[object],
    ) -> None:
        self._children.remove(task)
        if isinstance(outcome, Error):
            self._add_exc(outcome.error)
        self._check_nursery_closed()

    async def _nested_child_finished(
        self,
        nested_child_exc: BaseException | None,
    ) -> BaseException | None:
        # Returns ExceptionGroup instance (or any exception if the nursery is in loose mode
        # and there is just one contained exception) if there are pending exceptions
        if nested_child_exc is not None:
            self._add_exc(nested_child_exc)
        self._nested_child_running = False
        self._check_nursery_closed()

        if not self._closed:
            # If we have a KeyboardInterrupt injected, we want to save it in
            # the nursery's final exceptions list. But if it's just a
            # Cancelled, then we don't -- see gh-1457.
            def aborted(raise_cancel: _core.RaiseCancelT) -> Abort:
                exn = capture(raise_cancel).error
                if not isinstance(exn, Cancelled):
                    self._add_exc(exn)
                # see test_cancel_scope_exit_doesnt_create_cyclic_garbage
                del exn  # prevent cyclic garbage creation
                return Abort.FAILED

            self._parent_waiting_in_aexit = True
            await wait_task_rescheduled(aborted)
        else:
            # Nothing to wait for, so execute a schedule point, but don't
            # allow us to be cancelled, just like the other branch.  We
            # still need to catch and store non-Cancelled exceptions.
            try:
                await cancel_shielded_checkpoint()
            except BaseException as exc:
                self._add_exc(exc)

        popped = self._parent_task._child_nurseries.pop()
        assert popped is self
        if self._pending_excs:
            try:
                if not self._strict_exception_groups and len(self._pending_excs) == 1:
                    return self._pending_excs[0]
                exception = BaseExceptionGroup(
                    "Exceptions from Trio nursery",
                    self._pending_excs,
                )
                if not self._strict_exception_groups:
                    exception.add_note(NONSTRICT_EXCEPTIONGROUP_NOTE)
                return exception
            finally:
                # avoid a garbage cycle
                # (see test_locals_destroyed_promptly_on_cancel)
                del self._pending_excs
        return None

    def start_soon(
        self,
        async_fn: Callable[[Unpack[PosArgT]], Awaitable[object]],
        *args: Unpack[PosArgT],
        name: object = None,
    ) -> None:
        """Creates a child task, scheduling ``await async_fn(*args)``.

        If you want to run a function and immediately wait for its result,
        then you don't need a nursery; just use ``await async_fn(*args)``.
        If you want to wait for the task to initialize itself before
        continuing, see :meth:`start`, the other fundamental method for
        creating concurrent tasks in Trio.

        Note that this is *not* an async function and you don't use await
        when calling it. It sets up the new task, but then returns
        immediately, *before* the new task has a chance to do anything.
        New tasks may start running in any order, and at any checkpoint the
        scheduler chooses - at latest when the nursery is waiting to exit.

        It's possible to pass a nursery object into another task, which
        allows that task to start new child tasks in the first task's
        nursery.

        The child task inherits its parent nursery's cancel scopes.

        Args:
            async_fn: An async callable.
            args: Positional arguments for ``async_fn``. If you want
                  to pass keyword arguments, use
                  :func:`functools.partial`.
            name: The name for this task. Only used for
                  debugging/introspection
                  (e.g. ``repr(task_obj)``). If this isn't a string,
                  :meth:`start_soon` will try to make it one. A
                  common use case is if you're wrapping a function
                  before spawning a new task, you might pass the
                  original function as the ``name=`` to make
                  debugging easier.

        Raises:
            RuntimeError: If this nursery is no longer open
                          (i.e. its ``async with`` block has
                          exited).
        """
        GLOBAL_RUN_CONTEXT.runner.spawn_impl(async_fn, args, self, name)

    # Typing changes blocked by https://github.com/python/mypy/pull/17512
    async def start(  # type: ignore[explicit-any]
        self,
        async_fn: Callable[..., Awaitable[object]],
        *args: object,
        name: object = None,
    ) -> Any:
        r"""Creates and initializes a child task.

        Like :meth:`start_soon`, but blocks until the new task has
        finished initializing itself, and optionally returns some
        information from it.

        The ``async_fn`` must accept a ``task_status`` keyword argument,
        and it must make sure that it (or someone) eventually calls
        :meth:`task_status.started() <TaskStatus.started>`.

        The conventional way to define ``async_fn`` is like::

            async def async_fn(arg1, arg2, *, task_status=trio.TASK_STATUS_IGNORED):
                ...  # Caller is blocked waiting for this code to run
                task_status.started()
                ...  # This async code can be interleaved with the caller

        :attr:`trio.TASK_STATUS_IGNORED` is a special global object with
        a do-nothing ``started`` method. This way your function supports
        being called either like ``await nursery.start(async_fn, arg1,
        arg2)`` or directly like ``await async_fn(arg1, arg2)``, and
        either way it can call :meth:`task_status.started() <TaskStatus.started>`
        without worrying about which mode it's in. Defining your function like
        this will make it obvious to readers that it supports being used
        in both modes.

        Before the child calls :meth:`task_status.started() <TaskStatus.started>`,
        it's effectively run underneath the call to :meth:`start`: if it
        raises an exception then that exception is reported by
        :meth:`start`, and does *not* propagate out of the nursery. If
        :meth:`start` is cancelled, then the child task is also
        cancelled.

        When the child calls :meth:`task_status.started() <TaskStatus.started>`,
        it's moved out from underneath :meth:`start` and into the given nursery.

        If the child task passes a value to :meth:`task_status.started(value) <TaskStatus.started>`,
        then :meth:`start` returns this value. Otherwise, it returns ``None``.
        """
        if self._closed:
            raise RuntimeError("Nursery is closed to new arrivals")
        try:
            self._pending_starts += 1
            # wrap internal nursery in try-except to unroll any exceptiongroups
            # to avoid wrapping pre-started() exceptions in an extra ExceptionGroup.
            # See #2611.
            try:
                # set strict_exception_groups = True to make sure we always unwrap
                # *this* nursery's exceptiongroup
                async with open_nursery(strict_exception_groups=True) as old_nursery:
                    task_status: _TaskStatus[object | None] = _TaskStatus(
                        old_nursery,
                        self,
                    )
                    thunk = functools.partial(async_fn, task_status=task_status)
                    task = GLOBAL_RUN_CONTEXT.runner.spawn_impl(
                        thunk,
                        args,
                        old_nursery,
                        name,
                    )
                    task._eventual_parent_nursery = self
                    # Wait for either TaskStatus.started or an exception to
                    # cancel this nursery:
            except BaseExceptionGroup as exc:
                if len(exc.exceptions) == 1:
                    raise exc.exceptions[0] from None
                raise TrioInternalError(
                    "Internal nursery should not have multiple tasks. This can be "
                    'caused by the user managing to access the "old" nursery in '
                    "`task_status` and spawning tasks in it.",
                ) from exc

            # If we get here, then the child either got reparented or exited
            # normally. The complicated logic is all in TaskStatus.started().
            # (Any exceptions propagate directly out of the above.)
            if task_status._value is _NoStatus:
                raise RuntimeError("child exited without calling task_status.started()")
            return task_status._value
        finally:
            self._pending_starts -= 1
            self._check_nursery_closed()

    def __del__(self) -> None:
        assert not self._children


################################################################
# Task and friends
################################################################


@final
@attrs.define(eq=False, repr=False)
class Task(metaclass=NoPublicConstructor):  # type: ignore[explicit-any]
    _parent_nursery: Nursery | None
    coro: types.CoroutineType[Any, Outcome[object], Any]  # type: ignore[explicit-any]
    _runner: Runner
    name: str
    context: contextvars.Context
    _counter: int = attrs.field(init=False, factory=itertools.count().__next__)
    _ki_protected: bool

    # Invariant:
    # - for unscheduled tasks, _next_send_fn and _next_send are both None
    # - for scheduled tasks, _next_send_fn(_next_send) resumes the task;
    #   usually _next_send_fn is self.coro.send and _next_send is an
    #   Outcome. When recovering from a foreign await, _next_send_fn is
    #   self.coro.throw and _next_send is an exception. _next_send_fn
    #   will effectively be at the top of every task's call stack, so
    #   it should be written in C if you don't want to pollute Trio
    #   tracebacks with extraneous frames.
    # - for scheduled tasks, custom_sleep_data is None
    # Tasks start out unscheduled.
    _next_send_fn: Callable[[Any], object] | None = None  # type: ignore[explicit-any]
    _next_send: Outcome[Any] | BaseException | None = None  # type: ignore[explicit-any]
    _abort_func: Callable[[_core.RaiseCancelT], Abort] | None = None
    custom_sleep_data: Any = None  # type: ignore[explicit-any]

    # For introspection and nursery.start()
    _child_nurseries: list[Nursery] = attrs.Factory(list)
    _eventual_parent_nursery: Nursery | None = None

    # these are counts of how many cancel/schedule points this task has
    # executed, for assert{_no,}_checkpoints
    # XX maybe these should be exposed as part of a statistics() method?
    _cancel_points: int = 0
    _schedule_points: int = 0

    def __repr__(self) -> str:
        return f"<Task {self.name!r} at {id(self):#x}>"

    @property
    def parent_nursery(self) -> Nursery | None:
        """The nursery this task is inside (or None if this is the "init"
        task).

        Example use case: drawing a visualization of the task tree in a
        debugger.

        """
        return self._parent_nursery

    @property
    def eventual_parent_nursery(self) -> Nursery | None:
        """The nursery this task will be inside after it calls
        ``task_status.started()``.

        If this task has already called ``started()``, or if it was not
        spawned using `nursery.start() <trio.Nursery.start>`, then
        its `eventual_parent_nursery` is ``None``.

        """
        return self._eventual_parent_nursery

    @property
    def child_nurseries(self) -> list[Nursery]:
        """The nurseries this task contains.

        This is a list, with outer nurseries before inner nurseries.

        """
        return list(self._child_nurseries)

    def iter_await_frames(self) -> Iterator[tuple[types.FrameType, int]]:
        """Iterates recursively over the coroutine-like objects this
        task is waiting on, yielding the frame and line number at each
        frame.

        This is similar to `traceback.walk_stack` in a synchronous
        context. Note that `traceback.walk_stack` returns frames from
        the bottom of the call stack to the top, while this function
        starts from `Task.coro <trio.lowlevel.Task.coro>` and works it
        way down.

        Example usage: extracting a stack trace::

            import traceback

            def print_stack_for_task(task):
                ss = traceback.StackSummary.extract(task.iter_await_frames())
                print("".join(ss.format()))

        """
        # Ignore static typing as we're doing lots of dynamic introspection
        coro: Any = self.coro  # type: ignore[explicit-any]
        while coro is not None:
            if hasattr(coro, "cr_frame"):
                # A real coroutine
                yield coro.cr_frame, coro.cr_frame.f_lineno
                coro = coro.cr_await
            elif hasattr(coro, "gi_frame"):
                # A generator decorated with @types.coroutine
                yield coro.gi_frame, coro.gi_frame.f_lineno
                coro = coro.gi_yieldfrom
            elif coro.__class__.__name__ in [
                "async_generator_athrow",
                "async_generator_asend",
            ]:
                # cannot extract the generator directly, see https://github.com/python/cpython/issues/76991
                # we can however use the gc to look through the object
                for referent in gc.get_referents(coro):
                    if hasattr(referent, "ag_frame"):  # pragma: no branch
                        yield referent.ag_frame, referent.ag_frame.f_lineno
                        coro = referent.ag_await
                        break
                else:  # pragma: no cover
                    # either cpython changed or we are running on an alternative python implementation
                    return
            else:  # pragma: no cover
                return

    ################
    # Cancellation
    ################

    # The CancelStatus object that is currently active for this task.
    # Don't change this directly; instead, use _activate_cancel_status().
    # This can be None, but only in the init task.
    _cancel_status: CancelStatus = attrs.field(default=None, repr=False)

    def _activate_cancel_status(self, cancel_status: CancelStatus | None) -> None:
        if self._cancel_status is not None:
            self._cancel_status._tasks.remove(self)
        self._cancel_status = cancel_status  # type: ignore[assignment]
        if self._cancel_status is not None:
            self._cancel_status._tasks.add(self)
            if self._cancel_status.effectively_cancelled:
                self._attempt_delivery_of_any_pending_cancel()

    def _attempt_abort(self, raise_cancel: _core.RaiseCancelT) -> None:
        # Either the abort succeeds, in which case we will reschedule the
        # task, or else it fails, in which case it will worry about
        # rescheduling itself (hopefully eventually calling reraise to raise
        # the given exception, but not necessarily).

        # This is only called by the functions immediately below, which both check
        # `self.abort_func is not None`.
        assert self._abort_func is not None, "FATAL INTERNAL ERROR"

        success = self._abort_func(raise_cancel)
        if type(success) is not Abort:
            raise TrioInternalError("abort function must return Abort enum")
        # We only attempt to abort once per blocking call, regardless of
        # whether we succeeded or failed.
        self._abort_func = None
        if success is Abort.SUCCEEDED:
            self._runner.reschedule(self, capture(raise_cancel))

    def _attempt_delivery_of_any_pending_cancel(self) -> None:
        if self._abort_func is None:
            return
        if not self._cancel_status.effectively_cancelled:
            return

        def raise_cancel() -> NoReturn:
            raise Cancelled._create()

        self._attempt_abort(raise_cancel)

    def _attempt_delivery_of_pending_ki(self) -> None:
        assert self._runner.ki_pending
        if self._abort_func is None:
            return

        def raise_cancel() -> NoReturn:
            self._runner.ki_pending = False
            raise KeyboardInterrupt

        self._attempt_abort(raise_cancel)


################################################################
# The central Runner object
################################################################


@attrs.frozen
class RunStatistics:
    """An object containing run-loop-level debugging information.

    Currently, the following fields are defined:

    * ``tasks_living`` (int): The number of tasks that have been spawned
      and not yet exited.
    * ``tasks_runnable`` (int): The number of tasks that are currently
      queued on the run queue (as opposed to blocked waiting for something
      to happen).
    * ``seconds_to_next_deadline`` (float): The time until the next
      pending cancel scope deadline. May be negative if the deadline has
      expired but we haven't yet processed cancellations. May be
      :data:`~math.inf` if there are no pending deadlines.
    * ``run_sync_soon_queue_size`` (int): The number of
      unprocessed callbacks queued via
      :meth:`trio.lowlevel.TrioToken.run_sync_soon`.
    * ``io_statistics`` (object): Some statistics from Trio's I/O
      backend. This always has an attribute ``backend`` which is a string
      naming which operating-system-specific I/O backend is in use; the
      other attributes vary between backends.
    """

    tasks_living: int
    tasks_runnable: int
    seconds_to_next_deadline: float
    io_statistics: IOStatistics
    run_sync_soon_queue_size: int


# This holds all the state that gets trampolined back and forth between
# callbacks when we're running in guest mode.
#
# It has to be a separate object from Runner, and Runner *cannot* hold
# references to it (directly or indirectly)!
#
# The idea is that we want a chance to detect if our host loop quits and stops
# driving us forward. We detect that by unrolled_run_gen being garbage
# collected, and hitting its 'except GeneratorExit:' block. So this only
# happens if unrolled_run_gen is GCed.
#
# The Runner state is referenced from the global GLOBAL_RUN_CONTEXT. The only
# way it gets *un*referenced is by unrolled_run_gen completing, e.g. by being
# GCed. But if Runner has a direct or indirect reference to it, and the host
# loop has abandoned it, then this will never happen!
#
# So this object can reference Runner, but Runner can't reference it. The only
# references to it are the "in flight" callback chain on the host loop /
# worker thread.


@attrs.define(eq=False)
class GuestState:  # type: ignore[explicit-any]
    runner: Runner
    run_sync_soon_threadsafe: Callable[[Callable[[], object]], object]
    run_sync_soon_not_threadsafe: Callable[[Callable[[], object]], object]
    done_callback: Callable[[Outcome[Any]], object]  # type: ignore[explicit-any]
    unrolled_run_gen: Generator[float, EventResult, None]
    unrolled_run_next_send: Outcome[Any] = attrs.Factory(lambda: Value(None))  # type: ignore[explicit-any]

    def guest_tick(self) -> None:
        prev_library, sniffio_library.name = sniffio_library.name, "trio"
        try:
            timeout = self.unrolled_run_next_send.send(self.unrolled_run_gen)
        except StopIteration:
            assert self.runner.main_task_outcome is not None
            self.done_callback(self.runner.main_task_outcome)
            return
        except TrioInternalError as exc:
            self.done_callback(Error(exc))
            return
        finally:
            sniffio_library.name = prev_library

        # Optimization: try to skip going into the thread if we can avoid it
        events_outcome: Value[EventResult] | Error = capture(
            self.runner.io_manager.get_events,
            0,
        )
        if timeout <= 0 or isinstance(events_outcome, Error) or events_outcome.value:
            # No need to go into the thread
            self.unrolled_run_next_send = events_outcome
            self.runner.guest_tick_scheduled = True
            self.run_sync_soon_not_threadsafe(self.guest_tick)
        else:
            # Need to go into the thread and call get_events() there
            self.runner.guest_tick_scheduled = False

            def get_events() -> EventResult:
                return self.runner.io_manager.get_events(timeout)

            def deliver(events_outcome: Outcome[EventResult]) -> None:
                def in_main_thread() -> None:
                    self.unrolled_run_next_send = events_outcome
                    self.runner.guest_tick_scheduled = True
                    self.guest_tick()

                self.run_sync_soon_threadsafe(in_main_thread)

            start_thread_soon(get_events, deliver)


@attrs.define(eq=False)
class Runner:  # type: ignore[explicit-any]
    clock: Clock
    instruments: Instruments
    io_manager: TheIOManager
    ki_manager: KIManager
    strict_exception_groups: bool

    # Run-local values, see _local.py
    _locals: dict[_core.RunVar[Any], object] = attrs.Factory(dict)  # type: ignore[explicit-any]

    runq: deque[Task] = attrs.Factory(deque)
    tasks: set[Task] = attrs.Factory(set)

    deadlines: Deadlines = attrs.Factory(Deadlines)

    init_task: Task | None = None
    system_nursery: Nursery | None = None
    system_context: contextvars.Context = attrs.field(kw_only=True)
    main_task: Task | None = None
    main_task_outcome: Outcome[object] | None = None

    entry_queue: EntryQueue = attrs.Factory(EntryQueue)
    trio_token: TrioToken | None = None
    asyncgens: AsyncGenerators = attrs.Factory(AsyncGenerators)

    # If everything goes idle for this long, we call clock._autojump()
    clock_autojump_threshold: float = inf

    # Guest mode stuff
    is_guest: bool = False
    guest_tick_scheduled: bool = False

    def force_guest_tick_asap(self) -> None:
        if self.guest_tick_scheduled:
            return
        self.guest_tick_scheduled = True
        self.io_manager.force_wakeup()

    def close(self) -> None:
        self.io_manager.close()
        self.entry_queue.close()
        self.asyncgens.close()
        if "after_run" in self.instruments:
            self.instruments.call("after_run")
        # This is where KI protection gets disabled, so we do it last
        self.ki_manager.close()

    @_public
    def current_statistics(self) -> RunStatistics:
        """Returns ``RunStatistics``, which contains run-loop-level debugging information.

        Currently, the following fields are defined:

        * ``tasks_living`` (int): The number of tasks that have been spawned
          and not yet exited.
        * ``tasks_runnable`` (int): The number of tasks that are currently
          queued on the run queue (as opposed to blocked waiting for something
          to happen).
        * ``seconds_to_next_deadline`` (float): The time until the next
          pending cancel scope deadline. May be negative if the deadline has
          expired but we haven't yet processed cancellations. May be
          :data:`~math.inf` if there are no pending deadlines.
        * ``run_sync_soon_queue_size`` (int): The number of
          unprocessed callbacks queued via
          :meth:`trio.lowlevel.TrioToken.run_sync_soon`.
        * ``io_statistics`` (object): Some statistics from Trio's I/O
          backend. This always has an attribute ``backend`` which is a string
          naming which operating-system-specific I/O backend is in use; the
          other attributes vary between backends.

        """
        seconds_to_next_deadline = self.deadlines.next_deadline() - self.current_time()
        return RunStatistics(
            tasks_living=len(self.tasks),
            tasks_runnable=len(self.runq),
            seconds_to_next_deadline=seconds_to_next_deadline,
            io_statistics=self.io_manager.statistics(),
            run_sync_soon_queue_size=self.entry_queue.size(),
        )

    @_public
    def current_time(self) -> float:
        """Returns the current time according to Trio's internal clock.

        Returns:
            float: The current time.

        Raises:
            RuntimeError: if not inside a call to :func:`trio.run`.

        """
        return self.clock.current_time()

    @_public
    def current_clock(self) -> Clock:
        """Returns the current :class:`~trio.abc.Clock`."""
        return self.clock

    @_public
    def current_root_task(self) -> Task | None:
        """Returns the current root :class:`Task`.

        This is the task that is the ultimate parent of all other tasks.

        """
        return self.init_task

    ################
    # Core task handling primitives
    ################

    @_public
    def reschedule(self, task: Task, next_send: Outcome[object] = _NO_SEND) -> None:
        """Reschedule the given task with the given
        :class:`outcome.Outcome`.

        See :func:`wait_task_rescheduled` for the gory details.

        There must be exactly one call to :func:`reschedule` for every call to
        :func:`wait_task_rescheduled`. (And when counting, keep in mind that
        returning :data:`Abort.SUCCEEDED` from an abort callback is equivalent
        to calling :func:`reschedule` once.)

        Args:
          task (trio.lowlevel.Task): the task to be rescheduled. Must be blocked
              in a call to :func:`wait_task_rescheduled`.
          next_send (outcome.Outcome): the value (or error) to return (or
              raise) from :func:`wait_task_rescheduled`.

        """
        if next_send is _NO_SEND:
            next_send = Value(None)

        assert task._runner is self
        assert task._next_send_fn is None
        task._next_send_fn = task.coro.send
        task._next_send = next_send
        task._abort_func = None
        task.custom_sleep_data = None
        if not self.runq and self.is_guest:
            self.force_guest_tick_asap()
        self.runq.append(task)
        if "task_scheduled" in self.instruments:
            self.instruments.call("task_scheduled", task)

    def spawn_impl(
        self,
        async_fn: Callable[[Unpack[PosArgT]], Awaitable[object]],
        args: tuple[Unpack[PosArgT]],
        nursery: Nursery | None,
        name: object,
        *,
        system_task: bool = False,
        context: contextvars.Context | None = None,
    ) -> Task:
        ######
        # Make sure the nursery is in working order
        ######

        # This sorta feels like it should be a method on nursery, except it
        # has to handle nursery=None for init. And it touches the internals of
        # all kinds of objects.
        if nursery is not None and nursery._closed:
            raise RuntimeError("Nursery is closed to new arrivals")
        if nursery is None:
            assert self.init_task is None

        ######
        # Propagate contextvars
        ######
        if context is None:
            context = self.system_context.copy() if system_task else copy_context()

        ######
        # Call the function and get the coroutine object, while giving helpful
        # errors for common mistakes.
        ######
        # TypeVarTuple passed into ParamSpec function confuses Mypy.
        coro = context.run(coroutine_or_error, async_fn, *args)  # type: ignore[arg-type]

        if name is None:
            name = async_fn
        if isinstance(name, functools.partial):
            name = name.func
        if not isinstance(name, str):
            try:
                name = f"{name.__module__}.{name.__qualname__}"  # type: ignore[attr-defined]
            except AttributeError:
                name = repr(name)

        # very old Cython versions (<0.29.24) has the attribute, but with a value of None
        if getattr(coro, "cr_frame", None) is None:
            # This async function is implemented in C or Cython
            async def python_wrapper(orig_coro: Awaitable[RetT]) -> RetT:
                return await orig_coro

            coro = python_wrapper(coro)
        assert coro.cr_frame is not None, "Coroutine frame should exist"  # type: ignore[attr-defined]

        ######
        # Set up the Task object
        ######
        task = Task._create(
            coro=coro,
            parent_nursery=nursery,
            runner=self,
            name=name,
            context=context,
            ki_protected=system_task,
        )

        self.tasks.add(task)
        if nursery is not None:
            nursery._children.add(task)
            task._activate_cancel_status(nursery._cancel_status)

        if "task_spawned" in self.instruments:
            self.instruments.call("task_spawned", task)
        # Special case: normally next_send should be an Outcome, but for the
        # very first send we have to send a literal unboxed None.
        self.reschedule(task, None)  # type: ignore[arg-type]
        return task

    def task_exited(self, task: Task, outcome: Outcome[object]) -> None:
        # break parking lots associated with the exiting task
        if task in GLOBAL_PARKING_LOT_BREAKER:
            for lot in GLOBAL_PARKING_LOT_BREAKER[task]:
                lot.break_lot(task)
            del GLOBAL_PARKING_LOT_BREAKER[task]

        if (
            task._cancel_status is not None
            and task._cancel_status.abandoned_by_misnesting
            and task._cancel_status.parent is None
        ):
            # The cancel scope surrounding this task's nursery was closed
            # before the task exited. Force the task to exit with an error,
            # since the error might not have been caught elsewhere. See the
            # comments in CancelStatus.close().
            try:
                # Raise this, rather than just constructing it, to get a
                # traceback frame included
                raise RuntimeError(
                    "Cancel scope stack corrupted: cancel scope surrounding "
                    f"{task!r} was closed before the task exited\n{MISNESTING_ADVICE}",
                )
            except RuntimeError as new_exc:
                if isinstance(outcome, Error):
                    new_exc.__context__ = outcome.error
                outcome = Error(new_exc)

        task._activate_cancel_status(None)
        self.tasks.remove(task)
        if task is self.init_task:
            # If the init task crashed, then something is very wrong and we
            # let the error propagate. (It'll eventually be wrapped in a
            # TrioInternalError.)
            outcome.unwrap()
            # the init task should be the last task to exit. If not, then
            # something is very wrong.
            if self.tasks:  # pragma: no cover
                raise TrioInternalError
        else:
            if task is self.main_task:
                self.main_task_outcome = outcome
                outcome = Value(None)
            assert task._parent_nursery is not None, task
            task._parent_nursery._child_finished(task, outcome)

        if "task_exited" in self.instruments:
            self.instruments.call("task_exited", task)

    ################
    # System tasks and init
    ################

    @_public
    def spawn_system_task(
        self,
        async_fn: Callable[[Unpack[PosArgT]], Awaitable[object]],
        *args: Unpack[PosArgT],
        name: object = None,
        context: contextvars.Context | None = None,
    ) -> Task:
        """Spawn a "system" task.

        System tasks have a few differences from regular tasks:

        * They don't need an explicit nursery; instead they go into the
          internal "system nursery".

        * If a system task raises an exception, then it's converted into a
          :exc:`~trio.TrioInternalError` and *all* tasks are cancelled. If you
          write a system task, you should be careful to make sure it doesn't
          crash.

        * System tasks are automatically cancelled when the main task exits.

        * By default, system tasks have :exc:`KeyboardInterrupt` protection
          *enabled*. If you want your task to be interruptible by control-C,
          then you need to use :func:`disable_ki_protection` explicitly (and
          come up with some plan for what to do with a
          :exc:`KeyboardInterrupt`, given that system tasks aren't allowed to
          raise exceptions).

        * System tasks do not inherit context variables from their creator.

        Towards the end of a call to :meth:`trio.run`, after the main
        task and all system tasks have exited, the system nursery
        becomes closed. At this point, new calls to
        :func:`spawn_system_task` will raise ``RuntimeError("Nursery
        is closed to new arrivals")`` instead of creating a system
        task. It's possible to encounter this state either in
        a ``finally`` block in an async generator, or in a callback
        passed to :meth:`TrioToken.run_sync_soon` at the right moment.

        Args:
          async_fn: An async callable.
          args: Positional arguments for ``async_fn``. If you want to pass
              keyword arguments, use :func:`functools.partial`.
          name: The name for this task. Only used for debugging/introspection
              (e.g. ``repr(task_obj)``). If this isn't a string,
              :func:`spawn_system_task` will try to make it one. A common use
              case is if you're wrapping a function before spawning a new
              task, you might pass the original function as the ``name=`` to
              make debugging easier.
          context: An optional ``contextvars.Context`` object with context variables
              to use for this task. You would normally get a copy of the current
              context with ``context = contextvars.copy_context()`` and then you would
              pass that ``context`` object here.

        Returns:
          Task: the newly spawned task

        """
        return self.spawn_impl(
            async_fn,
            args,
            self.system_nursery,
            name,
            system_task=True,
            context=context,
        )

    async def init(
        self,
        async_fn: Callable[[Unpack[PosArgT]], Awaitable[object]],
        args: tuple[Unpack[PosArgT]],
    ) -> None:
        # run_sync_soon task runs here:
        async with open_nursery() as run_sync_soon_nursery:
            # All other system tasks run here:
            async with open_nursery() as self.system_nursery:
                # Only the main task runs here:
                async with open_nursery() as main_task_nursery:
                    try:
                        self.main_task = self.spawn_impl(
                            async_fn,
                            args,
                            main_task_nursery,
                            None,
                        )
                    except BaseException as exc:
                        self.main_task_outcome = Error(exc)
                        return
                    self.spawn_impl(
                        self.entry_queue.task,
                        (),
                        run_sync_soon_nursery,
                        "<TrioToken.run_sync_soon task>",
                        system_task=True,
                    )

                # Main task is done; start shutting down system tasks
                self.system_nursery.cancel_scope.cancel()

            # System nursery is closed; finalize remaining async generators
            await self.asyncgens.finalize_remaining(self)

            # There are no more asyncgens, which means no more user-provided
            # code except possibly run_sync_soon callbacks. It's finally safe
            # to stop the run_sync_soon task and exit run().
            run_sync_soon_nursery.cancel_scope.cancel()

    ################
    # Outside context problems
    ################

    @_public
    def current_trio_token(self) -> TrioToken:
        """Retrieve the :class:`TrioToken` for the current call to
        :func:`trio.run`.

        """
        if self.trio_token is None:
            self.trio_token = TrioToken._create(self.entry_queue)
        return self.trio_token

    ################
    # KI handling
    ################

    ki_pending: bool = False

    # deliver_ki is broke. Maybe move all the actual logic and state into
    # RunToken, and we'll only have one instance per runner? But then we can't
    # have a public constructor. Eh, but current_run_token() returning a
    # unique object per run feels pretty nice. Maybe let's just go for it. And
    # keep the class public so people can isinstance() it if they want.

    # This gets called from signal context
    def deliver_ki(self) -> None:
        self.ki_pending = True
        with suppress(RunFinishedError):
            self.entry_queue.run_sync_soon(self._deliver_ki_cb)

    def _deliver_ki_cb(self) -> None:
        if not self.ki_pending:
            return
        # Can't happen because main_task and run_sync_soon_task are created at
        # the same time -- so even if KI arrives before main_task is created,
        # we won't get here until afterwards.
        assert self.main_task is not None
        if self.main_task_outcome is not None:
            # We're already in the process of exiting -- leave ki_pending set
            # and we'll check it again on our way out of run().
            return
        self.main_task._attempt_delivery_of_pending_ki()

    ################
    # Quiescing
    ################

    # sortedcontainers doesn't have types, and is reportedly very hard to type:
    # https://github.com/grantjenks/python-sortedcontainers/issues/68
    waiting_for_idle: Any = attrs.Factory(SortedDict)  # type: ignore[explicit-any]

    @_public
    async def wait_all_tasks_blocked(self, cushion: float = 0.0) -> None:
        """Block until there are no runnable tasks.

        This is useful in testing code when you want to give other tasks a
        chance to "settle down". The calling task is blocked, and doesn't wake
        up until all other tasks are also blocked for at least ``cushion``
        seconds. (Setting a non-zero ``cushion`` is intended to handle cases
        like two tasks talking to each other over a local socket, where we
        want to ignore the potential brief moment between a send and receive
        when all tasks are blocked.)

        Note that ``cushion`` is measured in *real* time, not the Trio clock
        time.

        If there are multiple tasks blocked in :func:`wait_all_tasks_blocked`,
        then the one with the shortest ``cushion`` is the one woken (and
        this task becoming unblocked resets the timers for the remaining
        tasks). If there are multiple tasks that have exactly the same
        ``cushion``, then all are woken.

        You should also consider :class:`trio.testing.Sequencer`, which
        provides a more explicit way to control execution ordering within a
        test, and will often produce more readable tests.

        Example:
          Here's an example of one way to test that Trio's locks are fair: we
          take the lock in the parent, start a child, wait for the child to be
          blocked waiting for the lock (!), and then check that we can't
          release and immediately re-acquire the lock::

             async def lock_taker(lock):
                 await lock.acquire()
                 lock.release()

             async def test_lock_fairness():
                 lock = trio.Lock()
                 await lock.acquire()
                 async with trio.open_nursery() as nursery:
                     nursery.start_soon(lock_taker, lock)
                     # child hasn't run yet, we have the lock
                     assert lock.locked()
                     assert lock._owner is trio.lowlevel.current_task()
                     await trio.testing.wait_all_tasks_blocked()
                     # now the child has run and is blocked on lock.acquire(), we
                     # still have the lock
                     assert lock.locked()
                     assert lock._owner is trio.lowlevel.current_task()
                     lock.release()
                     try:
                         # The child has a prior claim, so we can't have it
                         lock.acquire_nowait()
                     except trio.WouldBlock:
                         assert lock._owner is not trio.lowlevel.current_task()
                         print("PASS")
                     else:
                         print("FAIL")

        """
        task = current_task()
        key = (cushion, id(task))
        self.waiting_for_idle[key] = task

        def abort(_: _core.RaiseCancelT) -> Abort:
            del self.waiting_for_idle[key]
            return Abort.SUCCEEDED

        await wait_task_rescheduled(abort)


################################################################
# run
################################################################
#
# Trio's core task scheduler and coroutine runner is in 'unrolled_run'. It's
# called that because it has an unusual feature: it's actually a generator.
# Whenever it needs to fetch IO events from the OS, it yields, and waits for
# its caller to send the IO events back in. So the loop is "unrolled" into a
# sequence of generator send() calls.
#
# The reason for this unusual design is to support two different modes of
# operation, where the IO is handled differently.
#
# In normal mode using trio.run, the scheduler and IO run in the same thread:
#
# Main thread:
#
# +---------------------------+
# | Run tasks                 |
# | (unrolled_run)            |
# +---------------------------+
# | Block waiting for I/O     |
# | (io_manager.get_events)   |
# +---------------------------+
# | Run tasks                 |
# | (unrolled_run)            |
# +---------------------------+
# | Block waiting for I/O     |
# | (io_manager.get_events)   |
# +---------------------------+
# :
#
#
# In guest mode using trio.lowlevel.start_guest_run, the scheduler runs on the
# main thread as a host loop callback, but blocking for IO gets pushed into a
# worker thread:
#
# Main thread executing host loop:           Trio I/O thread:
#
# +---------------------------+
# | Run Trio tasks            |
# | (unrolled_run)            |
# +---------------------------+ --------------+
#                                             v
# +---------------------------+              +----------------------------+
# | Host loop does whatever   |              | Block waiting for Trio I/O |
# | it wants                  |              | (io_manager.get_events)    |
# +---------------------------+              +----------------------------+
#                                             |
# +---------------------------+ <-------------+
# | Run Trio tasks            |
# | (unrolled_run)            |
# +---------------------------+ --------------+
#                                             v
# +---------------------------+              +----------------------------+
# | Host loop does whatever   |              | Block waiting for Trio I/O |
# | it wants                  |              | (io_manager.get_events)    |
# +---------------------------+              +----------------------------+
# :                                            :
#
# Most of Trio's internals don't need to care about this difference. The main
# complication it creates is that in guest mode, we might need to wake up not
# just due to OS-reported IO events, but also because of code running on the
# host loop calling reschedule() or changing task deadlines. Search for
# 'is_guest' to see the special cases we need to handle this.


def setup_runner(
    clock: Clock | None,
    instruments: Sequence[Instrument],
    restrict_keyboard_interrupt_to_checkpoints: bool,
    strict_exception_groups: bool,
) -> Runner:
    """Create a Runner object and install it as the GLOBAL_RUN_CONTEXT."""
    # It wouldn't be *hard* to support nested calls to run(), but I can't
    # think of a single good reason for it, so let's be conservative for
    # now:
    if in_trio_run():
        raise RuntimeError("Attempted to call run() from inside a run()")

    if clock is None:
        clock = SystemClock()
    instrument_group = Instruments(instruments)
    io_manager = TheIOManager()
    system_context = copy_context()
    ki_manager = KIManager()

    runner = Runner(
        clock=clock,
        instruments=instrument_group,
        io_manager=io_manager,
        system_context=system_context,
        ki_manager=ki_manager,
        strict_exception_groups=strict_exception_groups,
    )
    runner.asyncgens.install_hooks(runner)

    # This is where KI protection gets enabled, so we want to do it early - in
    # particular before we start modifying global state like GLOBAL_RUN_CONTEXT
    ki_manager.install(runner.deliver_ki, restrict_keyboard_interrupt_to_checkpoints)

    GLOBAL_RUN_CONTEXT.runner = runner
    return runner


def run(
    async_fn: Callable[[Unpack[PosArgT]], Awaitable[RetT]],
    *args: Unpack[PosArgT],
    clock: Clock | None = None,
    instruments: Sequence[Instrument] = (),
    restrict_keyboard_interrupt_to_checkpoints: bool = False,
    strict_exception_groups: bool = True,
) -> RetT:
    """Run a Trio-flavored async function, and return the result.

    Calling::

       run(async_fn, *args)

    is the equivalent of::

       await async_fn(*args)

    except that :func:`run` can (and must) be called from a synchronous
    context.

    This is Trio's main entry point. Almost every other function in Trio
    requires that you be inside a call to :func:`run`.

    Args:
      async_fn: An async function.

      args: Positional arguments to be passed to *async_fn*. If you need to
          pass keyword arguments, then use :func:`functools.partial`.

      clock: ``None`` to use the default system-specific monotonic clock;
          otherwise, an object implementing the :class:`trio.abc.Clock`
          interface, like (for example) a :class:`trio.testing.MockClock`
          instance.

      instruments (list of :class:`trio.abc.Instrument` objects): Any
          instrumentation you want to apply to this run. This can also be
          modified during the run; see :ref:`instrumentation`.

      restrict_keyboard_interrupt_to_checkpoints (bool): What happens if the
          user hits control-C while :func:`run` is running? If this argument
          is False (the default), then you get the standard Python behavior: a
          :exc:`KeyboardInterrupt` exception will immediately interrupt
          whatever task is running (or if no task is running, then Trio will
          wake up a task to be interrupted). Alternatively, if you set this
          argument to True, then :exc:`KeyboardInterrupt` delivery will be
          delayed: it will be *only* be raised at :ref:`checkpoints
          <checkpoints>`, like a :exc:`Cancelled` exception.

          The default behavior is nice because it means that even if you
          accidentally write an infinite loop that never executes any
          checkpoints, then you can still break out of it using control-C.
          The alternative behavior is nice if you're paranoid about a
          :exc:`KeyboardInterrupt` at just the wrong place leaving your
          program in an inconsistent state, because it means that you only
          have to worry about :exc:`KeyboardInterrupt` at the exact same
          places where you already have to worry about :exc:`Cancelled`.

          This setting has no effect if your program has registered a custom
          SIGINT handler, or if :func:`run` is called from anywhere but the
          main thread (this is a Python limitation), or if you use
          :func:`open_signal_receiver` to catch SIGINT.

      strict_exception_groups (bool): Unless set to False, nurseries will always wrap
          even a single raised exception in an exception group. This can be overridden
          on the level of individual nurseries. Setting it to False will be deprecated
          and ultimately removed in a future version of Trio.

    Returns:
      Whatever ``async_fn`` returns.

    Raises:
      TrioInternalError: if an unexpected error is encountered inside Trio's
          internal machinery. This is a bug and you should `let us know
          <https://github.com/python-trio/trio/issues>`__.

      Anything else: if ``async_fn`` raises an exception, then :func:`run`
          propagates it.

    """
    if strict_exception_groups is not None and not strict_exception_groups:
        warn_deprecated(
            "trio.run(..., strict_exception_groups=False)",
            version="0.25.0",
            issue=2929,
            instead=(
                "the default value of True and rewrite exception handlers to handle ExceptionGroups. "
                "See https://trio.readthedocs.io/en/stable/reference-core.html#designing-for-multiple-errors"
            ),
            use_triodeprecationwarning=True,
        )

    __tracebackhide__ = True

    runner = setup_runner(
        clock,
        instruments,
        restrict_keyboard_interrupt_to_checkpoints,
        strict_exception_groups,
    )

    prev_library, sniffio_library.name = sniffio_library.name, "trio"
    try:
        gen = unrolled_run(runner, async_fn, args)
        # Need to send None in the first time.
        next_send: EventResult = None  # type: ignore[assignment]
        while True:
            try:
                timeout = gen.send(next_send)
            except StopIteration:
                break
            next_send = runner.io_manager.get_events(timeout)
    finally:
        sniffio_library.name = prev_library
    # Inlined copy of runner.main_task_outcome.unwrap() to avoid
    # cluttering every single Trio traceback with an extra frame.
    if isinstance(runner.main_task_outcome, Value):
        return cast("RetT", runner.main_task_outcome.value)
    elif isinstance(runner.main_task_outcome, Error):
        raise runner.main_task_outcome.error
    else:  # pragma: no cover
        raise AssertionError(runner.main_task_outcome)


def start_guest_run(  # type: ignore[explicit-any]
    async_fn: Callable[..., Awaitable[RetT]],
    *args: object,
    run_sync_soon_threadsafe: Callable[[Callable[[], object]], object],
    done_callback: Callable[[outcome.Outcome[RetT]], object],
    run_sync_soon_not_threadsafe: (
        Callable[[Callable[[], object]], object] | None
    ) = None,
    host_uses_signal_set_wakeup_fd: bool = False,
    clock: Clock | None = None,
    instruments: Sequence[Instrument] = (),
    restrict_keyboard_interrupt_to_checkpoints: bool = False,
    strict_exception_groups: bool = True,
) -> None:
    """Start a "guest" run of Trio on top of some other "host" event loop.

    Each host loop can only have one guest run at a time.

    You should always let the Trio run finish before stopping the host loop;
    if not, it may leave Trio's internal data structures in an inconsistent
    state. You might be able to get away with it if you immediately exit the
    program, but it's safest not to go there in the first place.

    Generally, the best way to do this is wrap this in a function that starts
    the host loop and then immediately starts the guest run, and then shuts
    down the host when the guest run completes.

    Once :func:`start_guest_run` returns successfully, the guest run
    has been set up enough that you can invoke sync-colored Trio
    functions such as :func:`~trio.current_time`, :func:`spawn_system_task`,
    and :func:`current_trio_token`. If a `~trio.TrioInternalError` occurs
    during this early setup of the guest run, it will be raised out of
    :func:`start_guest_run`.  All other errors, including all errors
    raised by the *async_fn*, will be delivered to your
    *done_callback* at some point after :func:`start_guest_run` returns
    successfully.

    Args:

      run_sync_soon_threadsafe: An arbitrary callable, which will be passed a
         function as its sole argument::

            def my_run_sync_soon_threadsafe(fn):
                ...

         This callable should schedule ``fn()`` to be run by the host on its
         next pass through its loop. **Must support being called from
         arbitrary threads.**

      done_callback: An arbitrary callable::

            def my_done_callback(run_outcome):
                ...

         When the Trio run has finished, Trio will invoke this callback to let
         you know. The argument is an `outcome.Outcome`, reporting what would
         have been returned or raised by `trio.run`. This function can do
         anything you want, but commonly you'll want it to shut down the
         host loop, unwrap the outcome, etc.

      run_sync_soon_not_threadsafe: Like ``run_sync_soon_threadsafe``, but
         will only be called from inside the host loop's main thread.
         Optional, but if your host loop allows you to implement this more
         efficiently than ``run_sync_soon_threadsafe`` then passing it will
         make things a bit faster.

      host_uses_signal_set_wakeup_fd (bool): Pass `True` if your host loop
         uses `signal.set_wakeup_fd`, and `False` otherwise. For more details,
         see :ref:`guest-run-implementation`.

    For the meaning of other arguments, see `trio.run`.

    """
    if strict_exception_groups is not None and not strict_exception_groups:
        warn_deprecated(
            "trio.start_guest_run(..., strict_exception_groups=False)",
            version="0.25.0",
            issue=2929,
            instead=(
                "the default value of True and rewrite exception handlers to handle ExceptionGroups. "
                "See https://trio.readthedocs.io/en/stable/reference-core.html#designing-for-multiple-errors"
            ),
            use_triodeprecationwarning=True,
        )

    runner = setup_runner(
        clock,
        instruments,
        restrict_keyboard_interrupt_to_checkpoints,
        strict_exception_groups,
    )
    runner.is_guest = True
    runner.guest_tick_scheduled = True

    if run_sync_soon_not_threadsafe is None:
        run_sync_soon_not_threadsafe = run_sync_soon_threadsafe

    guest_state = GuestState(
        runner=runner,
        run_sync_soon_threadsafe=run_sync_soon_threadsafe,
        run_sync_soon_not_threadsafe=run_sync_soon_not_threadsafe,
        done_callback=done_callback,
        unrolled_run_gen=unrolled_run(
            runner,
            async_fn,
            args,
            host_uses_signal_set_wakeup_fd=host_uses_signal_set_wakeup_fd,
        ),
    )

    # Run a few ticks of the guest run synchronously, so that by the
    # time we return, the system nursery exists and callers can use
    # spawn_system_task. We don't actually run any user code during
    # this time, so it shouldn't be possible to get an exception here,
    # except for a TrioInternalError.
    next_send = cast(
        "EventResult",
        None,
    )  # First iteration must be `None`, every iteration after that is EventResult
    for _tick in range(5):  # expected need is 2 iterations + leave some wiggle room
        if runner.system_nursery is not None:
            # We're initialized enough to switch to async guest ticks
            break
        try:
            timeout = guest_state.unrolled_run_gen.send(next_send)
        except StopIteration:  # pragma: no cover
            raise TrioInternalError(
                "Guest runner exited before system nursery was initialized",
            ) from None
        if timeout != 0:  # pragma: no cover
            guest_state.unrolled_run_gen.throw(
                TrioInternalError(
                    "Guest runner blocked before system nursery was initialized",
                ),
            )
        # next_send should be the return value of
        # IOManager.get_events() if no I/O was waiting, which is
        # platform-dependent. We don't actually check for I/O during
        # this init phase because no one should be expecting any yet.
        if sys.platform == "win32":
            next_send = 0
        else:
            next_send = []
    else:  # pragma: no cover
        guest_state.unrolled_run_gen.throw(
            TrioInternalError(
                "Guest runner yielded too many times before "
                "system nursery was initialized",
            ),
        )

    guest_state.unrolled_run_next_send = Value(next_send)
    run_sync_soon_not_threadsafe(guest_state.guest_tick)


# 24 hours is arbitrary, but it avoids issues like people setting timeouts of
# 10**20 and then getting integer overflows in the underlying system calls.
_MAX_TIMEOUT: Final = 24 * 60 * 60


# Weird quirk: this is written as a generator in order to support "guest
# mode", where our core event loop gets unrolled into a series of callbacks on
# the host loop. If you're doing a regular trio.run then this gets run
# straight through.
@enable_ki_protection
def unrolled_run(
    runner: Runner,
    async_fn: Callable[[Unpack[PosArgT]], Awaitable[object]],
    args: tuple[Unpack[PosArgT]],
    host_uses_signal_set_wakeup_fd: bool = False,
) -> Generator[float, EventResult, None]:
    __tracebackhide__ = True

    try:
        if not host_uses_signal_set_wakeup_fd:
            runner.entry_queue.wakeup.wakeup_on_signals()

        if "before_run" in runner.instruments:
            runner.instruments.call("before_run")
        runner.clock.start_clock()
        runner.init_task = runner.spawn_impl(
            runner.init,
            (async_fn, args),
            None,
            "<init>",
            system_task=True,
        )

        # You know how people talk about "event loops"? This 'while' loop right
        # here is our event loop:
        while runner.tasks:
            if runner.runq:
                timeout: float = 0
            else:
                deadline = runner.deadlines.next_deadline()
                timeout = runner.clock.deadline_to_sleep_time(deadline)
            timeout = min(max(0, timeout), _MAX_TIMEOUT)

            idle_primed = None
            if runner.waiting_for_idle:
                cushion, _ = runner.waiting_for_idle.keys()[0]
                if cushion < timeout:
                    timeout = cushion
                    idle_primed = IdlePrimedTypes.WAITING_FOR_IDLE
            # We use 'elif' here because if there are tasks in
            # wait_all_tasks_blocked, then those tasks will wake up without
            # jumping the clock, so we don't need to autojump.
            elif runner.clock_autojump_threshold < timeout:
                timeout = runner.clock_autojump_threshold
                idle_primed = IdlePrimedTypes.AUTOJUMP_CLOCK

            if "before_io_wait" in runner.instruments:
                runner.instruments.call("before_io_wait", timeout)

            # Driver will call io_manager.get_events(timeout) and pass it back
            # in through the yield
            events = yield timeout
            runner.io_manager.process_events(events)

            if "after_io_wait" in runner.instruments:
                runner.instruments.call("after_io_wait", timeout)

            # Process cancellations due to deadline expiry
            now = runner.clock.current_time()
            if runner.deadlines.expire(now):
                idle_primed = None

            # idle_primed != None means: if the IO wait hit the timeout, and
            # still nothing is happening, then we should start waking up
            # wait_all_tasks_blocked tasks or autojump the clock. But there
            # are some subtleties in defining "nothing is happening".
            #
            # 'not runner.runq' means that no tasks are currently runnable.
            # 'not events' means that the last IO wait call hit its full
            # timeout. These are very similar, and if idle_primed != None and
            # we're running in regular mode then they always go together. But,
            # in *guest* mode, they can happen independently, even when
            # idle_primed=True:
            #
            # - runner.runq=empty and events=True: the host loop adjusted a
            #   deadline and that forced an IO wakeup before the timeout expired,
            #   even though no actual tasks were scheduled.
            #
            # - runner.runq=nonempty and events=False: the IO wait hit its
            #   timeout, but then some code in the host thread rescheduled a task
            #   before we got here.
            #
            # So we need to check both.
            if idle_primed is not None and not runner.runq and not events:
                if idle_primed is IdlePrimedTypes.WAITING_FOR_IDLE:
                    while runner.waiting_for_idle:
                        key, task = runner.waiting_for_idle.peekitem(0)
                        if key[0] == cushion:
                            del runner.waiting_for_idle[key]
                            runner.reschedule(task)
                        else:
                            break
                else:
                    assert idle_primed is IdlePrimedTypes.AUTOJUMP_CLOCK
                    assert isinstance(runner.clock, _core.MockClock)
                    runner.clock._autojump()

            # Process all runnable tasks, but only the ones that are already
            # runnable now. Anything that becomes runnable during this cycle
            # needs to wait until the next pass. This avoids various
            # starvation issues by ensuring that there's never an unbounded
            # delay between successive checks for I/O.
            #
            # Also, we randomize the order of each batch to avoid assumptions
            # about scheduling order sneaking in. In the long run, I suspect
            # we'll either (a) use strict FIFO ordering and document that for
            # predictability/determinism, or (b) implement a more
            # sophisticated scheduler (e.g. some variant of fair queueing),
            # for better behavior under load. For now, this is the worst of
            # both worlds - but it keeps our options open. (If we do decide to
            # go all in on deterministic scheduling, then there are other
            # things that will probably need to change too, like the deadlines
            # tie-breaker and the non-deterministic ordering of
            # task._notify_queues.)
            batch = list(runner.runq)
            runner.runq.clear()
            if _ALLOW_DETERMINISTIC_SCHEDULING:
                # We're running under Hypothesis, and pytest-trio has patched
                # this in to make the scheduler deterministic and avoid flaky
                # tests. It's not worth the (small) performance cost in normal
                # operation, since we'll shuffle the list and _r is only
                # seeded for tests.
                batch.sort(key=lambda t: t._counter)
                _r.shuffle(batch)
            else:
                # 50% chance of reversing the batch, this way each task
                # can appear before/after any other task.
                if _r.random() < 0.5:
                    batch.reverse()
            while batch:
                task = batch.pop()
                GLOBAL_RUN_CONTEXT.task = task

                if "before_task_step" in runner.instruments:
                    runner.instruments.call("before_task_step", task)

                next_send_fn = task._next_send_fn
                next_send = task._next_send
                task._next_send_fn = task._next_send = None
                final_outcome: Outcome[object] | None = None
                try:
                    # We used to unwrap the Outcome object here and send/throw
                    # its contents in directly, but it turns out that .throw()
                    # is buggy on CPython (all versions at time of writing):
                    #   https://bugs.python.org/issue29587
                    #   https://bugs.python.org/issue29590
                    #   https://bugs.python.org/issue40694
                    #   https://github.com/python/cpython/issues/108668
                    # So now we send in the Outcome object and unwrap it on the
                    # other side.
                    msg = task.context.run(next_send_fn, next_send)
                except StopIteration as stop_iteration:
                    final_outcome = Value(stop_iteration.value)
                except BaseException as task_exc:
                    # Store for later, removing uninteresting top frames: 1
                    # frame we always remove, because it's this function
                    # catching it, and then in addition we remove however many
                    # more Context.run adds.
                    tb = task_exc.__traceback__
                    for _ in range(1 + CONTEXT_RUN_TB_FRAMES):
                        if tb is not None:  # pragma: no branch
                            tb = tb.tb_next
                    final_outcome = Error(task_exc.with_traceback(tb))
                    # Remove local refs so that e.g. cancelled coroutine locals
                    # are not kept alive by this frame until another exception
                    # comes along.
                    del tb

                if final_outcome is not None:
                    # We can't call this directly inside the except: blocks
                    # above, because then the exceptions end up attaching
                    # themselves to other exceptions as __context__ in
                    # unwanted ways.
                    runner.task_exited(task, final_outcome)
                    # final_outcome may contain a traceback ref. It's not as
                    # crucial compared to the above, but this will allow more
                    # prompt release of resources in coroutine locals.
                    final_outcome = None
                else:
                    task._schedule_points += 1
                    if msg is CancelShieldedCheckpoint:
                        runner.reschedule(task)
                    elif type(msg) is WaitTaskRescheduled:
                        task._cancel_points += 1
                        task._abort_func = msg.abort_func
                        # KI is "outside" all cancel scopes, so check for it
                        # before checking for regular cancellation:
                        if runner.ki_pending and task is runner.main_task:
                            task._attempt_delivery_of_pending_ki()
                        task._attempt_delivery_of_any_pending_cancel()
                    elif type(msg) is PermanentlyDetachCoroutineObject:
                        # Pretend the task just exited with the given outcome
                        runner.task_exited(task, msg.final_outcome)
                    else:
                        exc = TypeError(
                            f"trio.run received unrecognized yield message {msg!r}. "
                            "Are you trying to use a library written for some "
                            "other framework like asyncio? That won't work "
                            "without some kind of compatibility shim.",
                        )
                        # The foreign library probably doesn't adhere to our
                        # protocol of unwrapping whatever outcome gets sent in.
                        # Instead, we'll arrange to throw `exc` in directly,
                        # which works for at least asyncio and curio.
                        runner.reschedule(task, exc)  # type: ignore[arg-type]
                        task._next_send_fn = task.coro.throw
                    # prevent long-lived reference
                    # TODO: develop test for this deletion
                    del msg

                if "after_task_step" in runner.instruments:
                    runner.instruments.call("after_task_step", task)
                del GLOBAL_RUN_CONTEXT.task
                # prevent long-lived references
                # TODO: develop test for this deletion
                del task, next_send, next_send_fn

    except GeneratorExit:
        # The run-loop generator has been garbage collected without finishing
        warnings.warn(
            RuntimeWarning(
                "Trio guest run got abandoned without properly finishing... "
                "weird stuff might happen",
            ),
            stacklevel=1,
        )
    except TrioInternalError:
        raise
    except BaseException as exc:
        raise TrioInternalError("internal error in Trio - please file a bug!") from exc
    finally:
        runner.close()
        GLOBAL_RUN_CONTEXT.__dict__.clear()

        # Have to do this after runner.close() has disabled KI protection,
        # because otherwise there's a race where ki_pending could get set
        # after we check it.
        if runner.ki_pending:
            ki = KeyboardInterrupt()
            if isinstance(runner.main_task_outcome, Error):
                ki.__context__ = runner.main_task_outcome.error
            runner.main_task_outcome = Error(ki)


################################################################
# Other public API functions
################################################################


class _TaskStatusIgnored(TaskStatus[object]):
    def __repr__(self) -> str:
        return "TASK_STATUS_IGNORED"

    def started(self, value: object = None) -> None:
        pass


TASK_STATUS_IGNORED: Final[TaskStatus[object]] = _TaskStatusIgnored()


def current_task() -> Task:
    """Return the :class:`Task` object representing the current task.

    Returns:
      Task: the :class:`Task` that called :func:`current_task`.

    """

    try:
        return GLOBAL_RUN_CONTEXT.task
    except AttributeError:
        raise RuntimeError("must be called from async context") from None


def current_effective_deadline() -> float:
    """Returns the current effective deadline for the current task.

    This function examines all the cancellation scopes that are currently in
    effect (taking into account shielding), and returns the deadline that will
    expire first.

    One example of where this might be is useful is if your code is trying to
    decide whether to begin an expensive operation like an RPC call, but wants
    to skip it if it knows that it can't possibly complete in the available
    time. Another example would be if you're using a protocol like gRPC that
    `propagates timeout information to the remote peer
    <http://www.grpc.io/docs/guides/concepts.html#deadlines>`__; this function
    gives a way to fetch that information so you can send it along.

    If this is called in a context where a cancellation is currently active
    (i.e., a blocking call will immediately raise :exc:`Cancelled`), then
    returned deadline is ``-inf``. If it is called in a context where no
    scopes have a deadline set, it returns ``inf``.

    Returns:
        float: the effective deadline, as an absolute time.

    """
    return current_task()._cancel_status.effective_deadline()


async def checkpoint() -> None:
    """A pure :ref:`checkpoint <checkpoints>`.

    This checks for cancellation and allows other tasks to be scheduled,
    without otherwise blocking.

    Note that the scheduler has the option of ignoring this and continuing to
    run the current task if it decides this is appropriate (e.g. for increased
    efficiency).

    Equivalent to ``await trio.sleep(0)`` (which is implemented by calling
    :func:`checkpoint`.)

    """
    # The scheduler is what checks timeouts and converts them into
    # cancellations. So by doing the schedule point first, we ensure that the
    # cancel point has the most up-to-date info.
    await cancel_shielded_checkpoint()
    task = current_task()
    task._cancel_points += 1
    if task._cancel_status.effectively_cancelled or (
        task is task._runner.main_task and task._runner.ki_pending
    ):
        with CancelScope(deadline=-inf):
            await _core.wait_task_rescheduled(lambda _: _core.Abort.SUCCEEDED)


async def checkpoint_if_cancelled() -> None:
    """Issue a :ref:`checkpoint <checkpoints>` if the calling context has been
    cancelled.

    Equivalent to (but potentially more efficient than)::

        if trio.current_effective_deadline() == -inf:
            await trio.lowlevel.checkpoint()

    This is either a no-op, or else it allow other tasks to be scheduled and
    then raises :exc:`trio.Cancelled`.

    Typically used together with :func:`cancel_shielded_checkpoint`.

    """
    task = current_task()
    if task._cancel_status.effectively_cancelled or (
        task is task._runner.main_task and task._runner.ki_pending
    ):
        await _core.checkpoint()
        raise AssertionError("this should never happen")  # pragma: no cover
    task._cancel_points += 1


def in_trio_run() -> bool:
    """Check whether we are in a Trio run.
    This returns `True` if and only if :func:`~trio.current_time` will succeed.

    See also the discussion of differing ways of :ref:`detecting Trio <trio_contexts>`.
    """
    return hasattr(GLOBAL_RUN_CONTEXT, "runner")


def in_trio_task() -> bool:
    """Check whether we are in a Trio task.
    This returns `True` if and only if :func:`~trio.lowlevel.current_task` will succeed.

    See also the discussion of differing ways of :ref:`detecting Trio <trio_contexts>`.
    """
    return hasattr(GLOBAL_RUN_CONTEXT, "task")


if sys.platform == "win32":
    from ._generated_io_windows import *
    from ._io_windows import (
        EventResult as EventResult,
        WindowsIOManager as TheIOManager,
        _WindowsStatistics as IOStatistics,
    )
elif sys.platform == "linux" or (not TYPE_CHECKING and hasattr(select, "epoll")):
    from ._generated_io_epoll import *
    from ._io_epoll import (
        EpollIOManager as TheIOManager,
        EventResult as EventResult,
        _EpollStatistics as IOStatistics,
    )
elif TYPE_CHECKING or hasattr(select, "kqueue"):
    from ._generated_io_kqueue import *
    from ._io_kqueue import (
        EventResult as EventResult,
        KqueueIOManager as TheIOManager,
        _KqueueStatistics as IOStatistics,
    )
else:  # pragma: no cover
    _patchers = sorted({"eventlet", "gevent"}.intersection(sys.modules))
    if _patchers:
        raise NotImplementedError(
            "unsupported platform or primitives trio depends on are monkey-patched out by "
            + ", ".join(_patchers),
        )

    raise NotImplementedError("unsupported platform")

from ._generated_instrumentation import *
from ._generated_run import *

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\single.py ===
#
# This is the module for ODE solver classes for single ODEs.
#

from __future__ import annotations
from typing import ClassVar, Iterator

from .riccati import match_riccati, solve_riccati
from sympy.core import Add, S, Pow, Rational
from sympy.core.cache import cached_property
from sympy.core.exprtools import factor_terms
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, diff, Function, expand, Subs, _mexpand
from sympy.core.numbers import zoo
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Dummy, Wild
from sympy.core.mul import Mul
from sympy.functions import exp, tan, log, sqrt, besselj, bessely, cbrt, airyai, airybi
from sympy.integrals import Integral
from sympy.polys import Poly
from sympy.polys.polytools import cancel, factor, degree
from sympy.simplify import collect, simplify, separatevars, logcombine, posify # type: ignore
from sympy.simplify.radsimp import fraction
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.solvers.deutils import ode_order, _preprocess
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.solvers import PolyNonlinearError
from .hypergeometric import equivalence_hypergeometric, match_2nd_2F1_hypergeometric, \
    get_sol_2F1_hypergeometric, match_2nd_hypergeometric
from .nonhomogeneous import _get_euler_characteristic_eq_sols, _get_const_characteristic_eq_sols, \
    _solve_undetermined_coefficients, _solve_variation_of_parameters, _test_term, _undetermined_coefficients_match, \
        _get_simplified_sol
from .lie_group import _ode_lie_group


class ODEMatchError(NotImplementedError):
    """Raised if a SingleODESolver is asked to solve an ODE it does not match"""
    pass


class SingleODEProblem:
    """Represents an ordinary differential equation (ODE)

    This class is used internally in the by dsolve and related
    functions/classes so that properties of an ODE can be computed
    efficiently.

    Examples
    ========

    This class is used internally by dsolve. To instantiate an instance
    directly first define an ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now you can create a SingleODEProblem instance and query its properties:

    >>> from sympy.solvers.ode.single import SingleODEProblem
    >>> problem = SingleODEProblem(f(x).diff(x), f(x), x)
    >>> problem.eq
    Derivative(f(x), x)
    >>> problem.func
    f(x)
    >>> problem.sym
    x
    """

    # Instance attributes:
    eq: Expr
    func: AppliedUndef
    sym: Symbol
    _order: int
    _eq_expanded: Expr
    _eq_preprocessed: Expr
    _eq_high_order_free = None

    def __init__(self, eq, func, sym, prep=True, **kwargs):
        assert isinstance(eq, Expr)
        assert isinstance(func, AppliedUndef)
        assert isinstance(sym, Symbol)
        assert isinstance(prep, bool)
        self.eq = eq
        self.func = func
        self.sym = sym
        self.prep = prep
        self.params = kwargs

    @cached_property
    def order(self) -> int:
        return ode_order(self.eq, self.func)

    @cached_property
    def eq_preprocessed(self) -> Expr:
        return self._get_eq_preprocessed()

    @cached_property
    def eq_high_order_free(self) -> Expr:
        a = Wild('a', exclude=[self.func])
        c1 = Wild('c1', exclude=[self.sym])
        # Precondition to try remove f(x) from highest order derivative
        reduced_eq = None
        if self.eq.is_Add:
            deriv_coef = self.eq.coeff(self.func.diff(self.sym, self.order))
            if deriv_coef not in (1, 0):
                r = deriv_coef.match(a*self.func**c1)
                if r and r[c1]:
                    den = self.func**r[c1]
                    reduced_eq = Add(*[arg/den for arg in self.eq.args])
        if reduced_eq is None:
            reduced_eq = expand(self.eq)
        return reduced_eq

    @cached_property
    def eq_expanded(self) -> Expr:
        return expand(self.eq_preprocessed)

    def _get_eq_preprocessed(self) -> Expr:
        if self.prep:
            process_eq, process_func = _preprocess(self.eq, self.func)
            if process_func != self.func:
                raise ValueError
        else:
            process_eq = self.eq
        return process_eq

    def get_numbered_constants(self, num=1, start=1, prefix='C') -> list[Symbol]:
        """
        Returns a list of constants that do not occur
        in eq already.
        """
        ncs = self.iter_numbered_constants(start, prefix)
        Cs = [next(ncs) for i in range(num)]
        return Cs

    def iter_numbered_constants(self, start=1, prefix='C') -> Iterator[Symbol]:
        """
        Returns an iterator of constants that do not occur
        in eq already.
        """
        atom_set = self.eq.free_symbols
        func_set = self.eq.atoms(Function)
        if func_set:
            atom_set |= {Symbol(str(f.func)) for f in func_set}
        return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)

    @cached_property
    def is_autonomous(self):
        u = Dummy('u')
        x = self.sym
        syms = self.eq.subs(self.func, u).free_symbols
        return x not in syms

    def get_linear_coefficients(self, eq, func, order):
        r"""
        Matches a differential equation to the linear form:

        .. math:: a_n(x) y^{(n)} + \cdots + a_1(x)y' + a_0(x) y + B(x) = 0

        Returns a dict of order:coeff terms, where order is the order of the
        derivative on each term, and coeff is the coefficient of that derivative.
        The key ``-1`` holds the function `B(x)`. Returns ``None`` if the ODE is
        not linear.  This function assumes that ``func`` has already been checked
        to be good.

        Examples
        ========

        >>> from sympy import Function, cos, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import SingleODEProblem
        >>> f = Function('f')
        >>> eq = f(x).diff(x, 3) + 2*f(x).diff(x) + \
        ... x*f(x).diff(x, 2) + cos(x)*f(x).diff(x) + x - f(x) - \
        ... sin(x)
        >>> obj = SingleODEProblem(eq, f(x), x)
        >>> obj.get_linear_coefficients(eq, f(x), 3)
        {-1: x - sin(x), 0: -1, 1: cos(x) + 2, 2: x, 3: 1}
        >>> eq = f(x).diff(x, 3) + 2*f(x).diff(x) + \
        ... x*f(x).diff(x, 2) + cos(x)*f(x).diff(x) + x - f(x) - \
        ... sin(f(x))
        >>> obj = SingleODEProblem(eq, f(x), x)
        >>> obj.get_linear_coefficients(eq, f(x), 3) == None
        True

        """
        f = func.func
        x = func.args[0]
        symset = {Derivative(f(x), x, i) for i in range(order+1)}
        try:
            rhs, lhs_terms = _lin_eq2dict(eq, symset)
        except PolyNonlinearError:
            return None

        if rhs.has(func) or any(c.has(func) for c in lhs_terms.values()):
            return None
        terms = {i: lhs_terms.get(f(x).diff(x, i), S.Zero) for i in range(order+1)}
        terms[-1] = rhs
        return terms

    # TODO: Add methods that can be used by many ODE solvers:
    # order
    # is_linear()
    # get_linear_coefficients()
    # eq_prepared (the ODE in prepared form)


class SingleODESolver:
    """
    Base class for Single ODE solvers.

    Subclasses should implement the _matches and _get_general_solution
    methods. This class is not intended to be instantiated directly but its
    subclasses are as part of dsolve.

    Examples
    ========

    You can use a subclass of SingleODEProblem to solve a particular type of
    ODE. We first define a particular ODE problem:

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)

    Now we solve this problem using the NthAlgebraic solver which is a
    subclass of SingleODESolver:

    >>> from sympy.solvers.ode.single import NthAlgebraic, SingleODEProblem
    >>> problem = SingleODEProblem(eq, f(x), x)
    >>> solver = NthAlgebraic(problem)
    >>> solver.get_general_solution()
    [Eq(f(x), _C*x + _C)]

    The normal way to solve an ODE is to use dsolve (which would use
    NthAlgebraic and other solvers internally). When using dsolve a number of
    other things are done such as evaluating integrals, simplifying the
    solution and renumbering the constants:

    >>> from sympy import dsolve
    >>> dsolve(eq, hint='nth_algebraic')
    Eq(f(x), C1 + C2*x)
    """

    # Subclasses should store the hint name (the argument to dsolve) in this
    # attribute
    hint: ClassVar[str]

    # Subclasses should define this to indicate if they support an _Integral
    # hint.
    has_integral: ClassVar[bool]

    # The ODE to be solved
    ode_problem: SingleODEProblem

    # Cache whether or not the equation has matched the method
    _matched: bool | None = None

    # Subclasses should store in this attribute the list of order(s) of ODE
    # that subclass can solve or leave it to None if not specific to any order
    order: list | None = None

    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def matches(self) -> bool:
        if self.order is not None and self.ode_problem.order not in self.order:
            self._matched = False
            return self._matched

        if self._matched is None:
            self._matched = self._matches()
        return self._matched

    def get_general_solution(self, *, simplify: bool = True) -> list[Equality]:
        if not self.matches():
            msg = "%s solver cannot solve:\n%s"
            raise ODEMatchError(msg % (self.hint, self.ode_problem.eq))
        return self._get_general_solution(simplify_flag=simplify)

    def _matches(self) -> bool:
        msg = "Subclasses of SingleODESolver should implement matches."
        raise NotImplementedError(msg)

    def _get_general_solution(self, *, simplify_flag: bool = True) -> list[Equality]:
        msg = "Subclasses of SingleODESolver should implement get_general_solution."
        raise NotImplementedError(msg)


class SinglePatternODESolver(SingleODESolver):
    '''Superclass for ODE solvers based on pattern matching'''

    def wilds(self):
        prob = self.ode_problem
        f = prob.func.func
        x = prob.sym
        order = prob.order
        return self._wilds(f, x, order)

    def wilds_match(self):
        match = self._wilds_match
        return [match.get(w, S.Zero) for w in self.wilds()]

    def _matches(self):
        eq = self.ode_problem.eq_expanded
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        df = f(x).diff(x, order)

        if order not in [1, 2]:
            return False

        pattern = self._equation(f(x), x, order)

        if not pattern.coeff(df).has(Wild):
            eq = expand(eq / eq.coeff(df))
        eq = eq.collect([f(x).diff(x), f(x)], func = cancel)

        self._wilds_match = match = eq.match(pattern)
        if match is not None:
            return self._verify(f(x))
        return False

    def _verify(self, fx) -> bool:
        return True

    def _wilds(self, f, x, order):
        msg = "Subclasses of SingleODESolver should implement _wilds"
        raise NotImplementedError(msg)

    def _equation(self, fx, x, order):
        msg = "Subclasses of SingleODESolver should implement _equation"
        raise NotImplementedError(msg)


class NthAlgebraic(SingleODESolver):
    r"""
    Solves an `n`\th order ordinary differential equation using algebra and
    integrals.

    There is no general form for the kind of equation that this can solve. The
    the equation is solved algebraically treating differentiation as an
    invertible algebraic function.

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = Eq(f(x) * (f(x).diff(x)**2 - 1), 0)
    >>> dsolve(eq, f(x), hint='nth_algebraic')
    [Eq(f(x), 0), Eq(f(x), C1 - x), Eq(f(x), C1 + x)]

    Note that this solver can return algebraic solutions that do not have any
    integration constants (f(x) = 0 in the above example).
    """

    hint = 'nth_algebraic'
    has_integral = True  # nth_algebraic_Integral hint

    def _matches(self):
        r"""
        Matches any differential equation that nth_algebraic can solve. Uses
        `sympy.solve` but teaches it how to integrate derivatives.

        This involves calling `sympy.solve` and does most of the work of finding a
        solution (apart from evaluating the integrals).
        """
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        var = self.ode_problem.sym

        # Derivative that solve can handle:
        diffx = self._get_diffx(var)

        # Replace derivatives wrt the independent variable with diffx
        def replace(eq, var):
            def expand_diffx(*args):
                differand, diffs = args[0], args[1:]
                toreplace = differand
                for v, n in diffs:
                    for _ in range(n):
                        if v == var:
                            toreplace = diffx(toreplace)
                        else:
                            toreplace = Derivative(toreplace, v)
                return toreplace
            return eq.replace(Derivative, expand_diffx)

        # Restore derivatives in solution afterwards
        def unreplace(eq, var):
            return eq.replace(diffx, lambda e: Derivative(e, var))

        subs_eqn = replace(eq, var)
        try:
            # turn off simplification to protect Integrals that have
            # _t instead of fx in them and would otherwise factor
            # as t_*Integral(1, x)
            solns = solve(subs_eqn, func, simplify=False)
        except NotImplementedError:
            solns = []

        solns = [simplify(unreplace(soln, var)) for soln in solns]
        solns = [Equality(func, soln) for soln in solns]

        self.solutions = solns
        return len(solns) != 0

    def _get_general_solution(self, *, simplify_flag: bool = True):
        return self.solutions

    # This needs to produce an invertible function but the inverse depends
    # which variable we are integrating with respect to. Since the class can
    # be stored in cached results we need to ensure that we always get the
    # same class back for each particular integration variable so we store these
    # classes in a global dict:
    _diffx_stored: dict[Symbol, type[Function]] = {}

    @staticmethod
    def _get_diffx(var):
        diffcls = NthAlgebraic._diffx_stored.get(var, None)

        if diffcls is None:
            # A class that behaves like Derivative wrt var but is "invertible".
            class diffx(Function):
                def inverse(self):
                    # don't use integrate here because fx has been replaced by _t
                    # in the equation; integrals will not be correct while solve
                    # is at work.
                    return lambda expr: Integral(expr, var) + Dummy('C')

            diffcls = NthAlgebraic._diffx_stored.setdefault(var, diffx)

        return diffcls


class FirstExact(SinglePatternODESolver):
    r"""
    Solves 1st order exact ordinary differential equations.

    A 1st order differential equation is called exact if it is the total
    differential of a function. That is, the differential equation

    .. math:: P(x, y) \,\partial{}x + Q(x, y) \,\partial{}y = 0

    is exact if there is some function `F(x, y)` such that `P(x, y) =
    \partial{}F/\partial{}x` and `Q(x, y) = \partial{}F/\partial{}y`.  It can
    be shown that a necessary and sufficient condition for a first order ODE
    to be exact is that `\partial{}P/\partial{}y = \partial{}Q/\partial{}x`.
    Then, the solution will be as given below::

        >>> from sympy import Function, Eq, Integral, symbols, pprint
        >>> x, y, t, x0, y0, C1= symbols('x,y,t,x0,y0,C1')
        >>> P, Q, F= map(Function, ['P', 'Q', 'F'])
        >>> pprint(Eq(Eq(F(x, y), Integral(P(t, y), (t, x0, x)) +
        ... Integral(Q(x0, t), (t, y0, y))), C1))
                    x                y
                    /                /
                   |                |
        F(x, y) =  |  P(t, y) dt +  |  Q(x0, t) dt = C1
                   |                |
                  /                /
                  x0               y0

    Where the first partials of `P` and `Q` exist and are continuous in a
    simply connected region.

    A note: SymPy currently has no way to represent inert substitution on an
    expression, so the hint ``1st_exact_Integral`` will return an integral
    with `dy`.  This is supposed to represent the function that you are
    solving for.

    Examples
    ========

    >>> from sympy import Function, dsolve, cos, sin
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x),
    ... f(x), hint='1st_exact')
    Eq(x*cos(f(x)) + f(x)**3/3, C1)

    References
    ==========

    - https://en.wikipedia.org/wiki/Exact_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 73

    # indirect doctest

    """
    hint = "1st_exact"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return P + Q*fx.diff(x)

    def _verify(self, fx) -> bool:
        P, Q = self.wilds()
        x = self.ode_problem.sym
        y = Dummy('y')

        m, n = self.wilds_match()

        m = m.subs(fx, y)
        n = n.subs(fx, y)
        numerator = cancel(m.diff(y) - n.diff(x))

        if numerator.is_zero:
            # Is exact
            return True
        else:
            # The following few conditions try to convert a non-exact
            # differential equation into an exact one.
            # References:
            # 1. Differential equations with applications
            # and historical notes - George E. Simmons
            # 2. https://math.okstate.edu/people/binegar/2233-S99/2233-l12.pdf

            factor_n = cancel(numerator/n)
            factor_m = cancel(-numerator/m)
            if y not in factor_n.free_symbols:
                # If (dP/dy - dQ/dx) / Q = f(x)
                # then exp(integral(f(x))*equation becomes exact
                factor = factor_n
                integration_variable = x
            elif x not in factor_m.free_symbols:
                # If (dP/dy - dQ/dx) / -P = f(y)
                # then exp(integral(f(y))*equation becomes exact
                factor = factor_m
                integration_variable = y
            else:
                # Couldn't convert to exact
                return False

            factor = exp(Integral(factor, integration_variable))
            m *= factor
            n *= factor
            self._wilds_match[P] = m.subs(y, fx)
            self._wilds_match[Q] = n.subs(y, fx)
            return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        m, n = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        y = Dummy('y')

        m = m.subs(fx, y)
        n = n.subs(fx, y)

        gen_sol = Eq(Subs(Integral(m, x)
                          + Integral(n - Integral(m, x).diff(y), y), y, fx), C1)
        return [gen_sol]


class FirstLinear(SinglePatternODESolver):
    r"""
    Solves 1st order linear differential equations.

    These are differential equations of the form

    .. math:: dy/dx + P(x) y = Q(x)\text{.}

    These kinds of differential equations can be solved in a general way.  The
    integrating factor `e^{\int P(x) \,dx}` will turn the equation into a
    separable equation.  The general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint, diff, sin
        >>> from sympy.abc import x
        >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
        >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x))
        >>> pprint(genform)
                    d
        P(x)*f(x) + --(f(x)) = Q(x)
                    dx
        >>> pprint(dsolve(genform, f(x), hint='1st_linear_Integral'))
                /       /                   \
                |      |                    |
                |      |         /          |     /
                |      |        |           |    |
                |      |        | P(x) dx   |  - | P(x) dx
                |      |        |           |    |
                |      |       /            |   /
        f(x) = |C1 +  | Q(x)*e           dx|*e
                |      |                    |
                \     /                     /


    Examples
    ========

    >>> f = Function('f')
    >>> pprint(dsolve(Eq(x*diff(f(x), x) - f(x), x**2*sin(x)),
    ... f(x), '1st_linear'))
    f(x) = x*(C1 - cos(x))

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation#First-order_equation_with_variable_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 92

    # indirect doctest

    """
    hint = '1st_linear'
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x), f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return fx.diff(x) + P*fx - Q

    def _get_general_solution(self, *, simplify_flag: bool = True):
        P, Q = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        gensol = Eq(fx, ((C1 + Integral(Q*exp(Integral(P, x)), x))
            * exp(-Integral(P, x))))
        return [gensol]


class AlmostLinear(SinglePatternODESolver):
    r"""
    Solves an almost-linear differential equation.

    The general form of an almost linear differential equation is

    .. math:: a(x) g'(f(x)) f'(x) + b(x) g(f(x)) + c(x)

    Here `f(x)` is the function to be solved for (the dependent variable).
    The substitution `g(f(x)) = u(x)` leads to a linear differential equation
    for `u(x)` of the form `a(x) u' + b(x) u + c(x) = 0`. This can be solved
    for `u(x)` by the `first_linear` hint and then `f(x)` is found by solving
    `g(f(x)) = u(x)`.

    See Also
    ========
    :obj:`sympy.solvers.ode.single.FirstLinear`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint, sin, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = x*d + x*f(x) + 1
    >>> dsolve(eq, f(x), hint='almost_linear')
    Eq(f(x), (C1 - Ei(x))*exp(-x))
    >>> pprint(dsolve(eq, f(x), hint='almost_linear'))
                        -x
    f(x) = (C1 - Ei(x))*e
    >>> example = cos(f(x))*f(x).diff(x) + sin(f(x)) + 1
    >>> pprint(example)
                        d
    sin(f(x)) + cos(f(x))*--(f(x)) + 1
                        dx
    >>> pprint(dsolve(example, f(x), hint='almost_linear'))
                    /    -x    \             /    -x    \
    [f(x) = pi - asin\C1*e   - 1/, f(x) = asin\C1*e   - 1/]


    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "almost_linear"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x).diff(x)])
        Q = Wild('Q', exclude=[f(x).diff(x)])
        return P, Q

    def _equation(self, fx, x, order):
        P, Q = self.wilds()
        return P*fx.diff(x) + Q

    def _verify(self, fx):
        a, b = self.wilds_match()
        c, b = b.as_independent(fx) if b.is_Add else (S.Zero, b)
        # a, b and c are the function a(x), b(x) and c(x) respectively.
        # c(x) is obtained by separating out b as terms with and without fx i.e, l(y)
        # The following conditions checks if the given equation is an almost-linear differential equation using the fact that
        # a(x)*(l(y))' / l(y)' is independent of l(y)

        if b.diff(fx) != 0 and not simplify(b.diff(fx)/a).has(fx):
            self.ly = factor_terms(b).as_independent(fx, as_Add=False)[1] # Gives the term containing fx i.e., l(y)
            self.ax = a / self.ly.diff(fx)
            self.cx = -c  # cx is taken as -c(x) to simplify expression in the solution integral
            self.bx = factor_terms(b) / self.ly
            return True

        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        x = self.ode_problem.sym
        (C1,)  = self.ode_problem.get_numbered_constants(num=1)
        gensol = Eq(self.ly, ((C1 + Integral((self.cx/self.ax)*exp(Integral(self.bx/self.ax, x)), x))
                * exp(-Integral(self.bx/self.ax, x))))

        return [gensol]


class Bernoulli(SinglePatternODESolver):
    r"""
    Solves Bernoulli differential equations.

    These are equations of the form

    .. math:: dy/dx + P(x) y = Q(x) y^n\text{, }n \ne 1`\text{.}

    The substitution `w = 1/y^{1-n}` will transform an equation of this form
    into one that is linear (see the docstring of
    :obj:`~sympy.solvers.ode.single.FirstLinear`).  The general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint
        >>> from sympy.abc import x, n
        >>> f, P, Q = map(Function, ['f', 'P', 'Q'])
        >>> genform = Eq(f(x).diff(x) + P(x)*f(x), Q(x)*f(x)**n)
        >>> pprint(genform)
                    d                n
        P(x)*f(x) + --(f(x)) = Q(x)*f (x)
                    dx
        >>> pprint(dsolve(genform, f(x), hint='Bernoulli_Integral'), num_columns=110)
                                                                                                                -1
                                                                                                               -----
                                                                                                               n - 1
               //         /                                 /                            \                    \
               ||        |                                 |                             |                    |
               ||        |                  /              |                  /          |            /       |
               ||        |                 |               |                 |           |           |        |
               ||        |       -(n - 1)* | P(x) dx       |       -(n - 1)* | P(x) dx   |  (n - 1)* | P(x) dx|
               ||        |                 |               |                 |           |           |        |
               ||        |                /                |                /            |          /         |
        f(x) = ||C1 - n* | Q(x)*e                    dx +  | Q(x)*e                    dx|*e                  |
               ||        |                                 |                             |                    |
               \\       /                                 /                              /                    /


    Note that the equation is separable when `n = 1` (see the docstring of
    :obj:`~sympy.solvers.ode.single.Separable`).

    >>> pprint(dsolve(Eq(f(x).diff(x) + P(x)*f(x), Q(x)*f(x)), f(x),
    ... hint='separable_Integral'))
    f(x)
        /
    |                /
    |  1            |
    |  - dy = C1 +  | (-P(x) + Q(x)) dx
    |  y            |
    |              /
    /


    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, pprint, log
    >>> from sympy.abc import x
    >>> f = Function('f')

    >>> pprint(dsolve(Eq(x*f(x).diff(x) + f(x), log(x)*f(x)**2),
    ... f(x), hint='Bernoulli'))
                    1
    f(x) =  -----------------
            C1*x + log(x) + 1

    References
    ==========

    - https://en.wikipedia.org/wiki/Bernoulli_differential_equation

    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 95

    # indirect doctest

    """
    hint = "Bernoulli"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        P = Wild('P', exclude=[f(x)])
        Q = Wild('Q', exclude=[f(x)])
        n = Wild('n', exclude=[x, f(x), f(x).diff(x)])
        return P, Q, n

    def _equation(self, fx, x, order):
        P, Q, n = self.wilds()
        return fx.diff(x) + P*fx - Q*fx**n

    def _get_general_solution(self, *, simplify_flag: bool = True):
        P, Q, n = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        if n==1:
            gensol = Eq(log(fx), (
            C1 + Integral((-P + Q), x)
        ))
        else:
            gensol = Eq(fx**(1-n), (
                (C1 - (n - 1) * Integral(Q*exp(-n*Integral(P, x))
                            * exp(Integral(P, x)), x)
                ) * exp(-(1 - n)*Integral(P, x)))
            )
        return [gensol]


class Factorable(SingleODESolver):
    r"""
        Solves equations having a solvable factor.

        This function is used to solve the equation having factors. Factors may be of type algebraic or ode. It
        will try to solve each factor independently. Factors will be solved by calling dsolve. We will return the
        list of solutions.

        Examples
        ========

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x
        >>> f = Function('f')
        >>> eq = (f(x)**2-4)*(f(x).diff(x)+f(x))
        >>> pprint(dsolve(eq, f(x)))
                                        -x
        [f(x) = 2, f(x) = -2, f(x) = C1*e  ]


        """
    hint = "factorable"
    has_integral = False

    def _matches(self):
        eq_orig = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        df = f(x).diff(x)
        self.eqs = []
        eq = eq_orig.collect(f(x), func = cancel)
        eq = fraction(factor(eq))[0]
        factors = Mul.make_args(factor(eq))
        roots = [fac.as_base_exp() for fac in factors if len(fac.args)!=0]
        if len(roots)>1 or roots[0][1]>1:
            for base, expo in roots:
                if base.has(f(x)):
                    self.eqs.append(base)
            if len(self.eqs)>0:
                return True
        roots = solve(eq, df)
        if len(roots)>0:
            self.eqs = [(df - root) for root in roots]
            # Avoid infinite recursion
            matches = self.eqs != [eq_orig]
            return matches
        for i in factors:
            if i.has(f(x)):
                self.eqs.append(i)
        return len(self.eqs)>0 and len(factors)>1

    def _get_general_solution(self, *, simplify_flag: bool = True):
        func = self.ode_problem.func.func
        x = self.ode_problem.sym
        eqns = self.eqs
        sols = []
        for eq in eqns:
            try:
                sol = dsolve(eq, func(x))
            except NotImplementedError:
                continue
            else:
                if isinstance(sol, list):
                    sols.extend(sol)
                else:
                    sols.append(sol)

        if sols == []:
            raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                + " the factorable group method")
        return sols


class RiccatiSpecial(SinglePatternODESolver):
    r"""
    The general Riccati equation has the form

    .. math:: dy/dx = f(x) y^2 + g(x) y + h(x)\text{.}

    While it does not have a general solution [1], the "special" form, `dy/dx
    = a y^2 - b x^c`, does have solutions in many cases [2].  This routine
    returns a solution for `a(dy/dx) = b y^2 + c y/x + d/x^2` that is obtained
    by using a suitable change of variables to reduce it to the special form
    and is valid when neither `a` nor `b` are zero and either `c` or `d` is
    zero.

    >>> from sympy.abc import x, a, b, c, d
    >>> from sympy import dsolve, checkodesol, pprint, Function
    >>> f = Function('f')
    >>> y = f(x)
    >>> genform = a*y.diff(x) - (b*y**2 + c*y/x + d/x**2)
    >>> sol = dsolve(genform, y, hint="Riccati_special_minus2")
    >>> pprint(sol, wrap_line=False)
            /                                 /        __________________       \\
            |           __________________    |       /                2        ||
            |          /                2     |     \/  4*b*d - (a + c)  *log(x)||
           -|a + c - \/  4*b*d - (a + c)  *tan|C1 + ----------------------------||
            \                                 \                 2*a             //
    f(x) = ------------------------------------------------------------------------
                                            2*b*x

    >>> checkodesol(genform, sol, order=1)[0]
    True

    References
    ==========

    - https://www.maplesoft.com/support/help/Maple/view.aspx?path=odeadvisor/Riccati
    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0106.pdf -
      https://eqworld.ipmnet.ru/en/solutions/ode/ode0123.pdf
    """
    hint = "Riccati_special_minus2"
    has_integral = False
    order = [1]

    def _wilds(self, f, x, order):
        a = Wild('a', exclude=[x, f(x), f(x).diff(x), 0])
        b = Wild('b', exclude=[x, f(x), f(x).diff(x), 0])
        c = Wild('c', exclude=[x, f(x), f(x).diff(x)])
        d = Wild('d', exclude=[x, f(x), f(x).diff(x)])
        return a, b, c, d

    def _equation(self, fx, x, order):
        a, b, c, d = self.wilds()
        return a*fx.diff(x) + b*fx**2 + c*fx/x + d/x**2

    def _get_general_solution(self, *, simplify_flag: bool = True):
        a, b, c, d = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        mu = sqrt(4*d*b - (a - c)**2)

        gensol = Eq(fx, (a - c - mu*tan(mu/(2*a)*log(x) + C1))/(2*b*x))
        return [gensol]


class RationalRiccati(SinglePatternODESolver):
    r"""
    Gives general solutions to the first order Riccati differential
    equations that have atleast one rational particular solution.

    .. math :: y' = b_0(x) + b_1(x) y + b_2(x) y^2

    where `b_0`, `b_1` and `b_2` are rational functions of `x`
    with `b_2 \ne 0` (`b_2 = 0` would make it a Bernoulli equation).

    Examples
    ========

    >>> from sympy import Symbol, Function, dsolve, checkodesol
    >>> f = Function('f')
    >>> x = Symbol('x')

    >>> eq = -x**4*f(x)**2 + x**3*f(x).diff(x) + x**2*f(x) + 20
    >>> sol = dsolve(eq, hint="1st_rational_riccati")
    >>> sol
    Eq(f(x), (4*C1 - 5*x**9 - 4)/(x**2*(C1 + x**9 - 1)))
    >>> checkodesol(eq, sol)
    (True, 0)

    References
    ==========

    - Riccati ODE:  https://en.wikipedia.org/wiki/Riccati_equation
    - N. Thieu Vo - Rational and Algebraic Solutions of First-Order Algebraic ODEs:
      Algorithm 11, pp. 78 - https://www3.risc.jku.at/publications/download/risc_5387/PhDThesisThieu.pdf
    """
    has_integral = False
    hint = "1st_rational_riccati"
    order = [1]

    def _wilds(self, f, x, order):
        b0 = Wild('b0', exclude=[f(x), f(x).diff(x)])
        b1 = Wild('b1', exclude=[f(x), f(x).diff(x)])
        b2 = Wild('b2', exclude=[f(x), f(x).diff(x)])
        return (b0, b1, b2)

    def _equation(self, fx, x, order):
        b0, b1, b2 = self.wilds()
        return fx.diff(x) - b0 - b1*fx - b2*fx**2

    def _matches(self):
        eq = self.ode_problem.eq_expanded
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order

        if order != 1:
            return False

        match, funcs = match_riccati(eq, f, x)
        if not match:
            return False
        _b0, _b1, _b2 = funcs
        b0, b1, b2 = self.wilds()
        self._wilds_match = match = {b0: _b0, b1: _b1, b2: _b2}
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # Match the equation
        b0, b1, b2 = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return solve_riccati(fx, x, b0, b1, b2, gensol=True)


class SecondNonlinearAutonomousConserved(SinglePatternODESolver):
    r"""
    Gives solution for the autonomous second order nonlinear
    differential equation of the form

    .. math :: f''(x) = g(f(x))

    The solution for this differential equation can be computed
    by multiplying by `f'(x)` and integrating on both sides,
    converting it into a first order differential equation.

    Examples
    ========

    >>> from sympy import Function, symbols, dsolve
    >>> f, g = symbols('f g', cls=Function)
    >>> x = symbols('x')

    >>> eq = f(x).diff(x, 2) - g(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(C1 + 2*Integral(g(_u), _u)), (_u, f(x))), C2 - x)]

    >>> from sympy import exp, log
    >>> eq = f(x).diff(x, 2) - exp(f(x)) + log(f(x))
    >>> dsolve(eq, simplify=False)
    [Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(-2*_u*log(_u) + 2*_u + C1 + 2*exp(_u)), (_u, f(x))), C2 - x)]

    References
    ==========

    - https://eqworld.ipmnet.ru/en/solutions/ode/ode0301.pdf
    """
    hint = "2nd_nonlinear_autonomous_conserved"
    has_integral = True
    order = [2]

    def _wilds(self, f, x, order):
        fy = Wild('fy', exclude=[0, f(x).diff(x), f(x).diff(x, 2)])
        return (fy, )

    def _equation(self, fx, x, order):
        fy = self.wilds()[0]
        return fx.diff(x, 2) + fy

    def _verify(self, fx):
        return self.ode_problem.is_autonomous

    def _get_general_solution(self, *, simplify_flag: bool = True):
        g = self.wilds_match()[0]
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        u = Dummy('u')
        g = g.subs(fx, u)
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        inside = -2*Integral(g, u) + C1
        lhs = Integral(1/sqrt(inside), (u, fx))
        return [Eq(lhs, C2 + x), Eq(lhs, C2 - x)]


class Liouville(SinglePatternODESolver):
    r"""
    Solves 2nd order Liouville differential equations.

    The general form of a Liouville ODE is

    .. math:: \frac{d^2 y}{dx^2} + g(y) \left(\!
                \frac{dy}{dx}\!\right)^2 + h(x)
                \frac{dy}{dx}\text{.}

    The general solution is:

        >>> from sympy import Function, dsolve, Eq, pprint, diff
        >>> from sympy.abc import x
        >>> f, g, h = map(Function, ['f', 'g', 'h'])
        >>> genform = Eq(diff(f(x),x,x) + g(f(x))*diff(f(x),x)**2 +
        ... h(x)*diff(f(x),x), 0)
        >>> pprint(genform)
                          2                    2
                /d       \         d          d
        g(f(x))*|--(f(x))|  + h(x)*--(f(x)) + ---(f(x)) = 0
                \dx      /         dx           2
                                              dx
        >>> pprint(dsolve(genform, f(x), hint='Liouville_Integral'))
                                          f(x)
                  /                     /
                 |                     |
                 |     /               |     /
                 |    |                |    |
                 |  - | h(x) dx        |    | g(y) dy
                 |    |                |    |
                 |   /                 |   /
        C1 + C2* | e            dx +   |  e           dy = 0
                 |                     |
                /                     /

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(diff(f(x), x, x) + diff(f(x), x)**2/f(x) +
    ... diff(f(x), x)/x, f(x), hint='Liouville'))
               ________________           ________________
    [f(x) = -\/ C1 + C2*log(x) , f(x) = \/ C1 + C2*log(x) ]

    References
    ==========

    - Goldstein and Braun, "Advanced Methods for the Solution of Differential
      Equations", pp. 98
    - https://www.maplesoft.com/support/help/Maple/view.aspx?path=odeadvisor/Liouville

    # indirect doctest

    """
    hint = "Liouville"
    has_integral = True
    order = [2]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        k = Wild('k', exclude=[f(x).diff(x)])
        return d, e, k

    def _equation(self, fx, x, order):
        # Liouville ODE in the form
        # f(x).diff(x, 2) + g(f(x))*(f(x).diff(x))**2 + h(x)*f(x).diff(x)
        # See Goldstein and Braun, "Advanced Methods for the Solution of
        # Differential Equations", pg. 98
        d, e, k = self.wilds()
        return d*fx.diff(x, 2) + e*fx.diff(x)**2 + k*fx.diff(x)

    def _verify(self, fx):
        d, e, k = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.g = simplify(e/d).subs(fx, self.y)
        self.h = simplify(k/d).subs(fx, self.y)
        if self.y in self.h.free_symbols or x in self.g.free_symbols:
            return False
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, k = self.wilds_match()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        C1, C2 = self.ode_problem.get_numbered_constants(num=2)
        int = Integral(exp(Integral(self.g, self.y)), (self.y, None, fx))
        gen_sol = Eq(int + C1*Integral(exp(-Integral(self.h, x)), x) + C2, 0)

        return [gen_sol]


class Separable(SinglePatternODESolver):
    r"""
    Solves separable 1st order differential equations.

    This is any differential equation that can be written as `P(y)
    \tfrac{dy}{dx} = Q(x)`.  The solution can then just be found by
    rearranging terms and integrating: `\int P(y) \,dy = \int Q(x) \,dx`.
    This hint uses :py:meth:`sympy.simplify.simplify.separatevars` as its back
    end, so if a separable equation is not caught by this solver, it is most
    likely the fault of that function.
    :py:meth:`~sympy.simplify.simplify.separatevars` is
    smart enough to do most expansion and factoring necessary to convert a
    separable equation `F(x, y)` into the proper form `P(x)\cdot{}Q(y)`.  The
    general solution is::

        >>> from sympy import Function, dsolve, Eq, pprint
        >>> from sympy.abc import x
        >>> a, b, c, d, f = map(Function, ['a', 'b', 'c', 'd', 'f'])
        >>> genform = Eq(a(x)*b(f(x))*f(x).diff(x), c(x)*d(f(x)))
        >>> pprint(genform)
                     d
        a(x)*b(f(x))*--(f(x)) = c(x)*d(f(x))
                     dx
        >>> pprint(dsolve(genform, f(x), hint='separable_Integral'))
             f(x)
           /                  /
          |                  |
          |  b(y)            | c(x)
          |  ---- dy = C1 +  | ---- dx
          |  d(y)            | a(x)
          |                  |
         /                  /

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(Eq(f(x)*f(x).diff(x) + x, 3*x*f(x)**2), f(x),
    ... hint='separable', simplify=False))
       /   2       \         2
    log\3*f (x) - 1/        x
    ---------------- = C1 + --
           6                2

    References
    ==========

    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 52

    # indirect doctest

    """
    hint = "separable"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        d, e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        d = separatevars(d.subs(fx, self.y))
        e = separatevars(e.subs(fx, self.y))
        # m1[coeff]*m1[x]*m1[y] + m2[coeff]*m2[x]*m2[y]*y'
        self.m1 = separatevars(d, dict=True, symbols=(x, self.y))
        self.m2 = separatevars(e, dict=True, symbols=(x, self.y))
        return bool(self.m1 and self.m2)

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        return self.m1, self.m2, x, fx

    def _get_general_solution(self, *, simplify_flag: bool = True):
        m1, m2, x, fx = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(m2['coeff']*m2[self.y]/m1[self.y],
        (self.y, None, fx))
        gen_sol = Eq(int, Integral(-m1['coeff']*m1[x]/
        m2[x], x) + C1)
        return [gen_sol]


class SeparableReduced(Separable):
    r"""
    Solves a differential equation that can be reduced to the separable form.

    The general form of this equation is

    .. math:: y' + (y/x) H(x^n y) = 0\text{}.

    This can be solved by substituting `u(y) = x^n y`.  The equation then
    reduces to the separable form `\frac{u'}{u (\mathrm{power} - H(u))} -
    \frac{1}{x} = 0`.

    The general solution is:

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x, n
        >>> f, g = map(Function, ['f', 'g'])
        >>> genform = f(x).diff(x) + (f(x)/x)*g(x**n*f(x))
        >>> pprint(genform)
                         / n     \
        d          f(x)*g\x *f(x)/
        --(f(x)) + ---------------
        dx                x
        >>> pprint(dsolve(genform, hint='separable_reduced'))
         n
        x *f(x)
          /
         |
         |         1
         |    ------------ dy = C1 + log(x)
         |    y*(n - g(y))
         |
         /

    See Also
    ========
    :obj:`sympy.solvers.ode.single.Separable`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> d = f(x).diff(x)
    >>> eq = (x - x**2*f(x))*d - f(x)
    >>> dsolve(eq, hint='separable_reduced')
    [Eq(f(x), (1 - sqrt(C1*x**2 + 1))/x), Eq(f(x), (sqrt(C1*x**2 + 1) + 1)/x)]
    >>> pprint(dsolve(eq, hint='separable_reduced'))
                   ___________            ___________
                  /     2                /     2
            1 - \/  C1*x  + 1          \/  C1*x  + 1  + 1
    [f(x) = ------------------, f(x) = ------------------]
                    x                          x

    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "separable_reduced"
    has_integral = True
    order = [1]

    def _degree(self, expr, x):
        # Made this function to calculate the degree of
        # x in an expression. If expr will be of form
        # x**p*y, (wheare p can be variables/rationals) then it
        # will return p.
        for val in expr:
            if val.has(x):
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    return (val.as_base_exp()[1])
                elif val == x:
                    return (val.as_base_exp()[1])
                else:
                    return self._degree(val.args, x)
        return 0

    def _powers(self, expr):
        # this function will return all the different relative power of x w.r.t f(x).
        # expr = x**p * f(x)**q then it will return {p/q}.
        pows = set()
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.y = Dummy('y')
        if isinstance(expr, Add):
            exprs = expr.atoms(Add)
        elif isinstance(expr, Mul):
            exprs = expr.atoms(Mul)
        elif isinstance(expr, Pow):
            exprs = expr.atoms(Pow)
        else:
            exprs = {expr}

        for arg in exprs:
            if arg.has(x):
                _, u = arg.as_independent(x, fx)
                pow = self._degree((u.subs(fx, self.y), ), x)/self._degree((u.subs(fx, self.y), ), self.y)
                pows.add(pow)
        return pows

    def _verify(self, fx):
        num, den = self.wilds_match()
        x = self.ode_problem.sym
        factor = simplify(x/fx*num/den)
        # Try representing factor in terms of x^n*y
        # where n is lowest power of x in factor;
        # first remove terms like sqrt(2)*3 from factor.atoms(Mul)
        num, dem = factor.as_numer_denom()
        num = expand(num)
        dem = expand(dem)
        pows = self._powers(num)
        pows.update(self._powers(dem))
        pows = list(pows)
        if(len(pows)==1) and pows[0]!=zoo:
            self.t = Dummy('t')
            self.r2 = {'t': self.t}
            num = num.subs(x**pows[0]*fx, self.t)
            dem = dem.subs(x**pows[0]*fx, self.t)
            test = num/dem
            free = test.free_symbols
            if len(free) == 1 and free.pop() == self.t:
                self.r2.update({'power' : pows[0], 'u' : test})
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        u = self.r2['u'].subs(self.r2['t'], self.y)
        ycoeff = 1/(self.y*(self.r2['power'] - u))
        m1 = {self.y: 1, x: -1/x, 'coeff': 1}
        m2 = {self.y: ycoeff, x: 1, 'coeff': 1}
        return m1, m2, x, x**self.r2['power']*fx


class HomogeneousCoeffSubsDepDivIndep(SinglePatternODESolver):
    r"""
    Solves a 1st order differential equation with homogeneous coefficients
    using the substitution `u_1 = \frac{\text{<dependent
    variable>}}{\text{<independent variable>}}`.

    This is a differential equation

    .. math:: P(x, y) + Q(x, y) dy/dx = 0

    such that `P` and `Q` are homogeneous and of the same order.  A function
    `F(x, y)` is homogeneous of order `n` if `F(x t, y t) = t^n F(x, y)`.
    Equivalently, `F(x, y)` can be rewritten as `G(y/x)` or `H(x/y)`.  See
    also the docstring of :py:meth:`~sympy.solvers.ode.homogeneous_order`.

    If the coefficients `P` and `Q` in the differential equation above are
    homogeneous functions of the same order, then it can be shown that the
    substitution `y = u_1 x` (i.e. `u_1 = y/x`) will turn the differential
    equation into an equation separable in the variables `x` and `u`.  If
    `h(u_1)` is the function that results from making the substitution `u_1 =
    f(x)/x` on `P(x, f(x))` and `g(u_2)` is the function that results from the
    substitution on `Q(x, f(x))` in the differential equation `P(x, f(x)) +
    Q(x, f(x)) f'(x) = 0`, then the general solution is::

        >>> from sympy import Function, dsolve, pprint
        >>> from sympy.abc import x
        >>> f, g, h = map(Function, ['f', 'g', 'h'])
        >>> genform = g(f(x)/x) + h(f(x)/x)*f(x).diff(x)
        >>> pprint(genform)
         /f(x)\    /f(x)\ d
        g|----| + h|----|*--(f(x))
         \ x  /    \ x  / dx
        >>> pprint(dsolve(genform, f(x),
        ... hint='1st_homogeneous_coeff_subs_dep_div_indep_Integral'))
                       f(x)
                       ----
                        x
                         /
                        |
                        |       -h(u1)
        log(x) = C1 +   |  ---------------- d(u1)
                        |  u1*h(u1) + g(u1)
                        |
                       /

    Where `u_1 h(u_1) + g(u_1) \ne 0` and `x \ne 0`.

    See also the docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`.

    Examples
    ========

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_subs_dep_div_indep', simplify=False))
                          /          3   \
                          |3*f(x)   f (x)|
                       log|------ + -----|
                          |  x         3 |
                          \           x  /
    log(x) = log(C1) - -------------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_subs_dep_div_indep"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.d = separatevars(self.d.subs(fx, self.y))
        self.e = separatevars(self.e.subs(fx, self.y))
        ordera = homogeneous_order(self.d, x, self.y)
        orderb = homogeneous_order(self.e, x, self.y)
        if ordera == orderb and ordera is not None:
            self.u = Dummy('u')
            if simplify((self.d + self.u*self.e).subs({x: 1, self.y: self.u})) != 0:
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        xarg = 0
        yarg = 0
        return [self.d, self.e, fx, x, self.u, self.u1, self.y, xarg, yarg]

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, fx, x, u, u1, y, xarg, yarg = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(
            (-e/(d + u1*e)).subs({x: 1, y: u1}),
            (u1, None, fx/x))
        sol = logcombine(Eq(log(x), int + log(C1)), force=True)
        gen_sol = sol.subs(fx, u).subs(((u, u - yarg), (x, x - xarg), (u, fx)))
        return [gen_sol]


class HomogeneousCoeffSubsIndepDivDep(SinglePatternODESolver):
    r"""
    Solves a 1st order differential equation with homogeneous coefficients
    using the substitution `u_2 = \frac{\text{<independent
    variable>}}{\text{<dependent variable>}}`.

    This is a differential equation

    .. math:: P(x, y) + Q(x, y) dy/dx = 0

    such that `P` and `Q` are homogeneous and of the same order.  A function
    `F(x, y)` is homogeneous of order `n` if `F(x t, y t) = t^n F(x, y)`.
    Equivalently, `F(x, y)` can be rewritten as `G(y/x)` or `H(x/y)`.  See
    also the docstring of :py:meth:`~sympy.solvers.ode.homogeneous_order`.

    If the coefficients `P` and `Q` in the differential equation above are
    homogeneous functions of the same order, then it can be shown that the
    substitution `x = u_2 y` (i.e. `u_2 = x/y`) will turn the differential
    equation into an equation separable in the variables `y` and `u_2`.  If
    `h(u_2)` is the function that results from making the substitution `u_2 =
    x/f(x)` on `P(x, f(x))` and `g(u_2)` is the function that results from the
    substitution on `Q(x, f(x))` in the differential equation `P(x, f(x)) +
    Q(x, f(x)) f'(x) = 0`, then the general solution is:

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f, g, h = map(Function, ['f', 'g', 'h'])
    >>> genform = g(x/f(x)) + h(x/f(x))*f(x).diff(x)
    >>> pprint(genform)
     / x  \    / x  \ d
    g|----| + h|----|*--(f(x))
     \f(x)/    \f(x)/ dx
    >>> pprint(dsolve(genform, f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral'))
                 x
                ----
                f(x)
                  /
                 |
                 |       -g(u1)
                 |  ---------------- d(u1)
                 |  u1*g(u1) + h(u1)
                 |
                /
    <BLANKLINE>
    f(x) = C1*e

    Where `u_1 g(u_1) + h(u_1) \ne 0` and `f(x) \ne 0`.

    See also the docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`.

    Examples
    ========

    >>> from sympy import Function, pprint, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep',
    ... simplify=False))
                             /   2     \
                             |3*x      |
                          log|----- + 1|
                             | 2       |
                             \f (x)    /
    log(f(x)) = log(C1) - --------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_subs_indep_div_dep"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        self.y = Dummy('y')
        x = self.ode_problem.sym
        self.d = separatevars(self.d.subs(fx, self.y))
        self.e = separatevars(self.e.subs(fx, self.y))
        ordera = homogeneous_order(self.d, x, self.y)
        orderb = homogeneous_order(self.e, x, self.y)
        if ordera == orderb and ordera is not None:
            self.u = Dummy('u')
            if simplify((self.e + self.u*self.d).subs({x: self.u, self.y: 1})) != 0:
                return True
            return False
        return False

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        xarg = 0
        yarg = 0
        return [self.d, self.e, fx, x, self.u, self.u1, self.y, xarg, yarg]

    def _get_general_solution(self, *, simplify_flag: bool = True):
        d, e, fx, x, u, u1, y, xarg, yarg = self._get_match_object()
        (C1,) = self.ode_problem.get_numbered_constants(num=1)
        int = Integral(simplify((-d/(e + u1*d)).subs({x: u1, y: 1})), (u1, None, x/fx)) # type: ignore
        sol = logcombine(Eq(log(fx), int + log(C1)), force=True)
        gen_sol = sol.subs(fx, u).subs(((u, u - yarg), (x, x - xarg), (u, fx)))
        return [gen_sol]


class HomogeneousCoeffBest(HomogeneousCoeffSubsIndepDivDep, HomogeneousCoeffSubsDepDivIndep):
    r"""
    Returns the best solution to an ODE from the two hints
    ``1st_homogeneous_coeff_subs_dep_div_indep`` and
    ``1st_homogeneous_coeff_subs_indep_div_dep``.

    This is as determined by :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity`.

    See the
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`
    and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`
    docstrings for more information on these hints.  Note that there is no
    ``ode_1st_homogeneous_coeff_best_Integral`` hint.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x), f(x),
    ... hint='1st_homogeneous_coeff_best', simplify=False))
                             /   2     \
                             |3*x      |
                          log|----- + 1|
                             | 2       |
                             \f (x)    /
    log(f(x)) = log(C1) - --------------
                                3

    References
    ==========

    - https://en.wikipedia.org/wiki/Homogeneous_differential_equation
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 59

    # indirect doctest

    """
    hint = "1st_homogeneous_coeff_best"
    has_integral = False
    order = [1]

    def _verify(self, fx):
        return HomogeneousCoeffSubsIndepDivDep._verify(self, fx) and \
               HomogeneousCoeffSubsDepDivIndep._verify(self, fx)

    def _get_general_solution(self, *, simplify_flag: bool = True):
        # There are two substitutions that solve the equation, u1=y/x and u2=x/y
        # # They produce different integrals, so try them both and see which
        # # one is easier
        sol1 = HomogeneousCoeffSubsIndepDivDep._get_general_solution(self)
        sol2 = HomogeneousCoeffSubsDepDivIndep._get_general_solution(self)
        fx = self.ode_problem.func
        if simplify_flag:
            sol1 = odesimp(self.ode_problem.eq, *sol1, fx, "1st_homogeneous_coeff_subs_indep_div_dep")
            sol2 = odesimp(self.ode_problem.eq, *sol2, fx, "1st_homogeneous_coeff_subs_dep_div_indep")
        # XXX: not simplify should be not simplify_flag. mypy correctly complains
        return min([sol1, sol2], key=lambda x: ode_sol_simplicity(x, fx, trysolving=not simplify)) # type: ignore


class LinearCoefficients(HomogeneousCoeffBest):
    r"""
    Solves a differential equation with linear coefficients.

    The general form of a differential equation with linear coefficients is

    .. math:: y' + F\left(\!\frac{a_1 x + b_1 y + c_1}{a_2 x + b_2 y +
                c_2}\!\right) = 0\text{,}

    where `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are constants and `a_1 b_2
    - a_2 b_1 \ne 0`.

    This can be solved by substituting:

    .. math:: x = x' + \frac{b_2 c_1 - b_1 c_2}{a_2 b_1 - a_1 b_2}

              y = y' + \frac{a_1 c_2 - a_2 c_1}{a_2 b_1 - a_1
                  b_2}\text{.}

    This substitution reduces the equation to a homogeneous differential
    equation.

    See Also
    ========
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffBest`
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`
    :obj:`sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep`

    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> df = f(x).diff(x)
    >>> eq = (x + f(x) + 1)*df + (f(x) - 6*x + 1)
    >>> dsolve(eq, hint='linear_coefficients')
    [Eq(f(x), -x - sqrt(C1 + 7*x**2) - 1), Eq(f(x), -x + sqrt(C1 + 7*x**2) - 1)]
    >>> pprint(dsolve(eq, hint='linear_coefficients'))
                      ___________                     ___________
                   /         2                     /         2
    [f(x) = -x - \/  C1 + 7*x   - 1, f(x) = -x + \/  C1 + 7*x   - 1]


    References
    ==========

    - Joel Moses, "Symbolic Integration - The Stormy Decade", Communications
      of the ACM, Volume 14, Number 8, August 1971, pp. 558
    """
    hint = "linear_coefficients"
    has_integral = True
    order = [1]

    def _wilds(self, f, x, order):
        d = Wild('d', exclude=[f(x).diff(x), f(x).diff(x, 2)])
        e = Wild('e', exclude=[f(x).diff(x)])
        return d, e

    def _equation(self, fx, x, order):
        d, e = self.wilds()
        return d + e*fx.diff(x)

    def _verify(self, fx):
        self.d, self.e = self.wilds_match()
        a, b = self.wilds()
        F = self.d/self.e
        x = self.ode_problem.sym
        params = self._linear_coeff_match(F, fx)
        if params:
            self.xarg, self.yarg = params
            u = Dummy('u')
            t = Dummy('t')
            self.y = Dummy('y')
            # Dummy substitution for df and f(x).
            dummy_eq = self.ode_problem.eq.subs(((fx.diff(x), t), (fx, u)))
            reps = ((x, x + self.xarg), (u, u + self.yarg), (t, fx.diff(x)), (u, fx))
            dummy_eq = simplify(dummy_eq.subs(reps))
            # get the re-cast values for e and d
            r2 = collect(expand(dummy_eq), [fx.diff(x), fx]).match(a*fx.diff(x) + b)
            if r2:
                self.d, self.e = r2[b], r2[a]
                orderd = homogeneous_order(self.d, x, fx)
                ordere = homogeneous_order(self.e, x, fx)
                if orderd == ordere and orderd is not None:
                    self.d = self.d.subs(fx, self.y)
                    self.e = self.e.subs(fx, self.y)
                    return True
                return False
            return False

    def _linear_coeff_match(self, expr, func):
        r"""
        Helper function to match hint ``linear_coefficients``.

        Matches the expression to the form `(a_1 x + b_1 f(x) + c_1)/(a_2 x + b_2
        f(x) + c_2)` where the following conditions hold:

        1. `a_1`, `b_1`, `c_1`, `a_2`, `b_2`, `c_2` are Rationals;
        2. `c_1` or `c_2` are not equal to zero;
        3. `a_2 b_1 - a_1 b_2` is not equal to zero.

        Return ``xarg``, ``yarg`` where

        1. ``xarg`` = `(b_2 c_1 - b_1 c_2)/(a_2 b_1 - a_1 b_2)`
        2. ``yarg`` = `(a_1 c_2 - a_2 c_1)/(a_2 b_1 - a_1 b_2)`


        Examples
        ========

        >>> from sympy import Function, sin
        >>> from sympy.abc import x
        >>> from sympy.solvers.ode.single import LinearCoefficients
        >>> f = Function('f')
        >>> eq = (-25*f(x) - 8*x + 62)/(4*f(x) + 11*x - 11)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (1/9, 22/9)
        >>> eq = sin((-5*f(x) - 8*x + 6)/(4*f(x) + x - 1))
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))
        (19/27, 2/27)
        >>> eq = sin(f(x)/x)
        >>> obj = LinearCoefficients(eq)
        >>> obj._linear_coeff_match(eq, f(x))

        """
        f = func.func
        x = func.args[0]
        def abc(eq):
            r'''
            Internal function of _linear_coeff_match
            that returns Rationals a, b, c
            if eq is a*x + b*f(x) + c, else None.
            '''
            eq = _mexpand(eq)
            c = eq.as_independent(x, f(x), as_Add=True)[0]
            if not c.is_Rational:
                return
            a = eq.coeff(x)
            if not a.is_Rational:
                return
            b = eq.coeff(f(x))
            if not b.is_Rational:
                return
            if eq == a*x + b*f(x) + c:
                return a, b, c

        def match(arg):
            r'''
            Internal function of _linear_coeff_match that returns Rationals a1,
            b1, c1, a2, b2, c2 and a2*b1 - a1*b2 of the expression (a1*x + b1*f(x)
            + c1)/(a2*x + b2*f(x) + c2) if one of c1 or c2 and a2*b1 - a1*b2 is
            non-zero, else None.
            '''
            n, d = arg.together().as_numer_denom()
            m = abc(n)
            if m is not None:
                a1, b1, c1 = m
                m = abc(d)
                if m is not None:
                    a2, b2, c2 = m
                    d = a2*b1 - a1*b2
                    if (c1 or c2) and d:
                        return a1, b1, c1, a2, b2, c2, d

        m = [fi.args[0] for fi in expr.atoms(Function) if fi.func != f and
            len(fi.args) == 1 and not fi.args[0].is_Function] or {expr}
        m1 = match(m.pop())
        if m1 and all(match(mi) == m1 for mi in m):
            a1, b1, c1, a2, b2, c2, denom = m1
            return (b2*c1 - b1*c2)/denom, (a1*c2 - a2*c1)/denom

    def _get_match_object(self):
        fx = self.ode_problem.func
        x = self.ode_problem.sym
        self.u1 = Dummy('u1')
        u = Dummy('u')
        return [self.d, self.e, fx, x, u, self.u1, self.y, self.xarg, self.yarg]


class NthOrderReducible(SingleODESolver):
    r"""
    Solves ODEs that only involve derivatives of the dependent variable using
    a substitution of the form `f^n(x) = g(x)`.

    For example any second order ODE of the form `f''(x) = h(f'(x), x)` can be
    transformed into a pair of 1st order ODEs `g'(x) = h(g(x), x)` and
    `f'(x) = g(x)`. Usually the 1st order ODE for `g` is easier to solve. If
    that gives an explicit solution for `g` then `f` is found simply by
    integration.


    Examples
    ========

    >>> from sympy import Function, dsolve, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = Eq(x*f(x).diff(x)**2 + f(x).diff(x, 2), 0)
    >>> dsolve(eq, f(x), hint='nth_order_reducible')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), C1 - sqrt(-1/C2)*log(-C2*sqrt(-1/C2) + x) + sqrt(-1/C2)*log(C2*sqrt(-1/C2) + x))

    """
    hint = "nth_order_reducible"
    has_integral = False

    def _matches(self):
        # Any ODE that can be solved with a substitution and
        # repeated integration e.g.:
        # `d^2/dx^2(y) + x*d/dx(y) = constant
        #f'(x) must be finite for this to work
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        x = self.ode_problem.sym
        r"""
        Matches any differential equation that can be rewritten with a smaller
        order. Only derivatives of ``func`` alone, wrt a single variable,
        are considered, and only in them should ``func`` appear.
        """
        # ODE only handles functions of 1 variable so this affirms that state
        assert len(func.args) == 1
        vc = [d.variable_count[0] for d in eq.atoms(Derivative)
            if d.expr == func and len(d.variable_count) == 1]
        ords = [c for v, c in vc if v == x]
        if len(ords) < 2:
            return False
        self.smallest = min(ords)
        # make sure func does not appear outside of derivatives
        D = Dummy()
        if eq.subs(func.diff(x, self.smallest), D).has(func):
            return False
        return True

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        n = self.smallest
        # get a unique function name for g
        names = [a.name for a in eq.atoms(AppliedUndef)]
        while True:
            name = Dummy().name
            if name not in names:
                g = Function(name)
                break
        w = f(x).diff(x, n)
        geq = eq.subs(w, g(x))
        gsol = dsolve(geq, g(x))

        if not isinstance(gsol, list):
            gsol = [gsol]

        # Might be multiple solutions to the reduced ODE:
        fsol = []
        for gsoli in gsol:
            fsoli = dsolve(gsoli.subs(g(x), w), f(x))  # or do integration n times
            fsol.append(fsoli)

        return fsol


class SecondHypergeometric(SingleODESolver):
    r"""
    Solves 2nd order linear differential equations.

    It computes special function solutions which can be expressed using the
    2F1, 1F1 or 0F1 hypergeometric functions.

    .. math:: y'' + A(x) y' + B(x) y = 0\text{,}

    where `A` and `B` are rational functions.

    These kinds of differential equations have solution of non-Liouvillian form.

    Given linear ODE can be obtained from 2F1 given by

    .. math:: (x^2 - x) y'' + ((a + b + 1) x - c) y' + b a y = 0\text{,}

    where {a, b, c} are arbitrary constants.

    Notes
    =====

    The algorithm should find any solution of the form

    .. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

    where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
    Currently only the 2F1 case is implemented in SymPy but the other cases are
    described in the paper and could be implemented in future (contributions
    welcome!).


    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = (x*x - x)*f(x).diff(x,2) + (5*x - 1)*f(x).diff(x) + 4*f(x)
    >>> pprint(dsolve(eq, f(x), '2nd_hypergeometric'))
                                        _
           /        /           4  \\  |_  /-1, -1 |  \
           |C1 + C2*|log(x) + -----||* |   |       | x|
           \        \         x + 1// 2  1 \  1    |  /
    f(x) = --------------------------------------------
                                    3
                             (x - 1)


    References
    ==========

    - "Non-Liouvillian solutions for second order linear ODEs" by L. Chan, E.S. Cheb-Terrab

    """
    hint = "2nd_hypergeometric"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        func = self.ode_problem.func
        r = match_2nd_hypergeometric(eq, func)
        self.match_object = None
        if r:
            A, B = r
            d = equivalence_hypergeometric(A, B, func)
            if d:
                if d['type'] == "2F1":
                    self.match_object = match_2nd_2F1_hypergeometric(d['I0'], d['k'], d['sing_point'], func)
                    if self.match_object is not None:
                        self.match_object.update({'A':A, 'B':B})
            # We can extend it for 1F1 and 0F1 type also.
        return self.match_object is not None

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        func = self.ode_problem.func
        if self.match_object['type'] == "2F1":
            sol = get_sol_2F1_hypergeometric(eq, func, self.match_object)
            if sol is None:
                raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                    + " the hypergeometric method")

        return [sol]


class NthLinearConstantCoeffHomogeneous(SingleODESolver):
    r"""
    Solves an `n`\th order linear homogeneous differential equation with
    constant coefficients.

    This is an equation of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x)
                + a_0 f(x) = 0\text{.}

    These equations can be solved in a general manner, by taking the roots of
    the characteristic equation `a_n m^n + a_{n-1} m^{n-1} + \cdots + a_1 m +
    a_0 = 0`.  The solution will then be the sum of `C_n x^i e^{r x}` terms,
    for each where `C_n` is an arbitrary constant, `r` is a root of the
    characteristic equation and `i` is one of each from 0 to the multiplicity
    of the root - 1 (for example, a root 3 of multiplicity 2 would create the
    terms `C_1 e^{3 x} + C_2 x e^{3 x}`).  The exponential is usually expanded
    for complex roots using Euler's equation `e^{I x} = \cos(x) + I \sin(x)`.
    Complex roots always come in conjugate pairs in polynomials with real
    coefficients, so the two roots will be represented (after simplifying the
    constants) as `e^{a x} \left(C_1 \cos(b x) + C_2 \sin(b x)\right)`.

    If SymPy cannot find exact roots to the characteristic equation, a
    :py:class:`~sympy.polys.rootoftools.ComplexRootOf` instance will be return
    instead.

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(f(x).diff(x, 5) + 10*f(x).diff(x) - 2*f(x), f(x),
    ... hint='nth_linear_constant_coeff_homogeneous')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), C5*exp(x*CRootOf(_x**5 + 10*_x - 2, 0))
    + (C1*sin(x*im(CRootOf(_x**5 + 10*_x - 2, 1)))
    + C2*cos(x*im(CRootOf(_x**5 + 10*_x - 2, 1))))*exp(x*re(CRootOf(_x**5 + 10*_x - 2, 1)))
    + (C3*sin(x*im(CRootOf(_x**5 + 10*_x - 2, 3)))
    + C4*cos(x*im(CRootOf(_x**5 + 10*_x - 2, 3))))*exp(x*re(CRootOf(_x**5 + 10*_x - 2, 3))))

    Note that because this method does not involve integration, there is no
    ``nth_linear_constant_coeff_homogeneous_Integral`` hint.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 4) + 2*f(x).diff(x, 3) -
    ... 2*f(x).diff(x, 2) - 6*f(x).diff(x) + 5*f(x), f(x),
    ... hint='nth_linear_constant_coeff_homogeneous'))
                        x                            -2*x
    f(x) = (C1 + C2*x)*e  + (C3*sin(x) + C4*cos(x))*e

    References
    ==========

    - https://en.wikipedia.org/wiki/Linear_differential_equation section:
      Nonhomogeneous_equation_with_constant_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 211

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_homogeneous"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if not self.r[-1]:
                return True
            else:
                return False
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        fx = self.ode_problem.func
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, fx, order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        gsol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        gsol = Eq(fx, gsol_rhs)
        if simplify_flag:
            gsol = _get_simplified_sol([gsol], fx, collectterms)

        return [gsol]


class NthLinearConstantCoeffVariationOfParameters(SingleODESolver):
    r"""
    Solves an `n`\th order linear differential equation with constant
    coefficients using the method of variation of parameters.

    This method works on any differential equations of the form

    .. math:: f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x) + a_0
                f(x) = P(x)\text{.}

    This method works by assuming that the particular solution takes the form

    .. math:: \sum_{x=1}^{n} c_i(x) y_i(x)\text{,}

    where `y_i` is the `i`\th solution to the homogeneous equation.  The
    solution is then solved using Wronskian's and Cramer's Rule.  The
    particular solution is given by

    .. math:: \sum_{x=1}^n \left( \int \frac{W_i(x)}{W(x)} \,dx
                \right) y_i(x) \text{,}

    where `W(x)` is the Wronskian of the fundamental system (the system of `n`
    linearly independent solutions to the homogeneous equation), and `W_i(x)`
    is the Wronskian of the fundamental system with the `i`\th column replaced
    with `[0, 0, \cdots, 0, P(x)]`.

    This method is general enough to solve any `n`\th order inhomogeneous
    linear differential equation with constant coefficients, but sometimes
    SymPy cannot simplify the Wronskian well enough to integrate it.  If this
    method hangs, try using the
    ``nth_linear_constant_coeff_variation_of_parameters_Integral`` hint and
    simplifying the integrals manually.  Also, prefer using
    ``nth_linear_constant_coeff_undetermined_coefficients`` when it
    applies, because it does not use integration, making it faster and more
    reliable.

    Warning, using simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters' in
    :py:meth:`~sympy.solvers.ode.dsolve` may cause it to hang, because it will
    not attempt to simplify the Wronskian before integrating.  It is
    recommended that you only use simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters_Integral' for this
    method, especially if the solution to the homogeneous equation has
    trigonometric functions in it.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, log
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 3) - 3*f(x).diff(x, 2) +
    ... 3*f(x).diff(x) - f(x) - exp(x)*log(x), f(x),
    ... hint='nth_linear_constant_coeff_variation_of_parameters'))
           /       /       /     x*log(x)   11*x\\\  x
    f(x) = |C1 + x*|C2 + x*|C3 + -------- - ----|||*e
           \       \       \        6        36 ///

    References
    ==========

    - https://en.wikipedia.org/wiki/Variation_of_parameters
    - https://planetmath.org/VariationOfParameters
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 233

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_variation_of_parameters"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)

        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if self.r[-1]:
                return True
            else:
                return False
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        homogen_sol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        homogen_sol = Eq(f(x), homogen_sol_rhs)
        homogen_sol = _solve_variation_of_parameters(eq, f(x), roots, homogen_sol, order, self.r, simplify_flag)
        if simplify_flag:
            homogen_sol = _get_simplified_sol([homogen_sol], f(x), collectterms)
        return [homogen_sol]


class NthLinearConstantCoeffUndeterminedCoefficients(SingleODESolver):
    r"""
    Solves an `n`\th order linear differential equation with constant
    coefficients using the method of undetermined coefficients.

    This method works on differential equations of the form

    .. math:: a_n f^{(n)}(x) + a_{n-1} f^{(n-1)}(x) + \cdots + a_1 f'(x)
                + a_0 f(x) = P(x)\text{,}

    where `P(x)` is a function that has a finite number of linearly
    independent derivatives.

    Functions that fit this requirement are finite sums functions of the form
    `a x^i e^{b x} \sin(c x + d)` or `a x^i e^{b x} \cos(c x + d)`, where `i`
    is a non-negative integer and `a`, `b`, `c`, and `d` are constants.  For
    example any polynomial in `x`, functions like `x^2 e^{2 x}`, `x \sin(x)`,
    and `e^x \cos(x)` can all be used.  Products of `\sin`'s and `\cos`'s have
    a finite number of derivatives, because they can be expanded into `\sin(a
    x)` and `\cos(b x)` terms.  However, SymPy currently cannot do that
    expansion, so you will need to manually rewrite the expression in terms of
    the above to use this method.  So, for example, you will need to manually
    convert `\sin^2(x)` into `(1 + \cos(2 x))/2` to properly apply the method
    of undetermined coefficients on it.

    This method works by creating a trial function from the expression and all
    of its linear independent derivatives and substituting them into the
    original ODE.  The coefficients for each term will be a system of linear
    equations, which are be solved for and substituted, giving the solution.
    If any of the trial functions are linearly dependent on the solution to
    the homogeneous equation, they are multiplied by sufficient `x` to make
    them linearly independent.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint, exp, cos
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x, 2) + 2*f(x).diff(x) + f(x) -
    ... 4*exp(-x)*x**2 + cos(2*x), f(x),
    ... hint='nth_linear_constant_coeff_undetermined_coefficients'))
           /       /      3\\
           |       |     x ||  -x   4*sin(2*x)   3*cos(2*x)
    f(x) = |C1 + x*|C2 + --||*e   - ---------- + ----------
           \       \     3 //           25           25

    References
    ==========

    - https://en.wikipedia.org/wiki/Method_of_undetermined_coefficients
    - M. Tenenbaum & H. Pollard, "Ordinary Differential Equations",
      Dover 1963, pp. 221

    # indirect doctest

    """
    hint = "nth_linear_constant_coeff_undetermined_coefficients"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        func = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        self.r = self.ode_problem.get_linear_coefficients(eq, func, order)
        does_match = False
        if order and self.r and not any(self.r[i].has(x) for i in self.r if i >= 0):
            if self.r[-1]:
                eq_homogeneous = Add(eq, -self.r[-1])
                undetcoeff = _undetermined_coefficients_match(self.r[-1], x, func, eq_homogeneous)
                if undetcoeff['test']:
                    self.trialset = undetcoeff['trialset']
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        roots, collectterms = _get_const_characteristic_eq_sols(self.r, f(x), order)
        # A generator of constants
        constants = self.ode_problem.get_numbered_constants(num=len(roots))
        homogen_sol_rhs = Add(*[i*j for (i, j) in zip(constants, roots)])
        homogen_sol = Eq(f(x), homogen_sol_rhs)
        self.r.update({'list': roots, 'sol': homogen_sol, 'simpliy_flag': simplify_flag})
        gsol = _solve_undetermined_coefficients(eq, f(x), order, self.r, self.trialset)
        if simplify_flag:
            gsol = _get_simplified_sol([gsol], f(x), collectterms)
        return [gsol]


class NthLinearEulerEqHomogeneous(SingleODESolver):
    r"""
    Solves an `n`\th order linear homogeneous variable-coefficient
    Cauchy-Euler equidimensional ordinary differential equation.

    This is an equation with form `0 = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    These equations can be solved in a general manner, by substituting
    solutions of the form `f(x) = x^r`, and deriving a characteristic equation
    for `r`.  When there are repeated roots, we include extra terms of the
    form `C_{r k} \ln^k(x) x^r`, where `C_{r k}` is an arbitrary integration
    constant, `r` is a root of the characteristic equation, and `k` ranges
    over the multiplicity of `r`.  In the cases where the roots are complex,
    solutions of the form `C_1 x^a \sin(b \log(x)) + C_2 x^a \cos(b \log(x))`
    are returned, based on expansions with Euler's formula.  The general
    solution is the sum of the terms found.  If SymPy cannot find exact roots
    to the characteristic equation, a
    :py:obj:`~.ComplexRootOf` instance will be returned
    instead.

    >>> from sympy import Function, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(4*x**2*f(x).diff(x, 2) + f(x), f(x),
    ... hint='nth_linear_euler_eq_homogeneous')
    ... # doctest: +NORMALIZE_WHITESPACE
    Eq(f(x), sqrt(x)*(C1 + C2*log(x)))

    Note that because this method does not involve integration, there is no
    ``nth_linear_euler_eq_homogeneous_Integral`` hint.

    The following is for internal use:

    - ``returns = 'sol'`` returns the solution to the ODE.
    - ``returns = 'list'`` returns a list of linearly independent solutions,
      corresponding to the fundamental solution set, for use with non
      homogeneous solution methods like variation of parameters and
      undetermined coefficients.  Note that, though the solutions should be
      linearly independent, this function does not explicitly check that.  You
      can do ``assert simplify(wronskian(sollist)) != 0`` to check for linear
      independence.  Also, ``assert len(sollist) == order`` will need to pass.
    - ``returns = 'both'``, return a dictionary ``{'sol': <solution to ODE>,
      'list': <list of linearly independent solutions>}``.

    Examples
    ========

    >>> from sympy import Function, dsolve, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = f(x).diff(x, 2)*x**2 - 4*f(x).diff(x)*x + 6*f(x)
    >>> pprint(dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_homogeneous'))
            2
    f(x) = x *(C1 + C2*x)

    References
    ==========

    - https://en.wikipedia.org/wiki/Cauchy%E2%80%93Euler_equation
    - C. Bender & S. Orszag, "Advanced Mathematical Methods for Scientists and
      Engineers", Springer 1999, pp. 12

    # indirect doctest

    """
    hint = "nth_linear_euler_eq_homogeneous"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if not self.r[-1]:
                does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        fx = self.ode_problem.func
        eq = self.ode_problem.eq
        homogen_sol = _get_euler_characteristic_eq_sols(eq, fx, self.r)[0]
        return [homogen_sol]


class NthLinearEulerEqNonhomogeneousVariationOfParameters(SingleODESolver):
    r"""
    Solves an `n`\th order linear non homogeneous Cauchy-Euler equidimensional
    ordinary differential equation using variation of parameters.

    This is an equation with form `g(x) = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    This method works by assuming that the particular solution takes the form

    .. math:: \sum_{x=1}^{n} c_i(x) y_i(x) {a_n} {x^n} \text{, }

    where `y_i` is the `i`\th solution to the homogeneous equation.  The
    solution is then solved using Wronskian's and Cramer's Rule.  The
    particular solution is given by multiplying eq given below with `a_n x^{n}`

    .. math:: \sum_{x=1}^n \left( \int \frac{W_i(x)}{W(x)} \, dx
                \right) y_i(x) \text{, }

    where `W(x)` is the Wronskian of the fundamental system (the system of `n`
    linearly independent solutions to the homogeneous equation), and `W_i(x)`
    is the Wronskian of the fundamental system with the `i`\th column replaced
    with `[0, 0, \cdots, 0, \frac{x^{- n}}{a_n} g{\left(x \right)}]`.

    This method is general enough to solve any `n`\th order inhomogeneous
    linear differential equation, but sometimes SymPy cannot simplify the
    Wronskian well enough to integrate it.  If this method hangs, try using the
    ``nth_linear_constant_coeff_variation_of_parameters_Integral`` hint and
    simplifying the integrals manually.  Also, prefer using
    ``nth_linear_constant_coeff_undetermined_coefficients`` when it
    applies, because it does not use integration, making it faster and more
    reliable.

    Warning, using simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters' in
    :py:meth:`~sympy.solvers.ode.dsolve` may cause it to hang, because it will
    not attempt to simplify the Wronskian before integrating.  It is
    recommended that you only use simplify=False with
    'nth_linear_constant_coeff_variation_of_parameters_Integral' for this
    method, especially if the solution to the homogeneous equation has
    trigonometric functions in it.

    Examples
    ========

    >>> from sympy import Function, dsolve, Derivative
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - x**4
    >>> dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_nonhomogeneous_variation_of_parameters').expand()
    Eq(f(x), C1*x + C2*x**2 + x**4/6)

    """
    hint = "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters"
    has_integral = True

    def _matches(self):
        eq = self.ode_problem.eq_preprocessed
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if self.r[-1]:
                does_match = True

        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        order = self.ode_problem.order
        homogen_sol, roots = _get_euler_characteristic_eq_sols(eq, f(x), self.r)
        self.r[-1] = self.r[-1]/self.r[order]
        sol = _solve_variation_of_parameters(eq, f(x), roots, homogen_sol, order, self.r, simplify_flag)

        return [Eq(f(x), homogen_sol.rhs + (sol.rhs - homogen_sol.rhs)*self.r[order])]


class NthLinearEulerEqNonhomogeneousUndeterminedCoefficients(SingleODESolver):
    r"""
    Solves an `n`\th order linear non homogeneous Cauchy-Euler equidimensional
    ordinary differential equation using undetermined coefficients.

    This is an equation with form `g(x) = a_0 f(x) + a_1 x f'(x) + a_2 x^2 f''(x)
    \cdots`.

    These equations can be solved in a general manner, by substituting
    solutions of the form `x = exp(t)`, and deriving a characteristic equation
    of form `g(exp(t)) = b_0 f(t) + b_1 f'(t) + b_2 f''(t) \cdots` which can
    be then solved by nth_linear_constant_coeff_undetermined_coefficients if
    g(exp(t)) has finite number of linearly independent derivatives.

    Functions that fit this requirement are finite sums functions of the form
    `a x^i e^{b x} \sin(c x + d)` or `a x^i e^{b x} \cos(c x + d)`, where `i`
    is a non-negative integer and `a`, `b`, `c`, and `d` are constants.  For
    example any polynomial in `x`, functions like `x^2 e^{2 x}`, `x \sin(x)`,
    and `e^x \cos(x)` can all be used.  Products of `\sin`'s and `\cos`'s have
    a finite number of derivatives, because they can be expanded into `\sin(a
    x)` and `\cos(b x)` terms.  However, SymPy currently cannot do that
    expansion, so you will need to manually rewrite the expression in terms of
    the above to use this method.  So, for example, you will need to manually
    convert `\sin^2(x)` into `(1 + \cos(2 x))/2` to properly apply the method
    of undetermined coefficients on it.

    After replacement of x by exp(t), this method works by creating a trial function
    from the expression and all of its linear independent derivatives and
    substituting them into the original ODE.  The coefficients for each term
    will be a system of linear equations, which are be solved for and
    substituted, giving the solution. If any of the trial functions are linearly
    dependent on the solution to the homogeneous equation, they are multiplied
    by sufficient `x` to make them linearly independent.

    Examples
    ========

    >>> from sympy import dsolve, Function, Derivative, log
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - log(x)
    >>> dsolve(eq, f(x),
    ... hint='nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients').expand()
    Eq(f(x), C1*x + C2*x**2 + log(x)/2 + 3/4)

    """
    hint = "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        match = self.ode_problem.get_linear_coefficients(eq, f(x), order)
        self.r = None
        does_match = False

        if order and match:
            coeff = match[order]
            factor = x**order / coeff
            self.r = {i: factor*match[i] for i in match}
        if self.r and all(_test_term(self.r[i], f(x), i) for i in
                          self.r if i >= 0):
            if self.r[-1]:
                e, re = posify(self.r[-1].subs(x, exp(x)))
                undetcoeff = _undetermined_coefficients_match(e.subs(re), x)
                if undetcoeff['test']:
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        chareq, eq, symbol = S.Zero, S.Zero, Dummy('x')
        for i in self.r.keys():
            if i >= 0:
                chareq += (self.r[i]*diff(x**symbol, x, i)*x**-symbol).expand()

        for i in range(1, degree(Poly(chareq, symbol))+1):
            eq += chareq.coeff(symbol**i)*diff(f(x), x, i)

        if chareq.as_coeff_add(symbol)[0]:
            eq += chareq.as_coeff_add(symbol)[0]*f(x)
        e, re = posify(self.r[-1].subs(x, exp(x)))
        eq += e.subs(re)

        self.const_undet_instance = NthLinearConstantCoeffUndeterminedCoefficients(SingleODEProblem(eq, f(x), x))
        sol = self.const_undet_instance.get_general_solution(simplify = simplify_flag)[0]
        sol = sol.subs(x, log(x)) # type: ignore
        sol = sol.subs(f(log(x)), f(x)).expand() # type: ignore

        return [sol]


class SecondLinearBessel(SingleODESolver):
    r"""
    Gives solution of the Bessel differential equation

    .. math :: x^2 \frac{d^2y}{dx^2} + x \frac{dy}{dx} y(x) + (x^2-n^2) y(x)

    if `n` is integer then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))`` as both the solutions are linearly independent else if
    `n` is a fraction then the solution is of the form ``Eq(f(x), C0 besselj(n,x)
    + C1 besselj(-n,x))`` which can also transform into ``Eq(f(x), C0 besselj(n,x)
    + C1 bessely(n,x))``.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import Symbol
    >>> v = Symbol('v', positive=True)
    >>> from sympy import dsolve, Function
    >>> f = Function('f')
    >>> y = f(x)
    >>> genform = x**2*y.diff(x, 2) + x*y.diff(x) + (x**2 - v**2)*y
    >>> dsolve(genform)
    Eq(f(x), C1*besselj(v, x) + C2*bessely(v, x))

    References
    ==========

    https://math24.net/bessel-differential-equation.html

    """
    hint = "2nd_linear_bessel"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f.diff(x)
        a = Wild('a', exclude=[f,df])
        b = Wild('b', exclude=[x, f,df])
        a4 = Wild('a4', exclude=[x,f,df])
        b4 = Wild('b4', exclude=[x,f,df])
        c4 = Wild('c4', exclude=[x,f,df])
        d4 = Wild('d4', exclude=[x,f,df])
        a3 = Wild('a3', exclude=[f, df, f.diff(x, 2)])
        b3 = Wild('b3', exclude=[f, df, f.diff(x, 2)])
        c3 = Wild('c3', exclude=[f, df, f.diff(x, 2)])
        deq = a3*(f.diff(x, 2)) + b3*df + c3*f
        r = collect(eq,
            [f.diff(x, 2), df, f]).match(deq)
        if order == 2 and r:
            if not all(r[key].is_polynomial() for key in r):
                n, d = eq.as_numer_denom()
                eq = expand(n)
                r = collect(eq,
                    [f.diff(x, 2), df, f]).match(deq)

        if r and r[a3] != 0:
            # leading coeff of f(x).diff(x, 2)
            coeff = factor(r[a3]).match(a4*(x-b)**b4)

            if coeff:
            # if coeff[b4] = 0 means constant coefficient
                if coeff[b4] == 0:
                    return False
                point = coeff[b]
            else:
                return False

            if point:
                r[a3] = simplify(r[a3].subs(x, x+point))
                r[b3] = simplify(r[b3].subs(x, x+point))
                r[c3] = simplify(r[c3].subs(x, x+point))

            # making a3 in the form of x**2
            r[a3] = cancel(r[a3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            r[b3] = cancel(r[b3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            r[c3] = cancel(r[c3]/(coeff[a4]*(x)**(-2+coeff[b4])))
            # checking if b3 is of form c*(x-b)
            coeff1 = factor(r[b3]).match(a4*(x))
            if coeff1 is None:
                return False
            # c3 maybe of very complex form so I am simply checking (a - b) form
            # if yes later I will match with the standard form of bessel in a and b
            # a, b are wild variable defined above.
            _coeff2 = expand(r[c3]).match(a - b)
            if _coeff2 is None:
                return False
            # matching with standard form for c3
            coeff2 = factor(_coeff2[a]).match(c4**2*(x)**(2*a4))
            if coeff2 is None:
                return False

            if _coeff2[b] == 0:
                coeff2[d4] = 0
            else:
                coeff2[d4] = factor(_coeff2[b]).match(d4**2)[d4]

            self.rn = {'n':coeff2[d4], 'a4':coeff2[c4], 'd4':coeff2[a4]}
            self.rn['c4'] = coeff1[a4]
            self.rn['b4'] = point
            return True
        return False

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        n = self.rn['n']
        a4 = self.rn['a4']
        c4 = self.rn['c4']
        d4 = self.rn['d4']
        b4 = self.rn['b4']
        n = sqrt(n**2 + Rational(1, 4)*(c4 - 1)**2)
        (C1, C2) = self.ode_problem.get_numbered_constants(num=2)
        return [Eq(f(x), ((x**(Rational(1-c4,2)))*(C1*besselj(n/d4,a4*x**d4/d4)
            + C2*bessely(n/d4,a4*x**d4/d4))).subs(x, x-b4))]


class SecondLinearAiry(SingleODESolver):
    r"""
    Gives solution of the Airy differential equation

    .. math :: \frac{d^2y}{dx^2} + (a + b x) y(x) = 0

    in terms of Airy special functions airyai and airybi.

    Examples
    ========

    >>> from sympy import dsolve, Function
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = f(x).diff(x, 2) - x*f(x)
    >>> dsolve(eq)
    Eq(f(x), C1*airyai(x) + C2*airybi(x))
    """
    hint = "2nd_linear_airy"
    has_integral = False

    def _matches(self):
        eq = self.ode_problem.eq_high_order_free
        f = self.ode_problem.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f.diff(x)
        a4 = Wild('a4', exclude=[x,f,df])
        b4 = Wild('b4', exclude=[x,f,df])
        match = self.ode_problem.get_linear_coefficients(eq, f, order)
        does_match = False
        if order == 2 and match and match[2] != 0:
            if match[1].is_zero:
                self.rn = cancel(match[0]/match[2]).match(a4+b4*x)
                if self.rn and self.rn[b4] != 0:
                    self.rn = {'b':self.rn[a4],'m':self.rn[b4]}
                    does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        f = self.ode_problem.func.func
        x = self.ode_problem.sym
        (C1, C2) = self.ode_problem.get_numbered_constants(num=2)
        b = self.rn['b']
        m = self.rn['m']
        if m.is_positive:
            arg = - b/cbrt(m)**2 - cbrt(m)*x
        elif m.is_negative:
            arg = - b/cbrt(-m)**2 + cbrt(-m)*x
        else:
            arg = - b/cbrt(-m)**2 + cbrt(-m)*x

        return [Eq(f(x), C1*airyai(arg) + C2*airybi(arg))]


class LieGroup(SingleODESolver):
    r"""
    This hint implements the Lie group method of solving first order differential
    equations. The aim is to convert the given differential equation from the
    given coordinate system into another coordinate system where it becomes
    invariant under the one-parameter Lie group of translations. The converted
    ODE can be easily solved by quadrature. It makes use of the
    :py:meth:`sympy.solvers.ode.infinitesimals` function which returns the
    infinitesimals of the transformation.

    The coordinates `r` and `s` can be found by solving the following Partial
    Differential Equations.

    .. math :: \xi\frac{\partial r}{\partial x} + \eta\frac{\partial r}{\partial y}
                  = 0

    .. math :: \xi\frac{\partial s}{\partial x} + \eta\frac{\partial s}{\partial y}
                  = 1

    The differential equation becomes separable in the new coordinate system

    .. math :: \frac{ds}{dr} = \frac{\frac{\partial s}{\partial x} +
                 h(x, y)\frac{\partial s}{\partial y}}{
                 \frac{\partial r}{\partial x} + h(x, y)\frac{\partial r}{\partial y}}

    After finding the solution by integration, it is then converted back to the original
    coordinate system by substituting `r` and `s` in terms of `x` and `y` again.

    Examples
    ========

    >>> from sympy import Function, dsolve, exp, pprint
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> pprint(dsolve(f(x).diff(x) + 2*x*f(x) - x*exp(-x**2), f(x),
    ... hint='lie_group'))
           /      2\    2
           |     x |  -x
    f(x) = |C1 + --|*e
           \     2 /


    References
    ==========

    - Solving differential equations by Symmetry Groups,
      John Starrett, pp. 1 - pp. 14

    """
    hint = "lie_group"
    has_integral = False

    def _has_additional_params(self):
        return 'xi' in self.ode_problem.params and 'eta' in self.ode_problem.params

    def _matches(self):
        eq = self.ode_problem.eq
        f = self.ode_problem.func.func
        order = self.ode_problem.order
        x = self.ode_problem.sym
        df = f(x).diff(x)
        y = Dummy('y')
        d = Wild('d', exclude=[df, f(x).diff(x, 2)])
        e = Wild('e', exclude=[df])
        does_match = False
        if self._has_additional_params() and order == 1:
            xi = self.ode_problem.params['xi']
            eta = self.ode_problem.params['eta']
            self.r3 = {'xi': xi, 'eta': eta}
            r = collect(eq, df, exact=True).match(d + e * df)
            if r:
                r['d'] = d
                r['e'] = e
                r['y'] = y
                r[d] = r[d].subs(f(x), y)
                r[e] = r[e].subs(f(x), y)
                self.r3.update(r)
            does_match = True
        return does_match

    def _get_general_solution(self, *, simplify_flag: bool = True):
        eq = self.ode_problem.eq
        x = self.ode_problem.sym
        func = self.ode_problem.func
        order = self.ode_problem.order
        df = func.diff(x)

        try:
            eqsol = solve(eq, df)
        except NotImplementedError:
            eqsol = []

        desols = []
        for s in eqsol:
            sol = _ode_lie_group(s, func, order, match=self.r3)
            if sol:
                desols.extend(sol)

        if desols == []:
            raise NotImplementedError("The given ODE " + str(eq) + " cannot be solved by"
                + " the lie group method")
        return desols


solver_map = {
    'factorable': Factorable,
    'nth_linear_constant_coeff_homogeneous': NthLinearConstantCoeffHomogeneous,
    'nth_linear_euler_eq_homogeneous': NthLinearEulerEqHomogeneous,
    'nth_linear_constant_coeff_undetermined_coefficients': NthLinearConstantCoeffUndeterminedCoefficients,
    'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients': NthLinearEulerEqNonhomogeneousUndeterminedCoefficients,
    'separable': Separable,
    '1st_exact': FirstExact,
    '1st_linear': FirstLinear,
    'Bernoulli': Bernoulli,
    'Riccati_special_minus2': RiccatiSpecial,
    '1st_rational_riccati': RationalRiccati,
    '1st_homogeneous_coeff_best': HomogeneousCoeffBest,
    '1st_homogeneous_coeff_subs_indep_div_dep': HomogeneousCoeffSubsIndepDivDep,
    '1st_homogeneous_coeff_subs_dep_div_indep': HomogeneousCoeffSubsDepDivIndep,
    'almost_linear': AlmostLinear,
    'linear_coefficients': LinearCoefficients,
    'separable_reduced': SeparableReduced,
    'nth_linear_constant_coeff_variation_of_parameters': NthLinearConstantCoeffVariationOfParameters,
    'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters': NthLinearEulerEqNonhomogeneousVariationOfParameters,
    'Liouville': Liouville,
    '2nd_linear_airy': SecondLinearAiry,
    '2nd_linear_bessel': SecondLinearBessel,
    '2nd_hypergeometric': SecondHypergeometric,
    'nth_order_reducible': NthOrderReducible,
    '2nd_nonlinear_autonomous_conserved': SecondNonlinearAutonomousConserved,
    'nth_algebraic': NthAlgebraic,
    'lie_group': LieGroup,
    }

# Avoid circular import:
from .ode import dsolve, ode_sol_simplicity, odesimp, homogeneous_order

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\sqlite\base.py ===
# dialects/sqlite/base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


r'''
.. dialect:: sqlite
    :name: SQLite
    :normal_support: 3.12+
    :best_effort: 3.7.16+

.. _sqlite_datetime:

Date and Time Types
-------------------

SQLite does not have built-in DATE, TIME, or DATETIME types, and pysqlite does
not provide out of the box functionality for translating values between Python
`datetime` objects and a SQLite-supported format. SQLAlchemy's own
:class:`~sqlalchemy.types.DateTime` and related types provide date formatting
and parsing functionality when SQLite is used. The implementation classes are
:class:`_sqlite.DATETIME`, :class:`_sqlite.DATE` and :class:`_sqlite.TIME`.
These types represent dates and times as ISO formatted strings, which also
nicely support ordering. There's no reliance on typical "libc" internals for
these functions so historical dates are fully supported.

Ensuring Text affinity
^^^^^^^^^^^^^^^^^^^^^^

The DDL rendered for these types is the standard ``DATE``, ``TIME``
and ``DATETIME`` indicators.    However, custom storage formats can also be
applied to these types.   When the
storage format is detected as containing no alpha characters, the DDL for
these types is rendered as ``DATE_CHAR``, ``TIME_CHAR``, and ``DATETIME_CHAR``,
so that the column continues to have textual affinity.

.. seealso::

    `Type Affinity <https://www.sqlite.org/datatype3.html#affinity>`_ -
    in the SQLite documentation

.. _sqlite_autoincrement:

SQLite Auto Incrementing Behavior
----------------------------------

Background on SQLite's autoincrement is at: https://sqlite.org/autoinc.html

Key concepts:

* SQLite has an implicit "auto increment" feature that takes place for any
  non-composite primary-key column that is specifically created using
  "INTEGER PRIMARY KEY" for the type + primary key.

* SQLite also has an explicit "AUTOINCREMENT" keyword, that is **not**
  equivalent to the implicit autoincrement feature; this keyword is not
  recommended for general use.  SQLAlchemy does not render this keyword
  unless a special SQLite-specific directive is used (see below).  However,
  it still requires that the column's type is named "INTEGER".

Using the AUTOINCREMENT Keyword
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To specifically render the AUTOINCREMENT keyword on the primary key column
when rendering DDL, add the flag ``sqlite_autoincrement=True`` to the Table
construct::

    Table(
        "sometable",
        metadata,
        Column("id", Integer, primary_key=True),
        sqlite_autoincrement=True,
    )

Allowing autoincrement behavior SQLAlchemy types other than Integer/INTEGER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SQLite's typing model is based on naming conventions.  Among other things, this
means that any type name which contains the substring ``"INT"`` will be
determined to be of "integer affinity".  A type named ``"BIGINT"``,
``"SPECIAL_INT"`` or even ``"XYZINTQPR"``, will be considered by SQLite to be
of "integer" affinity.  However, **the SQLite autoincrement feature, whether
implicitly or explicitly enabled, requires that the name of the column's type
is exactly the string "INTEGER"**.  Therefore, if an application uses a type
like :class:`.BigInteger` for a primary key, on SQLite this type will need to
be rendered as the name ``"INTEGER"`` when emitting the initial ``CREATE
TABLE`` statement in order for the autoincrement behavior to be available.

One approach to achieve this is to use :class:`.Integer` on SQLite
only using :meth:`.TypeEngine.with_variant`::

    table = Table(
        "my_table",
        metadata,
        Column(
            "id",
            BigInteger().with_variant(Integer, "sqlite"),
            primary_key=True,
        ),
    )

Another is to use a subclass of :class:`.BigInteger` that overrides its DDL
name to be ``INTEGER`` when compiled against SQLite::

    from sqlalchemy import BigInteger
    from sqlalchemy.ext.compiler import compiles


    class SLBigInteger(BigInteger):
        pass


    @compiles(SLBigInteger, "sqlite")
    def bi_c(element, compiler, **kw):
        return "INTEGER"


    @compiles(SLBigInteger)
    def bi_c(element, compiler, **kw):
        return compiler.visit_BIGINT(element, **kw)


    table = Table(
        "my_table", metadata, Column("id", SLBigInteger(), primary_key=True)
    )

.. seealso::

    :meth:`.TypeEngine.with_variant`

    :ref:`sqlalchemy.ext.compiler_toplevel`

    `Datatypes In SQLite Version 3 <https://sqlite.org/datatype3.html>`_

.. _sqlite_transactions:

Transactions with SQLite and the sqlite3 driver
-----------------------------------------------

As a file-based database, SQLite's approach to transactions differs from
traditional databases in many ways.  Additionally, the ``sqlite3`` driver
standard with Python (as well as the async version ``aiosqlite`` which builds
on top of it) has several quirks, workarounds, and API features in the
area of transaction control, all of which generally need to be addressed when
constructing a SQLAlchemy application that uses SQLite.

Legacy Transaction Mode with the sqlite3 driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most important aspect of transaction handling with the sqlite3 driver is
that it defaults (which will continue through Python 3.15 before being
removed in Python 3.16) to legacy transactional behavior which does
not strictly follow :pep:`249`.  The way in which the driver diverges from the
PEP is that it does not "begin" a transaction automatically as dictated by
:pep:`249` except in the case of DML statements, e.g. INSERT, UPDATE, and
DELETE.   Normally, :pep:`249` dictates that a BEGIN must be emitted upon
the first SQL statement of any kind, so that all subsequent operations will
be established within a transaction until ``connection.commit()`` has been
called.   The ``sqlite3`` driver, in an effort to be easier to use in
highly concurrent environments, skips this step for DQL (e.g. SELECT) statements,
and also skips it for DDL (e.g. CREATE TABLE etc.) statements for more legacy
reasons.  Statements such as SAVEPOINT are also skipped.

In modern versions of the ``sqlite3`` driver as of Python 3.12, this legacy
mode of operation is referred to as
`"legacy transaction control" <https://docs.python.org/3/library/sqlite3.html#sqlite3-transaction-control-isolation-level>`_, and is in
effect by default due to the ``Connection.autocommit`` parameter being set to
the constant ``sqlite3.LEGACY_TRANSACTION_CONTROL``.  Prior to Python 3.12,
the ``Connection.autocommit`` attribute did not exist.

The implications of legacy transaction mode include:

* **Incorrect support for transactional DDL** - statements like CREATE TABLE, ALTER TABLE,
  CREATE INDEX etc. will not automatically BEGIN a transaction if one were not
  started already, leading to the changes by each statement being
  "autocommitted" immediately unless BEGIN were otherwise emitted first.   Very
  old (pre Python 3.6) versions of SQLite would also force a COMMIT for these
  operations even if a transaction were present, however this is no longer the
  case.
* **SERIALIZABLE behavior not fully functional** - SQLite's transaction isolation
  behavior is normally consistent with SERIALIZABLE isolation, as it is a file-
  based system that locks the database file entirely for write operations,
  preventing COMMIT until all reader transactions (and associated file locks)
  have completed.  However, sqlite3's legacy transaction mode fails to emit BEGIN for SELECT
  statements, which causes these SELECT statements to no longer be "repeatable",
  failing one of the consistency guarantees of SERIALIZABLE.
* **Incorrect behavior for SAVEPOINT** - as the SAVEPOINT statement does not
  imply a BEGIN, a new SAVEPOINT emitted before a BEGIN will function on its
  own but fails to participate in the enclosing transaction, meaning a ROLLBACK
  of the transaction will not rollback elements that were part of a released
  savepoint.

Legacy transaction mode first existed in order to faciliate working around
SQLite's file locks.  Because SQLite relies upon whole-file locks, it is easy to
get "database is locked" errors, particularly when newer features like "write
ahead logging" are disabled.   This is a key reason why ``sqlite3``'s legacy
transaction mode is still the default mode of operation; disabling it will
produce behavior that is more susceptible to locked database errors.  However
note that **legacy transaction mode will no longer be the default** in a future
Python version (3.16 as of this writing).

.. _sqlite_enabling_transactions:

Enabling Non-Legacy SQLite Transactional Modes with the sqlite3 or aiosqlite driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current SQLAlchemy support allows either for setting the
``.Connection.autocommit`` attribute, most directly by using a
:func:`._sa.create_engine` parameter, or if on an older version of Python where
the attribute is not available, using event hooks to control the behavior of
BEGIN.

* **Enabling modern sqlite3 transaction control via the autocommit connect parameter** (Python 3.12 and above)

  To use SQLite in the mode described at `Transaction control via the autocommit attribute <https://docs.python.org/3/library/sqlite3.html#transaction-control-via-the-autocommit-attribute>`_,
  the most straightforward approach is to set the attribute to its recommended value
  of ``False`` at the connect level using :paramref:`_sa.create_engine.connect_args``::

    from sqlalchemy import create_engine

    engine = create_engine(
        "sqlite:///myfile.db", connect_args={"autocommit": False}
    )

  This parameter is also passed through when using the aiosqlite driver::

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(
        "sqlite+aiosqlite:///myfile.db", connect_args={"autocommit": False}
    )

  The parameter can also be set at the attribute level using the :meth:`.PoolEvents.connect`
  event hook, however this will only work for sqlite3, as aiosqlite does not yet expose this
  attribute on its ``Connection`` object::

    from sqlalchemy import create_engine, event

    engine = create_engine("sqlite:///myfile.db")


    @event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # enable autocommit=False mode
        dbapi_connection.autocommit = False

* **Using SQLAlchemy to emit BEGIN in lieu of SQLite's transaction control** (all Python versions, sqlite3 and aiosqlite)

  For older versions of ``sqlite3`` or for cross-compatiblity with older and
  newer versions, SQLAlchemy can also take over the job of transaction control.
  This is achieved by using the :meth:`.ConnectionEvents.begin` hook
  to emit the "BEGIN" command directly, while also disabling SQLite's control
  of this command using the :meth:`.PoolEvents.connect` event hook to set the
  ``Connection.isolation_level`` attribute to ``None``::


    from sqlalchemy import create_engine, event

    engine = create_engine("sqlite:///myfile.db")


    @event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # disable sqlite3's emitting of the BEGIN statement entirely.
        dbapi_connection.isolation_level = None


    @event.listens_for(engine, "begin")
    def do_begin(conn):
        # emit our own BEGIN.   sqlite3 still emits COMMIT/ROLLBACK correctly
        conn.exec_driver_sql("BEGIN")

  When using the asyncio variant ``aiosqlite``, refer to ``engine.sync_engine``
  as in the example below::

    from sqlalchemy import create_engine, event
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///myfile.db")


    @event.listens_for(engine.sync_engine, "connect")
    def do_connect(dbapi_connection, connection_record):
        # disable aiosqlite's emitting of the BEGIN statement entirely.
        dbapi_connection.isolation_level = None


    @event.listens_for(engine.sync_engine, "begin")
    def do_begin(conn):
        # emit our own BEGIN.  aiosqlite still emits COMMIT/ROLLBACK correctly
        conn.exec_driver_sql("BEGIN")

.. _sqlite_isolation_level:

Using SQLAlchemy's Driver Level AUTOCOMMIT Feature with SQLite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SQLAlchemy has a comprehensive database isolation feature with optional
autocommit support that is introduced in the section :ref:`dbapi_autocommit`.

For the ``sqlite3`` and ``aiosqlite`` drivers, SQLAlchemy only includes
built-in support for "AUTOCOMMIT".    Note that this mode is currently incompatible
with the non-legacy isolation mode hooks documented in the previous
section at :ref:`sqlite_enabling_transactions`.

To use the ``sqlite3`` driver with SQLAlchemy driver-level autocommit,
create an engine setting the :paramref:`_sa.create_engine.isolation_level`
parameter to "AUTOCOMMIT"::

    eng = create_engine("sqlite:///myfile.db", isolation_level="AUTOCOMMIT")

When using the above mode, any event hooks that set the sqlite3 ``Connection.autocommit``
parameter away from its default of ``sqlite3.LEGACY_TRANSACTION_CONTROL``
as well as hooks that emit ``BEGIN`` should be disabled.

Additional Reading for SQLite / sqlite3 transaction control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Links with important information on SQLite, the sqlite3 driver,
as well as long historical conversations on how things got to their current state:

* `Isolation in SQLite <https://www.sqlite.org/isolation.html>`_ - on the SQLite website
* `Transaction control <https://docs.python.org/3/library/sqlite3.html#transaction-control>`_ - describes the sqlite3 autocommit attribute as well
  as the legacy isolation_level attribute.
* `sqlite3 SELECT does not BEGIN a transaction, but should according to spec <https://github.com/python/cpython/issues/54133>`_ - imported Python standard library issue on github
* `sqlite3 module breaks transactions and potentially corrupts data <https://github.com/python/cpython/issues/54949>`_ - imported Python standard library issue on github


INSERT/UPDATE/DELETE...RETURNING
---------------------------------

The SQLite dialect supports SQLite 3.35's  ``INSERT|UPDATE|DELETE..RETURNING``
syntax.   ``INSERT..RETURNING`` may be used
automatically in some cases in order to fetch newly generated identifiers in
place of the traditional approach of using ``cursor.lastrowid``, however
``cursor.lastrowid`` is currently still preferred for simple single-statement
cases for its better performance.

To specify an explicit ``RETURNING`` clause, use the
:meth:`._UpdateBase.returning` method on a per-statement basis::

    # INSERT..RETURNING
    result = connection.execute(
        table.insert().values(name="foo").returning(table.c.col1, table.c.col2)
    )
    print(result.all())

    # UPDATE..RETURNING
    result = connection.execute(
        table.update()
        .where(table.c.name == "foo")
        .values(name="bar")
        .returning(table.c.col1, table.c.col2)
    )
    print(result.all())

    # DELETE..RETURNING
    result = connection.execute(
        table.delete()
        .where(table.c.name == "foo")
        .returning(table.c.col1, table.c.col2)
    )
    print(result.all())

.. versionadded:: 2.0  Added support for SQLite RETURNING


.. _sqlite_foreign_keys:

Foreign Key Support
-------------------

SQLite supports FOREIGN KEY syntax when emitting CREATE statements for tables,
however by default these constraints have no effect on the operation of the
table.

Constraint checking on SQLite has three prerequisites:

* At least version 3.6.19 of SQLite must be in use
* The SQLite library must be compiled *without* the SQLITE_OMIT_FOREIGN_KEY
  or SQLITE_OMIT_TRIGGER symbols enabled.
* The ``PRAGMA foreign_keys = ON`` statement must be emitted on all
  connections before use -- including the initial call to
  :meth:`sqlalchemy.schema.MetaData.create_all`.

SQLAlchemy allows for the ``PRAGMA`` statement to be emitted automatically for
new connections through the usage of events::

    from sqlalchemy.engine import Engine
    from sqlalchemy import event


    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

.. warning::

    When SQLite foreign keys are enabled, it is **not possible**
    to emit CREATE or DROP statements for tables that contain
    mutually-dependent foreign key constraints;
    to emit the DDL for these tables requires that ALTER TABLE be used to
    create or drop these constraints separately, for which SQLite has
    no support.

.. seealso::

    `SQLite Foreign Key Support <https://www.sqlite.org/foreignkeys.html>`_
    - on the SQLite web site.

    :ref:`event_toplevel` - SQLAlchemy event API.

    :ref:`use_alter` - more information on SQLAlchemy's facilities for handling
     mutually-dependent foreign key constraints.

.. _sqlite_on_conflict_ddl:

ON CONFLICT support for constraints
-----------------------------------

.. seealso:: This section describes the :term:`DDL` version of "ON CONFLICT" for
   SQLite, which occurs within a CREATE TABLE statement.  For "ON CONFLICT" as
   applied to an INSERT statement, see :ref:`sqlite_on_conflict_insert`.

SQLite supports a non-standard DDL clause known as ON CONFLICT which can be applied
to primary key, unique, check, and not null constraints.   In DDL, it is
rendered either within the "CONSTRAINT" clause or within the column definition
itself depending on the location of the target constraint.    To render this
clause within DDL, the extension parameter ``sqlite_on_conflict`` can be
specified with a string conflict resolution algorithm within the
:class:`.PrimaryKeyConstraint`, :class:`.UniqueConstraint`,
:class:`.CheckConstraint` objects.  Within the :class:`_schema.Column` object,
there
are individual parameters ``sqlite_on_conflict_not_null``,
``sqlite_on_conflict_primary_key``, ``sqlite_on_conflict_unique`` which each
correspond to the three types of relevant constraint types that can be
indicated from a :class:`_schema.Column` object.

.. seealso::

    `ON CONFLICT <https://www.sqlite.org/lang_conflict.html>`_ - in the SQLite
    documentation

.. versionadded:: 1.3


The ``sqlite_on_conflict`` parameters accept a  string argument which is just
the resolution name to be chosen, which on SQLite can be one of ROLLBACK,
ABORT, FAIL, IGNORE, and REPLACE.   For example, to add a UNIQUE constraint
that specifies the IGNORE algorithm::

    some_table = Table(
        "some_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("data", Integer),
        UniqueConstraint("id", "data", sqlite_on_conflict="IGNORE"),
    )

The above renders CREATE TABLE DDL as:

.. sourcecode:: sql

    CREATE TABLE some_table (
        id INTEGER NOT NULL,
        data INTEGER,
        PRIMARY KEY (id),
        UNIQUE (id, data) ON CONFLICT IGNORE
    )


When using the :paramref:`_schema.Column.unique`
flag to add a UNIQUE constraint
to a single column, the ``sqlite_on_conflict_unique`` parameter can
be added to the :class:`_schema.Column` as well, which will be added to the
UNIQUE constraint in the DDL::

    some_table = Table(
        "some_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column(
            "data", Integer, unique=True, sqlite_on_conflict_unique="IGNORE"
        ),
    )

rendering:

.. sourcecode:: sql

    CREATE TABLE some_table (
        id INTEGER NOT NULL,
        data INTEGER,
        PRIMARY KEY (id),
        UNIQUE (data) ON CONFLICT IGNORE
    )

To apply the FAIL algorithm for a NOT NULL constraint,
``sqlite_on_conflict_not_null`` is used::

    some_table = Table(
        "some_table",
        metadata,
        Column("id", Integer, primary_key=True),
        Column(
            "data", Integer, nullable=False, sqlite_on_conflict_not_null="FAIL"
        ),
    )

this renders the column inline ON CONFLICT phrase:

.. sourcecode:: sql

    CREATE TABLE some_table (
        id INTEGER NOT NULL,
        data INTEGER NOT NULL ON CONFLICT FAIL,
        PRIMARY KEY (id)
    )


Similarly, for an inline primary key, use ``sqlite_on_conflict_primary_key``::

    some_table = Table(
        "some_table",
        metadata,
        Column(
            "id",
            Integer,
            primary_key=True,
            sqlite_on_conflict_primary_key="FAIL",
        ),
    )

SQLAlchemy renders the PRIMARY KEY constraint separately, so the conflict
resolution algorithm is applied to the constraint itself:

.. sourcecode:: sql

    CREATE TABLE some_table (
        id INTEGER NOT NULL,
        PRIMARY KEY (id) ON CONFLICT FAIL
    )

.. _sqlite_on_conflict_insert:

INSERT...ON CONFLICT (Upsert)
-----------------------------

.. seealso:: This section describes the :term:`DML` version of "ON CONFLICT" for
   SQLite, which occurs within an INSERT statement.  For "ON CONFLICT" as
   applied to a CREATE TABLE statement, see :ref:`sqlite_on_conflict_ddl`.

From version 3.24.0 onwards, SQLite supports "upserts" (update or insert)
of rows into a table via the ``ON CONFLICT`` clause of the ``INSERT``
statement. A candidate row will only be inserted if that row does not violate
any unique or primary key constraints. In the case of a unique constraint violation, a
secondary action can occur which can be either "DO UPDATE", indicating that
the data in the target row should be updated, or "DO NOTHING", which indicates
to silently skip this row.

Conflicts are determined using columns that are part of existing unique
constraints and indexes.  These constraints are identified by stating the
columns and conditions that comprise the indexes.

SQLAlchemy provides ``ON CONFLICT`` support via the SQLite-specific
:func:`_sqlite.insert()` function, which provides
the generative methods :meth:`_sqlite.Insert.on_conflict_do_update`
and :meth:`_sqlite.Insert.on_conflict_do_nothing`:

.. sourcecode:: pycon+sql

    >>> from sqlalchemy.dialects.sqlite import insert

    >>> insert_stmt = insert(my_table).values(
    ...     id="some_existing_id", data="inserted value"
    ... )

    >>> do_update_stmt = insert_stmt.on_conflict_do_update(
    ...     index_elements=["id"], set_=dict(data="updated value")
    ... )

    >>> print(do_update_stmt)
    {printsql}INSERT INTO my_table (id, data) VALUES (?, ?)
    ON CONFLICT (id) DO UPDATE SET data = ?{stop}

    >>> do_nothing_stmt = insert_stmt.on_conflict_do_nothing(index_elements=["id"])

    >>> print(do_nothing_stmt)
    {printsql}INSERT INTO my_table (id, data) VALUES (?, ?)
    ON CONFLICT (id) DO NOTHING

.. versionadded:: 1.4

.. seealso::

    `Upsert
    <https://sqlite.org/lang_UPSERT.html>`_
    - in the SQLite documentation.


Specifying the Target
^^^^^^^^^^^^^^^^^^^^^

Both methods supply the "target" of the conflict using column inference:

* The :paramref:`_sqlite.Insert.on_conflict_do_update.index_elements` argument
  specifies a sequence containing string column names, :class:`_schema.Column`
  objects, and/or SQL expression elements, which would identify a unique index
  or unique constraint.

* When using :paramref:`_sqlite.Insert.on_conflict_do_update.index_elements`
  to infer an index, a partial index can be inferred by also specifying the
  :paramref:`_sqlite.Insert.on_conflict_do_update.index_where` parameter:

  .. sourcecode:: pycon+sql

        >>> stmt = insert(my_table).values(user_email="a@b.com", data="inserted data")

        >>> do_update_stmt = stmt.on_conflict_do_update(
        ...     index_elements=[my_table.c.user_email],
        ...     index_where=my_table.c.user_email.like("%@gmail.com"),
        ...     set_=dict(data=stmt.excluded.data),
        ... )

        >>> print(do_update_stmt)
        {printsql}INSERT INTO my_table (data, user_email) VALUES (?, ?)
        ON CONFLICT (user_email)
        WHERE user_email LIKE '%@gmail.com'
        DO UPDATE SET data = excluded.data

The SET Clause
^^^^^^^^^^^^^^^

``ON CONFLICT...DO UPDATE`` is used to perform an update of the already
existing row, using any combination of new values as well as values
from the proposed insertion. These values are specified using the
:paramref:`_sqlite.Insert.on_conflict_do_update.set_` parameter.  This
parameter accepts a dictionary which consists of direct values
for UPDATE:

.. sourcecode:: pycon+sql

    >>> stmt = insert(my_table).values(id="some_id", data="inserted value")

    >>> do_update_stmt = stmt.on_conflict_do_update(
    ...     index_elements=["id"], set_=dict(data="updated value")
    ... )

    >>> print(do_update_stmt)
    {printsql}INSERT INTO my_table (id, data) VALUES (?, ?)
    ON CONFLICT (id) DO UPDATE SET data = ?

.. warning::

    The :meth:`_sqlite.Insert.on_conflict_do_update` method does **not** take
    into account Python-side default UPDATE values or generation functions,
    e.g. those specified using :paramref:`_schema.Column.onupdate`. These
    values will not be exercised for an ON CONFLICT style of UPDATE, unless
    they are manually specified in the
    :paramref:`_sqlite.Insert.on_conflict_do_update.set_` dictionary.

Updating using the Excluded INSERT Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to refer to the proposed insertion row, the special alias
:attr:`~.sqlite.Insert.excluded` is available as an attribute on
the :class:`_sqlite.Insert` object; this object creates an "excluded." prefix
on a column, that informs the DO UPDATE to update the row with the value that
would have been inserted had the constraint not failed:

.. sourcecode:: pycon+sql

    >>> stmt = insert(my_table).values(
    ...     id="some_id", data="inserted value", author="jlh"
    ... )

    >>> do_update_stmt = stmt.on_conflict_do_update(
    ...     index_elements=["id"],
    ...     set_=dict(data="updated value", author=stmt.excluded.author),
    ... )

    >>> print(do_update_stmt)
    {printsql}INSERT INTO my_table (id, data, author) VALUES (?, ?, ?)
    ON CONFLICT (id) DO UPDATE SET data = ?, author = excluded.author

Additional WHERE Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`_sqlite.Insert.on_conflict_do_update` method also accepts
a WHERE clause using the :paramref:`_sqlite.Insert.on_conflict_do_update.where`
parameter, which will limit those rows which receive an UPDATE:

.. sourcecode:: pycon+sql

    >>> stmt = insert(my_table).values(
    ...     id="some_id", data="inserted value", author="jlh"
    ... )

    >>> on_update_stmt = stmt.on_conflict_do_update(
    ...     index_elements=["id"],
    ...     set_=dict(data="updated value", author=stmt.excluded.author),
    ...     where=(my_table.c.status == 2),
    ... )
    >>> print(on_update_stmt)
    {printsql}INSERT INTO my_table (id, data, author) VALUES (?, ?, ?)
    ON CONFLICT (id) DO UPDATE SET data = ?, author = excluded.author
    WHERE my_table.status = ?


Skipping Rows with DO NOTHING
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ON CONFLICT`` may be used to skip inserting a row entirely
if any conflict with a unique constraint occurs; below this is illustrated
using the :meth:`_sqlite.Insert.on_conflict_do_nothing` method:

.. sourcecode:: pycon+sql

    >>> stmt = insert(my_table).values(id="some_id", data="inserted value")
    >>> stmt = stmt.on_conflict_do_nothing(index_elements=["id"])
    >>> print(stmt)
    {printsql}INSERT INTO my_table (id, data) VALUES (?, ?) ON CONFLICT (id) DO NOTHING


If ``DO NOTHING`` is used without specifying any columns or constraint,
it has the effect of skipping the INSERT for any unique violation which
occurs:

.. sourcecode:: pycon+sql

    >>> stmt = insert(my_table).values(id="some_id", data="inserted value")
    >>> stmt = stmt.on_conflict_do_nothing()
    >>> print(stmt)
    {printsql}INSERT INTO my_table (id, data) VALUES (?, ?) ON CONFLICT DO NOTHING

.. _sqlite_type_reflection:

Type Reflection
---------------

SQLite types are unlike those of most other database backends, in that
the string name of the type usually does not correspond to a "type" in a
one-to-one fashion.  Instead, SQLite links per-column typing behavior
to one of five so-called "type affinities" based on a string matching
pattern for the type.

SQLAlchemy's reflection process, when inspecting types, uses a simple
lookup table to link the keywords returned to provided SQLAlchemy types.
This lookup table is present within the SQLite dialect as it is for all
other dialects.  However, the SQLite dialect has a different "fallback"
routine for when a particular type name is not located in the lookup map;
it instead implements the SQLite "type affinity" scheme located at
https://www.sqlite.org/datatype3.html section 2.1.

The provided typemap will make direct associations from an exact string
name match for the following types:

:class:`_types.BIGINT`, :class:`_types.BLOB`,
:class:`_types.BOOLEAN`, :class:`_types.BOOLEAN`,
:class:`_types.CHAR`, :class:`_types.DATE`,
:class:`_types.DATETIME`, :class:`_types.FLOAT`,
:class:`_types.DECIMAL`, :class:`_types.FLOAT`,
:class:`_types.INTEGER`, :class:`_types.INTEGER`,
:class:`_types.NUMERIC`, :class:`_types.REAL`,
:class:`_types.SMALLINT`, :class:`_types.TEXT`,
:class:`_types.TIME`, :class:`_types.TIMESTAMP`,
:class:`_types.VARCHAR`, :class:`_types.NVARCHAR`,
:class:`_types.NCHAR`

When a type name does not match one of the above types, the "type affinity"
lookup is used instead:

* :class:`_types.INTEGER` is returned if the type name includes the
  string ``INT``
* :class:`_types.TEXT` is returned if the type name includes the
  string ``CHAR``, ``CLOB`` or ``TEXT``
* :class:`_types.NullType` is returned if the type name includes the
  string ``BLOB``
* :class:`_types.REAL` is returned if the type name includes the string
  ``REAL``, ``FLOA`` or ``DOUB``.
* Otherwise, the :class:`_types.NUMERIC` type is used.

.. _sqlite_partial_index:

Partial Indexes
---------------

A partial index, e.g. one which uses a WHERE clause, can be specified
with the DDL system using the argument ``sqlite_where``::

    tbl = Table("testtbl", m, Column("data", Integer))
    idx = Index(
        "test_idx1",
        tbl.c.data,
        sqlite_where=and_(tbl.c.data > 5, tbl.c.data < 10),
    )

The index will be rendered at create time as:

.. sourcecode:: sql

    CREATE INDEX test_idx1 ON testtbl (data)
    WHERE data > 5 AND data < 10

.. _sqlite_dotted_column_names:

Dotted Column Names
-------------------

Using table or column names that explicitly have periods in them is
**not recommended**.   While this is generally a bad idea for relational
databases in general, as the dot is a syntactically significant character,
the SQLite driver up until version **3.10.0** of SQLite has a bug which
requires that SQLAlchemy filter out these dots in result sets.

The bug, entirely outside of SQLAlchemy, can be illustrated thusly::

    import sqlite3

    assert sqlite3.sqlite_version_info < (
        3,
        10,
        0,
    ), "bug is fixed in this version"

    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute("create table x (a integer, b integer)")
    cursor.execute("insert into x (a, b) values (1, 1)")
    cursor.execute("insert into x (a, b) values (2, 2)")

    cursor.execute("select x.a, x.b from x")
    assert [c[0] for c in cursor.description] == ["a", "b"]

    cursor.execute(
        """
        select x.a, x.b from x where a=1
        union
        select x.a, x.b from x where a=2
        """
    )
    assert [c[0] for c in cursor.description] == ["a", "b"], [
        c[0] for c in cursor.description
    ]

The second assertion fails:

.. sourcecode:: text

    Traceback (most recent call last):
      File "test.py", line 19, in <module>
        [c[0] for c in cursor.description]
    AssertionError: ['x.a', 'x.b']

Where above, the driver incorrectly reports the names of the columns
including the name of the table, which is entirely inconsistent vs.
when the UNION is not present.

SQLAlchemy relies upon column names being predictable in how they match
to the original statement, so the SQLAlchemy dialect has no choice but
to filter these out::


    from sqlalchemy import create_engine

    eng = create_engine("sqlite://")
    conn = eng.connect()

    conn.exec_driver_sql("create table x (a integer, b integer)")
    conn.exec_driver_sql("insert into x (a, b) values (1, 1)")
    conn.exec_driver_sql("insert into x (a, b) values (2, 2)")

    result = conn.exec_driver_sql("select x.a, x.b from x")
    assert result.keys() == ["a", "b"]

    result = conn.exec_driver_sql(
        """
        select x.a, x.b from x where a=1
        union
        select x.a, x.b from x where a=2
        """
    )
    assert result.keys() == ["a", "b"]

Note that above, even though SQLAlchemy filters out the dots, *both
names are still addressable*::

    >>> row = result.first()
    >>> row["a"]
    1
    >>> row["x.a"]
    1
    >>> row["b"]
    1
    >>> row["x.b"]
    1

Therefore, the workaround applied by SQLAlchemy only impacts
:meth:`_engine.CursorResult.keys` and :meth:`.Row.keys()` in the public API. In
the very specific case where an application is forced to use column names that
contain dots, and the functionality of :meth:`_engine.CursorResult.keys` and
:meth:`.Row.keys()` is required to return these dotted names unmodified,
the ``sqlite_raw_colnames`` execution option may be provided, either on a
per-:class:`_engine.Connection` basis::

    result = conn.execution_options(sqlite_raw_colnames=True).exec_driver_sql(
        """
        select x.a, x.b from x where a=1
        union
        select x.a, x.b from x where a=2
        """
    )
    assert result.keys() == ["x.a", "x.b"]

or on a per-:class:`_engine.Engine` basis::

    engine = create_engine(
        "sqlite://", execution_options={"sqlite_raw_colnames": True}
    )

When using the per-:class:`_engine.Engine` execution option, note that
**Core and ORM queries that use UNION may not function properly**.

SQLite-specific table options
-----------------------------

One option for CREATE TABLE is supported directly by the SQLite
dialect in conjunction with the :class:`_schema.Table` construct:

* ``WITHOUT ROWID``::

    Table("some_table", metadata, ..., sqlite_with_rowid=False)

*
  ``STRICT``::

    Table("some_table", metadata, ..., sqlite_strict=True)

  .. versionadded:: 2.0.37

.. seealso::

    `SQLite CREATE TABLE options
    <https://www.sqlite.org/lang_createtable.html>`_

.. _sqlite_include_internal:

Reflecting internal schema tables
----------------------------------

Reflection methods that return lists of tables will omit so-called
"SQLite internal schema object" names, which are considered by SQLite
as any object name that is prefixed with ``sqlite_``.  An example of
such an object is the ``sqlite_sequence`` table that's generated when
the ``AUTOINCREMENT`` column parameter is used.   In order to return
these objects, the parameter ``sqlite_include_internal=True`` may be
passed to methods such as :meth:`_schema.MetaData.reflect` or
:meth:`.Inspector.get_table_names`.

.. versionadded:: 2.0  Added the ``sqlite_include_internal=True`` parameter.
   Previously, these tables were not ignored by SQLAlchemy reflection
   methods.

.. note::

    The ``sqlite_include_internal`` parameter does not refer to the
    "system" tables that are present in schemas such as ``sqlite_master``.

.. seealso::

    `SQLite Internal Schema Objects <https://www.sqlite.org/fileformat2.html#intschema>`_ - in the SQLite
    documentation.

'''  # noqa
from __future__ import annotations

import datetime
import numbers
import re
from typing import Optional

from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa


class _SQliteJson(JSON):
    def result_processor(self, dialect, coltype):
        default_processor = super().result_processor(dialect, coltype)

        def process(value):
            try:
                return default_processor(value)
            except TypeError:
                if isinstance(value, numbers.Number):
                    return value
                else:
                    raise

        return process


class _DateTimeMixin:
    _reg = None
    _storage_format = None

    def __init__(self, storage_format=None, regexp=None, **kw):
        super().__init__(**kw)
        if regexp is not None:
            self._reg = re.compile(regexp)
        if storage_format is not None:
            self._storage_format = storage_format

    @property
    def format_is_text_affinity(self):
        """return True if the storage format will automatically imply
        a TEXT affinity.

        If the storage format contains no non-numeric characters,
        it will imply a NUMERIC storage format on SQLite; in this case,
        the type will generate its DDL as DATE_CHAR, DATETIME_CHAR,
        TIME_CHAR.

        """
        spec = self._storage_format % {
            "year": 0,
            "month": 0,
            "day": 0,
            "hour": 0,
            "minute": 0,
            "second": 0,
            "microsecond": 0,
        }
        return bool(re.search(r"[^0-9]", spec))

    def adapt(self, cls, **kw):
        if issubclass(cls, _DateTimeMixin):
            if self._storage_format:
                kw["storage_format"] = self._storage_format
            if self._reg:
                kw["regexp"] = self._reg
        return super().adapt(cls, **kw)

    def literal_processor(self, dialect):
        bp = self.bind_processor(dialect)

        def process(value):
            return "'%s'" % bp(value)

        return process


class DATETIME(_DateTimeMixin, sqltypes.DateTime):
    r"""Represent a Python datetime object in SQLite using a string.

    The default string storage format is::

        "%(year)04d-%(month)02d-%(day)02d %(hour)02d:%(minute)02d:%(second)02d.%(microsecond)06d"

    e.g.:

    .. sourcecode:: text

        2021-03-15 12:05:57.105542

    The incoming storage format is by default parsed using the
    Python ``datetime.fromisoformat()`` function.

    .. versionchanged:: 2.0  ``datetime.fromisoformat()`` is used for default
       datetime string parsing.

    The storage format can be customized to some degree using the
    ``storage_format`` and ``regexp`` parameters, such as::

        import re
        from sqlalchemy.dialects.sqlite import DATETIME

        dt = DATETIME(
            storage_format=(
                "%(year)04d/%(month)02d/%(day)02d %(hour)02d:%(minute)02d:%(second)02d"
            ),
            regexp=r"(\d+)/(\d+)/(\d+) (\d+)-(\d+)-(\d+)",
        )

    :param truncate_microseconds: when ``True`` microseconds will be truncated
     from the datetime. Can't be specified together with ``storage_format``
     or ``regexp``.

    :param storage_format: format string which will be applied to the dict
     with keys year, month, day, hour, minute, second, and microsecond.

    :param regexp: regular expression which will be applied to incoming result
     rows, replacing the use of ``datetime.fromisoformat()`` to parse incoming
     strings. If the regexp contains named groups, the resulting match dict is
     applied to the Python datetime() constructor as keyword arguments.
     Otherwise, if positional groups are used, the datetime() constructor
     is called with positional arguments via
     ``*map(int, match_obj.groups(0))``.

    """  # noqa

    _storage_format = (
        "%(year)04d-%(month)02d-%(day)02d "
        "%(hour)02d:%(minute)02d:%(second)02d.%(microsecond)06d"
    )

    def __init__(self, *args, **kwargs):
        truncate_microseconds = kwargs.pop("truncate_microseconds", False)
        super().__init__(*args, **kwargs)
        if truncate_microseconds:
            assert "storage_format" not in kwargs, (
                "You can specify only "
                "one of truncate_microseconds or storage_format."
            )
            assert "regexp" not in kwargs, (
                "You can specify only one of "
                "truncate_microseconds or regexp."
            )
            self._storage_format = (
                "%(year)04d-%(month)02d-%(day)02d "
                "%(hour)02d:%(minute)02d:%(second)02d"
            )

    def bind_processor(self, dialect):
        datetime_datetime = datetime.datetime
        datetime_date = datetime.date
        format_ = self._storage_format

        def process(value):
            if value is None:
                return None
            elif isinstance(value, datetime_datetime):
                return format_ % {
                    "year": value.year,
                    "month": value.month,
                    "day": value.day,
                    "hour": value.hour,
                    "minute": value.minute,
                    "second": value.second,
                    "microsecond": value.microsecond,
                }
            elif isinstance(value, datetime_date):
                return format_ % {
                    "year": value.year,
                    "month": value.month,
                    "day": value.day,
                    "hour": 0,
                    "minute": 0,
                    "second": 0,
                    "microsecond": 0,
                }
            else:
                raise TypeError(
                    "SQLite DateTime type only accepts Python "
                    "datetime and date objects as input."
                )

        return process

    def result_processor(self, dialect, coltype):
        if self._reg:
            return processors.str_to_datetime_processor_factory(
                self._reg, datetime.datetime
            )
        else:
            return processors.str_to_datetime


class DATE(_DateTimeMixin, sqltypes.Date):
    r"""Represent a Python date object in SQLite using a string.

    The default string storage format is::

        "%(year)04d-%(month)02d-%(day)02d"

    e.g.:

    .. sourcecode:: text

        2011-03-15

    The incoming storage format is by default parsed using the
    Python ``date.fromisoformat()`` function.

    .. versionchanged:: 2.0  ``date.fromisoformat()`` is used for default
       date string parsing.


    The storage format can be customized to some degree using the
    ``storage_format`` and ``regexp`` parameters, such as::

        import re
        from sqlalchemy.dialects.sqlite import DATE

        d = DATE(
            storage_format="%(month)02d/%(day)02d/%(year)04d",
            regexp=re.compile("(?P<month>\d+)/(?P<day>\d+)/(?P<year>\d+)"),
        )

    :param storage_format: format string which will be applied to the
     dict with keys year, month, and day.

    :param regexp: regular expression which will be applied to
     incoming result rows, replacing the use of ``date.fromisoformat()`` to
     parse incoming strings. If the regexp contains named groups, the resulting
     match dict is applied to the Python date() constructor as keyword
     arguments. Otherwise, if positional groups are used, the date()
     constructor is called with positional arguments via
     ``*map(int, match_obj.groups(0))``.

    """

    _storage_format = "%(year)04d-%(month)02d-%(day)02d"

    def bind_processor(self, dialect):
        datetime_date = datetime.date
        format_ = self._storage_format

        def process(value):
            if value is None:
                return None
            elif isinstance(value, datetime_date):
                return format_ % {
                    "year": value.year,
                    "month": value.month,
                    "day": value.day,
                }
            else:
                raise TypeError(
                    "SQLite Date type only accepts Python "
                    "date objects as input."
                )

        return process

    def result_processor(self, dialect, coltype):
        if self._reg:
            return processors.str_to_datetime_processor_factory(
                self._reg, datetime.date
            )
        else:
            return processors.str_to_date


class TIME(_DateTimeMixin, sqltypes.Time):
    r"""Represent a Python time object in SQLite using a string.

    The default string storage format is::

        "%(hour)02d:%(minute)02d:%(second)02d.%(microsecond)06d"

    e.g.:

    .. sourcecode:: text

        12:05:57.10558

    The incoming storage format is by default parsed using the
    Python ``time.fromisoformat()`` function.

    .. versionchanged:: 2.0  ``time.fromisoformat()`` is used for default
       time string parsing.

    The storage format can be customized to some degree using the
    ``storage_format`` and ``regexp`` parameters, such as::

        import re
        from sqlalchemy.dialects.sqlite import TIME

        t = TIME(
            storage_format="%(hour)02d-%(minute)02d-%(second)02d-%(microsecond)06d",
            regexp=re.compile("(\d+)-(\d+)-(\d+)-(?:-(\d+))?"),
        )

    :param truncate_microseconds: when ``True`` microseconds will be truncated
     from the time. Can't be specified together with ``storage_format``
     or ``regexp``.

    :param storage_format: format string which will be applied to the dict
     with keys hour, minute, second, and microsecond.

    :param regexp: regular expression which will be applied to incoming result
     rows, replacing the use of ``datetime.fromisoformat()`` to parse incoming
     strings. If the regexp contains named groups, the resulting match dict is
     applied to the Python time() constructor as keyword arguments. Otherwise,
     if positional groups are used, the time() constructor is called with
     positional arguments via ``*map(int, match_obj.groups(0))``.

    """

    _storage_format = "%(hour)02d:%(minute)02d:%(second)02d.%(microsecond)06d"

    def __init__(self, *args, **kwargs):
        truncate_microseconds = kwargs.pop("truncate_microseconds", False)
        super().__init__(*args, **kwargs)
        if truncate_microseconds:
            assert "storage_format" not in kwargs, (
                "You can specify only "
                "one of truncate_microseconds or storage_format."
            )
            assert "regexp" not in kwargs, (
                "You can specify only one of "
                "truncate_microseconds or regexp."
            )
            self._storage_format = "%(hour)02d:%(minute)02d:%(second)02d"

    def bind_processor(self, dialect):
        datetime_time = datetime.time
        format_ = self._storage_format

        def process(value):
            if value is None:
                return None
            elif isinstance(value, datetime_time):
                return format_ % {
                    "hour": value.hour,
                    "minute": value.minute,
                    "second": value.second,
                    "microsecond": value.microsecond,
                }
            else:
                raise TypeError(
                    "SQLite Time type only accepts Python "
                    "time objects as input."
                )

        return process

    def result_processor(self, dialect, coltype):
        if self._reg:
            return processors.str_to_datetime_processor_factory(
                self._reg, datetime.time
            )
        else:
            return processors.str_to_time


colspecs = {
    sqltypes.Date: DATE,
    sqltypes.DateTime: DATETIME,
    sqltypes.JSON: _SQliteJson,
    sqltypes.JSON.JSONIndexType: JSONIndexType,
    sqltypes.JSON.JSONPathType: JSONPathType,
    sqltypes.Time: TIME,
}

ischema_names = {
    "BIGINT": sqltypes.BIGINT,
    "BLOB": sqltypes.BLOB,
    "BOOL": sqltypes.BOOLEAN,
    "BOOLEAN": sqltypes.BOOLEAN,
    "CHAR": sqltypes.CHAR,
    "DATE": sqltypes.DATE,
    "DATE_CHAR": sqltypes.DATE,
    "DATETIME": sqltypes.DATETIME,
    "DATETIME_CHAR": sqltypes.DATETIME,
    "DOUBLE": sqltypes.DOUBLE,
    "DECIMAL": sqltypes.DECIMAL,
    "FLOAT": sqltypes.FLOAT,
    "INT": sqltypes.INTEGER,
    "INTEGER": sqltypes.INTEGER,
    "JSON": JSON,
    "NUMERIC": sqltypes.NUMERIC,
    "REAL": sqltypes.REAL,
    "SMALLINT": sqltypes.SMALLINT,
    "TEXT": sqltypes.TEXT,
    "TIME": sqltypes.TIME,
    "TIME_CHAR": sqltypes.TIME,
    "TIMESTAMP": sqltypes.TIMESTAMP,
    "VARCHAR": sqltypes.VARCHAR,
    "NVARCHAR": sqltypes.NVARCHAR,
    "NCHAR": sqltypes.NCHAR,
}


class SQLiteCompiler(compiler.SQLCompiler):
    extract_map = util.update_copy(
        compiler.SQLCompiler.extract_map,
        {
            "month": "%m",
            "day": "%d",
            "year": "%Y",
            "second": "%S",
            "hour": "%H",
            "doy": "%j",
            "minute": "%M",
            "epoch": "%s",
            "dow": "%w",
            "week": "%W",
        },
    )

    def visit_truediv_binary(self, binary, operator, **kw):
        return (
            self.process(binary.left, **kw)
            + " / "
            + "(%s + 0.0)" % self.process(binary.right, **kw)
        )

    def visit_now_func(self, fn, **kw):
        return "CURRENT_TIMESTAMP"

    def visit_localtimestamp_func(self, func, **kw):
        return "DATETIME(CURRENT_TIMESTAMP, 'localtime')"

    def visit_true(self, expr, **kw):
        return "1"

    def visit_false(self, expr, **kw):
        return "0"

    def visit_char_length_func(self, fn, **kw):
        return "length%s" % self.function_argspec(fn)

    def visit_aggregate_strings_func(self, fn, **kw):
        return "group_concat%s" % self.function_argspec(fn)

    def visit_cast(self, cast, **kwargs):
        if self.dialect.supports_cast:
            return super().visit_cast(cast, **kwargs)
        else:
            return self.process(cast.clause, **kwargs)

    def visit_extract(self, extract, **kw):
        try:
            return "CAST(STRFTIME('%s', %s) AS INTEGER)" % (
                self.extract_map[extract.field],
                self.process(extract.expr, **kw),
            )
        except KeyError as err:
            raise exc.CompileError(
                "%s is not a valid extract argument." % extract.field
            ) from err

    def returning_clause(
        self,
        stmt,
        returning_cols,
        *,
        populate_result_map,
        **kw,
    ):
        kw["include_table"] = False
        return super().returning_clause(
            stmt, returning_cols, populate_result_map=populate_result_map, **kw
        )

    def limit_clause(self, select, **kw):
        text = ""
        if select._limit_clause is not None:
            text += "\n LIMIT " + self.process(select._limit_clause, **kw)
        if select._offset_clause is not None:
            if select._limit_clause is None:
                text += "\n LIMIT " + self.process(sql.literal(-1))
            text += " OFFSET " + self.process(select._offset_clause, **kw)
        else:
            text += " OFFSET " + self.process(sql.literal(0), **kw)
        return text

    def for_update_clause(self, select, **kw):
        # sqlite has no "FOR UPDATE" AFAICT
        return ""

    def update_from_clause(
        self, update_stmt, from_table, extra_froms, from_hints, **kw
    ):
        kw["asfrom"] = True
        return "FROM " + ", ".join(
            t._compiler_dispatch(self, fromhints=from_hints, **kw)
            for t in extra_froms
        )

    def visit_is_distinct_from_binary(self, binary, operator, **kw):
        return "%s IS NOT %s" % (
            self.process(binary.left),
            self.process(binary.right),
        )

    def visit_is_not_distinct_from_binary(self, binary, operator, **kw):
        return "%s IS %s" % (
            self.process(binary.left),
            self.process(binary.right),
        )

    def visit_json_getitem_op_binary(self, binary, operator, **kw):
        if binary.type._type_affinity is sqltypes.JSON:
            expr = "JSON_QUOTE(JSON_EXTRACT(%s, %s))"
        else:
            expr = "JSON_EXTRACT(%s, %s)"

        return expr % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    def visit_json_path_getitem_op_binary(self, binary, operator, **kw):
        if binary.type._type_affinity is sqltypes.JSON:
            expr = "JSON_QUOTE(JSON_EXTRACT(%s, %s))"
        else:
            expr = "JSON_EXTRACT(%s, %s)"

        return expr % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    def visit_empty_set_op_expr(self, type_, expand_op, **kw):
        # slightly old SQLite versions don't seem to be able to handle
        # the empty set impl
        return self.visit_empty_set_expr(type_)

    def visit_empty_set_expr(self, element_types, **kw):
        return "SELECT %s FROM (SELECT %s) WHERE 1!=1" % (
            ", ".join("1" for type_ in element_types or [INTEGER()]),
            ", ".join("1" for type_ in element_types or [INTEGER()]),
        )

    def visit_regexp_match_op_binary(self, binary, operator, **kw):
        return self._generate_generic_binary(binary, " REGEXP ", **kw)

    def visit_not_regexp_match_op_binary(self, binary, operator, **kw):
        return self._generate_generic_binary(binary, " NOT REGEXP ", **kw)

    def _on_conflict_target(self, clause, **kw):
        if clause.inferred_target_elements is not None:
            target_text = "(%s)" % ", ".join(
                (
                    self.preparer.quote(c)
                    if isinstance(c, str)
                    else self.process(c, include_table=False, use_schema=False)
                )
                for c in clause.inferred_target_elements
            )
            if clause.inferred_target_whereclause is not None:
                target_text += " WHERE %s" % self.process(
                    clause.inferred_target_whereclause,
                    include_table=False,
                    use_schema=False,
                    literal_execute=True,
                )

        else:
            target_text = ""

        return target_text

    def visit_on_conflict_do_nothing(self, on_conflict, **kw):
        target_text = self._on_conflict_target(on_conflict, **kw)

        if target_text:
            return "ON CONFLICT %s DO NOTHING" % target_text
        else:
            return "ON CONFLICT DO NOTHING"

    def visit_on_conflict_do_update(self, on_conflict, **kw):
        clause = on_conflict

        target_text = self._on_conflict_target(on_conflict, **kw)

        action_set_ops = []

        set_parameters = dict(clause.update_values_to_set)
        # create a list of column assignment clauses as tuples

        insert_statement = self.stack[-1]["selectable"]
        cols = insert_statement.table.c
        for c in cols:
            col_key = c.key

            if col_key in set_parameters:
                value = set_parameters.pop(col_key)
            elif c in set_parameters:
                value = set_parameters.pop(c)
            else:
                continue

            if coercions._is_literal(value):
                value = elements.BindParameter(None, value, type_=c.type)

            else:
                if (
                    isinstance(value, elements.BindParameter)
                    and value.type._isnull
                ):
                    value = value._clone()
                    value.type = c.type
            value_text = self.process(value.self_group(), use_schema=False)

            key_text = self.preparer.quote(c.name)
            action_set_ops.append("%s = %s" % (key_text, value_text))

        # check for names that don't match columns
        if set_parameters:
            util.warn(
                "Additional column names not matching "
                "any column keys in table '%s': %s"
                % (
                    self.current_executable.table.name,
                    (", ".join("'%s'" % c for c in set_parameters)),
                )
            )
            for k, v in set_parameters.items():
                key_text = (
                    self.preparer.quote(k)
                    if isinstance(k, str)
                    else self.process(k, use_schema=False)
                )
                value_text = self.process(
                    coercions.expect(roles.ExpressionElementRole, v),
                    use_schema=False,
                )
                action_set_ops.append("%s = %s" % (key_text, value_text))

        action_text = ", ".join(action_set_ops)
        if clause.update_whereclause is not None:
            action_text += " WHERE %s" % self.process(
                clause.update_whereclause, include_table=True, use_schema=False
            )

        return "ON CONFLICT %s DO UPDATE SET %s" % (target_text, action_text)

    def visit_bitwise_xor_op_binary(self, binary, operator, **kw):
        # sqlite has no xor. Use "a XOR b" = "(a | b) - (a & b)".
        kw["eager_grouping"] = True
        or_ = self._generate_generic_binary(binary, " | ", **kw)
        and_ = self._generate_generic_binary(binary, " & ", **kw)
        return f"({or_} - {and_})"


class SQLiteDDLCompiler(compiler.DDLCompiler):
    def get_column_specification(self, column, **kwargs):
        coltype = self.dialect.type_compiler_instance.process(
            column.type, type_expression=column
        )
        colspec = self.preparer.format_column(column) + " " + coltype
        default = self.get_column_default_string(column)
        if default is not None:

            if not re.match(r"""^\s*[\'\"\(]""", default) and re.match(
                r".*\W.*", default
            ):
                colspec += f" DEFAULT ({default})"
            else:
                colspec += f" DEFAULT {default}"

        if not column.nullable:
            colspec += " NOT NULL"

            on_conflict_clause = column.dialect_options["sqlite"][
                "on_conflict_not_null"
            ]
            if on_conflict_clause is not None:
                colspec += " ON CONFLICT " + on_conflict_clause

        if column.primary_key:
            if (
                column.autoincrement is True
                and len(column.table.primary_key.columns) != 1
            ):
                raise exc.CompileError(
                    "SQLite does not support autoincrement for "
                    "composite primary keys"
                )

            if (
                column.table.dialect_options["sqlite"]["autoincrement"]
                and len(column.table.primary_key.columns) == 1
                and issubclass(column.type._type_affinity, sqltypes.Integer)
                and not column.foreign_keys
            ):
                colspec += " PRIMARY KEY"

                on_conflict_clause = column.dialect_options["sqlite"][
                    "on_conflict_primary_key"
                ]
                if on_conflict_clause is not None:
                    colspec += " ON CONFLICT " + on_conflict_clause

                colspec += " AUTOINCREMENT"

        if column.computed is not None:
            colspec += " " + self.process(column.computed)

        return colspec

    def visit_primary_key_constraint(self, constraint, **kw):
        # for columns with sqlite_autoincrement=True,
        # the PRIMARY KEY constraint can only be inline
        # with the column itself.
        if len(constraint.columns) == 1:
            c = list(constraint)[0]
            if (
                c.primary_key
                and c.table.dialect_options["sqlite"]["autoincrement"]
                and issubclass(c.type._type_affinity, sqltypes.Integer)
                and not c.foreign_keys
            ):
                return None

        text = super().visit_primary_key_constraint(constraint)

        on_conflict_clause = constraint.dialect_options["sqlite"][
            "on_conflict"
        ]
        if on_conflict_clause is None and len(constraint.columns) == 1:
            on_conflict_clause = list(constraint)[0].dialect_options["sqlite"][
                "on_conflict_primary_key"
            ]

        if on_conflict_clause is not None:
            text += " ON CONFLICT " + on_conflict_clause

        return text

    def visit_unique_constraint(self, constraint, **kw):
        text = super().visit_unique_constraint(constraint)

        on_conflict_clause = constraint.dialect_options["sqlite"][
            "on_conflict"
        ]
        if on_conflict_clause is None and len(constraint.columns) == 1:
            col1 = list(constraint)[0]
            if isinstance(col1, schema.SchemaItem):
                on_conflict_clause = list(constraint)[0].dialect_options[
                    "sqlite"
                ]["on_conflict_unique"]

        if on_conflict_clause is not None:
            text += " ON CONFLICT " + on_conflict_clause

        return text

    def visit_check_constraint(self, constraint, **kw):
        text = super().visit_check_constraint(constraint)

        on_conflict_clause = constraint.dialect_options["sqlite"][
            "on_conflict"
        ]

        if on_conflict_clause is not None:
            text += " ON CONFLICT " + on_conflict_clause

        return text

    def visit_column_check_constraint(self, constraint, **kw):
        text = super().visit_column_check_constraint(constraint)

        if constraint.dialect_options["sqlite"]["on_conflict"] is not None:
            raise exc.CompileError(
                "SQLite does not support on conflict clause for "
                "column check constraint"
            )

        return text

    def visit_foreign_key_constraint(self, constraint, **kw):
        local_table = constraint.elements[0].parent.table
        remote_table = constraint.elements[0].column.table

        if local_table.schema != remote_table.schema:
            return None
        else:
            return super().visit_foreign_key_constraint(constraint)

    def define_constraint_remote_table(self, constraint, table, preparer):
        """Format the remote table clause of a CREATE CONSTRAINT clause."""

        return preparer.format_table(table, use_schema=False)

    def visit_create_index(
        self, create, include_schema=False, include_table_schema=True, **kw
    ):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = "CREATE "
        if index.unique:
            text += "UNIQUE "

        text += "INDEX "

        if create.if_not_exists:
            text += "IF NOT EXISTS "

        text += "%s ON %s (%s)" % (
            self._prepared_index_name(index, include_schema=True),
            preparer.format_table(index.table, use_schema=False),
            ", ".join(
                self.sql_compiler.process(
                    expr, include_table=False, literal_binds=True
                )
                for expr in index.expressions
            ),
        )

        whereclause = index.dialect_options["sqlite"]["where"]
        if whereclause is not None:
            where_compiled = self.sql_compiler.process(
                whereclause, include_table=False, literal_binds=True
            )
            text += " WHERE " + where_compiled

        return text

    def post_create_table(self, table):
        table_options = []

        if not table.dialect_options["sqlite"]["with_rowid"]:
            table_options.append("WITHOUT ROWID")

        if table.dialect_options["sqlite"]["strict"]:
            table_options.append("STRICT")

        if table_options:
            return "\n " + ",\n ".join(table_options)
        else:
            return ""


class SQLiteTypeCompiler(compiler.GenericTypeCompiler):
    def visit_large_binary(self, type_, **kw):
        return self.visit_BLOB(type_)

    def visit_DATETIME(self, type_, **kw):
        if (
            not isinstance(type_, _DateTimeMixin)
            or type_.format_is_text_affinity
        ):
            return super().visit_DATETIME(type_)
        else:
            return "DATETIME_CHAR"

    def visit_DATE(self, type_, **kw):
        if (
            not isinstance(type_, _DateTimeMixin)
            or type_.format_is_text_affinity
        ):
            return super().visit_DATE(type_)
        else:
            return "DATE_CHAR"

    def visit_TIME(self, type_, **kw):
        if (
            not isinstance(type_, _DateTimeMixin)
            or type_.format_is_text_affinity
        ):
            return super().visit_TIME(type_)
        else:
            return "TIME_CHAR"

    def visit_JSON(self, type_, **kw):
        # note this name provides NUMERIC affinity, not TEXT.
        # should not be an issue unless the JSON value consists of a single
        # numeric value.   JSONTEXT can be used if this case is required.
        return "JSON"


class SQLiteIdentifierPreparer(compiler.IdentifierPreparer):
    reserved_words = {
        "add",
        "after",
        "all",
        "alter",
        "analyze",
        "and",
        "as",
        "asc",
        "attach",
        "autoincrement",
        "before",
        "begin",
        "between",
        "by",
        "cascade",
        "case",
        "cast",
        "check",
        "collate",
        "column",
        "commit",
        "conflict",
        "constraint",
        "create",
        "cross",
        "current_date",
        "current_time",
        "current_timestamp",
        "database",
        "default",
        "deferrable",
        "deferred",
        "delete",
        "desc",
        "detach",
        "distinct",
        "drop",
        "each",
        "else",
        "end",
        "escape",
        "except",
        "exclusive",
        "exists",
        "explain",
        "false",
        "fail",
        "for",
        "foreign",
        "from",
        "full",
        "glob",
        "group",
        "having",
        "if",
        "ignore",
        "immediate",
        "in",
        "index",
        "indexed",
        "initially",
        "inner",
        "insert",
        "instead",
        "intersect",
        "into",
        "is",
        "isnull",
        "join",
        "key",
        "left",
        "like",
        "limit",
        "match",
        "natural",
        "not",
        "notnull",
        "null",
        "of",
        "offset",
        "on",
        "or",
        "order",
        "outer",
        "plan",
        "pragma",
        "primary",
        "query",
        "raise",
        "references",
        "reindex",
        "rename",
        "replace",
        "restrict",
        "right",
        "rollback",
        "row",
        "select",
        "set",
        "table",
        "temp",
        "temporary",
        "then",
        "to",
        "transaction",
        "trigger",
        "true",
        "union",
        "unique",
        "update",
        "using",
        "vacuum",
        "values",
        "view",
        "virtual",
        "when",
        "where",
    }


class SQLiteExecutionContext(default.DefaultExecutionContext):
    @util.memoized_property
    def _preserve_raw_colnames(self):
        return (
            not self.dialect._broken_dotted_colnames
            or self.execution_options.get("sqlite_raw_colnames", False)
        )

    def _translate_colname(self, colname):
        # TODO: detect SQLite version 3.10.0 or greater;
        # see [ticket:3633]

        # adjust for dotted column names.  SQLite
        # in the case of UNION may store col names as
        # "tablename.colname", or if using an attached database,
        # "database.tablename.colname", in cursor.description
        if not self._preserve_raw_colnames and "." in colname:
            return colname.split(".")[-1], colname
        else:
            return colname, None


class SQLiteDialect(default.DefaultDialect):
    name = "sqlite"
    supports_alter = False

    # SQlite supports "DEFAULT VALUES" but *does not* support
    # "VALUES (DEFAULT)"
    supports_default_values = True
    supports_default_metavalue = False

    # sqlite issue:
    # https://github.com/python/cpython/issues/93421
    # note this parameter is no longer used by the ORM or default dialect
    # see #9414
    supports_sane_rowcount_returning = False

    supports_empty_insert = False
    supports_cast = True
    supports_multivalues_insert = True
    use_insertmanyvalues = True
    tuple_in_values = True
    supports_statement_cache = True
    insert_null_pk_still_autoincrements = True
    insert_returning = True
    update_returning = True
    update_returning_multifrom = True
    delete_returning = True
    update_returning_multifrom = True

    supports_default_metavalue = True
    """dialect supports INSERT... VALUES (DEFAULT) syntax"""

    default_metavalue_token = "NULL"
    """for INSERT... VALUES (DEFAULT) syntax, the token to put in the
    parenthesis."""

    default_paramstyle = "qmark"
    execution_ctx_cls = SQLiteExecutionContext
    statement_compiler = SQLiteCompiler
    ddl_compiler = SQLiteDDLCompiler
    type_compiler_cls = SQLiteTypeCompiler
    preparer = SQLiteIdentifierPreparer
    ischema_names = ischema_names
    colspecs = colspecs

    construct_arguments = [
        (
            sa_schema.Table,
            {
                "autoincrement": False,
                "with_rowid": True,
                "strict": False,
            },
        ),
        (sa_schema.Index, {"where": None}),
        (
            sa_schema.Column,
            {
                "on_conflict_primary_key": None,
                "on_conflict_not_null": None,
                "on_conflict_unique": None,
            },
        ),
        (sa_schema.Constraint, {"on_conflict": None}),
    ]

    _broken_fk_pragma_quotes = False
    _broken_dotted_colnames = False

    @util.deprecated_params(
        _json_serializer=(
            "1.3.7",
            "The _json_serializer argument to the SQLite dialect has "
            "been renamed to the correct name of json_serializer.  The old "
            "argument name will be removed in a future release.",
        ),
        _json_deserializer=(
            "1.3.7",
            "The _json_deserializer argument to the SQLite dialect has "
            "been renamed to the correct name of json_deserializer.  The old "
            "argument name will be removed in a future release.",
        ),
    )
    def __init__(
        self,
        native_datetime=False,
        json_serializer=None,
        json_deserializer=None,
        _json_serializer=None,
        _json_deserializer=None,
        **kwargs,
    ):
        default.DefaultDialect.__init__(self, **kwargs)

        if _json_serializer:
            json_serializer = _json_serializer
        if _json_deserializer:
            json_deserializer = _json_deserializer
        self._json_serializer = json_serializer
        self._json_deserializer = json_deserializer

        # this flag used by pysqlite dialect, and perhaps others in the
        # future, to indicate the driver is handling date/timestamp
        # conversions (and perhaps datetime/time as well on some hypothetical
        # driver ?)
        self.native_datetime = native_datetime

        if self.dbapi is not None:
            if self.dbapi.sqlite_version_info < (3, 7, 16):
                util.warn(
                    "SQLite version %s is older than 3.7.16, and will not "
                    "support right nested joins, as are sometimes used in "
                    "more complex ORM scenarios.  SQLAlchemy 1.4 and above "
                    "no longer tries to rewrite these joins."
                    % (self.dbapi.sqlite_version_info,)
                )

            # NOTE: python 3.7 on fedora for me has SQLite 3.34.1.  These
            # version checks are getting very stale.
            self._broken_dotted_colnames = self.dbapi.sqlite_version_info < (
                3,
                10,
                0,
            )
            self.supports_default_values = self.dbapi.sqlite_version_info >= (
                3,
                3,
                8,
            )
            self.supports_cast = self.dbapi.sqlite_version_info >= (3, 2, 3)
            self.supports_multivalues_insert = (
                # https://www.sqlite.org/releaselog/3_7_11.html
                self.dbapi.sqlite_version_info
                >= (3, 7, 11)
            )
            # see https://www.sqlalchemy.org/trac/ticket/2568
            # as well as https://www.sqlite.org/src/info/600482d161
            self._broken_fk_pragma_quotes = self.dbapi.sqlite_version_info < (
                3,
                6,
                14,
            )

            if self.dbapi.sqlite_version_info < (3, 35) or util.pypy:
                self.update_returning = self.delete_returning = (
                    self.insert_returning
                ) = False

            if self.dbapi.sqlite_version_info < (3, 32, 0):
                # https://www.sqlite.org/limits.html
                self.insertmanyvalues_max_parameters = 999

    _isolation_lookup = util.immutabledict(
        {"READ UNCOMMITTED": 1, "SERIALIZABLE": 0}
    )

    def get_isolation_level_values(self, dbapi_connection):
        return list(self._isolation_lookup)

    def set_isolation_level(self, dbapi_connection, level):
        isolation_level = self._isolation_lookup[level]

        cursor = dbapi_connection.cursor()
        cursor.execute(f"PRAGMA read_uncommitted = {isolation_level}")
        cursor.close()

    def get_isolation_level(self, dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA read_uncommitted")
        res = cursor.fetchone()
        if res:
            value = res[0]
        else:
            # https://www.sqlite.org/changes.html#version_3_3_3
            # "Optional READ UNCOMMITTED isolation (instead of the
            # default isolation level of SERIALIZABLE) and
            # table level locking when database connections
            # share a common cache.""
            # pre-SQLite 3.3.0 default to 0
            value = 0
        cursor.close()
        if value == 0:
            return "SERIALIZABLE"
        elif value == 1:
            return "READ UNCOMMITTED"
        else:
            assert False, "Unknown isolation level %s" % value

    @reflection.cache
    def get_schema_names(self, connection, **kw):
        s = "PRAGMA database_list"
        dl = connection.exec_driver_sql(s)

        return [db[1] for db in dl if db[1] != "temp"]

    def _format_schema(self, schema, table_name):
        if schema is not None:
            qschema = self.identifier_preparer.quote_identifier(schema)
            name = f"{qschema}.{table_name}"
        else:
            name = table_name
        return name

    def _sqlite_main_query(
        self,
        table: str,
        type_: str,
        schema: Optional[str],
        sqlite_include_internal: bool,
    ):
        main = self._format_schema(schema, table)
        if not sqlite_include_internal:
            filter_table = " AND name NOT LIKE 'sqlite~_%' ESCAPE '~'"
        else:
            filter_table = ""
        query = (
            f"SELECT name FROM {main} "
            f"WHERE type='{type_}'{filter_table} "
            "ORDER BY name"
        )
        return query

    @reflection.cache
    def get_table_names(
        self, connection, schema=None, sqlite_include_internal=False, **kw
    ):
        query = self._sqlite_main_query(
            "sqlite_master", "table", schema, sqlite_include_internal
        )
        names = connection.exec_driver_sql(query).scalars().all()
        return names

    @reflection.cache
    def get_temp_table_names(
        self, connection, sqlite_include_internal=False, **kw
    ):
        query = self._sqlite_main_query(
            "sqlite_temp_master", "table", None, sqlite_include_internal
        )
        names = connection.exec_driver_sql(query).scalars().all()
        return names

    @reflection.cache
    def get_temp_view_names(
        self, connection, sqlite_include_internal=False, **kw
    ):
        query = self._sqlite_main_query(
            "sqlite_temp_master", "view", None, sqlite_include_internal
        )
        names = connection.exec_driver_sql(query).scalars().all()
        return names

    @reflection.cache
    def has_table(self, connection, table_name, schema=None, **kw):
        self._ensure_has_table_connection(connection)

        if schema is not None and schema not in self.get_schema_names(
            connection, **kw
        ):
            return False

        info = self._get_table_pragma(
            connection, "table_info", table_name, schema=schema
        )
        return bool(info)

    def _get_default_schema_name(self, connection):
        return "main"

    @reflection.cache
    def get_view_names(
        self, connection, schema=None, sqlite_include_internal=False, **kw
    ):
        query = self._sqlite_main_query(
            "sqlite_master", "view", schema, sqlite_include_internal
        )
        names = connection.exec_driver_sql(query).scalars().all()
        return names

    @reflection.cache
    def get_view_definition(self, connection, view_name, schema=None, **kw):
        if schema is not None:
            qschema = self.identifier_preparer.quote_identifier(schema)
            master = f"{qschema}.sqlite_master"
            s = ("SELECT sql FROM %s WHERE name = ? AND type='view'") % (
                master,
            )
            rs = connection.exec_driver_sql(s, (view_name,))
        else:
            try:
                s = (
                    "SELECT sql FROM "
                    " (SELECT * FROM sqlite_master UNION ALL "
                    "  SELECT * FROM sqlite_temp_master) "
                    "WHERE name = ? "
                    "AND type='view'"
                )
                rs = connection.exec_driver_sql(s, (view_name,))
            except exc.DBAPIError:
                s = (
                    "SELECT sql FROM sqlite_master WHERE name = ? "
                    "AND type='view'"
                )
                rs = connection.exec_driver_sql(s, (view_name,))

        result = rs.fetchall()
        if result:
            return result[0].sql
        else:
            raise exc.NoSuchTableError(
                f"{schema}.{view_name}" if schema else view_name
            )

    @reflection.cache
    def get_columns(self, connection, table_name, schema=None, **kw):
        pragma = "table_info"
        # computed columns are threaded as hidden, they require table_xinfo
        if self.server_version_info >= (3, 31):
            pragma = "table_xinfo"
        info = self._get_table_pragma(
            connection, pragma, table_name, schema=schema
        )
        columns = []
        tablesql = None
        for row in info:
            name = row[1]
            type_ = row[2].upper()
            nullable = not row[3]
            default = row[4]
            primary_key = row[5]
            hidden = row[6] if pragma == "table_xinfo" else 0

            # hidden has value 0 for normal columns, 1 for hidden columns,
            # 2 for computed virtual columns and 3 for computed stored columns
            # https://www.sqlite.org/src/info/069351b85f9a706f60d3e98fbc8aaf40c374356b967c0464aede30ead3d9d18b
            if hidden == 1:
                continue

            generated = bool(hidden)
            persisted = hidden == 3

            if tablesql is None and generated:
                tablesql = self._get_table_sql(
                    connection, table_name, schema, **kw
                )
                # remove create table
                match = re.match(
                    r"create table .*?\((.*)\)$",
                    tablesql.strip(),
                    re.DOTALL | re.IGNORECASE,
                )
                assert match, f"create table not found in {tablesql}"
                tablesql = match.group(1).strip()

            columns.append(
                self._get_column_info(
                    name,
                    type_,
                    nullable,
                    default,
                    primary_key,
                    generated,
                    persisted,
                    tablesql,
                )
            )
        if columns:
            return columns
        elif not self.has_table(connection, table_name, schema):
            raise exc.NoSuchTableError(
                f"{schema}.{table_name}" if schema else table_name
            )
        else:
            return ReflectionDefaults.columns()

    def _get_column_info(
        self,
        name,
        type_,
        nullable,
        default,
        primary_key,
        generated,
        persisted,
        tablesql,
    ):
        if generated:
            # the type of a column "cc INTEGER GENERATED ALWAYS AS (1 + 42)"
            # somehow is "INTEGER GENERATED ALWAYS"
            type_ = re.sub("generated", "", type_, flags=re.IGNORECASE)
            type_ = re.sub("always", "", type_, flags=re.IGNORECASE).strip()

        coltype = self._resolve_type_affinity(type_)

        if default is not None:
            default = str(default)

        colspec = {
            "name": name,
            "type": coltype,
            "nullable": nullable,
            "default": default,
            "primary_key": primary_key,
        }
        if generated:
            sqltext = ""
            if tablesql:
                pattern = (
                    r"[^,]*\s+GENERATED\s+ALWAYS\s+AS"
                    r"\s+\((.*)\)\s*(?:virtual|stored)?"
                )
                match = re.search(
                    re.escape(name) + pattern, tablesql, re.IGNORECASE
                )
                if match:
                    sqltext = match.group(1)
            colspec["computed"] = {"sqltext": sqltext, "persisted": persisted}
        return colspec

    def _resolve_type_affinity(self, type_):
        """Return a data type from a reflected column, using affinity rules.

        SQLite's goal for universal compatibility introduces some complexity
        during reflection, as a column's defined type might not actually be a
        type that SQLite understands - or indeed, my not be defined *at all*.
        Internally, SQLite handles this with a 'data type affinity' for each
        column definition, mapping to one of 'TEXT', 'NUMERIC', 'INTEGER',
        'REAL', or 'NONE' (raw bits). The algorithm that determines this is
        listed in https://www.sqlite.org/datatype3.html section 2.1.

        This method allows SQLAlchemy to support that algorithm, while still
        providing access to smarter reflection utilities by recognizing
        column definitions that SQLite only supports through affinity (like
        DATE and DOUBLE).

        """
        match = re.match(r"([\w ]+)(\(.*?\))?", type_)
        if match:
            coltype = match.group(1)
            args = match.group(2)
        else:
            coltype = ""
            args = ""

        if coltype in self.ischema_names:
            coltype = self.ischema_names[coltype]
        elif "INT" in coltype:
            coltype = sqltypes.INTEGER
        elif "CHAR" in coltype or "CLOB" in coltype or "TEXT" in coltype:
            coltype = sqltypes.TEXT
        elif "BLOB" in coltype or not coltype:
            coltype = sqltypes.NullType
        elif "REAL" in coltype or "FLOA" in coltype or "DOUB" in coltype:
            coltype = sqltypes.REAL
        else:
            coltype = sqltypes.NUMERIC

        if args is not None:
            args = re.findall(r"(\d+)", args)
            try:
                coltype = coltype(*[int(a) for a in args])
            except TypeError:
                util.warn(
                    "Could not instantiate type %s with "
                    "reflected arguments %s; using no arguments."
                    % (coltype, args)
                )
                coltype = coltype()
        else:
            coltype = coltype()

        return coltype

    @reflection.cache
    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        constraint_name = None
        table_data = self._get_table_sql(connection, table_name, schema=schema)
        if table_data:
            PK_PATTERN = r"CONSTRAINT (\w+) PRIMARY KEY"
            result = re.search(PK_PATTERN, table_data, re.I)
            constraint_name = result.group(1) if result else None

        cols = self.get_columns(connection, table_name, schema, **kw)
        # consider only pk columns. This also avoids sorting the cached
        # value returned by get_columns
        cols = [col for col in cols if col.get("primary_key", 0) > 0]
        cols.sort(key=lambda col: col.get("primary_key"))
        pkeys = [col["name"] for col in cols]

        if pkeys:
            return {"constrained_columns": pkeys, "name": constraint_name}
        else:
            return ReflectionDefaults.pk_constraint()

    @reflection.cache
    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        # sqlite makes this *extremely difficult*.
        # First, use the pragma to get the actual FKs.
        pragma_fks = self._get_table_pragma(
            connection, "foreign_key_list", table_name, schema=schema
        )

        fks = {}

        for row in pragma_fks:
            (numerical_id, rtbl, lcol, rcol) = (row[0], row[2], row[3], row[4])

            if not rcol:
                # no referred column, which means it was not named in the
                # original DDL.  The referred columns of the foreign key
                # constraint are therefore the primary key of the referred
                # table.
                try:
                    referred_pk = self.get_pk_constraint(
                        connection, rtbl, schema=schema, **kw
                    )
                    referred_columns = referred_pk["constrained_columns"]
                except exc.NoSuchTableError:
                    # ignore not existing parents
                    referred_columns = []
            else:
                # note we use this list only if this is the first column
                # in the constraint.  for subsequent columns we ignore the
                # list and append "rcol" if present.
                referred_columns = []

            if self._broken_fk_pragma_quotes:
                rtbl = re.sub(r"^[\"\[`\']|[\"\]`\']$", "", rtbl)

            if numerical_id in fks:
                fk = fks[numerical_id]
            else:
                fk = fks[numerical_id] = {
                    "name": None,
                    "constrained_columns": [],
                    "referred_schema": schema,
                    "referred_table": rtbl,
                    "referred_columns": referred_columns,
                    "options": {},
                }
                fks[numerical_id] = fk

            fk["constrained_columns"].append(lcol)

            if rcol:
                fk["referred_columns"].append(rcol)

        def fk_sig(constrained_columns, referred_table, referred_columns):
            return (
                tuple(constrained_columns)
                + (referred_table,)
                + tuple(referred_columns)
            )

        # then, parse the actual SQL and attempt to find DDL that matches
        # the names as well.   SQLite saves the DDL in whatever format
        # it was typed in as, so need to be liberal here.

        keys_by_signature = {
            fk_sig(
                fk["constrained_columns"],
                fk["referred_table"],
                fk["referred_columns"],
            ): fk
            for fk in fks.values()
        }

        table_data = self._get_table_sql(connection, table_name, schema=schema)

        def parse_fks():
            if table_data is None:
                # system tables, etc.
                return

            # note that we already have the FKs from PRAGMA above.  This whole
            # regexp thing is trying to locate additional detail about the
            # FKs, namely the name of the constraint and other options.
            # so parsing the columns is really about matching it up to what
            # we already have.
            FK_PATTERN = (
                r"(?:CONSTRAINT (\w+) +)?"
                r"FOREIGN KEY *\( *(.+?) *\) +"
                r'REFERENCES +(?:(?:"(.+?)")|([a-z0-9_]+)) *\( *((?:(?:"[^"]+"|[a-z0-9_]+) *(?:, *)?)+)\) *'  # noqa: E501
                r"((?:ON (?:DELETE|UPDATE) "
                r"(?:SET NULL|SET DEFAULT|CASCADE|RESTRICT|NO ACTION) *)*)"
                r"((?:NOT +)?DEFERRABLE)?"
                r"(?: +INITIALLY +(DEFERRED|IMMEDIATE))?"
            )
            for match in re.finditer(FK_PATTERN, table_data, re.I):
                (
                    constraint_name,
                    constrained_columns,
                    referred_quoted_name,
                    referred_name,
                    referred_columns,
                    onupdatedelete,
                    deferrable,
                    initially,
                ) = match.group(1, 2, 3, 4, 5, 6, 7, 8)
                constrained_columns = list(
                    self._find_cols_in_sig(constrained_columns)
                )
                if not referred_columns:
                    referred_columns = constrained_columns
                else:
                    referred_columns = list(
                        self._find_cols_in_sig(referred_columns)
                    )
                referred_name = referred_quoted_name or referred_name
                options = {}

                for token in re.split(r" *\bON\b *", onupdatedelete.upper()):
                    if token.startswith("DELETE"):
                        ondelete = token[6:].strip()
                        if ondelete and ondelete != "NO ACTION":
                            options["ondelete"] = ondelete
                    elif token.startswith("UPDATE"):
                        onupdate = token[6:].strip()
                        if onupdate and onupdate != "NO ACTION":
                            options["onupdate"] = onupdate

                if deferrable:
                    options["deferrable"] = "NOT" not in deferrable.upper()
                if initially:
                    options["initially"] = initially.upper()

                yield (
                    constraint_name,
                    constrained_columns,
                    referred_name,
                    referred_columns,
                    options,
                )

        fkeys = []

        for (
            constraint_name,
            constrained_columns,
            referred_name,
            referred_columns,
            options,
        ) in parse_fks():
            sig = fk_sig(constrained_columns, referred_name, referred_columns)
            if sig not in keys_by_signature:
                util.warn(
                    "WARNING: SQL-parsed foreign key constraint "
                    "'%s' could not be located in PRAGMA "
                    "foreign_keys for table %s" % (sig, table_name)
                )
                continue
            key = keys_by_signature.pop(sig)
            key["name"] = constraint_name
            key["options"] = options
            fkeys.append(key)
        # assume the remainders are the unnamed, inline constraints, just
        # use them as is as it's extremely difficult to parse inline
        # constraints
        fkeys.extend(keys_by_signature.values())
        if fkeys:
            return fkeys
        else:
            return ReflectionDefaults.foreign_keys()

    def _find_cols_in_sig(self, sig):
        for match in re.finditer(r'(?:"(.+?)")|([a-z0-9_]+)', sig, re.I):
            yield match.group(1) or match.group(2)

    @reflection.cache
    def get_unique_constraints(
        self, connection, table_name, schema=None, **kw
    ):
        auto_index_by_sig = {}
        for idx in self.get_indexes(
            connection,
            table_name,
            schema=schema,
            include_auto_indexes=True,
            **kw,
        ):
            if not idx["name"].startswith("sqlite_autoindex"):
                continue
            sig = tuple(idx["column_names"])
            auto_index_by_sig[sig] = idx

        table_data = self._get_table_sql(
            connection, table_name, schema=schema, **kw
        )
        unique_constraints = []

        def parse_uqs():
            if table_data is None:
                return
            UNIQUE_PATTERN = r'(?:CONSTRAINT "?(.+?)"? +)?UNIQUE *\((.+?)\)'
            INLINE_UNIQUE_PATTERN = (
                r'(?:(".+?")|(?:[\[`])?([a-z0-9_]+)(?:[\]`])?)[\t ]'
                r"+[a-z0-9_ ]+?[\t ]+UNIQUE"
            )

            for match in re.finditer(UNIQUE_PATTERN, table_data, re.I):
                name, cols = match.group(1, 2)
                yield name, list(self._find_cols_in_sig(cols))

            # we need to match inlines as well, as we seek to differentiate
            # a UNIQUE constraint from a UNIQUE INDEX, even though these
            # are kind of the same thing :)
            for match in re.finditer(INLINE_UNIQUE_PATTERN, table_data, re.I):
                cols = list(
                    self._find_cols_in_sig(match.group(1) or match.group(2))
                )
                yield None, cols

        for name, cols in parse_uqs():
            sig = tuple(cols)
            if sig in auto_index_by_sig:
                auto_index_by_sig.pop(sig)
                parsed_constraint = {"name": name, "column_names": cols}
                unique_constraints.append(parsed_constraint)
        # NOTE: auto_index_by_sig might not be empty here,
        # the PRIMARY KEY may have an entry.
        if unique_constraints:
            return unique_constraints
        else:
            return ReflectionDefaults.unique_constraints()

    @reflection.cache
    def get_check_constraints(self, connection, table_name, schema=None, **kw):
        table_data = self._get_table_sql(
            connection, table_name, schema=schema, **kw
        )

        # NOTE NOTE NOTE
        # DO NOT CHANGE THIS REGULAR EXPRESSION.   There is no known way
        # to parse CHECK constraints that contain newlines themselves using
        # regular expressions, and the approach here relies upon each
        # individual
        # CHECK constraint being on a single line by itself.   This
        # necessarily makes assumptions as to how the CREATE TABLE
        # was emitted.   A more comprehensive DDL parsing solution would be
        # needed to improve upon the current situation. See #11840 for
        # background
        CHECK_PATTERN = r"(?:CONSTRAINT (.+) +)?CHECK *\( *(.+) *\),? *"
        cks = []

        for match in re.finditer(CHECK_PATTERN, table_data or "", re.I):

            name = match.group(1)

            if name:
                name = re.sub(r'^"|"$', "", name)

            cks.append({"sqltext": match.group(2), "name": name})
        cks.sort(key=lambda d: d["name"] or "~")  # sort None as last
        if cks:
            return cks
        else:
            return ReflectionDefaults.check_constraints()

    @reflection.cache
    def get_indexes(self, connection, table_name, schema=None, **kw):
        pragma_indexes = self._get_table_pragma(
            connection, "index_list", table_name, schema=schema
        )
        indexes = []

        # regular expression to extract the filter predicate of a partial
        # index. this could fail to extract the predicate correctly on
        # indexes created like
        #   CREATE INDEX i ON t (col || ') where') WHERE col <> ''
        # but as this function does not support expression-based indexes
        # this case does not occur.
        partial_pred_re = re.compile(r"\)\s+where\s+(.+)", re.IGNORECASE)

        if schema:
            schema_expr = "%s." % self.identifier_preparer.quote_identifier(
                schema
            )
        else:
            schema_expr = ""

        include_auto_indexes = kw.pop("include_auto_indexes", False)
        for row in pragma_indexes:
            # ignore implicit primary key index.
            # https://www.mail-archive.com/sqlite-users@sqlite.org/msg30517.html
            if not include_auto_indexes and row[1].startswith(
                "sqlite_autoindex"
            ):
                continue
            indexes.append(
                dict(
                    name=row[1],
                    column_names=[],
                    unique=row[2],
                    dialect_options={},
                )
            )

            # check partial indexes
            if len(row) >= 5 and row[4]:
                s = (
                    "SELECT sql FROM %(schema)ssqlite_master "
                    "WHERE name = ? "
                    "AND type = 'index'" % {"schema": schema_expr}
                )
                rs = connection.exec_driver_sql(s, (row[1],))
                index_sql = rs.scalar()
                predicate_match = partial_pred_re.search(index_sql)
                if predicate_match is None:
                    # unless the regex is broken this case shouldn't happen
                    # because we know this is a partial index, so the
                    # definition sql should match the regex
                    util.warn(
                        "Failed to look up filter predicate of "
                        "partial index %s" % row[1]
                    )
                else:
                    predicate = predicate_match.group(1)
                    indexes[-1]["dialect_options"]["sqlite_where"] = text(
                        predicate
                    )

        # loop thru unique indexes to get the column names.
        for idx in list(indexes):
            pragma_index = self._get_table_pragma(
                connection, "index_info", idx["name"], schema=schema
            )

            for row in pragma_index:
                if row[2] is None:
                    util.warn(
                        "Skipped unsupported reflection of "
                        "expression-based index %s" % idx["name"]
                    )
                    indexes.remove(idx)
                    break
                else:
                    idx["column_names"].append(row[2])

        indexes.sort(key=lambda d: d["name"] or "~")  # sort None as last
        if indexes:
            return indexes
        elif not self.has_table(connection, table_name, schema):
            raise exc.NoSuchTableError(
                f"{schema}.{table_name}" if schema else table_name
            )
        else:
            return ReflectionDefaults.indexes()

    def _is_sys_table(self, table_name):
        return table_name in {
            "sqlite_schema",
            "sqlite_master",
            "sqlite_temp_schema",
            "sqlite_temp_master",
        }

    @reflection.cache
    def _get_table_sql(self, connection, table_name, schema=None, **kw):
        if schema:
            schema_expr = "%s." % (
                self.identifier_preparer.quote_identifier(schema)
            )
        else:
            schema_expr = ""
        try:
            s = (
                "SELECT sql FROM "
                " (SELECT * FROM %(schema)ssqlite_master UNION ALL "
                "  SELECT * FROM %(schema)ssqlite_temp_master) "
                "WHERE name = ? "
                "AND type in ('table', 'view')" % {"schema": schema_expr}
            )
            rs = connection.exec_driver_sql(s, (table_name,))
        except exc.DBAPIError:
            s = (
                "SELECT sql FROM %(schema)ssqlite_master "
                "WHERE name = ? "
                "AND type in ('table', 'view')" % {"schema": schema_expr}
            )
            rs = connection.exec_driver_sql(s, (table_name,))
        value = rs.scalar()
        if value is None and not self._is_sys_table(table_name):
            raise exc.NoSuchTableError(f"{schema_expr}{table_name}")
        return value

    def _get_table_pragma(self, connection, pragma, table_name, schema=None):
        quote = self.identifier_preparer.quote_identifier
        if schema is not None:
            statements = [f"PRAGMA {quote(schema)}."]
        else:
            # because PRAGMA looks in all attached databases if no schema
            # given, need to specify "main" schema, however since we want
            # 'temp' tables in the same namespace as 'main', need to run
            # the PRAGMA twice
            statements = ["PRAGMA main.", "PRAGMA temp."]

        qtable = quote(table_name)
        for statement in statements:
            statement = f"{statement}{pragma}({qtable})"
            cursor = connection.exec_driver_sql(statement)
            if not cursor._soft_closed:
                # work around SQLite issue whereby cursor.description
                # is blank when PRAGMA returns no rows:
                # https://www.sqlite.org/cvstrac/tktview?tn=1884
                result = cursor.fetchall()
            else:
                result = []
            if result:
                return result
        else:
            return []

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\pretty\pretty.py ===
import itertools

from sympy.core import S
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Number, Rational
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.str import sstr
from sympy.utilities.iterables import has_variety
from sympy.utilities.exceptions import sympy_deprecation_warning

from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \
    xsym, pretty_symbol, pretty_atom, pretty_use_unicode, greek_unicode, U, \
    pretty_try_use_unicode, annotated, is_subscriptable_in_unicode, center_pad,  root as nth_root

# rename for usage from outside
pprint_use_unicode = pretty_use_unicode
pprint_try_use_unicode = pretty_try_use_unicode


class PrettyPrinter(Printer):
    """Printer, which converts an expression into 2D ASCII-art figure."""
    printmethod = "_pretty"

    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "use_unicode": None,
        "wrap_line": True,
        "num_columns": None,
        "use_unicode_sqrt_char": True,
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "perm_cyclic": True
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)

        if not isinstance(self._settings['imaginary_unit'], str):
            raise TypeError("'imaginary_unit' must a string, not {}".format(self._settings['imaginary_unit']))
        elif self._settings['imaginary_unit'] not in ("i", "j"):
            raise ValueError("'imaginary_unit' must be either 'i' or 'j', not '{}'".format(self._settings['imaginary_unit']))

    def emptyPrinter(self, expr):
        return prettyForm(str(expr))

    @property
    def _use_unicode(self):
        if self._settings['use_unicode']:
            return True
        else:
            return pretty_use_unicode()

    def doprint(self, expr):
        return self._print(expr).render(**self._settings)

    # empty op so _print(stringPict) returns the same
    def _print_stringPict(self, e):
        return e

    def _print_basestring(self, e):
        return prettyForm(e)

    def _print_atan2(self, e):
        pform = prettyForm(*self._print_seq(e.args).parens())
        pform = prettyForm(*pform.left('atan2'))
        return pform

    def _print_Symbol(self, e, bold_name=False):
        symb = pretty_symbol(e.name, bold_name)
        return prettyForm(symb)
    _print_RandomSymbol = _print_Symbol
    def _print_MatrixSymbol(self, e):
        return self._print_Symbol(e, self._settings['mat_symbol_style'] == "bold")

    def _print_Float(self, e):
        # we will use StrPrinter's Float printer, but we need to handle the
        # full_prec ourselves, according to the self._print_level
        full_prec = self._settings["full_prec"]
        if full_prec == "auto":
            full_prec = self._print_level == 1
        return prettyForm(sstr(e, full_prec=full_prec))

    def _print_Cross(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('MULTIPLICATION SIGN'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform

    def _print_Curl(self, e):
        vec = e._expr
        pform = self._print(vec)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('MULTIPLICATION SIGN'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Divergence(self, e):
        vec = e._expr
        pform = self._print(vec)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Dot(self, e):
        vec1 = e._expr1
        vec2 = e._expr2
        pform = self._print(vec2)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('DOT OPERATOR'))))
        pform = prettyForm(*pform.left(')'))
        pform = prettyForm(*pform.left(self._print(vec1)))
        pform = prettyForm(*pform.left('('))
        return pform

    def _print_Gradient(self, e):
        func = e._expr
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('NABLA'))))
        return pform

    def _print_Laplacian(self, e):
        func = e._expr
        pform = self._print(func)
        pform = prettyForm(*pform.left('('))
        pform = prettyForm(*pform.right(')'))
        pform = prettyForm(*pform.left(self._print(U('INCREMENT'))))
        return pform

    def _print_Atom(self, e):
        try:
            # print atoms like Exp1 or Pi
            return prettyForm(pretty_atom(e.__class__.__name__, printer=self))
        except KeyError:
            return self.emptyPrinter(e)

    # Infinity inherits from Number, so we have to override _print_XXX order
    _print_Infinity = _print_Atom
    _print_NegativeInfinity = _print_Atom
    _print_EmptySet = _print_Atom
    _print_Naturals = _print_Atom
    _print_Naturals0 = _print_Atom
    _print_Integers = _print_Atom
    _print_Rationals = _print_Atom
    _print_Complexes = _print_Atom

    _print_EmptySequence = _print_Atom

    def _print_Reals(self, e):
        if self._use_unicode:
            return self._print_Atom(e)
        else:
            inf_list = ['-oo', 'oo']
            return self._print_seq(inf_list, '(', ')')

    def _print_subfactorial(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('!'))
        return pform

    def _print_factorial(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right('!'))
        return pform

    def _print_factorial2(self, e):
        x = e.args[0]
        pform = self._print(x)
        # Add parentheses if needed
        if not ((x.is_Integer and x.is_nonnegative) or x.is_Symbol):
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right('!!'))
        return pform

    def _print_binomial(self, e):
        n, k = e.args

        n_pform = self._print(n)
        k_pform = self._print(k)

        bar = ' '*max(n_pform.width(), k_pform.width())

        pform = prettyForm(*k_pform.above(bar))
        pform = prettyForm(*pform.above(n_pform))
        pform = prettyForm(*pform.parens('(', ')'))

        pform.baseline = (pform.baseline + 1)//2

        return pform

    def _print_Relational(self, e):
        op = prettyForm(' ' + xsym(e.rel_op) + ' ')

        l = self._print(e.lhs)
        r = self._print(e.rhs)
        pform = prettyForm(*stringPict.next(l, op, r), binding=prettyForm.OPEN)
        return pform

    def _print_Not(self, e):
        from sympy.logic.boolalg import (Equivalent, Implies)
        if self._use_unicode:
            arg = e.args[0]
            pform = self._print(arg)
            if isinstance(arg, Equivalent):
                return self._print_Equivalent(arg, altchar=pretty_atom('NotEquiv'))
            if isinstance(arg, Implies):
                return self._print_Implies(arg, altchar=pretty_atom('NotArrow'))

            if arg.is_Boolean and not arg.is_Not:
                pform = prettyForm(*pform.parens())

            return prettyForm(*pform.left(pretty_atom('Not')))
        else:
            return self._print_Function(e)

    def __print_Boolean(self, e, char, sort=True):
        args = e.args
        if sort:
            args = sorted(e.args, key=default_sort_key)
        arg = args[0]
        pform = self._print(arg)

        if arg.is_Boolean and not arg.is_Not:
            pform = prettyForm(*pform.parens())

        for arg in args[1:]:
            pform_arg = self._print(arg)

            if arg.is_Boolean and not arg.is_Not:
                pform_arg = prettyForm(*pform_arg.parens())

            pform = prettyForm(*pform.right(' %s ' % char))
            pform = prettyForm(*pform.right(pform_arg))

        return pform

    def _print_And(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('And'))
        else:
            return self._print_Function(e, sort=True)

    def _print_Or(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Or'))
        else:
            return self._print_Function(e, sort=True)

    def _print_Xor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom("Xor"))
        else:
            return self._print_Function(e, sort=True)

    def _print_Nand(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Nand'))
        else:
            return self._print_Function(e, sort=True)

    def _print_Nor(self, e):
        if self._use_unicode:
            return self.__print_Boolean(e, pretty_atom('Nor'))
        else:
            return self._print_Function(e, sort=True)

    def _print_Implies(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or pretty_atom('Arrow'), sort=False)
        else:
            return self._print_Function(e)

    def _print_Equivalent(self, e, altchar=None):
        if self._use_unicode:
            return self.__print_Boolean(e, altchar or pretty_atom('Equiv'))
        else:
            return self._print_Function(e, sort=True)

    def _print_conjugate(self, e):
        pform = self._print(e.args[0])
        return prettyForm( *pform.above( hobj('_', pform.width())) )

    def _print_Abs(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens('|', '|'))
        return pform

    def _print_floor(self, e):
        if self._use_unicode:
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens('lfloor', 'rfloor'))
            return pform
        else:
            return self._print_Function(e)

    def _print_ceiling(self, e):
        if self._use_unicode:
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens('lceil', 'rceil'))
            return pform
        else:
            return self._print_Function(e)

    def _print_Derivative(self, deriv):
        if requires_partial(deriv.expr) and self._use_unicode:
            deriv_symbol = U('PARTIAL DIFFERENTIAL')
        else:
            deriv_symbol = r'd'
        x = None
        count_total_deriv = 0

        for sym, num in reversed(deriv.variable_count):
            s = self._print(sym)
            ds = prettyForm(*s.left(deriv_symbol))
            count_total_deriv += num

            if (not num.is_Integer) or (num > 1):
                ds = ds**prettyForm(str(num))

            if x is None:
                x = ds
            else:
                x = prettyForm(*x.right(' '))
                x = prettyForm(*x.right(ds))

        f = prettyForm(
            binding=prettyForm.FUNC, *self._print(deriv.expr).parens())

        pform = prettyForm(deriv_symbol)

        if (count_total_deriv > 1) != False:
            pform = pform**prettyForm(str(count_total_deriv))

        pform = prettyForm(*pform.below(stringPict.LINE, x))
        pform.baseline = pform.baseline + 1
        pform = prettyForm(*stringPict.next(pform, f))
        pform.binding = prettyForm.MUL

        return pform

    def _print_Cycle(self, dc):
        from sympy.combinatorics.permutations import Permutation, Cycle
        # for Empty Cycle
        if dc == Cycle():
            cyc = stringPict('')
            return prettyForm(*cyc.parens())

        dc_list = Permutation(dc.list()).cyclic_form
        # for Identity Cycle
        if dc_list == []:
            cyc = self._print(dc.size - 1)
            return prettyForm(*cyc.parens())

        cyc = stringPict('')
        for i in dc_list:
            l = self._print(str(tuple(i)).replace(',', ''))
            cyc = prettyForm(*cyc.right(l))
        return cyc

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=7,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            return self._print_Cycle(Cycle(expr))

        lower = expr.array_form
        upper = list(range(len(lower)))

        result = stringPict('')
        first = True
        for u, l in zip(upper, lower):
            s1 = self._print(u)
            s2 = self._print(l)
            col = prettyForm(*s1.below(s2))
            if first:
                first = False
            else:
                col = prettyForm(*col.left(" "))
            result = prettyForm(*result.right(col))
        return prettyForm(*result.parens())


    def _print_Integral(self, integral):
        f = integral.function

        # Add parentheses if arg involves addition of terms and
        # create a pretty form for the argument
        prettyF = self._print(f)
        # XXX generalize parens
        if f.is_Add:
            prettyF = prettyForm(*prettyF.parens())

        # dx dy dz ...
        arg = prettyF
        for x in integral.limits:
            prettyArg = self._print(x[0])
            # XXX qparens (parens if needs-parens)
            if prettyArg.width() > 1:
                prettyArg = prettyForm(*prettyArg.parens())

            arg = prettyForm(*arg.right(' d', prettyArg))

        # \int \int \int ...
        firstterm = True
        s = None
        for lim in integral.limits:
            # Create bar based on the height of the argument
            h = arg.height()
            H = h + 2

            # XXX hack!
            ascii_mode = not self._use_unicode
            if ascii_mode:
                H += 2

            vint = vobj('int', H)

            # Construct the pretty form with the integral sign and the argument
            pform = prettyForm(vint)
            pform.baseline = arg.baseline + (
                H - h)//2    # covering the whole argument

            if len(lim) > 1:
                # Create pretty forms for endpoints, if definite integral.
                # Do not print empty endpoints.
                if len(lim) == 2:
                    prettyA = prettyForm("")
                    prettyB = self._print(lim[1])
                if len(lim) == 3:
                    prettyA = self._print(lim[1])
                    prettyB = self._print(lim[2])

                if ascii_mode:  # XXX hack
                    # Add spacing so that endpoint can more easily be
                    # identified with the correct integral sign
                    spc = max(1, 3 - prettyB.width())
                    prettyB = prettyForm(*prettyB.left(' ' * spc))

                    spc = max(1, 4 - prettyA.width())
                    prettyA = prettyForm(*prettyA.right(' ' * spc))

                pform = prettyForm(*pform.above(prettyB))
                pform = prettyForm(*pform.below(prettyA))

            if not ascii_mode:  # XXX hack
                pform = prettyForm(*pform.right(' '))

            if firstterm:
                s = pform   # first term
                firstterm = False
            else:
                s = prettyForm(*s.left(pform))

        pform = prettyForm(*arg.left(s))
        pform.binding = prettyForm.MUL
        return pform

    def _print_Product(self, expr):
        func = expr.term
        pretty_func = self._print(func)

        horizontal_chr = xobj('_', 1)
        corner_chr = xobj('_', 1)
        vertical_chr = xobj('|', 1)

        if self._use_unicode:
            # use unicode corners
            horizontal_chr = xobj('-', 1)
            corner_chr = xobj('UpTack', 1)

        func_height = pretty_func.height()

        first = True
        max_upper = 0
        sign_height = 0

        for lim in expr.limits:
            pretty_lower, pretty_upper = self.__print_SumProduct_Limits(lim)

            width = (func_height + 2) * 5 // 3 - 2
            sign_lines = [horizontal_chr + corner_chr + (horizontal_chr * (width-2)) + corner_chr + horizontal_chr]
            for _ in range(func_height + 1):
                sign_lines.append(' ' + vertical_chr + (' ' * (width-2)) + vertical_chr + ' ')

            pretty_sign = stringPict('')
            pretty_sign = prettyForm(*pretty_sign.stack(*sign_lines))


            max_upper = max(max_upper, pretty_upper.height())

            if first:
                sign_height = pretty_sign.height()

            pretty_sign = prettyForm(*pretty_sign.above(pretty_upper))
            pretty_sign = prettyForm(*pretty_sign.below(pretty_lower))

            if first:
                pretty_func.baseline = 0
                first = False

            height = pretty_sign.height()
            padding = stringPict('')
            padding = prettyForm(*padding.stack(*[' ']*(height - 1)))
            pretty_sign = prettyForm(*pretty_sign.right(padding))

            pretty_func = prettyForm(*pretty_sign.right(pretty_func))

        pretty_func.baseline = max_upper + sign_height//2
        pretty_func.binding = prettyForm.MUL
        return pretty_func

    def __print_SumProduct_Limits(self, lim):
        def print_start(lhs, rhs):
            op = prettyForm(' ' + xsym("==") + ' ')
            l = self._print(lhs)
            r = self._print(rhs)
            pform = prettyForm(*stringPict.next(l, op, r))
            return pform

        prettyUpper = self._print(lim[2])
        prettyLower = print_start(lim[0], lim[1])
        return prettyLower, prettyUpper

    def _print_Sum(self, expr):
        ascii_mode = not self._use_unicode

        def asum(hrequired, lower, upper, use_ascii):
            def adjust(s, wid=None, how='<^>'):
                if not wid or len(s) > wid:
                    return s
                need = wid - len(s)
                if how in ('<^>', "<") or how not in list('<^>'):
                    return s + ' '*need
                half = need//2
                lead = ' '*half
                if how == ">":
                    return " "*need + s
                return lead + s + ' '*(need - len(lead))

            h = max(hrequired, 2)
            d = h//2
            w = d + 1
            more = hrequired % 2

            lines = []
            if use_ascii:
                lines.append("_"*(w) + ' ')
                lines.append(r"\%s`" % (' '*(w - 1)))
                for i in range(1, d):
                    lines.append('%s\\%s' % (' '*i, ' '*(w - i)))
                if more:
                    lines.append('%s)%s' % (' '*(d), ' '*(w - d)))
                for i in reversed(range(1, d)):
                    lines.append('%s/%s' % (' '*i, ' '*(w - i)))
                lines.append("/" + "_"*(w - 1) + ',')
                return d, h + more, lines, more
            else:
                w = w + more
                d = d + more
                vsum = vobj('sum', 4)
                lines.append("_"*(w))
                for i in range(0, d):
                    lines.append('%s%s%s' % (' '*i, vsum[2], ' '*(w - i - 1)))
                for i in reversed(range(0, d)):
                    lines.append('%s%s%s' % (' '*i, vsum[4], ' '*(w - i - 1)))
                lines.append(vsum[8]*(w))
                return d, h + 2*more, lines, more

        f = expr.function

        prettyF = self._print(f)

        if f.is_Add:  # add parens
            prettyF = prettyForm(*prettyF.parens())

        H = prettyF.height() + 2

        # \sum \sum \sum ...
        first = True
        max_upper = 0
        sign_height = 0

        for lim in expr.limits:
            prettyLower, prettyUpper = self.__print_SumProduct_Limits(lim)

            max_upper = max(max_upper, prettyUpper.height())

            # Create sum sign based on the height of the argument
            d, h, slines, adjustment = asum(
                H, prettyLower.width(), prettyUpper.width(), ascii_mode)
            prettySign = stringPict('')
            prettySign = prettyForm(*prettySign.stack(*slines))

            if first:
                sign_height = prettySign.height()

            prettySign = prettyForm(*prettySign.above(prettyUpper))
            prettySign = prettyForm(*prettySign.below(prettyLower))

            if first:
                # change F baseline so it centers on the sign
                prettyF.baseline -= d - (prettyF.height()//2 -
                                         prettyF.baseline)
                first = False

            # put padding to the right
            pad = stringPict('')
            pad = prettyForm(*pad.stack(*[' ']*h))
            prettySign = prettyForm(*prettySign.right(pad))
            # put the present prettyF to the right
            prettyF = prettyForm(*prettySign.right(prettyF))

        # adjust baseline of ascii mode sigma with an odd height so that it is
        # exactly through the center
        ascii_adjustment = ascii_mode if not adjustment else 0
        prettyF.baseline = max_upper + sign_height//2 + ascii_adjustment

        prettyF.binding = prettyForm.MUL
        return prettyF

    def _print_Limit(self, l):
        e, z, z0, dir = l.args

        E = self._print(e)
        if precedence(e) <= PRECEDENCE["Mul"]:
            E = prettyForm(*E.parens('(', ')'))
        Lim = prettyForm('lim')

        LimArg = self._print(z)
        if self._use_unicode:
            LimArg = prettyForm(*LimArg.right(f"{xobj('-', 1)}{pretty_atom('Arrow')}"))
        else:
            LimArg = prettyForm(*LimArg.right('->'))
        LimArg = prettyForm(*LimArg.right(self._print(z0)))

        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            dir = ""
        else:
            if self._use_unicode:
                dir = pretty_atom('SuperscriptPlus') if str(dir) == "+" else pretty_atom('SuperscriptMinus')

        LimArg = prettyForm(*LimArg.right(self._print(dir)))

        Lim = prettyForm(*Lim.below(LimArg))
        Lim = prettyForm(*Lim.right(E), binding=prettyForm.MUL)

        return Lim

    def _print_matrix_contents(self, e):
        """
        This method factors out what is essentially grid printing.
        """
        M = e   # matrix
        Ms = {}  # i,j -> pretty(M[i,j])
        for i in range(M.rows):
            for j in range(M.cols):
                Ms[i, j] = self._print(M[i, j])

        # h- and v- spacers
        hsep = 2
        vsep = 1

        # max width for columns
        maxw = [-1] * M.cols

        for j in range(M.cols):
            maxw[j] = max([Ms[i, j].width() for i in range(M.rows)] or [0])

        # drawing result
        D = None

        for i in range(M.rows):

            D_row = None
            for j in range(M.cols):
                s = Ms[i, j]

                # reshape s to maxw
                # XXX this should be generalized, and go to stringPict.reshape ?
                assert s.width() <= maxw[j]

                # hcenter it, +0.5 to the right                        2
                # ( it's better to align formula starts for say 0 and r )
                # XXX this is not good in all cases -- maybe introduce vbaseline?
                left, right = center_pad(s.width(), maxw[j])

                s = prettyForm(*s.right(right))
                s = prettyForm(*s.left(left))

                # we don't need vcenter cells -- this is automatically done in
                # a pretty way because when their baselines are taking into
                # account in .right()

                if D_row is None:
                    D_row = s   # first box in a row
                    continue

                D_row = prettyForm(*D_row.right(' '*hsep))  # h-spacer
                D_row = prettyForm(*D_row.right(s))

            if D is None:
                D = D_row       # first row in a picture
                continue

            # v-spacer
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))

            D = prettyForm(*D.below(D_row))

        if D is None:
            D = prettyForm('')  # Empty Matrix

        return D

    def _print_MatrixBase(self, e, lparens='[', rparens=']'):
        D = self._print_matrix_contents(e)
        D.baseline = D.height()//2
        D = prettyForm(*D.parens(lparens, rparens))
        return D

    def _print_Determinant(self, e):
        mat = e.arg
        if mat.is_MatrixExpr:
            from sympy.matrices.expressions.blockmatrix import BlockMatrix
            if isinstance(mat, BlockMatrix):
                return self._print_MatrixBase(mat.blocks, lparens='|', rparens='|')
            D = self._print(mat)
            D.baseline = D.height()//2
            return prettyForm(*D.parens('|', '|'))
        else:
            return self._print_MatrixBase(mat, lparens='|', rparens='|')

    def _print_TensorProduct(self, expr):
        # This should somehow share the code with _print_WedgeProduct:
        if self._use_unicode:
            circled_times = "\u2297"
        else:
            circled_times = ".*"
        return self._print_seq(expr.args, None, None, circled_times,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_WedgeProduct(self, expr):
        # This should somehow share the code with _print_TensorProduct:
        if self._use_unicode:
            wedge_symbol = "\u2227"
        else:
            wedge_symbol = '/\\'
        return self._print_seq(expr.args, None, None, wedge_symbol,
            parenthesize=lambda x: precedence_traditional(x) <= PRECEDENCE["Mul"])

    def _print_Trace(self, e):
        D = self._print(e.arg)
        D = prettyForm(*D.parens('(',')'))
        D.baseline = D.height()//2
        D = prettyForm(*D.left('\n'*(0) + 'tr'))
        return D


    def _print_MatrixElement(self, expr):
        from sympy.matrices import MatrixSymbol
        if (isinstance(expr.parent, MatrixSymbol)
                and expr.i.is_number and expr.j.is_number):
            return self._print(
                    Symbol(expr.parent.name + '_%d%d' % (expr.i, expr.j)))
        else:
            prettyFunc = self._print(expr.parent)
            prettyFunc = prettyForm(*prettyFunc.parens())
            prettyIndices = self._print_seq((expr.i, expr.j), delimiter=', '
                    ).parens(left='[', right=']')[0]
            pform = prettyForm(binding=prettyForm.FUNC,
                    *stringPict.next(prettyFunc, prettyIndices))

            # store pform parts so it can be reassembled e.g. when powered
            pform.prettyFunc = prettyFunc
            pform.prettyArgs = prettyIndices

            return pform


    def _print_MatrixSlice(self, m):
        # XXX works only for applied functions
        from sympy.matrices import MatrixSymbol
        prettyFunc = self._print(m.parent)
        if not isinstance(m.parent, MatrixSymbol):
            prettyFunc = prettyForm(*prettyFunc.parens())
        def ppslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return prettyForm(*self._print_seq(x, delimiter=':'))
        prettyArgs = self._print_seq((ppslice(m.rowslice, m.parent.rows),
            ppslice(m.colslice, m.parent.cols)), delimiter=', ').parens(left='[', right=']')[0]

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform

    def _print_Transpose(self, expr):
        mat = expr.arg
        pform = self._print(mat)
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            pform = prettyForm(*pform.parens())
        pform = pform**(prettyForm('T'))
        return pform

    def _print_Adjoint(self, expr):
        mat = expr.arg
        pform = self._print(mat)
        if self._use_unicode:
            dag = prettyForm(pretty_atom('Dagger'))
        else:
            dag = prettyForm('+')
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            pform = prettyForm(*pform.parens())
        pform = pform**dag
        return pform

    def _print_BlockMatrix(self, B):
        if B.blocks.shape == (1, 1):
            return self._print(B.blocks[0, 0])
        return self._print(B.blocks)

    def _print_MatAdd(self, expr):
        s = None
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # First element
            else:
                coeff = item.as_coeff_mmul()[0]
                if S(coeff).could_extract_minus_sign():
                    s = prettyForm(*stringPict.next(s, ' '))
                    pform = self._print(item)
                else:
                    s = prettyForm(*stringPict.next(s, ' + '))
                s = prettyForm(*stringPict.next(s, pform))

        return s

    def _print_MatMul(self, expr):
        args = list(expr.args)
        from sympy.matrices.expressions.hadamard import HadamardProduct
        from sympy.matrices.expressions.kronecker import KroneckerProduct
        from sympy.matrices.expressions.matadd import MatAdd
        for i, a in enumerate(args):
            if (isinstance(a, (Add, MatAdd, HadamardProduct, KroneckerProduct))
                    and len(expr.args) > 1):
                args[i] = prettyForm(*self._print(a).parens())
            else:
                args[i] = self._print(a)

        return prettyForm.__mul__(*args)

    def _print_Identity(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom('IdentityMatrix'))
        else:
            return prettyForm('I')

    def _print_ZeroMatrix(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom('ZeroMatrix'))
        else:
            return prettyForm('0')

    def _print_OneMatrix(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom("OneMatrix"))
        else:
            return prettyForm('1')

    def _print_DotProduct(self, expr):
        args = list(expr.args)

        for i, a in enumerate(args):
            args[i] = self._print(a)
        return prettyForm.__mul__(*args)

    def _print_MatPow(self, expr):
        pform = self._print(expr.base)
        from sympy.matrices import MatrixSymbol
        if not isinstance(expr.base, MatrixSymbol) and expr.base.is_MatrixExpr:
            pform = prettyForm(*pform.parens())
        pform = pform**(self._print(expr.exp))
        return pform

    def _print_HadamardProduct(self, expr):
        from sympy.matrices.expressions.hadamard import HadamardProduct
        from sympy.matrices.expressions.matadd import MatAdd
        from sympy.matrices.expressions.matmul import MatMul
        if self._use_unicode:
            delim = pretty_atom('Ring')
        else:
            delim = '.*'
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul, HadamardProduct)))

    def _print_HadamardPower(self, expr):
        # from sympy import MatAdd, MatMul
        if self._use_unicode:
            circ = pretty_atom('Ring')
        else:
            circ = self._print('.')
        pretty_base = self._print(expr.base)
        pretty_exp = self._print(expr.exp)
        if precedence(expr.exp) < PRECEDENCE["Mul"]:
            pretty_exp = prettyForm(*pretty_exp.parens())
        pretty_circ_exp = prettyForm(
            binding=prettyForm.LINE,
            *stringPict.next(circ, pretty_exp)
        )
        return pretty_base**pretty_circ_exp

    def _print_KroneckerProduct(self, expr):
        from sympy.matrices.expressions.matadd import MatAdd
        from sympy.matrices.expressions.matmul import MatMul
        if self._use_unicode:
            delim = f" {pretty_atom('TensorProduct')} "
        else:
            delim = ' x '
        return self._print_seq(expr.args, None, None, delim,
                parenthesize=lambda x: isinstance(x, (MatAdd, MatMul)))

    def _print_FunctionMatrix(self, X):
        D = self._print(X.lamda.expr)
        D = prettyForm(*D.parens('[', ']'))
        return D

    def _print_TransferFunction(self, expr):
        if not expr.num == 1:
            num, den = expr.num, expr.den
            res = Mul(num, Pow(den, -1, evaluate=False), evaluate=False)
            return self._print_Mul(res)
        else:
            return self._print(1)/self._print(expr.den)

    def _print_Series(self, expr):
        args = list(expr.args)
        for i, a in enumerate(expr.args):
            args[i] = prettyForm(*self._print(a).parens())
        return prettyForm.__mul__(*args)

    def _print_MIMOSeries(self, expr):
        from sympy.physics.control.lti import MIMOParallel
        args = list(expr.args)
        pretty_args = []
        for a in reversed(args):
            if (isinstance(a, MIMOParallel) and len(expr.args) > 1):
                expression = self._print(a)
                expression.baseline = expression.height()//2
                pretty_args.append(prettyForm(*expression.parens()))
            else:
                expression = self._print(a)
                expression.baseline = expression.height()//2
                pretty_args.append(expression)
        return prettyForm.__mul__(*pretty_args)

    def _print_Parallel(self, expr):
        s = None
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # First element
            else:
                s = prettyForm(*stringPict.next(s))
                s.baseline = s.height()//2
                s = prettyForm(*stringPict.next(s, ' + '))
                s = prettyForm(*stringPict.next(s, pform))
        return s

    def _print_MIMOParallel(self, expr):
        from sympy.physics.control.lti import TransferFunctionMatrix
        s = None
        for item in expr.args:
            pform = self._print(item)
            if s is None:
                s = pform     # First element
            else:
                s = prettyForm(*stringPict.next(s))
                s.baseline = s.height()//2
                s = prettyForm(*stringPict.next(s, ' + '))
                if isinstance(item, TransferFunctionMatrix):
                    s.baseline = s.height() - 1
                s = prettyForm(*stringPict.next(s, pform))
            # s.baseline = s.height()//2
        return s

    def _print_Feedback(self, expr):
        from sympy.physics.control import TransferFunction, Series

        num, tf = expr.sys1, TransferFunction(1, 1, expr.var)
        num_arg_list = list(num.args) if isinstance(num, Series) else [num]
        den_arg_list = list(expr.sys2.args) if \
            isinstance(expr.sys2, Series) else [expr.sys2]

        if isinstance(num, Series) and isinstance(expr.sys2, Series):
            den = Series(*num_arg_list, *den_arg_list)
        elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
            if expr.sys2 == tf:
                den = Series(*num_arg_list)
            else:
                den = Series(*num_arg_list, expr.sys2)
        elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
            if num == tf:
                den = Series(*den_arg_list)
            else:
                den = Series(num, *den_arg_list)
        else:
            if num == tf:
                den = Series(*den_arg_list)
            elif expr.sys2 == tf:
                den = Series(*num_arg_list)
            else:
                den = Series(*num_arg_list, *den_arg_list)

        denom = prettyForm(*stringPict.next(self._print(tf)))
        denom.baseline = denom.height()//2
        denom = prettyForm(*stringPict.next(denom, ' + ')) if expr.sign == -1 \
            else prettyForm(*stringPict.next(denom, ' - '))
        denom = prettyForm(*stringPict.next(denom, self._print(den)))

        return self._print(num)/denom

    def _print_MIMOFeedback(self, expr):
        from sympy.physics.control import MIMOSeries, TransferFunctionMatrix

        inv_mat = self._print(MIMOSeries(expr.sys2, expr.sys1))
        plant = self._print(expr.sys1)
        _feedback = prettyForm(*stringPict.next(inv_mat))
        _feedback = prettyForm(*stringPict.right("I + ", _feedback)) if expr.sign == -1 \
            else prettyForm(*stringPict.right("I - ", _feedback))
        _feedback = prettyForm(*stringPict.parens(_feedback))
        _feedback.baseline = 0
        _feedback = prettyForm(*stringPict.right(_feedback, '-1 '))
        _feedback.baseline = _feedback.height()//2
        _feedback = prettyForm.__mul__(_feedback, prettyForm(" "))
        if isinstance(expr.sys1, TransferFunctionMatrix):
            _feedback.baseline = _feedback.height() - 1
        _feedback = prettyForm(*stringPict.next(_feedback, plant))
        return _feedback

    def _print_TransferFunctionMatrix(self, expr):
        mat = self._print(expr._expr_mat)
        mat.baseline = mat.height() - 1
        subscript = greek_unicode['tau'] if self._use_unicode else r'{t}'
        mat = prettyForm(*mat.right(subscript))
        return mat

    def _print_StateSpace(self, expr):
        from sympy.matrices.expressions.blockmatrix import BlockMatrix
        A = expr._A
        B = expr._B
        C = expr._C
        D = expr._D
        mat = BlockMatrix([[A, B], [C, D]])
        return self._print(mat.blocks)

    def _print_BasisDependent(self, expr):
        from sympy.vector import Vector

        if not self._use_unicode:
            raise NotImplementedError("ASCII pretty printing of BasisDependent is not implemented")

        if expr == expr.zero:
            return prettyForm(expr.zero._pretty_form)
        o1 = []
        vectstrs = []
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]
        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key = lambda x: x[0].__str__())
            for k, v in inneritems:
                #if the coef of the basis vector is 1
                #we skip the 1
                if v == 1:
                    o1.append("" +
                              k._pretty_form)
                #Same for -1
                elif v == -1:
                    o1.append("(-1) " +
                              k._pretty_form)
                #For a general expr
                else:
                    #We always wrap the measure numbers in
                    #parentheses
                    arg_str = self._print(
                        v).parens()[0]

                    o1.append(arg_str + ' ' + k._pretty_form)
                vectstrs.append(k._pretty_form)

        #outstr = u("").join(o1)
        if o1[0].startswith(" + "):
            o1[0] = o1[0][3:]
        elif o1[0].startswith(" "):
            o1[0] = o1[0][1:]
        #Fixing the newlines
        lengths = []
        strs = ['']
        flag = []
        for i, partstr in enumerate(o1):
            flag.append(0)
            # XXX: What is this hack?
            if '\n' in partstr:
                tempstr = partstr
                tempstr = tempstr.replace(vectstrs[i], '')
                if xobj(')_ext', 1) in tempstr:   # If scalar is a fraction
                    for paren in range(len(tempstr)):
                        flag[i] = 1
                        if tempstr[paren] == xobj(')_ext', 1) and tempstr[paren + 1] == '\n':
                            # We want to place the vector string after all the right parentheses, because
                            # otherwise, the vector will be in the middle of the string
                            tempstr = tempstr[:paren] + xobj(')_ext', 1)\
                                         + ' '  + vectstrs[i] + tempstr[paren + 1:]
                            break
                elif xobj(')_lower_hook', 1) in tempstr:
                    # We want to place the vector string after all the right parentheses, because
                    # otherwise, the vector will be in the middle of the string. For this reason,
                    # we insert the vector string at the rightmost index.
                    index = tempstr.rfind(xobj(')_lower_hook', 1))
                    if index != -1: # then this character was found in this string
                        flag[i] = 1
                        tempstr = tempstr[:index] + xobj(')_lower_hook', 1)\
                                     + ' '  + vectstrs[i] + tempstr[index + 1:]
                o1[i] = tempstr

        o1 = [x.split('\n') for x in o1]
        n_newlines = max(len(x) for x in o1)  # Width of part in its pretty form

        if 1 in flag:                           # If there was a fractional scalar
            for i, parts in enumerate(o1):
                if len(parts) == 1:             # If part has no newline
                    parts.insert(0, ' ' * (len(parts[0])))
                    flag[i] = 1

        for i, parts in enumerate(o1):
            lengths.append(len(parts[flag[i]]))
            for j in range(n_newlines):
                if j+1 <= len(parts):
                    if j >= len(strs):
                        strs.append(' ' * (sum(lengths[:-1]) +
                                           3*(len(lengths)-1)))
                    if j == flag[i]:
                        strs[flag[i]] += parts[flag[i]] + ' + '
                    else:
                        strs[j] += parts[j] + ' '*(lengths[-1] -
                                                   len(parts[j])+
                                                   3)
                else:
                    if j >= len(strs):
                        strs.append(' ' * (sum(lengths[:-1]) +
                                           3*(len(lengths)-1)))
                    strs[j] += ' '*(lengths[-1]+3)

        return prettyForm('\n'.join([s[:-3] for s in strs]))

    def _print_NDimArray(self, expr):
        from sympy.matrices.immutable import ImmutableMatrix

        if expr.rank() == 0:
            return self._print(expr[()])

        level_str = [[]] + [[] for i in range(expr.rank())]
        shape_ranges = [list(range(i)) for i in expr.shape]
        # leave eventual matrix elements unflattened
        mat = lambda x: ImmutableMatrix(x, evaluate=False)
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(expr[outer_i])
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(level_str[back_outer_i+1])
                else:
                    level_str[back_outer_i].append(mat(
                        level_str[back_outer_i+1]))
                    if len(level_str[back_outer_i + 1]) == 1:
                        level_str[back_outer_i][-1] = mat(
                            [[level_str[back_outer_i][-1]]])
                even = not even
                level_str[back_outer_i+1] = []

        out_expr = level_str[0][0]
        if expr.rank() % 2 == 1:
            out_expr = mat([out_expr])

        return self._print(out_expr)

    def _printer_tensor_indices(self, name, indices, index_map={}):
        center = stringPict(name)
        top = stringPict(" "*center.width())
        bot = stringPict(" "*center.width())

        last_valence = None
        prev_map = None

        for index in indices:
            indpic = self._print(index.args[0])
            if ((index in index_map) or prev_map) and last_valence == index.is_up:
                if index.is_up:
                    top = prettyForm(*stringPict.next(top, ","))
                else:
                    bot = prettyForm(*stringPict.next(bot, ","))
            if index in index_map:
                indpic = prettyForm(*stringPict.next(indpic, "="))
                indpic = prettyForm(*stringPict.next(indpic, self._print(index_map[index])))
                prev_map = True
            else:
                prev_map = False
            if index.is_up:
                top = stringPict(*top.right(indpic))
                center = stringPict(*center.right(" "*indpic.width()))
                bot = stringPict(*bot.right(" "*indpic.width()))
            else:
                bot = stringPict(*bot.right(indpic))
                center = stringPict(*center.right(" "*indpic.width()))
                top = stringPict(*top.right(" "*indpic.width()))
            last_valence = index.is_up

        pict = prettyForm(*center.above(top))
        pict = prettyForm(*pict.below(bot))
        return pict

    def _print_Tensor(self, expr):
        name = expr.args[0].name
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices)

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].name
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        sign, args = expr._get_args_for_traditional_printer()
        args = [
            prettyForm(*self._print(i).parens()) if
            precedence_traditional(i) < PRECEDENCE["Mul"] else self._print(i)
            for i in args
        ]
        pform = prettyForm.__mul__(*args)
        if sign:
            return prettyForm(*pform.left(sign))
        else:
            return pform

    def _print_TensAdd(self, expr):
        args = [
            prettyForm(*self._print(i).parens()) if
            precedence_traditional(i) < PRECEDENCE["Mul"] else self._print(i)
            for i in expr.args
        ]
        return prettyForm.__add__(*args)

    def _print_TensorIndex(self, expr):
        sym = expr.args[0]
        if not expr.is_up:
            sym = -sym
        return self._print(sym)

    def _print_PartialDerivative(self, deriv):
        if self._use_unicode:
            deriv_symbol = U('PARTIAL DIFFERENTIAL')
        else:
            deriv_symbol = r'd'
        x = None

        for variable in reversed(deriv.variables):
            s = self._print(variable)
            ds = prettyForm(*s.left(deriv_symbol))

            if x is None:
                x = ds
            else:
                x = prettyForm(*x.right(' '))
                x = prettyForm(*x.right(ds))

        f = prettyForm(
            binding=prettyForm.FUNC, *self._print(deriv.expr).parens())

        pform = prettyForm(deriv_symbol)

        if len(deriv.variables) > 1:
            pform = pform**self._print(len(deriv.variables))

        pform = prettyForm(*pform.below(stringPict.LINE, x))
        pform.baseline = pform.baseline + 1
        pform = prettyForm(*stringPict.next(pform, f))
        pform.binding = prettyForm.MUL

        return pform

    def _print_Piecewise(self, pexpr):

        P = {}
        for n, ec in enumerate(pexpr.args):
            P[n, 0] = self._print(ec.expr)
            if ec.cond == True:
                P[n, 1] = prettyForm('otherwise')
            else:
                P[n, 1] = prettyForm(
                    *prettyForm('for ').right(self._print(ec.cond)))
        hsep = 2
        vsep = 1
        len_args = len(pexpr.args)

        # max widths
        maxw = [max(P[i, j].width() for i in range(len_args))
                for j in range(2)]

        # FIXME: Refactor this code and matrix into some tabular environment.
        # drawing result
        D = None

        for i in range(len_args):
            D_row = None
            for j in range(2):
                p = P[i, j]
                assert p.width() <= maxw[j]

                wdelta = maxw[j] - p.width()
                wleft = wdelta // 2
                wright = wdelta - wleft

                p = prettyForm(*p.right(' '*wright))
                p = prettyForm(*p.left(' '*wleft))

                if D_row is None:
                    D_row = p
                    continue

                D_row = prettyForm(*D_row.right(' '*hsep))  # h-spacer
                D_row = prettyForm(*D_row.right(p))
            if D is None:
                D = D_row       # first row in a picture
                continue

            # v-spacer
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))

            D = prettyForm(*D.below(D_row))

        D = prettyForm(*D.parens('{', ''))
        D.baseline = D.height()//2
        D.binding = prettyForm.OPEN
        return D

    def _print_ITE(self, ite):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(ite.rewrite(Piecewise))

    def _hprint_vec(self, v):
        D = None

        for a in v:
            p = a
            if D is None:
                D = p
            else:
                D = prettyForm(*D.right(', '))
                D = prettyForm(*D.right(p))
        if D is None:
            D = stringPict(' ')

        return D

    def _hprint_vseparator(self, p1, p2, left=None, right=None, delimiter='', ifascii_nougly=False):
        if ifascii_nougly and not self._use_unicode:
            return self._print_seq((p1, '|', p2), left=left, right=right,
                                   delimiter=delimiter, ifascii_nougly=True)
        tmp = self._print_seq((p1, p2,), left=left, right=right, delimiter=delimiter)
        sep = stringPict(vobj('|', tmp.height()), baseline=tmp.baseline)
        return self._print_seq((p1, sep, p2), left=left, right=right,
                               delimiter=delimiter)

    def _print_hyper(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment
        ap = [self._print(a) for a in e.ap]
        bq = [self._print(b) for b in e.bq]

        P = self._print(e.argument)
        P.baseline = P.height()//2

        # Drawing result - first create the ap, bq vectors
        D = None
        for v in [ap, bq]:
            D_row = self._hprint_vec(v)
            if D is None:
                D = D_row       # first row in a picture
            else:
                D = prettyForm(*D.below(' '))
                D = prettyForm(*D.below(D_row))

        # make sure that the argument `z' is centred vertically
        D.baseline = D.height()//2

        # insert horizontal separator
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # insert separating `|`
        D = self._hprint_vseparator(D, P)

        # add parens
        D = prettyForm(*D.parens('(', ')'))

        # create the F symbol
        above = D.height()//2 - 1
        below = D.height() - above - 1

        sz, t, b, add, img = annotated('F')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)
        add = (sz + 1)//2

        F = prettyForm(*F.left(self._print(len(e.ap))))
        F = prettyForm(*F.right(self._print(len(e.bq))))
        F.baseline = above + add

        D = prettyForm(*F.right(' ', D))

        return D

    def _print_meijerg(self, e):
        # FIXME refactor Matrix, Piecewise, and this into a tabular environment

        v = {}
        v[(0, 0)] = [self._print(a) for a in e.an]
        v[(0, 1)] = [self._print(a) for a in e.aother]
        v[(1, 0)] = [self._print(b) for b in e.bm]
        v[(1, 1)] = [self._print(b) for b in e.bother]

        P = self._print(e.argument)
        P.baseline = P.height()//2

        vp = {}
        for idx in v:
            vp[idx] = self._hprint_vec(v[idx])

        for i in range(2):
            maxw = max(vp[(0, i)].width(), vp[(1, i)].width())
            for j in range(2):
                s = vp[(j, i)]
                left = (maxw - s.width()) // 2
                right = maxw - left - s.width()
                s = prettyForm(*s.left(' ' * left))
                s = prettyForm(*s.right(' ' * right))
                vp[(j, i)] = s

        D1 = prettyForm(*vp[(0, 0)].right('  ', vp[(0, 1)]))
        D1 = prettyForm(*D1.below(' '))
        D2 = prettyForm(*vp[(1, 0)].right('  ', vp[(1, 1)]))
        D = prettyForm(*D1.below(D2))

        # make sure that the argument `z' is centred vertically
        D.baseline = D.height()//2

        # insert horizontal separator
        P = prettyForm(*P.left(' '))
        D = prettyForm(*D.right(' '))

        # insert separating `|`
        D = self._hprint_vseparator(D, P)

        # add parens
        D = prettyForm(*D.parens('(', ')'))

        # create the G symbol
        above = D.height()//2 - 1
        below = D.height() - above - 1

        sz, t, b, add, img = annotated('G')
        F = prettyForm('\n' * (above - t) + img + '\n' * (below - b),
                       baseline=above + sz)

        pp = self._print(len(e.ap))
        pq = self._print(len(e.bq))
        pm = self._print(len(e.bm))
        pn = self._print(len(e.an))

        def adjust(p1, p2):
            diff = p1.width() - p2.width()
            if diff == 0:
                return p1, p2
            elif diff > 0:
                return p1, prettyForm(*p2.left(' '*diff))
            else:
                return prettyForm(*p1.left(' '*-diff)), p2
        pp, pm = adjust(pp, pm)
        pq, pn = adjust(pq, pn)
        pu = prettyForm(*pm.right(', ', pn))
        pl = prettyForm(*pp.right(', ', pq))

        ht = F.baseline - above - 2
        if ht > 0:
            pu = prettyForm(*pu.below('\n'*ht))
        p = prettyForm(*pu.below(pl))

        F.baseline = above
        F = prettyForm(*F.right(p))

        F.baseline = above + add

        D = prettyForm(*F.right(' ', D))

        return D

    def _print_ExpBase(self, e):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        base = prettyForm(pretty_atom('Exp1', 'e'))
        return base ** self._print(e.args[0])

    def _print_Exp1(self, e):
        return prettyForm(pretty_atom('Exp1', 'e'))

    def _print_Function(self, e, sort=False, func_name=None, left='(',
                        right=')'):
        # optional argument func_name for supplying custom names
        # XXX works only for applied functions
        return self._helper_print_function(e.func, e.args, sort=sort, func_name=func_name, left=left, right=right)

    def _print_mathieuc(self, e):
        return self._print_Function(e, func_name='C')

    def _print_mathieus(self, e):
        return self._print_Function(e, func_name='S')

    def _print_mathieucprime(self, e):
        return self._print_Function(e, func_name="C'")

    def _print_mathieusprime(self, e):
        return self._print_Function(e, func_name="S'")

    def _helper_print_function(self, func, args, sort=False, func_name=None,
                               delimiter=', ', elementwise=False, left='(',
                               right=')'):
        if sort:
            args = sorted(args, key=default_sort_key)

        if not func_name and hasattr(func, "__name__"):
            func_name = func.__name__

        if func_name:
            prettyFunc = self._print(Symbol(func_name))
        else:
            prettyFunc = prettyForm(*self._print(func).parens())

        if elementwise:
            if self._use_unicode:
                circ = pretty_atom('Modifier Letter Low Ring')
            else:
                circ = '.'
            circ = self._print(circ)
            prettyFunc = prettyForm(
                binding=prettyForm.LINE,
                *stringPict.next(prettyFunc, circ)
            )

        prettyArgs = prettyForm(*self._print_seq(args, delimiter=delimiter).parens(
                                                 left=left, right=right))

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform

    def _print_ElementwiseApplyFunction(self, e):
        func = e.function
        arg = e.expr
        args = [arg]
        return self._helper_print_function(func, args, delimiter="", elementwise=True)

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.zeta_functions import lerchphi
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: [greek_unicode['delta'], 'delta'],
                gamma: [greek_unicode['Gamma'], 'Gamma'],
                lerchphi: [greek_unicode['Phi'], 'lerchphi'],
                lowergamma: [greek_unicode['gamma'], 'gamma'],
                beta: [greek_unicode['Beta'], 'B'],
                DiracDelta: [greek_unicode['delta'], 'delta'],
                Chi: ['Chi', 'Chi']}

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                if self._use_unicode:
                    return prettyForm(self._special_function_classes[cls][0])
                else:
                    return prettyForm(self._special_function_classes[cls][1])
        func_name = expr.__name__
        return prettyForm(pretty_symbol(func_name))

    def _print_GeometryEntity(self, expr):
        # GeometryEntity is based on Tuple but should not print like a Tuple
        return self.emptyPrinter(expr)

    def _print_polylog(self, e):
        subscript = self._print(e.args[0])
        if self._use_unicode and is_subscriptable_in_unicode(subscript):
            return self._print_Function(Function('Li_%s' % subscript)(e.args[1]))
        return self._print_Function(e)

    def _print_lerchphi(self, e):
        func_name = greek_unicode['Phi'] if self._use_unicode else 'lerchphi'
        return self._print_Function(e, func_name=func_name)

    def _print_dirichlet_eta(self, e):
        func_name = greek_unicode['eta'] if self._use_unicode else 'dirichlet_eta'
        return self._print_Function(e, func_name=func_name)

    def _print_Heaviside(self, e):
        func_name = greek_unicode['theta'] if self._use_unicode else 'Heaviside'
        if e.args[1] is S.Half:
            pform = prettyForm(*self._print(e.args[0]).parens())
            pform = prettyForm(*pform.left(func_name))
            return pform
        else:
            return self._print_Function(e, func_name=func_name)

    def _print_fresnels(self, e):
        return self._print_Function(e, func_name="S")

    def _print_fresnelc(self, e):
        return self._print_Function(e, func_name="C")

    def _print_airyai(self, e):
        return self._print_Function(e, func_name="Ai")

    def _print_airybi(self, e):
        return self._print_Function(e, func_name="Bi")

    def _print_airyaiprime(self, e):
        return self._print_Function(e, func_name="Ai'")

    def _print_airybiprime(self, e):
        return self._print_Function(e, func_name="Bi'")

    def _print_LambertW(self, e):
        return self._print_Function(e, func_name="W")

    def _print_Covariance(self, e):
        return self._print_Function(e, func_name="Cov")

    def _print_Variance(self, e):
        return self._print_Function(e, func_name="Var")

    def _print_Probability(self, e):
        return self._print_Function(e, func_name="P")

    def _print_Expectation(self, e):
        return self._print_Function(e, func_name="E", left='[', right=']')

    def _print_Lambda(self, e):
        expr = e.expr
        sig = e.signature
        if self._use_unicode:
            arrow = f" {pretty_atom('ArrowFromBar')} "
        else:
            arrow = " -> "
        if len(sig) == 1 and sig[0].is_symbol:
            sig = sig[0]
        var_form = self._print(sig)

        return prettyForm(*stringPict.next(var_form, arrow, self._print(expr)), binding=8)

    def _print_Order(self, expr):
        pform = self._print(expr.expr)
        if (expr.point and any(p != S.Zero for p in expr.point)) or \
           len(expr.variables) > 1:
            pform = prettyForm(*pform.right("; "))
            if len(expr.variables) > 1:
                pform = prettyForm(*pform.right(self._print(expr.variables)))
            elif len(expr.variables):
                pform = prettyForm(*pform.right(self._print(expr.variables[0])))
            if self._use_unicode:
                pform = prettyForm(*pform.right(f" {pretty_atom('Arrow')} "))
            else:
                pform = prettyForm(*pform.right(" -> "))
            if len(expr.point) > 1:
                pform = prettyForm(*pform.right(self._print(expr.point)))
            else:
                pform = prettyForm(*pform.right(self._print(expr.point[0])))
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left("O"))
        return pform

    def _print_SingularityFunction(self, e):
        if self._use_unicode:
            shift = self._print(e.args[0]-e.args[1])
            n = self._print(e.args[2])
            base = prettyForm("<")
            base = prettyForm(*base.right(shift))
            base = prettyForm(*base.right(">"))
            pform = base**n
            return pform
        else:
            n = self._print(e.args[2])
            shift = self._print(e.args[0]-e.args[1])
            base = self._print_seq(shift, "<", ">", ' ')
            return base**n

    def _print_beta(self, e):
        func_name = greek_unicode['Beta'] if self._use_unicode else 'B'
        return self._print_Function(e, func_name=func_name)

    def _print_betainc(self, e):
        func_name = "B'"
        return self._print_Function(e, func_name=func_name)

    def _print_betainc_regularized(self, e):
        func_name = 'I'
        return self._print_Function(e, func_name=func_name)

    def _print_gamma(self, e):
        func_name = greek_unicode['Gamma'] if self._use_unicode else 'Gamma'
        return self._print_Function(e, func_name=func_name)

    def _print_uppergamma(self, e):
        func_name = greek_unicode['Gamma'] if self._use_unicode else 'Gamma'
        return self._print_Function(e, func_name=func_name)

    def _print_lowergamma(self, e):
        func_name = greek_unicode['gamma'] if self._use_unicode else 'lowergamma'
        return self._print_Function(e, func_name=func_name)

    def _print_DiracDelta(self, e):
        if self._use_unicode:
            if len(e.args) == 2:
                a = prettyForm(greek_unicode['delta'])
                b = self._print(e.args[1])
                b = prettyForm(*b.parens())
                c = self._print(e.args[0])
                c = prettyForm(*c.parens())
                pform = a**b
                pform = prettyForm(*pform.right(' '))
                pform = prettyForm(*pform.right(c))
                return pform
            pform = self._print(e.args[0])
            pform = prettyForm(*pform.parens())
            pform = prettyForm(*pform.left(greek_unicode['delta']))
            return pform
        else:
            return self._print_Function(e)

    def _print_expint(self, e):
        subscript = self._print(e.args[0])
        if self._use_unicode and is_subscriptable_in_unicode(subscript):
            return self._print_Function(Function('E_%s' % subscript)(e.args[1]))
        return self._print_Function(e)

    def _print_Chi(self, e):
        # This needs a special case since otherwise it comes out as greek
        # letter chi...
        prettyFunc = prettyForm("Chi")
        prettyArgs = prettyForm(*self._print_seq(e.args).parens())

        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))

        # store pform parts so it can be reassembled e.g. when powered
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs

        return pform

    def _print_elliptic_e(self, e):
        pforma0 = self._print(e.args[0])
        if len(e.args) == 1:
            pform = pforma0
        else:
            pforma1 = self._print(e.args[1])
            pform = self._hprint_vseparator(pforma0, pforma1)
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('E'))
        return pform

    def _print_elliptic_k(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('K'))
        return pform

    def _print_elliptic_f(self, e):
        pforma0 = self._print(e.args[0])
        pforma1 = self._print(e.args[1])
        pform = self._hprint_vseparator(pforma0, pforma1)
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left('F'))
        return pform

    def _print_elliptic_pi(self, e):
        name = greek_unicode['Pi'] if self._use_unicode else 'Pi'
        pforma0 = self._print(e.args[0])
        pforma1 = self._print(e.args[1])
        if len(e.args) == 2:
            pform = self._hprint_vseparator(pforma0, pforma1)
        else:
            pforma2 = self._print(e.args[2])
            pforma = self._hprint_vseparator(pforma1, pforma2, ifascii_nougly=False)
            pforma = prettyForm(*pforma.left('; '))
            pform = prettyForm(*pforma.left(pforma0))
        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left(name))
        return pform

    def _print_GoldenRatio(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_symbol('phi'))
        return self._print(Symbol("GoldenRatio"))

    def _print_EulerGamma(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_symbol('gamma'))
        return self._print(Symbol("EulerGamma"))

    def _print_Catalan(self, expr):
        return self._print(Symbol("G"))

    def _print_Mod(self, expr):
        pform = self._print(expr.args[0])
        if pform.binding > prettyForm.MUL:
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right(' mod '))
        pform = prettyForm(*pform.right(self._print(expr.args[1])))
        pform.binding = prettyForm.OPEN
        return pform

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)
        pforms, indices = [], []

        def pretty_negative(pform, index):
            """Prepend a minus sign to a pretty form. """
            #TODO: Move this code to prettyForm
            if index == 0:
                if pform.height() > 1:
                    pform_neg = '- '
                else:
                    pform_neg = '-'
            else:
                pform_neg = ' - '

            if (pform.binding > prettyForm.NEG
                or pform.binding == prettyForm.ADD):
                p = stringPict(*pform.parens())
            else:
                p = pform
            p = stringPict.next(pform_neg, p)
            # Lower the binding to NEG, even if it was higher. Otherwise, it
            # will print as a + ( - (b)), instead of a - (b).
            return prettyForm(binding=prettyForm.NEG, *p)

        for i, term in enumerate(terms):
            if term.is_Mul and term.could_extract_minus_sign():
                coeff, other = term.as_coeff_mul(rational=False)
                if coeff == -1:
                    negterm = Mul(*other, evaluate=False)
                else:
                    negterm = Mul(-coeff, *other, evaluate=False)
                pform = self._print(negterm)
                pforms.append(pretty_negative(pform, i))
            elif term.is_Rational and term.q > 1:
                pforms.append(None)
                indices.append(i)
            elif term.is_Number and term < 0:
                pform = self._print(-term)
                pforms.append(pretty_negative(pform, i))
            elif term.is_Relational:
                pforms.append(prettyForm(*self._print(term).parens()))
            else:
                pforms.append(self._print(term))

        if indices:
            large = True

            for pform in pforms:
                if pform is not None and pform.height() > 1:
                    break
            else:
                large = False

            for i in indices:
                term, negative = terms[i], False

                if term < 0:
                    term, negative = -term, True

                if large:
                    pform = prettyForm(str(term.p))/prettyForm(str(term.q))
                else:
                    pform = self._print(term)

                if negative:
                    pform = pretty_negative(pform, i)

                pforms[i] = pform

        return prettyForm.__add__(*pforms)

    def _print_Mul(self, product):
        from sympy.physics.units import Quantity

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        args = product.args
        if args[0] is S.One or any(isinstance(arg, Number) for arg in args[1:]):
            strargs = list(map(self._print, args))
            # XXX: This is a hack to work around the fact that
            # prettyForm.__mul__ absorbs a leading -1 in the args. Probably it
            # would be better to fix this in prettyForm.__mul__ instead.
            negone = strargs[0] == '-1'
            if negone:
                strargs[0] = prettyForm('1', 0, 0)
            obj = prettyForm.__mul__(*strargs)
            if negone:
                obj = prettyForm('-' + obj.s, obj.baseline, obj.binding)
            return obj

        a = []  # items in the numerator
        b = []  # items that are in the denominator (if any)

        if self.order not in ('old', 'none'):
            args = product.as_ordered_factors()
        else:
            args = list(product.args)

        # If quantities are present append them at the back
        args = sorted(args, key=lambda x: isinstance(x, Quantity) or
                     (isinstance(x, Pow) and isinstance(x.base, Quantity)))

        # Gather terms for numerator/denominator
        for item in args:
            if item.is_commutative and item.is_Pow and item.exp.is_Rational and item.exp.is_negative:
                if item.exp != -1:
                    b.append(Pow(item.base, -item.exp, evaluate=False))
                else:
                    b.append(Pow(item.base, -item.exp))
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append( Rational(item.p) )
                if item.q != 1:
                    b.append( Rational(item.q) )
            else:
                a.append(item)

        # Convert to pretty forms. Parentheses are added by `__mul__`.
        a = [self._print(ai) for ai in a]
        b = [self._print(bi) for bi in b]

        # Construct a pretty form
        if len(b) == 0:
            return prettyForm.__mul__(*a)
        else:
            if len(a) == 0:
                a.append( self._print(S.One) )
            return prettyForm.__mul__(*a)/prettyForm.__mul__(*b)

    # A helper function for _print_Pow to print x**(1/n)
    def _print_nth_root(self, base, root):
        bpretty = self._print(base)

        # In very simple cases, use a single-char root sign
        if (self._settings['use_unicode_sqrt_char'] and self._use_unicode
            and root == 2 and bpretty.height() == 1
            and (bpretty.width() == 1
                 or (base.is_Integer and base.is_nonnegative))):
            return prettyForm(*bpretty.left(nth_root[2]))

        # Construct root sign, start with the \/ shape
        _zZ = xobj('/', 1)
        rootsign = xobj('\\', 1) + _zZ
        # Constructing the number to put on root
        rpretty = self._print(root)
        # roots look bad if they are not a single line
        if rpretty.height() != 1:
            return self._print(base)**self._print(1/root)
        # If power is half, no number should appear on top of root sign
        exp = '' if root == 2 else str(rpretty).ljust(2)
        if len(exp) > 2:
            rootsign = ' '*(len(exp) - 2) + rootsign
        # Stack the exponent
        rootsign = stringPict(exp + '\n' + rootsign)
        rootsign.baseline = 0
        # Diagonal: length is one less than height of base
        linelength = bpretty.height() - 1
        diagonal = stringPict('\n'.join(
            ' '*(linelength - i - 1) + _zZ + ' '*i
            for i in range(linelength)
        ))
        # Put baseline just below lowest line: next to exp
        diagonal.baseline = linelength - 1
        # Make the root symbol
        rootsign = prettyForm(*rootsign.right(diagonal))
        # Det the baseline to match contents to fix the height
        # but if the height of bpretty is one, the rootsign must be one higher
        rootsign.baseline = max(1, bpretty.baseline)
        #build result
        s = prettyForm(hobj('_', 2 + bpretty.width()))
        s = prettyForm(*bpretty.above(s))
        s = prettyForm(*s.left(rootsign))
        return s

    def _print_Pow(self, power):
        from sympy.simplify.simplify import fraction
        b, e = power.as_base_exp()
        if power.is_commutative:
            if e is S.NegativeOne:
                return prettyForm("1")/self._print(b)
            n, d = fraction(e)
            if n is S.One and d.is_Atom and not e.is_Integer and (e.is_Rational or d.is_Symbol) \
                    and self._settings['root_notation']:
                return self._print_nth_root(b, d)
            if e.is_Rational and e < 0:
                return prettyForm("1")/self._print(Pow(b, -e, evaluate=False))

        if b.is_Relational:
            return prettyForm(*self._print(b).parens()).__pow__(self._print(e))

        return self._print(b)**self._print(e)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def __print_numer_denom(self, p, q):
        if q == 1:
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)
            else:
                return prettyForm(str(p))
        elif abs(p) >= 10 and abs(q) >= 10:
            # If more than one digit in numer and denom, print larger fraction
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)/prettyForm(str(q))
                # Old printing method:
                #pform = prettyForm(str(-p))/prettyForm(str(q))
                #return prettyForm(binding=prettyForm.NEG, *pform.left('- '))
            else:
                return prettyForm(str(p))/prettyForm(str(q))
        else:
            return None

    def _print_Rational(self, expr):
        result = self.__print_numer_denom(expr.p, expr.q)

        if result is not None:
            return result
        else:
            return self.emptyPrinter(expr)

    def _print_Fraction(self, expr):
        result = self.__print_numer_denom(expr.numerator, expr.denominator)

        if result is not None:
            return result
        else:
            return self.emptyPrinter(expr)

    def _print_ProductSet(self, p):
        if len(p.sets) >= 1 and not has_variety(p.sets):
            return self._print(p.sets[0]) ** self._print(len(p.sets))
        else:
            prod_char = pretty_atom('Multiplication') if self._use_unicode else 'x'
            return self._print_seq(p.sets, None, None, ' %s ' % prod_char,
                                   parenthesize=lambda set: set.is_Union or
                                   set.is_Intersection or set.is_ProductSet)

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_seq(items, '{', '}', ', ' )

    def _print_Range(self, s):

        if self._use_unicode:
            dots = pretty_atom('Dots')
        else:
            dots = '...'

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

        return self._print_seq(printset, '{', '}', ', ' )

    def _print_Interval(self, i):
        if i.start == i.end:
            return self._print_seq(i.args[:1], '{', '}')

        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return self._print_seq(i.args[:2], left, right)

    def _print_AccumulationBounds(self, i):
        left = '<'
        right = '>'

        return self._print_seq(i.args[:2], left, right)

    def _print_Intersection(self, u):

        delimiter = ' %s ' % pretty_atom('Intersection', 'n')

        return self._print_seq(u.args, None, None, delimiter,
                               parenthesize=lambda set: set.is_ProductSet or
                               set.is_Union or set.is_Complement)

    def _print_Union(self, u):

        union_delimiter = ' %s ' % pretty_atom('Union', 'U')

        return self._print_seq(u.args, None, None, union_delimiter,
                               parenthesize=lambda set: set.is_ProductSet or
                               set.is_Intersection or set.is_Complement)

    def _print_SymmetricDifference(self, u):
        if not self._use_unicode:
            raise NotImplementedError("ASCII pretty printing of SymmetricDifference is not implemented")

        sym_delimeter = ' %s ' % pretty_atom('SymmetricDifference')

        return self._print_seq(u.args, None, None, sym_delimeter)

    def _print_Complement(self, u):

        delimiter = r' \ '

        return self._print_seq(u.args, None, None, delimiter,
             parenthesize=lambda set: set.is_ProductSet or set.is_Intersection
                               or set.is_Union)

    def _print_ImageSet(self, ts):
        if self._use_unicode:
            inn = pretty_atom("SmallElementOf")
        else:
            inn = 'in'
        fun = ts.lamda
        sets = ts.base_sets
        signature = fun.signature
        expr = self._print(fun.expr)

        # TODO: the stuff to the left of the | and the stuff to the right of
        # the | should have independent baselines, that way something like
        # ImageSet(Lambda(x, 1/x**2), S.Naturals) prints the "x in N" part
        # centered on the right instead of aligned with the fraction bar on
        # the left. The same also applies to ConditionSet and ComplexRegion
        if len(signature) == 1:
            S = self._print_seq((signature[0], inn, sets[0]),
                                delimiter=' ')
            return self._hprint_vseparator(expr, S,
                                           left='{', right='}',
                                           ifascii_nougly=True, delimiter=' ')
        else:
            pargs = tuple(j for var, setv in zip(signature, sets) for j in
                          (var, ' ', inn, ' ', setv, ", "))
            S = self._print_seq(pargs[:-1], delimiter='')
            return self._hprint_vseparator(expr, S,
                                           left='{', right='}',
                                           ifascii_nougly=True, delimiter=' ')

    def _print_ConditionSet(self, ts):
        if self._use_unicode:
            inn = pretty_atom('SmallElementOf')
            # using _and because and is a keyword and it is bad practice to
            # overwrite them
            _and = pretty_atom('And')
        else:
            inn = 'in'
            _and = 'and'

        variables = self._print_seq(Tuple(ts.sym))
        as_expr = getattr(ts.condition, 'as_expr', None)
        if as_expr is not None:
            cond = self._print(ts.condition.as_expr())
        else:
            cond = self._print(ts.condition)
            if self._use_unicode:
                cond = self._print(cond)
                cond = prettyForm(*cond.parens())

        if ts.base_set is S.UniversalSet:
            return self._hprint_vseparator(variables, cond, left="{",
                                           right="}", ifascii_nougly=True,
                                           delimiter=' ')

        base = self._print(ts.base_set)
        C = self._print_seq((variables, inn, base, _and, cond),
                            delimiter=' ')
        return self._hprint_vseparator(variables, C, left="{", right="}",
                                       ifascii_nougly=True, delimiter=' ')

    def _print_ComplexRegion(self, ts):
        if self._use_unicode:
            inn = pretty_atom('SmallElementOf')
        else:
            inn = 'in'
        variables = self._print_seq(ts.variables)
        expr = self._print(ts.expr)
        prodsets = self._print(ts.sets)

        C = self._print_seq((variables, inn, prodsets),
                            delimiter=' ')
        return self._hprint_vseparator(expr, C, left="{", right="}",
                                       ifascii_nougly=True, delimiter=' ')

    def _print_Contains(self, e):
        var, set = e.args
        if self._use_unicode:
            el = f" {pretty_atom('ElementOf')} "
            return prettyForm(*stringPict.next(self._print(var),
                                               el, self._print(set)), binding=8)
        else:
            return prettyForm(sstr(e))

    def _print_FourierSeries(self, s):
        if s.an.formula is S.Zero and s.bn.formula is S.Zero:
            return self._print(s.a0)
        if self._use_unicode:
            dots = pretty_atom('Dots')
        else:
            dots = '...'
        return self._print_Add(s.truncate()) + self._print(dots)

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_SetExpr(self, se):
        pretty_set = prettyForm(*self._print(se.set).parens())
        pretty_name = self._print(Symbol("SetExpr"))
        return prettyForm(*pretty_name.right(pretty_set))

    def _print_SeqFormula(self, s):
        if self._use_unicode:
            dots = pretty_atom('Dots')
        else:
            dots = '...'

        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            raise NotImplementedError("Pretty printing of sequences with symbolic bound not implemented")

        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2),
                s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(dots)
            printset = tuple(printset)
        else:
            printset = tuple(s)
        return self._print_list(printset)

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_seq(self, seq, left=None, right=None, delimiter=', ',
            parenthesize=lambda x: False, ifascii_nougly=True):

        pforms = []
        for item in seq:
            pform = self._print(item)
            if parenthesize(item):
                pform = prettyForm(*pform.parens())
            if pforms:
                pforms.append(delimiter)
            pforms.append(pform)

        if not pforms:
            s = stringPict('')
        else:
            s = prettyForm(*stringPict.next(*pforms))

        s = prettyForm(*s.parens(left, right, ifascii_nougly=ifascii_nougly))
        return s

    def join(self, delimiter, args):
        pform = None

        for arg in args:
            if pform is None:
                pform = arg
            else:
                pform = prettyForm(*pform.right(delimiter))
                pform = prettyForm(*pform.right(arg))

        if pform is None:
            return prettyForm("")
        else:
            return pform

    def _print_list(self, l):
        return self._print_seq(l, '[', ']')

    def _print_tuple(self, t):
        if len(t) == 1:
            ptuple = prettyForm(*stringPict.next(self._print(t[0]), ','))
            return prettyForm(*ptuple.parens('(', ')', ifascii_nougly=True))
        else:
            return self._print_seq(t, '(', ')')

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for k in keys:
            K = self._print(k)
            V = self._print(d[k])
            s = prettyForm(*stringPict.next(K, ': ', V))

            items.append(s)

        return self._print_seq(items, '{', '}')

    def _print_Dict(self, d):
        return self._print_dict(d)

    def _print_set(self, s):
        if not s:
            return prettyForm('set()')
        items = sorted(s, key=default_sort_key)
        pretty = self._print_seq(items)
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        return pretty

    def _print_frozenset(self, s):
        if not s:
            return prettyForm('frozenset()')
        items = sorted(s, key=default_sort_key)
        pretty = self._print_seq(items)
        pretty = prettyForm(*pretty.parens('{', '}', ifascii_nougly=True))
        pretty = prettyForm(*pretty.parens('(', ')', ifascii_nougly=True))
        pretty = prettyForm(*stringPict.next(type(s).__name__, pretty))
        return pretty

    def _print_UniversalSet(self, s):
        if self._use_unicode:
            return prettyForm(pretty_atom('Universe'))
        else:
            return prettyForm('UniversalSet')

    def _print_PolyRing(self, ring):
        return prettyForm(sstr(ring))

    def _print_FracField(self, field):
        return prettyForm(sstr(field))

    def _print_FreeGroupElement(self, elm):
        return prettyForm(str(elm))

    def _print_PolyElement(self, poly):
        return prettyForm(sstr(poly))

    def _print_FracElement(self, frac):
        return prettyForm(sstr(frac))

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_ComplexRootOf(self, expr):
        args = [self._print_Add(expr.expr, order='lex'), expr.index]
        pform = prettyForm(*self._print_seq(args).parens())
        pform = prettyForm(*pform.left('CRootOf'))
        return pform

    def _print_RootSum(self, expr):
        args = [self._print_Add(expr.expr, order='lex')]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        pform = prettyForm(*self._print_seq(args).parens())
        pform = prettyForm(*pform.left('RootSum'))

        return pform

    def _print_FiniteField(self, expr):
        if self._use_unicode:
            form = f"{pretty_atom('Integers')}_%d"
        else:
            form = 'GF(%d)'

        return prettyForm(pretty_symbol(form % expr.mod))

    def _print_IntegerRing(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom('Integers'))
        else:
            return prettyForm('ZZ')

    def _print_RationalField(self, expr):
        if self._use_unicode:
            return prettyForm(pretty_atom('Rationals'))
        else:
            return prettyForm('QQ')

    def _print_RealField(self, domain):
        if self._use_unicode:
            prefix = pretty_atom("Reals")
        else:
            prefix = 'RR'

        if domain.has_default_precision:
            return prettyForm(prefix)
        else:
            return self._print(pretty_symbol(prefix + "_" + str(domain.precision)))

    def _print_ComplexField(self, domain):
        if self._use_unicode:
            prefix = pretty_atom('Complexes')
        else:
            prefix = 'CC'

        if domain.has_default_precision:
            return prettyForm(prefix)
        else:
            return self._print(pretty_symbol(prefix + "_" + str(domain.precision)))

    def _print_PolynomialRing(self, expr):
        args = list(expr.symbols)

        if not expr.order.is_default:
            order = prettyForm(*prettyForm("order=").right(self._print(expr.order)))
            args.append(order)

        pform = self._print_seq(args, '[', ']')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    def _print_FractionField(self, expr):
        args = list(expr.symbols)

        if not expr.order.is_default:
            order = prettyForm(*prettyForm("order=").right(self._print(expr.order)))
            args.append(order)

        pform = self._print_seq(args, '(', ')')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    def _print_PolynomialRingBase(self, expr):
        g = expr.symbols
        if str(expr.order) != str(expr.default_order):
            g = g + ("order=" + str(expr.order),)
        pform = self._print_seq(g, '[', ']')
        pform = prettyForm(*pform.left(self._print(expr.domain)))

        return pform

    def _print_GroebnerBasis(self, basis):
        exprs = [ self._print_Add(arg, order=basis.order)
                  for arg in basis.exprs ]
        exprs = prettyForm(*self.join(", ", exprs).parens(left="[", right="]"))

        gens = [ self._print(gen) for gen in basis.gens ]

        domain = prettyForm(
            *prettyForm("domain=").right(self._print(basis.domain)))
        order = prettyForm(
            *prettyForm("order=").right(self._print(basis.order)))

        pform = self.join(", ", [exprs] + gens + [domain, order])

        pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.left(basis.__class__.__name__))

        return pform

    def _print_Subs(self, e):
        pform = self._print(e.expr)
        pform = prettyForm(*pform.parens())

        h = pform.height() if pform.height() > 1 else 2
        rvert = stringPict(vobj('|', h), baseline=pform.baseline)
        pform = prettyForm(*pform.right(rvert))

        b = pform.baseline
        pform.baseline = pform.height() - 1
        pform = prettyForm(*pform.right(self._print_seq([
            self._print_seq((self._print(v[0]), xsym('=='), self._print(v[1])),
                delimiter='') for v in zip(e.variables, e.point) ])))

        pform.baseline = b
        return pform

    def _print_number_function(self, e, name):
        # Print name_arg[0] for one argument or name_arg[0](arg[1])
        # for more than one argument
        pform = prettyForm(name)
        arg = self._print(e.args[0])
        pform_arg = prettyForm(" "*arg.width())
        pform_arg = prettyForm(*pform_arg.below(arg))
        pform = prettyForm(*pform.right(pform_arg))
        if len(e.args) == 1:
            return pform
        m, x = e.args
        # TODO: copy-pasted from _print_Function: can we do better?
        prettyFunc = pform
        prettyArgs = prettyForm(*self._print_seq([x]).parens())
        pform = prettyForm(
            binding=prettyForm.FUNC, *stringPict.next(prettyFunc, prettyArgs))
        pform.prettyFunc = prettyFunc
        pform.prettyArgs = prettyArgs
        return pform

    def _print_euler(self, e):
        return self._print_number_function(e, "E")

    def _print_catalan(self, e):
        return self._print_number_function(e, "C")

    def _print_bernoulli(self, e):
        return self._print_number_function(e, "B")

    _print_bell = _print_bernoulli

    def _print_lucas(self, e):
        return self._print_number_function(e, "L")

    def _print_fibonacci(self, e):
        return self._print_number_function(e, "F")

    def _print_tribonacci(self, e):
        return self._print_number_function(e, "T")

    def _print_stieltjes(self, e):
        if self._use_unicode:
            return self._print_number_function(e, greek_unicode['gamma'])
        else:
            return self._print_number_function(e, "stieltjes")

    def _print_KroneckerDelta(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.right(prettyForm(',')))
        pform = prettyForm(*pform.right(self._print(e.args[1])))
        if self._use_unicode:
            a = stringPict(pretty_symbol('delta'))
        else:
            a = stringPict('d')
        b = pform
        top = stringPict(*b.left(' '*a.width()))
        bot = stringPict(*a.right(' '*b.width()))
        return prettyForm(binding=prettyForm.POW, *bot.below(top))

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            pform = self._print('Domain: ')
            pform = prettyForm(*pform.right(self._print(d.as_boolean())))
            return pform
        elif hasattr(d, 'set'):
            pform = self._print('Domain: ')
            pform = prettyForm(*pform.right(self._print(d.symbols)))
            pform = prettyForm(*pform.right(self._print(' in ')))
            pform = prettyForm(*pform.right(self._print(d.set)))
            return pform
        elif hasattr(d, 'symbols'):
            pform = self._print('Domain on ')
            pform = prettyForm(*pform.right(self._print(d.symbols)))
            return pform
        else:
            return self._print(None)

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(pretty_symbol(object.name))

    def _print_Morphism(self, morphism):
        arrow = xsym("-->")

        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        tail = domain.right(arrow, codomain)[0]

        return prettyForm(tail)

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(pretty_symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return prettyForm(pretty_name.right(":", pretty_morphism)[0])

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(
            NamedMorphism(morphism.domain, morphism.codomain, "id"))

    def _print_CompositeMorphism(self, morphism):

        circle = xsym(".")

        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [pretty_symbol(component.name) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = circle.join(component_names_list) + ":"

        pretty_name = self._print(component_names)
        pretty_morphism = self._print_Morphism(morphism)
        return prettyForm(pretty_name.right(pretty_morphism)[0])

    def _print_Category(self, category):
        return self._print(pretty_symbol(category.name))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        pretty_result = self._print(diagram.premises)
        if diagram.conclusions:
            results_arrow = " %s " % xsym("==>")

            pretty_conclusions = self._print(diagram.conclusions)[0]
            pretty_result = pretty_result.right(
                results_arrow, pretty_conclusions)

        return prettyForm(pretty_result[0])

    def _print_DiagramGrid(self, grid):
        from sympy.matrices import Matrix
        matrix = Matrix([[grid[i, j] if grid[i, j] else Symbol(" ")
                          for j in range(grid.width)]
                         for i in range(grid.height)])
        return self._print_matrix_contents(matrix)

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return self._print_seq(m, '[', ']')

    def _print_SubModule(self, M):
        gens = [[M.ring.to_sympy(g) for g in gen] for gen in M.gens]
        return self._print_seq(gens, '<', '>')

    def _print_FreeModule(self, M):
        return self._print(M.ring)**self._print(M.rank)

    def _print_ModuleImplementedIdeal(self, M):
        sym = M.ring.to_sympy
        return self._print_seq([sym(x) for [x] in M._module.gens], '<', '>')

    def _print_QuotientRing(self, R):
        return self._print(R.ring) / self._print(R.base_ideal)

    def _print_QuotientRingElement(self, R):
        return self._print(R.ring.to_sympy(R)) + self._print(R.ring.base_ideal)

    def _print_QuotientModuleElement(self, m):
        return self._print(m.data) + self._print(m.module.killed_module)

    def _print_QuotientModule(self, M):
        return self._print(M.base) / self._print(M.killed_module)

    def _print_MatrixHomomorphism(self, h):
        matrix = self._print(h._sympy_matrix())
        matrix.baseline = matrix.height() // 2
        pform = prettyForm(*matrix.right(' : ', self._print(h.domain),
            ' %s> ' % hobj('-', 2), self._print(h.codomain)))
        return pform

    def _print_Manifold(self, manifold):
        return self._print(manifold.name)

    def _print_Patch(self, patch):
        return self._print(patch.name)

    def _print_CoordSystem(self, coords):
        return self._print(coords.name)

    def _print_BaseScalarField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return self._print(pretty_symbol(string))

    def _print_BaseVectorField(self, field):
        s = U('PARTIAL DIFFERENTIAL') + '_' + field._coord_sys.symbols[field._index].name
        return self._print(pretty_symbol(s))

    def _print_Differential(self, diff):
        if self._use_unicode:
            d = pretty_atom('Differential')
        else:
            d = 'd'
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys.symbols[field._index].name
            return self._print(d + ' ' + pretty_symbol(string))
        else:
            pform = self._print(field)
            pform = prettyForm(*pform.parens())
            return prettyForm(*pform.left(d))

    def _print_Tr(self, p):
        #TODO: Handle indices
        pform = self._print(p.args[0])
        pform = prettyForm(*pform.left('%s(' % (p.__class__.__name__)))
        pform = prettyForm(*pform.right(')'))
        return pform

    def _print_primenu(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        if self._use_unicode:
            pform = prettyForm(*pform.left(greek_unicode['nu']))
        else:
            pform = prettyForm(*pform.left('nu'))
        return pform

    def _print_primeomega(self, e):
        pform = self._print(e.args[0])
        pform = prettyForm(*pform.parens())
        if self._use_unicode:
            pform = prettyForm(*pform.left(greek_unicode['Omega']))
        else:
            pform = prettyForm(*pform.left('Omega'))
        return pform

    def _print_Quantity(self, e):
        if e.name.name == 'degree':
            if self._use_unicode:
                pform = self._print(pretty_atom('Degree'))
            else:
                pform = self._print(chr(176))
            return pform
        else:
            return self.emptyPrinter(e)

    def _print_AssignmentBase(self, e):

        op = prettyForm(' ' + xsym(e.op) + ' ')

        l = self._print(e.lhs)
        r = self._print(e.rhs)
        pform = prettyForm(*stringPict.next(l, op, r))
        return pform

    def _print_Str(self, s):
        return self._print(s.name)


@print_function(PrettyPrinter)
def pretty(expr, **settings):
    """Returns a string containing the prettified form of expr.

    For information on keyword arguments see pretty_print function.

    """
    pp = PrettyPrinter(settings)

    # XXX: this is an ugly hack, but at least it works
    use_unicode = pp._settings['use_unicode']
    uflag = pretty_use_unicode(use_unicode)

    try:
        return pp.doprint(expr)
    finally:
        pretty_use_unicode(uflag)


def pretty_print(expr, **kwargs):
    """Prints expr in pretty form.

    pprint is just a shortcut for this function.

    Parameters
    ==========

    expr : expression
        The expression to print.

    wrap_line : bool, optional (default=True)
        Line wrapping enabled/disabled.

    num_columns : int or None, optional (default=None)
        Number of columns before line breaking (default to None which reads
        the terminal width), useful when using SymPy without terminal.

    use_unicode : bool or None, optional (default=None)
        Use unicode characters, such as the Greek letter pi instead of
        the string pi.

    full_prec : bool or string, optional (default="auto")
        Use full precision.

    order : bool or string, optional (default=None)
        Set to 'none' for long expressions if slow; default is None.

    use_unicode_sqrt_char : bool, optional (default=True)
        Use compact single-character square root symbol (when unambiguous).

    root_notation : bool, optional (default=True)
        Set to 'False' for printing exponents of the form 1/n in fractional form.
        By default exponent is printed in root form.

    mat_symbol_style : string, optional (default="plain")
        Set to "bold" for printing MatrixSymbols using a bold mathematical symbol face.
        By default the standard face is used.

    imaginary_unit : string, optional (default="i")
        Letter to use for imaginary unit when use_unicode is True.
        Can be "i" (default) or "j".
    """
    print(pretty(expr, **kwargs))

pprint = pretty_print


def pager_print(expr, **settings):
    """Prints expr using the pager, in pretty form.

    This invokes a pager command using pydoc. Lines are not wrapped
    automatically. This routine is meant to be used with a pager that allows
    sideways scrolling, like ``less -S``.

    Parameters are the same as for ``pretty_print``. If you wish to wrap lines,
    pass ``num_columns=None`` to auto-detect the width of the terminal.

    """
    from pydoc import pager
    from locale import getpreferredencoding
    if 'num_columns' not in settings:
        settings['num_columns'] = 500000  # disable line wrap
    pager(pretty(expr, **settings).encode(getpreferredencoding()))