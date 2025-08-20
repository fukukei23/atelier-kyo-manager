
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_results.py ===
# testing/suite/test_results.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

import datetime
import re

from .. import engines
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import DateTime
from ... import func
from ... import Integer
from ... import select
from ... import sql
from ... import String
from ... import testing
from ... import text


class RowFetchTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "plain_pk",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(50)),
        )
        Table(
            "has_dates",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("today", DateTime),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.plain_pk.insert(),
            [
                {"id": 1, "data": "d1"},
                {"id": 2, "data": "d2"},
                {"id": 3, "data": "d3"},
            ],
        )

        connection.execute(
            cls.tables.has_dates.insert(),
            [{"id": 1, "today": datetime.datetime(2006, 5, 12, 12, 0, 0)}],
        )

    def test_via_attr(self, connection):
        row = connection.execute(
            self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)
        ).first()

        eq_(row.id, 1)
        eq_(row.data, "d1")

    def test_via_string(self, connection):
        row = connection.execute(
            self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)
        ).first()

        eq_(row._mapping["id"], 1)
        eq_(row._mapping["data"], "d1")

    def test_via_int(self, connection):
        row = connection.execute(
            self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)
        ).first()

        eq_(row[0], 1)
        eq_(row[1], "d1")

    def test_via_col_object(self, connection):
        row = connection.execute(
            self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)
        ).first()

        eq_(row._mapping[self.tables.plain_pk.c.id], 1)
        eq_(row._mapping[self.tables.plain_pk.c.data], "d1")

    @requirements.duplicate_names_in_cursor_description
    def test_row_with_dupe_names(self, connection):
        result = connection.execute(
            select(
                self.tables.plain_pk.c.data,
                self.tables.plain_pk.c.data.label("data"),
            ).order_by(self.tables.plain_pk.c.id)
        )
        row = result.first()
        eq_(result.keys(), ["data", "data"])
        eq_(row, ("d1", "d1"))

    def test_row_w_scalar_select(self, connection):
        """test that a scalar select as a column is returned as such
        and that type conversion works OK.

        (this is half a SQLAlchemy Core test and half to catch database
        backends that may have unusual behavior with scalar selects.)

        """
        datetable = self.tables.has_dates
        s = select(datetable.alias("x").c.today).scalar_subquery()
        s2 = select(datetable.c.id, s.label("somelabel"))
        row = connection.execute(s2).first()

        eq_(row.somelabel, datetime.datetime(2006, 5, 12, 12, 0, 0))


class PercentSchemaNamesTest(fixtures.TablesTest):
    """tests using percent signs, spaces in table and column names.

    This didn't work for PostgreSQL / MySQL drivers for a long time
    but is now supported.

    """

    __requires__ = ("percent_schema_names",)

    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        cls.tables.percent_table = Table(
            "percent%table",
            metadata,
            Column("percent%", Integer),
            Column("spaces % more spaces", Integer),
        )
        cls.tables.lightweight_percent_table = sql.table(
            "percent%table",
            sql.column("percent%"),
            sql.column("spaces % more spaces"),
        )

    def test_single_roundtrip(self, connection):
        percent_table = self.tables.percent_table
        for params in [
            {"percent%": 5, "spaces % more spaces": 12},
            {"percent%": 7, "spaces % more spaces": 11},
            {"percent%": 9, "spaces % more spaces": 10},
            {"percent%": 11, "spaces % more spaces": 9},
        ]:
            connection.execute(percent_table.insert(), params)
        self._assert_table(connection)

    def test_executemany_roundtrip(self, connection):
        percent_table = self.tables.percent_table
        connection.execute(
            percent_table.insert(), {"percent%": 5, "spaces % more spaces": 12}
        )
        connection.execute(
            percent_table.insert(),
            [
                {"percent%": 7, "spaces % more spaces": 11},
                {"percent%": 9, "spaces % more spaces": 10},
                {"percent%": 11, "spaces % more spaces": 9},
            ],
        )
        self._assert_table(connection)

    @requirements.insert_executemany_returning
    def test_executemany_returning_roundtrip(self, connection):
        percent_table = self.tables.percent_table
        connection.execute(
            percent_table.insert(), {"percent%": 5, "spaces % more spaces": 12}
        )
        result = connection.execute(
            percent_table.insert().returning(
                percent_table.c["percent%"],
                percent_table.c["spaces % more spaces"],
            ),
            [
                {"percent%": 7, "spaces % more spaces": 11},
                {"percent%": 9, "spaces % more spaces": 10},
                {"percent%": 11, "spaces % more spaces": 9},
            ],
        )
        eq_(result.all(), [(7, 11), (9, 10), (11, 9)])
        self._assert_table(connection)

    def _assert_table(self, conn):
        percent_table = self.tables.percent_table
        lightweight_percent_table = self.tables.lightweight_percent_table

        for table in (
            percent_table,
            percent_table.alias(),
            lightweight_percent_table,
            lightweight_percent_table.alias(),
        ):
            eq_(
                list(
                    conn.execute(table.select().order_by(table.c["percent%"]))
                ),
                [(5, 12), (7, 11), (9, 10), (11, 9)],
            )

            eq_(
                list(
                    conn.execute(
                        table.select()
                        .where(table.c["spaces % more spaces"].in_([9, 10]))
                        .order_by(table.c["percent%"])
                    )
                ),
                [(9, 10), (11, 9)],
            )

            row = conn.execute(
                table.select().order_by(table.c["percent%"])
            ).first()
            eq_(row._mapping["percent%"], 5)
            eq_(row._mapping["spaces % more spaces"], 12)

            eq_(row._mapping[table.c["percent%"]], 5)
            eq_(row._mapping[table.c["spaces % more spaces"]], 12)

        conn.execute(
            percent_table.update().values(
                {percent_table.c["spaces % more spaces"]: 15}
            )
        )

        eq_(
            list(
                conn.execute(
                    percent_table.select().order_by(
                        percent_table.c["percent%"]
                    )
                )
            ),
            [(5, 15), (7, 15), (9, 15), (11, 15)],
        )


class ServerSideCursorsTest(
    fixtures.TestBase, testing.AssertsExecutionResults
):
    __requires__ = ("server_side_cursors",)

    __backend__ = True

    def _is_server_side(self, cursor):
        # TODO: this is a huge issue as it prevents these tests from being
        # usable by third party dialects.
        if self.engine.dialect.driver == "psycopg2":
            return bool(cursor.name)
        elif self.engine.dialect.driver == "pymysql":
            sscursor = __import__("pymysql.cursors").cursors.SSCursor
            return isinstance(cursor, sscursor)
        elif self.engine.dialect.driver in ("aiomysql", "asyncmy", "aioodbc"):
            return cursor.server_side
        elif self.engine.dialect.driver == "mysqldb":
            sscursor = __import__("MySQLdb.cursors").cursors.SSCursor
            return isinstance(cursor, sscursor)
        elif self.engine.dialect.driver == "mariadbconnector":
            return not cursor.buffered
        elif self.engine.dialect.driver == "mysqlconnector":
            return "buffered" not in type(cursor).__name__.lower()
        elif self.engine.dialect.driver in ("asyncpg", "aiosqlite"):
            return cursor.server_side
        elif self.engine.dialect.driver == "pg8000":
            return getattr(cursor, "server_side", False)
        elif self.engine.dialect.driver == "psycopg":
            return bool(getattr(cursor, "name", False))
        elif self.engine.dialect.driver == "oracledb":
            return getattr(cursor, "server_side", False)
        else:
            return False

    def _fixture(self, server_side_cursors):
        if server_side_cursors:
            with testing.expect_deprecated(
                "The create_engine.server_side_cursors parameter is "
                "deprecated and will be removed in a future release.  "
                "Please use the Connection.execution_options.stream_results "
                "parameter."
            ):
                self.engine = engines.testing_engine(
                    options={"server_side_cursors": server_side_cursors}
                )
        else:
            self.engine = engines.testing_engine(
                options={"server_side_cursors": server_side_cursors}
            )
        return self.engine

    def stringify(self, str_):
        return re.compile(r"SELECT (\d+)", re.I).sub(
            lambda m: str(select(int(m.group(1))).compile(testing.db)), str_
        )

    @testing.combinations(
        ("global_string", True, lambda stringify: stringify("select 1"), True),
        (
            "global_text",
            True,
            lambda stringify: text(stringify("select 1")),
            True,
        ),
        ("global_expr", True, select(1), True),
        (
            "global_off_explicit",
            False,
            lambda stringify: text(stringify("select 1")),
            False,
        ),
        (
            "stmt_option",
            False,
            select(1).execution_options(stream_results=True),
            True,
        ),
        (
            "stmt_option_disabled",
            True,
            select(1).execution_options(stream_results=False),
            False,
        ),
        ("for_update_expr", True, select(1).with_for_update(), True),
        # TODO: need a real requirement for this, or dont use this test
        (
            "for_update_string",
            True,
            lambda stringify: stringify("SELECT 1 FOR UPDATE"),
            True,
            testing.skip_if(["sqlite", "mssql"]),
        ),
        (
            "text_no_ss",
            False,
            lambda stringify: text(stringify("select 42")),
            False,
        ),
        (
            "text_ss_option",
            False,
            lambda stringify: text(stringify("select 42")).execution_options(
                stream_results=True
            ),
            True,
        ),
        id_="iaaa",
        argnames="engine_ss_arg, statement, cursor_ss_status",
    )
    def test_ss_cursor_status(
        self, engine_ss_arg, statement, cursor_ss_status
    ):
        engine = self._fixture(engine_ss_arg)
        with engine.begin() as conn:
            if callable(statement):
                statement = testing.resolve_lambda(
                    statement, stringify=self.stringify
                )

            if isinstance(statement, str):
                result = conn.exec_driver_sql(statement)
            else:
                result = conn.execute(statement)
            eq_(self._is_server_side(result.cursor), cursor_ss_status)
            result.close()

    def test_conn_option(self):
        engine = self._fixture(False)

        with engine.connect() as conn:
            # should be enabled for this one
            result = conn.execution_options(
                stream_results=True
            ).exec_driver_sql(self.stringify("select 1"))
            assert self._is_server_side(result.cursor)

            # the connection has autobegun, which means at the end of the
            # block, we will roll back, which on MySQL at least will fail
            # with "Commands out of sync" if the result set
            # is not closed, so we close it first.
            #
            # fun fact!  why did we not have this result.close() in this test
            # before 2.0? don't we roll back in the connection pool
            # unconditionally? yes!  and in fact if you run this test in 1.4
            # with stdout shown, there is in fact "Exception during reset or
            # similar" with "Commands out sync" emitted a warning!  2.0's
            # architecture finds and fixes what was previously an expensive
            # silent error condition.
            result.close()

    def test_stmt_enabled_conn_option_disabled(self):
        engine = self._fixture(False)

        s = select(1).execution_options(stream_results=True)

        with engine.connect() as conn:
            # not this one
            result = conn.execution_options(stream_results=False).execute(s)
            assert not self._is_server_side(result.cursor)

    def test_aliases_and_ss(self):
        engine = self._fixture(False)
        s1 = (
            select(sql.literal_column("1").label("x"))
            .execution_options(stream_results=True)
            .subquery()
        )

        # options don't propagate out when subquery is used as a FROM clause
        with engine.begin() as conn:
            result = conn.execute(s1.select())
            assert not self._is_server_side(result.cursor)
            result.close()

        s2 = select(1).select_from(s1)
        with engine.begin() as conn:
            result = conn.execute(s2)
            assert not self._is_server_side(result.cursor)
            result.close()

    def test_roundtrip_fetchall(self, metadata):
        md = self.metadata

        engine = self._fixture(True)
        test_table = Table(
            "test_table",
            md,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("data", String(50)),
        )

        with engine.begin() as connection:
            test_table.create(connection, checkfirst=True)
            connection.execute(test_table.insert(), dict(data="data1"))
            connection.execute(test_table.insert(), dict(data="data2"))
            eq_(
                connection.execute(
                    test_table.select().order_by(test_table.c.id)
                ).fetchall(),
                [(1, "data1"), (2, "data2")],
            )
            connection.execute(
                test_table.update()
                .where(test_table.c.id == 2)
                .values(data=test_table.c.data + " updated")
            )
            eq_(
                connection.execute(
                    test_table.select().order_by(test_table.c.id)
                ).fetchall(),
                [(1, "data1"), (2, "data2 updated")],
            )
            connection.execute(test_table.delete())
            eq_(
                connection.scalar(
                    select(func.count("*")).select_from(test_table)
                ),
                0,
            )

    def test_roundtrip_fetchmany(self, metadata):
        md = self.metadata

        engine = self._fixture(True)
        test_table = Table(
            "test_table",
            md,
            Column(
                "id", Integer, primary_key=True, test_needs_autoincrement=True
            ),
            Column("data", String(50)),
        )

        with engine.begin() as connection:
            test_table.create(connection, checkfirst=True)
            connection.execute(
                test_table.insert(),
                [dict(data="data%d" % i) for i in range(1, 20)],
            )

            result = connection.execute(
                test_table.select().order_by(test_table.c.id)
            )

            eq_(
                result.fetchmany(5),
                [(i, "data%d" % i) for i in range(1, 6)],
            )
            eq_(
                result.fetchmany(10),
                [(i, "data%d" % i) for i in range(6, 16)],
            )
            eq_(result.fetchall(), [(i, "data%d" % i) for i in range(16, 20)])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\testing\runtests.py ===
"""
This is our testing framework.

Goals:

* it should be compatible with py.test and operate very similarly
  (or identically)
* does not require any external dependencies
* preferably all the functionality should be in this file only
* no magic, just import the test file and execute the test functions, that's it
* portable

"""

import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from pathlib import Path

from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES

IS_WINDOWS = (os.name == 'nt')
ON_CI = os.getenv('CI', None)

# empirically generated list of the proportion of time spent running
# an even split of tests.  This should periodically be regenerated.
# A list of [.6, .1, .3] would mean that if the tests are evenly split
# into '1/3', '2/3', '3/3', the first split would take 60% of the time,
# the second 10% and the third 30%.  These lists are normalized to sum
# to 1, so [60, 10, 30] has the same behavior as [6, 1, 3] or [.6, .1, .3].
#
# This list can be generated with the code:
#     from time import time
#     import sympy
#     import os
#     os.environ["CI"] = 'true' # Mock CI to get more correct densities
#     delays, num_splits = [], 30
#     for i in range(1, num_splits + 1):
#         tic = time()
#         sympy.test(split='{}/{}'.format(i, num_splits), time_balance=False) # Add slow=True for slow tests
#         delays.append(time() - tic)
#     tot = sum(delays)
#     print([round(x / tot, 4) for x in delays])
SPLIT_DENSITY = [
    0.0059, 0.0027, 0.0068, 0.0011, 0.0006,
    0.0058, 0.0047, 0.0046, 0.004, 0.0257,
    0.0017, 0.0026, 0.004, 0.0032, 0.0016,
    0.0015, 0.0004, 0.0011, 0.0016, 0.0014,
    0.0077, 0.0137, 0.0217, 0.0074, 0.0043,
    0.0067, 0.0236, 0.0004, 0.1189, 0.0142,
    0.0234, 0.0003, 0.0003, 0.0047, 0.0006,
    0.0013, 0.0004, 0.0008, 0.0007, 0.0006,
    0.0139, 0.0013, 0.0007, 0.0051, 0.002,
    0.0004, 0.0005, 0.0213, 0.0048, 0.0016,
    0.0012, 0.0014, 0.0024, 0.0015, 0.0004,
    0.0005, 0.0007, 0.011, 0.0062, 0.0015,
    0.0021, 0.0049, 0.0006, 0.0006, 0.0011,
    0.0006, 0.0019, 0.003, 0.0044, 0.0054,
    0.0057, 0.0049, 0.0016, 0.0006, 0.0009,
    0.0006, 0.0012, 0.0006, 0.0149, 0.0532,
    0.0076, 0.0041, 0.0024, 0.0135, 0.0081,
    0.2209, 0.0459, 0.0438, 0.0488, 0.0137,
    0.002, 0.0003, 0.0008, 0.0039, 0.0024,
    0.0005, 0.0004, 0.003, 0.056, 0.0026]
SPLIT_DENSITY_SLOW = [0.0086, 0.0004, 0.0568, 0.0003, 0.0032, 0.0005, 0.0004, 0.0013, 0.0016, 0.0648, 0.0198, 0.1285, 0.098, 0.0005, 0.0064, 0.0003, 0.0004, 0.0026, 0.0007, 0.0051, 0.0089, 0.0024, 0.0033, 0.0057, 0.0005, 0.0003, 0.001, 0.0045, 0.0091, 0.0006, 0.0005, 0.0321, 0.0059, 0.1105, 0.216, 0.1489, 0.0004, 0.0003, 0.0006, 0.0483]

class Skipped(Exception):
    pass

class TimeOutError(Exception):
    pass

class DependencyError(Exception):
    pass


def _indent(s, indent=4):
    """
    Add the given number of space characters to the beginning of
    every non-blank line in ``s``, and return the result.
    If the string ``s`` is Unicode, it is encoded using the stdout
    encoding and the ``backslashreplace`` error handler.
    """
    # This regexp matches the start of non-blank lines:
    return re.sub('(?m)^(?!$)', indent*' ', s)


pdoctest._indent = _indent  # type: ignore

# override reporter to maintain windows and python3


def _report_failure(self, out, test, example, got):
    """
    Report that the given example failed.
    """
    s = self._checker.output_difference(example, got, self.optionflags)
    s = s.encode('raw_unicode_escape').decode('utf8', 'ignore')
    out(self._failure_header(test, example) + s)


if IS_WINDOWS:
    DocTestRunner.report_failure = _report_failure  # type: ignore


def convert_to_native_paths(lst):
    """
    Converts a list of '/' separated paths into a list of
    native (os.sep separated) paths and converts to lowercase
    if the system is case insensitive.
    """
    newlst = []
    for rv in lst:
        rv = os.path.join(*rv.split("/"))
        # on windows the slash after the colon is dropped
        if sys.platform == "win32":
            pos = rv.find(':')
            if pos != -1:
                if rv[pos + 1] != '\\':
                    rv = rv[:pos + 1] + '\\' + rv[pos + 1:]
        newlst.append(os.path.normcase(rv))
    return newlst


def get_sympy_dir():
    """
    Returns the root SymPy directory and set the global value
    indicating whether the system is case sensitive or not.
    """
    this_file = os.path.abspath(__file__)
    sympy_dir = os.path.join(os.path.dirname(this_file), "..", "..")
    sympy_dir = os.path.normpath(sympy_dir)
    return os.path.normcase(sympy_dir)


def setup_pprint(disable_line_wrap=True):
    from sympy.interactive.printing import init_printing
    from sympy.printing.pretty.pretty import pprint_use_unicode
    import sympy.interactive.printing as interactive_printing
    from sympy.printing.pretty import stringpict

    # Prevent init_printing() in doctests from affecting other doctests
    interactive_printing.NO_GLOBAL = True

    # force pprint to be in ascii mode in doctests
    use_unicode_prev = pprint_use_unicode(False)

    # disable line wrapping for pprint() outputs
    wrap_line_prev = stringpict._GLOBAL_WRAP_LINE
    if disable_line_wrap:
        stringpict._GLOBAL_WRAP_LINE = False

    # hook our nice, hash-stable strprinter
    init_printing(pretty_print=False)

    return use_unicode_prev, wrap_line_prev


@contextmanager
def raise_on_deprecated():
    """Context manager to make DeprecationWarning raise an error

    This is to catch SymPyDeprecationWarning from library code while running
    tests and doctests. It is important to use this context manager around
    each individual test/doctest in case some tests modify the warning
    filters.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('error', '.*', DeprecationWarning, module='sympy.*')
        yield


def run_in_subprocess_with_hash_randomization(
        function, function_args=(),
        function_kwargs=None, command=sys.executable,
        module='sympy.testing.runtests', force=False):
    """
    Run a function in a Python subprocess with hash randomization enabled.

    If hash randomization is not supported by the version of Python given, it
    returns False.  Otherwise, it returns the exit value of the command.  The
    function is passed to sys.exit(), so the return value of the function will
    be the return value.

    The environment variable PYTHONHASHSEED is used to seed Python's hash
    randomization.  If it is set, this function will return False, because
    starting a new subprocess is unnecessary in that case.  If it is not set,
    one is set at random, and the tests are run.  Note that if this
    environment variable is set when Python starts, hash randomization is
    automatically enabled.  To force a subprocess to be created even if
    PYTHONHASHSEED is set, pass ``force=True``.  This flag will not force a
    subprocess in Python versions that do not support hash randomization (see
    below), because those versions of Python do not support the ``-R`` flag.

    ``function`` should be a string name of a function that is importable from
    the module ``module``, like "_test".  The default for ``module`` is
    "sympy.testing.runtests".  ``function_args`` and ``function_kwargs``
    should be a repr-able tuple and dict, respectively.  The default Python
    command is sys.executable, which is the currently running Python command.

    This function is necessary because the seed for hash randomization must be
    set by the environment variable before Python starts.  Hence, in order to
    use a predetermined seed for tests, we must start Python in a separate
    subprocess.

    Hash randomization was added in the minor Python versions 2.6.8, 2.7.3,
    3.1.5, and 3.2.3, and is enabled by default in all Python versions after
    and including 3.3.0.

    Examples
    ========

    >>> from sympy.testing.runtests import (
    ... run_in_subprocess_with_hash_randomization)
    >>> # run the core tests in verbose mode
    >>> run_in_subprocess_with_hash_randomization("_test",
    ... function_args=("core",),
    ... function_kwargs={'verbose': True}) # doctest: +SKIP
    # Will return 0 if sys.executable supports hash randomization and tests
    # pass, 1 if they fail, and False if it does not support hash
    # randomization.

    """
    cwd = get_sympy_dir()
    # Note, we must return False everywhere, not None, as subprocess.call will
    # sometimes return None.

    # First check if the Python version supports hash randomization
    # If it does not have this support, it won't recognize the -R flag
    p = subprocess.Popen([command, "-RV"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, cwd=cwd)
    p.communicate()
    if p.returncode != 0:
        return False

    hash_seed = os.getenv("PYTHONHASHSEED")
    if not hash_seed:
        os.environ["PYTHONHASHSEED"] = str(random.randrange(2**32))
    else:
        if not force:
            return False

    function_kwargs = function_kwargs or {}

    # Now run the command
    commandstring = ("import sys; from %s import %s;sys.exit(%s(*%s, **%s))" %
                     (module, function, function, repr(function_args),
                      repr(function_kwargs)))

    try:
        p = subprocess.Popen([command, "-R", "-c", commandstring], cwd=cwd)
        p.communicate()
    except KeyboardInterrupt:
        p.wait()
    finally:
        # Put the environment variable back, so that it reads correctly for
        # the current Python process.
        if hash_seed is None:
            del os.environ["PYTHONHASHSEED"]
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed
        return p.returncode


def run_all_tests(test_args=(), test_kwargs=None,
                  doctest_args=(), doctest_kwargs=None,
                  examples_args=(), examples_kwargs=None):
    """
    Run all tests.

    Right now, this runs the regular tests (bin/test), the doctests
    (bin/doctest), and the examples (examples/all.py).

    This is what ``setup.py test`` uses.

    You can pass arguments and keyword arguments to the test functions that
    support them (for now, test,  doctest, and the examples). See the
    docstrings of those functions for a description of the available options.

    For example, to run the solvers tests with colors turned off:

    >>> from sympy.testing.runtests import run_all_tests
    >>> run_all_tests(test_args=("solvers",),
    ... test_kwargs={"colors:False"}) # doctest: +SKIP

    """
    tests_successful = True

    test_kwargs = test_kwargs or {}
    doctest_kwargs = doctest_kwargs or {}
    examples_kwargs = examples_kwargs or {'quiet': True}

    try:
        # Regular tests
        if not test(*test_args, **test_kwargs):
            # some regular test fails, so set the tests_successful
            # flag to false and continue running the doctests
            tests_successful = False

        # Doctests
        print()
        if not doctest(*doctest_args, **doctest_kwargs):
            tests_successful = False

        # Examples
        print()
        sys.path.append("examples")   # examples/all.py
        from all import run_examples  # type: ignore
        if not run_examples(*examples_args, **examples_kwargs):
            tests_successful = False

        if tests_successful:
            return
        else:
            # Return nonzero exit code
            sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("DO *NOT* COMMIT!")
        sys.exit(1)


def test(*paths, subprocess=True, rerun=0, **kwargs):
    """
    Run tests in the specified test_*.py files.

    Tests in a particular test_*.py file are run if any of the given strings
    in ``paths`` matches a part of the test file's path. If ``paths=[]``,
    tests in all test_*.py files are run.

    Notes:

    - If sort=False, tests are run in random order (not default).
    - Paths can be entered in native system format or in unix,
      forward-slash format.
    - Files that are on the blacklist can be tested by providing
      their path; they are only excluded if no paths are given.

    **Explanation of test results**

    ======  ===============================================================
    Output  Meaning
    ======  ===============================================================
    .       passed
    F       failed
    X       XPassed (expected to fail but passed)
    f       XFAILed (expected to fail and indeed failed)
    s       skipped
    w       slow
    T       timeout (e.g., when ``--timeout`` is used)
    K       KeyboardInterrupt (when running the slow tests with ``--slow``,
            you can interrupt one of them without killing the test runner)
    ======  ===============================================================


    Colors have no additional meaning and are used just to facilitate
    interpreting the output.

    Examples
    ========

    >>> import sympy

    Run all tests:

    >>> sympy.test()    # doctest: +SKIP

    Run one file:

    >>> sympy.test("sympy/core/tests/test_basic.py")    # doctest: +SKIP
    >>> sympy.test("_basic")    # doctest: +SKIP

    Run all tests in sympy/functions/ and some particular file:

    >>> sympy.test("sympy/core/tests/test_basic.py",
    ...        "sympy/functions")    # doctest: +SKIP

    Run all tests in sympy/core and sympy/utilities:

    >>> sympy.test("/core", "/util")    # doctest: +SKIP

    Run specific test from a file:

    >>> sympy.test("sympy/core/tests/test_basic.py",
    ...        kw="test_equality")    # doctest: +SKIP

    Run specific test from any file:

    >>> sympy.test(kw="subs")    # doctest: +SKIP

    Run the tests with verbose mode on:

    >>> sympy.test(verbose=True)    # doctest: +SKIP

    Do not sort the test output:

    >>> sympy.test(sort=False)    # doctest: +SKIP

    Turn on post-mortem pdb:

    >>> sympy.test(pdb=True)    # doctest: +SKIP

    Turn off colors:

    >>> sympy.test(colors=False)    # doctest: +SKIP

    Force colors, even when the output is not to a terminal (this is useful,
    e.g., if you are piping to ``less -r`` and you still want colors)

    >>> sympy.test(force_colors=False)    # doctest: +SKIP

    The traceback verboseness can be set to "short" or "no" (default is
    "short")

    >>> sympy.test(tb='no')    # doctest: +SKIP

    The ``split`` option can be passed to split the test run into parts. The
    split currently only splits the test files, though this may change in the
    future. ``split`` should be a string of the form 'a/b', which will run
    part ``a`` of ``b``. For instance, to run the first half of the test suite:

    >>> sympy.test(split='1/2')  # doctest: +SKIP

    The ``time_balance`` option can be passed in conjunction with ``split``.
    If ``time_balance=True`` (the default for ``sympy.test``), SymPy will attempt
    to split the tests such that each split takes equal time.  This heuristic
    for balancing is based on pre-recorded test data.

    >>> sympy.test(split='1/2', time_balance=True)  # doctest: +SKIP

    You can disable running the tests in a separate subprocess using
    ``subprocess=False``.  This is done to support seeding hash randomization,
    which is enabled by default in the Python versions where it is supported.
    If subprocess=False, hash randomization is enabled/disabled according to
    whether it has been enabled or not in the calling Python process.
    However, even if it is enabled, the seed cannot be printed unless it is
    called from a new Python process.

    Hash randomization was added in the minor Python versions 2.6.8, 2.7.3,
    3.1.5, and 3.2.3, and is enabled by default in all Python versions after
    and including 3.3.0.

    If hash randomization is not supported ``subprocess=False`` is used
    automatically.

    >>> sympy.test(subprocess=False)     # doctest: +SKIP

    To set the hash randomization seed, set the environment variable
    ``PYTHONHASHSEED`` before running the tests.  This can be done from within
    Python using

    >>> import os
    >>> os.environ['PYTHONHASHSEED'] = '42' # doctest: +SKIP

    Or from the command line using

    $ PYTHONHASHSEED=42 ./bin/test

    If the seed is not set, a random seed will be chosen.

    Note that to reproduce the same hash values, you must use both the same seed
    as well as the same architecture (32-bit vs. 64-bit).

    """
    # count up from 0, do not print 0
    print_counter = lambda i : (print("rerun %d" % (rerun-i))
                                if rerun-i else None)

    if subprocess:
        # loop backwards so last i is 0
        for i in range(rerun, -1, -1):
            print_counter(i)
            ret = run_in_subprocess_with_hash_randomization("_test",
                        function_args=paths, function_kwargs=kwargs)
            if ret is False:
                break
            val = not bool(ret)
            # exit on the first failure or if done
            if not val or i == 0:
                return val

    # rerun even if hash randomization is not supported
    for i in range(rerun, -1, -1):
        print_counter(i)
        val = not bool(_test(*paths, **kwargs))
        if not val or i == 0:
            return val


def _test(*paths,
        verbose=False, tb="short", kw=None, pdb=False, colors=True,
        force_colors=False, sort=True, seed=None, timeout=False,
        fail_on_timeout=False, slow=False, enhance_asserts=False, split=None,
        time_balance=True, blacklist=(),
        fast_threshold=None, slow_threshold=None):
    """
    Internal function that actually runs the tests.

    All keyword arguments from ``test()`` are passed to this function except for
    ``subprocess``.

    Returns 0 if tests passed and 1 if they failed.  See the docstring of
    ``test()`` for more information.
    """
    kw = kw or ()
    # ensure that kw is a tuple
    if isinstance(kw, str):
        kw = (kw,)
    post_mortem = pdb
    if seed is None:
        seed = random.randrange(100000000)
    if ON_CI and timeout is False:
        timeout = 595
        fail_on_timeout = True
    if ON_CI:
        blacklist = list(blacklist) + ['sympy/plotting/pygletplot/tests']
    blacklist = convert_to_native_paths(blacklist)
    r = PyTestReporter(verbose=verbose, tb=tb, colors=colors,
        force_colors=force_colors, split=split)
    # This won't strictly run the test for the corresponding file, but it is
    # good enough for copying and pasting the failing test.
    _paths = []
    for path in paths:
        if '::' in path:
            path, _kw = path.split('::', 1)
            kw += (_kw,)
        _paths.append(path)
    paths = _paths

    t = SymPyTests(r, kw, post_mortem, seed,
                   fast_threshold=fast_threshold,
                   slow_threshold=slow_threshold)

    test_files = t.get_test_files('sympy')

    not_blacklisted = [f for f in test_files
                       if not any(b in f for b in blacklist)]

    if len(paths) == 0:
        matched = not_blacklisted
    else:
        paths = convert_to_native_paths(paths)
        matched = []
        for f in not_blacklisted:
            basename = os.path.basename(f)
            for p in paths:
                if p in f or fnmatch(basename, p):
                    matched.append(f)
                    break

    density = None
    if time_balance:
        if slow:
            density = SPLIT_DENSITY_SLOW
        else:
            density = SPLIT_DENSITY

    if split:
        matched = split_list(matched, split, density=density)

    t._testfiles.extend(matched)

    return int(not t.test(sort=sort, timeout=timeout, slow=slow,
        enhance_asserts=enhance_asserts, fail_on_timeout=fail_on_timeout))


def doctest(*paths, subprocess=True, rerun=0, **kwargs):
    r"""
    Runs doctests in all \*.py files in the SymPy directory which match
    any of the given strings in ``paths`` or all tests if paths=[].

    Notes:

    - Paths can be entered in native system format or in unix,
      forward-slash format.
    - Files that are on the blacklist can be tested by providing
      their path; they are only excluded if no paths are given.

    Examples
    ========

    >>> import sympy

    Run all tests:

    >>> sympy.doctest() # doctest: +SKIP

    Run one file:

    >>> sympy.doctest("sympy/core/basic.py") # doctest: +SKIP
    >>> sympy.doctest("polynomial.rst") # doctest: +SKIP

    Run all tests in sympy/functions/ and some particular file:

    >>> sympy.doctest("/functions", "basic.py") # doctest: +SKIP

    Run any file having polynomial in its name, doc/src/modules/polynomial.rst,
    sympy/functions/special/polynomials.py, and sympy/polys/polynomial.py:

    >>> sympy.doctest("polynomial") # doctest: +SKIP

    The ``split`` option can be passed to split the test run into parts. The
    split currently only splits the test files, though this may change in the
    future. ``split`` should be a string of the form 'a/b', which will run
    part ``a`` of ``b``. Note that the regular doctests and the Sphinx
    doctests are split independently. For instance, to run the first half of
    the test suite:

    >>> sympy.doctest(split='1/2')  # doctest: +SKIP

    The ``subprocess`` and ``verbose`` options are the same as with the function
    ``test()`` (see the docstring of that function for more information) except
    that ``verbose`` may also be set equal to ``2`` in order to print
    individual doctest lines, as they are being tested.
    """
    # count up from 0, do not print 0
    print_counter = lambda i : (print("rerun %d" % (rerun-i))
                                if rerun-i else None)

    if subprocess:
        # loop backwards so last i is 0
        for i in range(rerun, -1, -1):
            print_counter(i)
            ret = run_in_subprocess_with_hash_randomization("_doctest",
                        function_args=paths, function_kwargs=kwargs)
            if ret is False:
                break
            val = not bool(ret)
            # exit on the first failure or if done
            if not val or i == 0:
                return val

    # rerun even if hash randomization is not supported
    for i in range(rerun, -1, -1):
        print_counter(i)
        val = not bool(_doctest(*paths, **kwargs))
        if not val or i == 0:
            return val


def _get_doctest_blacklist():
    '''Get the default blacklist for the doctests'''
    blacklist = []

    blacklist.extend([
        "doc/src/modules/plotting.rst",  # generates live plots
        "doc/src/modules/physics/mechanics/autolev_parser.rst",
        "sympy/codegen/array_utils.py", # raises deprecation warning
        "sympy/core/compatibility.py", # backwards compatibility shim, importing it triggers a deprecation warning
        "sympy/core/trace.py", # backwards compatibility shim, importing it triggers a deprecation warning
        "sympy/galgebra.py", # no longer part of SymPy
        "sympy/parsing/autolev/_antlr/autolevlexer.py", # generated code
        "sympy/parsing/autolev/_antlr/autolevlistener.py", # generated code
        "sympy/parsing/autolev/_antlr/autolevparser.py", # generated code
        "sympy/parsing/latex/_antlr/latexlexer.py", # generated code
        "sympy/parsing/latex/_antlr/latexparser.py", # generated code
        "sympy/plotting/pygletplot/__init__.py", # crashes on some systems
        "sympy/plotting/pygletplot/plot.py", # crashes on some systems
        "sympy/printing/ccode.py", # backwards compatibility shim, importing it breaks the codegen doctests
        "sympy/printing/cxxcode.py", # backwards compatibility shim, importing it breaks the codegen doctests
        "sympy/printing/fcode.py", # backwards compatibility shim, importing it breaks the codegen doctests
        "sympy/testing/randtest.py", # backwards compatibility shim, importing it triggers a deprecation warning
        "sympy/this.py", # prints text
    ])
    # autolev parser tests
    num = 12
    for i in range (1, num+1):
        blacklist.append("sympy/parsing/autolev/test-examples/ruletest" + str(i) + ".py")
    blacklist.extend(["sympy/parsing/autolev/test-examples/pydy-example-repo/mass_spring_damper.py",
                      "sympy/parsing/autolev/test-examples/pydy-example-repo/chaos_pendulum.py",
                      "sympy/parsing/autolev/test-examples/pydy-example-repo/double_pendulum.py",
                      "sympy/parsing/autolev/test-examples/pydy-example-repo/non_min_pendulum.py"])

    if import_module('numpy') is None:
        blacklist.extend([
            "sympy/plotting/experimental_lambdify.py",
            "sympy/plotting/plot_implicit.py",
            "examples/advanced/autowrap_integrators.py",
            "examples/advanced/autowrap_ufuncify.py",
            "examples/intermediate/sample.py",
            "examples/intermediate/mplot2d.py",
            "examples/intermediate/mplot3d.py",
            "doc/src/modules/numeric-computation.rst",
            "doc/src/explanation/best-practices.md",
            "doc/src/tutorials/physics/biomechanics/biomechanical-model-example.rst",
            "doc/src/tutorials/physics/biomechanics/biomechanics.rst",
        ])
    else:
        if import_module('matplotlib') is None:
            blacklist.extend([
                "examples/intermediate/mplot2d.py",
                "examples/intermediate/mplot3d.py"
            ])
        else:
            # Use a non-windowed backend, so that the tests work on CI
            import matplotlib
            matplotlib.use('Agg')

    if ON_CI or import_module('pyglet') is None:
        blacklist.extend(["sympy/plotting/pygletplot"])

    if import_module('aesara') is None:
        blacklist.extend([
            "sympy/printing/aesaracode.py",
            "doc/src/modules/numeric-computation.rst",
        ])

    if import_module('cupy') is None:
        blacklist.extend([
            "doc/src/modules/numeric-computation.rst",
        ])

    if import_module('jax') is None:
        blacklist.extend([
            "doc/src/modules/numeric-computation.rst",
        ])

    if import_module('antlr4') is None:
        blacklist.extend([
            "sympy/parsing/autolev/__init__.py",
            "sympy/parsing/latex/_parse_latex_antlr.py",
        ])

    if import_module('lfortran') is None:
        #throws ImportError when lfortran not installed
        blacklist.extend([
            "sympy/parsing/sym_expr.py",
        ])

    if import_module("scipy") is None:
        # throws ModuleNotFoundError when scipy not installed
        blacklist.extend([
            "doc/src/guides/solving/solve-numerically.md",
            "doc/src/guides/solving/solve-ode.md",
        ])

    if import_module("numpy") is None:
        # throws ModuleNotFoundError when numpy not installed
        blacklist.extend([
                "doc/src/guides/solving/solve-ode.md",
                "doc/src/guides/solving/solve-numerically.md",
        ])

    # disabled because of doctest failures in asmeurer's bot
    blacklist.extend([
        "sympy/utilities/autowrap.py",
        "examples/advanced/autowrap_integrators.py",
        "examples/advanced/autowrap_ufuncify.py"
        ])

    blacklist.extend([
        "sympy/conftest.py", # Depends on pytest
    ])

    # These are deprecated stubs to be removed:
    blacklist.extend([
        "sympy/utilities/tmpfiles.py",
        "sympy/utilities/pytest.py",
        "sympy/utilities/runtests.py",
        "sympy/utilities/quality_unicode.py",
        "sympy/utilities/randtest.py",
    ])

    blacklist = convert_to_native_paths(blacklist)
    return blacklist


def _doctest(*paths, **kwargs):
    """
    Internal function that actually runs the doctests.

    All keyword arguments from ``doctest()`` are passed to this function
    except for ``subprocess``.

    Returns 0 if tests passed and 1 if they failed.  See the docstrings of
    ``doctest()`` and ``test()`` for more information.
    """
    from sympy.printing.pretty.pretty import pprint_use_unicode
    from sympy.printing.pretty import stringpict

    normal = kwargs.get("normal", False)
    verbose = kwargs.get("verbose", False)
    colors = kwargs.get("colors", True)
    force_colors = kwargs.get("force_colors", False)
    blacklist = kwargs.get("blacklist", [])
    split  = kwargs.get('split', None)

    blacklist.extend(_get_doctest_blacklist())

    # Use a non-windowed backend, so that the tests work on CI
    if import_module('matplotlib') is not None:
        import matplotlib
        matplotlib.use('Agg')

    # Disable warnings for external modules
    import sympy.external
    sympy.external.importtools.WARN_OLD_VERSION = False
    sympy.external.importtools.WARN_NOT_INSTALLED = False

    # Disable showing up of plots
    from sympy.plotting.plot import unset_show
    unset_show()

    r = PyTestReporter(verbose, split=split, colors=colors,\
                       force_colors=force_colors)
    t = SymPyDocTests(r, normal)

    test_files = t.get_test_files('sympy')
    test_files.extend(t.get_test_files('examples', init_only=False))

    not_blacklisted = [f for f in test_files
                       if not any(b in f for b in blacklist)]
    if len(paths) == 0:
        matched = not_blacklisted
    else:
        # take only what was requested...but not blacklisted items
        # and allow for partial match anywhere or fnmatch of name
        paths = convert_to_native_paths(paths)
        matched = []
        for f in not_blacklisted:
            basename = os.path.basename(f)
            for p in paths:
                if p in f or fnmatch(basename, p):
                    matched.append(f)
                    break

    matched.sort()

    if split:
        matched = split_list(matched, split)

    t._testfiles.extend(matched)

    # run the tests and record the result for this *py portion of the tests
    if t._testfiles:
        failed = not t.test()
    else:
        failed = False

    # N.B.
    # --------------------------------------------------------------------
    # Here we test *.rst and *.md files at or below doc/src. Code from these
    # must be self supporting in terms of imports since there is no importing
    # of necessary modules by doctest.testfile. If you try to pass *.py files
    # through this they might fail because they will lack the needed imports
    # and smarter parsing that can be done with source code.
    #
    test_files_rst = t.get_test_files('doc/src', '*.rst', init_only=False)
    test_files_md = t.get_test_files('doc/src', '*.md', init_only=False)
    test_files = test_files_rst + test_files_md
    test_files.sort()

    not_blacklisted = [f for f in test_files
                       if not any(b in f for b in blacklist)]

    if len(paths) == 0:
        matched = not_blacklisted
    else:
        # Take only what was requested as long as it's not on the blacklist.
        # Paths were already made native in *py tests so don't repeat here.
        # There's no chance of having a *py file slip through since we
        # only have *rst files in test_files.
        matched = []
        for f in not_blacklisted:
            basename = os.path.basename(f)
            for p in paths:
                if p in f or fnmatch(basename, p):
                    matched.append(f)
                    break

    if split:
        matched = split_list(matched, split)

    first_report = True
    for rst_file in matched:
        if not os.path.isfile(rst_file):
            continue
        old_displayhook = sys.displayhook
        try:
            use_unicode_prev, wrap_line_prev = setup_pprint()
            out = sympytestfile(
                rst_file, module_relative=False, encoding='utf-8',
                optionflags=pdoctest.ELLIPSIS | pdoctest.NORMALIZE_WHITESPACE |
                pdoctest.IGNORE_EXCEPTION_DETAIL)
        finally:
            # make sure we return to the original displayhook in case some
            # doctest has changed that
            sys.displayhook = old_displayhook
            # The NO_GLOBAL flag overrides the no_global flag to init_printing
            # if True
            import sympy.interactive.printing as interactive_printing
            interactive_printing.NO_GLOBAL = False
            pprint_use_unicode(use_unicode_prev)
            stringpict._GLOBAL_WRAP_LINE = wrap_line_prev

        rstfailed, tested = out
        if tested:
            failed = rstfailed or failed
            if first_report:
                first_report = False
                msg = 'rst/md doctests start'
                if not t._testfiles:
                    r.start(msg=msg)
                else:
                    r.write_center(msg)
                    print()
            # use as the id, everything past the first 'sympy'
            file_id = rst_file[rst_file.find('sympy') + len('sympy') + 1:]
            print(file_id, end=" ")
                # get at least the name out so it is know who is being tested
            wid = r.terminal_width - len(file_id) - 1  # update width
            test_file = '[%s]' % (tested)
            report = '[%s]' % (rstfailed or 'OK')
            print(''.join(
                [test_file, ' '*(wid - len(test_file) - len(report)), report])
            )

    # the doctests for *py will have printed this message already if there was
    # a failure, so now only print it if there was intervening reporting by
    # testing the *rst as evidenced by first_report no longer being True.
    if not first_report and failed:
        print()
        print("DO *NOT* COMMIT!")

    return int(failed)

sp = re.compile(r'([0-9]+)/([1-9][0-9]*)')

def split_list(l, split, density=None):
    """
    Splits a list into part a of b

    split should be a string of the form 'a/b'. For instance, '1/3' would give
    the split one of three.

    If the length of the list is not divisible by the number of splits, the
    last split will have more items.

    `density` may be specified as a list.  If specified,
    tests will be balanced so that each split has as equal-as-possible
    amount of mass according to `density`.

    >>> from sympy.testing.runtests import split_list
    >>> a = list(range(10))
    >>> split_list(a, '1/3')
    [0, 1, 2]
    >>> split_list(a, '2/3')
    [3, 4, 5]
    >>> split_list(a, '3/3')
    [6, 7, 8, 9]
    """
    m = sp.match(split)
    if not m:
        raise ValueError("split must be a string of the form a/b where a and b are ints")
    i, t = map(int, m.groups())

    if not density:
        return l[(i - 1)*len(l)//t : i*len(l)//t]

    # normalize density
    tot = sum(density)
    density = [x / tot for x in density]

    def density_inv(x):
        """Interpolate the inverse to the cumulative
        distribution function given by density"""
        if x <= 0:
            return 0
        if x >= sum(density):
            return 1

        # find the first time the cumulative sum surpasses x
        # and linearly interpolate
        cumm = 0
        for i, d in enumerate(density):
            cumm += d
            if cumm >= x:
                break
        frac = (d - (cumm - x)) / d
        return (i + frac) / len(density)

    lower_frac = density_inv((i - 1) / t)
    higher_frac = density_inv(i / t)
    return l[int(lower_frac*len(l)) : int(higher_frac*len(l))]

from collections import namedtuple
SymPyTestResults = namedtuple('SymPyTestResults', 'failed attempted')

def sympytestfile(filename, module_relative=True, name=None, package=None,
             globs=None, verbose=None, report=True, optionflags=0,
             extraglobs=None, raise_on_error=False,
             parser=pdoctest.DocTestParser(), encoding=None):

    """
    Test examples in the given file.  Return (#failures, #tests).

    Optional keyword arg ``module_relative`` specifies how filenames
    should be interpreted:

    - If ``module_relative`` is True (the default), then ``filename``
      specifies a module-relative path.  By default, this path is
      relative to the calling module's directory; but if the
      ``package`` argument is specified, then it is relative to that
      package.  To ensure os-independence, ``filename`` should use
      "/" characters to separate path segments, and should not
      be an absolute path (i.e., it may not begin with "/").

    - If ``module_relative`` is False, then ``filename`` specifies an
      os-specific path.  The path may be absolute or relative (to
      the current working directory).

    Optional keyword arg ``name`` gives the name of the test; by default
    use the file's basename.

    Optional keyword argument ``package`` is a Python package or the
    name of a Python package whose directory should be used as the
    base directory for a module relative filename.  If no package is
    specified, then the calling module's directory is used as the base
    directory for module relative filenames.  It is an error to
    specify ``package`` if ``module_relative`` is False.

    Optional keyword arg ``globs`` gives a dict to be used as the globals
    when executing examples; by default, use {}.  A copy of this dict
    is actually used for each docstring, so that each docstring's
    examples start with a clean slate.

    Optional keyword arg ``extraglobs`` gives a dictionary that should be
    merged into the globals that are used to execute examples.  By
    default, no extra globals are used.

    Optional keyword arg ``verbose`` prints lots of stuff if true, prints
    only failures if false; by default, it's true iff "-v" is in sys.argv.

    Optional keyword arg ``report`` prints a summary at the end when true,
    else prints nothing at the end.  In verbose mode, the summary is
    detailed, else very brief (in fact, empty if all tests passed).

    Optional keyword arg ``optionflags`` or's together module constants,
    and defaults to 0.  Possible values (see the docs for details):

    - DONT_ACCEPT_TRUE_FOR_1
    - DONT_ACCEPT_BLANKLINE
    - NORMALIZE_WHITESPACE
    - ELLIPSIS
    - SKIP
    - IGNORE_EXCEPTION_DETAIL
    - REPORT_UDIFF
    - REPORT_CDIFF
    - REPORT_NDIFF
    - REPORT_ONLY_FIRST_FAILURE

    Optional keyword arg ``raise_on_error`` raises an exception on the
    first unexpected exception or failure. This allows failures to be
    post-mortem debugged.

    Optional keyword arg ``parser`` specifies a DocTestParser (or
    subclass) that should be used to extract tests from the files.

    Optional keyword arg ``encoding`` specifies an encoding that should
    be used to convert the file to unicode.

    Advanced tomfoolery:  testmod runs methods of a local instance of
    class doctest.Tester, then merges the results into (or creates)
    global Tester instance doctest.master.  Methods of doctest.master
    can be called directly too, if you want to do something unusual.
    Passing report=0 to testmod is especially useful then, to delay
    displaying a summary.  Invoke doctest.master.summarize(verbose)
    when you're done fiddling.
    """
    if package and not module_relative:
        raise ValueError("Package may only be specified for module-"
                         "relative paths.")

    # Relativize the path
    text, filename = pdoctest._load_testfile(
        filename, package, module_relative, encoding)

    # If no name was given, then use the file's name.
    if name is None:
        name = os.path.basename(filename)

    # Assemble the globals.
    if globs is None:
        globs = {}
    else:
        globs = globs.copy()
    if extraglobs is not None:
        globs.update(extraglobs)
    if '__name__' not in globs:
        globs['__name__'] = '__main__'

    if raise_on_error:
        runner = pdoctest.DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        runner = SymPyDocTestRunner(verbose=verbose, optionflags=optionflags)
        runner._checker = SymPyOutputChecker()

    # Read the file, convert it to a test, and run it.
    test = parser.get_doctest(text, globs, name, filename, 0)
    runner.run(test)

    if report:
        runner.summarize()

    if pdoctest.master is None:
        pdoctest.master = runner
    else:
        pdoctest.master.merge(runner)

    return SymPyTestResults(runner.failures, runner.tries)


class SymPyTests:

    def __init__(self, reporter, kw="", post_mortem=False,
                 seed=None, fast_threshold=None, slow_threshold=None):
        self._post_mortem = post_mortem
        self._kw = kw
        self._count = 0
        self._root_dir = get_sympy_dir()
        self._reporter = reporter
        self._reporter.root_dir(self._root_dir)
        self._testfiles = []
        self._seed = seed if seed is not None else random.random()

        # Defaults in seconds, from human / UX design limits
        # http://www.nngroup.com/articles/response-times-3-important-limits/
        #
        # These defaults are *NOT* set in stone as we are measuring different
        # things, so others feel free to come up with a better yardstick :)
        if fast_threshold:
            self._fast_threshold = float(fast_threshold)
        else:
            self._fast_threshold = 8
        if slow_threshold:
            self._slow_threshold = float(slow_threshold)
        else:
            self._slow_threshold = 10

    def test(self, sort=False, timeout=False, slow=False,
            enhance_asserts=False, fail_on_timeout=False):
        """
        Runs the tests returning True if all tests pass, otherwise False.

        If sort=False run tests in random order.
        """
        if sort:
            self._testfiles.sort()
        elif slow:
            pass
        else:
            random.seed(self._seed)
            random.shuffle(self._testfiles)
        self._reporter.start(self._seed)
        for f in self._testfiles:
            try:
                self.test_file(f, sort, timeout, slow,
                    enhance_asserts, fail_on_timeout)
            except KeyboardInterrupt:
                print(" interrupted by user")
                self._reporter.finish()
                raise
        return self._reporter.finish()

    def _enhance_asserts(self, source):
        from ast import (NodeTransformer, Compare, Name, Store, Load, Tuple,
            Assign, BinOp, Str, Mod, Assert, parse, fix_missing_locations)

        ops = {"Eq": '==', "NotEq": '!=', "Lt": '<', "LtE": '<=',
                "Gt": '>', "GtE": '>=', "Is": 'is', "IsNot": 'is not',
                "In": 'in', "NotIn": 'not in'}

        class Transform(NodeTransformer):
            def visit_Assert(self, stmt):
                if isinstance(stmt.test, Compare):
                    compare = stmt.test
                    values = [compare.left] + compare.comparators
                    names = [ "_%s" % i for i, _ in enumerate(values) ]
                    names_store = [ Name(n, Store()) for n in names ]
                    names_load = [ Name(n, Load()) for n in names ]
                    target = Tuple(names_store, Store())
                    value = Tuple(values, Load())
                    assign = Assign([target], value)
                    new_compare = Compare(names_load[0], compare.ops, names_load[1:])
                    msg_format = "\n%s " + "\n%s ".join([ ops[op.__class__.__name__] for op in compare.ops ]) + "\n%s"
                    msg = BinOp(Str(msg_format), Mod(), Tuple(names_load, Load()))
                    test = Assert(new_compare, msg, lineno=stmt.lineno, col_offset=stmt.col_offset)
                    return [assign, test]
                else:
                    return stmt

        tree = parse(source)
        new_tree = Transform().visit(tree)
        return fix_missing_locations(new_tree)

    def test_file(self, filename, sort=True, timeout=False, slow=False,
            enhance_asserts=False, fail_on_timeout=False):
        reporter = self._reporter
        funcs = []
        try:
            gl = {'__file__': filename}
            try:
                open_file = lambda: open(filename, encoding="utf8")

                with open_file() as f:
                    source = f.read()
                    if self._kw:
                        for l in source.splitlines():
                            if l.lstrip().startswith('def '):
                                if any(l.lower().find(k.lower()) != -1 for k in self._kw):
                                    break
                        else:
                            return

                if enhance_asserts:
                    try:
                        source = self._enhance_asserts(source)
                    except ImportError:
                        pass

                code = compile(source, filename, "exec", flags=0, dont_inherit=True)
                exec(code, gl)
            except (SystemExit, KeyboardInterrupt):
                raise
            except ImportError:
                reporter.import_error(filename, sys.exc_info())
                return
            except Exception:
                reporter.test_exception(sys.exc_info())

            clear_cache()
            self._count += 1
            random.seed(self._seed)
            disabled = gl.get("disabled", False)
            if not disabled:
                # we need to filter only those functions that begin with 'test_'
                # We have to be careful about decorated functions. As long as
                # the decorator uses functools.wraps, we can detect it.
                funcs = []
                for f in gl:
                    if (f.startswith("test_") and (inspect.isfunction(gl[f])
                        or inspect.ismethod(gl[f]))):
                        func = gl[f]
                        # Handle multiple decorators
                        while hasattr(func, '__wrapped__'):
                            func = func.__wrapped__

                        if inspect.getsourcefile(func) == filename:
                            funcs.append(gl[f])
                if slow:
                    funcs = [f for f in funcs if getattr(f, '_slow', False)]
                # Sorting of XFAILed functions isn't fixed yet :-(
                funcs.sort(key=lambda x: inspect.getsourcelines(x)[1])
                i = 0
                while i < len(funcs):
                    if inspect.isgeneratorfunction(funcs[i]):
                    # some tests can be generators, that return the actual
                    # test functions. We unpack it below:
                        f = funcs.pop(i)
                        for fg in f():
                            func = fg[0]
                            args = fg[1:]
                            fgw = lambda: func(*args)
                            funcs.insert(i, fgw)
                            i += 1
                    else:
                        i += 1
                # drop functions that are not selected with the keyword expression:
                funcs = [x for x in funcs if self.matches(x)]

            if not funcs:
                return
        except Exception:
            reporter.entering_filename(filename, len(funcs))
            raise

        reporter.entering_filename(filename, len(funcs))
        if not sort:
            random.shuffle(funcs)

        for f in funcs:
            start = time.time()
            reporter.entering_test(f)
            try:
                if getattr(f, '_slow', False) and not slow:
                    raise Skipped("Slow")
                with raise_on_deprecated():
                    if timeout:
                        self._timeout(f, timeout, fail_on_timeout)
                    else:
                        random.seed(self._seed)
                        f()
            except KeyboardInterrupt:
                if getattr(f, '_slow', False):
                    reporter.test_skip("KeyboardInterrupt")
                else:
                    raise
            except Exception:
                if timeout:
                    signal.alarm(0)  # Disable the alarm. It could not be handled before.
                t, v, tr = sys.exc_info()
                if t is AssertionError:
                    reporter.test_fail((t, v, tr))
                    if self._post_mortem:
                        pdb.post_mortem(tr)
                elif t.__name__ == "Skipped":
                    reporter.test_skip(v)
                elif t.__name__ == "XFail":
                    reporter.test_xfail()
                elif t.__name__ == "XPass":
                    reporter.test_xpass(v)
                else:
                    reporter.test_exception((t, v, tr))
                    if self._post_mortem:
                        pdb.post_mortem(tr)
            else:
                reporter.test_pass()
            taken = time.time() - start
            if taken > self._slow_threshold:
                filename = os.path.relpath(filename, reporter._root_dir)
                reporter.slow_test_functions.append(
                    (filename + "::" + f.__name__, taken))
            if getattr(f, '_slow', False) and slow:
                if taken < self._fast_threshold:
                    filename = os.path.relpath(filename, reporter._root_dir)
                    reporter.fast_test_functions.append(
                        (filename + "::" + f.__name__, taken))
        reporter.leaving_filename()

    def _timeout(self, function, timeout, fail_on_timeout):
        def callback(x, y):
            signal.alarm(0)
            if fail_on_timeout:
                raise TimeOutError("Timed out after %d seconds" % timeout)
            else:
                raise Skipped("Timeout")
        signal.signal(signal.SIGALRM, callback)
        signal.alarm(timeout)  # Set an alarm with a given timeout
        function()
        signal.alarm(0)  # Disable the alarm

    def matches(self, x):
        """
        Does the keyword expression self._kw match "x"? Returns True/False.

        Always returns True if self._kw is "".
        """
        if not self._kw:
            return True
        for kw in self._kw:
            if x.__name__.lower().find(kw.lower()) != -1:
                return True
        return False

    def get_test_files(self, dir, pat='test_*.py'):
        """
        Returns the list of test_*.py (default) files at or below directory
        ``dir`` relative to the SymPy home directory.
        """
        dir = os.path.join(self._root_dir, convert_to_native_paths([dir])[0])

        g = []
        for path, folders, files in os.walk(dir):
            g.extend([os.path.join(path, f) for f in files if fnmatch(f, pat)])

        return sorted([os.path.normcase(gi) for gi in g])


class SymPyDocTests:

    def __init__(self, reporter, normal):
        self._count = 0
        self._root_dir = get_sympy_dir()
        self._reporter = reporter
        self._reporter.root_dir(self._root_dir)
        self._normal = normal

        self._testfiles = []

    def test(self):
        """
        Runs the tests and returns True if all tests pass, otherwise False.
        """
        self._reporter.start()
        for f in self._testfiles:
            try:
                self.test_file(f)
            except KeyboardInterrupt:
                print(" interrupted by user")
                self._reporter.finish()
                raise
        return self._reporter.finish()

    def test_file(self, filename):
        clear_cache()

        from io import StringIO
        import sympy.interactive.printing as interactive_printing
        from sympy.printing.pretty.pretty import pprint_use_unicode
        from sympy.printing.pretty import stringpict

        rel_name = filename[len(self._root_dir) + 1:]
        dirname, file = os.path.split(filename)
        module = rel_name.replace(os.sep, '.')[:-3]

        if rel_name.startswith("examples"):
            # Examples files do not have __init__.py files,
            # So we have to temporarily extend sys.path to import them
            sys.path.insert(0, dirname)
            module = file[:-3]  # remove ".py"
        try:
            module = pdoctest._normalize_module(module)
            tests = SymPyDocTestFinder().find(module)
        except (SystemExit, KeyboardInterrupt):
            raise
        except ImportError:
            self._reporter.import_error(filename, sys.exc_info())
            return
        finally:
            if rel_name.startswith("examples"):
                del sys.path[0]

        tests = [test for test in tests if len(test.examples) > 0]
        # By default tests are sorted by alphabetical order by function name.
        # We sort by line number so one can edit the file sequentially from
        # bottom to top. However, if there are decorated functions, their line
        # numbers will be too large and for now one must just search for these
        # by text and function name.
        tests.sort(key=lambda x: -x.lineno)

        if not tests:
            return
        self._reporter.entering_filename(filename, len(tests))
        for test in tests:
            assert len(test.examples) != 0

            if self._reporter._verbose:
                self._reporter.write("\n{} ".format(test.name))

            # check if there are external dependencies which need to be met
            if '_doctest_depends_on' in test.globs:
                try:
                    self._check_dependencies(**test.globs['_doctest_depends_on'])
                except DependencyError as e:
                    self._reporter.test_skip(v=str(e))
                    continue

            runner = SymPyDocTestRunner(verbose=self._reporter._verbose==2,
                    optionflags=pdoctest.ELLIPSIS |
                    pdoctest.NORMALIZE_WHITESPACE |
                    pdoctest.IGNORE_EXCEPTION_DETAIL)
            runner._checker = SymPyOutputChecker()
            old = sys.stdout
            new = old if self._reporter._verbose==2 else StringIO()
            sys.stdout = new
            # If the testing is normal, the doctests get importing magic to
            # provide the global namespace. If not normal (the default) then
            # then must run on their own; all imports must be explicit within
            # a function's docstring. Once imported that import will be
            # available to the rest of the tests in a given function's
            # docstring (unless clear_globs=True below).
            if not self._normal:
                test.globs = {}
                # if this is uncommented then all the test would get is what
                # comes by default with a "from sympy import *"
                #exec('from sympy import *') in test.globs
            old_displayhook = sys.displayhook
            use_unicode_prev, wrap_line_prev = setup_pprint()

            try:
                f, t = runner.run(test,
                                  out=new.write, clear_globs=False)
            except KeyboardInterrupt:
                raise
            finally:
                sys.stdout = old
            if f > 0:
                self._reporter.doctest_fail(test.name, new.getvalue())
            else:
                self._reporter.test_pass()
                sys.displayhook = old_displayhook
                interactive_printing.NO_GLOBAL = False
                pprint_use_unicode(use_unicode_prev)
                stringpict._GLOBAL_WRAP_LINE = wrap_line_prev

        self._reporter.leaving_filename()

    def get_test_files(self, dir, pat='*.py', init_only=True):
        r"""
        Returns the list of \*.py files (default) from which docstrings
        will be tested which are at or below directory ``dir``. By default,
        only those that have an __init__.py in their parent directory
        and do not start with ``test_`` will be included.
        """
        def importable(x):
            """
            Checks if given pathname x is an importable module by checking for
            __init__.py file.

            Returns True/False.

            Currently we only test if the __init__.py file exists in the
            directory with the file "x" (in theory we should also test all the
            parent dirs).
            """
            init_py = os.path.join(os.path.dirname(x), "__init__.py")
            return os.path.exists(init_py)

        dir = os.path.join(self._root_dir, convert_to_native_paths([dir])[0])

        g = []
        for path, folders, files in os.walk(dir):
            g.extend([os.path.join(path, f) for f in files
                      if not f.startswith('test_') and fnmatch(f, pat)])
        if init_only:
            # skip files that are not importable (i.e. missing __init__.py)
            g = [x for x in g if importable(x)]

        return [os.path.normcase(gi) for gi in g]

    def _check_dependencies(self,
                            executables=(),
                            modules=(),
                            disable_viewers=(),
                            python_version=(3, 5),
                            ground_types=None):
        """
        Checks if the dependencies for the test are installed.

        Raises ``DependencyError`` it at least one dependency is not installed.
        """

        for executable in executables:
            if not shutil.which(executable):
                raise DependencyError("Could not find %s" % executable)

        for module in modules:
            if module == 'matplotlib':
                matplotlib = import_module(
                    'matplotlib',
                    import_kwargs={'fromlist':
                                      ['pyplot', 'cm', 'collections']},
                    min_module_version='1.0.0', catch=(RuntimeError,))
                if matplotlib is None:
                    raise DependencyError("Could not import matplotlib")
            else:
                if not import_module(module):
                    raise DependencyError("Could not import %s" % module)

        if disable_viewers:
            tempdir = tempfile.mkdtemp()
            os.environ['PATH'] = '%s:%s' % (tempdir, os.environ['PATH'])

            vw = ('#!/usr/bin/env python3\n'
                  'import sys\n'
                  'if len(sys.argv) <= 1:\n'
                  '    exit("wrong number of args")\n')

            for viewer in disable_viewers:
                Path(os.path.join(tempdir, viewer)).write_text(vw)

                # make the file executable
                os.chmod(os.path.join(tempdir, viewer),
                         stat.S_IREAD | stat.S_IWRITE | stat.S_IXUSR)

        if python_version:
            if sys.version_info < python_version:
                raise DependencyError("Requires Python >= " + '.'.join(map(str, python_version)))

        if ground_types is not None:
            if GROUND_TYPES not in ground_types:
                raise DependencyError("Requires ground_types in " + str(ground_types))

        if 'pyglet' in modules:
            # monkey-patch pyglet s.t. it does not open a window during
            # doctesting
            import pyglet
            class DummyWindow:
                def __init__(self, *args, **kwargs):
                    self.has_exit = True
                    self.width = 600
                    self.height = 400

                def set_vsync(self, x):
                    pass

                def switch_to(self):
                    pass

                def push_handlers(self, x):
                    pass

                def close(self):
                    pass

            pyglet.window.Window = DummyWindow


class SymPyDocTestFinder(DocTestFinder):
    """
    A class used to extract the DocTests that are relevant to a given
    object, from its docstring and the docstrings of its contained
    objects.  Doctests can currently be extracted from the following
    object types: modules, functions, classes, methods, staticmethods,
    classmethods, and properties.

    Modified from doctest's version to look harder for code that
    appears comes from a different module. For example, the @vectorize
    decorator makes it look like functions come from multidimensional.py
    even though their code exists elsewhere.
    """

    def _find(self, tests, obj, name, module, source_lines, globs, seen):
        """
        Find tests for the given object and any contained objects, and
        add them to ``tests``.
        """
        if self._verbose:
            print('Finding tests in %s' % name)

        # If we've already processed this object, then ignore it.
        if id(obj) in seen:
            return
        seen[id(obj)] = 1

        # Make sure we don't run doctests for classes outside of sympy, such
        # as in numpy or scipy.
        if inspect.isclass(obj):
            if obj.__module__.split('.')[0] != 'sympy':
                return

        # Find a test for this object, and add it to the list of tests.
        test = self._get_test(obj, name, module, globs, source_lines)
        if test is not None:
            tests.append(test)

        if not self._recurse:
            return

        # Look for tests in a module's contained objects.
        if inspect.ismodule(obj):
            for rawname, val in obj.__dict__.items():
                # Recurse to functions & classes.
                if inspect.isfunction(val) or inspect.isclass(val):
                    # Make sure we don't run doctests functions or classes
                    # from different modules
                    if val.__module__ != module.__name__:
                        continue

                    assert self._from_module(module, val), \
                        "%s is not in module %s (rawname %s)" % (val, module, rawname)

                    try:
                        valname = '%s.%s' % (name, rawname)
                        self._find(tests, val, valname, module,
                                   source_lines, globs, seen)
                    except KeyboardInterrupt:
                        raise

            # Look for tests in a module's __test__ dictionary.
            for valname, val in getattr(obj, '__test__', {}).items():
                if not isinstance(valname, str):
                    raise ValueError("SymPyDocTestFinder.find: __test__ keys "
                                     "must be strings: %r" %
                                     (type(valname),))
                if not (inspect.isfunction(val) or inspect.isclass(val) or
                        inspect.ismethod(val) or inspect.ismodule(val) or
                        isinstance(val, str)):
                    raise ValueError("SymPyDocTestFinder.find: __test__ values "
                                     "must be strings, functions, methods, "
                                     "classes, or modules: %r" %
                                     (type(val),))
                valname = '%s.__test__.%s' % (name, valname)
                self._find(tests, val, valname, module, source_lines,
                           globs, seen)


        # Look for tests in a class's contained objects.
        if inspect.isclass(obj):
            for valname, val in obj.__dict__.items():
                # Special handling for staticmethod/classmethod.
                if isinstance(val, staticmethod):
                    val = getattr(obj, valname)
                if isinstance(val, classmethod):
                    val = getattr(obj, valname).__func__


                # Recurse to methods, properties, and nested classes.
                if ((inspect.isfunction(unwrap(val)) or
                        inspect.isclass(val) or
                        isinstance(val, property)) and
                    self._from_module(module, val)):
                    # Make sure we don't run doctests functions or classes
                    # from different modules
                    if isinstance(val, property):
                        if hasattr(val.fget, '__module__'):
                            if val.fget.__module__ != module.__name__:
                                continue
                    else:
                        if val.__module__ != module.__name__:
                            continue

                    assert self._from_module(module, val), \
                        "%s is not in module %s (valname %s)" % (
                            val, module, valname)

                    valname = '%s.%s' % (name, valname)
                    self._find(tests, val, valname, module, source_lines,
                               globs, seen)

    def _get_test(self, obj, name, module, globs, source_lines):
        """
        Return a DocTest for the given object, if it defines a docstring;
        otherwise, return None.
        """

        lineno = None

        # Extract the object's docstring.  If it does not have one,
        # then return None (no test for this object).
        if isinstance(obj, str):
            # obj is a string in the case for objects in the polys package.
            # Note that source_lines is a binary string (compiled polys
            # modules), which can't be handled by _find_lineno so determine
            # the line number here.

            docstring = obj

            matches = re.findall(r"line \d+", name)
            assert len(matches) == 1, \
                "string '%s' does not contain lineno " % name

            # NOTE: this is not the exact linenumber but its better than no
            # lineno ;)
            lineno = int(matches[0][5:])

        else:
            docstring = getattr(obj, '__doc__', '')
            if docstring is None:
                docstring = ''
            if not isinstance(docstring, str):
                docstring = str(docstring)

        # Don't bother if the docstring is empty.
        if self._exclude_empty and not docstring:
            return None

        # check that properties have a docstring because _find_lineno
        # assumes it
        if isinstance(obj, property):
            if obj.fget.__doc__ is None:
                return None

        # Find the docstring's location in the file.
        if lineno is None:
            obj = unwrap(obj)
            # handling of properties is not implemented in _find_lineno so do
            # it here
            if hasattr(obj, 'func_closure') and obj.func_closure is not None:
                tobj = obj.func_closure[0].cell_contents
            elif isinstance(obj, property):
                tobj = obj.fget
            else:
                tobj = obj
            lineno = self._find_lineno(tobj, source_lines)

        if lineno is None:
            return None

        # Return a DocTest for this object.
        if module is None:
            filename = None
        else:
            filename = getattr(module, '__file__', module.__name__)
            if filename[-4:] in (".pyc", ".pyo"):
                filename = filename[:-1]

        globs['_doctest_depends_on'] = getattr(obj, '_doctest_depends_on', {})

        return self._parser.get_doctest(docstring, globs, name,
                                        filename, lineno)


class SymPyDocTestRunner(DocTestRunner):
    """
    A class used to run DocTest test cases, and accumulate statistics.
    The ``run`` method is used to process a single DocTest case.  It
    returns a tuple ``(f, t)``, where ``t`` is the number of test cases
    tried, and ``f`` is the number of test cases that failed.

    Modified from the doctest version to not reset the sys.displayhook (see
    issue 5140).

    See the docstring of the original DocTestRunner for more information.
    """

    def run(self, test, compileflags=None, out=None, clear_globs=True):
        """
        Run the examples in ``test``, and display the results using the
        writer function ``out``.

        The examples are run in the namespace ``test.globs``.  If
        ``clear_globs`` is true (the default), then this namespace will
        be cleared after the test runs, to help with garbage
        collection.  If you would like to examine the namespace after
        the test completes, then use ``clear_globs=False``.

        ``compileflags`` gives the set of flags that should be used by
        the Python compiler when running the examples.  If not
        specified, then it will default to the set of future-import
        flags that apply to ``globs``.

        The output of each example is checked using
        ``SymPyDocTestRunner.check_output``, and the results are
        formatted by the ``SymPyDocTestRunner.report_*`` methods.
        """
        self.test = test

        # Remove ``` from the end of example, which may appear in Markdown
        # files
        for example in test.examples:
            example.want = example.want.replace('```\n', '')
            example.exc_msg = example.exc_msg and example.exc_msg.replace('```\n', '')


        if compileflags is None:
            compileflags = pdoctest._extract_future_flags(test.globs)

        save_stdout = sys.stdout
        if out is None:
            out = save_stdout.write
        sys.stdout = self._fakeout

        # Patch pdb.set_trace to restore sys.stdout during interactive
        # debugging (so it's not still redirected to self._fakeout).
        # Note that the interactive output will go to *our*
        # save_stdout, even if that's not the real sys.stdout; this
        # allows us to write test cases for the set_trace behavior.
        save_set_trace = pdb.set_trace
        self.debugger = pdoctest._OutputRedirectingPdb(save_stdout)
        self.debugger.reset()
        pdb.set_trace = self.debugger.set_trace

        # Patch linecache.getlines, so we can see the example's source
        # when we're inside the debugger.
        self.save_linecache_getlines = pdoctest.linecache.getlines
        linecache.getlines = self.__patched_linecache_getlines

        # Fail for deprecation warnings
        with raise_on_deprecated():
            try:
                return self.__run(test, compileflags, out)
            finally:
                sys.stdout = save_stdout
                pdb.set_trace = save_set_trace
                linecache.getlines = self.save_linecache_getlines
                if clear_globs:
                    test.globs.clear()


# We have to override the name mangled methods.
monkeypatched_methods = [
    'patched_linecache_getlines',
    'run',
    'record_outcome'
]
for method in monkeypatched_methods:
    oldname = '_DocTestRunner__' + method
    newname = '_SymPyDocTestRunner__' + method
    setattr(SymPyDocTestRunner, newname, getattr(DocTestRunner, oldname))


class SymPyOutputChecker(pdoctest.OutputChecker):
    """
    Compared to the OutputChecker from the stdlib our OutputChecker class
    supports numerical comparison of floats occurring in the output of the
    doctest examples
    """

    def __init__(self):
        # NOTE OutputChecker is an old-style class with no __init__ method,
        # so we can't call the base class version of __init__ here

        got_floats = r'(\d+\.\d*|\.\d+)'

        # floats in the 'want' string may contain ellipses
        want_floats = got_floats + r'(\.{3})?'

        front_sep = r'\s|\+|\-|\*|,'
        back_sep = front_sep + r'|j|e'

        fbeg = r'^%s(?=%s|$)' % (got_floats, back_sep)
        fmidend = r'(?<=%s)%s(?=%s|$)' % (front_sep, got_floats, back_sep)
        self.num_got_rgx = re.compile(r'(%s|%s)' %(fbeg, fmidend))

        fbeg = r'^%s(?=%s|$)' % (want_floats, back_sep)
        fmidend = r'(?<=%s)%s(?=%s|$)' % (front_sep, want_floats, back_sep)
        self.num_want_rgx = re.compile(r'(%s|%s)' %(fbeg, fmidend))

    def check_output(self, want, got, optionflags):
        """
        Return True iff the actual output from an example (`got`)
        matches the expected output (`want`).  These strings are
        always considered to match if they are identical; but
        depending on what option flags the test runner is using,
        several non-exact match types are also possible.  See the
        documentation for `TestRunner` for more information about
        option flags.
        """
        # Handle the common case first, for efficiency:
        # if they're string-identical, always return true.
        if got == want:
            return True

        # TODO parse integers as well ?
        # Parse floats and compare them. If some of the parsed floats contain
        # ellipses, skip the comparison.
        matches = self.num_got_rgx.finditer(got)
        numbers_got = [match.group(1) for match in matches] # list of strs
        matches = self.num_want_rgx.finditer(want)
        numbers_want = [match.group(1) for match in matches] # list of strs
        if len(numbers_got) != len(numbers_want):
            return False

        if len(numbers_got) > 0:
            nw_  = []
            for ng, nw in zip(numbers_got, numbers_want):
                if '...' in nw:
                    nw_.append(ng)
                    continue
                else:
                    nw_.append(nw)

                if abs(float(ng)-float(nw)) > 1e-5:
                    return False

            got = self.num_got_rgx.sub(r'%s', got)
            got = got % tuple(nw_)

        # <BLANKLINE> can be used as a special sequence to signify a
        # blank line, unless the DONT_ACCEPT_BLANKLINE flag is used.
        if not (optionflags & pdoctest.DONT_ACCEPT_BLANKLINE):
            # Replace <BLANKLINE> in want with a blank line.
            want = re.sub(r'(?m)^%s\s*?$' % re.escape(pdoctest.BLANKLINE_MARKER),
                          '', want)
            # If a line in got contains only spaces, then remove the
            # spaces.
            got = re.sub(r'(?m)^\s*?$', '', got)
            if got == want:
                return True

        # This flag causes doctest to ignore any differences in the
        # contents of whitespace strings.  Note that this can be used
        # in conjunction with the ELLIPSIS flag.
        if optionflags & pdoctest.NORMALIZE_WHITESPACE:
            got = ' '.join(got.split())
            want = ' '.join(want.split())
            if got == want:
                return True

        # The ELLIPSIS flag says to let the sequence "..." in `want`
        # match any substring in `got`.
        if optionflags & pdoctest.ELLIPSIS:
            if pdoctest._ellipsis_match(want, got):
                return True

        # We didn't find any match; return false.
        return False


class Reporter:
    """
    Parent class for all reporters.
    """
    pass


class PyTestReporter(Reporter):
    """
    Py.test like reporter. Should produce output identical to py.test.
    """

    def __init__(self, verbose=False, tb="short", colors=True,
                 force_colors=False, split=None):
        self._verbose = verbose
        self._tb_style = tb
        self._colors = colors
        self._force_colors = force_colors
        self._xfailed = 0
        self._xpassed = []
        self._failed = []
        self._failed_doctest = []
        self._passed = 0
        self._skipped = 0
        self._exceptions = []
        self._terminal_width = None
        self._default_width = 80
        self._split = split
        self._active_file = ''
        self._active_f = None

        # TODO: Should these be protected?
        self.slow_test_functions = []
        self.fast_test_functions = []

        # this tracks the x-position of the cursor (useful for positioning
        # things on the screen), without the need for any readline library:
        self._write_pos = 0
        self._line_wrap = False

    def root_dir(self, dir):
        self._root_dir = dir

    @property
    def terminal_width(self):
        if self._terminal_width is not None:
            return self._terminal_width

        def findout_terminal_width():
            if sys.platform == "win32":
                # Windows support is based on:
                #
                #  http://code.activestate.com/recipes/
                #  440694-determine-size-of-console-window-on-windows/

                from ctypes import windll, create_string_buffer

                h = windll.kernel32.GetStdHandle(-12)
                csbi = create_string_buffer(22)
                res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)

                if res:
                    import struct
                    (_, _, _, _, _, left, _, right, _, _, _) = \
                        struct.unpack("hhhhHhhhhhh", csbi.raw)
                    return right - left
                else:
                    return self._default_width

            if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
                return self._default_width  # leave PIPEs alone

            try:
                process = subprocess.Popen(['stty', '-a'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                stdout = stdout.decode("utf-8")
            except OSError:
                pass
            else:
                # We support the following output formats from stty:
                #
                # 1) Linux   -> columns 80
                # 2) OS X    -> 80 columns
                # 3) Solaris -> columns = 80

                re_linux = r"columns\s+(?P<columns>\d+);"
                re_osx = r"(?P<columns>\d+)\s*columns;"
                re_solaris = r"columns\s+=\s+(?P<columns>\d+);"

                for regex in (re_linux, re_osx, re_solaris):
                    match = re.search(regex, stdout)

                    if match is not None:
                        columns = match.group('columns')

                        try:
                            width = int(columns)
                        except ValueError:
                            pass
                        if width != 0:
                            return width

            return self._default_width

        width = findout_terminal_width()
        self._terminal_width = width

        return width

    def write(self, text, color="", align="left", width=None,
              force_colors=False):
        """
        Prints a text on the screen.

        It uses sys.stdout.write(), so no readline library is necessary.

        Parameters
        ==========

        color : choose from the colors below, "" means default color
        align : "left"/"right", "left" is a normal print, "right" is aligned on
                the right-hand side of the screen, filled with spaces if
                necessary
        width : the screen width

        """
        color_templates = (
            ("Black", "0;30"),
            ("Red", "0;31"),
            ("Green", "0;32"),
            ("Brown", "0;33"),
            ("Blue", "0;34"),
            ("Purple", "0;35"),
            ("Cyan", "0;36"),
            ("LightGray", "0;37"),
            ("DarkGray", "1;30"),
            ("LightRed", "1;31"),
            ("LightGreen", "1;32"),
            ("Yellow", "1;33"),
            ("LightBlue", "1;34"),
            ("LightPurple", "1;35"),
            ("LightCyan", "1;36"),
            ("White", "1;37"),
        )

        colors = {}

        for name, value in color_templates:
            colors[name] = value
        c_normal = '\033[0m'
        c_color = '\033[%sm'

        if width is None:
            width = self.terminal_width

        if align == "right":
            if self._write_pos + len(text) > width:
                # we don't fit on the current line, create a new line
                self.write("\n")
            self.write(" "*(width - self._write_pos - len(text)))

        if not self._force_colors and hasattr(sys.stdout, 'isatty') and not \
                sys.stdout.isatty():
            # the stdout is not a terminal, this for example happens if the
            # output is piped to less, e.g. "bin/test | less". In this case,
            # the terminal control sequences would be printed verbatim, so
            # don't use any colors.
            color = ""
        elif sys.platform == "win32":
            # Windows consoles don't support ANSI escape sequences
            color = ""
        elif not self._colors:
            color = ""

        if self._line_wrap:
            if text[0] != "\n":
                sys.stdout.write("\n")

        # Avoid UnicodeEncodeError when printing out test failures
        if IS_WINDOWS:
            text = text.encode('raw_unicode_escape').decode('utf8', 'ignore')
        elif not sys.stdout.encoding.lower().startswith('utf'):
            text = text.encode(sys.stdout.encoding, 'backslashreplace'
                              ).decode(sys.stdout.encoding)

        if color == "":
            sys.stdout.write(text)
        else:
            sys.stdout.write("%s%s%s" %
                (c_color % colors[color], text, c_normal))
        sys.stdout.flush()
        l = text.rfind("\n")
        if l == -1:
            self._write_pos += len(text)
        else:
            self._write_pos = len(text) - l - 1
        self._line_wrap = self._write_pos >= width
        self._write_pos %= width

    def write_center(self, text, delim="="):
        width = self.terminal_width
        if text != "":
            text = " %s " % text
        idx = (width - len(text)) // 2
        t = delim*idx + text + delim*(width - idx - len(text))
        self.write(t + "\n")

    def write_exception(self, e, val, tb):
        # remove the first item, as that is always runtests.py
        tb = tb.tb_next
        t = traceback.format_exception(e, val, tb)
        self.write("".join(t))

    def start(self, seed=None, msg="test process starts"):
        self.write_center(msg)
        executable = sys.executable
        v = tuple(sys.version_info)
        python_version = "%s.%s.%s-%s-%s" % v
        implementation = platform.python_implementation()
        if implementation == 'PyPy':
            implementation += " %s.%s.%s-%s-%s" % sys.pypy_version_info
        self.write("executable:         %s  (%s) [%s]\n" %
            (executable, python_version, implementation))
        from sympy.utilities.misc import ARCH
        self.write("architecture:       %s\n" % ARCH)
        from sympy.core.cache import USE_CACHE
        self.write("cache:              %s\n" % USE_CACHE)
        version = ''
        if GROUND_TYPES =='gmpy':
            import gmpy2 as gmpy
            version = gmpy.version()
        self.write("ground types:       %s %s\n" % (GROUND_TYPES, version))
        numpy = import_module('numpy')
        self.write("numpy:              %s\n" % (None if not numpy else numpy.__version__))
        if seed is not None:
            self.write("random seed:        %d\n" % seed)
        from sympy.utilities.misc import HASH_RANDOMIZATION
        self.write("hash randomization: ")
        hash_seed = os.getenv("PYTHONHASHSEED") or '0'
        if HASH_RANDOMIZATION and (hash_seed == "random" or int(hash_seed)):
            self.write("on (PYTHONHASHSEED=%s)\n" % hash_seed)
        else:
            self.write("off\n")
        if self._split:
            self.write("split:              %s\n" % self._split)
        self.write('\n')
        self._t_start = clock()

    def finish(self):
        self._t_end = clock()
        self.write("\n")
        global text, linelen
        text = "tests finished: %d passed, " % self._passed
        linelen = len(text)

        def add_text(mytext):
            global text, linelen
            """Break new text if too long."""
            if linelen + len(mytext) > self.terminal_width:
                text += '\n'
                linelen = 0
            text += mytext
            linelen += len(mytext)

        if len(self._failed) > 0:
            add_text("%d failed, " % len(self._failed))
        if len(self._failed_doctest) > 0:
            add_text("%d failed, " % len(self._failed_doctest))
        if self._skipped > 0:
            add_text("%d skipped, " % self._skipped)
        if self._xfailed > 0:
            add_text("%d expected to fail, " % self._xfailed)
        if len(self._xpassed) > 0:
            add_text("%d expected to fail but passed, " % len(self._xpassed))
        if len(self._exceptions) > 0:
            add_text("%d exceptions, " % len(self._exceptions))
        add_text("in %.2f seconds" % (self._t_end - self._t_start))

        if self.slow_test_functions:
            self.write_center('slowest tests', '_')
            sorted_slow = sorted(self.slow_test_functions, key=lambda r: r[1])
            for slow_func_name, taken in sorted_slow:
                print('%s - Took %.3f seconds' % (slow_func_name, taken))

        if self.fast_test_functions:
            self.write_center('unexpectedly fast tests', '_')
            sorted_fast = sorted(self.fast_test_functions,
                                 key=lambda r: r[1])
            for fast_func_name, taken in sorted_fast:
                print('%s - Took %.3f seconds' % (fast_func_name, taken))

        if len(self._xpassed) > 0:
            self.write_center("xpassed tests", "_")
            for e in self._xpassed:
                self.write("%s: %s\n" % (e[0], e[1]))
            self.write("\n")

        if self._tb_style != "no" and len(self._exceptions) > 0:
            for e in self._exceptions:
                filename, f, (t, val, tb) = e
                self.write_center("", "_")
                if f is None:
                    s = "%s" % filename
                else:
                    s = "%s:%s" % (filename, f.__name__)
                self.write_center(s, "_")
                self.write_exception(t, val, tb)
            self.write("\n")

        if self._tb_style != "no" and len(self._failed) > 0:
            for e in self._failed:
                filename, f, (t, val, tb) = e
                self.write_center("", "_")
                self.write_center("%s::%s" % (filename, f.__name__), "_")
                self.write_exception(t, val, tb)
            self.write("\n")

        if self._tb_style != "no" and len(self._failed_doctest) > 0:
            for e in self._failed_doctest:
                filename, msg = e
                self.write_center("", "_")
                self.write_center("%s" % filename, "_")
                self.write(msg)
            self.write("\n")

        self.write_center(text)
        ok = len(self._failed) == 0 and len(self._exceptions) == 0 and \
            len(self._failed_doctest) == 0
        if not ok:
            self.write("DO *NOT* COMMIT!\n")
        return ok

    def entering_filename(self, filename, n):
        rel_name = filename[len(self._root_dir) + 1:]
        self._active_file = rel_name
        self._active_file_error = False
        self.write(rel_name)
        self.write("[%d] " % n)

    def leaving_filename(self):
        self.write(" ")
        if self._active_file_error:
            self.write("[FAIL]", "Red", align="right")
        else:
            self.write("[OK]", "Green", align="right")
        self.write("\n")
        if self._verbose:
            self.write("\n")

    def entering_test(self, f):
        self._active_f = f
        if self._verbose:
            self.write("\n" + f.__name__ + " ")

    def test_xfail(self):
        self._xfailed += 1
        self.write("f", "Green")

    def test_xpass(self, v):
        message = str(v)
        self._xpassed.append((self._active_file, message))
        self.write("X", "Green")

    def test_fail(self, exc_info):
        self._failed.append((self._active_file, self._active_f, exc_info))
        self.write("F", "Red")
        self._active_file_error = True

    def doctest_fail(self, name, error_msg):
        # the first line contains "******", remove it:
        error_msg = "\n".join(error_msg.split("\n")[1:])
        self._failed_doctest.append((name, error_msg))
        self.write("F", "Red")
        self._active_file_error = True

    def test_pass(self, char="."):
        self._passed += 1
        if self._verbose:
            self.write("ok", "Green")
        else:
            self.write(char, "Green")

    def test_skip(self, v=None):
        char = "s"
        self._skipped += 1
        if v is not None:
            message = str(v)
            if message == "KeyboardInterrupt":
                char = "K"
            elif message == "Timeout":
                char = "T"
            elif message == "Slow":
                char = "w"
        if self._verbose:
            if v is not None:
                self.write(message + ' ', "Blue")
            else:
                self.write(" - ", "Blue")
        self.write(char, "Blue")

    def test_exception(self, exc_info):
        self._exceptions.append((self._active_file, self._active_f, exc_info))
        if exc_info[0] is TimeOutError:
            self.write("T", "Red")
        else:
            self.write("E", "Red")
        self._active_file_error = True

    def import_error(self, filename, exc_info):
        self._exceptions.append((filename, None, exc_info))
        rel_name = filename[len(self._root_dir) + 1:]
        self.write(rel_name)
        self.write("[?]   Failed to import", "Red")
        self.write(" ")
        self.write("[FAIL]", "Red", align="right")
        self.write("\n")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\util.py ===
# orm/util.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

from __future__ import annotations

import enum
import functools
import re
import types
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import attributes  # noqa
from . import exc
from . import exc as orm_exc
from ._typing import _O
from ._typing import insp_is_aliased_class
from ._typing import insp_is_mapper
from ._typing import prop_is_relationship
from .base import _class_to_mapper as _class_to_mapper
from .base import _MappedAnnotationBase
from .base import _never_set as _never_set  # noqa: F401
from .base import _none_only_set as _none_only_set  # noqa: F401
from .base import _none_set as _none_set  # noqa: F401
from .base import attribute_str as attribute_str  # noqa: F401
from .base import class_mapper as class_mapper
from .base import DynamicMapped
from .base import InspectionAttr as InspectionAttr
from .base import instance_str as instance_str  # noqa: F401
from .base import Mapped
from .base import object_mapper as object_mapper
from .base import object_state as object_state  # noqa: F401
from .base import opt_manager_of_class
from .base import ORMDescriptor
from .base import state_attribute_str as state_attribute_str  # noqa: F401
from .base import state_class_str as state_class_str  # noqa: F401
from .base import state_str as state_str  # noqa: F401
from .base import WriteOnlyMapped
from .interfaces import CriteriaOption
from .interfaces import MapperProperty as MapperProperty
from .interfaces import ORMColumnsClauseRole
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .path_registry import PathRegistry as PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import sql
from .. import util
from ..engine.result import result_tuple
from ..sql import coercions
from ..sql import expression
from ..sql import lambdas
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import is_selectable
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import ColumnCollection
from ..sql.cache_key import HasCacheKey
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import ColumnElement
from ..sql.elements import KeyedColumnElement
from ..sql.selectable import FromClause
from ..util.langhelpers import MemoizedSlots
from ..util.typing import de_stringify_annotation as _de_stringify_annotation
from ..util.typing import eval_name_only as _eval_name_only
from ..util.typing import fixup_container_fwd_refs
from ..util.typing import get_origin
from ..util.typing import is_origin_of_cls
from ..util.typing import Literal
from ..util.typing import Protocol

if typing.TYPE_CHECKING:
    from ._typing import _EntityType
    from ._typing import _IdentityKeyType
    from ._typing import _InternalEntityType
    from ._typing import _ORMCOLEXPR
    from .context import _MapperEntity
    from .context import ORMCompileState
    from .mapper import Mapper
    from .path_registry import AbstractEntityRegistry
    from .query import Query
    from .relationships import RelationshipProperty
    from ..engine import Row
    from ..engine import RowMapping
    from ..sql._typing import _CE
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _EquivalentColumnMap
    from ..sql._typing import _FromClauseArgument
    from ..sql._typing import _OnClauseArgument
    from ..sql._typing import _PropagateAttrsType
    from ..sql.annotation import _SA
    from ..sql.base import ReadOnlyColumnCollection
    from ..sql.elements import BindParameter
    from ..sql.selectable import _ColumnsClauseElement
    from ..sql.selectable import Select
    from ..sql.selectable import Selectable
    from ..sql.visitors import anon_map
    from ..util.typing import _AnnotationScanType

_T = TypeVar("_T", bound=Any)

all_cascades = frozenset(
    (
        "delete",
        "delete-orphan",
        "all",
        "merge",
        "expunge",
        "save-update",
        "refresh-expire",
        "none",
    )
)

_de_stringify_partial = functools.partial(
    functools.partial,
    locals_=util.immutabledict(
        {
            "Mapped": Mapped,
            "WriteOnlyMapped": WriteOnlyMapped,
            "DynamicMapped": DynamicMapped,
        }
    ),
)

# partial is practically useless as we have to write out the whole
# function and maintain the signature anyway


class _DeStringifyAnnotation(Protocol):
    def __call__(
        self,
        cls: Type[Any],
        annotation: _AnnotationScanType,
        originating_module: str,
        *,
        str_cleanup_fn: Optional[Callable[[str, str], str]] = None,
        include_generic: bool = False,
    ) -> Type[Any]: ...


de_stringify_annotation = cast(
    _DeStringifyAnnotation, _de_stringify_partial(_de_stringify_annotation)
)


class _EvalNameOnly(Protocol):
    def __call__(self, name: str, module_name: str) -> Any: ...


eval_name_only = cast(_EvalNameOnly, _de_stringify_partial(_eval_name_only))


class CascadeOptions(FrozenSet[str]):
    """Keeps track of the options sent to
    :paramref:`.relationship.cascade`"""

    _add_w_all_cascades = all_cascades.difference(
        ["all", "none", "delete-orphan"]
    )
    _allowed_cascades = all_cascades

    _viewonly_cascades = ["expunge", "all", "none", "refresh-expire", "merge"]

    __slots__ = (
        "save_update",
        "delete",
        "refresh_expire",
        "merge",
        "expunge",
        "delete_orphan",
    )

    save_update: bool
    delete: bool
    refresh_expire: bool
    merge: bool
    expunge: bool
    delete_orphan: bool

    def __new__(
        cls, value_list: Optional[Union[Iterable[str], str]]
    ) -> CascadeOptions:
        if isinstance(value_list, str) or value_list is None:
            return cls.from_string(value_list)  # type: ignore
        values = set(value_list)
        if values.difference(cls._allowed_cascades):
            raise sa_exc.ArgumentError(
                "Invalid cascade option(s): %s"
                % ", ".join(
                    [
                        repr(x)
                        for x in sorted(
                            values.difference(cls._allowed_cascades)
                        )
                    ]
                )
            )

        if "all" in values:
            values.update(cls._add_w_all_cascades)
        if "none" in values:
            values.clear()
        values.discard("all")

        self = super().__new__(cls, values)
        self.save_update = "save-update" in values
        self.delete = "delete" in values
        self.refresh_expire = "refresh-expire" in values
        self.merge = "merge" in values
        self.expunge = "expunge" in values
        self.delete_orphan = "delete-orphan" in values

        if self.delete_orphan and not self.delete:
            util.warn("The 'delete-orphan' cascade option requires 'delete'.")
        return self

    def __repr__(self):
        return "CascadeOptions(%r)" % (",".join([x for x in sorted(self)]))

    @classmethod
    def from_string(cls, arg):
        values = [c for c in re.split(r"\s*,\s*", arg or "") if c]
        return cls(values)


def _validator_events(desc, key, validator, include_removes, include_backrefs):
    """Runs a validation method on an attribute value to be set or
    appended.
    """

    if not include_backrefs:

        def detect_is_backref(state, initiator):
            impl = state.manager[key].impl
            return initiator.impl is not impl

    if include_removes:

        def append(state, value, initiator):
            if initiator.op is not attributes.OP_BULK_REPLACE and (
                include_backrefs or not detect_is_backref(state, initiator)
            ):
                return validator(state.obj(), key, value, False)
            else:
                return value

        def bulk_set(state, values, initiator):
            if include_backrefs or not detect_is_backref(state, initiator):
                obj = state.obj()
                values[:] = [
                    validator(obj, key, value, False) for value in values
                ]

        def set_(state, value, oldvalue, initiator):
            if include_backrefs or not detect_is_backref(state, initiator):
                return validator(state.obj(), key, value, False)
            else:
                return value

        def remove(state, value, initiator):
            if include_backrefs or not detect_is_backref(state, initiator):
                validator(state.obj(), key, value, True)

    else:

        def append(state, value, initiator):
            if initiator.op is not attributes.OP_BULK_REPLACE and (
                include_backrefs or not detect_is_backref(state, initiator)
            ):
                return validator(state.obj(), key, value)
            else:
                return value

        def bulk_set(state, values, initiator):
            if include_backrefs or not detect_is_backref(state, initiator):
                obj = state.obj()
                values[:] = [validator(obj, key, value) for value in values]

        def set_(state, value, oldvalue, initiator):
            if include_backrefs or not detect_is_backref(state, initiator):
                return validator(state.obj(), key, value)
            else:
                return value

    event.listen(desc, "append", append, raw=True, retval=True)
    event.listen(desc, "bulk_replace", bulk_set, raw=True)
    event.listen(desc, "set", set_, raw=True, retval=True)
    if include_removes:
        event.listen(desc, "remove", remove, raw=True, retval=True)


def polymorphic_union(
    table_map, typecolname, aliasname="p_union", cast_nulls=True
):
    """Create a ``UNION`` statement used by a polymorphic mapper.

    See  :ref:`concrete_inheritance` for an example of how
    this is used.

    :param table_map: mapping of polymorphic identities to
     :class:`_schema.Table` objects.
    :param typecolname: string name of a "discriminator" column, which will be
     derived from the query, producing the polymorphic identity for
     each row.  If ``None``, no polymorphic discriminator is generated.
    :param aliasname: name of the :func:`~sqlalchemy.sql.expression.alias()`
     construct generated.
    :param cast_nulls: if True, non-existent columns, which are represented
     as labeled NULLs, will be passed into CAST.   This is a legacy behavior
     that is problematic on some backends such as Oracle - in which case it
     can be set to False.

    """

    colnames: util.OrderedSet[str] = util.OrderedSet()
    colnamemaps = {}
    types = {}
    for key in table_map:
        table = table_map[key]

        table = coercions.expect(
            roles.StrictFromClauseRole, table, allow_select=True
        )
        table_map[key] = table

        m = {}
        for c in table.c:
            if c.key == typecolname:
                raise sa_exc.InvalidRequestError(
                    "Polymorphic union can't use '%s' as the discriminator "
                    "column due to mapped column %r; please apply the "
                    "'typecolname' "
                    "argument; this is available on "
                    "ConcreteBase as '_concrete_discriminator_name'"
                    % (typecolname, c)
                )
            colnames.add(c.key)
            m[c.key] = c
            types[c.key] = c.type
        colnamemaps[table] = m

    def col(name, table):
        try:
            return colnamemaps[table][name]
        except KeyError:
            if cast_nulls:
                return sql.cast(sql.null(), types[name]).label(name)
            else:
                return sql.type_coerce(sql.null(), types[name]).label(name)

    result = []
    for type_, table in table_map.items():
        if typecolname is not None:
            result.append(
                sql.select(
                    *(
                        [col(name, table) for name in colnames]
                        + [
                            sql.literal_column(
                                sql_util._quote_ddl_expr(type_)
                            ).label(typecolname)
                        ]
                    )
                ).select_from(table)
            )
        else:
            result.append(
                sql.select(
                    *[col(name, table) for name in colnames]
                ).select_from(table)
            )
    return sql.union_all(*result).alias(aliasname)


def identity_key(
    class_: Optional[Type[_T]] = None,
    ident: Union[Any, Tuple[Any, ...]] = None,
    *,
    instance: Optional[_T] = None,
    row: Optional[Union[Row[Any], RowMapping]] = None,
    identity_token: Optional[Any] = None,
) -> _IdentityKeyType[_T]:
    r"""Generate "identity key" tuples, as are used as keys in the
    :attr:`.Session.identity_map` dictionary.

    This function has several call styles:

    * ``identity_key(class, ident, identity_token=token)``

      This form receives a mapped class and a primary key scalar or
      tuple as an argument.

      E.g.::

        >>> identity_key(MyClass, (1, 2))
        (<class '__main__.MyClass'>, (1, 2), None)

      :param class: mapped class (must be a positional argument)
      :param ident: primary key, may be a scalar or tuple argument.
      :param identity_token: optional identity token

        .. versionadded:: 1.2 added identity_token


    * ``identity_key(instance=instance)``

      This form will produce the identity key for a given instance.  The
      instance need not be persistent, only that its primary key attributes
      are populated (else the key will contain ``None`` for those missing
      values).

      E.g.::

        >>> instance = MyClass(1, 2)
        >>> identity_key(instance=instance)
        (<class '__main__.MyClass'>, (1, 2), None)

      In this form, the given instance is ultimately run though
      :meth:`_orm.Mapper.identity_key_from_instance`, which will have the
      effect of performing a database check for the corresponding row
      if the object is expired.

      :param instance: object instance (must be given as a keyword arg)

    * ``identity_key(class, row=row, identity_token=token)``

      This form is similar to the class/tuple form, except is passed a
      database result row as a :class:`.Row` or :class:`.RowMapping` object.

      E.g.::

        >>> row = engine.execute(text("select * from table where a=1 and b=2")).first()
        >>> identity_key(MyClass, row=row)
        (<class '__main__.MyClass'>, (1, 2), None)

      :param class: mapped class (must be a positional argument)
      :param row: :class:`.Row` row returned by a :class:`_engine.CursorResult`
       (must be given as a keyword arg)
      :param identity_token: optional identity token

        .. versionadded:: 1.2 added identity_token

    """  # noqa: E501
    if class_ is not None:
        mapper = class_mapper(class_)
        if row is None:
            if ident is None:
                raise sa_exc.ArgumentError("ident or row is required")
            return mapper.identity_key_from_primary_key(
                tuple(util.to_list(ident)), identity_token=identity_token
            )
        else:
            return mapper.identity_key_from_row(
                row, identity_token=identity_token
            )
    elif instance is not None:
        mapper = object_mapper(instance)
        return mapper.identity_key_from_instance(instance)
    else:
        raise sa_exc.ArgumentError("class or instance is required")


class _TraceAdaptRole(enum.Enum):
    """Enumeration of all the use cases for ORMAdapter.

    ORMAdapter remains one of the most complicated aspects of the ORM, as it is
    used for in-place adaption of column expressions to be applied to a SELECT,
    replacing :class:`.Table` and other objects that are mapped to classes with
    aliases of those tables in the case of joined eager loading, or in the case
    of polymorphic loading as used with concrete mappings or other custom "with
    polymorphic" parameters, with whole user-defined subqueries. The
    enumerations provide an overview of all the use cases used by ORMAdapter, a
    layer of formality as to the introduction of new ORMAdapter use cases (of
    which none are anticipated), as well as a means to trace the origins of a
    particular ORMAdapter within runtime debugging.

    SQLAlchemy 2.0 has greatly scaled back ORM features which relied heavily on
    open-ended statement adaption, including the ``Query.with_polymorphic()``
    method and the ``Query.select_from_entity()`` methods, favoring
    user-explicit aliasing schemes using the ``aliased()`` and
    ``with_polymorphic()`` standalone constructs; these still use adaption,
    however the adaption is applied in a narrower scope.

    """

    # aliased() use that is used to adapt individual attributes at query
    # construction time
    ALIASED_INSP = enum.auto()

    # joinedload cases; typically adapt an ON clause of a relationship
    # join
    JOINEDLOAD_USER_DEFINED_ALIAS = enum.auto()
    JOINEDLOAD_PATH_WITH_POLYMORPHIC = enum.auto()
    JOINEDLOAD_MEMOIZED_ADAPTER = enum.auto()

    # polymorphic cases - these are complex ones that replace FROM
    # clauses, replacing tables with subqueries
    MAPPER_POLYMORPHIC_ADAPTER = enum.auto()
    WITH_POLYMORPHIC_ADAPTER = enum.auto()
    WITH_POLYMORPHIC_ADAPTER_RIGHT_JOIN = enum.auto()
    DEPRECATED_JOIN_ADAPT_RIGHT_SIDE = enum.auto()

    # the from_statement() case, used only to adapt individual attributes
    # from a given statement to local ORM attributes at result fetching
    # time.  assigned to ORMCompileState._from_obj_alias
    ADAPT_FROM_STATEMENT = enum.auto()

    # the joinedload for queries that have LIMIT/OFFSET/DISTINCT case;
    # the query is placed inside of a subquery with the LIMIT/OFFSET/etc.,
    # joinedloads are then placed on the outside.
    # assigned to ORMCompileState.compound_eager_adapter
    COMPOUND_EAGER_STATEMENT = enum.auto()

    # the legacy Query._set_select_from() case.
    # this is needed for Query's set operations (i.e. UNION, etc. )
    # as well as "legacy from_self()", which while removed from 2.0 as
    # public API, is used for the Query.count() method.  this one
    # still does full statement traversal
    # assigned to ORMCompileState._from_obj_alias
    LEGACY_SELECT_FROM_ALIAS = enum.auto()


class ORMStatementAdapter(sql_util.ColumnAdapter):
    """ColumnAdapter which includes a role attribute."""

    __slots__ = ("role",)

    def __init__(
        self,
        role: _TraceAdaptRole,
        selectable: Selectable,
        *,
        equivalents: Optional[_EquivalentColumnMap] = None,
        adapt_required: bool = False,
        allow_label_resolve: bool = True,
        anonymize_labels: bool = False,
        adapt_on_names: bool = False,
        adapt_from_selectables: Optional[AbstractSet[FromClause]] = None,
    ):
        self.role = role
        super().__init__(
            selectable,
            equivalents=equivalents,
            adapt_required=adapt_required,
            allow_label_resolve=allow_label_resolve,
            anonymize_labels=anonymize_labels,
            adapt_on_names=adapt_on_names,
            adapt_from_selectables=adapt_from_selectables,
        )


class ORMAdapter(sql_util.ColumnAdapter):
    """ColumnAdapter subclass which excludes adaptation of entities from
    non-matching mappers.

    """

    __slots__ = ("role", "mapper", "is_aliased_class", "aliased_insp")

    is_aliased_class: bool
    aliased_insp: Optional[AliasedInsp[Any]]

    def __init__(
        self,
        role: _TraceAdaptRole,
        entity: _InternalEntityType[Any],
        *,
        equivalents: Optional[_EquivalentColumnMap] = None,
        adapt_required: bool = False,
        allow_label_resolve: bool = True,
        anonymize_labels: bool = False,
        selectable: Optional[Selectable] = None,
        limit_on_entity: bool = True,
        adapt_on_names: bool = False,
        adapt_from_selectables: Optional[AbstractSet[FromClause]] = None,
    ):
        self.role = role
        self.mapper = entity.mapper
        if selectable is None:
            selectable = entity.selectable
        if insp_is_aliased_class(entity):
            self.is_aliased_class = True
            self.aliased_insp = entity
        else:
            self.is_aliased_class = False
            self.aliased_insp = None

        super().__init__(
            selectable,
            equivalents,
            adapt_required=adapt_required,
            allow_label_resolve=allow_label_resolve,
            anonymize_labels=anonymize_labels,
            include_fn=self._include_fn if limit_on_entity else None,
            adapt_on_names=adapt_on_names,
            adapt_from_selectables=adapt_from_selectables,
        )

    def _include_fn(self, elem):
        entity = elem._annotations.get("parentmapper", None)

        return not entity or entity.isa(self.mapper) or self.mapper.isa(entity)


class AliasedClass(
    inspection.Inspectable["AliasedInsp[_O]"], ORMColumnsClauseRole[_O]
):
    r"""Represents an "aliased" form of a mapped class for usage with Query.

    The ORM equivalent of a :func:`~sqlalchemy.sql.expression.alias`
    construct, this object mimics the mapped class using a
    ``__getattr__`` scheme and maintains a reference to a
    real :class:`~sqlalchemy.sql.expression.Alias` object.

    A primary purpose of :class:`.AliasedClass` is to serve as an alternate
    within a SQL statement generated by the ORM, such that an existing
    mapped entity can be used in multiple contexts.   A simple example::

        # find all pairs of users with the same name
        user_alias = aliased(User)
        session.query(User, user_alias).join(
            (user_alias, User.id > user_alias.id)
        ).filter(User.name == user_alias.name)

    :class:`.AliasedClass` is also capable of mapping an existing mapped
    class to an entirely new selectable, provided this selectable is column-
    compatible with the existing mapped selectable, and it can also be
    configured in a mapping as the target of a :func:`_orm.relationship`.
    See the links below for examples.

    The :class:`.AliasedClass` object is constructed typically using the
    :func:`_orm.aliased` function.   It also is produced with additional
    configuration when using the :func:`_orm.with_polymorphic` function.

    The resulting object is an instance of :class:`.AliasedClass`.
    This object implements an attribute scheme which produces the
    same attribute and method interface as the original mapped
    class, allowing :class:`.AliasedClass` to be compatible
    with any attribute technique which works on the original class,
    including hybrid attributes (see :ref:`hybrids_toplevel`).

    The :class:`.AliasedClass` can be inspected for its underlying
    :class:`_orm.Mapper`, aliased selectable, and other information
    using :func:`_sa.inspect`::

        from sqlalchemy import inspect

        my_alias = aliased(MyClass)
        insp = inspect(my_alias)

    The resulting inspection object is an instance of :class:`.AliasedInsp`.


    .. seealso::

        :func:`.aliased`

        :func:`.with_polymorphic`

        :ref:`relationship_aliased_class`

        :ref:`relationship_to_window_function`


    """

    __name__: str

    def __init__(
        self,
        mapped_class_or_ac: _EntityType[_O],
        alias: Optional[FromClause] = None,
        name: Optional[str] = None,
        flat: bool = False,
        adapt_on_names: bool = False,
        with_polymorphic_mappers: Optional[Sequence[Mapper[Any]]] = None,
        with_polymorphic_discriminator: Optional[ColumnElement[Any]] = None,
        base_alias: Optional[AliasedInsp[Any]] = None,
        use_mapper_path: bool = False,
        represents_outer_join: bool = False,
    ):
        insp = cast(
            "_InternalEntityType[_O]", inspection.inspect(mapped_class_or_ac)
        )
        mapper = insp.mapper

        nest_adapters = False

        if alias is None:
            if insp.is_aliased_class and insp.selectable._is_subquery:
                alias = insp.selectable.alias()
            else:
                alias = (
                    mapper._with_polymorphic_selectable._anonymous_fromclause(
                        name=name,
                        flat=flat,
                    )
                )
        elif insp.is_aliased_class:
            nest_adapters = True

        assert alias is not None
        self._aliased_insp = AliasedInsp(
            self,
            insp,
            alias,
            name,
            (
                with_polymorphic_mappers
                if with_polymorphic_mappers
                else mapper.with_polymorphic_mappers
            ),
            (
                with_polymorphic_discriminator
                if with_polymorphic_discriminator is not None
                else mapper.polymorphic_on
            ),
            base_alias,
            use_mapper_path,
            adapt_on_names,
            represents_outer_join,
            nest_adapters,
        )

        self.__name__ = f"aliased({mapper.class_.__name__})"

    @classmethod
    def _reconstitute_from_aliased_insp(
        cls, aliased_insp: AliasedInsp[_O]
    ) -> AliasedClass[_O]:
        obj = cls.__new__(cls)
        obj.__name__ = f"aliased({aliased_insp.mapper.class_.__name__})"
        obj._aliased_insp = aliased_insp

        if aliased_insp._is_with_polymorphic:
            for sub_aliased_insp in aliased_insp._with_polymorphic_entities:
                if sub_aliased_insp is not aliased_insp:
                    ent = AliasedClass._reconstitute_from_aliased_insp(
                        sub_aliased_insp
                    )
                    setattr(obj, sub_aliased_insp.class_.__name__, ent)

        return obj

    def __getattr__(self, key: str) -> Any:
        try:
            _aliased_insp = self.__dict__["_aliased_insp"]
        except KeyError:
            raise AttributeError()
        else:
            target = _aliased_insp._target
            # maintain all getattr mechanics
            attr = getattr(target, key)

        # attribute is a method, that will be invoked against a
        # "self"; so just return a new method with the same function and
        # new self
        if hasattr(attr, "__call__") and hasattr(attr, "__self__"):
            return types.MethodType(attr.__func__, self)

        # attribute is a descriptor, that will be invoked against a
        # "self"; so invoke the descriptor against this self
        if hasattr(attr, "__get__"):
            attr = attr.__get__(None, self)

        # attributes within the QueryableAttribute system will want this
        # to be invoked so the object can be adapted
        if hasattr(attr, "adapt_to_entity"):
            attr = attr.adapt_to_entity(_aliased_insp)
            setattr(self, key, attr)

        return attr

    def _get_from_serialized(
        self, key: str, mapped_class: _O, aliased_insp: AliasedInsp[_O]
    ) -> Any:
        # this method is only used in terms of the
        # sqlalchemy.ext.serializer extension
        attr = getattr(mapped_class, key)
        if hasattr(attr, "__call__") and hasattr(attr, "__self__"):
            return types.MethodType(attr.__func__, self)

        # attribute is a descriptor, that will be invoked against a
        # "self"; so invoke the descriptor against this self
        if hasattr(attr, "__get__"):
            attr = attr.__get__(None, self)

        # attributes within the QueryableAttribute system will want this
        # to be invoked so the object can be adapted
        if hasattr(attr, "adapt_to_entity"):
            aliased_insp._weak_entity = weakref.ref(self)
            attr = attr.adapt_to_entity(aliased_insp)
            setattr(self, key, attr)

        return attr

    def __repr__(self) -> str:
        return "<AliasedClass at 0x%x; %s>" % (
            id(self),
            self._aliased_insp._target.__name__,
        )

    def __str__(self) -> str:
        return str(self._aliased_insp)


@inspection._self_inspects
class AliasedInsp(
    ORMEntityColumnsClauseRole[_O],
    ORMFromClauseRole,
    HasCacheKey,
    InspectionAttr,
    MemoizedSlots,
    inspection.Inspectable["AliasedInsp[_O]"],
    Generic[_O],
):
    """Provide an inspection interface for an
    :class:`.AliasedClass` object.

    The :class:`.AliasedInsp` object is returned
    given an :class:`.AliasedClass` using the
    :func:`_sa.inspect` function::

        from sqlalchemy import inspect
        from sqlalchemy.orm import aliased

        my_alias = aliased(MyMappedClass)
        insp = inspect(my_alias)

    Attributes on :class:`.AliasedInsp`
    include:

    * ``entity`` - the :class:`.AliasedClass` represented.
    * ``mapper`` - the :class:`_orm.Mapper` mapping the underlying class.
    * ``selectable`` - the :class:`_expression.Alias`
      construct which ultimately
      represents an aliased :class:`_schema.Table` or
      :class:`_expression.Select`
      construct.
    * ``name`` - the name of the alias.  Also is used as the attribute
      name when returned in a result tuple from :class:`_query.Query`.
    * ``with_polymorphic_mappers`` - collection of :class:`_orm.Mapper`
      objects
      indicating all those mappers expressed in the select construct
      for the :class:`.AliasedClass`.
    * ``polymorphic_on`` - an alternate column or SQL expression which
      will be used as the "discriminator" for a polymorphic load.

    .. seealso::

        :ref:`inspection_toplevel`

    """

    __slots__ = (
        "__weakref__",
        "_weak_entity",
        "mapper",
        "selectable",
        "name",
        "_adapt_on_names",
        "with_polymorphic_mappers",
        "polymorphic_on",
        "_use_mapper_path",
        "_base_alias",
        "represents_outer_join",
        "persist_selectable",
        "local_table",
        "_is_with_polymorphic",
        "_with_polymorphic_entities",
        "_adapter",
        "_target",
        "__clause_element__",
        "_memoized_values",
        "_all_column_expressions",
        "_nest_adapters",
    )

    _cache_key_traversal = [
        ("name", visitors.ExtendedInternalTraversal.dp_string),
        ("_adapt_on_names", visitors.ExtendedInternalTraversal.dp_boolean),
        ("_use_mapper_path", visitors.ExtendedInternalTraversal.dp_boolean),
        ("_target", visitors.ExtendedInternalTraversal.dp_inspectable),
        ("selectable", visitors.ExtendedInternalTraversal.dp_clauseelement),
        (
            "with_polymorphic_mappers",
            visitors.InternalTraversal.dp_has_cache_key_list,
        ),
        ("polymorphic_on", visitors.InternalTraversal.dp_clauseelement),
    ]

    mapper: Mapper[_O]
    selectable: FromClause
    _adapter: ORMAdapter
    with_polymorphic_mappers: Sequence[Mapper[Any]]
    _with_polymorphic_entities: Sequence[AliasedInsp[Any]]

    _weak_entity: weakref.ref[AliasedClass[_O]]
    """the AliasedClass that refers to this AliasedInsp"""

    _target: Union[Type[_O], AliasedClass[_O]]
    """the thing referenced by the AliasedClass/AliasedInsp.

    In the vast majority of cases, this is the mapped class.  However
    it may also be another AliasedClass (alias of alias).

    """

    def __init__(
        self,
        entity: AliasedClass[_O],
        inspected: _InternalEntityType[_O],
        selectable: FromClause,
        name: Optional[str],
        with_polymorphic_mappers: Optional[Sequence[Mapper[Any]]],
        polymorphic_on: Optional[ColumnElement[Any]],
        _base_alias: Optional[AliasedInsp[Any]],
        _use_mapper_path: bool,
        adapt_on_names: bool,
        represents_outer_join: bool,
        nest_adapters: bool,
    ):
        mapped_class_or_ac = inspected.entity
        mapper = inspected.mapper

        self._weak_entity = weakref.ref(entity)
        self.mapper = mapper
        self.selectable = self.persist_selectable = self.local_table = (
            selectable
        )
        self.name = name
        self.polymorphic_on = polymorphic_on
        self._base_alias = weakref.ref(_base_alias or self)
        self._use_mapper_path = _use_mapper_path
        self.represents_outer_join = represents_outer_join
        self._nest_adapters = nest_adapters

        if with_polymorphic_mappers:
            self._is_with_polymorphic = True
            self.with_polymorphic_mappers = with_polymorphic_mappers
            self._with_polymorphic_entities = []
            for poly in self.with_polymorphic_mappers:
                if poly is not mapper:
                    ent = AliasedClass(
                        poly.class_,
                        selectable,
                        base_alias=self,
                        adapt_on_names=adapt_on_names,
                        use_mapper_path=_use_mapper_path,
                    )

                    setattr(self.entity, poly.class_.__name__, ent)
                    self._with_polymorphic_entities.append(ent._aliased_insp)

        else:
            self._is_with_polymorphic = False
            self.with_polymorphic_mappers = [mapper]

        self._adapter = ORMAdapter(
            _TraceAdaptRole.ALIASED_INSP,
            mapper,
            selectable=selectable,
            equivalents=mapper._equivalent_columns,
            adapt_on_names=adapt_on_names,
            anonymize_labels=True,
            # make sure the adapter doesn't try to grab other tables that
            # are not even the thing we are mapping, such as embedded
            # selectables in subqueries or CTEs.  See issue #6060
            adapt_from_selectables={
                m.selectable
                for m in self.with_polymorphic_mappers
                if not adapt_on_names
            },
            limit_on_entity=False,
        )

        if nest_adapters:
            # supports "aliased class of aliased class" use case
            assert isinstance(inspected, AliasedInsp)
            self._adapter = inspected._adapter.wrap(self._adapter)

        self._adapt_on_names = adapt_on_names
        self._target = mapped_class_or_ac

    @classmethod
    def _alias_factory(
        cls,
        element: Union[_EntityType[_O], FromClause],
        alias: Optional[FromClause] = None,
        name: Optional[str] = None,
        flat: bool = False,
        adapt_on_names: bool = False,
    ) -> Union[AliasedClass[_O], FromClause]:
        if isinstance(element, FromClause):
            if adapt_on_names:
                raise sa_exc.ArgumentError(
                    "adapt_on_names only applies to ORM elements"
                )
            if name:
                return element.alias(name=name, flat=flat)
            else:
                return coercions.expect(
                    roles.AnonymizedFromClauseRole, element, flat=flat
                )
        else:
            return AliasedClass(
                element,
                alias=alias,
                flat=flat,
                name=name,
                adapt_on_names=adapt_on_names,
            )

    @classmethod
    def _with_polymorphic_factory(
        cls,
        base: Union[Type[_O], Mapper[_O]],
        classes: Union[Literal["*"], Iterable[_EntityType[Any]]],
        selectable: Union[Literal[False, None], FromClause] = False,
        flat: bool = False,
        polymorphic_on: Optional[ColumnElement[Any]] = None,
        aliased: bool = False,
        innerjoin: bool = False,
        adapt_on_names: bool = False,
        name: Optional[str] = None,
        _use_mapper_path: bool = False,
    ) -> AliasedClass[_O]:
        primary_mapper = _class_to_mapper(base)

        if selectable not in (None, False) and flat:
            raise sa_exc.ArgumentError(
                "the 'flat' and 'selectable' arguments cannot be passed "
                "simultaneously to with_polymorphic()"
            )

        mappers, selectable = primary_mapper._with_polymorphic_args(
            classes, selectable, innerjoin=innerjoin
        )
        if aliased or flat:
            assert selectable is not None
            selectable = selectable._anonymous_fromclause(flat=flat)

        return AliasedClass(
            base,
            selectable,
            name=name,
            with_polymorphic_mappers=mappers,
            adapt_on_names=adapt_on_names,
            with_polymorphic_discriminator=polymorphic_on,
            use_mapper_path=_use_mapper_path,
            represents_outer_join=not innerjoin,
        )

    @property
    def entity(self) -> AliasedClass[_O]:
        # to eliminate reference cycles, the AliasedClass is held weakly.
        # this produces some situations where the AliasedClass gets lost,
        # particularly when one is created internally and only the AliasedInsp
        # is passed around.
        # to work around this case, we just generate a new one when we need
        # it, as it is a simple class with very little initial state on it.
        ent = self._weak_entity()
        if ent is None:
            ent = AliasedClass._reconstitute_from_aliased_insp(self)
            self._weak_entity = weakref.ref(ent)
        return ent

    is_aliased_class = True
    "always returns True"

    def _memoized_method___clause_element__(self) -> FromClause:
        return self.selectable._annotate(
            {
                "parentmapper": self.mapper,
                "parententity": self,
                "entity_namespace": self,
            }
        )._set_propagate_attrs(
            {"compile_state_plugin": "orm", "plugin_subject": self}
        )

    @property
    def entity_namespace(self) -> AliasedClass[_O]:
        return self.entity

    @property
    def class_(self) -> Type[_O]:
        """Return the mapped class ultimately represented by this
        :class:`.AliasedInsp`."""
        return self.mapper.class_

    @property
    def _path_registry(self) -> AbstractEntityRegistry:
        if self._use_mapper_path:
            return self.mapper._path_registry
        else:
            return PathRegistry.per_mapper(self)

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "mapper": self.mapper,
            "alias": self.selectable,
            "name": self.name,
            "adapt_on_names": self._adapt_on_names,
            "with_polymorphic_mappers": self.with_polymorphic_mappers,
            "with_polymorphic_discriminator": self.polymorphic_on,
            "base_alias": self._base_alias(),
            "use_mapper_path": self._use_mapper_path,
            "represents_outer_join": self.represents_outer_join,
            "nest_adapters": self._nest_adapters,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__init__(  # type: ignore
            state["entity"],
            state["mapper"],
            state["alias"],
            state["name"],
            state["with_polymorphic_mappers"],
            state["with_polymorphic_discriminator"],
            state["base_alias"],
            state["use_mapper_path"],
            state["adapt_on_names"],
            state["represents_outer_join"],
            state["nest_adapters"],
        )

    def _merge_with(self, other: AliasedInsp[_O]) -> AliasedInsp[_O]:
        # assert self._is_with_polymorphic
        # assert other._is_with_polymorphic

        primary_mapper = other.mapper

        assert self.mapper is primary_mapper

        our_classes = util.to_set(
            mp.class_ for mp in self.with_polymorphic_mappers
        )
        new_classes = {mp.class_ for mp in other.with_polymorphic_mappers}
        if our_classes == new_classes:
            return other
        else:
            classes = our_classes.union(new_classes)

        mappers, selectable = primary_mapper._with_polymorphic_args(
            classes, None, innerjoin=not other.represents_outer_join
        )
        selectable = selectable._anonymous_fromclause(flat=True)
        return AliasedClass(
            primary_mapper,
            selectable,
            with_polymorphic_mappers=mappers,
            with_polymorphic_discriminator=other.polymorphic_on,
            use_mapper_path=other._use_mapper_path,
            represents_outer_join=other.represents_outer_join,
        )._aliased_insp

    def _adapt_element(
        self, expr: _ORMCOLEXPR, key: Optional[str] = None
    ) -> _ORMCOLEXPR:
        assert isinstance(expr, ColumnElement)
        d: Dict[str, Any] = {
            "parententity": self,
            "parentmapper": self.mapper,
        }
        if key:
            d["proxy_key"] = key

        # IMO mypy should see this one also as returning the same type
        # we put into it, but it's not
        return (
            self._adapter.traverse(expr)
            ._annotate(d)
            ._set_propagate_attrs(
                {"compile_state_plugin": "orm", "plugin_subject": self}
            )
        )

    if TYPE_CHECKING:
        # establish compatibility with the _ORMAdapterProto protocol,
        # which in turn is compatible with _CoreAdapterProto.

        def _orm_adapt_element(
            self,
            obj: _CE,
            key: Optional[str] = None,
        ) -> _CE: ...

    else:
        _orm_adapt_element = _adapt_element

    def _entity_for_mapper(self, mapper):
        self_poly = self.with_polymorphic_mappers
        if mapper in self_poly:
            if mapper is self.mapper:
                return self
            else:
                return getattr(
                    self.entity, mapper.class_.__name__
                )._aliased_insp
        elif mapper.isa(self.mapper):
            return self
        else:
            assert False, "mapper %s doesn't correspond to %s" % (mapper, self)

    def _memoized_attr__get_clause(self):
        onclause, replacemap = self.mapper._get_clause
        return (
            self._adapter.traverse(onclause),
            {
                self._adapter.traverse(col): param
                for col, param in replacemap.items()
            },
        )

    def _memoized_attr__memoized_values(self):
        return {}

    def _memoized_attr__all_column_expressions(self):
        if self._is_with_polymorphic:
            cols_plus_keys = self.mapper._columns_plus_keys(
                [ent.mapper for ent in self._with_polymorphic_entities]
            )
        else:
            cols_plus_keys = self.mapper._columns_plus_keys()

        cols_plus_keys = [
            (key, self._adapt_element(col)) for key, col in cols_plus_keys
        ]

        return ColumnCollection(cols_plus_keys)

    def _memo(self, key, callable_, *args, **kw):
        if key in self._memoized_values:
            return self._memoized_values[key]
        else:
            self._memoized_values[key] = value = callable_(*args, **kw)
            return value

    def __repr__(self):
        if self.with_polymorphic_mappers:
            with_poly = "(%s)" % ", ".join(
                mp.class_.__name__ for mp in self.with_polymorphic_mappers
            )
        else:
            with_poly = ""
        return "<AliasedInsp at 0x%x; %s%s>" % (
            id(self),
            self.class_.__name__,
            with_poly,
        )

    def __str__(self):
        if self._is_with_polymorphic:
            return "with_polymorphic(%s, [%s])" % (
                self._target.__name__,
                ", ".join(
                    mp.class_.__name__
                    for mp in self.with_polymorphic_mappers
                    if mp is not self.mapper
                ),
            )
        else:
            return "aliased(%s)" % (self._target.__name__,)


class _WrapUserEntity:
    """A wrapper used within the loader_criteria lambda caller so that
    we can bypass declared_attr descriptors on unmapped mixins, which
    normally emit a warning for such use.

    might also be useful for other per-lambda instrumentations should
    the need arise.

    """

    __slots__ = ("subject",)

    def __init__(self, subject):
        self.subject = subject

    @util.preload_module("sqlalchemy.orm.decl_api")
    def __getattribute__(self, name):
        decl_api = util.preloaded.orm.decl_api

        subject = object.__getattribute__(self, "subject")
        if name in subject.__dict__ and isinstance(
            subject.__dict__[name], decl_api.declared_attr
        ):
            return subject.__dict__[name].fget(subject)
        else:
            return getattr(subject, name)


class LoaderCriteriaOption(CriteriaOption):
    """Add additional WHERE criteria to the load for all occurrences of
    a particular entity.

    :class:`_orm.LoaderCriteriaOption` is invoked using the
    :func:`_orm.with_loader_criteria` function; see that function for
    details.

    .. versionadded:: 1.4

    """

    __slots__ = (
        "root_entity",
        "entity",
        "deferred_where_criteria",
        "where_criteria",
        "_where_crit_orig",
        "include_aliases",
        "propagate_to_loaders",
    )

    _traverse_internals = [
        ("root_entity", visitors.ExtendedInternalTraversal.dp_plain_obj),
        ("entity", visitors.ExtendedInternalTraversal.dp_has_cache_key),
        ("where_criteria", visitors.InternalTraversal.dp_clauseelement),
        ("include_aliases", visitors.InternalTraversal.dp_boolean),
        ("propagate_to_loaders", visitors.InternalTraversal.dp_boolean),
    ]

    root_entity: Optional[Type[Any]]
    entity: Optional[_InternalEntityType[Any]]
    where_criteria: Union[ColumnElement[bool], lambdas.DeferredLambdaElement]
    deferred_where_criteria: bool
    include_aliases: bool
    propagate_to_loaders: bool

    _where_crit_orig: Any

    def __init__(
        self,
        entity_or_base: _EntityType[Any],
        where_criteria: Union[
            _ColumnExpressionArgument[bool],
            Callable[[Any], _ColumnExpressionArgument[bool]],
        ],
        loader_only: bool = False,
        include_aliases: bool = False,
        propagate_to_loaders: bool = True,
        track_closure_variables: bool = True,
    ):
        entity = cast(
            "_InternalEntityType[Any]",
            inspection.inspect(entity_or_base, False),
        )
        if entity is None:
            self.root_entity = cast("Type[Any]", entity_or_base)
            self.entity = None
        else:
            self.root_entity = None
            self.entity = entity

        self._where_crit_orig = where_criteria
        if callable(where_criteria):
            if self.root_entity is not None:
                wrap_entity = self.root_entity
            else:
                assert entity is not None
                wrap_entity = entity.entity

            self.deferred_where_criteria = True
            self.where_criteria = lambdas.DeferredLambdaElement(
                where_criteria,
                roles.WhereHavingRole,
                lambda_args=(_WrapUserEntity(wrap_entity),),
                opts=lambdas.LambdaOptions(
                    track_closure_variables=track_closure_variables
                ),
            )
        else:
            self.deferred_where_criteria = False
            self.where_criteria = coercions.expect(
                roles.WhereHavingRole, where_criteria
            )

        self.include_aliases = include_aliases
        self.propagate_to_loaders = propagate_to_loaders

    @classmethod
    def _unreduce(
        cls, entity, where_criteria, include_aliases, propagate_to_loaders
    ):
        return LoaderCriteriaOption(
            entity,
            where_criteria,
            include_aliases=include_aliases,
            propagate_to_loaders=propagate_to_loaders,
        )

    def __reduce__(self):
        return (
            LoaderCriteriaOption._unreduce,
            (
                self.entity.class_ if self.entity else self.root_entity,
                self._where_crit_orig,
                self.include_aliases,
                self.propagate_to_loaders,
            ),
        )

    def _all_mappers(self) -> Iterator[Mapper[Any]]:
        if self.entity:
            yield from self.entity.mapper.self_and_descendants
        else:
            assert self.root_entity
            stack = list(self.root_entity.__subclasses__())
            while stack:
                subclass = stack.pop(0)
                ent = cast(
                    "_InternalEntityType[Any]",
                    inspection.inspect(subclass, raiseerr=False),
                )
                if ent:
                    yield from ent.mapper.self_and_descendants
                else:
                    stack.extend(subclass.__subclasses__())

    def _should_include(self, compile_state: ORMCompileState) -> bool:
        if (
            compile_state.select_statement._annotations.get(
                "for_loader_criteria", None
            )
            is self
        ):
            return False
        return True

    def _resolve_where_criteria(
        self, ext_info: _InternalEntityType[Any]
    ) -> ColumnElement[bool]:
        if self.deferred_where_criteria:
            crit = cast(
                "ColumnElement[bool]",
                self.where_criteria._resolve_with_args(ext_info.entity),
            )
        else:
            crit = self.where_criteria  # type: ignore
        assert isinstance(crit, ColumnElement)
        return sql_util._deep_annotate(
            crit,
            {"for_loader_criteria": self},
            detect_subquery_cols=True,
            ind_cols_on_fromclause=True,
        )

    def process_compile_state_replaced_entities(
        self,
        compile_state: ORMCompileState,
        mapper_entities: Iterable[_MapperEntity],
    ) -> None:
        self.process_compile_state(compile_state)

    def process_compile_state(self, compile_state: ORMCompileState) -> None:
        """Apply a modification to a given :class:`.CompileState`."""

        # if options to limit the criteria to immediate query only,
        # use compile_state.attributes instead

        self.get_global_criteria(compile_state.global_attributes)

    def get_global_criteria(self, attributes: Dict[Any, Any]) -> None:
        for mp in self._all_mappers():
            load_criteria = attributes.setdefault(
                ("additional_entity_criteria", mp), []
            )

            load_criteria.append(self)


inspection._inspects(AliasedClass)(lambda target: target._aliased_insp)


@inspection._inspects(type)
def _inspect_mc(
    class_: Type[_O],
) -> Optional[Mapper[_O]]:
    try:
        class_manager = opt_manager_of_class(class_)
        if class_manager is None or not class_manager.is_mapped:
            return None
        mapper = class_manager.mapper
    except exc.NO_STATE:
        return None
    else:
        return mapper


GenericAlias = type(List[Any])


@inspection._inspects(GenericAlias)
def _inspect_generic_alias(
    class_: Type[_O],
) -> Optional[Mapper[_O]]:
    origin = cast("Type[_O]", get_origin(class_))
    return _inspect_mc(origin)


@inspection._self_inspects
class Bundle(
    ORMColumnsClauseRole[_T],
    SupportsCloneAnnotations,
    MemoizedHasCacheKey,
    inspection.Inspectable["Bundle[_T]"],
    InspectionAttr,
):
    """A grouping of SQL expressions that are returned by a :class:`.Query`
    under one namespace.

    The :class:`.Bundle` essentially allows nesting of the tuple-based
    results returned by a column-oriented :class:`_query.Query` object.
    It also
    is extensible via simple subclassing, where the primary capability
    to override is that of how the set of expressions should be returned,
    allowing post-processing as well as custom return types, without
    involving ORM identity-mapped classes.

    .. seealso::

        :ref:`bundles`


    """

    single_entity = False
    """If True, queries for a single Bundle will be returned as a single
    entity, rather than an element within a keyed tuple."""

    is_clause_element = False

    is_mapper = False

    is_aliased_class = False

    is_bundle = True

    _propagate_attrs: _PropagateAttrsType = util.immutabledict()

    proxy_set = util.EMPTY_SET  # type: ignore

    exprs: List[_ColumnsClauseElement]

    def __init__(
        self, name: str, *exprs: _ColumnExpressionArgument[Any], **kw: Any
    ):
        r"""Construct a new :class:`.Bundle`.

        e.g.::

            bn = Bundle("mybundle", MyClass.x, MyClass.y)

            for row in session.query(bn).filter(bn.c.x == 5).filter(bn.c.y == 4):
                print(row.mybundle.x, row.mybundle.y)

        :param name: name of the bundle.
        :param \*exprs: columns or SQL expressions comprising the bundle.
        :param single_entity=False: if True, rows for this :class:`.Bundle`
         can be returned as a "single entity" outside of any enclosing tuple
         in the same manner as a mapped entity.

        """  # noqa: E501
        self.name = self._label = name
        coerced_exprs = [
            coercions.expect(
                roles.ColumnsClauseRole, expr, apply_propagate_attrs=self
            )
            for expr in exprs
        ]
        self.exprs = coerced_exprs

        self.c = self.columns = ColumnCollection(
            (getattr(col, "key", col._label), col)
            for col in [e._annotations.get("bundle", e) for e in coerced_exprs]
        ).as_readonly()
        self.single_entity = kw.pop("single_entity", self.single_entity)

    def _gen_cache_key(
        self, anon_map: anon_map, bindparams: List[BindParameter[Any]]
    ) -> Tuple[Any, ...]:
        return (self.__class__, self.name, self.single_entity) + tuple(
            [expr._gen_cache_key(anon_map, bindparams) for expr in self.exprs]
        )

    @property
    def mapper(self) -> Optional[Mapper[Any]]:
        mp: Optional[Mapper[Any]] = self.exprs[0]._annotations.get(
            "parentmapper", None
        )
        return mp

    @property
    def entity(self) -> Optional[_InternalEntityType[Any]]:
        ie: Optional[_InternalEntityType[Any]] = self.exprs[
            0
        ]._annotations.get("parententity", None)
        return ie

    @property
    def entity_namespace(
        self,
    ) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        return self.c

    columns: ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]

    """A namespace of SQL expressions referred to by this :class:`.Bundle`.

        e.g.::

            bn = Bundle("mybundle", MyClass.x, MyClass.y)

            q = sess.query(bn).filter(bn.c.x == 5)

        Nesting of bundles is also supported::

            b1 = Bundle(
                "b1",
                Bundle("b2", MyClass.a, MyClass.b),
                Bundle("b3", MyClass.x, MyClass.y),
            )

            q = sess.query(b1).filter(b1.c.b2.c.a == 5).filter(b1.c.b3.c.y == 9)

    .. seealso::

        :attr:`.Bundle.c`

    """  # noqa: E501

    c: ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]
    """An alias for :attr:`.Bundle.columns`."""

    def _clone(self, **kw):
        cloned = self.__class__.__new__(self.__class__)
        cloned.__dict__.update(self.__dict__)
        return cloned

    def __clause_element__(self):
        # ensure existing entity_namespace remains
        annotations = {"bundle": self, "entity_namespace": self}
        annotations.update(self._annotations)

        plugin_subject = self.exprs[0]._propagate_attrs.get(
            "plugin_subject", self.entity
        )
        return (
            expression.ClauseList(
                _literal_as_text_role=roles.ColumnsClauseRole,
                group=False,
                *[e._annotations.get("bundle", e) for e in self.exprs],
            )
            ._annotate(annotations)
            ._set_propagate_attrs(
                # the Bundle *must* use the orm plugin no matter what.  the
                # subject can be None but it's much better if it's not.
                {
                    "compile_state_plugin": "orm",
                    "plugin_subject": plugin_subject,
                }
            )
        )

    @property
    def clauses(self):
        return self.__clause_element__().clauses

    def label(self, name):
        """Provide a copy of this :class:`.Bundle` passing a new label."""

        cloned = self._clone()
        cloned.name = name
        return cloned

    def create_row_processor(
        self,
        query: Select[Any],
        procs: Sequence[Callable[[Row[Any]], Any]],
        labels: Sequence[str],
    ) -> Callable[[Row[Any]], Any]:
        """Produce the "row processing" function for this :class:`.Bundle`.

        May be overridden by subclasses to provide custom behaviors when
        results are fetched. The method is passed the statement object and a
        set of "row processor" functions at query execution time; these
        processor functions when given a result row will return the individual
        attribute value, which can then be adapted into any kind of return data
        structure.

        The example below illustrates replacing the usual :class:`.Row`
        return structure with a straight Python dictionary::

            from sqlalchemy.orm import Bundle


            class DictBundle(Bundle):
                def create_row_processor(self, query, procs, labels):
                    "Override create_row_processor to return values as dictionaries"

                    def proc(row):
                        return dict(zip(labels, (proc(row) for proc in procs)))

                    return proc

        A result from the above :class:`_orm.Bundle` will return dictionary
        values::

            bn = DictBundle("mybundle", MyClass.data1, MyClass.data2)
            for row in session.execute(select(bn)).where(bn.c.data1 == "d1"):
                print(row.mybundle["data1"], row.mybundle["data2"])

        """  # noqa: E501
        keyed_tuple = result_tuple(labels, [() for l in labels])

        def proc(row: Row[Any]) -> Any:
            return keyed_tuple([proc(row) for proc in procs])

        return proc


def _orm_annotate(element: _SA, exclude: Optional[Any] = None) -> _SA:
    """Deep copy the given ClauseElement, annotating each element with the
    "_orm_adapt" flag.

    Elements within the exclude collection will be cloned but not annotated.

    """
    return sql_util._deep_annotate(element, {"_orm_adapt": True}, exclude)


def _orm_deannotate(element: _SA) -> _SA:
    """Remove annotations that link a column to a particular mapping.

    Note this doesn't affect "remote" and "foreign" annotations
    passed by the :func:`_orm.foreign` and :func:`_orm.remote`
    annotators.

    """

    return sql_util._deep_deannotate(
        element, values=("_orm_adapt", "parententity")
    )


def _orm_full_deannotate(element: _SA) -> _SA:
    return sql_util._deep_deannotate(element)


class _ORMJoin(expression.Join):
    """Extend Join to support ORM constructs as input."""

    __visit_name__ = expression.Join.__visit_name__

    inherit_cache = True

    def __init__(
        self,
        left: _FromClauseArgument,
        right: _FromClauseArgument,
        onclause: Optional[_OnClauseArgument] = None,
        isouter: bool = False,
        full: bool = False,
        _left_memo: Optional[Any] = None,
        _right_memo: Optional[Any] = None,
        _extra_criteria: Tuple[ColumnElement[bool], ...] = (),
    ):
        left_info = cast(
            "Union[FromClause, _InternalEntityType[Any]]",
            inspection.inspect(left),
        )

        right_info = cast(
            "Union[FromClause, _InternalEntityType[Any]]",
            inspection.inspect(right),
        )
        adapt_to = right_info.selectable

        # used by joined eager loader
        self._left_memo = _left_memo
        self._right_memo = _right_memo

        if isinstance(onclause, attributes.QueryableAttribute):
            if TYPE_CHECKING:
                assert isinstance(
                    onclause.comparator, RelationshipProperty.Comparator
                )
            on_selectable = onclause.comparator._source_selectable()
            prop = onclause.property
            _extra_criteria += onclause._extra_criteria
        elif isinstance(onclause, MapperProperty):
            # used internally by joined eager loader...possibly not ideal
            prop = onclause
            on_selectable = prop.parent.selectable
        else:
            prop = None
            on_selectable = None

        left_selectable = left_info.selectable
        if prop:
            adapt_from: Optional[FromClause]
            if sql_util.clause_is_present(on_selectable, left_selectable):
                adapt_from = on_selectable
            else:
                assert isinstance(left_selectable, FromClause)
                adapt_from = left_selectable

            (
                pj,
                sj,
                source,
                dest,
                secondary,
                target_adapter,
            ) = prop._create_joins(
                source_selectable=adapt_from,
                dest_selectable=adapt_to,
                source_polymorphic=True,
                of_type_entity=right_info,
                alias_secondary=True,
                extra_criteria=_extra_criteria,
            )

            if sj is not None:
                if isouter:
                    # note this is an inner join from secondary->right
                    right = sql.join(secondary, right, sj)
                    onclause = pj
                else:
                    left = sql.join(left, secondary, pj, isouter)
                    onclause = sj
            else:
                onclause = pj

            self._target_adapter = target_adapter

        # we don't use the normal coercions logic for _ORMJoin
        # (probably should), so do some gymnastics to get the entity.
        # logic here is for #8721, which was a major bug in 1.4
        # for almost two years, not reported/fixed until 1.4.43 (!)
        if is_selectable(left_info):
            parententity = left_selectable._annotations.get(
                "parententity", None
            )
        elif insp_is_mapper(left_info) or insp_is_aliased_class(left_info):
            parententity = left_info
        else:
            parententity = None

        if parententity is not None:
            self._annotations = self._annotations.union(
                {"parententity": parententity}
            )

        augment_onclause = bool(_extra_criteria) and not prop
        expression.Join.__init__(self, left, right, onclause, isouter, full)

        assert self.onclause is not None

        if augment_onclause:
            self.onclause &= sql.and_(*_extra_criteria)

        if (
            not prop
            and getattr(right_info, "mapper", None)
            and right_info.mapper.single  # type: ignore
        ):
            right_info = cast("_InternalEntityType[Any]", right_info)
            # if single inheritance target and we are using a manual
            # or implicit ON clause, augment it the same way we'd augment the
            # WHERE.
            single_crit = right_info.mapper._single_table_criterion
            if single_crit is not None:
                if insp_is_aliased_class(right_info):
                    single_crit = right_info._adapter.traverse(single_crit)
                self.onclause = self.onclause & single_crit

    def _splice_into_center(self, other):
        """Splice a join into the center.

        Given join(a, b) and join(b, c), return join(a, b).join(c)

        """
        leftmost = other
        while isinstance(leftmost, sql.Join):
            leftmost = leftmost.left

        assert self.right is leftmost

        left = _ORMJoin(
            self.left,
            other.left,
            self.onclause,
            isouter=self.isouter,
            _left_memo=self._left_memo,
            _right_memo=other._left_memo._path_registry,
        )

        return _ORMJoin(
            left,
            other.right,
            other.onclause,
            isouter=other.isouter,
            _right_memo=other._right_memo,
        )

    def join(
        self,
        right: _FromClauseArgument,
        onclause: Optional[_OnClauseArgument] = None,
        isouter: bool = False,
        full: bool = False,
    ) -> _ORMJoin:
        return _ORMJoin(self, right, onclause, full=full, isouter=isouter)

    def outerjoin(
        self,
        right: _FromClauseArgument,
        onclause: Optional[_OnClauseArgument] = None,
        full: bool = False,
    ) -> _ORMJoin:
        return _ORMJoin(self, right, onclause, isouter=True, full=full)


def with_parent(
    instance: object,
    prop: attributes.QueryableAttribute[Any],
    from_entity: Optional[_EntityType[Any]] = None,
) -> ColumnElement[bool]:
    """Create filtering criterion that relates this query's primary entity
    to the given related instance, using established
    :func:`_orm.relationship()`
    configuration.

    E.g.::

        stmt = select(Address).where(with_parent(some_user, User.addresses))

    The SQL rendered is the same as that rendered when a lazy loader
    would fire off from the given parent on that attribute, meaning
    that the appropriate state is taken from the parent object in
    Python without the need to render joins to the parent table
    in the rendered statement.

    The given property may also make use of :meth:`_orm.PropComparator.of_type`
    to indicate the left side of the criteria::


        a1 = aliased(Address)
        a2 = aliased(Address)
        stmt = select(a1, a2).where(with_parent(u1, User.addresses.of_type(a2)))

    The above use is equivalent to using the
    :func:`_orm.with_parent.from_entity` argument::

        a1 = aliased(Address)
        a2 = aliased(Address)
        stmt = select(a1, a2).where(
            with_parent(u1, User.addresses, from_entity=a2)
        )

    :param instance:
      An instance which has some :func:`_orm.relationship`.

    :param property:
      Class-bound attribute, which indicates
      what relationship from the instance should be used to reconcile the
      parent/child relationship.

    :param from_entity:
      Entity in which to consider as the left side.  This defaults to the
      "zero" entity of the :class:`_query.Query` itself.

      .. versionadded:: 1.2

    """  # noqa: E501
    prop_t: RelationshipProperty[Any]

    if isinstance(prop, str):
        raise sa_exc.ArgumentError(
            "with_parent() accepts class-bound mapped attributes, not strings"
        )
    elif isinstance(prop, attributes.QueryableAttribute):
        if prop._of_type:
            from_entity = prop._of_type
        mapper_property = prop.property
        if mapper_property is None or not prop_is_relationship(
            mapper_property
        ):
            raise sa_exc.ArgumentError(
                f"Expected relationship property for with_parent(), "
                f"got {mapper_property}"
            )
        prop_t = mapper_property
    else:
        prop_t = prop

    return prop_t._with_parent(instance, from_entity=from_entity)


def has_identity(object_: object) -> bool:
    """Return True if the given object has a database
    identity.

    This typically corresponds to the object being
    in either the persistent or detached state.

    .. seealso::

        :func:`.was_deleted`

    """
    state = attributes.instance_state(object_)
    return state.has_identity


def was_deleted(object_: object) -> bool:
    """Return True if the given object was deleted
    within a session flush.

    This is regardless of whether or not the object is
    persistent or detached.

    .. seealso::

        :attr:`.InstanceState.was_deleted`

    """

    state = attributes.instance_state(object_)
    return state.was_deleted


def _entity_corresponds_to(
    given: _InternalEntityType[Any], entity: _InternalEntityType[Any]
) -> bool:
    """determine if 'given' corresponds to 'entity', in terms
    of an entity passed to Query that would match the same entity
    being referred to elsewhere in the query.

    """
    if insp_is_aliased_class(entity):
        if insp_is_aliased_class(given):
            if entity._base_alias() is given._base_alias():
                return True
        return False
    elif insp_is_aliased_class(given):
        if given._use_mapper_path:
            return entity in given.with_polymorphic_mappers
        else:
            return entity is given

    assert insp_is_mapper(given)
    return entity.common_parent(given)


def _entity_corresponds_to_use_path_impl(
    given: _InternalEntityType[Any], entity: _InternalEntityType[Any]
) -> bool:
    """determine if 'given' corresponds to 'entity', in terms
    of a path of loader options where a mapped attribute is taken to
    be a member of a parent entity.

    e.g.::

        someoption(A).someoption(A.b)  # -> fn(A, A) -> True
        someoption(A).someoption(C.d)  # -> fn(A, C) -> False

        a1 = aliased(A)
        someoption(a1).someoption(A.b)  # -> fn(a1, A) -> False
        someoption(a1).someoption(a1.b)  # -> fn(a1, a1) -> True

        wp = with_polymorphic(A, [A1, A2])
        someoption(wp).someoption(A1.foo)  # -> fn(wp, A1) -> False
        someoption(wp).someoption(wp.A1.foo)  # -> fn(wp, wp.A1) -> True

    """
    if insp_is_aliased_class(given):
        return (
            insp_is_aliased_class(entity)
            and not entity._use_mapper_path
            and (given is entity or entity in given._with_polymorphic_entities)
        )
    elif not insp_is_aliased_class(entity):
        return given.isa(entity.mapper)
    else:
        return (
            entity._use_mapper_path
            and given in entity.with_polymorphic_mappers
        )


def _entity_isa(given: _InternalEntityType[Any], mapper: Mapper[Any]) -> bool:
    """determine if 'given' "is a" mapper, in terms of the given
    would load rows of type 'mapper'.

    """
    if given.is_aliased_class:
        return mapper in given.with_polymorphic_mappers or given.mapper.isa(
            mapper
        )
    elif given.with_polymorphic_mappers:
        return mapper in given.with_polymorphic_mappers or given.isa(mapper)
    else:
        return given.isa(mapper)


def _getitem(iterable_query: Query[Any], item: Any) -> Any:
    """calculate __getitem__ in terms of an iterable query object
    that also has a slice() method.

    """

    def _no_negative_indexes():
        raise IndexError(
            "negative indexes are not accepted by SQL "
            "index / slice operators"
        )

    if isinstance(item, slice):
        start, stop, step = util.decode_slice(item)

        if (
            isinstance(stop, int)
            and isinstance(start, int)
            and stop - start <= 0
        ):
            return []

        elif (isinstance(start, int) and start < 0) or (
            isinstance(stop, int) and stop < 0
        ):
            _no_negative_indexes()

        res = iterable_query.slice(start, stop)
        if step is not None:
            return list(res)[None : None : item.step]
        else:
            return list(res)
    else:
        if item == -1:
            _no_negative_indexes()
        else:
            return list(iterable_query[item : item + 1])[0]


def _is_mapped_annotation(
    raw_annotation: _AnnotationScanType,
    cls: Type[Any],
    originating_cls: Type[Any],
) -> bool:
    try:
        annotated = de_stringify_annotation(
            cls, raw_annotation, originating_cls.__module__
        )
    except NameError:
        # in most cases, at least within our own tests, we can raise
        # here, which is more accurate as it prevents us from returning
        # false negatives.  However, in the real world, try to avoid getting
        # involved with end-user annotations that have nothing to do with us.
        # see issue #8888 where we bypass using this function in the case
        # that we want to detect an unresolvable Mapped[] type.
        return False
    else:
        return is_origin_of_cls(annotated, _MappedAnnotationBase)


class _CleanupError(Exception):
    pass


def _cleanup_mapped_str_annotation(
    annotation: str, originating_module: str
) -> str:
    # fix up an annotation that comes in as the form:
    # 'Mapped[List[Address]]'  so that it instead looks like:
    # 'Mapped[List["Address"]]' , which will allow us to get
    # "Address" as a string

    # additionally, resolve symbols for these names since this is where
    # we'd have to do it

    inner: Optional[Match[str]]

    mm = re.match(r"^([^ \|]+?)\[(.+)\]$", annotation)

    if not mm:
        return annotation

    # ticket #8759.  Resolve the Mapped name to a real symbol.
    # originally this just checked the name.
    try:
        obj = eval_name_only(mm.group(1), originating_module)
    except NameError as ne:
        raise _CleanupError(
            f'For annotation "{annotation}", could not resolve '
            f'container type "{mm.group(1)}".  '
            "Please ensure this type is imported at the module level "
            "outside of TYPE_CHECKING blocks"
        ) from ne

    if obj is typing.ClassVar:
        real_symbol = "ClassVar"
    else:
        try:
            if issubclass(obj, _MappedAnnotationBase):
                real_symbol = obj.__name__
            else:
                return annotation
        except TypeError:
            # avoid isinstance(obj, type) check, just catch TypeError
            return annotation

    # note: if one of the codepaths above didn't define real_symbol and
    # then didn't return, real_symbol raises UnboundLocalError
    # which is actually a NameError, and the calling routines don't
    # notice this since they are catching NameError anyway.   Just in case
    # this is being modified in the future, something to be aware of.

    stack = []
    inner = mm
    while True:
        stack.append(real_symbol if mm is inner else inner.group(1))
        g2 = inner.group(2)
        inner = re.match(r"^([^ \|]+?)\[(.+)\]$", g2)
        if inner is None:
            stack.append(g2)
            break

    # stacks we want to rewrite, that is, quote the last entry which
    # we think is a relationship class name:
    #
    #   ['Mapped', 'List', 'Address']
    #   ['Mapped', 'A']
    #
    # stacks we dont want to rewrite, which are generally MappedColumn
    # use cases:
    #
    # ['Mapped', "'Optional[Dict[str, str]]'"]
    # ['Mapped', 'dict[str, str] | None']

    if (
        # avoid already quoted symbols such as
        # ['Mapped', "'Optional[Dict[str, str]]'"]
        not re.match(r"""^["'].*["']$""", stack[-1])
        # avoid further generics like Dict[] such as
        # ['Mapped', 'dict[str, str] | None'],
        # ['Mapped', 'list[int] | list[str]'],
        # ['Mapped', 'Union[list[int], list[str]]'],
        and not re.search(r"[\[\]]", stack[-1])
    ):
        stripchars = "\"' "
        stack[-1] = ", ".join(
            f'"{elem.strip(stripchars)}"' for elem in stack[-1].split(",")
        )

        annotation = "[".join(stack) + ("]" * (len(stack) - 1))

    return annotation


def _extract_mapped_subtype(
    raw_annotation: Optional[_AnnotationScanType],
    cls: type,
    originating_module: str,
    key: str,
    attr_cls: Type[Any],
    required: bool,
    is_dataclass_field: bool,
    expect_mapped: bool = True,
    raiseerr: bool = True,
) -> Optional[Tuple[Union[_AnnotationScanType, str], Optional[type]]]:
    """given an annotation, figure out if it's ``Mapped[something]`` and if
    so, return the ``something`` part.

    Includes error raise scenarios and other options.

    """

    if raw_annotation is None:
        if required:
            raise orm_exc.MappedAnnotationError(
                f"Python typing annotation is required for attribute "
                f'"{cls.__name__}.{key}" when primary argument(s) for '
                f'"{attr_cls.__name__}" construct are None or not present'
            )
        return None

    try:
        # destringify the "outside" of the annotation.  note we are not
        # adding include_generic so it will *not* dig into generic contents,
        # which will remain as ForwardRef or plain str under future annotations
        # mode.  The full destringify happens later when mapped_column goes
        # to do a full lookup in the registry type_annotations_map.
        annotated = de_stringify_annotation(
            cls,
            raw_annotation,
            originating_module,
            str_cleanup_fn=_cleanup_mapped_str_annotation,
        )
    except _CleanupError as ce:
        raise orm_exc.MappedAnnotationError(
            f"Could not interpret annotation {raw_annotation}.  "
            "Check that it uses names that are correctly imported at the "
            "module level. See chained stack trace for more hints."
        ) from ce
    except NameError as ne:
        if raiseerr and "Mapped[" in raw_annotation:  # type: ignore
            raise orm_exc.MappedAnnotationError(
                f"Could not interpret annotation {raw_annotation}.  "
                "Check that it uses names that are correctly imported at the "
                "module level. See chained stack trace for more hints."
            ) from ne

        annotated = raw_annotation  # type: ignore

    if is_dataclass_field:
        return annotated, None
    else:
        if not hasattr(annotated, "__origin__") or not is_origin_of_cls(
            annotated, _MappedAnnotationBase
        ):
            if expect_mapped:
                if not raiseerr:
                    return None

                origin = getattr(annotated, "__origin__", None)
                if origin is typing.ClassVar:
                    return None

                # check for other kind of ORM descriptor like AssociationProxy,
                # don't raise for that (issue #9957)
                elif isinstance(origin, type) and issubclass(
                    origin, ORMDescriptor
                ):
                    return None

                raise orm_exc.MappedAnnotationError(
                    f'Type annotation for "{cls.__name__}.{key}" '
                    "can't be correctly interpreted for "
                    "Annotated Declarative Table form.  ORM annotations "
                    "should normally make use of the ``Mapped[]`` generic "
                    "type, or other ORM-compatible generic type, as a "
                    "container for the actual type, which indicates the "
                    "intent that the attribute is mapped. "
                    "Class variables that are not intended to be mapped "
                    "by the ORM should use ClassVar[].  "
                    "To allow Annotated Declarative to disregard legacy "
                    "annotations which don't use Mapped[] to pass, set "
                    '"__allow_unmapped__ = True" on the class or a '
                    "superclass this class.",
                    code="zlpr",
                )

            else:
                return annotated, None

        if len(annotated.__args__) != 1:
            raise orm_exc.MappedAnnotationError(
                "Expected sub-type for Mapped[] annotation"
            )

        return (
            # fix dict/list/set args to be ForwardRef, see #11814
            fixup_container_fwd_refs(annotated.__args__[0]),
            annotated.__origin__,
        )


def _mapper_property_as_plain_name(prop: Type[Any]) -> str:
    if hasattr(prop, "_mapper_property_name"):
        name = prop._mapper_property_name()
    else:
        name = None
    return util.clsname_as_plain_name(prop, name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\result.py ===
# engine/result.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Define generic result set constructs."""

from __future__ import annotations

from enum import Enum
import functools
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .row import Row
from .row import RowMapping
from .. import exc
from .. import util
from ..sql.base import _generative
from ..sql.base import HasMemoized
from ..sql.base import InPlaceGenerative
from ..util import HasMemoized_ro_memoized_attribute
from ..util import NONE_SET
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Self

if typing.TYPE_CHECKING or not HAS_CYEXTENSION:
    from ._py_row import tuplegetter as tuplegetter
else:
    from sqlalchemy.cyextension.resultproxy import tuplegetter as tuplegetter

if typing.TYPE_CHECKING:
    from ..sql.elements import SQLCoreOperations
    from ..sql.type_api import _ResultProcessorType

_KeyType = Union[str, "SQLCoreOperations[Any]"]
_KeyIndexType = Union[_KeyType, int]

# is overridden in cursor using _CursorKeyMapRecType
_KeyMapRecType = Any

_KeyMapType = Mapping[_KeyType, _KeyMapRecType]


_RowData = Union[Row[Any], RowMapping, Any]
"""A generic form of "row" that accommodates for the different kinds of
"rows" that different result objects return, including row, row mapping, and
scalar values"""

_RawRowType = Tuple[Any, ...]
"""represents the kind of row we get from a DBAPI cursor"""

_R = TypeVar("_R", bound=_RowData)
_T = TypeVar("_T", bound=Any)
_TP = TypeVar("_TP", bound=Tuple[Any, ...])

_InterimRowType = Union[_R, _RawRowType]
"""a catchall "anything" kind of return type that can be applied
across all the result types

"""

_InterimSupportsScalarsRowType = Union[Row[Any], Any]

_ProcessorsType = Sequence[Optional["_ResultProcessorType[Any]"]]
_TupleGetterType = Callable[[Sequence[Any]], Sequence[Any]]
_UniqueFilterType = Callable[[Any], Any]
_UniqueFilterStateType = Tuple[Set[Any], Optional[_UniqueFilterType]]


class ResultMetaData:
    """Base for metadata about result rows."""

    __slots__ = ()

    _tuplefilter: Optional[_TupleGetterType] = None
    _translated_indexes: Optional[Sequence[int]] = None
    _unique_filters: Optional[Sequence[Callable[[Any], Any]]] = None
    _keymap: _KeyMapType
    _keys: Sequence[str]
    _processors: Optional[_ProcessorsType]
    _key_to_index: Mapping[_KeyType, int]

    @property
    def keys(self) -> RMKeyView:
        return RMKeyView(self)

    def _has_key(self, key: object) -> bool:
        raise NotImplementedError()

    def _for_freeze(self) -> ResultMetaData:
        raise NotImplementedError()

    @overload
    def _key_fallback(
        self, key: Any, err: Optional[Exception], raiseerr: Literal[True] = ...
    ) -> NoReturn: ...

    @overload
    def _key_fallback(
        self,
        key: Any,
        err: Optional[Exception],
        raiseerr: Literal[False] = ...,
    ) -> None: ...

    @overload
    def _key_fallback(
        self, key: Any, err: Optional[Exception], raiseerr: bool = ...
    ) -> Optional[NoReturn]: ...

    def _key_fallback(
        self, key: Any, err: Optional[Exception], raiseerr: bool = True
    ) -> Optional[NoReturn]:
        assert raiseerr
        raise KeyError(key) from err

    def _raise_for_ambiguous_column_name(
        self, rec: _KeyMapRecType
    ) -> NoReturn:
        raise NotImplementedError(
            "ambiguous column name logic is implemented for "
            "CursorResultMetaData"
        )

    def _index_for_key(
        self, key: _KeyIndexType, raiseerr: bool
    ) -> Optional[int]:
        raise NotImplementedError()

    def _indexes_for_keys(
        self, keys: Sequence[_KeyIndexType]
    ) -> Sequence[int]:
        raise NotImplementedError()

    def _metadata_for_keys(
        self, keys: Sequence[_KeyIndexType]
    ) -> Iterator[_KeyMapRecType]:
        raise NotImplementedError()

    def _reduce(self, keys: Sequence[_KeyIndexType]) -> ResultMetaData:
        raise NotImplementedError()

    def _getter(
        self, key: Any, raiseerr: bool = True
    ) -> Optional[Callable[[Row[Any]], Any]]:
        index = self._index_for_key(key, raiseerr)

        if index is not None:
            return operator.itemgetter(index)
        else:
            return None

    def _row_as_tuple_getter(
        self, keys: Sequence[_KeyIndexType]
    ) -> _TupleGetterType:
        indexes = self._indexes_for_keys(keys)
        return tuplegetter(*indexes)

    def _make_key_to_index(
        self, keymap: Mapping[_KeyType, Sequence[Any]], index: int
    ) -> Mapping[_KeyType, int]:
        return {
            key: rec[index]
            for key, rec in keymap.items()
            if rec[index] is not None
        }

    def _key_not_found(self, key: Any, attr_error: bool) -> NoReturn:
        if key in self._keymap:
            # the index must be none in this case
            self._raise_for_ambiguous_column_name(self._keymap[key])
        else:
            # unknown key
            if attr_error:
                try:
                    self._key_fallback(key, None)
                except KeyError as ke:
                    raise AttributeError(ke.args[0]) from ke
            else:
                self._key_fallback(key, None)

    @property
    def _effective_processors(self) -> Optional[_ProcessorsType]:
        if not self._processors or NONE_SET.issuperset(self._processors):
            return None
        else:
            return self._processors


class RMKeyView(typing.KeysView[Any]):
    __slots__ = ("_parent", "_keys")

    _parent: ResultMetaData
    _keys: Sequence[str]

    def __init__(self, parent: ResultMetaData):
        self._parent = parent
        self._keys = [k for k in parent._keys if k is not None]

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return "{0.__class__.__name__}({0._keys!r})".format(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, int):
            return False

        # note this also includes special key fallback behaviors
        # which also don't seem to be tested in test_resultset right now
        return self._parent._has_key(item)

    def __eq__(self, other: Any) -> bool:
        return list(other) == list(self)

    def __ne__(self, other: Any) -> bool:
        return list(other) != list(self)


class SimpleResultMetaData(ResultMetaData):
    """result metadata for in-memory collections."""

    __slots__ = (
        "_keys",
        "_keymap",
        "_processors",
        "_tuplefilter",
        "_translated_indexes",
        "_unique_filters",
        "_key_to_index",
    )

    _keys: Sequence[str]

    def __init__(
        self,
        keys: Sequence[str],
        extra: Optional[Sequence[Any]] = None,
        _processors: Optional[_ProcessorsType] = None,
        _tuplefilter: Optional[_TupleGetterType] = None,
        _translated_indexes: Optional[Sequence[int]] = None,
        _unique_filters: Optional[Sequence[Callable[[Any], Any]]] = None,
    ):
        self._keys = list(keys)
        self._tuplefilter = _tuplefilter
        self._translated_indexes = _translated_indexes
        self._unique_filters = _unique_filters
        if extra:
            recs_names = [
                (
                    (name,) + (extras if extras else ()),
                    (index, name, extras),
                )
                for index, (name, extras) in enumerate(zip(self._keys, extra))
            ]
        else:
            recs_names = [
                ((name,), (index, name, ()))
                for index, name in enumerate(self._keys)
            ]

        self._keymap = {key: rec for keys, rec in recs_names for key in keys}

        self._processors = _processors

        self._key_to_index = self._make_key_to_index(self._keymap, 0)

    def _has_key(self, key: object) -> bool:
        return key in self._keymap

    def _for_freeze(self) -> ResultMetaData:
        unique_filters = self._unique_filters
        if unique_filters and self._tuplefilter:
            unique_filters = self._tuplefilter(unique_filters)

        # TODO: are we freezing the result with or without uniqueness
        # applied?
        return SimpleResultMetaData(
            self._keys,
            extra=[self._keymap[key][2] for key in self._keys],
            _unique_filters=unique_filters,
        )

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "_keys": self._keys,
            "_translated_indexes": self._translated_indexes,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if state["_translated_indexes"]:
            _translated_indexes = state["_translated_indexes"]
            _tuplefilter = tuplegetter(*_translated_indexes)
        else:
            _translated_indexes = _tuplefilter = None
        self.__init__(  # type: ignore
            state["_keys"],
            _translated_indexes=_translated_indexes,
            _tuplefilter=_tuplefilter,
        )

    def _index_for_key(self, key: Any, raiseerr: bool = True) -> int:
        if int in key.__class__.__mro__:
            key = self._keys[key]
        try:
            rec = self._keymap[key]
        except KeyError as ke:
            rec = self._key_fallback(key, ke, raiseerr)

        return rec[0]  # type: ignore[no-any-return]

    def _indexes_for_keys(self, keys: Sequence[Any]) -> Sequence[int]:
        return [self._keymap[key][0] for key in keys]

    def _metadata_for_keys(
        self, keys: Sequence[Any]
    ) -> Iterator[_KeyMapRecType]:
        for key in keys:
            if int in key.__class__.__mro__:
                key = self._keys[key]

            try:
                rec = self._keymap[key]
            except KeyError as ke:
                rec = self._key_fallback(key, ke, True)

            yield rec

    def _reduce(self, keys: Sequence[Any]) -> ResultMetaData:
        try:
            metadata_for_keys = [
                self._keymap[
                    self._keys[key] if int in key.__class__.__mro__ else key
                ]
                for key in keys
            ]
        except KeyError as ke:
            self._key_fallback(ke.args[0], ke, True)

        indexes: Sequence[int]
        new_keys: Sequence[str]
        extra: Sequence[Any]
        indexes, new_keys, extra = zip(*metadata_for_keys)

        if self._translated_indexes:
            indexes = [self._translated_indexes[idx] for idx in indexes]

        tup = tuplegetter(*indexes)

        new_metadata = SimpleResultMetaData(
            new_keys,
            extra=extra,
            _tuplefilter=tup,
            _translated_indexes=indexes,
            _processors=self._processors,
            _unique_filters=self._unique_filters,
        )

        return new_metadata


def result_tuple(
    fields: Sequence[str], extra: Optional[Any] = None
) -> Callable[[Iterable[Any]], Row[Any]]:
    parent = SimpleResultMetaData(fields, extra)
    return functools.partial(
        Row, parent, parent._effective_processors, parent._key_to_index
    )


# a symbol that indicates to internal Result methods that
# "no row is returned".  We can't use None for those cases where a scalar
# filter is applied to rows.
class _NoRow(Enum):
    _NO_ROW = 0


_NO_ROW = _NoRow._NO_ROW


class ResultInternal(InPlaceGenerative, Generic[_R]):
    __slots__ = ()

    _real_result: Optional[Result[Any]] = None
    _generate_rows: bool = True
    _row_logging_fn: Optional[Callable[[Any], Any]]

    _unique_filter_state: Optional[_UniqueFilterStateType] = None
    _post_creational_filter: Optional[Callable[[Any], Any]] = None
    _is_cursor = False

    _metadata: ResultMetaData

    _source_supports_scalars: bool

    def _fetchiter_impl(self) -> Iterator[_InterimRowType[Row[Any]]]:
        raise NotImplementedError()

    def _fetchone_impl(
        self, hard_close: bool = False
    ) -> Optional[_InterimRowType[Row[Any]]]:
        raise NotImplementedError()

    def _fetchmany_impl(
        self, size: Optional[int] = None
    ) -> List[_InterimRowType[Row[Any]]]:
        raise NotImplementedError()

    def _fetchall_impl(self) -> List[_InterimRowType[Row[Any]]]:
        raise NotImplementedError()

    def _soft_close(self, hard: bool = False) -> None:
        raise NotImplementedError()

    @HasMemoized_ro_memoized_attribute
    def _row_getter(self) -> Optional[Callable[..., _R]]:
        real_result: Result[Any] = (
            self._real_result
            if self._real_result
            else cast("Result[Any]", self)
        )

        if real_result._source_supports_scalars:
            if not self._generate_rows:
                return None
            else:
                _proc = Row

                def process_row(
                    metadata: ResultMetaData,
                    processors: Optional[_ProcessorsType],
                    key_to_index: Mapping[_KeyType, int],
                    scalar_obj: Any,
                ) -> Row[Any]:
                    return _proc(
                        metadata, processors, key_to_index, (scalar_obj,)
                    )

        else:
            process_row = Row  # type: ignore

        metadata = self._metadata

        key_to_index = metadata._key_to_index
        processors = metadata._effective_processors
        tf = metadata._tuplefilter

        if tf and not real_result._source_supports_scalars:
            if processors:
                processors = tf(processors)

            _make_row_orig: Callable[..., _R] = functools.partial(  # type: ignore  # noqa E501
                process_row, metadata, processors, key_to_index
            )

            fixed_tf = tf

            def make_row(row: _InterimRowType[Row[Any]]) -> _R:
                return _make_row_orig(fixed_tf(row))

        else:
            make_row = functools.partial(  # type: ignore
                process_row, metadata, processors, key_to_index
            )

        if real_result._row_logging_fn:
            _log_row = real_result._row_logging_fn
            _make_row = make_row

            def make_row(row: _InterimRowType[Row[Any]]) -> _R:
                return _log_row(_make_row(row))  # type: ignore

        return make_row

    @HasMemoized_ro_memoized_attribute
    def _iterator_getter(self) -> Callable[..., Iterator[_R]]:
        make_row = self._row_getter

        post_creational_filter = self._post_creational_filter

        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy

            def iterrows(self: Result[Any]) -> Iterator[_R]:
                for raw_row in self._fetchiter_impl():
                    obj: _InterimRowType[Any] = (
                        make_row(raw_row) if make_row else raw_row
                    )
                    hashed = strategy(obj) if strategy else obj
                    if hashed in uniques:
                        continue
                    uniques.add(hashed)
                    if post_creational_filter:
                        obj = post_creational_filter(obj)
                    yield obj  # type: ignore

        else:

            def iterrows(self: Result[Any]) -> Iterator[_R]:
                for raw_row in self._fetchiter_impl():
                    row: _InterimRowType[Any] = (
                        make_row(raw_row) if make_row else raw_row
                    )
                    if post_creational_filter:
                        row = post_creational_filter(row)
                    yield row  # type: ignore

        return iterrows

    def _raw_all_rows(self) -> List[_R]:
        make_row = self._row_getter
        assert make_row is not None
        rows = self._fetchall_impl()
        return [make_row(row) for row in rows]

    def _allrows(self) -> List[_R]:
        post_creational_filter = self._post_creational_filter

        make_row = self._row_getter

        rows = self._fetchall_impl()
        made_rows: List[_InterimRowType[_R]]
        if make_row:
            made_rows = [make_row(row) for row in rows]
        else:
            made_rows = rows  # type: ignore

        interim_rows: List[_R]

        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy

            interim_rows = [
                made_row  # type: ignore
                for made_row, sig_row in [
                    (
                        made_row,
                        strategy(made_row) if strategy else made_row,
                    )
                    for made_row in made_rows
                ]
                if sig_row not in uniques and not uniques.add(sig_row)  # type: ignore # noqa: E501
            ]
        else:
            interim_rows = made_rows  # type: ignore

        if post_creational_filter:
            interim_rows = [
                post_creational_filter(row) for row in interim_rows
            ]
        return interim_rows

    @HasMemoized_ro_memoized_attribute
    def _onerow_getter(
        self,
    ) -> Callable[..., Union[Literal[_NoRow._NO_ROW], _R]]:
        make_row = self._row_getter

        post_creational_filter = self._post_creational_filter

        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy

            def onerow(self: Result[Any]) -> Union[_NoRow, _R]:
                _onerow = self._fetchone_impl
                while True:
                    row = _onerow()
                    if row is None:
                        return _NO_ROW
                    else:
                        obj: _InterimRowType[Any] = (
                            make_row(row) if make_row else row
                        )
                        hashed = strategy(obj) if strategy else obj
                        if hashed in uniques:
                            continue
                        else:
                            uniques.add(hashed)
                        if post_creational_filter:
                            obj = post_creational_filter(obj)
                        return obj  # type: ignore

        else:

            def onerow(self: Result[Any]) -> Union[_NoRow, _R]:
                row = self._fetchone_impl()
                if row is None:
                    return _NO_ROW
                else:
                    interim_row: _InterimRowType[Any] = (
                        make_row(row) if make_row else row
                    )
                    if post_creational_filter:
                        interim_row = post_creational_filter(interim_row)
                    return interim_row  # type: ignore

        return onerow

    @HasMemoized_ro_memoized_attribute
    def _manyrow_getter(self) -> Callable[..., List[_R]]:
        make_row = self._row_getter

        post_creational_filter = self._post_creational_filter

        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy

            def filterrows(
                make_row: Optional[Callable[..., _R]],
                rows: List[Any],
                strategy: Optional[Callable[[List[Any]], Any]],
                uniques: Set[Any],
            ) -> List[_R]:
                if make_row:
                    rows = [make_row(row) for row in rows]

                if strategy:
                    made_rows = (
                        (made_row, strategy(made_row)) for made_row in rows
                    )
                else:
                    made_rows = ((made_row, made_row) for made_row in rows)
                return [
                    made_row
                    for made_row, sig_row in made_rows
                    if sig_row not in uniques and not uniques.add(sig_row)  # type: ignore  # noqa: E501
                ]

            def manyrows(
                self: ResultInternal[_R], num: Optional[int]
            ) -> List[_R]:
                collect: List[_R] = []

                _manyrows = self._fetchmany_impl

                if num is None:
                    # if None is passed, we don't know the default
                    # manyrows number, DBAPI has this as cursor.arraysize
                    # different DBAPIs / fetch strategies may be different.
                    # do a fetch to find what the number is.  if there are
                    # only fewer rows left, then it doesn't matter.
                    real_result = (
                        self._real_result
                        if self._real_result
                        else cast("Result[Any]", self)
                    )
                    if real_result._yield_per:
                        num_required = num = real_result._yield_per
                    else:
                        rows = _manyrows(num)
                        num = len(rows)
                        assert make_row is not None
                        collect.extend(
                            filterrows(make_row, rows, strategy, uniques)
                        )
                        num_required = num - len(collect)
                else:
                    num_required = num

                assert num is not None

                while num_required:
                    rows = _manyrows(num_required)
                    if not rows:
                        break

                    collect.extend(
                        filterrows(make_row, rows, strategy, uniques)
                    )
                    num_required = num - len(collect)

                if post_creational_filter:
                    collect = [post_creational_filter(row) for row in collect]
                return collect

        else:

            def manyrows(
                self: ResultInternal[_R], num: Optional[int]
            ) -> List[_R]:
                if num is None:
                    real_result = (
                        self._real_result
                        if self._real_result
                        else cast("Result[Any]", self)
                    )
                    num = real_result._yield_per

                rows: List[_InterimRowType[Any]] = self._fetchmany_impl(num)
                if make_row:
                    rows = [make_row(row) for row in rows]
                if post_creational_filter:
                    rows = [post_creational_filter(row) for row in rows]
                return rows  # type: ignore

        return manyrows

    @overload
    def _only_one_row(
        self: ResultInternal[Row[Any]],
        raise_for_second_row: bool,
        raise_for_none: bool,
        scalar: Literal[True],
    ) -> Any: ...

    @overload
    def _only_one_row(
        self,
        raise_for_second_row: bool,
        raise_for_none: Literal[True],
        scalar: bool,
    ) -> _R: ...

    @overload
    def _only_one_row(
        self,
        raise_for_second_row: bool,
        raise_for_none: bool,
        scalar: bool,
    ) -> Optional[_R]: ...

    def _only_one_row(
        self,
        raise_for_second_row: bool,
        raise_for_none: bool,
        scalar: bool,
    ) -> Optional[_R]:
        onerow = self._fetchone_impl

        row: Optional[_InterimRowType[Any]] = onerow(hard_close=True)
        if row is None:
            if raise_for_none:
                raise exc.NoResultFound(
                    "No row was found when one was required"
                )
            else:
                return None

        if scalar and self._source_supports_scalars:
            self._generate_rows = False
            make_row = None
        else:
            make_row = self._row_getter

        try:
            row = make_row(row) if make_row else row
        except:
            self._soft_close(hard=True)
            raise

        if raise_for_second_row:
            if self._unique_filter_state:
                # for no second row but uniqueness, need to essentially
                # consume the entire result :(
                uniques, strategy = self._unique_strategy

                existing_row_hash = strategy(row) if strategy else row

                while True:
                    next_row: Any = onerow(hard_close=True)
                    if next_row is None:
                        next_row = _NO_ROW
                        break

                    try:
                        next_row = make_row(next_row) if make_row else next_row

                        if strategy:
                            assert next_row is not _NO_ROW
                            if existing_row_hash == strategy(next_row):
                                continue
                        elif row == next_row:
                            continue
                        # here, we have a row and it's different
                        break
                    except:
                        self._soft_close(hard=True)
                        raise
            else:
                next_row = onerow(hard_close=True)
                if next_row is None:
                    next_row = _NO_ROW

            if next_row is not _NO_ROW:
                self._soft_close(hard=True)
                raise exc.MultipleResultsFound(
                    "Multiple rows were found when exactly one was required"
                    if raise_for_none
                    else "Multiple rows were found when one or none "
                    "was required"
                )
        else:
            # if we checked for second row then that would have
            # closed us :)
            self._soft_close(hard=True)

        if not scalar:
            post_creational_filter = self._post_creational_filter
            if post_creational_filter:
                row = post_creational_filter(row)

        if scalar and make_row:
            return row[0]  # type: ignore
        else:
            return row  # type: ignore

    def _iter_impl(self) -> Iterator[_R]:
        return self._iterator_getter(self)

    def _next_impl(self) -> _R:
        row = self._onerow_getter(self)
        if row is _NO_ROW:
            raise StopIteration()
        else:
            return row

    @_generative
    def _column_slices(self, indexes: Sequence[_KeyIndexType]) -> Self:
        real_result = (
            self._real_result
            if self._real_result
            else cast("Result[Any]", self)
        )

        if not real_result._source_supports_scalars or len(indexes) != 1:
            self._metadata = self._metadata._reduce(indexes)

        assert self._generate_rows

        return self

    @HasMemoized.memoized_attribute
    def _unique_strategy(self) -> _UniqueFilterStateType:
        assert self._unique_filter_state is not None
        uniques, strategy = self._unique_filter_state

        real_result = (
            self._real_result
            if self._real_result is not None
            else cast("Result[Any]", self)
        )

        if not strategy and self._metadata._unique_filters:
            if (
                real_result._source_supports_scalars
                and not self._generate_rows
            ):
                strategy = self._metadata._unique_filters[0]
            else:
                filters = self._metadata._unique_filters
                if self._metadata._tuplefilter:
                    filters = self._metadata._tuplefilter(filters)

                strategy = operator.methodcaller("_filter_on_values", filters)
        return uniques, strategy


class _WithKeys:
    __slots__ = ()

    _metadata: ResultMetaData

    # used mainly to share documentation on the keys method.
    def keys(self) -> RMKeyView:
        """Return an iterable view which yields the string keys that would
        be represented by each :class:`_engine.Row`.

        The keys can represent the labels of the columns returned by a core
        statement or the names of the orm classes returned by an orm
        execution.

        The view also can be tested for key containment using the Python
        ``in`` operator, which will test both for the string keys represented
        in the view, as well as for alternate keys such as column objects.

        .. versionchanged:: 1.4 a key view object is returned rather than a
           plain list.


        """
        return self._metadata.keys


class Result(_WithKeys, ResultInternal[Row[_TP]]):
    """Represent a set of database results.

    .. versionadded:: 1.4  The :class:`_engine.Result` object provides a
       completely updated usage model and calling facade for SQLAlchemy
       Core and SQLAlchemy ORM.   In Core, it forms the basis of the
       :class:`_engine.CursorResult` object which replaces the previous
       :class:`_engine.ResultProxy` interface.   When using the ORM, a
       higher level object called :class:`_engine.ChunkedIteratorResult`
       is normally used.

    .. note:: In SQLAlchemy 1.4 and above, this object is
       used for ORM results returned by :meth:`_orm.Session.execute`, which can
       yield instances of ORM mapped objects either individually or within
       tuple-like rows. Note that the :class:`_engine.Result` object does not
       deduplicate instances or rows automatically as is the case with the
       legacy :class:`_orm.Query` object. For in-Python de-duplication of
       instances or rows, use the :meth:`_engine.Result.unique` modifier
       method.

    .. seealso::

        :ref:`tutorial_fetching_rows` - in the :doc:`/tutorial/index`

    """

    __slots__ = ("_metadata", "__dict__")

    _row_logging_fn: Optional[Callable[[Row[Any]], Row[Any]]] = None

    _source_supports_scalars: bool = False

    _yield_per: Optional[int] = None

    _attributes: util.immutabledict[Any, Any] = util.immutabledict()

    def __init__(self, cursor_metadata: ResultMetaData):
        self._metadata = cursor_metadata

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """close this :class:`_engine.Result`.

        The behavior of this method is implementation specific, and is
        not implemented by default.    The method should generally end
        the resources in use by the result object and also cause any
        subsequent iteration or row fetching to raise
        :class:`.ResourceClosedError`.

        .. versionadded:: 1.4.27 - ``.close()`` was previously not generally
           available for all :class:`_engine.Result` classes, instead only
           being available on the :class:`_engine.CursorResult` returned for
           Core statement executions. As most other result objects, namely the
           ones used by the ORM, are proxying a :class:`_engine.CursorResult`
           in any case, this allows the underlying cursor result to be closed
           from the outside facade for the case when the ORM query is using
           the ``yield_per`` execution option where it does not immediately
           exhaust and autoclose the database cursor.

        """
        self._soft_close(hard=True)

    @property
    def _soft_closed(self) -> bool:
        raise NotImplementedError()

    @property
    def closed(self) -> bool:
        """return ``True`` if this :class:`_engine.Result` reports .closed

        .. versionadded:: 1.4.43

        """
        raise NotImplementedError()

    @_generative
    def yield_per(self, num: int) -> Self:
        """Configure the row-fetching strategy to fetch ``num`` rows at a time.

        This impacts the underlying behavior of the result when iterating over
        the result object, or otherwise making use of  methods such as
        :meth:`_engine.Result.fetchone` that return one row at a time.   Data
        from the underlying cursor or other data source will be buffered up to
        this many rows in memory, and the buffered collection will then be
        yielded out one row at a time or as many rows are requested. Each time
        the buffer clears, it will be refreshed to this many rows or as many
        rows remain if fewer remain.

        The :meth:`_engine.Result.yield_per` method is generally used in
        conjunction with the
        :paramref:`_engine.Connection.execution_options.stream_results`
        execution option, which will allow the database dialect in use to make
        use of a server side cursor, if the DBAPI supports a specific "server
        side cursor" mode separate from its default mode of operation.

        .. tip::

            Consider using the
            :paramref:`_engine.Connection.execution_options.yield_per`
            execution option, which will simultaneously set
            :paramref:`_engine.Connection.execution_options.stream_results`
            to ensure the use of server side cursors, as well as automatically
            invoke the :meth:`_engine.Result.yield_per` method to establish
            a fixed row buffer size at once.

            The :paramref:`_engine.Connection.execution_options.yield_per`
            execution option is available for ORM operations, with
            :class:`_orm.Session`-oriented use described at
            :ref:`orm_queryguide_yield_per`. The Core-only version which works
            with :class:`_engine.Connection` is new as of SQLAlchemy 1.4.40.

        .. versionadded:: 1.4

        :param num: number of rows to fetch each time the buffer is refilled.
         If set to a value below 1, fetches all rows for the next buffer.

        .. seealso::

            :ref:`engine_stream_results` - describes Core behavior for
            :meth:`_engine.Result.yield_per`

            :ref:`orm_queryguide_yield_per` - in the :ref:`queryguide_toplevel`

        """
        self._yield_per = num
        return self

    @_generative
    def unique(self, strategy: Optional[_UniqueFilterType] = None) -> Self:
        """Apply unique filtering to the objects returned by this
        :class:`_engine.Result`.

        When this filter is applied with no arguments, the rows or objects
        returned will filtered such that each row is returned uniquely. The
        algorithm used to determine this uniqueness is by default the Python
        hashing identity of the whole tuple.   In some cases a specialized
        per-entity hashing scheme may be used, such as when using the ORM, a
        scheme is applied which  works against the primary key identity of
        returned objects.

        The unique filter is applied **after all other filters**, which means
        if the columns returned have been refined using a method such as the
        :meth:`_engine.Result.columns` or :meth:`_engine.Result.scalars`
        method, the uniquing is applied to **only the column or columns
        returned**.   This occurs regardless of the order in which these
        methods have been called upon the :class:`_engine.Result` object.

        The unique filter also changes the calculus used for methods like
        :meth:`_engine.Result.fetchmany` and :meth:`_engine.Result.partitions`.
        When using :meth:`_engine.Result.unique`, these methods will continue
        to yield the number of rows or objects requested, after uniquing
        has been applied.  However, this necessarily impacts the buffering
        behavior of the underlying cursor or datasource, such that multiple
        underlying calls to ``cursor.fetchmany()`` may be necessary in order
        to accumulate enough objects in order to provide a unique collection
        of the requested size.

        :param strategy: a callable that will be applied to rows or objects
         being iterated, which should return an object that represents the
         unique value of the row.   A Python ``set()`` is used to store
         these identities.   If not passed, a default uniqueness strategy
         is used which may have been assembled by the source of this
         :class:`_engine.Result` object.

        """
        self._unique_filter_state = (set(), strategy)
        return self

    def columns(self, *col_expressions: _KeyIndexType) -> Self:
        r"""Establish the columns that should be returned in each row.

        This method may be used to limit the columns returned as well
        as to reorder them.   The given list of expressions are normally
        a series of integers or string key names.   They may also be
        appropriate :class:`.ColumnElement` objects which correspond to
        a given statement construct.

        .. versionchanged:: 2.0  Due to a bug in 1.4, the
           :meth:`_engine.Result.columns` method had an incorrect behavior
           where calling upon the method with just one index would cause the
           :class:`_engine.Result` object to yield scalar values rather than
           :class:`_engine.Row` objects.   In version 2.0, this behavior
           has been corrected such that calling upon
           :meth:`_engine.Result.columns` with a single index will
           produce a :class:`_engine.Result` object that continues
           to yield :class:`_engine.Row` objects, which include
           only a single column.

        E.g.::

            statement = select(table.c.x, table.c.y, table.c.z)
            result = connection.execute(statement)

            for z, y in result.columns("z", "y"):
                ...

        Example of using the column objects from the statement itself::

            for z, y in result.columns(
                statement.selected_columns.c.z, statement.selected_columns.c.y
            ):
                ...

        .. versionadded:: 1.4

        :param \*col_expressions: indicates columns to be returned.  Elements
         may be integer row indexes, string column names, or appropriate
         :class:`.ColumnElement` objects corresponding to a select construct.

        :return: this :class:`_engine.Result` object with the modifications
         given.

        """
        return self._column_slices(col_expressions)

    @overload
    def scalars(self: Result[Tuple[_T]]) -> ScalarResult[_T]: ...

    @overload
    def scalars(
        self: Result[Tuple[_T]], index: Literal[0]
    ) -> ScalarResult[_T]: ...

    @overload
    def scalars(self, index: _KeyIndexType = 0) -> ScalarResult[Any]: ...

    def scalars(self, index: _KeyIndexType = 0) -> ScalarResult[Any]:
        """Return a :class:`_engine.ScalarResult` filtering object which
        will return single elements rather than :class:`_row.Row` objects.

        E.g.::

            >>> result = conn.execute(text("select int_id from table"))
            >>> result.scalars().all()
            [1, 2, 3]

        When results are fetched from the :class:`_engine.ScalarResult`
        filtering object, the single column-row that would be returned by the
        :class:`_engine.Result` is instead returned as the column's value.

        .. versionadded:: 1.4

        :param index: integer or row key indicating the column to be fetched
         from each row, defaults to ``0`` indicating the first column.

        :return: a new :class:`_engine.ScalarResult` filtering object referring
         to this :class:`_engine.Result` object.

        """
        return ScalarResult(self, index)

    def _getter(
        self, key: _KeyIndexType, raiseerr: bool = True
    ) -> Optional[Callable[[Row[Any]], Any]]:
        """return a callable that will retrieve the given key from a
        :class:`_engine.Row`.

        """
        if self._source_supports_scalars:
            raise NotImplementedError(
                "can't use this function in 'only scalars' mode"
            )
        return self._metadata._getter(key, raiseerr)

    def _tuple_getter(self, keys: Sequence[_KeyIndexType]) -> _TupleGetterType:
        """return a callable that will retrieve the given keys from a
        :class:`_engine.Row`.

        """
        if self._source_supports_scalars:
            raise NotImplementedError(
                "can't use this function in 'only scalars' mode"
            )
        return self._metadata._row_as_tuple_getter(keys)

    def mappings(self) -> MappingResult:
        """Apply a mappings filter to returned rows, returning an instance of
        :class:`_engine.MappingResult`.

        When this filter is applied, fetching rows will return
        :class:`_engine.RowMapping` objects instead of :class:`_engine.Row`
        objects.

        .. versionadded:: 1.4

        :return: a new :class:`_engine.MappingResult` filtering object
         referring to this :class:`_engine.Result` object.

        """

        return MappingResult(self)

    @property
    def t(self) -> TupleResult[_TP]:
        """Apply a "typed tuple" typing filter to returned rows.

        The :attr:`_engine.Result.t` attribute is a synonym for
        calling the :meth:`_engine.Result.tuples` method.

        .. versionadded:: 2.0

        """
        return self  # type: ignore

    def tuples(self) -> TupleResult[_TP]:
        """Apply a "typed tuple" typing filter to returned rows.

        This method returns the same :class:`_engine.Result` object
        at runtime,
        however annotates as returning a :class:`_engine.TupleResult` object
        that will indicate to :pep:`484` typing tools that plain typed
        ``Tuple`` instances are returned rather than rows.  This allows
        tuple unpacking and ``__getitem__`` access of :class:`_engine.Row`
        objects to by typed, for those cases where the statement invoked
        itself included typing information.

        .. versionadded:: 2.0

        :return: the :class:`_engine.TupleResult` type at typing time.

        .. seealso::

            :attr:`_engine.Result.t` - shorter synonym

            :attr:`_engine.Row._t` - :class:`_engine.Row` version

        """

        return self  # type: ignore

    def _raw_row_iterator(self) -> Iterator[_RowData]:
        """Return a safe iterator that yields raw row data.

        This is used by the :meth:`_engine.Result.merge` method
        to merge multiple compatible results together.

        """
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Row[_TP]]:
        return self._iter_impl()

    def __next__(self) -> Row[_TP]:
        return self._next_impl()

    def partitions(
        self, size: Optional[int] = None
    ) -> Iterator[Sequence[Row[_TP]]]:
        """Iterate through sub-lists of rows of the size given.

        Each list will be of the size given, excluding the last list to
        be yielded, which may have a small number of rows.  No empty
        lists will be yielded.

        The result object is automatically closed when the iterator
        is fully consumed.

        Note that the backend driver will usually buffer the entire result
        ahead of time unless the
        :paramref:`.Connection.execution_options.stream_results` execution
        option is used indicating that the driver should not pre-buffer
        results, if possible.   Not all drivers support this option and
        the option is silently ignored for those who do not.

        When using the ORM, the :meth:`_engine.Result.partitions` method
        is typically more effective from a memory perspective when it is
        combined with use of the
        :ref:`yield_per execution option <orm_queryguide_yield_per>`,
        which instructs both the DBAPI driver to use server side cursors,
        if available, as well as instructs the ORM loading internals to only
        build a certain amount of ORM objects from a result at a time before
        yielding them out.

        .. versionadded:: 1.4

        :param size: indicate the maximum number of rows to be present
         in each list yielded.  If None, makes use of the value set by
         the :meth:`_engine.Result.yield_per`, method, if it were called,
         or the :paramref:`_engine.Connection.execution_options.yield_per`
         execution option, which is equivalent in this regard.  If
         yield_per weren't set, it makes use of the
         :meth:`_engine.Result.fetchmany` default, which may be backend
         specific and not well defined.

        :return: iterator of lists

        .. seealso::

            :ref:`engine_stream_results`

            :ref:`orm_queryguide_yield_per` - in the :ref:`queryguide_toplevel`

        """

        getter = self._manyrow_getter

        while True:
            partition = getter(self, size)
            if partition:
                yield partition
            else:
                break

    def fetchall(self) -> Sequence[Row[_TP]]:
        """A synonym for the :meth:`_engine.Result.all` method."""

        return self._allrows()

    def fetchone(self) -> Optional[Row[_TP]]:
        """Fetch one row.

        When all rows are exhausted, returns None.

        This method is provided for backwards compatibility with
        SQLAlchemy 1.x.x.

        To fetch the first row of a result only, use the
        :meth:`_engine.Result.first` method.  To iterate through all
        rows, iterate the :class:`_engine.Result` object directly.

        :return: a :class:`_engine.Row` object if no filters are applied,
         or ``None`` if no rows remain.

        """
        row = self._onerow_getter(self)
        if row is _NO_ROW:
            return None
        else:
            return row

    def fetchmany(self, size: Optional[int] = None) -> Sequence[Row[_TP]]:
        """Fetch many rows.

        When all rows are exhausted, returns an empty sequence.

        This method is provided for backwards compatibility with
        SQLAlchemy 1.x.x.

        To fetch rows in groups, use the :meth:`_engine.Result.partitions`
        method.

        :return: a sequence of :class:`_engine.Row` objects.

        .. seealso::

            :meth:`_engine.Result.partitions`

        """

        return self._manyrow_getter(self, size)

    def all(self) -> Sequence[Row[_TP]]:
        """Return all rows in a sequence.

        Closes the result set after invocation.   Subsequent invocations
        will return an empty sequence.

        .. versionadded:: 1.4

        :return: a sequence of :class:`_engine.Row` objects.

        .. seealso::

            :ref:`engine_stream_results` - How to stream a large result set
            without loading it completely in python.

        """

        return self._allrows()

    def first(self) -> Optional[Row[_TP]]:
        """Fetch the first row or ``None`` if no row is present.

        Closes the result set and discards remaining rows.

        .. note::  This method returns one **row**, e.g. tuple, by default.
           To return exactly one single scalar value, that is, the first
           column of the first row, use the
           :meth:`_engine.Result.scalar` method,
           or combine :meth:`_engine.Result.scalars` and
           :meth:`_engine.Result.first`.

           Additionally, in contrast to the behavior of the legacy  ORM
           :meth:`_orm.Query.first` method, **no limit is applied** to the
           SQL query which was invoked to produce this
           :class:`_engine.Result`;
           for a DBAPI driver that buffers results in memory before yielding
           rows, all rows will be sent to the Python process and all but
           the first row will be discarded.

           .. seealso::

                :ref:`migration_20_unify_select`

        :return: a :class:`_engine.Row` object, or None
         if no rows remain.

        .. seealso::

            :meth:`_engine.Result.scalar`

            :meth:`_engine.Result.one`

        """

        return self._only_one_row(
            raise_for_second_row=False, raise_for_none=False, scalar=False
        )

    def one_or_none(self) -> Optional[Row[_TP]]:
        """Return at most one result or raise an exception.

        Returns ``None`` if the result has no rows.
        Raises :class:`.MultipleResultsFound`
        if multiple rows are returned.

        .. versionadded:: 1.4

        :return: The first :class:`_engine.Row` or ``None`` if no row
         is available.

        :raises: :class:`.MultipleResultsFound`

        .. seealso::

            :meth:`_engine.Result.first`

            :meth:`_engine.Result.one`

        """
        return self._only_one_row(
            raise_for_second_row=True, raise_for_none=False, scalar=False
        )

    @overload
    def scalar_one(self: Result[Tuple[_T]]) -> _T: ...

    @overload
    def scalar_one(self) -> Any: ...

    def scalar_one(self) -> Any:
        """Return exactly one scalar result or raise an exception.

        This is equivalent to calling :meth:`_engine.Result.scalars` and
        then :meth:`_engine.ScalarResult.one`.

        .. seealso::

            :meth:`_engine.ScalarResult.one`

            :meth:`_engine.Result.scalars`

        """
        return self._only_one_row(
            raise_for_second_row=True, raise_for_none=True, scalar=True
        )

    @overload
    def scalar_one_or_none(self: Result[Tuple[_T]]) -> Optional[_T]: ...

    @overload
    def scalar_one_or_none(self) -> Optional[Any]: ...

    def scalar_one_or_none(self) -> Optional[Any]:
        """Return exactly one scalar result or ``None``.

        This is equivalent to calling :meth:`_engine.Result.scalars` and
        then :meth:`_engine.ScalarResult.one_or_none`.

        .. seealso::

            :meth:`_engine.ScalarResult.one_or_none`

            :meth:`_engine.Result.scalars`

        """
        return self._only_one_row(
            raise_for_second_row=True, raise_for_none=False, scalar=True
        )

    def one(self) -> Row[_TP]:
        """Return exactly one row or raise an exception.

        Raises :class:`_exc.NoResultFound` if the result returns no
        rows, or :class:`_exc.MultipleResultsFound` if multiple rows
        would be returned.

        .. note::  This method returns one **row**, e.g. tuple, by default.
           To return exactly one single scalar value, that is, the first
           column of the first row, use the
           :meth:`_engine.Result.scalar_one` method, or combine
           :meth:`_engine.Result.scalars` and
           :meth:`_engine.Result.one`.

        .. versionadded:: 1.4

        :return: The first :class:`_engine.Row`.

        :raises: :class:`.MultipleResultsFound`, :class:`.NoResultFound`

        .. seealso::

            :meth:`_engine.Result.first`

            :meth:`_engine.Result.one_or_none`

            :meth:`_engine.Result.scalar_one`

        """
        return self._only_one_row(
            raise_for_second_row=True, raise_for_none=True, scalar=False
        )

    @overload
    def scalar(self: Result[Tuple[_T]]) -> Optional[_T]: ...

    @overload
    def scalar(self) -> Any: ...

    def scalar(self) -> Any:
        """Fetch the first column of the first row, and close the result set.

        Returns ``None`` if there are no rows to fetch.

        No validation is performed to test if additional rows remain.

        After calling this method, the object is fully closed,
        e.g. the :meth:`_engine.CursorResult.close`
        method will have been called.

        :return: a Python scalar value, or ``None`` if no rows remain.

        """
        return self._only_one_row(
            raise_for_second_row=False, raise_for_none=False, scalar=True
        )

    def freeze(self) -> FrozenResult[_TP]:
        """Return a callable object that will produce copies of this
        :class:`_engine.Result` when invoked.

        The callable object returned is an instance of
        :class:`_engine.FrozenResult`.

        This is used for result set caching.  The method must be called
        on the result when it has been unconsumed, and calling the method
        will consume the result fully.   When the :class:`_engine.FrozenResult`
        is retrieved from a cache, it can be called any number of times where
        it will produce a new :class:`_engine.Result` object each time
        against its stored set of rows.

        .. seealso::

            :ref:`do_orm_execute_re_executing` - example usage within the
            ORM to implement a result-set cache.

        """

        return FrozenResult(self)

    def merge(self, *others: Result[Any]) -> MergedResult[_TP]:
        """Merge this :class:`_engine.Result` with other compatible result
        objects.

        The object returned is an instance of :class:`_engine.MergedResult`,
        which will be composed of iterators from the given result
        objects.

        The new result will use the metadata from this result object.
        The subsequent result objects must be against an identical
        set of result / cursor metadata, otherwise the behavior is
        undefined.

        """
        return MergedResult(self._metadata, (self,) + others)


class FilterResult(ResultInternal[_R]):
    """A wrapper for a :class:`_engine.Result` that returns objects other than
    :class:`_engine.Row` objects, such as dictionaries or scalar objects.

    :class:`_engine.FilterResult` is the common base for additional result
    APIs including :class:`_engine.MappingResult`,
    :class:`_engine.ScalarResult` and :class:`_engine.AsyncResult`.

    """

    __slots__ = (
        "_real_result",
        "_post_creational_filter",
        "_metadata",
        "_unique_filter_state",
        "__dict__",
    )

    _post_creational_filter: Optional[Callable[[Any], Any]]

    _real_result: Result[Any]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        self._real_result.__exit__(type_, value, traceback)

    @_generative
    def yield_per(self, num: int) -> Self:
        """Configure the row-fetching strategy to fetch ``num`` rows at a time.

        The :meth:`_engine.FilterResult.yield_per` method is a pass through
        to the :meth:`_engine.Result.yield_per` method.  See that method's
        documentation for usage notes.

        .. versionadded:: 1.4.40 - added :meth:`_engine.FilterResult.yield_per`
           so that the method is available on all result set implementations

        .. seealso::

            :ref:`engine_stream_results` - describes Core behavior for
            :meth:`_engine.Result.yield_per`

            :ref:`orm_queryguide_yield_per` - in the :ref:`queryguide_toplevel`

        """
        self._real_result = self._real_result.yield_per(num)
        return self

    def _soft_close(self, hard: bool = False) -> None:
        self._real_result._soft_close(hard=hard)

    @property
    def _soft_closed(self) -> bool:
        return self._real_result._soft_closed

    @property
    def closed(self) -> bool:
        """Return ``True`` if the underlying :class:`_engine.Result` reports
        closed

        .. versionadded:: 1.4.43

        """
        return self._real_result.closed

    def close(self) -> None:
        """Close this :class:`_engine.FilterResult`.

        .. versionadded:: 1.4.43

        """
        self._real_result.close()

    @property
    def _attributes(self) -> Dict[Any, Any]:
        return self._real_result._attributes

    def _fetchiter_impl(self) -> Iterator[_InterimRowType[Row[Any]]]:
        return self._real_result._fetchiter_impl()

    def _fetchone_impl(
        self, hard_close: bool = False
    ) -> Optional[_InterimRowType[Row[Any]]]:
        return self._real_result._fetchone_impl(hard_close=hard_close)

    def _fetchall_impl(self) -> List[_InterimRowType[Row[Any]]]:
        return self._real_result._fetchall_impl()

    def _fetchmany_impl(
        self, size: Optional[int] = None
    ) -> List[_InterimRowType[Row[Any]]]:
        return self._real_result._fetchmany_impl(size=size)


class ScalarResult(FilterResult[_R]):
    """A wrapper for a :class:`_engine.Result` that returns scalar values
    rather than :class:`_row.Row` values.

    The :class:`_engine.ScalarResult` object is acquired by calling the
    :meth:`_engine.Result.scalars` method.

    A special limitation of :class:`_engine.ScalarResult` is that it has
    no ``fetchone()`` method; since the semantics of ``fetchone()`` are that
    the ``None`` value indicates no more results, this is not compatible
    with :class:`_engine.ScalarResult` since there is no way to distinguish
    between ``None`` as a row value versus ``None`` as an indicator.  Use
    ``next(result)`` to receive values individually.

    """

    __slots__ = ()

    _generate_rows = False

    _post_creational_filter: Optional[Callable[[Any], Any]]

    def __init__(self, real_result: Result[Any], index: _KeyIndexType):
        self._real_result = real_result

        if real_result._source_supports_scalars:
            self._metadata = real_result._metadata
            self._post_creational_filter = None
        else:
            self._metadata = real_result._metadata._reduce([index])
            self._post_creational_filter = operator.itemgetter(0)

        self._unique_filter_state = real_result._unique_filter_state

    def unique(self, strategy: Optional[_UniqueFilterType] = None) -> Self:
        """Apply unique filtering to the objects returned by this
        :class:`_engine.ScalarResult`.

        See :meth:`_engine.Result.unique` for usage details.

        """
        self._unique_filter_state = (set(), strategy)
        return self

    def partitions(self, size: Optional[int] = None) -> Iterator[Sequence[_R]]:
        """Iterate through sub-lists of elements of the size given.

        Equivalent to :meth:`_engine.Result.partitions` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """

        getter = self._manyrow_getter

        while True:
            partition = getter(self, size)
            if partition:
                yield partition
            else:
                break

    def fetchall(self) -> Sequence[_R]:
        """A synonym for the :meth:`_engine.ScalarResult.all` method."""

        return self._allrows()

    def fetchmany(self, size: Optional[int] = None) -> Sequence[_R]:
        """Fetch many objects.

        Equivalent to :meth:`_engine.Result.fetchmany` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return self._manyrow_getter(self, size)

    def all(self) -> Sequence[_R]:
        """Return all scalar values in a sequence.

        Equivalent to :meth:`_engine.Result.all` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return self._allrows()

    def __iter__(self) -> Iterator[_R]:
        return self._iter_impl()

    def __next__(self) -> _R:
        return self._next_impl()

    def first(self) -> Optional[_R]:
        """Fetch the first object or ``None`` if no object is present.

        Equivalent to :meth:`_engine.Result.first` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.


        """
        return self._only_one_row(
            raise_for_second_row=False, raise_for_none=False, scalar=False
        )

    def one_or_none(self) -> Optional[_R]:
        """Return at most one object or raise an exception.

        Equivalent to :meth:`_engine.Result.one_or_none` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return self._only_one_row(
            raise_for_second_row=True, raise_for_none=False, scalar=False
        )

    def one(self) -> _R:
        """Return exactly one object or raise an exception.

        Equivalent to :meth:`_engine.Result.one` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return self._only_one_row(
            raise_for_second_row=True, raise_for_none=True, scalar=False
        )


class TupleResult(FilterResult[_R], util.TypingOnly):
    """A :class:`_engine.Result` that's typed as returning plain
    Python tuples instead of rows.

    Since :class:`_engine.Row` acts like a tuple in every way already,
    this class is a typing only class, regular :class:`_engine.Result` is
    still used at runtime.

    """

    __slots__ = ()

    if TYPE_CHECKING:

        def partitions(
            self, size: Optional[int] = None
        ) -> Iterator[Sequence[_R]]:
            """Iterate through sub-lists of elements of the size given.

            Equivalent to :meth:`_engine.Result.partitions` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        def fetchone(self) -> Optional[_R]:
            """Fetch one tuple.

            Equivalent to :meth:`_engine.Result.fetchone` except that
            tuple values, rather than :class:`_engine.Row`
            objects, are returned.

            """
            ...

        def fetchall(self) -> Sequence[_R]:
            """A synonym for the :meth:`_engine.ScalarResult.all` method."""
            ...

        def fetchmany(self, size: Optional[int] = None) -> Sequence[_R]:
            """Fetch many objects.

            Equivalent to :meth:`_engine.Result.fetchmany` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        def all(self) -> Sequence[_R]:  # noqa: A001
            """Return all scalar values in a sequence.

            Equivalent to :meth:`_engine.Result.all` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        def __iter__(self) -> Iterator[_R]: ...

        def __next__(self) -> _R: ...

        def first(self) -> Optional[_R]:
            """Fetch the first object or ``None`` if no object is present.

            Equivalent to :meth:`_engine.Result.first` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.


            """
            ...

        def one_or_none(self) -> Optional[_R]:
            """Return at most one object or raise an exception.

            Equivalent to :meth:`_engine.Result.one_or_none` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        def one(self) -> _R:
            """Return exactly one object or raise an exception.

            Equivalent to :meth:`_engine.Result.one` except that
            tuple values, rather than :class:`_engine.Row` objects,
            are returned.

            """
            ...

        @overload
        def scalar_one(self: TupleResult[Tuple[_T]]) -> _T: ...

        @overload
        def scalar_one(self) -> Any: ...

        def scalar_one(self) -> Any:
            """Return exactly one scalar result or raise an exception.

            This is equivalent to calling :meth:`_engine.Result.scalars`
            and then :meth:`_engine.ScalarResult.one`.

            .. seealso::

                :meth:`_engine.ScalarResult.one`

                :meth:`_engine.Result.scalars`

            """
            ...

        @overload
        def scalar_one_or_none(
            self: TupleResult[Tuple[_T]],
        ) -> Optional[_T]: ...

        @overload
        def scalar_one_or_none(self) -> Optional[Any]: ...

        def scalar_one_or_none(self) -> Optional[Any]:
            """Return exactly one or no scalar result.

            This is equivalent to calling :meth:`_engine.Result.scalars`
            and then :meth:`_engine.ScalarResult.one_or_none`.

            .. seealso::

                :meth:`_engine.ScalarResult.one_or_none`

                :meth:`_engine.Result.scalars`

            """
            ...

        @overload
        def scalar(self: TupleResult[Tuple[_T]]) -> Optional[_T]: ...

        @overload
        def scalar(self) -> Any: ...

        def scalar(self) -> Any:
            """Fetch the first column of the first row, and close the result
            set.

            Returns ``None`` if there are no rows to fetch.

            No validation is performed to test if additional rows remain.

            After calling this method, the object is fully closed,
            e.g. the :meth:`_engine.CursorResult.close`
            method will have been called.

            :return: a Python scalar value , or ``None`` if no rows remain.

            """
            ...


class MappingResult(_WithKeys, FilterResult[RowMapping]):
    """A wrapper for a :class:`_engine.Result` that returns dictionary values
    rather than :class:`_engine.Row` values.

    The :class:`_engine.MappingResult` object is acquired by calling the
    :meth:`_engine.Result.mappings` method.

    """

    __slots__ = ()

    _generate_rows = True

    _post_creational_filter = operator.attrgetter("_mapping")

    def __init__(self, result: Result[Any]):
        self._real_result = result
        self._unique_filter_state = result._unique_filter_state
        self._metadata = result._metadata
        if result._source_supports_scalars:
            self._metadata = self._metadata._reduce([0])

    def unique(self, strategy: Optional[_UniqueFilterType] = None) -> Self:
        """Apply unique filtering to the objects returned by this
        :class:`_engine.MappingResult`.

        See :meth:`_engine.Result.unique` for usage details.

        """
        self._unique_filter_state = (set(), strategy)
        return self

    def columns(self, *col_expressions: _KeyIndexType) -> Self:
        r"""Establish the columns that should be returned in each row."""
        return self._column_slices(col_expressions)

    def partitions(
        self, size: Optional[int] = None
    ) -> Iterator[Sequence[RowMapping]]:
        """Iterate through sub-lists of elements of the size given.

        Equivalent to :meth:`_engine.Result.partitions` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """

        getter = self._manyrow_getter

        while True:
            partition = getter(self, size)
            if partition:
                yield partition
            else:
                break

    def fetchall(self) -> Sequence[RowMapping]:
        """A synonym for the :meth:`_engine.MappingResult.all` method."""

        return self._allrows()

    def fetchone(self) -> Optional[RowMapping]:
        """Fetch one object.

        Equivalent to :meth:`_engine.Result.fetchone` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """

        row = self._onerow_getter(self)
        if row is _NO_ROW:
            return None
        else:
            return row

    def fetchmany(self, size: Optional[int] = None) -> Sequence[RowMapping]:
        """Fetch many objects.

        Equivalent to :meth:`_engine.Result.fetchmany` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """

        return self._manyrow_getter(self, size)

    def all(self) -> Sequence[RowMapping]:
        """Return all scalar values in a sequence.

        Equivalent to :meth:`_engine.Result.all` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """

        return self._allrows()

    def __iter__(self) -> Iterator[RowMapping]:
        return self._iter_impl()

    def __next__(self) -> RowMapping:
        return self._next_impl()

    def first(self) -> Optional[RowMapping]:
        """Fetch the first object or ``None`` if no object is present.

        Equivalent to :meth:`_engine.Result.first` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.


        """
        return self._only_one_row(
            raise_for_second_row=False, raise_for_none=False, scalar=False
        )

    def one_or_none(self) -> Optional[RowMapping]:
        """Return at most one object or raise an exception.

        Equivalent to :meth:`_engine.Result.one_or_none` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return self._only_one_row(
            raise_for_second_row=True, raise_for_none=False, scalar=False
        )

    def one(self) -> RowMapping:
        """Return exactly one object or raise an exception.

        Equivalent to :meth:`_engine.Result.one` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return self._only_one_row(
            raise_for_second_row=True, raise_for_none=True, scalar=False
        )


class FrozenResult(Generic[_TP]):
    """Represents a :class:`_engine.Result` object in a "frozen" state suitable
    for caching.

    The :class:`_engine.FrozenResult` object is returned from the
    :meth:`_engine.Result.freeze` method of any :class:`_engine.Result`
    object.

    A new iterable :class:`_engine.Result` object is generated from a fixed
    set of data each time the :class:`_engine.FrozenResult` is invoked as
    a callable::


        result = connection.execute(query)

        frozen = result.freeze()

        unfrozen_result_one = frozen()

        for row in unfrozen_result_one:
            print(row)

        unfrozen_result_two = frozen()
        rows = unfrozen_result_two.all()

        # ... etc

    .. versionadded:: 1.4

    .. seealso::

        :ref:`do_orm_execute_re_executing` - example usage within the
        ORM to implement a result-set cache.

        :func:`_orm.loading.merge_frozen_result` - ORM function to merge
        a frozen result back into a :class:`_orm.Session`.

    """

    data: Sequence[Any]

    def __init__(self, result: Result[_TP]):
        self.metadata = result._metadata._for_freeze()
        self._source_supports_scalars = result._source_supports_scalars
        self._attributes = result._attributes

        if self._source_supports_scalars:
            self.data = list(result._raw_row_iterator())
        else:
            self.data = result.fetchall()

    def rewrite_rows(self) -> Sequence[Sequence[Any]]:
        if self._source_supports_scalars:
            return [[elem] for elem in self.data]
        else:
            return [list(row) for row in self.data]

    def with_new_rows(
        self, tuple_data: Sequence[Row[_TP]]
    ) -> FrozenResult[_TP]:
        fr = FrozenResult.__new__(FrozenResult)
        fr.metadata = self.metadata
        fr._attributes = self._attributes
        fr._source_supports_scalars = self._source_supports_scalars

        if self._source_supports_scalars:
            fr.data = [d[0] for d in tuple_data]
        else:
            fr.data = tuple_data
        return fr

    def __call__(self) -> Result[_TP]:
        result: IteratorResult[_TP] = IteratorResult(
            self.metadata, iter(self.data)
        )
        result._attributes = self._attributes
        result._source_supports_scalars = self._source_supports_scalars
        return result


class IteratorResult(Result[_TP]):
    """A :class:`_engine.Result` that gets data from a Python iterator of
    :class:`_engine.Row` objects or similar row-like data.

    .. versionadded:: 1.4

    """

    _hard_closed = False
    _soft_closed = False

    def __init__(
        self,
        cursor_metadata: ResultMetaData,
        iterator: Iterator[_InterimSupportsScalarsRowType],
        raw: Optional[Result[Any]] = None,
        _source_supports_scalars: bool = False,
    ):
        self._metadata = cursor_metadata
        self.iterator = iterator
        self.raw = raw
        self._source_supports_scalars = _source_supports_scalars

    @property
    def closed(self) -> bool:
        """Return ``True`` if this :class:`_engine.IteratorResult` has
        been closed

        .. versionadded:: 1.4.43

        """
        return self._hard_closed

    def _soft_close(self, hard: bool = False, **kw: Any) -> None:
        if hard:
            self._hard_closed = True
        if self.raw is not None:
            self.raw._soft_close(hard=hard, **kw)
        self.iterator = iter([])
        self._reset_memoizations()
        self._soft_closed = True

    def _raise_hard_closed(self) -> NoReturn:
        raise exc.ResourceClosedError("This result object is closed.")

    def _raw_row_iterator(self) -> Iterator[_RowData]:
        return self.iterator

    def _fetchiter_impl(self) -> Iterator[_InterimSupportsScalarsRowType]:
        if self._hard_closed:
            self._raise_hard_closed()
        return self.iterator

    def _fetchone_impl(
        self, hard_close: bool = False
    ) -> Optional[_InterimRowType[Row[Any]]]:
        if self._hard_closed:
            self._raise_hard_closed()

        row = next(self.iterator, _NO_ROW)
        if row is _NO_ROW:
            self._soft_close(hard=hard_close)
            return None
        else:
            return row

    def _fetchall_impl(self) -> List[_InterimRowType[Row[Any]]]:
        if self._hard_closed:
            self._raise_hard_closed()
        try:
            return list(self.iterator)
        finally:
            self._soft_close()

    def _fetchmany_impl(
        self, size: Optional[int] = None
    ) -> List[_InterimRowType[Row[Any]]]:
        if self._hard_closed:
            self._raise_hard_closed()

        return list(itertools.islice(self.iterator, 0, size))


def null_result() -> IteratorResult[Any]:
    return IteratorResult(SimpleResultMetaData([]), iter([]))


class ChunkedIteratorResult(IteratorResult[_TP]):
    """An :class:`_engine.IteratorResult` that works from an
    iterator-producing callable.

    The given ``chunks`` argument is a function that is given a number of rows
    to return in each chunk, or ``None`` for all rows.  The function should
    then return an un-consumed iterator of lists, each list of the requested
    size.

    The function can be called at any time again, in which case it should
    continue from the same result set but adjust the chunk size as given.

    .. versionadded:: 1.4

    """

    def __init__(
        self,
        cursor_metadata: ResultMetaData,
        chunks: Callable[
            [Optional[int]], Iterator[Sequence[_InterimRowType[_R]]]
        ],
        source_supports_scalars: bool = False,
        raw: Optional[Result[Any]] = None,
        dynamic_yield_per: bool = False,
    ):
        self._metadata = cursor_metadata
        self.chunks = chunks
        self._source_supports_scalars = source_supports_scalars
        self.raw = raw
        self.iterator = itertools.chain.from_iterable(self.chunks(None))
        self.dynamic_yield_per = dynamic_yield_per

    @_generative
    def yield_per(self, num: int) -> Self:
        # TODO: this throws away the iterator which may be holding
        # onto a chunk.   the yield_per cannot be changed once any
        # rows have been fetched.   either find a way to enforce this,
        # or we can't use itertools.chain and will instead have to
        # keep track.

        self._yield_per = num
        self.iterator = itertools.chain.from_iterable(self.chunks(num))
        return self

    def _soft_close(self, hard: bool = False, **kw: Any) -> None:
        super()._soft_close(hard=hard, **kw)
        self.chunks = lambda size: []  # type: ignore

    def _fetchmany_impl(
        self, size: Optional[int] = None
    ) -> List[_InterimRowType[Row[Any]]]:
        if self.dynamic_yield_per:
            self.iterator = itertools.chain.from_iterable(self.chunks(size))
        return super()._fetchmany_impl(size=size)


class MergedResult(IteratorResult[_TP]):
    """A :class:`_engine.Result` that is merged from any number of
    :class:`_engine.Result` objects.

    Returned by the :meth:`_engine.Result.merge` method.

    .. versionadded:: 1.4

    """

    closed = False
    rowcount: Optional[int]

    def __init__(
        self, cursor_metadata: ResultMetaData, results: Sequence[Result[_TP]]
    ):
        self._results = results
        super().__init__(
            cursor_metadata,
            itertools.chain.from_iterable(
                r._raw_row_iterator() for r in results
            ),
        )

        self._unique_filter_state = results[0]._unique_filter_state
        self._yield_per = results[0]._yield_per

        # going to try something w/ this in next rev
        self._source_supports_scalars = results[0]._source_supports_scalars

        self._attributes = self._attributes.merge_with(
            *[r._attributes for r in results]
        )

    def _soft_close(self, hard: bool = False, **kw: Any) -> None:
        for r in self._results:
            r._soft_close(hard=hard, **kw)
        if hard:
            self.closed = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\parsers\readers.py ===
"""
Module contains tools for processing files into DataFrames or other objects

GH#48849 provides a convenient way of deprecating keyword arguments
"""
from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
import csv
import sys
from textwrap import fill
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    NamedTuple,
    TypedDict,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_copy_on_write

from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import (
    AbstractMethodError,
    ParserWarning,
)
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_file_like,
    is_float,
    is_hashable,
    is_integer,
    is_list_like,
    pandas_dtype,
)

from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs

from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper
from pandas.io.parsers.base_parser import (
    ParserBase,
    is_index_col,
    parser_defaults,
)
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
from pandas.io.parsers.python_parser import (
    FixedWidthFieldParser,
    PythonParser,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
        Sequence,
    )
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        CSVEngine,
        DtypeArg,
        DtypeBackend,
        FilePath,
        IndexLabel,
        ReadCsvBuffer,
        Self,
        StorageOptions,
        UsecolsArgType,
    )
_doc_read_csv_and_table = (
    r"""
{summary}

Also supports optionally iterating or breaking of the file
into chunks.

Additional help can be found in the online docs for
`IO Tools <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

Parameters
----------
filepath_or_buffer : str, path object or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is
    expected. A local file could be: file://localhost/path/to/table.csv.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method, such as
    a file handle (e.g. via builtin ``open`` function) or ``StringIO``.
sep : str, default {_default_sep}
    Character or regex pattern to treat as the delimiter. If ``sep=None``, the
    C engine cannot automatically detect
    the separator, but the Python parsing engine can, meaning the latter will
    be used and automatically detect the separator from only the first valid
    row of the file by Python's builtin sniffer tool, ``csv.Sniffer``.
    In addition, separators longer than 1 character and different from
    ``'\s+'`` will be interpreted as regular expressions and will also force
    the use of the Python parsing engine. Note that regex delimiters are prone
    to ignoring quoted data. Regex example: ``'\r\t'``.
delimiter : str, optional
    Alias for ``sep``.
header : int, Sequence of int, 'infer' or None, default 'infer'
    Row number(s) containing column labels and marking the start of the
    data (zero-indexed). Default behavior is to infer the column names: if no ``names``
    are passed the behavior is identical to ``header=0`` and column
    names are inferred from the first line of the file, if column
    names are passed explicitly to ``names`` then the behavior is identical to
    ``header=None``. Explicitly pass ``header=0`` to be able to
    replace existing names. The header can be a list of integers that
    specify row locations for a :class:`~pandas.MultiIndex` on the columns
    e.g. ``[0, 1, 3]``. Intervening rows that are not specified will be
    skipped (e.g. 2 in this example is skipped). Note that this
    parameter ignores commented lines and empty lines if
    ``skip_blank_lines=True``, so ``header=0`` denotes the first line of
    data rather than the first line of the file.
names : Sequence of Hashable, optional
    Sequence of column labels to apply. If the file contains a header row,
    then you should explicitly pass ``header=0`` to override the column names.
    Duplicates in this list are not allowed.
index_col : Hashable, Sequence of Hashable or False, optional
  Column(s) to use as row label(s), denoted either by column labels or column
  indices.  If a sequence of labels or indices is given, :class:`~pandas.MultiIndex`
  will be formed for the row labels.

  Note: ``index_col=False`` can be used to force pandas to *not* use the first
  column as the index, e.g., when you have a malformed file with delimiters at
  the end of each line.
usecols : Sequence of Hashable or Callable, optional
    Subset of columns to select, denoted either by column labels or column indices.
    If list-like, all elements must either
    be positional (i.e. integer indices into the document columns) or strings
    that correspond to column names provided either by the user in ``names`` or
    inferred from the document header row(s). If ``names`` are given, the document
    header row(s) are not taken into account. For example, a valid list-like
    ``usecols`` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
    Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.
    To instantiate a :class:`~pandas.DataFrame` from ``data`` with element order
    preserved use ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]``
    for columns in ``['foo', 'bar']`` order or
    ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``
    for ``['bar', 'foo']`` order.

    If callable, the callable function will be evaluated against the column
    names, returning names where the callable function evaluates to ``True``. An
    example of a valid callable argument would be ``lambda x: x.upper() in
    ['AAA', 'BBB', 'DDD']``. Using this parameter results in much faster
    parsing time and lower memory usage.
dtype : dtype or dict of {{Hashable : dtype}}, optional
    Data type(s) to apply to either the whole dataset or individual columns.
    E.g., ``{{'a': np.float64, 'b': np.int32, 'c': 'Int64'}}``
    Use ``str`` or ``object`` together with suitable ``na_values`` settings
    to preserve and not interpret ``dtype``.
    If ``converters`` are specified, they will be applied INSTEAD
    of ``dtype`` conversion.

    .. versionadded:: 1.5.0

        Support for ``defaultdict`` was added. Specify a ``defaultdict`` as input where
        the default determines the ``dtype`` of the columns which are not explicitly
        listed.
engine : {{'c', 'python', 'pyarrow'}}, optional
    Parser engine to use. The C and pyarrow engines are faster, while the python engine
    is currently more feature-complete. Multithreading is currently only supported by
    the pyarrow engine.

    .. versionadded:: 1.4.0

        The 'pyarrow' engine was added as an *experimental* engine, and some features
        are unsupported, or may not work correctly, with this engine.
converters : dict of {{Hashable : Callable}}, optional
    Functions for converting values in specified columns. Keys can either
    be column labels or column indices.
true_values : list, optional
    Values to consider as ``True`` in addition to case-insensitive variants of 'True'.
false_values : list, optional
    Values to consider as ``False`` in addition to case-insensitive variants of 'False'.
skipinitialspace : bool, default False
    Skip spaces after delimiter.
skiprows : int, list of int or Callable, optional
    Line numbers to skip (0-indexed) or number of lines to skip (``int``)
    at the start of the file.

    If callable, the callable function will be evaluated against the row
    indices, returning ``True`` if the row should be skipped and ``False`` otherwise.
    An example of a valid callable argument would be ``lambda x: x in [0, 2]``.
skipfooter : int, default 0
    Number of lines at bottom of file to skip (Unsupported with ``engine='c'``).
nrows : int, optional
    Number of rows of file to read. Useful for reading pieces of large files.
na_values : Hashable, Iterable of Hashable or dict of {{Hashable : Iterable}}, optional
    Additional strings to recognize as ``NA``/``NaN``. If ``dict`` passed, specific
    per-column ``NA`` values.  By default the following values are interpreted as
    ``NaN``: " """
    + fill('", "'.join(sorted(STR_NA_VALUES)), 70, subsequent_indent="    ")
    + """ ".

keep_default_na : bool, default True
    Whether or not to include the default ``NaN`` values when parsing the data.
    Depending on whether ``na_values`` is passed in, the behavior is as follows:

    * If ``keep_default_na`` is ``True``, and ``na_values`` are specified, ``na_values``
      is appended to the default ``NaN`` values used for parsing.
    * If ``keep_default_na`` is ``True``, and ``na_values`` are not specified, only
      the default ``NaN`` values are used for parsing.
    * If ``keep_default_na`` is ``False``, and ``na_values`` are specified, only
      the ``NaN`` values specified ``na_values`` are used for parsing.
    * If ``keep_default_na`` is ``False``, and ``na_values`` are not specified, no
      strings will be parsed as ``NaN``.

    Note that if ``na_filter`` is passed in as ``False``, the ``keep_default_na`` and
    ``na_values`` parameters will be ignored.
na_filter : bool, default True
    Detect missing value markers (empty strings and the value of ``na_values``). In
    data without any ``NA`` values, passing ``na_filter=False`` can improve the
    performance of reading a large file.
verbose : bool, default False
    Indicate number of ``NA`` values placed in non-numeric columns.

    .. deprecated:: 2.2.0
skip_blank_lines : bool, default True
    If ``True``, skip over blank lines rather than interpreting as ``NaN`` values.
parse_dates : bool, list of Hashable, list of lists or dict of {{Hashable : list}}, \
default False
    The behavior is as follows:

    * ``bool``. If ``True`` -> try parsing the index. Note: Automatically set to
      ``True`` if ``date_format`` or ``date_parser`` arguments have been passed.
    * ``list`` of ``int`` or names. e.g. If ``[1, 2, 3]`` -> try parsing columns 1, 2, 3
      each as a separate date column.
    * ``list`` of ``list``. e.g.  If ``[[1, 3]]`` -> combine columns 1 and 3 and parse
      as a single date column. Values are joined with a space before parsing.
    * ``dict``, e.g. ``{{'foo' : [1, 3]}}`` -> parse columns 1, 3 as date and call
      result 'foo'. Values are joined with a space before parsing.

    If a column or index cannot be represented as an array of ``datetime``,
    say because of an unparsable value or a mixture of timezones, the column
    or index will be returned unaltered as an ``object`` data type. For
    non-standard ``datetime`` parsing, use :func:`~pandas.to_datetime` after
    :func:`~pandas.read_csv`.

    Note: A fast-path exists for iso8601-formatted dates.
infer_datetime_format : bool, default False
    If ``True`` and ``parse_dates`` is enabled, pandas will attempt to infer the
    format of the ``datetime`` strings in the columns, and if it can be inferred,
    switch to a faster method of parsing them. In some cases this can increase
    the parsing speed by 5-10x.

    .. deprecated:: 2.0.0
        A strict version of this argument is now the default, passing it has no effect.

keep_date_col : bool, default False
    If ``True`` and ``parse_dates`` specifies combining multiple columns then
    keep the original columns.
date_parser : Callable, optional
    Function to use for converting a sequence of string columns to an array of
    ``datetime`` instances. The default uses ``dateutil.parser.parser`` to do the
    conversion. pandas will try to call ``date_parser`` in three different ways,
    advancing to the next if an exception occurs: 1) Pass one or more arrays
    (as defined by ``parse_dates``) as arguments; 2) concatenate (row-wise) the
    string values from the columns defined by ``parse_dates`` into a single array
    and pass that; and 3) call ``date_parser`` once for each row using one or
    more strings (corresponding to the columns defined by ``parse_dates``) as
    arguments.

    .. deprecated:: 2.0.0
       Use ``date_format`` instead, or read in as ``object`` and then apply
       :func:`~pandas.to_datetime` as-needed.
date_format : str or dict of column -> format, optional
    Format to use for parsing dates when used in conjunction with ``parse_dates``.
    The strftime to parse time, e.g. :const:`"%d/%m/%Y"`. See
    `strftime documentation
    <https://docs.python.org/3/library/datetime.html
    #strftime-and-strptime-behavior>`_ for more information on choices, though
    note that :const:`"%f"` will parse all the way up to nanoseconds.
    You can also pass:

    - "ISO8601", to parse any `ISO8601 <https://en.wikipedia.org/wiki/ISO_8601>`_
        time string (not necessarily in exactly the same format);
    - "mixed", to infer the format for each element individually. This is risky,
        and you should probably use it along with `dayfirst`.

    .. versionadded:: 2.0.0
dayfirst : bool, default False
    DD/MM format dates, international and European format.
cache_dates : bool, default True
    If ``True``, use a cache of unique, converted dates to apply the ``datetime``
    conversion. May produce significant speed-up when parsing duplicate
    date strings, especially ones with timezone offsets.

iterator : bool, default False
    Return ``TextFileReader`` object for iteration or getting chunks with
    ``get_chunk()``.
chunksize : int, optional
    Number of lines to read from the file per chunk. Passing a value will cause the
    function to return a ``TextFileReader`` object for iteration.
    See the `IO Tools docs
    <https://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking>`_
    for more information on ``iterator`` and ``chunksize``.

{decompression_options}

    .. versionchanged:: 1.4.0 Zstandard support.

thousands : str (length 1), optional
    Character acting as the thousands separator in numerical values.
decimal : str (length 1), default '.'
    Character to recognize as decimal point (e.g., use ',' for European data).
lineterminator : str (length 1), optional
    Character used to denote a line break. Only valid with C parser.
quotechar : str (length 1), optional
    Character used to denote the start and end of a quoted item. Quoted
    items can include the ``delimiter`` and it will be ignored.
quoting : {{0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL, 2 or csv.QUOTE_NONNUMERIC, \
3 or csv.QUOTE_NONE}}, default csv.QUOTE_MINIMAL
    Control field quoting behavior per ``csv.QUOTE_*`` constants. Default is
    ``csv.QUOTE_MINIMAL`` (i.e., 0) which implies that only fields containing special
    characters are quoted (e.g., characters defined in ``quotechar``, ``delimiter``,
    or ``lineterminator``.
doublequote : bool, default True
   When ``quotechar`` is specified and ``quoting`` is not ``QUOTE_NONE``, indicate
   whether or not to interpret two consecutive ``quotechar`` elements INSIDE a
   field as a single ``quotechar`` element.
escapechar : str (length 1), optional
    Character used to escape other characters.
comment : str (length 1), optional
    Character indicating that the remainder of line should not be parsed.
    If found at the beginning
    of a line, the line will be ignored altogether. This parameter must be a
    single character. Like empty lines (as long as ``skip_blank_lines=True``),
    fully commented lines are ignored by the parameter ``header`` but not by
    ``skiprows``. For example, if ``comment='#'``, parsing
    ``#empty\\na,b,c\\n1,2,3`` with ``header=0`` will result in ``'a,b,c'`` being
    treated as the header.
encoding : str, optional, default 'utf-8'
    Encoding to use for UTF when reading/writing (ex. ``'utf-8'``). `List of Python
    standard encodings
    <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .

encoding_errors : str, optional, default 'strict'
    How encoding errors are treated. `List of possible values
    <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

    .. versionadded:: 1.3.0

dialect : str or csv.Dialect, optional
    If provided, this parameter will override values (default or not) for the
    following parameters: ``delimiter``, ``doublequote``, ``escapechar``,
    ``skipinitialspace``, ``quotechar``, and ``quoting``. If it is necessary to
    override values, a ``ParserWarning`` will be issued. See ``csv.Dialect``
    documentation for more details.
on_bad_lines : {{'error', 'warn', 'skip'}} or Callable, default 'error'
    Specifies what to do upon encountering a bad line (a line with too many fields).
    Allowed values are :

    - ``'error'``, raise an Exception when a bad line is encountered.
    - ``'warn'``, raise a warning when a bad line is encountered and skip that line.
    - ``'skip'``, skip bad lines without raising or warning when they are encountered.

    .. versionadded:: 1.3.0

    .. versionadded:: 1.4.0

        - Callable, function with signature
          ``(bad_line: list[str]) -> list[str] | None`` that will process a single
          bad line. ``bad_line`` is a list of strings split by the ``sep``.
          If the function returns ``None``, the bad line will be ignored.
          If the function returns a new ``list`` of strings with more elements than
          expected, a ``ParserWarning`` will be emitted while dropping extra elements.
          Only supported when ``engine='python'``

    .. versionchanged:: 2.2.0

        - Callable, function with signature
          as described in `pyarrow documentation
          <https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html
          #pyarrow.csv.ParseOptions.invalid_row_handler>`_ when ``engine='pyarrow'``

delim_whitespace : bool, default False
    Specifies whether or not whitespace (e.g. ``' '`` or ``'\\t'``) will be
    used as the ``sep`` delimiter. Equivalent to setting ``sep='\\s+'``. If this option
    is set to ``True``, nothing should be passed in for the ``delimiter``
    parameter.

    .. deprecated:: 2.2.0
        Use ``sep="\\s+"`` instead.
low_memory : bool, default True
    Internally process the file in chunks, resulting in lower memory use
    while parsing, but possibly mixed type inference.  To ensure no mixed
    types either set ``False``, or specify the type with the ``dtype`` parameter.
    Note that the entire file is read into a single :class:`~pandas.DataFrame`
    regardless, use the ``chunksize`` or ``iterator`` parameter to return the data in
    chunks. (Only valid with C parser).
memory_map : bool, default False
    If a filepath is provided for ``filepath_or_buffer``, map the file object
    directly onto memory and access the data directly from there. Using this
    option can improve performance because there is no longer any I/O overhead.
float_precision : {{'high', 'legacy', 'round_trip'}}, optional
    Specifies which converter the C engine should use for floating-point
    values. The options are ``None`` or ``'high'`` for the ordinary converter,
    ``'legacy'`` for the original lower precision pandas converter, and
    ``'round_trip'`` for the round-trip converter.

{storage_options}

dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
    Back-end data type applied to the resultant :class:`DataFrame`
    (still experimental). Behaviour is as follows:

    * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
      (default).
    * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
      DataFrame.

    .. versionadded:: 2.0

Returns
-------
DataFrame or TextFileReader
    A comma-separated values (csv) file is returned as two-dimensional
    data structure with labeled axes.

See Also
--------
DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
{see_also_func_name} : {see_also_func_summary}
read_fwf : Read a table of fixed-width formatted lines into DataFrame.

Examples
--------
>>> pd.{func_name}('data.csv')  # doctest: +SKIP
"""
)


class _C_Parser_Defaults(TypedDict):
    delim_whitespace: Literal[False]
    na_filter: Literal[True]
    low_memory: Literal[True]
    memory_map: Literal[False]
    float_precision: None


_c_parser_defaults: _C_Parser_Defaults = {
    "delim_whitespace": False,
    "na_filter": True,
    "low_memory": True,
    "memory_map": False,
    "float_precision": None,
}


class _Fwf_Defaults(TypedDict):
    colspecs: Literal["infer"]
    infer_nrows: Literal[100]
    widths: None


_fwf_defaults: _Fwf_Defaults = {"colspecs": "infer", "infer_nrows": 100, "widths": None}
_c_unsupported = {"skipfooter"}
_python_unsupported = {"low_memory", "float_precision"}
_pyarrow_unsupported = {
    "skipfooter",
    "float_precision",
    "chunksize",
    "comment",
    "nrows",
    "thousands",
    "memory_map",
    "dialect",
    "delim_whitespace",
    "quoting",
    "lineterminator",
    "converters",
    "iterator",
    "dayfirst",
    "verbose",
    "skipinitialspace",
    "low_memory",
}


class _DeprecationConfig(NamedTuple):
    default_value: Any
    msg: str | None


@overload
def validate_integer(name: str, val: None, min_val: int = ...) -> None:
    ...


@overload
def validate_integer(name: str, val: float, min_val: int = ...) -> int:
    ...


@overload
def validate_integer(name: str, val: int | None, min_val: int = ...) -> int | None:
    ...


def validate_integer(
    name: str, val: int | float | None, min_val: int = 0
) -> int | None:
    """
    Checks whether the 'name' parameter for parsing is either
    an integer OR float that can SAFELY be cast to an integer
    without losing accuracy. Raises a ValueError if that is
    not the case.

    Parameters
    ----------
    name : str
        Parameter name (used for error reporting)
    val : int or float
        The value to check
    min_val : int
        Minimum allowed value (val < min_val will result in a ValueError)
    """
    if val is None:
        return val

    msg = f"'{name:s}' must be an integer >={min_val:d}"
    if is_float(val):
        if int(val) != val:
            raise ValueError(msg)
        val = int(val)
    elif not (is_integer(val) and val >= min_val):
        raise ValueError(msg)

    return int(val)


def _validate_names(names: Sequence[Hashable] | None) -> None:
    """
    Raise ValueError if the `names` parameter contains duplicates or has an
    invalid data type.

    Parameters
    ----------
    names : array-like or None
        An array containing a list of the names used for the output DataFrame.

    Raises
    ------
    ValueError
        If names are not unique or are not ordered (e.g. set).
    """
    if names is not None:
        if len(names) != len(set(names)):
            raise ValueError("Duplicate names are not allowed.")
        if not (
            is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView)
        ):
            raise ValueError("Names should be an ordered collection.")


def _read(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], kwds
) -> DataFrame | TextFileReader:
    """Generic reader of line files."""
    # if we pass a date_parser and parse_dates=False, we should not parse the
    # dates GH#44366
    if kwds.get("parse_dates", None) is None:
        if (
            kwds.get("date_parser", lib.no_default) is lib.no_default
            and kwds.get("date_format", None) is None
        ):
            kwds["parse_dates"] = False
        else:
            kwds["parse_dates"] = True

    # Extract some of the arguments (pass chunksize on).
    iterator = kwds.get("iterator", False)
    chunksize = kwds.get("chunksize", None)
    if kwds.get("engine") == "pyarrow":
        if iterator:
            raise ValueError(
                "The 'iterator' option is not supported with the 'pyarrow' engine"
            )

        if chunksize is not None:
            raise ValueError(
                "The 'chunksize' option is not supported with the 'pyarrow' engine"
            )
    else:
        chunksize = validate_integer("chunksize", chunksize, 1)

    nrows = kwds.get("nrows", None)

    # Check for duplicates in names.
    _validate_names(kwds.get("names", None))

    # Create the parser.
    parser = TextFileReader(filepath_or_buffer, **kwds)

    if chunksize or iterator:
        return parser

    with parser:
        return parser.read(nrows)


# iterator=True -> TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> TextFileReader:
    ...


# chunksize=int -> TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> TextFileReader:
    ...


# default case -> DataFrame
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


# Unions -> DataFrame | TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame | TextFileReader:
    ...


@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_csv",
        summary="Read a comma-separated values (csv) file into DataFrame.",
        see_also_func_name="read_table",
        see_also_func_summary="Read general delimited file into DataFrame.",
        _default_sep="','",
        storage_options=_shared_docs["storage_options"],
        decompression_options=_shared_docs["decompression_options"]
        % "filepath_or_buffer",
    )
)
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[Hashable, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool | lib.NoDefault = lib.no_default,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] | None = None,
    infer_datetime_format: bool | lib.NoDefault = lib.no_default,
    keep_date_col: bool | lib.NoDefault = lib.no_default,
    date_parser: Callable | lib.NoDefault = lib.no_default,
    date_format: str | dict[Hashable, str] | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = csv.QUOTE_MINIMAL,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    # Error Handling
    on_bad_lines: str = "error",
    # Internal
    delim_whitespace: bool | lib.NoDefault = lib.no_default,
    low_memory: bool = _c_parser_defaults["low_memory"],
    memory_map: bool = False,
    float_precision: Literal["high", "legacy"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | TextFileReader:
    if keep_date_col is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'keep_date_col' keyword in pd.read_csv is deprecated and "
            "will be removed in a future version. Explicitly remove unwanted "
            "columns after parsing instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        keep_date_col = False

    if lib.is_list_like(parse_dates):
        # GH#55569
        depr = False
        # error: Item "bool" of "bool | Sequence[Hashable] | None" has no
        # attribute "__iter__" (not iterable)
        if not all(is_hashable(x) for x in parse_dates):  # type: ignore[union-attr]
            depr = True
        elif isinstance(parse_dates, dict) and any(
            lib.is_list_like(x) for x in parse_dates.values()
        ):
            depr = True
        if depr:
            warnings.warn(
                "Support for nested sequences for 'parse_dates' in pd.read_csv "
                "is deprecated. Combine the desired columns with pd.to_datetime "
                "after parsing instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

    if infer_datetime_format is not lib.no_default:
        warnings.warn(
            "The argument 'infer_datetime_format' is deprecated and will "
            "be removed in a future version. "
            "A strict version of it is now the default, see "
            "https://pandas.pydata.org/pdeps/0004-consistent-to-datetime-parsing.html. "
            "You can safely remove this argument.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if delim_whitespace is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'delim_whitespace' keyword in pd.read_csv is deprecated and "
            "will be removed in a future version. Use ``sep='\\s+'`` instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        delim_whitespace = False

    if verbose is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'verbose' keyword in pd.read_csv is deprecated and "
            "will be removed in a future version.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        verbose = False

    # locals() should never be modified
    kwds = locals().copy()
    del kwds["filepath_or_buffer"]
    del kwds["sep"]

    kwds_defaults = _refine_defaults_read(
        dialect,
        delimiter,
        delim_whitespace,
        engine,
        sep,
        on_bad_lines,
        names,
        defaults={"delimiter": ","},
        dtype_backend=dtype_backend,
    )
    kwds.update(kwds_defaults)

    return _read(filepath_or_buffer, kwds)


# iterator=True -> TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> TextFileReader:
    ...


# chunksize=int -> TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> TextFileReader:
    ...


# default -> DataFrame
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


# Unions -> DataFrame | TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame | TextFileReader:
    ...


@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_table",
        summary="Read general delimited file into DataFrame.",
        see_also_func_name="read_csv",
        see_also_func_summary=(
            "Read a comma-separated values (csv) file into DataFrame."
        ),
        _default_sep=r"'\\t' (tab-stop)",
        storage_options=_shared_docs["storage_options"],
        decompression_options=_shared_docs["decompression_options"]
        % "filepath_or_buffer",
    )
)
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[Hashable, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool | lib.NoDefault = lib.no_default,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] = False,
    infer_datetime_format: bool | lib.NoDefault = lib.no_default,
    keep_date_col: bool | lib.NoDefault = lib.no_default,
    date_parser: Callable | lib.NoDefault = lib.no_default,
    date_format: str | dict[Hashable, str] | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = csv.QUOTE_MINIMAL,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    # Error Handling
    on_bad_lines: str = "error",
    # Internal
    delim_whitespace: bool | lib.NoDefault = lib.no_default,
    low_memory: bool = _c_parser_defaults["low_memory"],
    memory_map: bool = False,
    float_precision: str | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | TextFileReader:
    if keep_date_col is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'keep_date_col' keyword in pd.read_table is deprecated and "
            "will be removed in a future version. Explicitly remove unwanted "
            "columns after parsing instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        keep_date_col = False

    # error: Item "bool" of "bool | Sequence[Hashable]" has no attribute "__iter__"
    if lib.is_list_like(parse_dates) and not all(is_hashable(x) for x in parse_dates):  # type: ignore[union-attr]
        # GH#55569
        warnings.warn(
            "Support for nested sequences for 'parse_dates' in pd.read_table "
            "is deprecated. Combine the desired columns with pd.to_datetime "
            "after parsing instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if infer_datetime_format is not lib.no_default:
        warnings.warn(
            "The argument 'infer_datetime_format' is deprecated and will "
            "be removed in a future version. "
            "A strict version of it is now the default, see "
            "https://pandas.pydata.org/pdeps/0004-consistent-to-datetime-parsing.html. "
            "You can safely remove this argument.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if delim_whitespace is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'delim_whitespace' keyword in pd.read_table is deprecated and "
            "will be removed in a future version. Use ``sep='\\s+'`` instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        delim_whitespace = False

    if verbose is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'verbose' keyword in pd.read_table is deprecated and "
            "will be removed in a future version.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        verbose = False

    # locals() should never be modified
    kwds = locals().copy()
    del kwds["filepath_or_buffer"]
    del kwds["sep"]

    kwds_defaults = _refine_defaults_read(
        dialect,
        delimiter,
        delim_whitespace,
        engine,
        sep,
        on_bad_lines,
        names,
        defaults={"delimiter": "\t"},
        dtype_backend=dtype_backend,
    )
    kwds.update(kwds_defaults)

    return _read(filepath_or_buffer, kwds)


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwds,
) -> TextFileReader:
    ...


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    iterator: bool = ...,
    chunksize: int,
    **kwds,
) -> TextFileReader:
    ...


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    **kwds,
) -> DataFrame:
    ...


def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = "infer",
    widths: Sequence[int] | None = None,
    infer_nrows: int = 100,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwds,
) -> DataFrame | TextFileReader:
    r"""
    Read a table of fixed-width formatted lines into DataFrame.

    Also supports optionally iterating or breaking of the file
    into chunks.

    Additional help can be found in the `online docs for IO Tools
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a text ``read()`` function.The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.csv``.
    colspecs : list of tuple (int, int) or 'infer'. optional
        A list of tuples giving the extents of the fixed-width
        fields of each line as half-open intervals (i.e.,  [from, to[ ).
        String value 'infer' can be used to instruct the parser to try
        detecting the column specifications from the first 100 rows of
        the data which are not being skipped via skiprows (default='infer').
    widths : list of int, optional
        A list of field widths which can be used instead of 'colspecs' if
        the intervals are contiguous.
    infer_nrows : int, default 100
        The number of rows to consider when letting the parser determine the
        `colspecs`.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    **kwds : optional
        Optional keyword arguments can be passed to ``TextFileReader``.

    Returns
    -------
    DataFrame or TextFileReader
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.

    See Also
    --------
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.

    Examples
    --------
    >>> pd.read_fwf('data.csv')  # doctest: +SKIP
    """
    # Check input arguments.
    if colspecs is None and widths is None:
        raise ValueError("Must specify either colspecs or widths")
    if colspecs not in (None, "infer") and widths is not None:
        raise ValueError("You must specify only one of 'widths' and 'colspecs'")

    # Compute 'colspecs' from 'widths', if specified.
    if widths is not None:
        colspecs, col = [], 0
        for w in widths:
            colspecs.append((col, col + w))
            col += w

    # for mypy
    assert colspecs is not None

    # GH#40830
    # Ensure length of `colspecs` matches length of `names`
    names = kwds.get("names")
    if names is not None:
        if len(names) != len(colspecs) and colspecs != "infer":
            # need to check len(index_col) as it might contain
            # unnamed indices, in which case it's name is not required
            len_index = 0
            if kwds.get("index_col") is not None:
                index_col: Any = kwds.get("index_col")
                if index_col is not False:
                    if not is_list_like(index_col):
                        len_index = 1
                    else:
                        len_index = len(index_col)
            if kwds.get("usecols") is None and len(names) + len_index != len(colspecs):
                # If usecols is used colspec may be longer than names
                raise ValueError("Length of colspecs must match length of names")

    kwds["colspecs"] = colspecs
    kwds["infer_nrows"] = infer_nrows
    kwds["engine"] = "python-fwf"
    kwds["iterator"] = iterator
    kwds["chunksize"] = chunksize

    check_dtype_backend(dtype_backend)
    kwds["dtype_backend"] = dtype_backend
    return _read(filepath_or_buffer, kwds)


class TextFileReader(abc.Iterator):
    """

    Passed dialect overrides any of the related parser options

    """

    def __init__(
        self,
        f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list,
        engine: CSVEngine | None = None,
        **kwds,
    ) -> None:
        if engine is not None:
            engine_specified = True
        else:
            engine = "python"
            engine_specified = False
        self.engine = engine
        self._engine_specified = kwds.get("engine_specified", engine_specified)

        _validate_skipfooter(kwds)

        dialect = _extract_dialect(kwds)
        if dialect is not None:
            if engine == "pyarrow":
                raise ValueError(
                    "The 'dialect' option is not supported with the 'pyarrow' engine"
                )
            kwds = _merge_with_dialect_properties(dialect, kwds)

        if kwds.get("header", "infer") == "infer":
            kwds["header"] = 0 if kwds.get("names") is None else None

        self.orig_options = kwds

        # miscellanea
        self._currow = 0

        options = self._get_options_with_defaults(engine)
        options["storage_options"] = kwds.get("storage_options", None)

        self.chunksize = options.pop("chunksize", None)
        self.nrows = options.pop("nrows", None)

        self._check_file_or_buffer(f, engine)
        self.options, self.engine = self._clean_options(options, engine)

        if "has_index_names" in kwds:
            self.options["has_index_names"] = kwds["has_index_names"]

        self.handles: IOHandles | None = None
        self._engine = self._make_engine(f, self.engine)

    def close(self) -> None:
        if self.handles is not None:
            self.handles.close()
        self._engine.close()

    def _get_options_with_defaults(self, engine: CSVEngine) -> dict[str, Any]:
        kwds = self.orig_options

        options = {}
        default: object | None

        for argname, default in parser_defaults.items():
            value = kwds.get(argname, default)

            # see gh-12935
            if (
                engine == "pyarrow"
                and argname in _pyarrow_unsupported
                and value != default
                and value != getattr(value, "value", default)
            ):
                raise ValueError(
                    f"The {repr(argname)} option is not supported with the "
                    f"'pyarrow' engine"
                )
            options[argname] = value

        for argname, default in _c_parser_defaults.items():
            if argname in kwds:
                value = kwds[argname]

                if engine != "c" and value != default:
                    # TODO: Refactor this logic, its pretty convoluted
                    if "python" in engine and argname not in _python_unsupported:
                        pass
                    elif "pyarrow" in engine and argname not in _pyarrow_unsupported:
                        pass
                    else:
                        raise ValueError(
                            f"The {repr(argname)} option is not supported with the "
                            f"{repr(engine)} engine"
                        )
            else:
                value = default
            options[argname] = value

        if engine == "python-fwf":
            for argname, default in _fwf_defaults.items():
                options[argname] = kwds.get(argname, default)

        return options

    def _check_file_or_buffer(self, f, engine: CSVEngine) -> None:
        # see gh-16530
        if is_file_like(f) and engine != "c" and not hasattr(f, "__iter__"):
            # The C engine doesn't need the file-like to have the "__iter__"
            # attribute. However, the Python engine needs "__iter__(...)"
            # when iterating through such an object, meaning it
            # needs to have that attribute
            raise ValueError(
                "The 'python' engine cannot iterate through this file buffer."
            )

    def _clean_options(
        self, options: dict[str, Any], engine: CSVEngine
    ) -> tuple[dict[str, Any], CSVEngine]:
        result = options.copy()

        fallback_reason = None

        # C engine not supported yet
        if engine == "c":
            if options["skipfooter"] > 0:
                fallback_reason = "the 'c' engine does not support skipfooter"
                engine = "python"

        sep = options["delimiter"]
        delim_whitespace = options["delim_whitespace"]

        if sep is None and not delim_whitespace:
            if engine in ("c", "pyarrow"):
                fallback_reason = (
                    f"the '{engine}' engine does not support "
                    "sep=None with delim_whitespace=False"
                )
                engine = "python"
        elif sep is not None and len(sep) > 1:
            if engine == "c" and sep == r"\s+":
                result["delim_whitespace"] = True
                del result["delimiter"]
            elif engine not in ("python", "python-fwf"):
                # wait until regex engine integrated
                fallback_reason = (
                    f"the '{engine}' engine does not support "
                    "regex separators (separators > 1 char and "
                    r"different from '\s+' are interpreted as regex)"
                )
                engine = "python"
        elif delim_whitespace:
            if "python" in engine:
                result["delimiter"] = r"\s+"
        elif sep is not None:
            encodeable = True
            encoding = sys.getfilesystemencoding() or "utf-8"
            try:
                if len(sep.encode(encoding)) > 1:
                    encodeable = False
            except UnicodeDecodeError:
                encodeable = False
            if not encodeable and engine not in ("python", "python-fwf"):
                fallback_reason = (
                    f"the separator encoded in {encoding} "
                    f"is > 1 char long, and the '{engine}' engine "
                    "does not support such separators"
                )
                engine = "python"

        quotechar = options["quotechar"]
        if quotechar is not None and isinstance(quotechar, (str, bytes)):
            if (
                len(quotechar) == 1
                and ord(quotechar) > 127
                and engine not in ("python", "python-fwf")
            ):
                fallback_reason = (
                    "ord(quotechar) > 127, meaning the "
                    "quotechar is larger than one byte, "
                    f"and the '{engine}' engine does not support such quotechars"
                )
                engine = "python"

        if fallback_reason and self._engine_specified:
            raise ValueError(fallback_reason)

        if engine == "c":
            for arg in _c_unsupported:
                del result[arg]

        if "python" in engine:
            for arg in _python_unsupported:
                if fallback_reason and result[arg] != _c_parser_defaults.get(arg):
                    raise ValueError(
                        "Falling back to the 'python' engine because "
                        f"{fallback_reason}, but this causes {repr(arg)} to be "
                        "ignored as it is not supported by the 'python' engine."
                    )
                del result[arg]

        if fallback_reason:
            warnings.warn(
                (
                    "Falling back to the 'python' engine because "
                    f"{fallback_reason}; you can avoid this warning by specifying "
                    "engine='python'."
                ),
                ParserWarning,
                stacklevel=find_stack_level(),
            )

        index_col = options["index_col"]
        names = options["names"]
        converters = options["converters"]
        na_values = options["na_values"]
        skiprows = options["skiprows"]

        validate_header_arg(options["header"])

        if index_col is True:
            raise ValueError("The value of index_col couldn't be 'True'")
        if is_index_col(index_col):
            if not isinstance(index_col, (list, tuple, np.ndarray)):
                index_col = [index_col]
        result["index_col"] = index_col

        names = list(names) if names is not None else names

        # type conversion-related
        if converters is not None:
            if not isinstance(converters, dict):
                raise TypeError(
                    "Type converters must be a dict or subclass, "
                    f"input was a {type(converters).__name__}"
                )
        else:
            converters = {}

        # Converting values to NA
        keep_default_na = options["keep_default_na"]
        floatify = engine != "pyarrow"
        na_values, na_fvalues = _clean_na_values(
            na_values, keep_default_na, floatify=floatify
        )

        # handle skiprows; this is internally handled by the
        # c-engine, so only need for python and pyarrow parsers
        if engine == "pyarrow":
            if not is_integer(skiprows) and skiprows is not None:
                # pyarrow expects skiprows to be passed as an integer
                raise ValueError(
                    "skiprows argument must be an integer when using "
                    "engine='pyarrow'"
                )
        else:
            if is_integer(skiprows):
                skiprows = list(range(skiprows))
            if skiprows is None:
                skiprows = set()
            elif not callable(skiprows):
                skiprows = set(skiprows)

        # put stuff back
        result["names"] = names
        result["converters"] = converters
        result["na_values"] = na_values
        result["na_fvalues"] = na_fvalues
        result["skiprows"] = skiprows

        return result, engine

    def __next__(self) -> DataFrame:
        try:
            return self.get_chunk()
        except StopIteration:
            self.close()
            raise

    def _make_engine(
        self,
        f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list | IO,
        engine: CSVEngine = "c",
    ) -> ParserBase:
        mapping: dict[str, type[ParserBase]] = {
            "c": CParserWrapper,
            "python": PythonParser,
            "pyarrow": ArrowParserWrapper,
            "python-fwf": FixedWidthFieldParser,
        }
        if engine not in mapping:
            raise ValueError(
                f"Unknown engine: {engine} (valid options are {mapping.keys()})"
            )
        if not isinstance(f, list):
            # open file here
            is_text = True
            mode = "r"
            if engine == "pyarrow":
                is_text = False
                mode = "rb"
            elif (
                engine == "c"
                and self.options.get("encoding", "utf-8") == "utf-8"
                and isinstance(stringify_path(f), str)
            ):
                # c engine can decode utf-8 bytes, adding TextIOWrapper makes
                # the c-engine especially for memory_map=True far slower
                is_text = False
                if "b" not in mode:
                    mode += "b"
            self.handles = get_handle(
                f,
                mode,
                encoding=self.options.get("encoding", None),
                compression=self.options.get("compression", None),
                memory_map=self.options.get("memory_map", False),
                is_text=is_text,
                errors=self.options.get("encoding_errors", "strict"),
                storage_options=self.options.get("storage_options", None),
            )
            assert self.handles is not None
            f = self.handles.handle

        elif engine != "python":
            msg = f"Invalid file path or buffer object type: {type(f)}"
            raise ValueError(msg)

        try:
            return mapping[engine](f, **self.options)
        except Exception:
            if self.handles is not None:
                self.handles.close()
            raise

    def _failover_to_python(self) -> None:
        raise AbstractMethodError(self)

    def read(self, nrows: int | None = None) -> DataFrame:
        if self.engine == "pyarrow":
            try:
                # error: "ParserBase" has no attribute "read"
                df = self._engine.read()  # type: ignore[attr-defined]
            except Exception:
                self.close()
                raise
        else:
            nrows = validate_integer("nrows", nrows)
            try:
                # error: "ParserBase" has no attribute "read"
                (
                    index,
                    columns,
                    col_dict,
                ) = self._engine.read(  # type: ignore[attr-defined]
                    nrows
                )
            except Exception:
                self.close()
                raise

            if index is None:
                if col_dict:
                    # Any column is actually fine:
                    new_rows = len(next(iter(col_dict.values())))
                    index = RangeIndex(self._currow, self._currow + new_rows)
                else:
                    new_rows = 0
            else:
                new_rows = len(index)

            if hasattr(self, "orig_options"):
                dtype_arg = self.orig_options.get("dtype", None)
            else:
                dtype_arg = None

            if isinstance(dtype_arg, dict):
                dtype = defaultdict(lambda: None)  # type: ignore[var-annotated]
                dtype.update(dtype_arg)
            elif dtype_arg is not None and pandas_dtype(dtype_arg) in (
                np.str_,
                np.object_,
            ):
                dtype = defaultdict(lambda: dtype_arg)
            else:
                dtype = None

            if dtype is not None:
                new_col_dict = {}
                for k, v in col_dict.items():
                    d = (
                        dtype[k]
                        if pandas_dtype(dtype[k]) in (np.str_, np.object_)
                        else None
                    )
                    new_col_dict[k] = Series(v, index=index, dtype=d, copy=False)
            else:
                new_col_dict = col_dict

            df = DataFrame(
                new_col_dict,
                columns=columns,
                index=index,
                copy=not using_copy_on_write(),
            )

            self._currow += new_rows
        return df

    def get_chunk(self, size: int | None = None) -> DataFrame:
        if size is None:
            size = self.chunksize
        if self.nrows is not None:
            if self._currow >= self.nrows:
                raise StopIteration
            size = min(size, self.nrows - self._currow)
        return self.read(nrows=size)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def TextParser(*args, **kwds) -> TextFileReader:
    """
    Converts lists of lists/tuples into DataFrames with proper type inference
    and optional (e.g. string to datetime) conversion. Also enables iterating
    lazily over chunks of large files

    Parameters
    ----------
    data : file-like object or list
    delimiter : separator character to use
    dialect : str or csv.Dialect instance, optional
        Ignored if delimiter is longer than 1 character
    names : sequence, default
    header : int, default 0
        Row to use to parse column labels. Defaults to the first row. Prior
        rows will be discarded
    index_col : int or list, optional
        Column or columns to use as the (possibly hierarchical) index
    has_index_names: bool, default False
        True if the cols defined in index_col have an index name and are
        not in the header.
    na_values : scalar, str, list-like, or dict, optional
        Additional strings to recognize as NA/NaN.
    keep_default_na : bool, default True
    thousands : str, optional
        Thousands separator
    comment : str, optional
        Comment out remainder of line
    parse_dates : bool, default False
    keep_date_col : bool, default False
    date_parser : function, optional

        .. deprecated:: 2.0.0
    date_format : str or dict of column -> format, default ``None``

        .. versionadded:: 2.0.0
    skiprows : list of integers
        Row numbers to skip
    skipfooter : int
        Number of line at bottom of file to skip
    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the cell (not column) content, and return the
        transformed content.
    encoding : str, optional
        Encoding to use for UTF when reading/writing (ex. 'utf-8')
    float_precision : str, optional
        Specifies which converter the C engine should use for floating-point
        values. The options are `None` or `high` for the ordinary converter,
        `legacy` for the original lower precision pandas converter, and
        `round_trip` for the round-trip converter.
    """
    kwds["engine"] = "python"
    return TextFileReader(*args, **kwds)


def _clean_na_values(na_values, keep_default_na: bool = True, floatify: bool = True):
    na_fvalues: set | dict
    if na_values is None:
        if keep_default_na:
            na_values = STR_NA_VALUES
        else:
            na_values = set()
        na_fvalues = set()
    elif isinstance(na_values, dict):
        old_na_values = na_values.copy()
        na_values = {}  # Prevent aliasing.

        # Convert the values in the na_values dictionary
        # into array-likes for further use. This is also
        # where we append the default NaN values, provided
        # that `keep_default_na=True`.
        for k, v in old_na_values.items():
            if not is_list_like(v):
                v = [v]

            if keep_default_na:
                v = set(v) | STR_NA_VALUES

            na_values[k] = v
        na_fvalues = {k: _floatify_na_values(v) for k, v in na_values.items()}
    else:
        if not is_list_like(na_values):
            na_values = [na_values]
        na_values = _stringify_na_values(na_values, floatify)
        if keep_default_na:
            na_values = na_values | STR_NA_VALUES

        na_fvalues = _floatify_na_values(na_values)

    return na_values, na_fvalues


def _floatify_na_values(na_values):
    # create float versions of the na_values
    result = set()
    for v in na_values:
        try:
            v = float(v)
            if not np.isnan(v):
                result.add(v)
        except (TypeError, ValueError, OverflowError):
            pass
    return result


def _stringify_na_values(na_values, floatify: bool):
    """return a stringified and numeric for these values"""
    result: list[str | float] = []
    for x in na_values:
        result.append(str(x))
        result.append(x)
        try:
            v = float(x)

            # we are like 999 here
            if v == int(v):
                v = int(v)
                result.append(f"{v}.0")
                result.append(str(v))

            if floatify:
                result.append(v)
        except (TypeError, ValueError, OverflowError):
            pass
        if floatify:
            try:
                result.append(int(x))
            except (TypeError, ValueError, OverflowError):
                pass
    return set(result)


def _refine_defaults_read(
    dialect: str | csv.Dialect | None,
    delimiter: str | None | lib.NoDefault,
    delim_whitespace: bool,
    engine: CSVEngine | None,
    sep: str | None | lib.NoDefault,
    on_bad_lines: str | Callable,
    names: Sequence[Hashable] | None | lib.NoDefault,
    defaults: dict[str, Any],
    dtype_backend: DtypeBackend | lib.NoDefault,
):
    """Validate/refine default values of input parameters of read_csv, read_table.

    Parameters
    ----------
    dialect : str or csv.Dialect
        If provided, this parameter will override values (default or not) for the
        following parameters: `delimiter`, `doublequote`, `escapechar`,
        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to
        override values, a ParserWarning will be issued. See csv.Dialect
        documentation for more details.
    delimiter : str or object
        Alias for sep.
    delim_whitespace : bool
        Specifies whether or not whitespace (e.g. ``' '`` or ``'\t'``) will be
        used as the sep. Equivalent to setting ``sep='\\s+'``. If this option
        is set to True, nothing should be passed in for the ``delimiter``
        parameter.

        .. deprecated:: 2.2.0
            Use ``sep="\\s+"`` instead.
    engine : {{'c', 'python'}}
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    sep : str or object
        A delimiter provided by the user (str) or a sentinel value, i.e.
        pandas._libs.lib.no_default.
    on_bad_lines : str, callable
        An option for handling bad lines or a sentinel value(None).
    names : array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass ``header=0`` to override the column names.
        Duplicates in this list are not allowed.
    defaults: dict
        Default values of input parameters.

    Returns
    -------
    kwds : dict
        Input parameters with correct values.

    Raises
    ------
    ValueError :
        If a delimiter was specified with ``sep`` (or ``delimiter``) and
        ``delim_whitespace=True``.
    """
    # fix types for sep, delimiter to Union(str, Any)
    delim_default = defaults["delimiter"]
    kwds: dict[str, Any] = {}
    # gh-23761
    #
    # When a dialect is passed, it overrides any of the overlapping
    # parameters passed in directly. We don't want to warn if the
    # default parameters were passed in (since it probably means
    # that the user didn't pass them in explicitly in the first place).
    #
    # "delimiter" is the annoying corner case because we alias it to
    # "sep" before doing comparison to the dialect values later on.
    # Thus, we need a flag to indicate that we need to "override"
    # the comparison to dialect values by checking if default values
    # for BOTH "delimiter" and "sep" were provided.
    if dialect is not None:
        kwds["sep_override"] = delimiter is None and (
            sep is lib.no_default or sep == delim_default
        )

    if delimiter and (sep is not lib.no_default):
        raise ValueError("Specified a sep and a delimiter; you can only specify one.")

    kwds["names"] = None if names is lib.no_default else names

    # Alias sep -> delimiter.
    if delimiter is None:
        delimiter = sep

    if delim_whitespace and (delimiter is not lib.no_default):
        raise ValueError(
            "Specified a delimiter with both sep and "
            "delim_whitespace=True; you can only specify one."
        )

    if delimiter == "\n":
        raise ValueError(
            r"Specified \n as separator or delimiter. This forces the python engine "
            "which does not accept a line terminator. Hence it is not allowed to use "
            "the line terminator as separator.",
        )

    if delimiter is lib.no_default:
        # assign default separator value
        kwds["delimiter"] = delim_default
    else:
        kwds["delimiter"] = delimiter

    if engine is not None:
        kwds["engine_specified"] = True
    else:
        kwds["engine"] = "c"
        kwds["engine_specified"] = False

    if on_bad_lines == "error":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.ERROR
    elif on_bad_lines == "warn":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.WARN
    elif on_bad_lines == "skip":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.SKIP
    elif callable(on_bad_lines):
        if engine not in ["python", "pyarrow"]:
            raise ValueError(
                "on_bad_line can only be a callable function "
                "if engine='python' or 'pyarrow'"
            )
        kwds["on_bad_lines"] = on_bad_lines
    else:
        raise ValueError(f"Argument {on_bad_lines} is invalid for on_bad_lines")

    check_dtype_backend(dtype_backend)

    kwds["dtype_backend"] = dtype_backend

    return kwds


def _extract_dialect(kwds: dict[str, Any]) -> csv.Dialect | None:
    """
    Extract concrete csv dialect instance.

    Returns
    -------
    csv.Dialect or None
    """
    if kwds.get("dialect") is None:
        return None

    dialect = kwds["dialect"]
    if dialect in csv.list_dialects():
        dialect = csv.get_dialect(dialect)

    _validate_dialect(dialect)

    return dialect


MANDATORY_DIALECT_ATTRS = (
    "delimiter",
    "doublequote",
    "escapechar",
    "skipinitialspace",
    "quotechar",
    "quoting",
)


def _validate_dialect(dialect: csv.Dialect) -> None:
    """
    Validate csv dialect instance.

    Raises
    ------
    ValueError
        If incorrect dialect is provided.
    """
    for param in MANDATORY_DIALECT_ATTRS:
        if not hasattr(dialect, param):
            raise ValueError(f"Invalid dialect {dialect} provided")


def _merge_with_dialect_properties(
    dialect: csv.Dialect,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge default kwargs in TextFileReader with dialect parameters.

    Parameters
    ----------
    dialect : csv.Dialect
        Concrete csv dialect. See csv.Dialect documentation for more details.
    defaults : dict
        Keyword arguments passed to TextFileReader.

    Returns
    -------
    kwds : dict
        Updated keyword arguments, merged with dialect parameters.
    """
    kwds = defaults.copy()

    for param in MANDATORY_DIALECT_ATTRS:
        dialect_val = getattr(dialect, param)

        parser_default = parser_defaults[param]
        provided = kwds.get(param, parser_default)

        # Messages for conflicting values between the dialect
        # instance and the actual parameters provided.
        conflict_msgs = []

        # Don't warn if the default parameter was passed in,
        # even if it conflicts with the dialect (gh-23761).
        if provided not in (parser_default, dialect_val):
            msg = (
                f"Conflicting values for '{param}': '{provided}' was "
                f"provided, but the dialect specifies '{dialect_val}'. "
                "Using the dialect-specified value."
            )

            # Annoying corner case for not warning about
            # conflicts between dialect and delimiter parameter.
            # Refer to the outer "_read_" function for more info.
            if not (param == "delimiter" and kwds.pop("sep_override", False)):
                conflict_msgs.append(msg)

        if conflict_msgs:
            warnings.warn(
                "\n\n".join(conflict_msgs), ParserWarning, stacklevel=find_stack_level()
            )
        kwds[param] = dialect_val
    return kwds


def _validate_skipfooter(kwds: dict[str, Any]) -> None:
    """
    Check whether skipfooter is compatible with other kwargs in TextFileReader.

    Parameters
    ----------
    kwds : dict
        Keyword arguments passed to TextFileReader.

    Raises
    ------
    ValueError
        If skipfooter is not compatible with other parameters.
    """
    if kwds.get("skipfooter"):
        if kwds.get("iterator") or kwds.get("chunksize"):
            raise ValueError("'skipfooter' not supported for iteration")
        if kwds.get("nrows"):
            raise ValueError("'skipfooter' not supported with 'nrows'")