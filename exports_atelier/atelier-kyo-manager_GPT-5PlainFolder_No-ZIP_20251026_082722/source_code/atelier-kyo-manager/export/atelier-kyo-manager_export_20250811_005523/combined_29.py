
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_dialect.py ===
# testing/suite/test_dialect.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


import importlib

from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec


class PingTest(fixtures.TestBase):
    __backend__ = True

    def test_do_ping(self):
        with testing.db.connect() as conn:
            is_true(
                testing.db.dialect.do_ping(conn.connection.dbapi_connection)
            )


class ArgSignatureTest(fixtures.TestBase):
    """test that all visit_XYZ() in :class:`_sql.Compiler` subclasses have
    ``**kw``, for #8988.

    This test uses runtime code inspection.   Does not need to be a
    ``__backend__`` test as it only needs to run once provided all target
    dialects have been imported.

    For third party dialects, the suite would be run with that third
    party as a "--dburi", which means its compiler classes will have been
    imported by the time this test runs.

    """

    def _all_subclasses():  # type: ignore  # noqa
        for d in dialects.__all__:
            if not d.startswith("_"):
                importlib.import_module("sqlalchemy.dialects.%s" % d)

        stack = [Compiled]

        while stack:
            cls = stack.pop(0)
            stack.extend(cls.__subclasses__())
            yield cls

    @testing.fixture(params=list(_all_subclasses()))
    def all_subclasses(self, request):
        yield request.param

    def test_all_visit_methods_accept_kw(self, all_subclasses):
        cls = all_subclasses

        for k in cls.__dict__:
            if k.startswith("visit_"):
                meth = getattr(cls, k)

                insp = inspect_getfullargspec(meth)
                is_not_none(
                    insp.varkw,
                    f"Compiler visit method {cls.__name__}.{k}() does "
                    "not accommodate for **kw in its argument signature",
                )


class ExceptionTest(fixtures.TablesTest):
    """Test basic exception wrapping.

    DBAPIs vary a lot in exception behavior so to actually anticipate
    specific exceptions from real round trips, we need to be conservative.

    """

    run_deletes = "each"

    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "manual_pk",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("data", String(50)),
        )

    @requirements.duplicate_key_raises_integrity_error
    def test_integrity_error(self):
        with config.db.connect() as conn:
            trans = conn.begin()
            conn.execute(
                self.tables.manual_pk.insert(), {"id": 1, "data": "d1"}
            )

            assert_raises(
                exc.IntegrityError,
                conn.execute,
                self.tables.manual_pk.insert(),
                {"id": 1, "data": "d1"},
            )

            trans.rollback()

    def test_exception_with_non_ascii(self):
        with config.db.connect() as conn:
            try:
                # try to create an error message that likely has non-ascii
                # characters in the DBAPI's message string.  unfortunately
                # there's no way to make this happen with some drivers like
                # mysqlclient, pymysql.  this at least does produce a non-
                # ascii error message for cx_oracle, psycopg2
                conn.execute(select(literal_column("méil")))
                assert False
            except exc.DBAPIError as err:
                err_str = str(err)

                assert str(err.orig) in str(err)

            assert isinstance(err_str, str)


class IsolationLevelTest(fixtures.TestBase):
    __backend__ = True

    __requires__ = ("isolation_level",)

    def _get_non_default_isolation_level(self):
        levels = requirements.get_isolation_levels(config)

        default = levels["default"]
        supported = levels["supported"]

        s = set(supported).difference(["AUTOCOMMIT", default])
        if s:
            return s.pop()
        else:
            config.skip_test("no non-default isolation level available")

    def test_default_isolation_level(self):
        eq_(
            config.db.dialect.default_isolation_level,
            requirements.get_isolation_levels(config)["default"],
        )

    def test_non_default_isolation_level(self):
        non_default = self._get_non_default_isolation_level()

        with config.db.connect() as conn:
            existing = conn.get_isolation_level()

            ne_(existing, non_default)

            conn.execution_options(isolation_level=non_default)

            eq_(conn.get_isolation_level(), non_default)

            conn.dialect.reset_isolation_level(
                conn.connection.dbapi_connection
            )

            eq_(conn.get_isolation_level(), existing)

    def test_all_levels(self):
        levels = requirements.get_isolation_levels(config)

        all_levels = levels["supported"]

        for level in set(all_levels).difference(["AUTOCOMMIT"]):
            with config.db.connect() as conn:
                conn.execution_options(isolation_level=level)

                eq_(conn.get_isolation_level(), level)

                trans = conn.begin()
                trans.rollback()

                eq_(conn.get_isolation_level(), level)

            with config.db.connect() as conn:
                eq_(
                    conn.get_isolation_level(),
                    levels["default"],
                )

    @testing.requires.get_isolation_level_values
    def test_invalid_level_execution_option(self, connection_no_trans):
        """test for the new get_isolation_level_values() method"""

        connection = connection_no_trans
        with expect_raises_message(
            exc.ArgumentError,
            "Invalid value '%s' for isolation_level. "
            "Valid isolation levels for '%s' are %s"
            % (
                "FOO",
                connection.dialect.name,
                ", ".join(
                    requirements.get_isolation_levels(config)["supported"]
                ),
            ),
        ):
            connection.execution_options(isolation_level="FOO")

    @testing.requires.get_isolation_level_values
    @testing.requires.dialect_level_isolation_level_param
    def test_invalid_level_engine_param(self, testing_engine):
        """test for the new get_isolation_level_values() method
        and support for the dialect-level 'isolation_level' parameter.

        """

        eng = testing_engine(options=dict(isolation_level="FOO"))
        with expect_raises_message(
            exc.ArgumentError,
            "Invalid value '%s' for isolation_level. "
            "Valid isolation levels for '%s' are %s"
            % (
                "FOO",
                eng.dialect.name,
                ", ".join(
                    requirements.get_isolation_levels(config)["supported"]
                ),
            ),
        ):
            eng.connect()

    @testing.requires.independent_readonly_connections
    def test_dialect_user_setting_is_restored(self, testing_engine):
        levels = requirements.get_isolation_levels(config)
        default = levels["default"]
        supported = (
            sorted(
                set(levels["supported"]).difference([default, "AUTOCOMMIT"])
            )
        )[0]

        e = testing_engine(options={"isolation_level": supported})

        with e.connect() as conn:
            eq_(conn.get_isolation_level(), supported)

        with e.connect() as conn:
            conn.execution_options(isolation_level=default)
            eq_(conn.get_isolation_level(), default)

        with e.connect() as conn:
            eq_(conn.get_isolation_level(), supported)


class AutocommitIsolationTest(fixtures.TablesTest):
    run_deletes = "each"

    __requires__ = ("autocommit",)

    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("data", String(50)),
            test_needs_acid=True,
        )

    def _test_conn_autocommits(self, conn, autocommit):
        trans = conn.begin()
        conn.execute(
            self.tables.some_table.insert(), {"id": 1, "data": "some data"}
        )
        trans.rollback()

        eq_(
            conn.scalar(select(self.tables.some_table.c.id)),
            1 if autocommit else None,
        )
        conn.rollback()

        with conn.begin():
            conn.execute(self.tables.some_table.delete())

    def test_autocommit_on(self, connection_no_trans):
        conn = connection_no_trans
        c2 = conn.execution_options(isolation_level="AUTOCOMMIT")
        self._test_conn_autocommits(c2, True)

        c2.dialect.reset_isolation_level(c2.connection.dbapi_connection)

        self._test_conn_autocommits(conn, False)

    def test_autocommit_off(self, connection_no_trans):
        conn = connection_no_trans
        self._test_conn_autocommits(conn, False)

    def test_turn_autocommit_off_via_default_iso_level(
        self, connection_no_trans
    ):
        conn = connection_no_trans
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        self._test_conn_autocommits(conn, True)

        conn.execution_options(
            isolation_level=requirements.get_isolation_levels(config)[
                "default"
            ]
        )
        self._test_conn_autocommits(conn, False)

    @testing.requires.independent_readonly_connections
    @testing.variation("use_dialect_setting", [True, False])
    def test_dialect_autocommit_is_restored(
        self, testing_engine, use_dialect_setting
    ):
        """test #10147"""

        if use_dialect_setting:
            e = testing_engine(options={"isolation_level": "AUTOCOMMIT"})
        else:
            e = testing_engine().execution_options(
                isolation_level="AUTOCOMMIT"
            )

        levels = requirements.get_isolation_levels(config)

        default = levels["default"]

        with e.connect() as conn:
            self._test_conn_autocommits(conn, True)

        with e.connect() as conn:
            conn.execution_options(isolation_level=default)
            self._test_conn_autocommits(conn, False)

        with e.connect() as conn:
            self._test_conn_autocommits(conn, True)


class EscapingTest(fixtures.TestBase):
    @provide_metadata
    def test_percent_sign_round_trip(self):
        """test that the DBAPI accommodates for escaped / nonescaped
        percent signs in a way that matches the compiler

        """
        m = self.metadata
        t = Table("t", m, Column("data", String(50)))
        t.create(config.db)
        with config.db.begin() as conn:
            conn.execute(t.insert(), dict(data="some % value"))
            conn.execute(t.insert(), dict(data="some %% other value"))

            eq_(
                conn.scalar(
                    select(t.c.data).where(
                        t.c.data == literal_column("'some % value'")
                    )
                ),
                "some % value",
            )

            eq_(
                conn.scalar(
                    select(t.c.data).where(
                        t.c.data == literal_column("'some %% other value'")
                    )
                ),
                "some %% other value",
            )


class WeCanSetDefaultSchemaWEventsTest(fixtures.TestBase):
    __backend__ = True

    __requires__ = ("default_schema_name_switch",)

    def test_control_case(self):
        default_schema_name = config.db.dialect.default_schema_name

        eng = engines.testing_engine()
        with eng.connect():
            pass

        eq_(eng.dialect.default_schema_name, default_schema_name)

    def test_wont_work_wo_insert(self):
        default_schema_name = config.db.dialect.default_schema_name

        eng = engines.testing_engine()

        @event.listens_for(eng, "connect")
        def on_connect(dbapi_connection, connection_record):
            set_default_schema_on_connection(
                config, dbapi_connection, config.test_schema
            )

        with eng.connect() as conn:
            what_it_should_be = eng.dialect._get_default_schema_name(conn)
            eq_(what_it_should_be, config.test_schema)

        eq_(eng.dialect.default_schema_name, default_schema_name)

    def test_schema_change_on_connect(self):
        eng = engines.testing_engine()

        @event.listens_for(eng, "connect", insert=True)
        def on_connect(dbapi_connection, connection_record):
            set_default_schema_on_connection(
                config, dbapi_connection, config.test_schema
            )

        with eng.connect() as conn:
            what_it_should_be = eng.dialect._get_default_schema_name(conn)
            eq_(what_it_should_be, config.test_schema)

        eq_(eng.dialect.default_schema_name, config.test_schema)

    def test_schema_change_works_w_transactions(self):
        eng = engines.testing_engine()

        @event.listens_for(eng, "connect", insert=True)
        def on_connect(dbapi_connection, *arg):
            set_default_schema_on_connection(
                config, dbapi_connection, config.test_schema
            )

        with eng.connect() as conn:
            trans = conn.begin()
            what_it_should_be = eng.dialect._get_default_schema_name(conn)
            eq_(what_it_should_be, config.test_schema)
            trans.rollback()

            what_it_should_be = eng.dialect._get_default_schema_name(conn)
            eq_(what_it_should_be, config.test_schema)

        eq_(eng.dialect.default_schema_name, config.test_schema)


class FutureWeCanSetDefaultSchemaWEventsTest(
    fixtures.FutureEngineMixin, WeCanSetDefaultSchemaWEventsTest
):
    pass


class DifficultParametersTest(fixtures.TestBase):
    __backend__ = True

    tough_parameters = testing.combinations(
        ("boring",),
        ("per cent",),
        ("per % cent",),
        ("%percent",),
        ("par(ens)",),
        ("percent%(ens)yah",),
        ("col:ons",),
        ("_starts_with_underscore",),
        ("dot.s",),
        ("more :: %colons%",),
        ("_name",),
        ("___name",),
        ("[BracketsAndCase]",),
        ("42numbers",),
        ("percent%signs",),
        ("has spaces",),
        ("/slashes/",),
        ("more/slashes",),
        ("q?marks",),
        ("1param",),
        ("1col:on",),
        argnames="paramname",
    )

    @tough_parameters
    @config.requirements.unusual_column_name_characters
    def test_round_trip_same_named_column(
        self, paramname, connection, metadata
    ):
        name = paramname

        t = Table(
            "t",
            metadata,
            Column("id", Integer, primary_key=True),
            Column(name, String(50), nullable=False),
        )

        # table is created
        t.create(connection)

        # automatic param generated by insert
        connection.execute(t.insert().values({"id": 1, name: "some name"}))

        # automatic param generated by criteria, plus selecting the column
        stmt = select(t.c[name]).where(t.c[name] == "some name")

        eq_(connection.scalar(stmt), "some name")

        # use the name in a param explicitly
        stmt = select(t.c[name]).where(t.c[name] == bindparam(name))

        row = connection.execute(stmt, {name: "some name"}).first()

        # name works as the key from cursor.description
        eq_(row._mapping[name], "some name")

        # use expanding IN
        stmt = select(t.c[name]).where(
            t.c[name].in_(["some name", "some other_name"])
        )

        connection.execute(stmt).first()

    @testing.fixture
    def multirow_fixture(self, metadata, connection):
        mytable = Table(
            "mytable",
            metadata,
            Column("myid", Integer),
            Column("name", String(50)),
            Column("desc", String(50)),
        )

        mytable.create(connection)

        connection.execute(
            mytable.insert(),
            [
                {"myid": 1, "name": "a", "desc": "a_desc"},
                {"myid": 2, "name": "b", "desc": "b_desc"},
                {"myid": 3, "name": "c", "desc": "c_desc"},
                {"myid": 4, "name": "d", "desc": "d_desc"},
            ],
        )
        yield mytable

    @tough_parameters
    def test_standalone_bindparam_escape(
        self, paramname, connection, multirow_fixture
    ):
        tbl1 = multirow_fixture
        stmt = select(tbl1.c.myid).where(
            tbl1.c.name == bindparam(paramname, value="x")
        )
        res = connection.scalar(stmt, {paramname: "c"})
        eq_(res, 3)

    @tough_parameters
    def test_standalone_bindparam_escape_expanding(
        self, paramname, connection, multirow_fixture
    ):
        tbl1 = multirow_fixture
        stmt = (
            select(tbl1.c.myid)
            .where(tbl1.c.name.in_(bindparam(paramname, value=["a", "b"])))
            .order_by(tbl1.c.myid)
        )

        res = connection.scalars(stmt, {paramname: ["d", "a"]}).all()
        eq_(res, [1, 4])


class ReturningGuardsTest(fixtures.TablesTest):
    """test that the various 'returning' flags are set appropriately"""

    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "t",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=False),
            Column("data", String(50)),
        )

    @testing.fixture
    def run_stmt(self, connection):
        t = self.tables.t

        def go(stmt, executemany, id_param_name, expect_success):
            stmt = stmt.returning(t.c.id)

            if executemany:
                if not expect_success:
                    # for RETURNING executemany(), we raise our own
                    # error as this is independent of general RETURNING
                    # support
                    with expect_raises_message(
                        exc.StatementError,
                        rf"Dialect {connection.dialect.name}\+"
                        f"{connection.dialect.driver} with "
                        f"current server capabilities does not support "
                        f".*RETURNING when executemany is used",
                    ):
                        connection.execute(
                            stmt,
                            [
                                {id_param_name: 1, "data": "d1"},
                                {id_param_name: 2, "data": "d2"},
                                {id_param_name: 3, "data": "d3"},
                            ],
                        )
                else:
                    result = connection.execute(
                        stmt,
                        [
                            {id_param_name: 1, "data": "d1"},
                            {id_param_name: 2, "data": "d2"},
                            {id_param_name: 3, "data": "d3"},
                        ],
                    )
                    eq_(result.all(), [(1,), (2,), (3,)])
            else:
                if not expect_success:
                    # for RETURNING execute(), we pass all the way to the DB
                    # and let it fail
                    with expect_raises(exc.DBAPIError):
                        connection.execute(
                            stmt, {id_param_name: 1, "data": "d1"}
                        )
                else:
                    result = connection.execute(
                        stmt, {id_param_name: 1, "data": "d1"}
                    )
                    eq_(result.all(), [(1,)])

        return go

    def test_insert_single(self, connection, run_stmt):
        t = self.tables.t

        stmt = t.insert()

        run_stmt(stmt, False, "id", connection.dialect.insert_returning)

    def test_insert_many(self, connection, run_stmt):
        t = self.tables.t

        stmt = t.insert()

        run_stmt(
            stmt, True, "id", connection.dialect.insert_executemany_returning
        )

    def test_update_single(self, connection, run_stmt):
        t = self.tables.t

        connection.execute(
            t.insert(),
            [
                {"id": 1, "data": "d1"},
                {"id": 2, "data": "d2"},
                {"id": 3, "data": "d3"},
            ],
        )

        stmt = t.update().where(t.c.id == bindparam("b_id"))

        run_stmt(stmt, False, "b_id", connection.dialect.update_returning)

    def test_update_many(self, connection, run_stmt):
        t = self.tables.t

        connection.execute(
            t.insert(),
            [
                {"id": 1, "data": "d1"},
                {"id": 2, "data": "d2"},
                {"id": 3, "data": "d3"},
            ],
        )

        stmt = t.update().where(t.c.id == bindparam("b_id"))

        run_stmt(
            stmt, True, "b_id", connection.dialect.update_executemany_returning
        )

    def test_delete_single(self, connection, run_stmt):
        t = self.tables.t

        connection.execute(
            t.insert(),
            [
                {"id": 1, "data": "d1"},
                {"id": 2, "data": "d2"},
                {"id": 3, "data": "d3"},
            ],
        )

        stmt = t.delete().where(t.c.id == bindparam("b_id"))

        run_stmt(stmt, False, "b_id", connection.dialect.delete_returning)

    def test_delete_many(self, connection, run_stmt):
        t = self.tables.t

        connection.execute(
            t.insert(),
            [
                {"id": 1, "data": "d1"},
                {"id": 2, "data": "d2"},
                {"id": 3, "data": "d3"},
            ],
        )

        stmt = t.delete().where(t.c.id == bindparam("b_id"))

        run_stmt(
            stmt, True, "b_id", connection.dialect.delete_executemany_returning
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\ode.py ===
r"""
This module contains :py:meth:`~sympy.solvers.ode.dsolve` and different helper
functions that it uses.

:py:meth:`~sympy.solvers.ode.dsolve` solves ordinary differential equations.
See the docstring on the various functions for their uses.  Note that partial
differential equations support is in ``pde.py``.  Note that hint functions
have docstrings describing their various methods, but they are intended for
internal use.  Use ``dsolve(ode, func, hint=hint)`` to solve an ODE using a
specific hint.  See also the docstring on
:py:meth:`~sympy.solvers.ode.dsolve`.

**Functions in this module**

    These are the user functions in this module:

    - :py:meth:`~sympy.solvers.ode.dsolve` - Solves ODEs.
    - :py:meth:`~sympy.solvers.ode.classify_ode` - Classifies ODEs into
      possible hints for :py:meth:`~sympy.solvers.ode.dsolve`.
    - :py:meth:`~sympy.solvers.ode.checkodesol` - Checks if an equation is the
      solution to an ODE.
    - :py:meth:`~sympy.solvers.ode.homogeneous_order` - Returns the
      homogeneous order of an expression.
    - :py:meth:`~sympy.solvers.ode.infinitesimals` - Returns the infinitesimals
      of the Lie group of point transformations of an ODE, such that it is
      invariant.
    - :py:meth:`~sympy.solvers.ode.checkinfsol` - Checks if the given infinitesimals
      are the actual infinitesimals of a first order ODE.

    These are the non-solver helper functions that are for internal use.  The
    user should use the various options to
    :py:meth:`~sympy.solvers.ode.dsolve` to obtain the functionality provided
    by these functions:

    - :py:meth:`~sympy.solvers.ode.ode.odesimp` - Does all forms of ODE
      simplification.
    - :py:meth:`~sympy.solvers.ode.ode.ode_sol_simplicity` - A key function for
      comparing solutions by simplicity.
    - :py:meth:`~sympy.solvers.ode.constantsimp` - Simplifies arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode.constant_renumber` - Renumber arbitrary
      constants.
    - :py:meth:`~sympy.solvers.ode.ode._handle_Integral` - Evaluate unevaluated
      Integrals.

    See also the docstrings of these functions.

**Currently implemented solver methods**

The following methods are implemented for solving ordinary differential
equations.  See the docstrings of the various hint functions for more
information on each (run ``help(ode)``):

  - 1st order separable differential equations.
  - 1st order differential equations whose coefficients or `dx` and `dy` are
    functions homogeneous of the same order.
  - 1st order exact differential equations.
  - 1st order linear differential equations.
  - 1st order Bernoulli differential equations.
  - Power series solutions for first order differential equations.
  - Lie Group method of solving first order differential equations.
  - 2nd order Liouville differential equations.
  - Power series solutions for second order differential equations
    at ordinary and regular singular points.
  - `n`\th order differential equation that can be solved with algebraic
    rearrangement and integration.
  - `n`\th order linear homogeneous differential equation with constant
    coefficients.
  - `n`\th order linear inhomogeneous differential equation with constant
    coefficients using the method of undetermined coefficients.
  - `n`\th order linear inhomogeneous differential equation with constant
    coefficients using the method of variation of parameters.

**Philosophy behind this module**

This module is designed to make it easy to add new ODE solving methods without
having to mess with the solving code for other methods.  The idea is that
there is a :py:meth:`~sympy.solvers.ode.classify_ode` function, which takes in
an ODE and tells you what hints, if any, will solve the ODE.  It does this
without attempting to solve the ODE, so it is fast.  Each solving method is a
hint, and it has its own function, named ``ode_<hint>``.  That function takes
in the ODE and any match expression gathered by
:py:meth:`~sympy.solvers.ode.classify_ode` and returns a solved result.  If
this result has any integrals in it, the hint function will return an
unevaluated :py:class:`~sympy.integrals.integrals.Integral` class.
:py:meth:`~sympy.solvers.ode.dsolve`, which is the user wrapper function
around all of this, will then call :py:meth:`~sympy.solvers.ode.ode.odesimp` on
the result, which, among other things, will attempt to solve the equation for
the dependent variable (the function we are solving for), simplify the
arbitrary constants in the expression, and evaluate any integrals, if the hint
allows it.

**How to add new solution methods**

If you have an ODE that you want :py:meth:`~sympy.solvers.ode.dsolve` to be
able to solve, try to avoid adding special case code here.  Instead, try
finding a general method that will solve your ODE, as well as others.  This
way, the :py:mod:`~sympy.solvers.ode` module will become more robust, and
unhindered by special case hacks.  WolphramAlpha and Maple's
DETools[odeadvisor] function are two resources you can use to classify a
specific ODE.  It is also better for a method to work with an `n`\th order ODE
instead of only with specific orders, if possible.

To add a new method, there are a few things that you need to do.  First, you
need a hint name for your method.  Try to name your hint so that it is
unambiguous with all other methods, including ones that may not be implemented
yet.  If your method uses integrals, also include a ``hint_Integral`` hint.
If there is more than one way to solve ODEs with your method, include a hint
for each one, as well as a ``<hint>_best`` hint.  Your ``ode_<hint>_best()``
function should choose the best using min with ``ode_sol_simplicity`` as the
key argument.  See
:obj:`~sympy.solvers.ode.single.HomogeneousCoeffBest`, for example.
The function that uses your method will be called ``ode_<hint>()``, so the
hint must only use characters that are allowed in a Python function name
(alphanumeric characters and the underscore '``_``' character).  Include a
function for every hint, except for ``_Integral`` hints
(:py:meth:`~sympy.solvers.ode.dsolve` takes care of those automatically).
Hint names should be all lowercase, unless a word is commonly capitalized
(such as Integral or Bernoulli).  If you have a hint that you do not want to
run with ``all_Integral`` that does not have an ``_Integral`` counterpart (such
as a best hint that would defeat the purpose of ``all_Integral``), you will
need to remove it manually in the :py:meth:`~sympy.solvers.ode.dsolve` code.
See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for
guidelines on writing a hint name.

Determine *in general* how the solutions returned by your method compare with
other methods that can potentially solve the same ODEs.  Then, put your hints
in the :py:data:`~sympy.solvers.ode.allhints` tuple in the order that they
should be called.  The ordering of this tuple determines which hints are
default.  Note that exceptions are ok, because it is easy for the user to
choose individual hints with :py:meth:`~sympy.solvers.ode.dsolve`.  In
general, ``_Integral`` variants should go at the end of the list, and
``_best`` variants should go before the various hints they apply to.  For
example, the ``undetermined_coefficients`` hint comes before the
``variation_of_parameters`` hint because, even though variation of parameters
is more general than undetermined coefficients, undetermined coefficients
generally returns cleaner results for the ODEs that it can solve than
variation of parameters does, and it does not require integration, so it is
much faster.

Next, you need to have a match expression or a function that matches the type
of the ODE, which you should put in :py:meth:`~sympy.solvers.ode.classify_ode`
(if the match function is more than just a few lines.  It should match the
ODE without solving for it as much as possible, so that
:py:meth:`~sympy.solvers.ode.classify_ode` remains fast and is not hindered by
bugs in solving code.  Be sure to consider corner cases.  For example, if your
solution method involves dividing by something, make sure you exclude the case
where that division will be 0.

In most cases, the matching of the ODE will also give you the various parts
that you need to solve it.  You should put that in a dictionary (``.match()``
will do this for you), and add that as ``matching_hints['hint'] = matchdict``
in the relevant part of :py:meth:`~sympy.solvers.ode.classify_ode`.
:py:meth:`~sympy.solvers.ode.classify_ode` will then send this to
:py:meth:`~sympy.solvers.ode.dsolve`, which will send it to your function as
the ``match`` argument.  Your function should be named ``ode_<hint>(eq, func,
order, match)`.  If you need to send more information, put it in the ``match``
dictionary.  For example, if you had to substitute in a dummy variable in
:py:meth:`~sympy.solvers.ode.classify_ode` to match the ODE, you will need to
pass it to your function using the `match` dict to access it.  You can access
the independent variable using ``func.args[0]``, and the dependent variable
(the function you are trying to solve for) as ``func.func``.  If, while trying
to solve the ODE, you find that you cannot, raise ``NotImplementedError``.
:py:meth:`~sympy.solvers.ode.dsolve` will catch this error with the ``all``
meta-hint, rather than causing the whole routine to fail.

Add a docstring to your function that describes the method employed.  Like
with anything else in SymPy, you will need to add a doctest to the docstring,
in addition to real tests in ``test_ode.py``.  Try to maintain consistency
with the other hint functions' docstrings.  Add your method to the list at the
top of this docstring.  Also, add your method to ``ode.rst`` in the
``docs/src`` directory, so that the Sphinx docs will pull its docstring into
the main SymPy documentation.  Be sure to make the Sphinx documentation by
running ``make html`` from within the doc directory to verify that the
docstring formats correctly.

If your solution method involves integrating, use :py:obj:`~.Integral` instead of
:py:meth:`~sympy.core.expr.Expr.integrate`.  This allows the user to bypass
hard/slow integration by using the ``_Integral`` variant of your hint.  In
most cases, calling :py:meth:`sympy.core.basic.Basic.doit` will integrate your
solution.  If this is not the case, you will need to write special code in
:py:meth:`~sympy.solvers.ode.ode._handle_Integral`.  Arbitrary constants should be
symbols named ``C1``, ``C2``, and so on.  All solution methods should return
an equality instance.  If you need an arbitrary number of arbitrary constants,
you can use ``constants = numbered_symbols(prefix='C', cls=Symbol, start=1)``.
If it is possible to solve for the dependent function in a general way, do so.
Otherwise, do as best as you can, but do not call solve in your
``ode_<hint>()`` function.  :py:meth:`~sympy.solvers.ode.ode.odesimp` will attempt
to solve the solution for you, so you do not need to do that.  Lastly, if your
ODE has a common simplification that can be applied to your solutions, you can
add a special case in :py:meth:`~sympy.solvers.ode.ode.odesimp` for it.  For
example, solutions returned from the ``1st_homogeneous_coeff`` hints often
have many :obj:`~sympy.functions.elementary.exponential.log` terms, so
:py:meth:`~sympy.solvers.ode.ode.odesimp` calls
:py:meth:`~sympy.simplify.simplify.logcombine` on them (it also helps to write
the arbitrary constant as ``log(C1)`` instead of ``C1`` in this case).  Also
consider common ways that you can rearrange your solution to have
:py:meth:`~sympy.solvers.ode.constantsimp` take better advantage of it.  It is
better to put simplification in :py:meth:`~sympy.solvers.ode.ode.odesimp` than in
your method, because it can then be turned off with the simplify flag in
:py:meth:`~sympy.solvers.ode.dsolve`.  If you have any extraneous
simplification in your function, be sure to only run it using ``if
match.get('simplify', True):``, especially if it can be slow or if it can
reduce the domain of the solution.

Finally, as with every contribution to SymPy, your method will need to be
tested.  Add a test for each method in ``test_ode.py``.  Follow the
conventions there, i.e., test the solver using ``dsolve(eq, f(x),
hint=your_hint)``, and also test the solution using
:py:meth:`~sympy.solvers.ode.checkodesol` (you can put these in a separate
tests and skip/XFAIL if it runs too slow/does not work).  Be sure to call your
hint specifically in :py:meth:`~sympy.solvers.ode.dsolve`, that way the test
will not be broken simply by the introduction of another matching hint.  If your
method works for higher order (>1) ODEs, you will need to run ``sol =
constant_renumber(sol, 'C', 1, order)`` for each solution, where ``order`` is
the order of the ODE.  This is because ``constant_renumber`` renumbers the
arbitrary constants by printing order, which is platform dependent.  Try to
test every corner case of your solver, including a range of orders if it is a
`n`\th order solver, but if your solver is slow, such as if it involves hard
integration, try to keep the test run time down.

Feel free to refactor existing hints to avoid duplicating code or creating
inconsistencies.  If you can show that your method exactly duplicates an
existing method, including in the simplicity and speed of obtaining the
solutions, then you can remove the old, less general method.  The existing
code is tested extensively in ``test_ode.py``, so if anything is broken, one
of those tests will surely fail.

"""

from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import (Function, Derivative, AppliedUndef, diff,
    expand, expand_mul, Subs)
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal

from sympy.logic.boolalg import (BooleanAtom, BooleanTrue,
                                BooleanFalse)
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import (Poly, terms_gcd, PolynomialError, lcm)
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import (collect, logcombine, powsimp,  # type: ignore
    separatevars, simplify, cse)
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve

from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve


#: This is a list of hints in the order that they should be preferred by
#: :py:meth:`~sympy.solvers.ode.classify_ode`. In general, hints earlier in the
#: list should produce simpler solutions than those later in the list (for
#: ODEs that fit both).  For now, the order of this list is based on empirical
#: observations by the developers of SymPy.
#:
#: The hint used by :py:meth:`~sympy.solvers.ode.dsolve` for a specific ODE
#: can be overridden (see the docstring).
#:
#: In general, ``_Integral`` hints are grouped at the end of the list, unless
#: there is a method that returns an unevaluable integral most of the time
#: (which go near the end of the list anyway).  ``default``, ``all``,
#: ``best``, and ``all_Integral`` meta-hints should not be included in this
#: list, but ``_best`` and ``_Integral`` hints should be included.
allhints = (
    "factorable",
    "nth_algebraic",
    "separable",
    "1st_exact",
    "1st_linear",
    "Bernoulli",
    "1st_rational_riccati",
    "Riccati_special_minus2",
    "1st_homogeneous_coeff_best",
    "1st_homogeneous_coeff_subs_indep_div_dep",
    "1st_homogeneous_coeff_subs_dep_div_indep",
    "almost_linear",
    "linear_coefficients",
    "separable_reduced",
    "1st_power_series",
    "lie_group",
    "nth_linear_constant_coeff_homogeneous",
    "nth_linear_euler_eq_homogeneous",
    "nth_linear_constant_coeff_undetermined_coefficients",
    "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients",
    "nth_linear_constant_coeff_variation_of_parameters",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters",
    "Liouville",
    "2nd_linear_airy",
    "2nd_linear_bessel",
    "2nd_hypergeometric",
    "2nd_hypergeometric_Integral",
    "nth_order_reducible",
    "2nd_power_series_ordinary",
    "2nd_power_series_regular",
    "nth_algebraic_Integral",
    "separable_Integral",
    "1st_exact_Integral",
    "1st_linear_Integral",
    "Bernoulli_Integral",
    "1st_homogeneous_coeff_subs_indep_div_dep_Integral",
    "1st_homogeneous_coeff_subs_dep_div_indep_Integral",
    "almost_linear_Integral",
    "linear_coefficients_Integral",
    "separable_reduced_Integral",
    "nth_linear_constant_coeff_variation_of_parameters_Integral",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters_Integral",
    "Liouville_Integral",
    "2nd_nonlinear_autonomous_conserved",
    "2nd_nonlinear_autonomous_conserved_Integral",
    )



def get_numbered_constants(eq, num=1, start=1, prefix='C'):
    """
    Returns a list of constants that do not occur
    in eq already.
    """

    ncs = iter_numbered_constants(eq, start, prefix)
    Cs = [next(ncs) for i in range(num)]
    return (Cs[0] if num == 1 else tuple(Cs))


def iter_numbered_constants(eq, start=1, prefix='C'):
    """
    Returns an iterator of constants that do not occur
    in eq already.
    """

    if isinstance(eq, (Expr, Eq)):
        eq = [eq]
    elif not iterable(eq):
        raise ValueError("Expected Expr or iterable but got %s" % eq)

    atom_set = set().union(*[i.free_symbols for i in eq])
    func_set = set().union(*[i.atoms(Function) for i in eq])
    if func_set:
        atom_set |= {Symbol(str(f.func)) for f in func_set}
    return numbered_symbols(start=start, prefix=prefix, exclude=atom_set)


def dsolve(eq, func=None, hint="default", simplify=True,
    ics= None, xi=None, eta=None, x0=0, n=6, **kwargs):
    r"""
    Solves any (supported) kind of ordinary differential equation and
    system of ordinary differential equations.

    For single ordinary differential equation
    =========================================

    It is classified under this when number of equation in ``eq`` is one.
    **Usage**

        ``dsolve(eq, f(x), hint)`` -> Solve ordinary differential equation
        ``eq`` for function ``f(x)``, using method ``hint``.

    **Details**

        ``eq`` can be any supported ordinary differential equation (see the
            :py:mod:`~sympy.solvers.ode` docstring for supported methods).
            This can either be an :py:class:`~sympy.core.relational.Equality`,
            or an expression, which is assumed to be equal to ``0``.

        ``f(x)`` is a function of one variable whose derivatives in that
            variable make up the ordinary differential equation ``eq``.  In
            many cases it is not necessary to provide this; it will be
            autodetected (and an error raised if it could not be detected).

        ``hint`` is the solving method that you want dsolve to use.  Use
            ``classify_ode(eq, f(x))`` to get all of the possible hints for an
            ODE.  The default hint, ``default``, will use whatever hint is
            returned first by :py:meth:`~sympy.solvers.ode.classify_ode`.  See
            Hints below for more options that you can use for hint.

        ``simplify`` enables simplification by
            :py:meth:`~sympy.solvers.ode.ode.odesimp`.  See its docstring for more
            information.  Turn this off, for example, to disable solving of
            solutions for ``func`` or simplification of arbitrary constants.
            It will still integrate with this hint. Note that the solution may
            contain more arbitrary constants than the order of the ODE with
            this option enabled.

        ``xi`` and ``eta`` are the infinitesimal functions of an ordinary
            differential equation. They are the infinitesimals of the Lie group
            of point transformations for which the differential equation is
            invariant. The user can specify values for the infinitesimals. If
            nothing is specified, ``xi`` and ``eta`` are calculated using
            :py:meth:`~sympy.solvers.ode.infinitesimals` with the help of various
            heuristics.

        ``ics`` is the set of initial/boundary conditions for the differential equation.
          It should be given in the form of ``{f(x0): x1, f(x).diff(x).subs(x, x2):
          x3}`` and so on.  For power series solutions, if no initial
          conditions are specified ``f(0)`` is assumed to be ``C0`` and the power
          series solution is calculated about 0.

        ``x0`` is the point about which the power series solution of a differential
          equation is to be evaluated.

        ``n`` gives the exponent of the dependent variable up to which the power series
          solution of a differential equation is to be evaluated.

    **Hints**

        Aside from the various solving methods, there are also some meta-hints
        that you can pass to :py:meth:`~sympy.solvers.ode.dsolve`:

        ``default``:
                This uses whatever hint is returned first by
                :py:meth:`~sympy.solvers.ode.classify_ode`. This is the
                default argument to :py:meth:`~sympy.solvers.ode.dsolve`.

        ``all``:
                To make :py:meth:`~sympy.solvers.ode.dsolve` apply all
                relevant classification hints, use ``dsolve(ODE, func,
                hint="all")``.  This will return a dictionary of
                ``hint:solution`` terms.  If a hint causes dsolve to raise the
                ``NotImplementedError``, value of that hint's key will be the
                exception object raised.  The dictionary will also include
                some special keys:

                - ``order``: The order of the ODE.  See also
                  :py:meth:`~sympy.solvers.deutils.ode_order` in
                  ``deutils.py``.
                - ``best``: The simplest hint; what would be returned by
                  ``best`` below.
                - ``best_hint``: The hint that would produce the solution
                  given by ``best``.  If more than one hint produces the best
                  solution, the first one in the tuple returned by
                  :py:meth:`~sympy.solvers.ode.classify_ode` is chosen.
                - ``default``: The solution that would be returned by default.
                  This is the one produced by the hint that appears first in
                  the tuple returned by
                  :py:meth:`~sympy.solvers.ode.classify_ode`.

        ``all_Integral``:
                This is the same as ``all``, except if a hint also has a
                corresponding ``_Integral`` hint, it only returns the
                ``_Integral`` hint.  This is useful if ``all`` causes
                :py:meth:`~sympy.solvers.ode.dsolve` to hang because of a
                difficult or impossible integral.  This meta-hint will also be
                much faster than ``all``, because
                :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive
                routine.

        ``best``:
                To have :py:meth:`~sympy.solvers.ode.dsolve` try all methods
                and return the simplest one.  This takes into account whether
                the solution is solvable in the function, whether it contains
                any Integral classes (i.e.  unevaluatable integrals), and
                which one is the shortest in size.

        See also the :py:meth:`~sympy.solvers.ode.classify_ode` docstring for
        more info on hints, and the :py:mod:`~sympy.solvers.ode` docstring for
        a list of all supported hints.

    **Tips**

        - You can declare the derivative of an unknown function this way:

            >>> from sympy import Function, Derivative
            >>> from sympy.abc import x # x is the independent variable
            >>> f = Function("f")(x) # f is a function of x
            >>> # f_ will be the derivative of f with respect to x
            >>> f_ = Derivative(f, x)

        - See ``test_ode.py`` for many tests, which serves also as a set of
          examples for how to use :py:meth:`~sympy.solvers.ode.dsolve`.
        - :py:meth:`~sympy.solvers.ode.dsolve` always returns an
          :py:class:`~sympy.core.relational.Equality` class (except for the
          case when the hint is ``all`` or ``all_Integral``).  If possible, it
          solves the solution explicitly for the function being solved for.
          Otherwise, it returns an implicit solution.
        - Arbitrary constants are symbols named ``C1``, ``C2``, and so on.
        - Because all solutions should be mathematically equivalent, some
          hints may return the exact same result for an ODE. Often, though,
          two different hints will return the same solution formatted
          differently.  The two should be equivalent. Also note that sometimes
          the values of the arbitrary constants in two different solutions may
          not be the same, because one constant may have "absorbed" other
          constants into it.
        - Do ``help(ode.ode_<hintname>)`` to get help more information on a
          specific hint, where ``<hintname>`` is the name of a hint without
          ``_Integral``.

    For system of ordinary differential equations
    =============================================

    **Usage**
        ``dsolve(eq, func)`` -> Solve a system of ordinary differential
        equations ``eq`` for ``func`` being list of functions including
        `x(t)`, `y(t)`, `z(t)` where number of functions in the list depends
        upon the number of equations provided in ``eq``.

    **Details**

        ``eq`` can be any supported system of ordinary differential equations
        This can either be an :py:class:`~sympy.core.relational.Equality`,
        or an expression, which is assumed to be equal to ``0``.

        ``func`` holds ``x(t)`` and ``y(t)`` being functions of one variable which
        together with some of their derivatives make up the system of ordinary
        differential equation ``eq``. It is not necessary to provide this; it
        will be autodetected (and an error raised if it could not be detected).

    **Hints**

        The hints are formed by parameters returned by classify_sysode, combining
        them give hints name used later for forming method name.

    Examples
    ========

    >>> from sympy import Function, dsolve, Eq, Derivative, sin, cos, symbols
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))
    Eq(f(x), C1*sin(3*x) + C2*cos(3*x))

    >>> eq = sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x)
    >>> dsolve(eq, hint='1st_exact')
    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
    >>> dsolve(eq, hint='almost_linear')
    [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))]
    >>> t = symbols('t')
    >>> x, y = symbols('x, y', cls=Function)
    >>> eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))
    >>> dsolve(eq)
    [Eq(x(t), C1*x0(t) + C2*x0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t)),
    Eq(y(t), C1*y0(t) + C2*(y0(t)*Integral(8*exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)**2, t) +
    exp(Integral(7*t, t))*exp(Integral(12*t, t))/x0(t)))]
    >>> eq = (Eq(Derivative(x(t),t),x(t)*y(t)*sin(t)), Eq(Derivative(y(t),t),y(t)**2*sin(t)))
    >>> dsolve(eq)
    {Eq(x(t), -exp(C1)/(C2*exp(C1) - cos(t))), Eq(y(t), -1/(C1 - cos(t)))}
    """
    if iterable(eq):
        from sympy.solvers.ode.systems import dsolve_system

        # This may have to be changed in future
        # when we have weakly and strongly
        # connected components. This have to
        # changed to show the systems that haven't
        # been solved.
        try:
            sol = dsolve_system(eq, funcs=func, ics=ics, doit=True)
            return sol[0] if len(sol) == 1 else sol
        except NotImplementedError:
            pass

        match = classify_sysode(eq, func)

        eq = match['eq']
        order = match['order']
        func = match['func']
        t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]

        # keep highest order term coefficient positive
        for i in range(len(eq)):
            for func_ in func:
                if isinstance(func_, list):
                    pass
                else:
                    if eq[i].coeff(diff(func[i],t,ode_order(eq[i], func[i]))).is_negative:
                        eq[i] = -eq[i]
        match['eq'] = eq
        if len(set(order.values()))!=1:
            raise ValueError("It solves only those systems of equations whose orders are equal")
        match['order'] = list(order.values())[0]
        def recur_len(l):
            return sum(recur_len(item) if isinstance(item,list) else 1 for item in l)
        if recur_len(func) != len(eq):
            raise ValueError("dsolve() and classify_sysode() work with "
            "number of functions being equal to number of equations")
        if match['type_of_equation'] is None:
            raise NotImplementedError
        else:
            if match['is_linear'] == True:
                solvefunc = globals()['sysode_linear_%(no_of_equation)seq_order%(order)s' % match]
            else:
                solvefunc = globals()['sysode_nonlinear_%(no_of_equation)seq_order%(order)s' % match]
            sols = solvefunc(match)
            if ics:
                constants = Tuple(*sols).free_symbols - Tuple(*eq).free_symbols
                solved_constants = solve_ics(sols, func, constants, ics)
                return [sol.subs(solved_constants) for sol in sols]
            return sols
    else:
        given_hint = hint  # hint given by the user

        # See the docstring of _desolve for more details.
        hints = _desolve(eq, func=func,
            hint=hint, simplify=True, xi=xi, eta=eta, type='ode', ics=ics,
            x0=x0, n=n, **kwargs)
        eq = hints.pop('eq', eq)
        all_ = hints.pop('all', False)
        if all_:
            retdict = {}
            failed_hints = {}
            gethints = classify_ode(eq, dict=True, hint='all')
            orderedhints = gethints['ordered_hints']
            for hint in hints:
                try:
                    rv = _helper_simplify(eq, hint, hints[hint], simplify)
                except NotImplementedError as detail:
                    failed_hints[hint] = detail
                else:
                    retdict[hint] = rv
            func = hints[hint]['func']

            retdict['best'] = min(list(retdict.values()), key=lambda x:
                ode_sol_simplicity(x, func, trysolving=not simplify))
            if given_hint == 'best':
                return retdict['best']
            for i in orderedhints:
                if retdict['best'] == retdict.get(i, None):
                    retdict['best_hint'] = i
                    break
            retdict['default'] = gethints['default']
            retdict['order'] = gethints['order']
            retdict.update(failed_hints)
            return retdict

        else:
            # The key 'hint' stores the hint needed to be solved for.
            hint = hints['hint']
            return _helper_simplify(eq, hint, hints, simplify, ics=ics)


def _helper_simplify(eq, hint, match, simplify=True, ics=None, **kwargs):
    r"""
    Helper function of dsolve that calls the respective
    :py:mod:`~sympy.solvers.ode` functions to solve for the ordinary
    differential equations. This minimizes the computation in calling
    :py:meth:`~sympy.solvers.deutils._desolve` multiple times.
    """
    r = match
    func = r['func']
    order = r['order']
    match = r[hint]

    if isinstance(match, SingleODESolver):
        solvefunc = match
    else:
        solvefunc = globals()['ode_' + hint.removesuffix('_Integral')]

    free = eq.free_symbols
    cons = lambda s: s.free_symbols.difference(free)

    if simplify:
        # odesimp() will attempt to integrate, if necessary, apply constantsimp(),
        # attempt to solve for func, and apply any other hint specific
        # simplifications
        if isinstance(solvefunc, SingleODESolver):
            sols = solvefunc.get_general_solution()
        else:
            sols = solvefunc(eq, func, order, match)
        if iterable(sols):
            rv = []
            for s in sols:
                simp = odesimp(eq, s, func, hint)
                if iterable(simp):
                    rv.extend(simp)
                else:
                    rv.append(simp)
        else:
            rv = odesimp(eq, sols, func, hint)
    else:
        # We still want to integrate (you can disable it separately with the hint)
        if isinstance(solvefunc, SingleODESolver):
            exprs = solvefunc.get_general_solution(simplify=False)
        else:
            match['simplify'] = False  # Some hints can take advantage of this option
            exprs = solvefunc(eq, func, order, match)
        if isinstance(exprs, list):
            rv = [_handle_Integral(expr, func, hint) for expr in exprs]
        else:
            rv = _handle_Integral(exprs, func, hint)

    if isinstance(rv, list):
        assert all(isinstance(i, Eq) for i in rv), rv  # if not => internal error
        if simplify:
            rv = _remove_redundant_solutions(eq, rv, order, func.args[0])
        if len(rv) == 1:
            rv = rv[0]
    if ics and 'power_series' not in hint:
        if isinstance(rv, (Expr, Eq)):
            solved_constants = solve_ics([rv], [r['func']], cons(rv), ics)
            rv = rv.subs(solved_constants)
        else:
            rv1 = []
            for s in rv:
                try:
                    solved_constants = solve_ics([s], [r['func']], cons(s), ics)
                except ValueError:
                    continue
                rv1.append(s.subs(solved_constants))
            if len(rv1) == 1:
                return rv1[0]
            rv = rv1
    return rv


def solve_ics(sols, funcs, constants, ics):
    """
    Solve for the constants given initial conditions

    ``sols`` is a list of solutions.

    ``funcs`` is a list of functions.

    ``constants`` is a list of constants.

    ``ics`` is the set of initial/boundary conditions for the differential
    equation. It should be given in the form of ``{f(x0): x1,
    f(x).diff(x).subs(x, x2):  x3}`` and so on.

    Returns a dictionary mapping constants to values.
    ``solution.subs(constants)`` will replace the constants in ``solution``.

    Example
    =======
    >>> # From dsolve(f(x).diff(x) - f(x), f(x))
    >>> from sympy import symbols, Eq, exp, Function
    >>> from sympy.solvers.ode.ode import solve_ics
    >>> f = Function('f')
    >>> x, C1 = symbols('x C1')
    >>> sols = [Eq(f(x), C1*exp(x))]
    >>> funcs = [f(x)]
    >>> constants = [C1]
    >>> ics = {f(0): 2}
    >>> solved_constants = solve_ics(sols, funcs, constants, ics)
    >>> solved_constants
    {C1: 2}
    >>> sols[0].subs(solved_constants)
    Eq(f(x), 2*exp(x))

    """
    # Assume ics are of the form f(x0): value or Subs(diff(f(x), x, n), (x,
    # x0)): value (currently checked by classify_ode). To solve, replace x
    # with x0, f(x0) with value, then solve for constants. For f^(n)(x0),
    # differentiate the solution n times, so that f^(n)(x) appears.
    x = funcs[0].args[0]
    diff_sols = []
    subs_sols = []
    diff_variables = set()
    for funcarg, value in ics.items():
        if isinstance(funcarg, AppliedUndef):
            x0 = funcarg.args[0]
            matching_func = [f for f in funcs if f.func == funcarg.func][0]
            S = sols
        elif isinstance(funcarg, (Subs, Derivative)):
            if isinstance(funcarg, Subs):
                # Make sure it stays a subs. Otherwise subs below will produce
                # a different looking term.
                funcarg = funcarg.doit()
            if isinstance(funcarg, Subs):
                deriv = funcarg.expr
                x0 = funcarg.point[0]
                variables = funcarg.expr.variables
                matching_func = deriv
            elif isinstance(funcarg, Derivative):
                deriv = funcarg
                x0 = funcarg.variables[0]
                variables = (x,)*len(funcarg.variables)
                matching_func = deriv.subs(x0, x)
            for sol in sols:
                if sol.has(deriv.expr.func):
                    diff_sols.append(Eq(sol.lhs.diff(*variables), sol.rhs.diff(*variables)))
            diff_variables.add(variables)
            S = diff_sols
        else:
            raise NotImplementedError("Unrecognized initial condition")

        for sol in S:
            if sol.has(matching_func):
                sol2 = sol
                sol2 = sol2.subs(x, x0)
                sol2 = sol2.subs(funcarg, value)
                # This check is necessary because of issue #15724
                if not isinstance(sol2, BooleanAtom) or not subs_sols:
                    subs_sols = [s for s in subs_sols if not isinstance(s, BooleanAtom)]
                    subs_sols.append(sol2)

    # TODO: Use solveset here
    try:
        solved_constants = solve(subs_sols, constants, dict=True)
    except NotImplementedError:
        solved_constants = []

    # XXX: We can't differentiate between the solution not existing because of
    # invalid initial conditions, and not existing because solve is not smart
    # enough. If we could use solveset, this might be improvable, but for now,
    # we use NotImplementedError in this case.
    if not solved_constants:
        raise ValueError("Couldn't solve for initial conditions")

    if solved_constants == True:
        raise ValueError("Initial conditions did not produce any solutions for constants. Perhaps they are degenerate.")

    if len(solved_constants) > 1:
        raise NotImplementedError("Initial conditions produced too many solutions for constants")

    return solved_constants[0]

def classify_ode(eq, func=None, dict=False, ics=None, *, prep=True, xi=None, eta=None, n=None, **kwargs):
    r"""
    Returns a tuple of possible :py:meth:`~sympy.solvers.ode.dsolve`
    classifications for an ODE.

    The tuple is ordered so that first item is the classification that
    :py:meth:`~sympy.solvers.ode.dsolve` uses to solve the ODE by default.  In
    general, classifications at the near the beginning of the list will
    produce better solutions faster than those near the end, thought there are
    always exceptions.  To make :py:meth:`~sympy.solvers.ode.dsolve` use a
    different classification, use ``dsolve(ODE, func,
    hint=<classification>)``.  See also the
    :py:meth:`~sympy.solvers.ode.dsolve` docstring for different meta-hints
    you can use.

    If ``dict`` is true, :py:meth:`~sympy.solvers.ode.classify_ode` will
    return a dictionary of ``hint:match`` expression terms. This is intended
    for internal use by :py:meth:`~sympy.solvers.ode.dsolve`.  Note that
    because dictionaries are ordered arbitrarily, this will most likely not be
    in the same order as the tuple.

    You can get help on different hints by executing
    ``help(ode.ode_hintname)``, where ``hintname`` is the name of the hint
    without ``_Integral``.

    See :py:data:`~sympy.solvers.ode.allhints` or the
    :py:mod:`~sympy.solvers.ode` docstring for a list of all supported hints
    that can be returned from :py:meth:`~sympy.solvers.ode.classify_ode`.

    Notes
    =====

    These are remarks on hint names.

    ``_Integral``

        If a classification has ``_Integral`` at the end, it will return the
        expression with an unevaluated :py:class:`~.Integral`
        class in it.  Note that a hint may do this anyway if
        :py:meth:`~sympy.core.expr.Expr.integrate` cannot do the integral,
        though just using an ``_Integral`` will do so much faster.  Indeed, an
        ``_Integral`` hint will always be faster than its corresponding hint
        without ``_Integral`` because
        :py:meth:`~sympy.core.expr.Expr.integrate` is an expensive routine.
        If :py:meth:`~sympy.solvers.ode.dsolve` hangs, it is probably because
        :py:meth:`~sympy.core.expr.Expr.integrate` is hanging on a tough or
        impossible integral.  Try using an ``_Integral`` hint or
        ``all_Integral`` to get it return something.

        Note that some hints do not have ``_Integral`` counterparts. This is
        because :py:func:`~sympy.integrals.integrals.integrate` is not used in
        solving the ODE for those method. For example, `n`\th order linear
        homogeneous ODEs with constant coefficients do not require integration
        to solve, so there is no
        ``nth_linear_homogeneous_constant_coeff_Integrate`` hint. You can
        easily evaluate any unevaluated
        :py:class:`~sympy.integrals.integrals.Integral`\s in an expression by
        doing ``expr.doit()``.

    Ordinals

        Some hints contain an ordinal such as ``1st_linear``.  This is to help
        differentiate them from other hints, as well as from other methods
        that may not be implemented yet. If a hint has ``nth`` in it, such as
        the ``nth_linear`` hints, this means that the method used to applies
        to ODEs of any order.

    ``indep`` and ``dep``

        Some hints contain the words ``indep`` or ``dep``.  These reference
        the independent variable and the dependent function, respectively. For
        example, if an ODE is in terms of `f(x)`, then ``indep`` will refer to
        `x` and ``dep`` will refer to `f`.

    ``subs``

        If a hints has the word ``subs`` in it, it means that the ODE is solved
        by substituting the expression given after the word ``subs`` for a
        single dummy variable.  This is usually in terms of ``indep`` and
        ``dep`` as above.  The substituted expression will be written only in
        characters allowed for names of Python objects, meaning operators will
        be spelled out.  For example, ``indep``/``dep`` will be written as
        ``indep_div_dep``.

    ``coeff``

        The word ``coeff`` in a hint refers to the coefficients of something
        in the ODE, usually of the derivative terms.  See the docstring for
        the individual methods for more info (``help(ode)``).  This is
        contrast to ``coefficients``, as in ``undetermined_coefficients``,
        which refers to the common name of a method.

    ``_best``

        Methods that have more than one fundamental way to solve will have a
        hint for each sub-method and a ``_best`` meta-classification. This
        will evaluate all hints and return the best, using the same
        considerations as the normal ``best`` meta-hint.


    Examples
    ========

    >>> from sympy import Function, classify_ode, Eq
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> classify_ode(Eq(f(x).diff(x), 0), f(x))
    ('nth_algebraic',
    'separable',
    '1st_exact',
    '1st_linear',
    'Bernoulli',
    '1st_homogeneous_coeff_best',
    '1st_homogeneous_coeff_subs_indep_div_dep',
    '1st_homogeneous_coeff_subs_dep_div_indep',
    '1st_power_series', 'lie_group', 'nth_linear_constant_coeff_homogeneous',
    'nth_linear_euler_eq_homogeneous',
    'nth_algebraic_Integral', 'separable_Integral', '1st_exact_Integral',
    '1st_linear_Integral', 'Bernoulli_Integral',
    '1st_homogeneous_coeff_subs_indep_div_dep_Integral',
    '1st_homogeneous_coeff_subs_dep_div_indep_Integral')
    >>> classify_ode(f(x).diff(x, 2) + 3*f(x).diff(x) + 2*f(x) - 4)
    ('factorable', 'nth_linear_constant_coeff_undetermined_coefficients',
    'nth_linear_constant_coeff_variation_of_parameters',
    'nth_linear_constant_coeff_variation_of_parameters_Integral')

    """
    ics = sympify(ics)

    if func and len(func.args) != 1:
        raise ValueError("dsolve() and classify_ode() only "
        "work with functions of one variable, not %s" % func)

    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs

    # Some methods want the unprocessed equation
    eq_orig = eq

    if prep or func is None:
        eq, func_ = _preprocess(eq, func)
        if func is None:
            func = func_
    x = func.args[0]
    f = func.func
    y = Dummy('y')
    terms = 5 if n is None else n

    order = ode_order(eq, f(x))
    # hint:matchdict or hint:(tuple of matchdicts)
    # Also will contain "default":<default hint> and "order":order items.
    matching_hints = {"order": order}

    df = f(x).diff(x)
    a = Wild('a', exclude=[f(x)])
    d = Wild('d', exclude=[df, f(x).diff(x, 2)])
    e = Wild('e', exclude=[df])
    n = Wild('n', exclude=[x, f(x), df])
    c1 = Wild('c1', exclude=[x])
    a3 = Wild('a3', exclude=[f(x), df, f(x).diff(x, 2)])
    b3 = Wild('b3', exclude=[f(x), df, f(x).diff(x, 2)])
    c3 = Wild('c3', exclude=[f(x), df, f(x).diff(x, 2)])
    boundary = {}  # Used to extract initial conditions
    C1 = Symbol("C1")

    # Preprocessing to get the initial conditions out
    if ics is not None:
        for funcarg in ics:
            # Separating derivatives
            if isinstance(funcarg, (Subs, Derivative)):
                # f(x).diff(x).subs(x, 0) is a Subs, but f(x).diff(x).subs(x,
                # y) is a Derivative
                if isinstance(funcarg, Subs):
                    deriv = funcarg.expr
                    old = funcarg.variables[0]
                    new = funcarg.point[0]
                elif isinstance(funcarg, Derivative):
                    deriv = funcarg
                    # No information on this. Just assume it was x
                    old = x
                    new = funcarg.variables[0]

                if (isinstance(deriv, Derivative) and isinstance(deriv.args[0],
                    AppliedUndef) and deriv.args[0].func == f and
                    len(deriv.args[0].args) == 1 and old == x and not
                    new.has(x) and all(i == deriv.variables[0] for i in
                    deriv.variables) and x not in ics[funcarg].free_symbols):

                    dorder = ode_order(deriv, x)
                    temp = 'f' + str(dorder)
                    boundary.update({temp: new, temp + 'val': ics[funcarg]})
                else:
                    raise ValueError("Invalid boundary conditions for Derivatives")


            # Separating functions
            elif isinstance(funcarg, AppliedUndef):
                if (funcarg.func == f and len(funcarg.args) == 1 and
                    not funcarg.args[0].has(x) and x not in ics[funcarg].free_symbols):
                    boundary.update({'f0': funcarg.args[0], 'f0val': ics[funcarg]})
                else:
                    raise ValueError("Invalid boundary conditions for Function")

            else:
                raise ValueError("Enter boundary conditions of the form ics={f(point): value, f(x).diff(x, order).subs(x, point): value}")

    ode = SingleODEProblem(eq_orig, func, x, prep=prep, xi=xi, eta=eta)
    user_hint = kwargs.get('hint', 'default')
    # Used when dsolve is called without an explicit hint.
    # We exit early to return the first valid match
    early_exit = (user_hint=='default')
    user_hint = user_hint.removesuffix('_Integral')
    user_map = solver_map
    # An explicit hint has been given to dsolve
    # Skip matching code for other hints
    if user_hint not in ['default', 'all', 'all_Integral', 'best'] and user_hint in solver_map:
        user_map = {user_hint: solver_map[user_hint]}

    for hint in user_map:
        solver = user_map[hint](ode)
        if solver.matches():
            matching_hints[hint] = solver
            if user_map[hint].has_integral:
                matching_hints[hint + "_Integral"] = solver
            if dict and early_exit:
                matching_hints["default"] = hint
                return matching_hints

    eq = expand(eq)
    # Precondition to try remove f(x) from highest order derivative
    reduced_eq = None
    if eq.is_Add:
        deriv_coef = eq.coeff(f(x).diff(x, order))
        if deriv_coef not in (1, 0):
            r = deriv_coef.match(a*f(x)**c1)
            if r and r[c1]:
                den = f(x)**r[c1]
                reduced_eq = Add(*[arg/den for arg in eq.args])
    if not reduced_eq:
        reduced_eq = eq

    if order == 1:

        # NON-REDUCED FORM OF EQUATION matches
        r = collect(eq, df, exact=True).match(d + e * df)
        if r:
            r['d'] = d
            r['e'] = e
            r['y'] = y
            r[d] = r[d].subs(f(x), y)
            r[e] = r[e].subs(f(x), y)

            # FIRST ORDER POWER SERIES WHICH NEEDS INITIAL CONDITIONS
            # TODO: Hint first order series should match only if d/e is analytic.
            # For now, only d/e and (d/e).diff(arg) is checked for existence at
            # at a given point.
            # This is currently done internally in ode_1st_power_series.
            point = boundary.get('f0', 0)
            value = boundary.get('f0val', C1)
            check = cancel(r[d]/r[e])
            check1 = check.subs({x: point, y: value})
            if not check1.has(oo) and not check1.has(zoo) and \
                not check1.has(nan) and not check1.has(-oo):
                check2 = (check1.diff(x)).subs({x: point, y: value})
                if not check2.has(oo) and not check2.has(zoo) and \
                    not check2.has(nan) and not check2.has(-oo):
                    rseries = r.copy()
                    rseries.update({'terms': terms, 'f0': point, 'f0val': value})
                    matching_hints["1st_power_series"] = rseries

    elif order == 2:
        # Homogeneous second order differential equation of the form
        # a3*f(x).diff(x, 2) + b3*f(x).diff(x) + c3
        # It has a definite power series solution at point x0 if, b3/a3 and c3/a3
        # are analytic at x0.
        deq = a3*(f(x).diff(x, 2)) + b3*df + c3*f(x)
        r = collect(reduced_eq,
            [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
        ordinary = False
        if r:
            if not all(r[key].is_polynomial() for key in r):
                n, d = reduced_eq.as_numer_denom()
                reduced_eq = expand(n)
                r = collect(reduced_eq,
                    [f(x).diff(x, 2), f(x).diff(x), f(x)]).match(deq)
        if r and r[a3] != 0:
            p = cancel(r[b3]/r[a3])  # Used below
            q = cancel(r[c3]/r[a3])  # Used below
            point = kwargs.get('x0', 0)
            check = p.subs(x, point)
            if not check.has(oo, nan, zoo, -oo):
                check = q.subs(x, point)
                if not check.has(oo, nan, zoo, -oo):
                    ordinary = True
                    r.update({'a3': a3, 'b3': b3, 'c3': c3, 'x0': point, 'terms': terms})
                    matching_hints["2nd_power_series_ordinary"] = r

            # Checking if the differential equation has a regular singular point
            # at x0. It has a regular singular point at x0, if (b3/a3)*(x - x0)
            # and (c3/a3)*((x - x0)**2) are analytic at x0.
            if not ordinary:
                p = cancel((x - point)*p)
                check = p.subs(x, point)
                if not check.has(oo, nan, zoo, -oo):
                    q = cancel(((x - point)**2)*q)
                    check = q.subs(x, point)
                    if not check.has(oo, nan, zoo, -oo):
                        coeff_dict = {'p': p, 'q': q, 'x0': point, 'terms': terms}
                        matching_hints["2nd_power_series_regular"] = coeff_dict


    # Order keys based on allhints.
    retlist = [i for i in allhints if i in matching_hints]
    if dict:
        # Dictionaries are ordered arbitrarily, so make note of which
        # hint would come first for dsolve().  Use an ordered dict in Py 3.
        matching_hints["default"] = retlist[0] if retlist else None
        matching_hints["ordered_hints"] = tuple(retlist)
        return matching_hints
    else:
        return tuple(retlist)


def classify_sysode(eq, funcs=None, **kwargs):
    r"""
    Returns a dictionary of parameter names and values that define the system
    of ordinary differential equations in ``eq``.
    The parameters are further used in
    :py:meth:`~sympy.solvers.ode.dsolve` for solving that system.

    Some parameter names and values are:

    'is_linear' (boolean), which tells whether the given system is linear.
    Note that "linear" here refers to the operator: terms such as ``x*diff(x,t)`` are
    nonlinear, whereas terms like ``sin(t)*diff(x,t)`` are still linear operators.

    'func' (list) contains the :py:class:`~sympy.core.function.Function`s that
    appear with a derivative in the ODE, i.e. those that we are trying to solve
    the ODE for.

    'order' (dict) with the maximum derivative for each element of the 'func'
    parameter.

    'func_coeff' (dict or Matrix) with the coefficient for each triple ``(equation number,
    function, order)```. The coefficients are those subexpressions that do not
    appear in 'func', and hence can be considered constant for purposes of ODE
    solving. The value of this parameter can also be a  Matrix if the system of ODEs are
    linear first order of the form X' = AX where X is the vector of dependent variables.
    Here, this function returns the coefficient matrix A.

    'eq' (list) with the equations from ``eq``, sympified and transformed into
    expressions (we are solving for these expressions to be zero).

    'no_of_equations' (int) is the number of equations (same as ``len(eq)``).

    'type_of_equation' (string) is an internal classification of the type of
    ODE.

    'is_constant' (boolean), which tells if the system of ODEs is constant coefficient
    or not. This key is temporary addition for now and is in the match dict only when
    the system of ODEs is linear first order constant coefficient homogeneous. So, this
    key's value is True for now if it is available else it does not exist.

    'is_homogeneous' (boolean), which tells if the system of ODEs is homogeneous. Like the
    key 'is_constant', this key is a temporary addition and it is True since this key value
    is available only when the system is linear first order constant coefficient homogeneous.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode-toc1.htm
    -A. D. Polyanin and A. V. Manzhirov, Handbook of Mathematics for Engineers and Scientists

    Examples
    ========

    >>> from sympy import Function, Eq, symbols, diff
    >>> from sympy.solvers.ode.ode import classify_sysode
    >>> from sympy.abc import t
    >>> f, x, y = symbols('f, x, y', cls=Function)
    >>> k, l, m, n = symbols('k, l, m, n', Integer=True)
    >>> x1 = diff(x(t), t) ; y1 = diff(y(t), t)
    >>> x2 = diff(x(t), t, t) ; y2 = diff(y(t), t, t)
    >>> eq = (Eq(x1, 12*x(t) - 6*y(t)), Eq(y1, 11*x(t) + 3*y(t)))
    >>> classify_sysode(eq)
    {'eq': [-12*x(t) + 6*y(t) + Derivative(x(t), t), -11*x(t) - 3*y(t) + Derivative(y(t), t)], 'func': [x(t), y(t)],
     'func_coeff': {(0, x(t), 0): -12, (0, x(t), 1): 1, (0, y(t), 0): 6, (0, y(t), 1): 0, (1, x(t), 0): -11, (1, x(t), 1): 0, (1, y(t), 0): -3, (1, y(t), 1): 1}, 'is_linear': True, 'no_of_equation': 2, 'order': {x(t): 1, y(t): 1}, 'type_of_equation': None}
    >>> eq = (Eq(diff(x(t),t), 5*t*x(t) + t**2*y(t) + 2), Eq(diff(y(t),t), -t**2*x(t) + 5*t*y(t)))
    >>> classify_sysode(eq)
    {'eq': [-t**2*y(t) - 5*t*x(t) + Derivative(x(t), t) - 2, t**2*x(t) - 5*t*y(t) + Derivative(y(t), t)],
     'func': [x(t), y(t)], 'func_coeff': {(0, x(t), 0): -5*t, (0, x(t), 1): 1, (0, y(t), 0): -t**2, (0, y(t), 1): 0,
     (1, x(t), 0): t**2, (1, x(t), 1): 0, (1, y(t), 0): -5*t, (1, y(t), 1): 1}, 'is_linear': True, 'no_of_equation': 2,
      'order': {x(t): 1, y(t): 1}, 'type_of_equation': None}

    """

    # Sympify equations and convert iterables of equations into
    # a list of equations
    def _sympify(eq):
        return list(map(sympify, eq if iterable(eq) else [eq]))

    eq, funcs = (_sympify(w) for w in [eq, funcs])
    for i, fi in enumerate(eq):
        if isinstance(fi, Equality):
            eq[i] = fi.lhs - fi.rhs

    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    matching_hints = {"no_of_equation":i+1}
    matching_hints['eq'] = eq
    if i==0:
        raise ValueError("classify_sysode() works for systems of ODEs. "
        "For scalar ODEs, classify_ode should be used")

    # find all the functions if not given
    order = {}
    if funcs==[None]:
        funcs = _extract_funcs(eq)

    funcs = list(set(funcs))
    if len(funcs) != len(eq):
        raise ValueError("Number of functions given is not equal to the number of equations %s" % funcs)

    # This logic of list of lists in funcs to
    # be replaced later.
    func_dict = {}
    for func in funcs:
        if not order.get(func, False):
            max_order = 0
            for i, eqs_ in enumerate(eq):
                order_ = ode_order(eqs_,func)
                if max_order < order_:
                    max_order = order_
                    eq_no = i
            if eq_no in func_dict:
                func_dict[eq_no] = [func_dict[eq_no], func]
            else:
                func_dict[eq_no] = func
            order[func] = max_order

    funcs = [func_dict[i] for i in range(len(func_dict))]
    matching_hints['func'] = funcs
    for func in funcs:
        if isinstance(func, list):
            for func_elem in func:
                if len(func_elem.args) != 1:
                    raise ValueError("dsolve() and classify_sysode() work with "
                    "functions of one variable only, not %s" % func)
        else:
            if func and len(func.args) != 1:
                raise ValueError("dsolve() and classify_sysode() work with "
                "functions of one variable only, not %s" % func)

    # find the order of all equation in system of odes
    matching_hints["order"] = order

    # find coefficients of terms f(t), diff(f(t),t) and higher derivatives
    # and similarly for other functions g(t), diff(g(t),t) in all equations.
    # Here j denotes the equation number, funcs[l] denotes the function about
    # which we are talking about and k denotes the order of function funcs[l]
    # whose coefficient we are calculating.
    def linearity_check(eqs, j, func, is_linear_):
        for k in range(order[func] + 1):
            func_coef[j, func, k] = collect(eqs.expand(), [diff(func, t, k)]).coeff(diff(func, t, k))
            if is_linear_ == True:
                if func_coef[j, func, k] == 0:
                    if k == 0:
                        coef = eqs.as_independent(func, as_Add=True)[1]
                        for xr in range(1, ode_order(eqs,func) + 1):
                            coef -= eqs.as_independent(diff(func, t, xr), as_Add=True)[1]
                        if coef != 0:
                            is_linear_ = False
                    else:
                        if eqs.as_independent(diff(func, t, k), as_Add=True)[1]:
                            is_linear_ = False
                else:
                    for func_ in funcs:
                        if isinstance(func_, list):
                            for elem_func_ in func_:
                                dep = func_coef[j, func, k].as_independent(elem_func_, as_Add=True)[1]
                                if dep != 0:
                                    is_linear_ = False
                        else:
                            dep = func_coef[j, func, k].as_independent(func_, as_Add=True)[1]
                            if dep != 0:
                                is_linear_ = False
        return is_linear_

    func_coef = {}
    is_linear = True
    for j, eqs in enumerate(eq):
        for func in funcs:
            if isinstance(func, list):
                for func_elem in func:
                    is_linear = linearity_check(eqs, j, func_elem, is_linear)
            else:
                is_linear = linearity_check(eqs, j, func, is_linear)
    matching_hints['func_coeff'] = func_coef
    matching_hints['is_linear'] = is_linear


    if len(set(order.values())) == 1:
        order_eq = list(matching_hints['order'].values())[0]
        if matching_hints['is_linear'] == True:
            if matching_hints['no_of_equation'] == 2:
                if order_eq == 1:
                    type_of_equation = check_linear_2eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            # If the equation does not match up with any of the
            # general case solvers in systems.py and the number
            # of equations is greater than 2, then NotImplementedError
            # should be raised.
            else:
                type_of_equation = None

        else:
            if matching_hints['no_of_equation'] == 2:
                if order_eq == 1:
                    type_of_equation = check_nonlinear_2eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            elif matching_hints['no_of_equation'] == 3:
                if order_eq == 1:
                    type_of_equation = check_nonlinear_3eq_order1(eq, funcs, func_coef)
                else:
                    type_of_equation = None
            else:
                type_of_equation = None
    else:
        type_of_equation = None

    matching_hints['type_of_equation'] = type_of_equation

    return matching_hints


def check_linear_2eq_order1(eq, func, func_coef):
    x = func[0].func
    y = func[1].func
    fc = func_coef
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    r = {}
    # for equations Eq(a1*diff(x(t),t), b1*x(t) + c1*y(t) + d1)
    # and Eq(a2*diff(y(t),t), b2*x(t) + c2*y(t) + d2)
    r['a1'] = fc[0,x(t),1] ; r['a2'] = fc[1,y(t),1]
    r['b1'] = -fc[0,x(t),0]/fc[0,x(t),1] ; r['b2'] = -fc[1,x(t),0]/fc[1,y(t),1]
    r['c1'] = -fc[0,y(t),0]/fc[0,x(t),1] ; r['c2'] = -fc[1,y(t),0]/fc[1,y(t),1]
    forcing = [S.Zero,S.Zero]
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j
    if not (forcing[0].has(t) or forcing[1].has(t)):
        # We can handle homogeneous case and simple constant forcings
        r['d1'] = forcing[0]
        r['d2'] = forcing[1]
    else:
        # Issue #9244: nonhomogeneous linear systems are not supported
        return None

    # Conditions to check for type 6 whose equations are Eq(diff(x(t),t), f(t)*x(t) + g(t)*y(t)) and
    # Eq(diff(y(t),t), a*[f(t) + a*h(t)]x(t) + a*[g(t) - h(t)]*y(t))
    p = 0
    q = 0
    p1 = cancel(r['b2']/(cancel(r['b2']/r['c2']).as_numer_denom()[0]))
    p2 = cancel(r['b1']/(cancel(r['b1']/r['c1']).as_numer_denom()[0]))
    for n, i in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q and n==0:
                if ((r['b2']/j - r['b1'])/(r['c1'] - r['c2']/j)) == j:
                    p = 1
            elif q and n==1:
                if ((r['b1']/j - r['b2'])/(r['c2'] - r['c1']/j)) == j:
                    p = 2
    # End of condition for type 6

    if r['d1']!=0 or r['d2']!=0:
        return None
    else:
        if not any(r[k].has(t) for k in 'a1 a2 b1 b2 c1 c2'.split()):
            return None
        else:
            r['b1'] = r['b1']/r['a1'] ; r['b2'] = r['b2']/r['a2']
            r['c1'] = r['c1']/r['a1'] ; r['c2'] = r['c2']/r['a2']
            if p:
                return "type6"
            else:
                # Equations for type 7 are Eq(diff(x(t),t), f(t)*x(t) + g(t)*y(t)) and Eq(diff(y(t),t), h(t)*x(t) + p(t)*y(t))
                return "type7"
def check_nonlinear_2eq_order1(eq, func, func_coef):
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    f = Wild('f')
    g = Wild('g')
    u, v = symbols('u, v', cls=Dummy)
    def check_type(x, y):
        r1 = eq[0].match(t*diff(x(t),t) - x(t) + f)
        r2 = eq[1].match(t*diff(y(t),t) - y(t) + g)
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t),t) - x(t)/t + f/t)
            r2 = eq[1].match(diff(y(t),t) - y(t)/t + g/t)
        if not (r1 and r2):
            r1 = (-eq[0]).match(t*diff(x(t),t) - x(t) + f)
            r2 = (-eq[1]).match(t*diff(y(t),t) - y(t) + g)
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t),t) - x(t)/t + f/t)
            r2 = (-eq[1]).match(diff(y(t),t) - y(t)/t + g/t)
        if r1 and r2 and not (r1[f].subs(diff(x(t),t),u).subs(diff(y(t),t),v).has(t) \
        or r2[g].subs(diff(x(t),t),u).subs(diff(y(t),t),v).has(t)):
            return 'type5'
        else:
            return None
    for func_ in func:
        if isinstance(func_, list):
            x = func[0][0].func
            y = func[0][1].func
            eq_type = check_type(x, y)
            if not eq_type:
                eq_type = check_type(y, x)
            return eq_type
    x = func[0].func
    y = func[1].func
    fc = func_coef
    n = Wild('n', exclude=[x(t),y(t)])
    f1 = Wild('f1', exclude=[v,t])
    f2 = Wild('f2', exclude=[v,t])
    g1 = Wild('g1', exclude=[u,t])
    g2 = Wild('g2', exclude=[u,t])
    for i in range(2):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms/fc[i,func[i],1]
        eq[i] = eqs
    r = eq[0].match(diff(x(t),t) - x(t)**n*f)
    if r:
        g = (diff(y(t),t) - eq[1])/r[f]
    if r and not (g.has(x(t)) or g.subs(y(t),v).has(t) or r[f].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type1'
    r = eq[0].match(diff(x(t),t) - exp(n*x(t))*f)
    if r:
        g = (diff(y(t),t) - eq[1])/r[f]
    if r and not (g.has(x(t)) or g.subs(y(t),v).has(t) or r[f].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type2'
    g = Wild('g')
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    if r1 and r2 and not (r1[f].subs(x(t),u).subs(y(t),v).has(t) or \
    r2[g].subs(x(t),u).subs(y(t),v).has(t)):
        return 'type3'
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    num, den = (
        (r1[f].subs(x(t),u).subs(y(t),v))/
        (r2[g].subs(x(t),u).subs(y(t),v))).as_numer_denom()
    R1 = num.match(f1*g1)
    R2 = den.match(f2*g2)
    # phi = (r1[f].subs(x(t),u).subs(y(t),v))/num
    if R1 and R2:
        return 'type4'
    return None


def check_nonlinear_2eq_order2(eq, func, func_coef):
    return None

def check_nonlinear_3eq_order1(eq, func, func_coef):
    x = func[0].func
    y = func[1].func
    z = func[2].func
    fc = func_coef
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    u, v, w = symbols('u, v, w', cls=Dummy)
    a = Wild('a', exclude=[x(t), y(t), z(t), t])
    b = Wild('b', exclude=[x(t), y(t), z(t), t])
    c = Wild('c', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')
    F1 = Wild('F1')
    F2 = Wild('F2')
    F3 = Wild('F3')
    for i in range(3):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms/fc[i,func[i],1]
        eq[i] = eqs
    r1 = eq[0].match(diff(x(t),t) - a*y(t)*z(t))
    r2 = eq[1].match(diff(y(t),t) - b*z(t)*x(t))
    r3 = eq[2].match(diff(z(t),t) - c*x(t)*y(t))
    if r1 and r2 and r3:
        num1, den1 = r1[a].as_numer_denom()
        num2, den2 = r2[b].as_numer_denom()
        num3, den3 = r3[c].as_numer_denom()
        if solve([num1*u-den1*(v-w), num2*v-den2*(w-u), num3*w-den3*(u-v)],[u, v]):
            return 'type1'
    r = eq[0].match(diff(x(t),t) - y(t)*z(t)*f)
    if r:
        r1 = collect_const(r[f]).match(a*f)
        r2 = ((diff(y(t),t) - eq[1])/r1[f]).match(b*z(t)*x(t))
        r3 = ((diff(z(t),t) - eq[2])/r1[f]).match(c*x(t)*y(t))
    if r1 and r2 and r3:
        num1, den1 = r1[a].as_numer_denom()
        num2, den2 = r2[b].as_numer_denom()
        num3, den3 = r3[c].as_numer_denom()
        if solve([num1*u-den1*(v-w), num2*v-den2*(w-u), num3*w-den3*(u-v)],[u, v]):
            return 'type2'
    r = eq[0].match(diff(x(t),t) - (F2-F3))
    if r:
        r1 = collect_const(r[F2]).match(c*F2)
        r1.update(collect_const(r[F3]).match(b*F3))
        if r1:
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                r1[F2], r1[F3] = r1[F3], r1[F2]
                r1[c], r1[b] = -r1[b], -r1[c]
            r2 = eq[1].match(diff(y(t),t) - a*r1[F3] + r1[c]*F1)
        if r2:
            r3 = (eq[2] == diff(z(t),t) - r1[b]*r2[F1] + r2[a]*r1[F2])
        if r1 and r2 and r3:
            return 'type3'
    r = eq[0].match(diff(x(t),t) - z(t)*F2 + y(t)*F3)
    if r:
        r1 = collect_const(r[F2]).match(c*F2)
        r1.update(collect_const(r[F3]).match(b*F3))
        if r1:
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                r1[F2], r1[F3] = r1[F3], r1[F2]
                r1[c], r1[b] = -r1[b], -r1[c]
            r2 = (diff(y(t),t) - eq[1]).match(a*x(t)*r1[F3] - r1[c]*z(t)*F1)
        if r2:
            r3 = (diff(z(t),t) - eq[2] == r1[b]*y(t)*r2[F1] - r2[a]*x(t)*r1[F2])
        if r1 and r2 and r3:
            return 'type4'
    r = (diff(x(t),t) - eq[0]).match(x(t)*(F2 - F3))
    if r:
        r1 = collect_const(r[F2]).match(c*F2)
        r1.update(collect_const(r[F3]).match(b*F3))
        if r1:
            if eq[1].has(r1[F2]) and not eq[1].has(r1[F3]):
                r1[F2], r1[F3] = r1[F3], r1[F2]
                r1[c], r1[b] = -r1[b], -r1[c]
            r2 = (diff(y(t),t) - eq[1]).match(y(t)*(a*r1[F3] - r1[c]*F1))
        if r2:
            r3 = (diff(z(t),t) - eq[2] == z(t)*(r1[b]*r2[F1] - r2[a]*r1[F2]))
        if r1 and r2 and r3:
            return 'type5'
    return None


def check_nonlinear_3eq_order2(eq, func, func_coef):
    return None


@vectorize(0)
def odesimp(ode, eq, func, hint):
    r"""
    Simplifies solutions of ODEs, including trying to solve for ``func`` and
    running :py:meth:`~sympy.solvers.ode.constantsimp`.

    It may use knowledge of the type of solution that the hint returns to
    apply additional simplifications.

    It also attempts to integrate any :py:class:`~sympy.integrals.integrals.Integral`\s
    in the expression, if the hint is not an ``_Integral`` hint.

    This function should have no effect on expressions returned by
    :py:meth:`~sympy.solvers.ode.dsolve`, as
    :py:meth:`~sympy.solvers.ode.dsolve` already calls
    :py:meth:`~sympy.solvers.ode.ode.odesimp`, but the individual hint functions
    do not call :py:meth:`~sympy.solvers.ode.ode.odesimp` (because the
    :py:meth:`~sympy.solvers.ode.dsolve` wrapper does).  Therefore, this
    function is designed for mainly internal use.

    Examples
    ========

    >>> from sympy import sin, symbols, dsolve, pprint, Function
    >>> from sympy.solvers.ode.ode import odesimp
    >>> x, u2, C1= symbols('x,u2,C1')
    >>> f = Function('f')

    >>> eq = dsolve(x*f(x).diff(x) - f(x) - x*sin(f(x)/x), f(x),
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep_Integral',
    ... simplify=False)
    >>> pprint(eq, wrap_line=False)
                            x
                           ----
                           f(x)
                             /
                            |
                            |   /        1   \
                            |  -|u1 + -------|
                            |   |        /1 \|
                            |   |     sin|--||
                            |   \        \u1//
    log(f(x)) = log(C1) +   |  ---------------- d(u1)
                            |          2
                            |        u1
                            |
                           /

    >>> pprint(odesimp(eq, f(x), 1, {C1},
    ... hint='1st_homogeneous_coeff_subs_indep_div_dep'
    ... )) #doctest: +SKIP
        x
    --------- = C1
       /f(x)\
    tan|----|
       \2*x /

    """
    x = func.args[0]
    f = func.func
    C1 = get_numbered_constants(eq, num=1)
    constants = eq.free_symbols - ode.free_symbols

    # First, integrate if the hint allows it.
    eq = _handle_Integral(eq, func, hint)
    if hint.startswith("nth_linear_euler_eq_nonhomogeneous"):
        eq = simplify(eq)
    if not isinstance(eq, Equality):
        raise TypeError("eq should be an instance of Equality")

    # allow simplifications under assumption that symbols are nonzero
    eq = eq.xreplace((_:={i: Dummy(nonzero=True) for i in constants})).xreplace({_[i]: i for i in _})

    # Second, clean up the arbitrary constants.
    # Right now, nth linear hints can put as many as 2*order constants in an
    # expression.  If that number grows with another hint, the third argument
    # here should be raised accordingly, or constantsimp() rewritten to handle
    # an arbitrary number of constants.
    eq = constantsimp(eq, constants)

    # Lastly, now that we have cleaned up the expression, try solving for func.
    # When CRootOf is implemented in solve(), we will want to return a CRootOf
    # every time instead of an Equality.

    # Get the f(x) on the left if possible.
    if eq.rhs == func and not eq.lhs.has(func):
        eq = [Eq(eq.rhs, eq.lhs)]

    # make sure we are working with lists of solutions in simplified form.
    if eq.lhs == func and not eq.rhs.has(func):
        # The solution is already solved
        eq = [eq]

    else:
        # The solution is not solved, so try to solve it
        try:
            floats = any(i.is_Float for i in eq.atoms(Number))
            eqsol = solve(eq, func, force=True, rational=False if floats else None)
            if not eqsol:
                raise NotImplementedError
        except (NotImplementedError, PolynomialError):
            eq = [eq]
        else:
            def _expand(expr):
                numer, denom = expr.as_numer_denom()

                if denom.is_Add:
                    return expr
                else:
                    return powsimp(expr.expand(), combine='exp', deep=True)

            # XXX: the rest of odesimp() expects each ``t`` to be in a
            # specific normal form: rational expression with numerator
            # expanded, but with combined exponential functions (at
            # least in this setup all tests pass).
            eq = [Eq(f(x), _expand(t)) for t in eqsol]

        # special simplification of the lhs.
        if hint.startswith("1st_homogeneous_coeff"):
            for j, eqi in enumerate(eq):
                newi = logcombine(eqi, force=True)
                if isinstance(newi.lhs, log) and newi.rhs == 0:
                    newi = Eq(newi.lhs.args[0]/C1, C1)
                eq[j] = newi

    # We cleaned up the constants before solving to help the solve engine with
    # a simpler expression, but the solved expression could have introduced
    # things like -C1, so rerun constantsimp() one last time before returning.
    for i, eqi in enumerate(eq):
        eq[i] = constantsimp(eqi, constants)
        eq[i] = constant_renumber(eq[i], ode.free_symbols)

    # If there is only 1 solution, return it;
    # otherwise return the list of solutions.
    if len(eq) == 1:
        eq = eq[0]
    return eq


def ode_sol_simplicity(sol, func, trysolving=True):
    r"""
    Returns an extended integer representing how simple a solution to an ODE
    is.

    The following things are considered, in order from most simple to least:

    - ``sol`` is solved for ``func``.
    - ``sol`` is not solved for ``func``, but can be if passed to solve (e.g.,
      a solution returned by ``dsolve(ode, func, simplify=False``).
    - If ``sol`` is not solved for ``func``, then base the result on the
      length of ``sol``, as computed by ``len(str(sol))``.
    - If ``sol`` has any unevaluated :py:class:`~sympy.integrals.integrals.Integral`\s,
      this will automatically be considered less simple than any of the above.

    This function returns an integer such that if solution A is simpler than
    solution B by above metric, then ``ode_sol_simplicity(sola, func) <
    ode_sol_simplicity(solb, func)``.

    Currently, the following are the numbers returned, but if the heuristic is
    ever improved, this may change.  Only the ordering is guaranteed.

    +----------------------------------------------+-------------------+
    | Simplicity                                   | Return            |
    +==============================================+===================+
    | ``sol`` solved for ``func``                  | ``-2``            |
    +----------------------------------------------+-------------------+
    | ``sol`` not solved for ``func`` but can be   | ``-1``            |
    +----------------------------------------------+-------------------+
    | ``sol`` is not solved nor solvable for       | ``len(str(sol))`` |
    | ``func``                                     |                   |
    +----------------------------------------------+-------------------+
    | ``sol`` contains an                          | ``oo``            |
    | :obj:`~sympy.integrals.integrals.Integral`   |                   |
    +----------------------------------------------+-------------------+

    ``oo`` here means the SymPy infinity, which should compare greater than
    any integer.

    If you already know :py:meth:`~sympy.solvers.solvers.solve` cannot solve
    ``sol``, you can use ``trysolving=False`` to skip that step, which is the
    only potentially slow step.  For example,
    :py:meth:`~sympy.solvers.ode.dsolve` with the ``simplify=False`` flag
    should do this.

    If ``sol`` is a list of solutions, if the worst solution in the list
    returns ``oo`` it returns that, otherwise it returns ``len(str(sol))``,
    that is, the length of the string representation of the whole list.

    Examples
    ========

    This function is designed to be passed to ``min`` as the key argument,
    such as ``min(listofsolutions, key=lambda i: ode_sol_simplicity(i,
    f(x)))``.

    >>> from sympy import symbols, Function, Eq, tan, Integral
    >>> from sympy.solvers.ode.ode import ode_sol_simplicity
    >>> x, C1, C2 = symbols('x, C1, C2')
    >>> f = Function('f')

    >>> ode_sol_simplicity(Eq(f(x), C1*x**2), f(x))
    -2
    >>> ode_sol_simplicity(Eq(x**2 + f(x), C1), f(x))
    -1
    >>> ode_sol_simplicity(Eq(f(x), C1*Integral(2*x, x)), f(x))
    oo
    >>> eq1 = Eq(f(x)/tan(f(x)/(2*x)), C1)
    >>> eq2 = Eq(f(x)/tan(f(x)/(2*x) + f(x)), C2)
    >>> [ode_sol_simplicity(eq, f(x)) for eq in [eq1, eq2]]
    [28, 35]
    >>> min([eq1, eq2], key=lambda i: ode_sol_simplicity(i, f(x)))
    Eq(f(x)/tan(f(x)/(2*x)), C1)

    """
    # TODO: if two solutions are solved for f(x), we still want to be
    # able to get the simpler of the two

    # See the docstring for the coercion rules.  We check easier (faster)
    # things here first, to save time.

    if iterable(sol):
        # See if there are Integrals
        for i in sol:
            if ode_sol_simplicity(i, func, trysolving=trysolving) == oo:
                return oo

        return len(str(sol))

    if sol.has(Integral):
        return oo

    # Next, try to solve for func.  This code will change slightly when CRootOf
    # is implemented in solve().  Probably a CRootOf solution should fall
    # somewhere between a normal solution and an unsolvable expression.

    # First, see if they are already solved
    if sol.lhs == func and not sol.rhs.has(func) or \
            sol.rhs == func and not sol.lhs.has(func):
        return -2
    # We are not so lucky, try solving manually
    if trysolving:
        try:
            sols = solve(sol, func)
            if not sols:
                raise NotImplementedError
        except NotImplementedError:
            pass
        else:
            return -1

    # Finally, a naive computation based on the length of the string version
    # of the expression.  This may favor combined fractions because they
    # will not have duplicate denominators, and may slightly favor expressions
    # with fewer additions and subtractions, as those are separated by spaces
    # by the printer.

    # Additional ideas for simplicity heuristics are welcome, like maybe
    # checking if a equation has a larger domain, or if constantsimp has
    # introduced arbitrary constants numbered higher than the order of a
    # given ODE that sol is a solution of.
    return len(str(sol))


def _extract_funcs(eqs):
    funcs = []
    for eq in eqs:
        derivs = [node for node in preorder_traversal(eq) if isinstance(node, Derivative)]
        func = []
        for d in derivs:
            func += list(d.atoms(AppliedUndef))
        for func_ in func:
            funcs.append(func_)
    funcs = list(uniq(funcs))

    return funcs


def _get_constant_subexpressions(expr, Cs):
    Cs = set(Cs)
    Ces = []
    def _recursive_walk(expr):
        expr_syms = expr.free_symbols
        if expr_syms and expr_syms.issubset(Cs):
            Ces.append(expr)
        else:
            if expr.func == exp:
                expr = expr.expand(mul=True)
            if expr.func in (Add, Mul):
                d = sift(expr.args, lambda i : i.free_symbols.issubset(Cs))
                if len(d[True]) > 1:
                    x = expr.func(*d[True])
                    if not x.is_number:
                        Ces.append(x)
            elif isinstance(expr, Integral):
                if expr.free_symbols.issubset(Cs) and \
                            all(len(x) == 3 for x in expr.limits):
                    Ces.append(expr)
            for i in expr.args:
                _recursive_walk(i)
        return
    _recursive_walk(expr)
    return Ces

def __remove_linear_redundancies(expr, Cs):
    cnts = {i: expr.count(i) for i in Cs}
    Cs = [i for i in Cs if cnts[i] > 0]

    def _linear(expr):
        if isinstance(expr, Add):
            xs = [i for i in Cs if expr.count(i)==cnts[i] \
                and 0 == expr.diff(i, 2)]
            d = {}
            for x in xs:
                y = expr.diff(x)
                if y not in d:
                    d[y]=[]
                d[y].append(x)
            for y in d:
                if len(d[y]) > 1:
                    d[y].sort(key=str)
                    for x in d[y][1:]:
                        expr = expr.subs(x, 0)
        return expr

    def _recursive_walk(expr):
        if len(expr.args) != 0:
            expr = expr.func(*[_recursive_walk(i) for i in expr.args])
        expr = _linear(expr)
        return expr

    if isinstance(expr, Equality):
        lhs, rhs = [_recursive_walk(i) for i in expr.args]
        f = lambda i: isinstance(i, Number) or i in Cs
        if isinstance(lhs, Symbol) and lhs in Cs:
            rhs, lhs = lhs, rhs
        if lhs.func in (Add, Symbol) and rhs.func in (Add, Symbol):
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            drhs = sift([rhs] if isinstance(rhs, AtomicExpr) else rhs.args, f)
            for i in [True, False]:
                for hs in [dlhs, drhs]:
                    if i not in hs:
                        hs[i] = [0]
            # this calculation can be simplified
            lhs = Add(*dlhs[False]) - Add(*drhs[False])
            rhs = Add(*drhs[True]) - Add(*dlhs[True])
        elif lhs.func in (Mul, Symbol) and rhs.func in (Mul, Symbol):
            dlhs = sift([lhs] if isinstance(lhs, AtomicExpr) else lhs.args, f)
            if True in dlhs:
                if False not in dlhs:
                    dlhs[False] = [1]
                lhs = Mul(*dlhs[False])
                rhs = rhs/Mul(*dlhs[True])
        return Eq(lhs, rhs)
    else:
        return _recursive_walk(expr)

@vectorize(0)
def constantsimp(expr, constants):
    r"""
    Simplifies an expression with arbitrary constants in it.

    This function is written specifically to work with
    :py:meth:`~sympy.solvers.ode.dsolve`, and is not intended for general use.

    Simplification is done by "absorbing" the arbitrary constants into other
    arbitrary constants, numbers, and symbols that they are not independent
    of.

    The symbols must all have the same name with numbers after it, for
    example, ``C1``, ``C2``, ``C3``.  The ``symbolname`` here would be
    '``C``', the ``startnumber`` would be 1, and the ``endnumber`` would be 3.
    If the arbitrary constants are independent of the variable ``x``, then the
    independent symbol would be ``x``.  There is no need to specify the
    dependent function, such as ``f(x)``, because it already has the
    independent symbol, ``x``, in it.

    Because terms are "absorbed" into arbitrary constants and because
    constants are renumbered after simplifying, the arbitrary constants in
    expr are not necessarily equal to the ones of the same name in the
    returned result.

    If two or more arbitrary constants are added, multiplied, or raised to the
    power of each other, they are first absorbed together into a single
    arbitrary constant.  Then the new constant is combined into other terms if
    necessary.

    Absorption of constants is done with limited assistance:

    1. terms of :py:class:`~sympy.core.add.Add`\s are collected to try join
       constants so `e^x (C_1 \cos(x) + C_2 \cos(x))` will simplify to `e^x
       C_1 \cos(x)`;

    2. powers with exponents that are :py:class:`~sympy.core.add.Add`\s are
       expanded so `e^{C_1 + x}` will be simplified to `C_1 e^x`.

    Use :py:meth:`~sympy.solvers.ode.ode.constant_renumber` to renumber constants
    after simplification or else arbitrary numbers on constants may appear,
    e.g. `C_1 + C_3 x`.

    In rare cases, a single constant can be "simplified" into two constants.
    Every differential equation solution should have as many arbitrary
    constants as the order of the differential equation.  The result here will
    be technically correct, but it may, for example, have `C_1` and `C_2` in
    an expression, when `C_1` is actually equal to `C_2`.  Use your discretion
    in such situations, and also take advantage of the ability to use hints in
    :py:meth:`~sympy.solvers.ode.dsolve`.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.ode.ode import constantsimp
    >>> C1, C2, C3, x, y = symbols('C1, C2, C3, x, y')
    >>> constantsimp(2*C1*x, {C1, C2, C3})
    C1*x
    >>> constantsimp(C1 + 2 + x, {C1, C2, C3})
    C1 + x
    >>> constantsimp(C1*C2 + 2 + C2 + C3*x, {C1, C2, C3})
    C1 + C3*x

    """
    # This function works recursively.  The idea is that, for Mul,
    # Add, Pow, and Function, if the class has a constant in it, then
    # we can simplify it, which we do by recursing down and
    # simplifying up.  Otherwise, we can skip that part of the
    # expression.

    Cs = constants

    orig_expr = expr

    constant_subexprs = _get_constant_subexpressions(expr, Cs)
    for xe in constant_subexprs:
        xes = list(xe.free_symbols)
        if not xes:
            continue
        if all(expr.count(c) == xe.count(c) for c in xes):
            xes.sort(key=str)
            expr = expr.subs(xe, xes[0])

    # try to perform common sub-expression elimination of constant terms
    try:
        commons, rexpr = cse(expr)
        commons.reverse()
        rexpr = rexpr[0]
        for s in commons:
            cs = list(s[1].atoms(Symbol))
            if len(cs) == 1 and cs[0] in Cs and \
                cs[0] not in rexpr.atoms(Symbol) and \
                not any(cs[0] in ex for ex in commons if ex != s):
                rexpr = rexpr.subs(s[0], cs[0])
            else:
                rexpr = rexpr.subs(*s)
        expr = rexpr
    except IndexError:
        pass
    expr = __remove_linear_redundancies(expr, Cs)

    def _conditional_term_factoring(expr):
        new_expr = terms_gcd(expr, clear=False, deep=True, expand=False)

        # we do not want to factor exponentials, so handle this separately
        if new_expr.is_Mul:
            infac = False
            asfac = False
            for m in new_expr.args:
                if isinstance(m, exp):
                    asfac = True
                elif m.is_Add:
                    infac = any(isinstance(fi, exp) for t in m.args
                        for fi in Mul.make_args(t))
                if asfac and infac:
                    new_expr = expr
                    break
        return new_expr

    expr = _conditional_term_factoring(expr)

    # call recursively if more simplification is possible
    if orig_expr != expr:
        return constantsimp(expr, Cs)
    return expr


def constant_renumber(expr, variables=None, newconstants=None):
    r"""
    Renumber arbitrary constants in ``expr`` to use the symbol names as given
    in ``newconstants``. In the process, this reorders expression terms in a
    standard way.

    If ``newconstants`` is not provided then the new constant names will be
    ``C1``, ``C2`` etc. Otherwise ``newconstants`` should be an iterable
    giving the new symbols to use for the constants in order.

    The ``variables`` argument is a list of non-constant symbols. All other
    free symbols found in ``expr`` are assumed to be constants and will be
    renumbered. If ``variables`` is not given then any numbered symbol
    beginning with ``C`` (e.g. ``C1``) is assumed to be a constant.

    Symbols are renumbered based on ``.sort_key()``, so they should be
    numbered roughly in the order that they appear in the final, printed
    expression.  Note that this ordering is based in part on hashes, so it can
    produce different results on different machines.

    The structure of this function is very similar to that of
    :py:meth:`~sympy.solvers.ode.constantsimp`.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.ode.ode import constant_renumber
    >>> x, C1, C2, C3 = symbols('x,C1:4')
    >>> expr = C3 + C2*x + C1*x**2
    >>> expr
    C1*x**2  + C2*x + C3
    >>> constant_renumber(expr)
    C1 + C2*x + C3*x**2

    The ``variables`` argument specifies which are constants so that the
    other symbols will not be renumbered:

    >>> constant_renumber(expr, [C1, x])
    C1*x**2  + C2 + C3*x

    The ``newconstants`` argument is used to specify what symbols to use when
    replacing the constants:

    >>> constant_renumber(expr, [x], newconstants=symbols('E1:4'))
    E1 + E2*x + E3*x**2

    """

    # System of expressions
    if isinstance(expr, (set, list, tuple)):
        return type(expr)(constant_renumber(Tuple(*expr),
                        variables=variables, newconstants=newconstants))

    # Symbols in solution but not ODE are constants
    if variables is not None:
        variables = set(variables)
        free_symbols = expr.free_symbols
        constantsymbols = list(free_symbols - variables)
    # Any Cn is a constant...
    else:
        variables = set()
        isconstant = lambda s: s.startswith('C') and s[1:].isdigit()
        constantsymbols = [sym for sym in expr.free_symbols if isconstant(sym.name)]

    # Find new constants checking that they aren't already in the ODE
    if newconstants is None:
        iter_constants = numbered_symbols(start=1, prefix='C', exclude=variables)
    else:
        iter_constants = (sym for sym in newconstants if sym not in variables)

    constants_found = []

    # make a mapping to send all constantsymbols to S.One and use
    # that to make sure that term ordering is not dependent on
    # the indexed value of C
    C_1 = [(ci, S.One) for ci in constantsymbols]
    sort_key=lambda arg: default_sort_key(arg.subs(C_1))

    def _constant_renumber(expr):
        r"""
        We need to have an internal recursive function
        """

        # For system of expressions
        if isinstance(expr, Tuple):
            renumbered = [_constant_renumber(e) for e in expr]
            return Tuple(*renumbered)

        if isinstance(expr, Equality):
            return Eq(
                _constant_renumber(expr.lhs),
                _constant_renumber(expr.rhs))

        if type(expr) not in (Mul, Add, Pow) and not expr.is_Function and \
                not expr.has(*constantsymbols):
            # Base case, as above.  Hope there aren't constants inside
            # of some other class, because they won't be renumbered.
            return expr
        elif expr.is_Piecewise:
            return expr
        elif expr in constantsymbols:
            if expr not in constants_found:
                constants_found.append(expr)
            return expr
        elif expr.is_Function or expr.is_Pow:
            return expr.func(
                *[_constant_renumber(x) for x in expr.args])
        else:
            sortedargs = list(expr.args)
            sortedargs.sort(key=sort_key)
            return expr.func(*[_constant_renumber(x) for x in sortedargs])
    expr = _constant_renumber(expr)

    # Don't renumber symbols present in the ODE.
    constants_found = [c for c in constants_found if c not in variables]

    # Renumbering happens here
    subs_dict = dict(zip(constants_found, iter_constants))
    expr = expr.subs(subs_dict, simultaneous=True)

    return expr


def _handle_Integral(expr, func, hint):
    r"""
    Converts a solution with Integrals in it into an actual solution.

    For most hints, this simply runs ``expr.doit()``.

    """
    if hint == "nth_linear_constant_coeff_homogeneous":
        sol = expr
    elif not hint.endswith("_Integral"):
        sol = expr.doit()
    else:
        sol = expr
    return sol


# XXX: Should this function maybe go somewhere else?


def homogeneous_order(eq, *symbols):
    r"""
    Returns the order `n` if `g` is homogeneous and ``None`` if it is not
    homogeneous.

    Determines if a function is homogeneous and if so of what order.  A
    function `f(x, y, \cdots)` is homogeneous of order `n` if `f(t x, t y,
    \cdots) = t^n f(x, y, \cdots)`.

    If the function is of two variables, `F(x, y)`, then `f` being homogeneous
    of any order is equivalent to being able to rewrite `F(x, y)` as `G(x/y)`
    or `H(y/x)`.  This fact is used to solve 1st order ordinary differential
    equations whose coefficients are homogeneous of the same order (see the
    docstrings of
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsDepDivIndep` and
    :obj:`~sympy.solvers.ode.single.HomogeneousCoeffSubsIndepDivDep`).

    Symbols can be functions, but every argument of the function must be a
    symbol, and the arguments of the function that appear in the expression
    must match those given in the list of symbols.  If a declared function
    appears with different arguments than given in the list of symbols,
    ``None`` is returned.

    Examples
    ========

    >>> from sympy import Function, homogeneous_order, sqrt
    >>> from sympy.abc import x, y
    >>> f = Function('f')
    >>> homogeneous_order(f(x), f(x)) is None
    True
    >>> homogeneous_order(f(x,y), f(y, x), x, y) is None
    True
    >>> homogeneous_order(f(x), f(x), x)
    1
    >>> homogeneous_order(x**2*f(x)/sqrt(x**2+f(x)**2), x, f(x))
    2
    >>> homogeneous_order(x**2+f(x), x, f(x)) is None
    True

    """

    if not symbols:
        raise ValueError("homogeneous_order: no symbols were given.")
    symset = set(symbols)
    eq = sympify(eq)

    # The following are not supported
    if eq.has(Order, Derivative):
        return None

    # These are all constants
    if (eq.is_Number or
        eq.is_NumberSymbol or
        eq.is_number
            ):
        return S.Zero

    # Replace all functions with dummy variables
    dum = numbered_symbols(prefix='d', cls=Dummy)
    newsyms = set()
    for i in [j for j in symset if getattr(j, 'is_Function')]:
        iargs = set(i.args)
        if iargs.difference(symset):
            return None
        else:
            dummyvar = next(dum)
            eq = eq.subs(i, dummyvar)
            symset.remove(i)
            newsyms.add(dummyvar)
    symset.update(newsyms)

    if not eq.free_symbols & symset:
        return None

    # assuming order of a nested function can only be equal to zero
    if isinstance(eq, Function):
        return None if homogeneous_order(
            eq.args[0], *tuple(symset)) != 0 else S.Zero

    # make the replacement of x with x*t and see if t can be factored out
    t = Dummy('t', positive=True)  # It is sufficient that t > 0
    eqs = separatevars(eq.subs([(i, t*i) for i in symset]), [t], dict=True)[t]
    if eqs is S.One:
        return S.Zero  # there was no term with only t
    i, d = eqs.as_independent(t, as_Add=False)
    b, e = d.as_base_exp()
    if b == t:
        return e


def ode_2nd_power_series_ordinary(eq, func, order, match):
    r"""
    Gives a power series solution to a second order homogeneous differential
    equation with polynomial coefficients at an ordinary point. A homogeneous
    differential equation is of the form

    .. math :: P(x)\frac{d^2y}{dx^2} + Q(x)\frac{dy}{dx} + R(x) y(x) = 0

    For simplicity it is assumed that `P(x)`, `Q(x)` and `R(x)` are polynomials,
    it is sufficient that `\frac{Q(x)}{P(x)}` and `\frac{R(x)}{P(x)}` exists at
    `x_{0}`. A recurrence relation is obtained by substituting `y` as `\sum_{n=0}^\infty a_{n}x^{n}`,
    in the differential equation, and equating the nth term. Using this relation
    various terms can be generated.


    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = f(x).diff(x, 2) + f(x)
    >>> pprint(dsolve(eq, hint='2nd_power_series_ordinary'))
              / 4    2    \        /     2\
              |x    x     |        |    x |    / 6\
    f(x) = C2*|-- - -- + 1| + C1*x*|1 - --| + O\x /
              \24   2     /        \    6 /


    References
    ==========
    - https://tutorial.math.lamar.edu/Classes/DE/SeriesSolutions.aspx
    - George E. Simmons, "Differential Equations with Applications and
      Historical Notes", p.p 176 - 184

    """
    x = func.args[0]
    f = func.func
    C0, C1 = get_numbered_constants(eq, num=2)
    n = Dummy("n", integer=True)
    s = Wild("s")
    k = Wild("k", exclude=[x])
    x0 = match['x0']
    terms = match['terms']
    p = match[match['a3']]
    q = match[match['b3']]
    r = match[match['c3']]
    seriesdict = {}
    recurr = Function("r")

    # Generating the recurrence relation which works this way:
    # for the second order term the summation begins at n = 2. The coefficients
    # p is multiplied with an*(n - 1)*(n - 2)*x**n-2 and a substitution is made such that
    # the exponent of x becomes n.
    # For example, if p is x, then the second degree recurrence term is
    # an*(n - 1)*(n - 2)*x**n-1, substituting (n - 1) as n, it transforms to
    # an+1*n*(n - 1)*x**n.
    # A similar process is done with the first order and zeroth order term.

    coefflist = [(recurr(n), r), (n*recurr(n), q), (n*(n - 1)*recurr(n), p)]
    for index, coeff in enumerate(coefflist):
        if coeff[1]:
            f2 = powsimp(expand((coeff[1]*(x - x0)**(n - index)).subs(x, x + x0)))
            if f2.is_Add:
                addargs = f2.args
            else:
                addargs = [f2]
            for arg in addargs:
                powm = arg.match(s*x**k)
                term = coeff[0]*powm[s]
                if not powm[k].is_Symbol:
                    term = term.subs(n, n - powm[k].as_independent(n)[0])
                startind = powm[k].subs(n, index)
                # Seeing if the startterm can be reduced further.
                # If it vanishes for n lesser than startind, it is
                # equal to summation from n.
                if startind:
                    for i in reversed(range(startind)):
                        if not term.subs(n, i):
                            seriesdict[term] = i
                        else:
                            seriesdict[term] = i + 1
                            break
                else:
                    seriesdict[term] = S.Zero

    # Stripping of terms so that the sum starts with the same number.
    teq = S.Zero
    suminit = seriesdict.values()
    rkeys = seriesdict.keys()
    req = Add(*rkeys)
    if any(suminit):
        maxval = max(suminit)
        for term in seriesdict:
            val = seriesdict[term]
            if val != maxval:
                for i in range(val, maxval):
                    teq += term.subs(n, val)

    finaldict = {}
    if teq:
        fargs = teq.atoms(AppliedUndef)
        if len(fargs) == 1:
            finaldict[fargs.pop()] = 0
        else:
            maxf = max(fargs, key = lambda x: x.args[0])
            sol = solve(teq, maxf)
            if isinstance(sol, list):
                sol = sol[0]
            finaldict[maxf] = sol

    # Finding the recurrence relation in terms of the largest term.
    fargs = req.atoms(AppliedUndef)
    maxf = max(fargs, key = lambda x: x.args[0])
    minf = min(fargs, key = lambda x: x.args[0])
    if minf.args[0].is_Symbol:
        startiter = 0
    else:
        startiter = -minf.args[0].as_independent(n)[0]
    lhs = maxf
    rhs =  solve(req, maxf)
    if isinstance(rhs, list):
        rhs = rhs[0]

    # Checking how many values are already present
    tcounter = len([t for t in finaldict.values() if t])

    for _ in range(tcounter, terms - 3):  # Assuming c0 and c1 to be arbitrary
        check = rhs.subs(n, startiter)
        nlhs = lhs.subs(n, startiter)
        nrhs = check.subs(finaldict)
        finaldict[nlhs] = nrhs
        startiter += 1

    # Post processing
    series = C0 + C1*(x - x0)
    for term in finaldict:
        if finaldict[term]:
            fact = term.args[0]
            series += (finaldict[term].subs([(recurr(0), C0), (recurr(1), C1)])*(
                x - x0)**fact)
    series = collect(expand_mul(series), [C0, C1]) + Order(x**terms)
    return Eq(f(x), series)


def ode_2nd_power_series_regular(eq, func, order, match):
    r"""
    Gives a power series solution to a second order homogeneous differential
    equation with polynomial coefficients at a regular point. A second order
    homogeneous differential equation is of the form

    .. math :: P(x)\frac{d^2y}{dx^2} + Q(x)\frac{dy}{dx} + R(x) y(x) = 0

    A point is said to regular singular at `x0` if `x - x0\frac{Q(x)}{P(x)}`
    and `(x - x0)^{2}\frac{R(x)}{P(x)}` are analytic at `x0`. For simplicity
    `P(x)`, `Q(x)` and `R(x)` are assumed to be polynomials. The algorithm for
    finding the power series solutions is:

    1.  Try expressing `(x - x0)P(x)` and `((x - x0)^{2})Q(x)` as power series
        solutions about x0. Find `p0` and `q0` which are the constants of the
        power series expansions.
    2.  Solve the indicial equation `f(m) = m(m - 1) + m*p0 + q0`, to obtain the
        roots `m1` and `m2` of the indicial equation.
    3.  If `m1 - m2` is a non integer there exists two series solutions. If
        `m1 = m2`, there exists only one solution. If `m1 - m2` is an integer,
        then the existence of one solution is confirmed. The other solution may
        or may not exist.

    The power series solution is of the form `x^{m}\sum_{n=0}^\infty a_{n}x^{n}`. The
    coefficients are determined by the following recurrence relation.
    `a_{n} = -\frac{\sum_{k=0}^{n-1} q_{n-k} + (m + k)p_{n-k}}{f(m + n)}`. For the case
    in which `m1 - m2` is an integer, it can be seen from the recurrence relation
    that for the lower root `m`, when `n` equals the difference of both the
    roots, the denominator becomes zero. So if the numerator is not equal to zero,
    a second series solution exists.


    Examples
    ========

    >>> from sympy import dsolve, Function, pprint
    >>> from sympy.abc import x
    >>> f = Function("f")
    >>> eq = x*(f(x).diff(x, 2)) + 2*(f(x).diff(x)) + x*f(x)
    >>> pprint(dsolve(eq, hint='2nd_power_series_regular'))
                                  /   6     4    2    \
                                  |  x     x    x     |
              / 4     2    \   C1*|- --- + -- - -- + 1|
              |x     x     |      \  720   24   2     /    / 6\
    f(x) = C2*|--- - -- + 1| + ------------------------ + O\x /
              \120   6     /              x


    References
    ==========
    - George E. Simmons, "Differential Equations with Applications and
      Historical Notes", p.p 176 - 184

    """
    x = func.args[0]
    f = func.func
    C0, C1 = get_numbered_constants(eq, num=2)
    m = Dummy("m")  # for solving the indicial equation
    x0 = match['x0']
    terms = match['terms']
    p = match['p']
    q = match['q']

    # Generating the indicial equation
    indicial = []
    for term in [p, q]:
        if not term.has(x):
            indicial.append(term)
        else:
            term = series(term, x=x, n=1, x0=x0)
            if isinstance(term, Order):
                indicial.append(S.Zero)
            else:
                for arg in term.args:
                    if not arg.has(x):
                        indicial.append(arg)
                        break

    p0, q0 = indicial
    sollist = solve(m*(m - 1) + m*p0 + q0, m)
    if sollist and isinstance(sollist, list) and all(
        sol.is_real for sol in sollist):
        serdict1 = {}
        serdict2 = {}
        if len(sollist) == 1:
            # Only one series solution exists in this case.
            m1 = m2 = sollist.pop()
            if terms-m1-1 <= 0:
                return Eq(f(x), Order(terms))
            serdict1 = _frobenius(terms-m1-1, m1, p0, q0, p, q, x0, x, C0)

        else:
            m1 = sollist[0]
            m2 = sollist[1]
            if m1 < m2:
                m1, m2 = m2, m1
            # Irrespective of whether m1 - m2 is an integer or not, one
            # Frobenius series solution exists.
            serdict1 = _frobenius(terms-m1-1, m1, p0, q0, p, q, x0, x, C0)
            if not (m1 - m2).is_integer:
                # Second frobenius series solution exists.
                serdict2 = _frobenius(terms-m2-1, m2, p0, q0, p, q, x0, x, C1)
            else:
                # Check if second frobenius series solution exists.
                serdict2 = _frobenius(terms-m2-1, m2, p0, q0, p, q, x0, x, C1, check=m1)

        if serdict1:
            finalseries1 = C0
            for key in serdict1:
                power = int(key.name[1:])
                finalseries1 += serdict1[key]*(x - x0)**power
            finalseries1 = (x - x0)**m1*finalseries1
            finalseries2 = S.Zero
            if serdict2:
                for key in serdict2:
                    power = int(key.name[1:])
                    finalseries2 += serdict2[key]*(x - x0)**power
                finalseries2 += C1
                finalseries2 = (x - x0)**m2*finalseries2
            return Eq(f(x), collect(finalseries1 + finalseries2,
                [C0, C1]) + Order(x**terms))


def _frobenius(n, m, p0, q0, p, q, x0, x, c, check=None):
    r"""
    Returns a dict with keys as coefficients and values as their values in terms of C0
    """
    n = int(n)
    # In cases where m1 - m2 is not an integer
    m2 = check

    d = Dummy("d")
    numsyms = numbered_symbols("C", start=0)
    numsyms = [next(numsyms) for i in range(n + 1)]
    serlist = []
    for ser in [p, q]:
        # Order term not present
        if ser.is_polynomial(x) and Poly(ser, x).degree() <= n:
            if x0:
                ser = ser.subs(x, x + x0)
            dict_ = Poly(ser, x).as_dict()
        # Order term present
        else:
            tseries = series(ser, x=x0, n=n+1)
            # Removing order
            dict_ = Poly(list(ordered(tseries.args))[: -1], x).as_dict()
        # Fill in with zeros, if coefficients are zero.
        for i in range(n + 1):
            if (i,) not in dict_:
                dict_[(i,)] = S.Zero
        serlist.append(dict_)

    pseries = serlist[0]
    qseries = serlist[1]
    indicial = d*(d - 1) + d*p0 + q0
    frobdict = {}
    for i in range(1, n + 1):
        num = c*(m*pseries[(i,)] + qseries[(i,)])
        for j in range(1, i):
            sym = Symbol("C" + str(j))
            num += frobdict[sym]*((m + j)*pseries[(i - j,)] + qseries[(i - j,)])

        # Checking for cases when m1 - m2 is an integer. If num equals zero
        # then a second Frobenius series solution cannot be found. If num is not zero
        # then set constant as zero and proceed.
        if m2 is not None and i == m2 - m:
            if num:
                return False
            else:
                frobdict[numsyms[i]] = S.Zero
        else:
            frobdict[numsyms[i]] = -num/(indicial.subs(d, m+i))

    return frobdict

def _remove_redundant_solutions(eq, solns, order, var):
    r"""
    Remove redundant solutions from the set of solutions.

    This function is needed because otherwise dsolve can return
    redundant solutions. As an example consider:

        eq = Eq((f(x).diff(x, 2))*f(x).diff(x), 0)

    There are two ways to find solutions to eq. The first is to solve f(x).diff(x, 2) = 0
    leading to solution f(x)=C1 + C2*x. The second is to solve the equation f(x).diff(x) = 0
    leading to the solution f(x) = C1. In this particular case we then see
    that the second solution is a special case of the first and we do not
    want to return it.

    This does not always happen. If we have

        eq = Eq((f(x)**2-4)*(f(x).diff(x)-4), 0)

    then we get the algebraic solution f(x) = [-2, 2] and the integral solution
    f(x) = x + C1 and in this case the two solutions are not equivalent wrt
    initial conditions so both should be returned.
    """
    def is_special_case_of(soln1, soln2):
        return _is_special_case_of(soln1, soln2, eq, order, var)

    unique_solns = []
    for soln1 in solns:
        for soln2 in unique_solns.copy():
            if is_special_case_of(soln1, soln2):
                break
            elif is_special_case_of(soln2, soln1):
                unique_solns.remove(soln2)
        else:
            unique_solns.append(soln1)

    return unique_solns

def _is_special_case_of(soln1, soln2, eq, order, var):
    r"""
    True if soln1 is found to be a special case of soln2 wrt some value of the
    constants that appear in soln2. False otherwise.
    """
    # The solutions returned by dsolve may be given explicitly or implicitly.
    # We will equate the sol1=(soln1.rhs - soln1.lhs), sol2=(soln2.rhs - soln2.lhs)
    # of the two solutions.
    #
    # Since this is supposed to hold for all x it also holds for derivatives.
    # For an order n ode we should be able to differentiate
    # each solution n times to get n+1 equations.
    #
    # We then try to solve those n+1 equations for the integrations constants
    # in sol2. If we can find a solution that does not depend on x then it
    # means that some value of the constants in sol1 is a special case of
    # sol2 corresponding to a particular choice of the integration constants.

    # In case the solution is in implicit form we subtract the sides
    soln1 = soln1.rhs - soln1.lhs
    soln2 = soln2.rhs - soln2.lhs

    # Work for the series solution
    if soln1.has(Order) and soln2.has(Order):
        if soln1.getO() == soln2.getO():
            soln1 = soln1.removeO()
            soln2 = soln2.removeO()
        else:
            return False
    elif soln1.has(Order) or soln2.has(Order):
        return False

    constants1 = soln1.free_symbols.difference(eq.free_symbols)
    constants2 = soln2.free_symbols.difference(eq.free_symbols)

    constants1_new = get_numbered_constants(Tuple(soln1, soln2), len(constants1))
    if len(constants1) == 1:
        constants1_new = {constants1_new}
    for c_old, c_new in zip(constants1, constants1_new):
        soln1 = soln1.subs(c_old, c_new)

    # n equations for sol1 = sol2, sol1'=sol2', ...
    lhs = soln1
    rhs = soln2
    eqns = [Eq(lhs, rhs)]
    for n in range(1, order):
        lhs = lhs.diff(var)
        rhs = rhs.diff(var)
        eq = Eq(lhs, rhs)
        eqns.append(eq)

    # BooleanTrue/False awkwardly show up for trivial equations
    if any(isinstance(eq, BooleanFalse) for eq in eqns):
        return False
    eqns = [eq for eq in eqns if not isinstance(eq, BooleanTrue)]

    try:
        constant_solns = solve(eqns, constants2)
    except NotImplementedError:
        return False

    # Sometimes returns a dict and sometimes a list of dicts
    if isinstance(constant_solns, dict):
        constant_solns = [constant_solns]

    # after solving the issue 17418, maybe we don't need the following checksol code.
    for constant_soln in constant_solns:
        for eq in eqns:
            eq=eq.rhs-eq.lhs
            if checksol(eq, constant_soln) is not True:
                return False

    # If any solution gives all constants as expressions that don't depend on
    # x then there exists constants for soln2 that give soln1
    for constant_soln in constant_solns:
        if not any(c.has(var) for c in constant_soln.values()):
            return True

    return False


def ode_1st_power_series(eq, func, order, match):
    r"""
    The power series solution is a method which gives the Taylor series expansion
    to the solution of a differential equation.

    For a first order differential equation `\frac{dy}{dx} = h(x, y)`, a power
    series solution exists at a point `x = x_{0}` if `h(x, y)` is analytic at `x_{0}`.
    The solution is given by

    .. math:: y(x) = y(x_{0}) + \sum_{n = 1}^{\infty} \frac{F_{n}(x_{0},b)(x - x_{0})^n}{n!},

    where `y(x_{0}) = b` is the value of y at the initial value of `x_{0}`.
    To compute the values of the `F_{n}(x_{0},b)` the following algorithm is
    followed, until the required number of terms are generated.

    1. `F_1 = h(x_{0}, b)`
    2. `F_{n+1} = \frac{\partial F_{n}}{\partial x} + \frac{\partial F_{n}}{\partial y}F_{1}`

    Examples
    ========

    >>> from sympy import Function, pprint, exp, dsolve
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> eq = exp(x)*(f(x).diff(x)) - f(x)
    >>> pprint(dsolve(eq, hint='1st_power_series'))
                           3       4       5
                       C1*x    C1*x    C1*x     / 6\
    f(x) = C1 + C1*x - ----- + ----- + ----- + O\x /
                         6       24      60


    References
    ==========

    - Travis W. Walker, Analytic power series technique for solving first-order
      differential equations, p.p 17, 18

    """
    x = func.args[0]
    y = match['y']
    f = func.func
    h = -match[match['d']]/match[match['e']]
    point = match['f0']
    value = match['f0val']
    terms = match['terms']

    # First term
    F = h
    if not h:
        return Eq(f(x), value)

    # Initialization
    series = value
    if terms > 1:
        hc = h.subs({x: point, y: value})
        if hc.has(oo) or hc.has(nan) or hc.has(zoo):
            # Derivative does not exist, not analytic
            return Eq(f(x), oo)
        elif hc:
            series += hc*(x - point)

    for factcount in range(2, terms):
        Fnew = F.diff(x) + F.diff(y)*h
        Fnewc = Fnew.subs({x: point, y: value})
        # Same logic as above
        if Fnewc.has(oo) or Fnewc.has(nan) or Fnewc.has(-oo) or Fnewc.has(zoo):
            return Eq(f(x), oo)
        series += Fnewc*((x - point)**factcount)/factorial(factcount)
        F = Fnew
    series += Order(x**terms)
    return Eq(f(x), series)


def checkinfsol(eq, infinitesimals, func=None, order=None):
    r"""
    This function is used to check if the given infinitesimals are the
    actual infinitesimals of the given first order differential equation.
    This method is specific to the Lie Group Solver of ODEs.

    As of now, it simply checks, by substituting the infinitesimals in the
    partial differential equation.


    .. math:: \frac{\partial \eta}{\partial x} + \left(\frac{\partial \eta}{\partial y}
                - \frac{\partial \xi}{\partial x}\right)*h
                - \frac{\partial \xi}{\partial y}*h^{2}
                - \xi\frac{\partial h}{\partial x} - \eta\frac{\partial h}{\partial y} = 0


    where `\eta`, and `\xi` are the infinitesimals and `h(x,y) = \frac{dy}{dx}`

    The infinitesimals should be given in the form of a list of dicts
    ``[{xi(x, y): inf, eta(x, y): inf}]``, corresponding to the
    output of the function infinitesimals. It returns a list
    of values of the form ``[(True/False, sol)]`` where ``sol`` is the value
    obtained after substituting the infinitesimals in the PDE. If it
    is ``True``, then ``sol`` would be 0.

    """
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs
    if not func:
        eq, func = _preprocess(eq)
    variables = func.args
    if len(variables) != 1:
        raise ValueError("ODE's have only one independent variable")
    else:
        x = variables[0]
        if not order:
            order = ode_order(eq, func)
        if order != 1:
            raise NotImplementedError("Lie groups solver has been implemented "
            "only for first order differential equations")
        else:
            df = func.diff(x)
            a = Wild('a', exclude = [df])
            b = Wild('b', exclude = [df])
            match = collect(expand(eq), df).match(a*df + b)

            if match:
                h = -simplify(match[b]/match[a])
            else:
                try:
                    sol = solve(eq, df)
                except NotImplementedError:
                    raise NotImplementedError("Infinitesimals for the "
                        "first order ODE could not be found")
                else:
                    h = sol[0]  # Find infinitesimals for one solution

            y = Dummy('y')
            h = h.subs(func, y)
            xi = Function('xi')(x, y)
            eta = Function('eta')(x, y)
            dxi = Function('xi')(x, func)
            deta = Function('eta')(x, func)
            pde = (eta.diff(x) + (eta.diff(y) - xi.diff(x))*h -
                (xi.diff(y))*h**2 - xi*(h.diff(x)) - eta*(h.diff(y)))
            soltup = []
            for sol in infinitesimals:
                tsol = {xi: S(sol[dxi]).subs(func, y),
                    eta: S(sol[deta]).subs(func, y)}
                sol = simplify(pde.subs(tsol).doit())
                if sol:
                    soltup.append((False, sol.subs(y, func)))
                else:
                    soltup.append((True, 0))
            return soltup


def sysode_linear_2eq_order1(match_):
    x = match_['func'][0].func
    y = match_['func'][1].func
    func = match_['func']
    fc = match_['func_coeff']
    eq = match_['eq']
    r = {}
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    for i in range(2):
        eq[i] = Add(*[terms/fc[i,func[i],1] for terms in Add.make_args(eq[i])])

    # for equations Eq(a1*diff(x(t),t), a*x(t) + b*y(t) + k1)
    # and Eq(a2*diff(x(t),t), c*x(t) + d*y(t) + k2)
    r['a'] = -fc[0,x(t),0]/fc[0,x(t),1]
    r['c'] = -fc[1,x(t),0]/fc[1,y(t),1]
    r['b'] = -fc[0,y(t),0]/fc[0,x(t),1]
    r['d'] = -fc[1,y(t),0]/fc[1,y(t),1]
    forcing = [S.Zero,S.Zero]
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j
    if not (forcing[0].has(t) or forcing[1].has(t)):
        r['k1'] = forcing[0]
        r['k2'] = forcing[1]
    else:
        raise NotImplementedError("Only homogeneous problems are supported" +
                                  " (and constant inhomogeneity)")

    if match_['type_of_equation'] == 'type6':
        sol = _linear_2eq_order1_type6(x, y, t, r, eq)
    if match_['type_of_equation'] == 'type7':
        sol = _linear_2eq_order1_type7(x, y, t, r, eq)
    return sol

def _linear_2eq_order1_type6(x, y, t, r, eq):
    r"""
    The equations of this type of ode are .

    .. math:: x' = f(t) x + g(t) y

    .. math:: y' = a [f(t) + a h(t)] x + a [g(t) - h(t)] y

    This is solved by first multiplying the first equation by `-a` and adding
    it to the second equation to obtain

    .. math:: y' - a x' = -a h(t) (y - a x)

    Setting `U = y - ax` and integrating the equation we arrive at

    .. math:: y - ax = C_1 e^{-a \int h(t) \,dt}

    and on substituting the value of y in first equation give rise to first order ODEs. After solving for
    `x`, we can obtain `y` by substituting the value of `x` in second equation.

    """
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    p = 0
    q = 0
    p1 = cancel(r['c']/cancel(r['c']/r['d']).as_numer_denom()[0])
    p2 = cancel(r['a']/cancel(r['a']/r['b']).as_numer_denom()[0])
    for n, i in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q!=0 and n==0:
                if ((r['c']/j - r['a'])/(r['b'] - r['d']/j)) == j:
                    p = 1
                    s = j
                    break
            if q!=0 and n==1:
                if ((r['a']/j - r['c'])/(r['d'] - r['b']/j)) == j:
                    p = 2
                    s = j
                    break

    if p == 1:
        equ = diff(x(t),t) - r['a']*x(t) - r['b']*(s*x(t) + C1*exp(-s*Integral(r['b'] - r['d']/s, t)))
        hint1 = classify_ode(equ)[1]
        sol1 = dsolve(equ, hint=hint1+'_Integral').rhs
        sol2 = s*sol1 + C1*exp(-s*Integral(r['b'] - r['d']/s, t))
    elif p ==2:
        equ = diff(y(t),t) - r['c']*y(t) - r['d']*s*y(t) + C1*exp(-s*Integral(r['d'] - r['b']/s, t))
        hint1 = classify_ode(equ)[1]
        sol2 = dsolve(equ, hint=hint1+'_Integral').rhs
        sol1 = s*sol2 + C1*exp(-s*Integral(r['d'] - r['b']/s, t))
    return [Eq(x(t), sol1), Eq(y(t), sol2)]

def _linear_2eq_order1_type7(x, y, t, r, eq):
    r"""
    The equations of this type of ode are .

    .. math:: x' = f(t) x + g(t) y

    .. math:: y' = h(t) x + p(t) y

    Differentiating the first equation and substituting the value of `y`
    from second equation will give a second-order linear equation

    .. math:: g x'' - (fg + gp + g') x' + (fgp - g^{2} h + f g' - f' g) x = 0

    This above equation can be easily integrated if following conditions are satisfied.

    1. `fgp - g^{2} h + f g' - f' g = 0`

    2. `fgp - g^{2} h + f g' - f' g = ag, fg + gp + g' = bg`

    If first condition is satisfied then it is solved by current dsolve solver and in second case it becomes
    a constant coefficient differential equation which is also solved by current solver.

    Otherwise if the above condition fails then,
    a particular solution is assumed as `x = x_0(t)` and `y = y_0(t)`
    Then the general solution is expressed as

    .. math:: x = C_1 x_0(t) + C_2 x_0(t) \int \frac{g(t) F(t) P(t)}{x_0^{2}(t)} \,dt

    .. math:: y = C_1 y_0(t) + C_2 [\frac{F(t) P(t)}{x_0(t)} + y_0(t) \int \frac{g(t) F(t) P(t)}{x_0^{2}(t)} \,dt]

    where C1 and C2 are arbitrary constants and

    .. math:: F(t) = e^{\int f(t) \,dt}, P(t) = e^{\int p(t) \,dt}

    """
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    e1 = r['a']*r['b']*r['c'] - r['b']**2*r['c'] + r['a']*diff(r['b'],t) - diff(r['a'],t)*r['b']
    e2 = r['a']*r['c']*r['d'] - r['b']*r['c']**2 + diff(r['c'],t)*r['d'] - r['c']*diff(r['d'],t)
    m1 = r['a']*r['b'] + r['b']*r['d'] + diff(r['b'],t)
    m2 = r['a']*r['c'] + r['c']*r['d'] + diff(r['c'],t)
    if e1 == 0:
        sol1 = dsolve(r['b']*diff(x(t),t,t) - m1*diff(x(t),t)).rhs
        sol2 = dsolve(diff(y(t),t) - r['c']*sol1 - r['d']*y(t)).rhs
    elif e2 == 0:
        sol2 = dsolve(r['c']*diff(y(t),t,t) - m2*diff(y(t),t)).rhs
        sol1 = dsolve(diff(x(t),t) - r['a']*x(t) - r['b']*sol2).rhs
    elif not (e1/r['b']).has(t) and not (m1/r['b']).has(t):
        sol1 = dsolve(diff(x(t),t,t) - (m1/r['b'])*diff(x(t),t) - (e1/r['b'])*x(t)).rhs
        sol2 = dsolve(diff(y(t),t) - r['c']*sol1 - r['d']*y(t)).rhs
    elif not (e2/r['c']).has(t) and not (m2/r['c']).has(t):
        sol2 = dsolve(diff(y(t),t,t) - (m2/r['c'])*diff(y(t),t) - (e2/r['c'])*y(t)).rhs
        sol1 = dsolve(diff(x(t),t) - r['a']*x(t) - r['b']*sol2).rhs
    else:
        x0 = Function('x0')(t)    # x0 and y0 being particular solutions
        y0 = Function('y0')(t)
        F = exp(Integral(r['a'],t))
        P = exp(Integral(r['d'],t))
        sol1 = C1*x0 + C2*x0*Integral(r['b']*F*P/x0**2, t)
        sol2 = C1*y0 + C2*(F*P/x0 + y0*Integral(r['b']*F*P/x0**2, t))
    return [Eq(x(t), sol1), Eq(y(t), sol2)]


def sysode_nonlinear_2eq_order1(match_):
    func = match_['func']
    eq = match_['eq']
    fc = match_['func_coeff']
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    if match_['type_of_equation'] == 'type5':
        sol = _nonlinear_2eq_order1_type5(func, t, eq)
        return sol
    x = func[0].func
    y = func[1].func
    for i in range(2):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms/fc[i,func[i],1]
        eq[i] = eqs
    if match_['type_of_equation'] == 'type1':
        sol = _nonlinear_2eq_order1_type1(x, y, t, eq)
    elif match_['type_of_equation'] == 'type2':
        sol = _nonlinear_2eq_order1_type2(x, y, t, eq)
    elif match_['type_of_equation'] == 'type3':
        sol = _nonlinear_2eq_order1_type3(x, y, t, eq)
    elif match_['type_of_equation'] == 'type4':
        sol = _nonlinear_2eq_order1_type4(x, y, t, eq)
    return sol


def _nonlinear_2eq_order1_type1(x, y, t, eq):
    r"""
    Equations:

    .. math:: x' = x^n F(x,y)

    .. math:: y' = g(y) F(x,y)

    Solution:

    .. math:: x = \varphi(y), \int \frac{1}{g(y) F(\varphi(y),y)} \,dy = t + C_2

    where

    if `n \neq 1`

    .. math:: \varphi = [C_1 + (1-n) \int \frac{1}{g(y)} \,dy]^{\frac{1}{1-n}}

    if `n = 1`

    .. math:: \varphi = C_1 e^{\int \frac{1}{g(y)} \,dy}

    where `C_1` and `C_2` are arbitrary constants.

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    n = Wild('n', exclude=[x(t),y(t)])
    f = Wild('f')
    u, v = symbols('u, v')
    r = eq[0].match(diff(x(t),t) - x(t)**n*f)
    g = ((diff(y(t),t) - eq[1])/r[f]).subs(y(t),v)
    F = r[f].subs(x(t),u).subs(y(t),v)
    n = r[n]
    if n!=1:
        phi = (C1 + (1-n)*Integral(1/g, v))**(1/(1-n))
    else:
        phi = C1*exp(Integral(1/g, v))
    phi = phi.doit()
    sol2 = solve(Integral(1/(g*F.subs(u,phi)), v).doit() - t - C2, v)
    sol = []
    for sols in sol2:
        sol.append(Eq(x(t),phi.subs(v, sols)))
        sol.append(Eq(y(t), sols))
    return sol

def _nonlinear_2eq_order1_type2(x, y, t, eq):
    r"""
    Equations:

    .. math:: x' = e^{\lambda x} F(x,y)

    .. math:: y' = g(y) F(x,y)

    Solution:

    .. math:: x = \varphi(y), \int \frac{1}{g(y) F(\varphi(y),y)} \,dy = t + C_2

    where

    if `\lambda \neq 0`

    .. math:: \varphi = -\frac{1}{\lambda} log(C_1 - \lambda \int \frac{1}{g(y)} \,dy)

    if `\lambda = 0`

    .. math:: \varphi = C_1 + \int \frac{1}{g(y)} \,dy

    where `C_1` and `C_2` are arbitrary constants.

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    n = Wild('n', exclude=[x(t),y(t)])
    f = Wild('f')
    u, v = symbols('u, v')
    r = eq[0].match(diff(x(t),t) - exp(n*x(t))*f)
    g = ((diff(y(t),t) - eq[1])/r[f]).subs(y(t),v)
    F = r[f].subs(x(t),u).subs(y(t),v)
    n = r[n]
    if n:
        phi = -1/n*log(C1 - n*Integral(1/g, v))
    else:
        phi = C1 + Integral(1/g, v)
    phi = phi.doit()
    sol2 = solve(Integral(1/(g*F.subs(u,phi)), v).doit() - t - C2, v)
    sol = []
    for sols in sol2:
        sol.append(Eq(x(t),phi.subs(v, sols)))
        sol.append(Eq(y(t), sols))
    return sol

def _nonlinear_2eq_order1_type3(x, y, t, eq):
    r"""
    Autonomous system of general form

    .. math:: x' = F(x,y)

    .. math:: y' = G(x,y)

    Assuming `y = y(x, C_1)` where `C_1` is an arbitrary constant is the general
    solution of the first-order equation

    .. math:: F(x,y) y'_x = G(x,y)

    Then the general solution of the original system of equations has the form

    .. math:: \int \frac{1}{F(x,y(x,C_1))} \,dx = t + C_1

    """
    C1, C2, C3, C4 = get_numbered_constants(eq, num=4)
    v = Function('v')
    u = Symbol('u')
    f = Wild('f')
    g = Wild('g')
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    F = r1[f].subs(x(t), u).subs(y(t), v(u))
    G = r2[g].subs(x(t), u).subs(y(t), v(u))
    sol2r = dsolve(Eq(diff(v(u), u), G/F))
    if isinstance(sol2r, Equality):
        sol2r = [sol2r]
    for sol2s in sol2r:
        sol1 = solve(Integral(1/F.subs(v(u), sol2s.rhs), u).doit() - t - C2, u)
    sol = []
    for sols in sol1:
        sol.append(Eq(x(t), sols))
        sol.append(Eq(y(t), (sol2s.rhs).subs(u, sols)))
    return sol

def _nonlinear_2eq_order1_type4(x, y, t, eq):
    r"""
    Equation:

    .. math:: x' = f_1(x) g_1(y) \phi(x,y,t)

    .. math:: y' = f_2(x) g_2(y) \phi(x,y,t)

    First integral:

    .. math:: \int \frac{f_2(x)}{f_1(x)} \,dx - \int \frac{g_1(y)}{g_2(y)} \,dy = C

    where `C` is an arbitrary constant.

    On solving the first integral for `x` (resp., `y` ) and on substituting the
    resulting expression into either equation of the original solution, one
    arrives at a first-order equation for determining `y` (resp., `x` ).

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    u, v = symbols('u, v')
    U, V = symbols('U, V', cls=Function)
    f = Wild('f')
    g = Wild('g')
    f1 = Wild('f1', exclude=[v,t])
    f2 = Wild('f2', exclude=[v,t])
    g1 = Wild('g1', exclude=[u,t])
    g2 = Wild('g2', exclude=[u,t])
    r1 = eq[0].match(diff(x(t),t) - f)
    r2 = eq[1].match(diff(y(t),t) - g)
    num, den = (
        (r1[f].subs(x(t),u).subs(y(t),v))/
        (r2[g].subs(x(t),u).subs(y(t),v))).as_numer_denom()
    R1 = num.match(f1*g1)
    R2 = den.match(f2*g2)
    phi = (r1[f].subs(x(t),u).subs(y(t),v))/num
    F1 = R1[f1]; F2 = R2[f2]
    G1 = R1[g1]; G2 = R2[g2]
    sol1r = solve(Integral(F2/F1, u).doit() - Integral(G1/G2,v).doit() - C1, u)
    sol2r = solve(Integral(F2/F1, u).doit() - Integral(G1/G2,v).doit() - C1, v)
    sol = []
    for sols in sol1r:
        sol.append(Eq(y(t), dsolve(diff(V(t),t) - F2.subs(u,sols).subs(v,V(t))*G2.subs(v,V(t))*phi.subs(u,sols).subs(v,V(t))).rhs))
    for sols in sol2r:
        sol.append(Eq(x(t), dsolve(diff(U(t),t) - F1.subs(u,U(t))*G1.subs(v,sols).subs(u,U(t))*phi.subs(v,sols).subs(u,U(t))).rhs))
    return set(sol)

def _nonlinear_2eq_order1_type5(func, t, eq):
    r"""
    Clairaut system of ODEs

    .. math:: x = t x' + F(x',y')

    .. math:: y = t y' + G(x',y')

    The following are solutions of the system

    `(i)` straight lines:

    .. math:: x = C_1 t + F(C_1, C_2), y = C_2 t + G(C_1, C_2)

    where `C_1` and `C_2` are arbitrary constants;

    `(ii)` envelopes of the above lines;

    `(iii)` continuously differentiable lines made up from segments of the lines
    `(i)` and `(ii)`.

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    f = Wild('f')
    g = Wild('g')
    def check_type(x, y):
        r1 = eq[0].match(t*diff(x(t),t) - x(t) + f)
        r2 = eq[1].match(t*diff(y(t),t) - y(t) + g)
        if not (r1 and r2):
            r1 = eq[0].match(diff(x(t),t) - x(t)/t + f/t)
            r2 = eq[1].match(diff(y(t),t) - y(t)/t + g/t)
        if not (r1 and r2):
            r1 = (-eq[0]).match(t*diff(x(t),t) - x(t) + f)
            r2 = (-eq[1]).match(t*diff(y(t),t) - y(t) + g)
        if not (r1 and r2):
            r1 = (-eq[0]).match(diff(x(t),t) - x(t)/t + f/t)
            r2 = (-eq[1]).match(diff(y(t),t) - y(t)/t + g/t)
        return [r1, r2]
    for func_ in func:
        if isinstance(func_, list):
            x = func[0][0].func
            y = func[0][1].func
            [r1, r2] = check_type(x, y)
            if not (r1 and r2):
                [r1, r2] = check_type(y, x)
                x, y = y, x
    x1 = diff(x(t),t); y1 = diff(y(t),t)
    return {Eq(x(t), C1*t + r1[f].subs(x1,C1).subs(y1,C2)), Eq(y(t), C2*t + r2[g].subs(x1,C1).subs(y1,C2))}

def sysode_nonlinear_3eq_order1(match_):
    x = match_['func'][0].func
    y = match_['func'][1].func
    z = match_['func'][2].func
    eq = match_['eq']
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    if match_['type_of_equation'] == 'type1':
        sol = _nonlinear_3eq_order1_type1(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type2':
        sol = _nonlinear_3eq_order1_type2(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type3':
        sol = _nonlinear_3eq_order1_type3(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type4':
        sol = _nonlinear_3eq_order1_type4(x, y, z, t, eq)
    if match_['type_of_equation'] == 'type5':
        sol = _nonlinear_3eq_order1_type5(x, y, z, t, eq)
    return sol

def _nonlinear_3eq_order1_type1(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: a x' = (b - c) y z, \enspace b y' = (c - a) z x, \enspace c z' = (a - b) x y

    First Integrals:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2

    where `C_1` and `C_2` are arbitrary constants. On solving the integrals for `y` and
    `z` and on substituting the resulting expressions into the first equation of the
    system, we arrives at a separable first-order equation on `x`. Similarly doing that
    for other two equations, we will arrive at first order equation on `y` and `z` too.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0401.pdf

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    u, v, w = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    r = (diff(x(t),t) - eq[0]).match(p*y(t)*z(t))
    r.update((diff(y(t),t) - eq[1]).match(q*z(t)*x(t)))
    r.update((diff(z(t),t) - eq[2]).match(s*x(t)*y(t)))
    n1, d1 = r[p].as_numer_denom()
    n2, d2 = r[q].as_numer_denom()
    n3, d3 = r[s].as_numer_denom()
    val = solve([n1*u-d1*v+d1*w, d2*u+n2*v-d2*w, d3*u-d3*v-n3*w],[u,v])
    vals = [val[v], val[u]]
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])
    b = vals[0].subs(w, c)
    a = vals[1].subs(w, c)
    y_x = sqrt(((c*C1-C2) - a*(c-a)*x(t)**2)/(b*(c-b)))
    z_x = sqrt(((b*C1-C2) - a*(b-a)*x(t)**2)/(c*(b-c)))
    z_y = sqrt(((a*C1-C2) - b*(a-b)*y(t)**2)/(c*(a-c)))
    x_y = sqrt(((c*C1-C2) - b*(c-b)*y(t)**2)/(a*(c-a)))
    x_z = sqrt(((b*C1-C2) - c*(b-c)*z(t)**2)/(a*(b-a)))
    y_z = sqrt(((a*C1-C2) - c*(a-c)*z(t)**2)/(b*(a-b)))
    sol1 = dsolve(a*diff(x(t),t) - (b-c)*y_x*z_x)
    sol2 = dsolve(b*diff(y(t),t) - (c-a)*z_y*x_y)
    sol3 = dsolve(c*diff(z(t),t) - (a-b)*x_z*y_z)
    return [sol1, sol2, sol3]


def _nonlinear_3eq_order1_type2(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: a x' = (b - c) y z f(x, y, z, t)

    .. math:: b y' = (c - a) z x f(x, y, z, t)

    .. math:: c z' = (a - b) x y f(x, y, z, t)

    First Integrals:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    .. math:: a^{2} x^{2} + b^{2} y^{2} + c^{2} z^{2} = C_2

    where `C_1` and `C_2` are arbitrary constants. On solving the integrals for `y` and
    `z` and on substituting the resulting expressions into the first equation of the
    system, we arrives at a first-order differential equations on `x`. Similarly doing
    that for other two equations we will arrive at first order equation on `y` and `z`.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0402.pdf

    """
    C1, C2 = get_numbered_constants(eq, num=2)
    u, v, w = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')
    r1 = (diff(x(t),t) - eq[0]).match(y(t)*z(t)*f)
    r = collect_const(r1[f]).match(p*f)
    r.update(((diff(y(t),t) - eq[1])/r[f]).match(q*z(t)*x(t)))
    r.update(((diff(z(t),t) - eq[2])/r[f]).match(s*x(t)*y(t)))
    n1, d1 = r[p].as_numer_denom()
    n2, d2 = r[q].as_numer_denom()
    n3, d3 = r[s].as_numer_denom()
    val = solve([n1*u-d1*v+d1*w, d2*u+n2*v-d2*w, -d3*u+d3*v+n3*w],[u,v])
    vals = [val[v], val[u]]
    c = lcm(vals[0].as_numer_denom()[1], vals[1].as_numer_denom()[1])
    a = vals[0].subs(w, c)
    b = vals[1].subs(w, c)
    y_x = sqrt(((c*C1-C2) - a*(c-a)*x(t)**2)/(b*(c-b)))
    z_x = sqrt(((b*C1-C2) - a*(b-a)*x(t)**2)/(c*(b-c)))
    z_y = sqrt(((a*C1-C2) - b*(a-b)*y(t)**2)/(c*(a-c)))
    x_y = sqrt(((c*C1-C2) - b*(c-b)*y(t)**2)/(a*(c-a)))
    x_z = sqrt(((b*C1-C2) - c*(b-c)*z(t)**2)/(a*(b-a)))
    y_z = sqrt(((a*C1-C2) - c*(a-c)*z(t)**2)/(b*(a-b)))
    sol1 = dsolve(a*diff(x(t),t) - (b-c)*y_x*z_x*r[f])
    sol2 = dsolve(b*diff(y(t),t) - (c-a)*z_y*x_y*r[f])
    sol3 = dsolve(c*diff(z(t),t) - (a-b)*x_z*y_z*r[f])
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type3(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: x' = c F_2 - b F_3, \enspace y' = a F_3 - c F_1, \enspace z' = b F_1 - a F_2

    where `F_n = F_n(x, y, z, t)`.

    1. First Integral:

    .. math:: a x + b y + c z = C_1,

    where C is an arbitrary constant.

    2. If we assume function `F_n` to be independent of `t`,i.e, `F_n` = `F_n (x, y, z)`
    Then, on eliminating `t` and `z` from the first two equation of the system, one
    arrives at the first-order equation

    .. math:: \frac{dy}{dx} = \frac{a F_3 (x, y, z) - c F_1 (x, y, z)}{c F_2 (x, y, z) -
                b F_3 (x, y, z)}

    where `z = \frac{1}{c} (C_1 - a x - b y)`

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0404.pdf

    """
    C1 = get_numbered_constants(eq, num=1)
    u, v, w = symbols('u, v, w')
    fu, fv, fw = symbols('u, v, w', cls=Function)
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    r1 = (diff(x(t), t) - eq[0]).match(F2-F3)
    r = collect_const(r1[F2]).match(s*F2)
    r.update(collect_const(r1[F3]).match(q*F3))
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        r[F2], r[F3] = r[F3], r[F2]
        r[s], r[q] = -r[q], -r[s]
    r.update((diff(y(t), t) - eq[1]).match(p*r[F3] - r[s]*F1))
    a = r[p]; b = r[q]; c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t),v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t),v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t),v).subs(z(t), w)
    z_xy = (C1-a*u-b*v)/c
    y_zx = (C1-a*u-c*w)/b
    x_yz = (C1-b*v-c*w)/a
    y_x = dsolve(diff(fv(u),u) - ((a*F3-c*F1)/(c*F2-b*F3)).subs(w,z_xy).subs(v,fv(u))).rhs
    z_x = dsolve(diff(fw(u),u) - ((b*F1-a*F2)/(c*F2-b*F3)).subs(v,y_zx).subs(w,fw(u))).rhs
    z_y = dsolve(diff(fw(v),v) - ((b*F1-a*F2)/(a*F3-c*F1)).subs(u,x_yz).subs(w,fw(v))).rhs
    x_y = dsolve(diff(fu(v),v) - ((c*F2-b*F3)/(a*F3-c*F1)).subs(w,z_xy).subs(u,fu(v))).rhs
    y_z = dsolve(diff(fv(w),w) - ((a*F3-c*F1)/(b*F1-a*F2)).subs(u,x_yz).subs(v,fv(w))).rhs
    x_z = dsolve(diff(fu(w),w) - ((c*F2-b*F3)/(b*F1-a*F2)).subs(v,y_zx).subs(u,fu(w))).rhs
    sol1 = dsolve(diff(fu(t),t) - (c*F2 - b*F3).subs(v,y_x).subs(w,z_x).subs(u,fu(t))).rhs
    sol2 = dsolve(diff(fv(t),t) - (a*F3 - c*F1).subs(u,x_y).subs(w,z_y).subs(v,fv(t))).rhs
    sol3 = dsolve(diff(fw(t),t) - (b*F1 - a*F2).subs(u,x_z).subs(v,y_z).subs(w,fw(t))).rhs
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type4(x, y, z, t, eq):
    r"""
    Equations:

    .. math:: x' = c z F_2 - b y F_3, \enspace y' = a x F_3 - c z F_1, \enspace z' = b y F_1 - a x F_2

    where `F_n = F_n (x, y, z, t)`

    1. First integral:

    .. math:: a x^{2} + b y^{2} + c z^{2} = C_1

    where `C` is an arbitrary constant.

    2. Assuming the function `F_n` is independent of `t`: `F_n = F_n (x, y, z)`. Then on
    eliminating `t` and `z` from the first two equations of the system, one arrives at
    the first-order equation

    .. math:: \frac{dy}{dx} = \frac{a x F_3 (x, y, z) - c z F_1 (x, y, z)}
                {c z F_2 (x, y, z) - b y F_3 (x, y, z)}

    where `z = \pm \sqrt{\frac{1}{c} (C_1 - a x^{2} - b y^{2})}`

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0405.pdf

    """
    C1 = get_numbered_constants(eq, num=1)
    u, v, w = symbols('u, v, w')
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    r1 = eq[0].match(diff(x(t),t) - z(t)*F2 + y(t)*F3)
    r = collect_const(r1[F2]).match(s*F2)
    r.update(collect_const(r1[F3]).match(q*F3))
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        r[F2], r[F3] = r[F3], r[F2]
        r[s], r[q] = -r[q], -r[s]
    r.update((diff(y(t),t) - eq[1]).match(p*x(t)*r[F3] - r[s]*z(t)*F1))
    a = r[p]; b = r[q]; c = r[s]
    F1 = r[F1].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    F2 = r[F2].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    F3 = r[F3].subs(x(t),u).subs(y(t),v).subs(z(t),w)
    x_yz = sqrt((C1 - b*v**2 - c*w**2)/a)
    y_zx = sqrt((C1 - c*w**2 - a*u**2)/b)
    z_xy = sqrt((C1 - a*u**2 - b*v**2)/c)
    y_x = dsolve(diff(v(u),u) - ((a*u*F3-c*w*F1)/(c*w*F2-b*v*F3)).subs(w,z_xy).subs(v,v(u))).rhs
    z_x = dsolve(diff(w(u),u) - ((b*v*F1-a*u*F2)/(c*w*F2-b*v*F3)).subs(v,y_zx).subs(w,w(u))).rhs
    z_y = dsolve(diff(w(v),v) - ((b*v*F1-a*u*F2)/(a*u*F3-c*w*F1)).subs(u,x_yz).subs(w,w(v))).rhs
    x_y = dsolve(diff(u(v),v) - ((c*w*F2-b*v*F3)/(a*u*F3-c*w*F1)).subs(w,z_xy).subs(u,u(v))).rhs
    y_z = dsolve(diff(v(w),w) - ((a*u*F3-c*w*F1)/(b*v*F1-a*u*F2)).subs(u,x_yz).subs(v,v(w))).rhs
    x_z = dsolve(diff(u(w),w) - ((c*w*F2-b*v*F3)/(b*v*F1-a*u*F2)).subs(v,y_zx).subs(u,u(w))).rhs
    sol1 = dsolve(diff(u(t),t) - (c*w*F2 - b*v*F3).subs(v,y_x).subs(w,z_x).subs(u,u(t))).rhs
    sol2 = dsolve(diff(v(t),t) - (a*u*F3 - c*w*F1).subs(u,x_y).subs(w,z_y).subs(v,v(t))).rhs
    sol3 = dsolve(diff(w(t),t) - (b*v*F1 - a*u*F2).subs(u,x_z).subs(v,y_z).subs(w,w(t))).rhs
    return [sol1, sol2, sol3]

def _nonlinear_3eq_order1_type5(x, y, z, t, eq):
    r"""
    .. math:: x' = x (c F_2 - b F_3), \enspace y' = y (a F_3 - c F_1), \enspace z' = z (b F_1 - a F_2)

    where `F_n = F_n (x, y, z, t)` and are arbitrary functions.

    First Integral:

    .. math:: \left|x\right|^{a} \left|y\right|^{b} \left|z\right|^{c} = C_1

    where `C` is an arbitrary constant. If the function `F_n` is independent of `t`,
    then, by eliminating `t` and `z` from the first two equations of the system, one
    arrives at a first-order equation.

    References
    ==========
    -https://eqworld.ipmnet.ru/en/solutions/sysode/sode0406.pdf

    """
    C1 = get_numbered_constants(eq, num=1)
    u, v, w = symbols('u, v, w')
    fu, fv, fw = symbols('u, v, w', cls=Function)
    p = Wild('p', exclude=[x(t), y(t), z(t), t])
    q = Wild('q', exclude=[x(t), y(t), z(t), t])
    s = Wild('s', exclude=[x(t), y(t), z(t), t])
    F1, F2, F3 = symbols('F1, F2, F3', cls=Wild)
    r1 = eq[0].match(diff(x(t), t) - x(t)*F2 + x(t)*F3)
    r = collect_const(r1[F2]).match(s*F2)
    r.update(collect_const(r1[F3]).match(q*F3))
    if eq[1].has(r[F2]) and not eq[1].has(r[F3]):
        r[F2], r[F3] = r[F3], r[F2]
        r[s], r[q] = -r[q], -r[s]
    r.update((diff(y(t), t) - eq[1]).match(y(t)*(p*r[F3] - r[s]*F1)))
    a = r[p]; b = r[q]; c = r[s]
    F1 = r[F1].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F2 = r[F2].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    F3 = r[F3].subs(x(t), u).subs(y(t), v).subs(z(t), w)
    x_yz = (C1*v**-b*w**-c)**-a
    y_zx = (C1*w**-c*u**-a)**-b
    z_xy = (C1*u**-a*v**-b)**-c
    y_x = dsolve(diff(fv(u), u) - ((v*(a*F3 - c*F1))/(u*(c*F2 - b*F3))).subs(w, z_xy).subs(v, fv(u))).rhs
    z_x = dsolve(diff(fw(u), u) - ((w*(b*F1 - a*F2))/(u*(c*F2 - b*F3))).subs(v, y_zx).subs(w, fw(u))).rhs
    z_y = dsolve(diff(fw(v), v) - ((w*(b*F1 - a*F2))/(v*(a*F3 - c*F1))).subs(u, x_yz).subs(w, fw(v))).rhs
    x_y = dsolve(diff(fu(v), v) - ((u*(c*F2 - b*F3))/(v*(a*F3 - c*F1))).subs(w, z_xy).subs(u, fu(v))).rhs
    y_z = dsolve(diff(fv(w), w) - ((v*(a*F3 - c*F1))/(w*(b*F1 - a*F2))).subs(u, x_yz).subs(v, fv(w))).rhs
    x_z = dsolve(diff(fu(w), w) - ((u*(c*F2 - b*F3))/(w*(b*F1 - a*F2))).subs(v, y_zx).subs(u, fu(w))).rhs
    sol1 = dsolve(diff(fu(t), t) - (u*(c*F2 - b*F3)).subs(v, y_x).subs(w, z_x).subs(u, fu(t))).rhs
    sol2 = dsolve(diff(fv(t), t) - (v*(a*F3 - c*F1)).subs(u, x_y).subs(w, z_y).subs(v, fv(t))).rhs
    sol3 = dsolve(diff(fw(t), t) - (w*(b*F1 - a*F2)).subs(u, x_z).subs(v, y_z).subs(w, fw(t))).rhs
    return [sol1, sol2, sol3]


#This import is written at the bottom to avoid circular imports.
from .single import SingleODEProblem, SingleODESolver, solver_map

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\strings\accessor.py ===
from __future__ import annotations

import codecs
from functools import wraps
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import lib
from pandas._typing import (
    AlignJoin,
    DtypeObj,
    F,
    Scalar,
    npt,
)
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_object_dtype,
    is_re,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )

_shared_docs: dict[str, str] = {}
_cpython_optimized_encoders = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
_cpython_optimized_decoders = _cpython_optimized_encoders + ("utf-16", "utf-32")


def forbid_nonstring_types(
    forbidden: list[str] | None, name: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to forbid specific types for a method of StringMethods.

    For calling `.str.{method}` on a Series or Index, it is necessary to first
    initialize the :class:`StringMethods` object, and then call the method.
    However, different methods allow different input types, and so this can not
    be checked during :meth:`StringMethods.__init__`, but must be done on a
    per-method basis. This decorator exists to facilitate this process, and
    make it explicit which (inferred) types are disallowed by the method.

    :meth:`StringMethods.__init__` allows the *union* of types its different
    methods allow (after skipping NaNs; see :meth:`StringMethods._validate`),
    namely: ['string', 'empty', 'bytes', 'mixed', 'mixed-integer'].

    The default string types ['string', 'empty'] are allowed for all methods.
    For the additional types ['bytes', 'mixed', 'mixed-integer'], each method
    then needs to forbid the types it is not intended for.

    Parameters
    ----------
    forbidden : list-of-str or None
        List of forbidden non-string types, may be one or more of
        `['bytes', 'mixed', 'mixed-integer']`.
    name : str, default None
        Name of the method to use in the error message. By default, this is
        None, in which case the name from the method being wrapped will be
        copied. However, for working with further wrappers (like _pat_wrapper
        and _noarg_wrapper), it is necessary to specify the name.

    Returns
    -------
    func : wrapper
        The method to which the decorator is applied, with an added check that
        enforces the inferred type to not be in the list of forbidden types.

    Raises
    ------
    TypeError
        If the inferred type of the underlying data is in `forbidden`.
    """
    # deal with None
    forbidden = [] if forbidden is None else forbidden

    allowed_types = {"string", "empty", "bytes", "mixed", "mixed-integer"} - set(
        forbidden
    )

    def _forbid_nonstring_types(func: F) -> F:
        func_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._inferred_dtype not in allowed_types:
                msg = (
                    f"Cannot use .str.{func_name} with values of "
                    f"inferred dtype '{self._inferred_dtype}'."
                )
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        wrapper.__name__ = func_name
        return cast(F, wrapper)

    return _forbid_nonstring_types


def _map_and_wrap(name: str | None, docstring: str | None):
    @forbid_nonstring_types(["bytes"], name=name)
    def wrapper(self):
        result = getattr(self._data.array, f"_str_{name}")()
        return self._wrap_result(
            result, returns_string=name not in ("isnumeric", "isdecimal")
        )

    wrapper.__doc__ = docstring
    return wrapper


class StringMethods(NoNewAttributesMixin):
    """
    Vectorized string functions for Series and Index.

    NAs stay NA unless handled otherwise by a particular method.
    Patterned after Python's string methods, with some inspiration from
    R's stringr package.

    Examples
    --------
    >>> s = pd.Series(["A_Str_Series"])
    >>> s
    0    A_Str_Series
    dtype: object

    >>> s.str.split("_")
    0    [A, Str, Series]
    dtype: object

    >>> s.str.replace("_", "")
    0    AStrSeries
    dtype: object
    """

    # Note: see the docstring in pandas.core.strings.__init__
    # for an explanation of the implementation.
    # TODO: Dispatch all the methods
    # Currently the following are not dispatched to the array
    # * cat
    # * extractall

    def __init__(self, data) -> None:
        from pandas.core.arrays.string_ import StringDtype

        self._inferred_dtype = self._validate(data)
        self._is_categorical = isinstance(data.dtype, CategoricalDtype)
        self._is_string = isinstance(data.dtype, StringDtype)
        self._data = data

        self._index = self._name = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name

        # ._values.categories works for both Series/Index
        self._parent = data._values.categories if self._is_categorical else data
        # save orig to blow up categoricals to the right type
        self._orig = data
        self._freeze()

    @staticmethod
    def _validate(data):
        """
        Auxiliary function for StringMethods, infers and checks dtype of data.

        This is a "first line of defence" at the creation of the StringMethods-
        object, and just checks that the dtype is in the
        *union* of the allowed types over all string methods below; this
        restriction is then refined on a per-method basis using the decorator
        @forbid_nonstring_types (more info in the corresponding docstring).

        This really should exclude all series/index with any non-string values,
        but that isn't practical for performance reasons until we have a str
        dtype (GH 9343 / 13877)

        Parameters
        ----------
        data : The content of the Series

        Returns
        -------
        dtype : inferred dtype of data
        """
        if isinstance(data, ABCMultiIndex):
            raise AttributeError(
                "Can only use .str accessor with Index, not MultiIndex"
            )

        # see _libs/lib.pyx for list of inferred types
        allowed_types = ["string", "empty", "bytes", "mixed", "mixed-integer"]

        data = extract_array(data)

        values = getattr(data, "categories", data)  # categorical / normal

        inferred_dtype = lib.infer_dtype(values, skipna=True)

        if inferred_dtype not in allowed_types:
            raise AttributeError("Can only use .str accessor with string values!")
        return inferred_dtype

    def __getitem__(self, key):
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(
        self,
        result,
        name=None,
        expand: bool | None = None,
        fill_value=np.nan,
        returns_string: bool = True,
        returns_bool: bool = False,
        dtype=None,
    ):
        from pandas import (
            Index,
            MultiIndex,
        )

        if not hasattr(result, "ndim") or not hasattr(result, "dtype"):
            if isinstance(result, ABCDataFrame):
                result = result.__finalize__(self._orig, name="str")
            return result
        assert result.ndim < 3

        # We can be wrapping a string / object / categorical result, in which
        # case we'll want to return the same dtype as the input.
        # Or we can be wrapping a numeric output, in which case we don't want
        # to return a StringArray.
        # Ideally the array method returns the right array type.
        if expand is None:
            # infer from ndim if expand is not specified
            expand = result.ndim != 1
        elif expand is True and not isinstance(self._orig, ABCIndex):
            # required when expand=True is explicitly specified
            # not needed when inferred
            if isinstance(result.dtype, ArrowDtype):
                import pyarrow as pa

                from pandas.compat import pa_version_under11p0

                from pandas.core.arrays.arrow.array import ArrowExtensionArray

                value_lengths = pa.compute.list_value_length(result._pa_array)
                max_len = pa.compute.max(value_lengths).as_py()
                min_len = pa.compute.min(value_lengths).as_py()
                if result._hasna:
                    # ArrowExtensionArray.fillna doesn't work for list scalars
                    result = ArrowExtensionArray(
                        result._pa_array.fill_null([None] * max_len)
                    )
                if min_len < max_len:
                    # append nulls to each scalar list element up to max_len
                    if not pa_version_under11p0:
                        result = ArrowExtensionArray(
                            pa.compute.list_slice(
                                result._pa_array,
                                start=0,
                                stop=max_len,
                                return_fixed_size_list=True,
                            )
                        )
                    else:
                        all_null = np.full(max_len, fill_value=None, dtype=object)
                        values = result.to_numpy()
                        new_values = []
                        for row in values:
                            if len(row) < max_len:
                                nulls = all_null[: max_len - len(row)]
                                row = np.append(row, nulls)
                            new_values.append(row)
                        pa_type = result._pa_array.type
                        result = ArrowExtensionArray(pa.array(new_values, type=pa_type))
                if name is not None:
                    labels = name
                else:
                    labels = range(max_len)
                result = (
                    pa.compute.list_flatten(result._pa_array)
                    .to_numpy()
                    .reshape(len(result), max_len)
                )
                result = {
                    label: ArrowExtensionArray(pa.array(res))
                    for label, res in zip(labels, result.T)
                }
            elif is_object_dtype(result):

                def cons_row(x):
                    if is_list_like(x):
                        return x
                    else:
                        return [x]

                result = [cons_row(x) for x in result]
                if result and not self._is_string:
                    # propagate nan values to match longest sequence (GH 18450)
                    max_len = max(len(x) for x in result)
                    result = [
                        x * max_len if len(x) == 0 or x[0] is np.nan else x
                        for x in result
                    ]

        if not isinstance(expand, bool):
            raise ValueError("expand must be True or False")

        if expand is False:
            # if expand is False, result should have the same name
            # as the original otherwise specified
            if name is None:
                name = getattr(result, "name", None)
            if name is None:
                # do not use logical or, _orig may be a DataFrame
                # which has "name" column
                name = self._orig.name

        # Wait until we are sure result is a Series or Index before
        # checking attributes (GH 12180)
        if isinstance(self._orig, ABCIndex):
            # if result is a boolean np.array, return the np.array
            # instead of wrapping it into a boolean Index (GH 8875)
            if is_bool_dtype(result):
                return result

            if expand:
                result = list(result)
                out: Index = MultiIndex.from_tuples(result, names=name)
                if out.nlevels == 1:
                    # We had all tuples of length-one, which are
                    # better represented as a regular Index.
                    out = out.get_level_values(0)
                return out
            else:
                return Index(result, name=name, dtype=dtype)
        else:
            index = self._orig.index
            # This is a mess.
            _dtype: DtypeObj | str | None = dtype
            vdtype = getattr(result, "dtype", None)
            if _dtype is not None:
                pass
            elif self._is_string:
                if is_bool_dtype(vdtype):
                    _dtype = result.dtype
                elif returns_string:
                    _dtype = self._orig.dtype
                else:
                    _dtype = vdtype
            elif vdtype is not None:
                _dtype = vdtype

            if expand:
                cons = self._orig._constructor_expanddim
                result = cons(result, columns=name, index=index, dtype=_dtype)
            else:
                # Must be a Series
                cons = self._orig._constructor
                result = cons(result, name=name, index=index, dtype=_dtype)
            result = result.__finalize__(self._orig, method="str")
            if name is not None and result.ndim == 1:
                # __finalize__ might copy over the original name, but we may
                # want the new name (e.g. str.extract).
                result.name = name
            return result

    def _get_series_list(self, others):
        """
        Auxiliary function for :meth:`str.cat`. Turn potentially mixed input
        into a list of Series (elements without an index must match the length
        of the calling Series/Index).

        Parameters
        ----------
        others : Series, DataFrame, np.ndarray, list-like or list-like of
            Objects that are either Series, Index or np.ndarray (1-dim).

        Returns
        -------
        list of Series
            Others transformed into list of Series.
        """
        from pandas import (
            DataFrame,
            Series,
        )

        # self._orig is either Series or Index
        idx = self._orig if isinstance(self._orig, ABCIndex) else self._orig.index

        # Generally speaking, all objects without an index inherit the index
        # `idx` of the calling Series/Index - i.e. must have matching length.
        # Objects with an index (i.e. Series/Index/DataFrame) keep their own.
        if isinstance(others, ABCSeries):
            return [others]
        elif isinstance(others, ABCIndex):
            return [Series(others, index=idx, dtype=others.dtype)]
        elif isinstance(others, ABCDataFrame):
            return [others[x] for x in others]
        elif isinstance(others, np.ndarray) and others.ndim == 2:
            others = DataFrame(others, index=idx)
            return [others[x] for x in others]
        elif is_list_like(others, allow_sets=False):
            try:
                others = list(others)  # ensure iterators do not get read twice etc
            except TypeError:
                # e.g. ser.str, raise below
                pass
            else:
                # in case of list-like `others`, all elements must be
                # either Series/Index/np.ndarray (1-dim)...
                if all(
                    isinstance(x, (ABCSeries, ABCIndex, ExtensionArray))
                    or (isinstance(x, np.ndarray) and x.ndim == 1)
                    for x in others
                ):
                    los: list[Series] = []
                    while others:  # iterate through list and append each element
                        los = los + self._get_series_list(others.pop(0))
                    return los
                # ... or just strings
                elif all(not is_list_like(x) for x in others):
                    return [Series(others, index=idx)]
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarray "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )

    @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
    def cat(
        self,
        others=None,
        sep: str | None = None,
        na_rep=None,
        join: AlignJoin = "left",
    ) -> str | Series | Index:
        """
        Concatenate strings in the Series/Index with given separator.

        If `others` is specified, this function concatenates the Series/Index
        and elements of `others` element-wise.
        If `others` is not passed, then all values in the Series/Index are
        concatenated into a single string with a given `sep`.

        Parameters
        ----------
        others : Series, Index, DataFrame, np.ndarray or list-like
            Series, Index, DataFrame, np.ndarray (one- or two-dimensional) and
            other list-likes of strings must have the same length as the
            calling Series/Index, with the exception of indexed objects (i.e.
            Series/Index/DataFrame) if `join` is not None.

            If others is a list-like that contains a combination of Series,
            Index or np.ndarray (1-dim), then all elements will be unpacked and
            must satisfy the above criteria individually.

            If others is None, the method returns the concatenation of all
            strings in the calling Series/Index.
        sep : str, default ''
            The separator between the different elements/columns. By default
            the empty string `''` is used.
        na_rep : str or None, default None
            Representation that is inserted for all missing values:

            - If `na_rep` is None, and `others` is None, missing values in the
              Series/Index are omitted from the result.
            - If `na_rep` is None, and `others` is not None, a row containing a
              missing value in any of the columns (before concatenation) will
              have a missing value in the result.
        join : {'left', 'right', 'outer', 'inner'}, default 'left'
            Determines the join-style between the calling Series/Index and any
            Series/Index/DataFrame in `others` (objects without an index need
            to match the length of the calling Series/Index). To disable
            alignment, use `.values` on any Series/Index/DataFrame in `others`.

        Returns
        -------
        str, Series or Index
            If `others` is None, `str` is returned, otherwise a `Series/Index`
            (same type as caller) of objects is returned.

        See Also
        --------
        split : Split each string in the Series/Index.
        join : Join lists contained as elements in the Series/Index.

        Examples
        --------
        When not passing `others`, all values are concatenated into a single
        string:

        >>> s = pd.Series(['a', 'b', np.nan, 'd'])
        >>> s.str.cat(sep=' ')
        'a b d'

        By default, NA values in the Series are ignored. Using `na_rep`, they
        can be given a representation:

        >>> s.str.cat(sep=' ', na_rep='?')
        'a b ? d'

        If `others` is specified, corresponding values are concatenated with
        the separator. Result will be a Series of strings.

        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',')
        0    a,A
        1    b,B
        2    NaN
        3    d,D
        dtype: object

        Missing values will remain missing in the result, but can again be
        represented using `na_rep`

        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',', na_rep='-')
        0    a,A
        1    b,B
        2    -,C
        3    d,D
        dtype: object

        If `sep` is not specified, the values are concatenated without
        separation.

        >>> s.str.cat(['A', 'B', 'C', 'D'], na_rep='-')
        0    aA
        1    bB
        2    -C
        3    dD
        dtype: object

        Series with different indexes can be aligned before concatenation. The
        `join`-keyword works as in other methods.

        >>> t = pd.Series(['d', 'a', 'e', 'c'], index=[3, 0, 4, 2])
        >>> s.str.cat(t, join='left', na_rep='-')
        0    aa
        1    b-
        2    -c
        3    dd
        dtype: object
        >>>
        >>> s.str.cat(t, join='outer', na_rep='-')
        0    aa
        1    b-
        2    -c
        3    dd
        4    -e
        dtype: object
        >>>
        >>> s.str.cat(t, join='inner', na_rep='-')
        0    aa
        2    -c
        3    dd
        dtype: object
        >>>
        >>> s.str.cat(t, join='right', na_rep='-')
        3    dd
        0    aa
        4    -e
        2    -c
        dtype: object

        For more examples, see :ref:`here <text.concatenate>`.
        """
        # TODO: dispatch
        from pandas import (
            Index,
            Series,
            concat,
        )

        if isinstance(others, str):
            raise ValueError("Did you mean to supply a `sep` keyword?")
        if sep is None:
            sep = ""

        if isinstance(self._orig, ABCIndex):
            data = Series(self._orig, index=self._orig, dtype=self._orig.dtype)
        else:  # Series
            data = self._orig

        # concatenate Series/Index with itself if no "others"
        if others is None:
            # error: Incompatible types in assignment (expression has type
            # "ndarray", variable has type "Series")
            data = ensure_object(data)  # type: ignore[assignment]
            na_mask = isna(data)
            if na_rep is None and na_mask.any():
                return sep.join(data[~na_mask])
            elif na_rep is not None and na_mask.any():
                return sep.join(np.where(na_mask, na_rep, data))
            else:
                return sep.join(data)

        try:
            # turn anything in "others" into lists of Series
            others = self._get_series_list(others)
        except ValueError as err:  # do not catch TypeError raised by _get_series_list
            raise ValueError(
                "If `others` contains arrays or lists (or other "
                "list-likes without an index), these must all be "
                "of the same length as the calling Series/Index."
            ) from err

        # align if required
        if any(not data.index.equals(x.index) for x in others):
            # Need to add keys for uniqueness in case of duplicate columns
            others = concat(
                others,
                axis=1,
                join=(join if join == "inner" else "outer"),
                keys=range(len(others)),
                sort=False,
                copy=False,
            )
            data, others = data.align(others, join=join)
            others = [others[x] for x in others]  # again list of Series

        all_cols = [ensure_object(x) for x in [data] + others]
        na_masks = np.array([isna(x) for x in all_cols])
        union_mask = np.logical_or.reduce(na_masks, axis=0)

        if na_rep is None and union_mask.any():
            # no na_rep means NaNs for all rows where any column has a NaN
            # only necessary if there are actually any NaNs
            result = np.empty(len(data), dtype=object)
            np.putmask(result, union_mask, np.nan)

            not_masked = ~union_mask
            result[not_masked] = cat_safe([x[not_masked] for x in all_cols], sep)
        elif na_rep is not None and union_mask.any():
            # fill NaNs with na_rep in case there are actually any NaNs
            all_cols = [
                np.where(nm, na_rep, col) for nm, col in zip(na_masks, all_cols)
            ]
            result = cat_safe(all_cols, sep)
        else:
            # no NaNs - can just concatenate
            result = cat_safe(all_cols, sep)

        out: Index | Series
        if isinstance(self._orig.dtype, CategoricalDtype):
            # We need to infer the new categories.
            dtype = self._orig.dtype.categories.dtype
        else:
            dtype = self._orig.dtype
        if isinstance(self._orig, ABCIndex):
            # add dtype for case that result is all-NA
            if isna(result).all():
                dtype = object  # type: ignore[assignment]

            out = Index(result, dtype=dtype, name=self._orig.name)
        else:  # Series
            res_ser = Series(
                result, dtype=dtype, index=data.index, name=self._orig.name, copy=False
            )
            out = res_ser.__finalize__(self._orig, method="str_cat")
        return out

    _shared_docs[
        "str_split"
    ] = r"""
    Split strings around given separator/delimiter.

    Splits the string in the Series/Index from the %(side)s,
    at the specified delimiter string.

    Parameters
    ----------
    pat : str%(pat_regex)s, optional
        %(pat_description)s.
        If not specified, split on whitespace.
    n : int, default -1 (all)
        Limit number of splits in output.
        ``None``, 0 and -1 will be interpreted as return all splits.
    expand : bool, default False
        Expand the split strings into separate columns.

        - If ``True``, return DataFrame/MultiIndex expanding dimensionality.
        - If ``False``, return Series/Index, containing lists of strings.
    %(regex_argument)s
    Returns
    -------
    Series, Index, DataFrame or MultiIndex
        Type matches caller unless ``expand=True`` (see Notes).
    %(raises_split)s
    See Also
    --------
    Series.str.split : Split strings around given separator/delimiter.
    Series.str.rsplit : Splits string around given separator/delimiter,
        starting from the right.
    Series.str.join : Join lists contained as elements in the Series/Index
        with passed delimiter.
    str.split : Standard library version for split.
    str.rsplit : Standard library version for rsplit.

    Notes
    -----
    The handling of the `n` keyword depends on the number of found splits:

    - If found splits > `n`,  make first `n` splits only
    - If found splits <= `n`, make all splits
    - If for a certain row the number of found splits < `n`,
      append `None` for padding up to `n` if ``expand=True``

    If using ``expand=True``, Series and Index callers return DataFrame and
    MultiIndex objects, respectively.
    %(regex_pat_note)s
    Examples
    --------
    >>> s = pd.Series(
    ...     [
    ...         "this is a regular sentence",
    ...         "https://docs.python.org/3/tutorial/index.html",
    ...         np.nan
    ...     ]
    ... )
    >>> s
    0                       this is a regular sentence
    1    https://docs.python.org/3/tutorial/index.html
    2                                              NaN
    dtype: object

    In the default setting, the string is split by whitespace.

    >>> s.str.split()
    0                   [this, is, a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    Without the `n` parameter, the outputs of `rsplit` and `split`
    are identical.

    >>> s.str.rsplit()
    0                   [this, is, a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    The `n` parameter can be used to limit the number of splits on the
    delimiter. The outputs of `split` and `rsplit` are different.

    >>> s.str.split(n=2)
    0                     [this, is, a regular sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    >>> s.str.rsplit(n=2)
    0                     [this is a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    The `pat` parameter can be used to split by other characters.

    >>> s.str.split(pat="/")
    0                         [this is a regular sentence]
    1    [https:, , docs.python.org, 3, tutorial, index...
    2                                                  NaN
    dtype: object

    When using ``expand=True``, the split elements will expand out into
    separate columns. If NaN is present, it is propagated throughout
    the columns during the split.

    >>> s.str.split(expand=True)
                                                   0     1     2        3         4
    0                                           this    is     a  regular  sentence
    1  https://docs.python.org/3/tutorial/index.html  None  None     None      None
    2                                            NaN   NaN   NaN      NaN       NaN

    For slightly more complex use cases like splitting the html document name
    from a url, a combination of parameter settings can be used.

    >>> s.str.rsplit("/", n=1, expand=True)
                                        0           1
    0          this is a regular sentence        None
    1  https://docs.python.org/3/tutorial  index.html
    2                                 NaN         NaN
    %(regex_examples)s"""

    @Appender(
        _shared_docs["str_split"]
        % {
            "side": "beginning",
            "pat_regex": " or compiled regex",
            "pat_description": "String or regular expression to split on",
            "regex_argument": """
    regex : bool, default None
        Determines if the passed-in pattern is a regular expression:

        - If ``True``, assumes the passed-in pattern is a regular expression
        - If ``False``, treats the pattern as a literal string.
        - If ``None`` and `pat` length is 1, treats `pat` as a literal string.
        - If ``None`` and `pat` length is not 1, treats `pat` as a regular expression.
        - Cannot be set to False if `pat` is a compiled regex

        .. versionadded:: 1.4.0
         """,
            "raises_split": """
                      Raises
                      ------
                      ValueError
                          * if `regex` is False and `pat` is a compiled regex
                      """,
            "regex_pat_note": """
    Use of `regex =False` with a `pat` as a compiled regex will raise an error.
            """,
            "method": "split",
            "regex_examples": r"""
    Remember to escape special characters when explicitly using regular expressions.

    >>> s = pd.Series(["foo and bar plus baz"])
    >>> s.str.split(r"and|plus", expand=True)
        0   1   2
    0 foo bar baz

    Regular expressions can be used to handle urls or file names.
    When `pat` is a string and ``regex=None`` (the default), the given `pat` is compiled
    as a regex only if ``len(pat) != 1``.

    >>> s = pd.Series(['foojpgbar.jpg'])
    >>> s.str.split(r".", expand=True)
               0    1
    0  foojpgbar  jpg

    >>> s.str.split(r"\.jpg", expand=True)
               0 1
    0  foojpgbar

    When ``regex=True``, `pat` is interpreted as a regex

    >>> s.str.split(r"\.jpg", regex=True, expand=True)
               0 1
    0  foojpgbar

    A compiled regex can be passed as `pat`

    >>> import re
    >>> s.str.split(re.compile(r"\.jpg"), expand=True)
               0 1
    0  foojpgbar

    When ``regex=False``, `pat` is interpreted as the string itself

    >>> s.str.split(r"\.jpg", regex=False, expand=True)
                   0
    0  foojpgbar.jpg
    """,
        }
    )
    @forbid_nonstring_types(["bytes"])
    def split(
        self,
        pat: str | re.Pattern | None = None,
        *,
        n=-1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if regex is False and is_re(pat):
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )
        if is_re(pat):
            regex = True
        result = self._data.array._str_split(pat, n, expand, regex)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(
        _shared_docs["str_split"]
        % {
            "side": "end",
            "pat_regex": "",
            "pat_description": "String to split on",
            "regex_argument": "",
            "raises_split": "",
            "regex_pat_note": "",
            "method": "rsplit",
            "regex_examples": "",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rsplit(self, pat=None, *, n=-1, expand: bool = False):
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    _shared_docs[
        "str_partition"
    ] = """
    Split the string at the %(side)s occurrence of `sep`.

    This method splits the string at the %(side)s occurrence of `sep`,
    and returns 3 elements containing the part before the separator,
    the separator itself, and the part after the separator.
    If the separator is not found, return %(return)s.

    Parameters
    ----------
    sep : str, default whitespace
        String to split on.
    expand : bool, default True
        If True, return DataFrame/MultiIndex expanding dimensionality.
        If False, return Series/Index.

    Returns
    -------
    DataFrame/MultiIndex or Series/Index of objects

    See Also
    --------
    %(also)s
    Series.str.split : Split strings around given separators.
    str.partition : Standard library version.

    Examples
    --------

    >>> s = pd.Series(['Linda van der Berg', 'George Pitt-Rivers'])
    >>> s
    0    Linda van der Berg
    1    George Pitt-Rivers
    dtype: object

    >>> s.str.partition()
            0  1             2
    0   Linda     van der Berg
    1  George      Pitt-Rivers

    To partition by the last space instead of the first one:

    >>> s.str.rpartition()
                   0  1            2
    0  Linda van der            Berg
    1         George     Pitt-Rivers

    To partition by something different than a space:

    >>> s.str.partition('-')
                        0  1       2
    0  Linda van der Berg
    1         George Pitt  -  Rivers

    To return a Series containing tuples instead of a DataFrame:

    >>> s.str.partition('-', expand=False)
    0    (Linda van der Berg, , )
    1    (George Pitt, -, Rivers)
    dtype: object

    Also available on indices:

    >>> idx = pd.Index(['X 123', 'Y 999'])
    >>> idx
    Index(['X 123', 'Y 999'], dtype='object')

    Which will create a MultiIndex:

    >>> idx.str.partition()
    MultiIndex([('X', ' ', '123'),
                ('Y', ' ', '999')],
               )

    Or an index with tuples with ``expand=False``:

    >>> idx.str.partition(expand=False)
    Index([('X', ' ', '123'), ('Y', ' ', '999')], dtype='object')
    """

    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "first",
            "return": "3 elements containing the string itself, followed by two "
            "empty strings",
            "also": "rpartition : Split the string at the last occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def partition(self, sep: str = " ", expand: bool = True):
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "last",
            "return": "3 elements containing two empty strings, followed by the "
            "string itself",
            "also": "partition : Split the string at the first occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rpartition(self, sep: str = " ", expand: bool = True):
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    def get(self, i):
        """
        Extract element from each component at specified position or with specified key.

        Extract element from lists, tuples, dict, or strings in each element in the
        Series/Index.

        Parameters
        ----------
        i : int or hashable dict label
            Position or key of element to extract.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = pd.Series(["String",
        ...               (1, 2, 3),
        ...               ["a", "b", "c"],
        ...               123,
        ...               -456,
        ...               {1: "Hello", "2": "World"}])
        >>> s
        0                        String
        1                     (1, 2, 3)
        2                     [a, b, c]
        3                           123
        4                          -456
        5    {1: 'Hello', '2': 'World'}
        dtype: object

        >>> s.str.get(1)
        0        t
        1        2
        2        b
        3      NaN
        4      NaN
        5    Hello
        dtype: object

        >>> s.str.get(-1)
        0      g
        1      3
        2      c
        3    NaN
        4    NaN
        5    None
        dtype: object

        Return element with given key

        >>> s = pd.Series([{"name": "Hello", "value": "World"},
        ...               {"name": "Goodbye", "value": "Planet"}])
        >>> s.str.get('name')
        0      Hello
        1    Goodbye
        dtype: object
        """
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def join(self, sep: str):
        """
        Join lists contained as elements in the Series/Index with passed delimiter.

        If the elements of a Series are lists themselves, join the content of these
        lists using the delimiter passed to the function.
        This function is an equivalent to :meth:`str.join`.

        Parameters
        ----------
        sep : str
            Delimiter to use between list entries.

        Returns
        -------
        Series/Index: object
            The list entries concatenated by intervening occurrences of the
            delimiter.

        Raises
        ------
        AttributeError
            If the supplied Series contains neither strings nor lists.

        See Also
        --------
        str.join : Standard library version of this method.
        Series.str.split : Split strings around given separator/delimiter.

        Notes
        -----
        If any of the list items is not a string object, the result of the join
        will be `NaN`.

        Examples
        --------
        Example with a list that contains non-string elements.

        >>> s = pd.Series([['lion', 'elephant', 'zebra'],
        ...                [1.1, 2.2, 3.3],
        ...                ['cat', np.nan, 'dog'],
        ...                ['cow', 4.5, 'goat'],
        ...                ['duck', ['swan', 'fish'], 'guppy']])
        >>> s
        0        [lion, elephant, zebra]
        1                [1.1, 2.2, 3.3]
        2                [cat, nan, dog]
        3               [cow, 4.5, goat]
        4    [duck, [swan, fish], guppy]
        dtype: object

        Join all lists using a '-'. The lists containing object(s) of types other
        than str will produce a NaN.

        >>> s.str.join('-')
        0    lion-elephant-zebra
        1                    NaN
        2                    NaN
        3                    NaN
        4                    NaN
        dtype: object
        """
        result = self._data.array._str_join(sep)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        r"""
        Test if pattern or regex is contained within a string of a Series or Index.

        Return boolean Series or Index based on whether a given pattern or regex is
        contained within a string of a Series or Index.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Flags to pass through to the re module, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.
        regex : bool, default True
            If True, assumes the pat is a regular expression.

            If False, treats the pat as a literal string.

        Returns
        -------
        Series or Index of boolean values
            A Series or Index of boolean values indicating whether the
            given pattern is contained within the string of each element
            of the Series or Index.

        See Also
        --------
        match : Analogous, but stricter, relying on re.match instead of re.search.
        Series.str.startswith : Test if the start of each string element matches a
            pattern.
        Series.str.endswith : Same as startswith, but tests the end of string.

        Examples
        --------
        Returning a Series of booleans using only a literal pattern.

        >>> s1 = pd.Series(['Mouse', 'dog', 'house and parrot', '23', np.nan])
        >>> s1.str.contains('og', regex=False)
        0    False
        1     True
        2    False
        3    False
        4      NaN
        dtype: object

        Returning an Index of booleans using only a literal pattern.

        >>> ind = pd.Index(['Mouse', 'dog', 'house and parrot', '23.0', np.nan])
        >>> ind.str.contains('23', regex=False)
        Index([False, False, False, True, nan], dtype='object')

        Specifying case sensitivity using `case`.

        >>> s1.str.contains('oG', case=True, regex=True)
        0    False
        1    False
        2    False
        3    False
        4      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN` replaces NaN values
        with `False`. If Series or Index does not contain NaN values
        the resultant dtype will be `bool`, otherwise, an `object` dtype.

        >>> s1.str.contains('og', na=False, regex=True)
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        Returning 'house' or 'dog' when either expression occurs in a string.

        >>> s1.str.contains('house|dog', regex=True)
        0    False
        1     True
        2     True
        3    False
        4      NaN
        dtype: object

        Ignoring case sensitivity using `flags` with regex.

        >>> import re
        >>> s1.str.contains('PARROT', flags=re.IGNORECASE, regex=True)
        0    False
        1    False
        2     True
        3    False
        4      NaN
        dtype: object

        Returning any digit using regular expression.

        >>> s1.str.contains('\\d', regex=True)
        0    False
        1    False
        2    False
        3     True
        4      NaN
        dtype: object

        Ensure `pat` is a not a literal pattern when `regex` is set to True.
        Note in the following example one might expect only `s2[1]` and `s2[3]` to
        return `True`. However, '.0' as a regex matches any character
        followed by a 0.

        >>> s2 = pd.Series(['40', '40.0', '41', '41.0', '35'])
        >>> s2.str.contains('.0', regex=True)
        0     True
        1     True
        2    False
        3     True
        4    False
        dtype: bool
        """
        if regex and re.compile(pat).groups:
            warnings.warn(
                "This pattern is interpreted as a regular expression, and has "
                "match groups. To actually get the groups, use str.extract.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        result = self._data.array._str_contains(pat, case, flags, na, regex)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def match(self, pat: str, case: bool = True, flags: int = 0, na=lib.no_default):
        """
        Determine if each string starts with a match of a regular expression.

        Parameters
        ----------
        pat : str
            Character sequence.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.

        Returns
        -------
        Series/Index/array of boolean values

        See Also
        --------
        fullmatch : Stricter matching that requires the entire string to match.
        contains : Analogous, but less strict, relying on re.search instead of
            re.match.
        extract : Extract matched groups.

        Examples
        --------
        >>> ser = pd.Series(["horse", "eagle", "donkey"])
        >>> ser.str.match("e")
        0   False
        1   True
        2   False
        dtype: bool
        """
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def fullmatch(self, pat, case: bool = True, flags: int = 0, na=lib.no_default):
        """
        Determine if each string entirely matches a regular expression.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.

        Returns
        -------
        Series/Index/array of boolean values

        See Also
        --------
        match : Similar, but also returns `True` when only a *prefix* of the string
            matches the regular expression.
        extract : Extract matched groups.

        Examples
        --------
        >>> ser = pd.Series(["cat", "duck", "dove"])
        >>> ser.str.fullmatch(r'd.+')
        0   False
        1    True
        2    True
        dtype: bool
        """
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ):
        r"""
        Replace each occurrence of pattern/regex in the Series/Index.

        Equivalent to :meth:`str.replace` or :func:`re.sub`, depending on
        the regex value.

        Parameters
        ----------
        pat : str or compiled regex
            String can be a character sequence or regular expression.
        repl : str or callable
            Replacement string or a callable. The callable is passed the regex
            match object and must return a replacement string to be used.
            See :func:`re.sub`.
        n : int, default -1 (all)
            Number of replacements to make from start.
        case : bool, default None
            Determines if replace is case sensitive:

            - If True, case sensitive (the default if `pat` is a string)
            - Set to False for case insensitive
            - Cannot be set if `pat` is a compiled regex.

        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE. Cannot be set if `pat` is a compiled
            regex.
        regex : bool, default False
            Determines if the passed-in pattern is a regular expression:

            - If True, assumes the passed-in pattern is a regular expression.
            - If False, treats the pattern as a literal string
            - Cannot be set to False if `pat` is a compiled regex or `repl` is
              a callable.

        Returns
        -------
        Series or Index of object
            A copy of the object with all matching occurrences of `pat` replaced by
            `repl`.

        Raises
        ------
        ValueError
            * if `regex` is False and `repl` is a callable or `pat` is a compiled
              regex
            * if `pat` is a compiled regex and `case` or `flags` is set

        Notes
        -----
        When `pat` is a compiled regex, all flags should be included in the
        compiled regex. Use of `case`, `flags`, or `regex=False` with a compiled
        regex will raise an error.

        Examples
        --------
        When `pat` is a string and `regex` is True, the given `pat`
        is compiled as a regex. When `repl` is a string, it replaces matching
        regex patterns as with :meth:`re.sub`. NaN value(s) in the Series are
        left as is:

        >>> pd.Series(['foo', 'fuz', np.nan]).str.replace('f.', 'ba', regex=True)
        0    bao
        1    baz
        2    NaN
        dtype: object

        When `pat` is a string and `regex` is False, every `pat` is replaced with
        `repl` as with :meth:`str.replace`:

        >>> pd.Series(['f.o', 'fuz', np.nan]).str.replace('f.', 'ba', regex=False)
        0    bao
        1    fuz
        2    NaN
        dtype: object

        When `repl` is a callable, it is called on every `pat` using
        :func:`re.sub`. The callable should expect one positional argument
        (a regex object) and return a string.

        To get the idea:

        >>> pd.Series(['foo', 'fuz', np.nan]).str.replace('f', repr, regex=True)
        0    <re.Match object; span=(0, 1), match='f'>oo
        1    <re.Match object; span=(0, 1), match='f'>uz
        2                                            NaN
        dtype: object

        Reverse every lowercase alphabetic word:

        >>> repl = lambda m: m.group(0)[::-1]
        >>> ser = pd.Series(['foo 123', 'bar baz', np.nan])
        >>> ser.str.replace(r'[a-z]+', repl, regex=True)
        0    oof 123
        1    rab zab
        2        NaN
        dtype: object

        Using regex groups (extract second group and swap case):

        >>> pat = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
        >>> repl = lambda m: m.group('two').swapcase()
        >>> ser = pd.Series(['One Two Three', 'Foo Bar Baz'])
        >>> ser.str.replace(pat, repl, regex=True)
        0    tWO
        1    bAR
        dtype: object

        Using a compiled regex with flags

        >>> import re
        >>> regex_pat = re.compile(r'FUZ', flags=re.IGNORECASE)
        >>> pd.Series(['foo', 'fuz', np.nan]).str.replace(regex_pat, 'bar', regex=True)
        0    foo
        1    bar
        2    NaN
        dtype: object
        """
        # Check whether repl is valid (GH 13438, GH 15055)
        if not (isinstance(repl, str) or callable(repl)):
            raise TypeError("repl must be a string or callable")

        is_compiled_re = is_re(pat)
        if regex or regex is None:
            if is_compiled_re and (case is not None or flags != 0):
                raise ValueError(
                    "case and flags cannot be set when pat is a compiled regex"
                )

        elif is_compiled_re:
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )
        elif callable(repl):
            raise ValueError("Cannot use a callable replacement when regex=False")

        if case is None:
            case = True

        result = self._data.array._str_replace(
            pat, repl, n=n, case=case, flags=flags, regex=regex
        )
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def repeat(self, repeats):
        """
        Duplicate each string in the Series or Index.

        Parameters
        ----------
        repeats : int or sequence of int
            Same value for all (int) or different value per (sequence).

        Returns
        -------
        Series or pandas.Index
            Series or Index of repeated string objects specified by
            input parameter repeats.

        Examples
        --------
        >>> s = pd.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        dtype: object

        Single int repeats string in Series

        >>> s.str.repeat(repeats=2)
        0    aa
        1    bb
        2    cc
        dtype: object

        Sequence of int repeats corresponding string in Series

        >>> s.str.repeat(repeats=[1, 2, 3])
        0      a
        1     bb
        2    ccc
        dtype: object
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        """
        Pad strings in the Series/Index up to width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be filled
            with character defined in `fillchar`.
        side : {'left', 'right', 'both'}, default 'left'
            Side from which to fill resulting string.
        fillchar : str, default ' '
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series or Index of object
            Returns Series or Index with minimum number of char in object.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='left')``.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='right')``.
        Series.str.center : Fills both sides of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='both')``.
        Series.str.zfill : Pad strings in the Series/Index by prepending '0'
            character. Equivalent to ``Series.str.pad(side='left', fillchar='0')``.

        Examples
        --------
        >>> s = pd.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        dtype: object

        >>> s.str.pad(width=10)
        0       caribou
        1         tiger
        dtype: object

        >>> s.str.pad(width=10, side='right', fillchar='-')
        0    caribou---
        1    tiger-----
        dtype: object

        >>> s.str.pad(width=10, side='both', fillchar='-')
        0    -caribou--
        1    --tiger---
        dtype: object
        """
        if not isinstance(fillchar, str):
            msg = f"fillchar must be a character, not {type(fillchar).__name__}"
            raise TypeError(msg)

        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    _shared_docs[
        "str_pad"
    ] = """
    Pad %(side)s side of strings in the Series/Index.

    Equivalent to :meth:`str.%(method)s`.

    Parameters
    ----------
    width : int
        Minimum width of resulting string; additional characters will be filled
        with ``fillchar``.
    fillchar : str
        Additional character for filling, default is whitespace.

    Returns
    -------
    Series/Index of objects.

    Examples
    --------
    For Series.str.center:

    >>> ser = pd.Series(['dog', 'bird', 'mouse'])
    >>> ser.str.center(8, fillchar='.')
    0   ..dog...
    1   ..bird..
    2   .mouse..
    dtype: object

    For Series.str.ljust:

    >>> ser = pd.Series(['dog', 'bird', 'mouse'])
    >>> ser.str.ljust(8, fillchar='.')
    0   dog.....
    1   bird....
    2   mouse...
    dtype: object

    For Series.str.rjust:

    >>> ser = pd.Series(['dog', 'bird', 'mouse'])
    >>> ser.str.rjust(8, fillchar='.')
    0   .....dog
    1   ....bird
    2   ...mouse
    dtype: object
    """

    @Appender(_shared_docs["str_pad"] % {"side": "left and right", "method": "center"})
    @forbid_nonstring_types(["bytes"])
    def center(self, width: int, fillchar: str = " "):
        return self.pad(width, side="both", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "right", "method": "ljust"})
    @forbid_nonstring_types(["bytes"])
    def ljust(self, width: int, fillchar: str = " "):
        return self.pad(width, side="right", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "left", "method": "rjust"})
    @forbid_nonstring_types(["bytes"])
    def rjust(self, width: int, fillchar: str = " "):
        return self.pad(width, side="left", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def zfill(self, width: int):
        """
        Pad strings in the Series/Index by prepending '0' characters.

        Strings in the Series/Index are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the Series/Index with length greater or equal to `width` are
        unchanged.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less
            than `width` be prepended with '0' characters.

        Returns
        -------
        Series/Index of objects.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character.
        Series.str.pad : Fills the specified sides of strings with an arbitrary
            character.
        Series.str.center : Fills both sides of strings with an arbitrary
            character.

        Notes
        -----
        Differs from :meth:`str.zfill` which has special handling
        for '+'/'-' in the string.

        Examples
        --------
        >>> s = pd.Series(['-1', '1', '1000', 10, np.nan])
        >>> s
        0      -1
        1       1
        2    1000
        3      10
        4     NaN
        dtype: object

        Note that ``10`` and ``NaN`` are not strings, therefore they are
        converted to ``NaN``. The minus sign in ``'-1'`` is treated as a
        special character and the zero is added to the right of it
        (:meth:`str.zfill` would have moved it to the left). ``1000``
        remains unchanged as it is longer than `width`.

        >>> s.str.zfill(3)
        0     -01
        1     001
        2    1000
        3     NaN
        4     NaN
        dtype: object
        """
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        return self._wrap_result(result)

    def slice(self, start=None, stop=None, step=None):
        """
        Slice substrings from each element in the Series or Index.

        Parameters
        ----------
        start : int, optional
            Start position for slice operation.
        stop : int, optional
            Stop position for slice operation.
        step : int, optional
            Step size for slice operation.

        Returns
        -------
        Series or Index of object
            Series or Index from sliced substring from original string object.

        See Also
        --------
        Series.str.slice_replace : Replace a slice with a string.
        Series.str.get : Return element at position.
            Equivalent to `Series.str.slice(start=i, stop=i+1)` with `i`
            being the position.

        Examples
        --------
        >>> s = pd.Series(["koala", "dog", "chameleon"])
        >>> s
        0        koala
        1          dog
        2    chameleon
        dtype: object

        >>> s.str.slice(start=1)
        0        oala
        1          og
        2    hameleon
        dtype: object

        >>> s.str.slice(start=-1)
        0           a
        1           g
        2           n
        dtype: object

        >>> s.str.slice(stop=2)
        0    ko
        1    do
        2    ch
        dtype: object

        >>> s.str.slice(step=2)
        0      kaa
        1       dg
        2    caeen
        dtype: object

        >>> s.str.slice(start=0, stop=5, step=3)
        0    kl
        1     d
        2    cm
        dtype: object

        Equivalent behaviour to:

        >>> s.str[0:5:3]
        0    kl
        1     d
        2    cm
        dtype: object
        """
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def slice_replace(self, start=None, stop=None, repl=None):
        """
        Replace a positional slice of a string with another value.

        Parameters
        ----------
        start : int, optional
            Left index position to use for the slice. If not specified (None),
            the slice is unbounded on the left, i.e. slice from the start
            of the string.
        stop : int, optional
            Right index position to use for the slice. If not specified (None),
            the slice is unbounded on the right, i.e. slice until the
            end of the string.
        repl : str, optional
            String for replacement. If not specified (None), the sliced region
            is replaced with an empty string.

        Returns
        -------
        Series or Index
            Same type as the original object.

        See Also
        --------
        Series.str.slice : Just slicing without replacement.

        Examples
        --------
        >>> s = pd.Series(['a', 'ab', 'abc', 'abdc', 'abcde'])
        >>> s
        0        a
        1       ab
        2      abc
        3     abdc
        4    abcde
        dtype: object

        Specify just `start`, meaning replace `start` until the end of the
        string with `repl`.

        >>> s.str.slice_replace(1, repl='X')
        0    aX
        1    aX
        2    aX
        3    aX
        4    aX
        dtype: object

        Specify just `stop`, meaning the start of the string to `stop` is replaced
        with `repl`, and the rest of the string is included.

        >>> s.str.slice_replace(stop=2, repl='X')
        0       X
        1       X
        2      Xc
        3     Xdc
        4    Xcde
        dtype: object

        Specify `start` and `stop`, meaning the slice from `start` to `stop` is
        replaced with `repl`. Everything before or after `start` and `stop` is
        included as is.

        >>> s.str.slice_replace(start=1, stop=3, repl='X')
        0      aX
        1      aX
        2      aX
        3     aXc
        4    aXde
        dtype: object
        """
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def decode(
        self, encoding, errors: str = "strict", dtype: str | DtypeObj | None = None
    ):
        """
        Decode character string in the Series/Index using indicated encoding.

        Equivalent to :meth:`str.decode` in python2 and :meth:`bytes.decode` in
        python3.

        Parameters
        ----------
        encoding : str
        errors : str, optional
            Specifies the error handling scheme.
            Possible values are those supported by :meth:`bytes.decode`.
        dtype : str or dtype, optional
            The dtype of the result. When not ``None``, must be either a string or
            object dtype. When ``None``, the dtype of the result is determined by
            ``pd.options.future.infer_string``.

            .. versionadded:: 2.3.0

        Returns
        -------
        Series or Index

        Examples
        --------
        For Series:

        >>> ser = pd.Series([b'cow', b'123', b'()'])
        >>> ser.str.decode('ascii')
        0   cow
        1   123
        2   ()
        dtype: object
        """
        if dtype is not None and not is_string_dtype(dtype):
            raise ValueError(f"dtype must be string or object, got {dtype=}")
        if dtype is None and get_option("future.infer_string"):
            dtype = "str"
        # TODO: Add a similar _bytes interface.
        if encoding in _cpython_optimized_decoders:
            # CPython optimized implementation
            f = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        arr = self._data.array
        result = arr._str_map(f)
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def encode(self, encoding, errors: str = "strict"):
        """
        Encode character string in the Series/Index using indicated encoding.

        Equivalent to :meth:`str.encode`.

        Parameters
        ----------
        encoding : str
        errors : str, optional

        Returns
        -------
        Series/Index of objects

        Examples
        --------
        >>> ser = pd.Series(['cow', '123', '()'])
        >>> ser.str.encode(encoding='ascii')
        0     b'cow'
        1     b'123'
        2      b'()'
        dtype: object
        """
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    _shared_docs[
        "str_strip"
    ] = r"""
    Remove %(position)s characters.

    Strip whitespaces (including newlines) or a set of specified characters
    from each string in the Series/Index from %(side)s.
    Replaces any non-strings in Series with NaNs.
    Equivalent to :meth:`str.%(method)s`.

    Parameters
    ----------
    to_strip : str or None, default None
        Specifying the set of characters to be removed.
        All combinations of this set of characters will be stripped.
        If None then whitespaces are removed.

    Returns
    -------
    Series or Index of object

    See Also
    --------
    Series.str.strip : Remove leading and trailing characters in Series/Index.
    Series.str.lstrip : Remove leading characters in Series/Index.
    Series.str.rstrip : Remove trailing characters in Series/Index.

    Examples
    --------
    >>> s = pd.Series(['1. Ant.  ', '2. Bee!\n', '3. Cat?\t', np.nan, 10, True])
    >>> s
    0    1. Ant.
    1    2. Bee!\n
    2    3. Cat?\t
    3          NaN
    4           10
    5         True
    dtype: object

    >>> s.str.strip()
    0    1. Ant.
    1    2. Bee!
    2    3. Cat?
    3        NaN
    4        NaN
    5        NaN
    dtype: object

    >>> s.str.lstrip('123.')
    0    Ant.
    1    Bee!\n
    2    Cat?\t
    3       NaN
    4       NaN
    5       NaN
    dtype: object

    >>> s.str.rstrip('.!? \n\t')
    0    1. Ant
    1    2. Bee
    2    3. Cat
    3       NaN
    4       NaN
    5       NaN
    dtype: object

    >>> s.str.strip('123.!? \n\t')
    0    Ant
    1    Bee
    2    Cat
    3    NaN
    4    NaN
    5    NaN
    dtype: object
    """

    @Appender(
        _shared_docs["str_strip"]
        % {
            "side": "left and right sides",
            "method": "strip",
            "position": "leading and trailing",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def strip(self, to_strip=None):
        result = self._data.array._str_strip(to_strip)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_strip"]
        % {"side": "left side", "method": "lstrip", "position": "leading"}
    )
    @forbid_nonstring_types(["bytes"])
    def lstrip(self, to_strip=None):
        result = self._data.array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_strip"]
        % {"side": "right side", "method": "rstrip", "position": "trailing"}
    )
    @forbid_nonstring_types(["bytes"])
    def rstrip(self, to_strip=None):
        result = self._data.array._str_rstrip(to_strip)
        return self._wrap_result(result)

    _shared_docs[
        "str_removefix"
    ] = r"""
    Remove a %(side)s from an object series.

    If the %(side)s is not present, the original string will be returned.

    Parameters
    ----------
    %(side)s : str
        Remove the %(side)s of the string.

    Returns
    -------
    Series/Index: object
        The Series or Index with given %(side)s removed.

    See Also
    --------
    Series.str.remove%(other_side)s : Remove a %(other_side)s from an object series.

    Examples
    --------
    >>> s = pd.Series(["str_foo", "str_bar", "no_prefix"])
    >>> s
    0    str_foo
    1    str_bar
    2    no_prefix
    dtype: object
    >>> s.str.removeprefix("str_")
    0    foo
    1    bar
    2    no_prefix
    dtype: object

    >>> s = pd.Series(["foo_str", "bar_str", "no_suffix"])
    >>> s
    0    foo_str
    1    bar_str
    2    no_suffix
    dtype: object
    >>> s.str.removesuffix("_str")
    0    foo
    1    bar
    2    no_suffix
    dtype: object
    """

    @Appender(
        _shared_docs["str_removefix"] % {"side": "prefix", "other_side": "suffix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removeprefix(self, prefix: str):
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_removefix"] % {"side": "suffix", "other_side": "prefix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removesuffix(self, suffix: str):
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def wrap(self, width: int, **kwargs):
        r"""
        Wrap strings in Series/Index at specified line width.

        This method has the same keyword parameters and defaults as
        :class:`textwrap.TextWrapper`.

        Parameters
        ----------
        width : int
            Maximum line width.
        expand_tabs : bool, optional
            If True, tab characters will be expanded to spaces (default: True).
        replace_whitespace : bool, optional
            If True, each whitespace character (as defined by string.whitespace)
            remaining after tab expansion will be replaced by a single space
            (default: True).
        drop_whitespace : bool, optional
            If True, whitespace that, after wrapping, happens to end up at the
            beginning or end of a line is dropped (default: True).
        break_long_words : bool, optional
            If True, then words longer than width will be broken in order to ensure
            that no lines are longer than width. If it is false, long words will
            not be broken, and some lines may be longer than width (default: True).
        break_on_hyphens : bool, optional
            If True, wrapping will occur preferably on whitespace and right after
            hyphens in compound words, as it is customary in English. If false,
            only whitespaces will be considered as potentially good places for line
            breaks, but you need to set break_long_words to false if you want truly
            insecable words (default: True).

        Returns
        -------
        Series or Index

        Notes
        -----
        Internally, this method uses a :class:`textwrap.TextWrapper` instance with
        default settings. To achieve behavior matching R's stringr library str_wrap
        function, use the arguments:

        - expand_tabs = False
        - replace_whitespace = True
        - drop_whitespace = True
        - break_long_words = False
        - break_on_hyphens = False

        Examples
        --------
        >>> s = pd.Series(['line to be wrapped', 'another line to be wrapped'])
        >>> s.str.wrap(12)
        0             line to be\nwrapped
        1    another line\nto be\nwrapped
        dtype: object
        """
        result = self._data.array._str_wrap(width, **kwargs)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def get_dummies(self, sep: str = "|"):
        """
        Return DataFrame of dummy/indicator variables for Series.

        Each string in Series is split by sep and returned as a DataFrame
        of dummy/indicator variables.

        Parameters
        ----------
        sep : str, default "|"
            String to split on.

        Returns
        -------
        DataFrame
            Dummy variables corresponding to values of the Series.

        See Also
        --------
        get_dummies : Convert categorical variable into dummy/indicator
            variables.

        Examples
        --------
        >>> pd.Series(['a|b', 'a', 'a|c']).str.get_dummies()
           a  b  c
        0  1  1  0
        1  1  0  0
        2  1  0  1

        >>> pd.Series(['a|b', np.nan, 'a|c']).str.get_dummies()
           a  b  c
        0  1  1  0
        1  0  0  0
        2  1  0  1
        """
        # we need to cast to Series of strings as only that has all
        # methods available for making the dummies...
        result, name = self._data.array._str_get_dummies(sep)
        return self._wrap_result(
            result,
            name=name,
            expand=True,
            returns_string=False,
        )

    @forbid_nonstring_types(["bytes"])
    def translate(self, table):
        """
        Map all characters in the string through the given mapping table.

        Equivalent to standard :meth:`str.translate`.

        Parameters
        ----------
        table : dict
            Table is a mapping of Unicode ordinals to Unicode ordinals, strings, or
            None. Unmapped characters are left untouched.
            Characters mapped to None are deleted. :meth:`str.maketrans` is a
            helper function for making translation tables.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> ser = pd.Series(["El niño", "Françoise"])
        >>> mytable = str.maketrans({'ñ': 'n', 'ç': 'c'})
        >>> ser.str.translate(mytable)
        0   El nino
        1   Francoise
        dtype: object
        """
        result = self._data.array._str_translate(table)
        dtype = object if self._data.dtype == "object" else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def count(self, pat, flags: int = 0):
        r"""
        Count occurrences of pattern in each string of the Series/Index.

        This function is used to count the number of times a particular regex
        pattern is repeated in each of the string elements of the
        :class:`~pandas.Series`.

        Parameters
        ----------
        pat : str
            Valid regular expression.
        flags : int, default 0, meaning no flags
            Flags for the `re` module. For a complete list, `see here
            <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
        **kwargs
            For compatibility with other string methods. Not used.

        Returns
        -------
        Series or Index
            Same type as the calling object containing the integer counts.

        See Also
        --------
        re : Standard library module for regular expressions.
        str.count : Standard library version, without regular expression support.

        Notes
        -----
        Some characters need to be escaped when passing in `pat`.
        eg. ``'$'`` has a special meaning in regex and must be escaped when
        finding this literal character.

        Examples
        --------
        >>> s = pd.Series(['A', 'B', 'Aaba', 'Baca', np.nan, 'CABA', 'cat'])
        >>> s.str.count('a')
        0    0.0
        1    0.0
        2    2.0
        3    2.0
        4    NaN
        5    0.0
        6    1.0
        dtype: float64

        Escape ``'$'`` to find the literal dollar sign.

        >>> s = pd.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])
        >>> s.str.count('\\$')
        0    1
        1    0
        2    1
        3    2
        4    2
        5    0
        dtype: int64

        This is also available on Index

        >>> pd.Index(['A', 'A', 'Aaba', 'cat']).str.count('a')
        Index([0, 0, 2, 1], dtype='int64')
        """
        result = self._data.array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def startswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ) -> Series | Index:
        """
        Test if the start of each string element matches a pattern.

        Equivalent to :meth:`str.startswith`.

        Parameters
        ----------
        pat : str or tuple[str, ...]
            Character sequence or tuple of strings. Regular expressions are not
            accepted.
        na : scalar, optional
            Object shown if element tested is not a string. The default depends
            on dtype of the array. For object-dtype, ``numpy.nan`` is used.
            For the nullable ``StringDtype``, ``pandas.NA`` is used.
            For the ``"str"`` dtype, ``False`` is used.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given pattern matches
            the start of each string element.

        See Also
        --------
        str.startswith : Python standard library string method.
        Series.str.endswith : Same as startswith, but tests the end of string.
        Series.str.contains : Tests if string element contains a pattern.

        Examples
        --------
        >>> s = pd.Series(['bat', 'Bear', 'cat', np.nan])
        >>> s
        0     bat
        1    Bear
        2     cat
        3     NaN
        dtype: object

        >>> s.str.startswith('b')
        0     True
        1    False
        2    False
        3      NaN
        dtype: object

        >>> s.str.startswith(('b', 'B'))
        0     True
        1     True
        2    False
        3      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN`.

        >>> s.str.startswith('b', na=False)
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_startswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def endswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ) -> Series | Index:
        """
        Test if the end of each string element matches a pattern.

        Equivalent to :meth:`str.endswith`.

        Parameters
        ----------
        pat : str or tuple[str, ...]
            Character sequence or tuple of strings. Regular expressions are not
            accepted.
        na : scalar, optional
            Object shown if element tested is not a string. The default depends
            on dtype of the array. For object-dtype, ``numpy.nan`` is used.
            For the nullable ``StringDtype``, ``pandas.NA`` is used.
            For the ``"str"`` dtype, ``False`` is used.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given pattern matches
            the end of each string element.

        See Also
        --------
        str.endswith : Python standard library string method.
        Series.str.startswith : Same as endswith, but tests the start of string.
        Series.str.contains : Tests if string element contains a pattern.

        Examples
        --------
        >>> s = pd.Series(['bat', 'bear', 'caT', np.nan])
        >>> s
        0     bat
        1    bear
        2     caT
        3     NaN
        dtype: object

        >>> s.str.endswith('t')
        0     True
        1    False
        2    False
        3      NaN
        dtype: object

        >>> s.str.endswith(('t', 'T'))
        0     True
        1    False
        2     True
        3      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN`.

        >>> s.str.endswith('t', na=False)
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_endswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def findall(self, pat, flags: int = 0):
        """
        Find all occurrences of pattern or regular expression in the Series/Index.

        Equivalent to applying :func:`re.findall` to all the elements in the
        Series/Index.

        Parameters
        ----------
        pat : str
            Pattern or regular expression.
        flags : int, default 0
            Flags from ``re`` module, e.g. `re.IGNORECASE` (default is 0, which
            means no flags).

        Returns
        -------
        Series/Index of lists of strings
            All non-overlapping matches of pattern or regular expression in each
            string of this Series/Index.

        See Also
        --------
        count : Count occurrences of pattern or regular expression in each string
            of the Series/Index.
        extractall : For each string in the Series, extract groups from all matches
            of regular expression and return a DataFrame with one row for each
            match and one column for each group.
        re.findall : The equivalent ``re`` function to all non-overlapping matches
            of pattern or regular expression in string, as a list of strings.

        Examples
        --------
        >>> s = pd.Series(['Lion', 'Monkey', 'Rabbit'])

        The search for the pattern 'Monkey' returns one match:

        >>> s.str.findall('Monkey')
        0          []
        1    [Monkey]
        2          []
        dtype: object

        On the other hand, the search for the pattern 'MONKEY' doesn't return any
        match:

        >>> s.str.findall('MONKEY')
        0    []
        1    []
        2    []
        dtype: object

        Flags can be added to the pattern or regular expression. For instance,
        to find the pattern 'MONKEY' ignoring the case:

        >>> import re
        >>> s.str.findall('MONKEY', flags=re.IGNORECASE)
        0          []
        1    [Monkey]
        2          []
        dtype: object

        When the pattern matches more than one string in the Series, all matches
        are returned:

        >>> s.str.findall('on')
        0    [on]
        1    [on]
        2      []
        dtype: object

        Regular expressions are supported too. For instance, the search for all the
        strings ending with the word 'on' is shown next:

        >>> s.str.findall('on$')
        0    [on]
        1      []
        2      []
        dtype: object

        If the pattern is found more than once in the same string, then a list of
        multiple strings is returned:

        >>> s.str.findall('b')
        0        []
        1        []
        2    [b, b]
        dtype: object
        """
        result = self._data.array._str_findall(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def extract(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> DataFrame | Series | Index:
        r"""
        Extract capture groups in the regex `pat` as columns in a DataFrame.

        For each subject string in the Series, extract groups from the
        first match of regular expression `pat`.

        Parameters
        ----------
        pat : str
            Regular expression pattern with capturing groups.
        flags : int, default 0 (no flags)
            Flags from the ``re`` module, e.g. ``re.IGNORECASE``, that
            modify regular expression matching for things like case,
            spaces, etc. For more details, see :mod:`re`.
        expand : bool, default True
            If True, return DataFrame with one column per capture group.
            If False, return a Series/Index if there is one capture group
            or DataFrame if there are multiple capture groups.

        Returns
        -------
        DataFrame or Series or Index
            A DataFrame with one row for each subject string, and one
            column for each group. Any capture group names in regular
            expression pat will be used for column names; otherwise
            capture group numbers will be used. The dtype of each result
            column is always object, even when no match is found. If
            ``expand=False`` and pat has only one capture group, then
            return a Series (if subject is a Series) or Index (if subject
            is an Index).

        See Also
        --------
        extractall : Returns all matches (not just the first match).

        Examples
        --------
        A pattern with two groups will return a DataFrame with two columns.
        Non-matches will be NaN.

        >>> s = pd.Series(['a1', 'b2', 'c3'])
        >>> s.str.extract(r'([ab])(\d)')
            0    1
        0    a    1
        1    b    2
        2  NaN  NaN

        A pattern may contain optional groups.

        >>> s.str.extract(r'([ab])?(\d)')
            0  1
        0    a  1
        1    b  2
        2  NaN  3

        Named groups will become column names in the result.

        >>> s.str.extract(r'(?P<letter>[ab])(?P<digit>\d)')
        letter digit
        0      a     1
        1      b     2
        2    NaN   NaN

        A pattern with one group will return a DataFrame with one column
        if expand=True.

        >>> s.str.extract(r'[ab](\d)', expand=True)
            0
        0    1
        1    2
        2  NaN

        A pattern with one group will return a Series if expand=False.

        >>> s.str.extract(r'[ab](\d)', expand=False)
        0      1
        1      2
        2    NaN
        dtype: object
        """
        from pandas import DataFrame

        if not isinstance(expand, bool):
            raise ValueError("expand must be True or False")

        regex = re.compile(pat, flags=flags)
        if regex.groups == 0:
            raise ValueError("pattern contains no capture groups")

        if not expand and regex.groups > 1 and isinstance(self._data, ABCIndex):
            raise ValueError("only one regex group is supported with Index")

        obj = self._data
        result_dtype = _result_dtype(obj)

        returns_df = regex.groups > 1 or expand

        if returns_df:
            name = None
            columns = _get_group_names(regex)

            if obj.array.size == 0:
                result = DataFrame(columns=columns, dtype=result_dtype)

            else:
                result_list = self._data.array._str_extract(
                    pat, flags=flags, expand=returns_df
                )

                result_index: Index | None
                if isinstance(obj, ABCSeries):
                    result_index = obj.index
                else:
                    result_index = None

                result = DataFrame(
                    result_list, columns=columns, index=result_index, dtype=result_dtype
                )

        else:
            name = _get_single_group_name(regex)
            result = self._data.array._str_extract(pat, flags=flags, expand=returns_df)
        return self._wrap_result(result, name=name, dtype=result_dtype)

    @forbid_nonstring_types(["bytes"])
    def extractall(self, pat, flags: int = 0) -> DataFrame:
        r"""
        Extract capture groups in the regex `pat` as columns in DataFrame.

        For each subject string in the Series, extract groups from all
        matches of regular expression pat. When each subject string in the
        Series has exactly one match, extractall(pat).xs(0, level='match')
        is the same as extract(pat).

        Parameters
        ----------
        pat : str
            Regular expression pattern with capturing groups.
        flags : int, default 0 (no flags)
            A ``re`` module flag, for example ``re.IGNORECASE``. These allow
            to modify regular expression matching for things like case, spaces,
            etc. Multiple flags can be combined with the bitwise OR operator,
            for example ``re.IGNORECASE | re.MULTILINE``.

        Returns
        -------
        DataFrame
            A ``DataFrame`` with one row for each match, and one column for each
            group. Its rows have a ``MultiIndex`` with first levels that come from
            the subject ``Series``. The last level is named 'match' and indexes the
            matches in each item of the ``Series``. Any capture group names in
            regular expression pat will be used for column names; otherwise capture
            group numbers will be used.

        See Also
        --------
        extract : Returns first match only (not all matches).

        Examples
        --------
        A pattern with one group will return a DataFrame with one column.
        Indices with no matches will not appear in the result.

        >>> s = pd.Series(["a1a2", "b1", "c1"], index=["A", "B", "C"])
        >>> s.str.extractall(r"[ab](\d)")
                0
        match
        A 0      1
          1      2
        B 0      1

        Capture group names are used for column names of the result.

        >>> s.str.extractall(r"[ab](?P<digit>\d)")
                digit
        match
        A 0         1
          1         2
        B 0         1

        A pattern with two groups will return a DataFrame with two columns.

        >>> s.str.extractall(r"(?P<letter>[ab])(?P<digit>\d)")
                letter digit
        match
        A 0          a     1
          1          a     2
        B 0          b     1

        Optional groups that do not match are NaN in the result.

        >>> s.str.extractall(r"(?P<letter>[ab])?(?P<digit>\d)")
                letter digit
        match
        A 0          a     1
          1          a     2
        B 0          b     1
        C 0        NaN     1
        """
        # TODO: dispatch
        return str_extractall(self._orig, pat, flags)

    _shared_docs[
        "find"
    ] = """
    Return %(side)s indexes in each strings in the Series/Index.

    Each of returned indexes corresponds to the position where the
    substring is fully contained between [start:end]. Return -1 on
    failure. Equivalent to standard :meth:`str.%(method)s`.

    Parameters
    ----------
    sub : str
        Substring being searched.
    start : int
        Left edge index.
    end : int
        Right edge index.

    Returns
    -------
    Series or Index of int.

    See Also
    --------
    %(also)s

    Examples
    --------
    For Series.str.find:

    >>> ser = pd.Series(["cow_", "duck_", "do_ve"])
    >>> ser.str.find("_")
    0   3
    1   4
    2   2
    dtype: int64

    For Series.str.rfind:

    >>> ser = pd.Series(["_cow_", "duck_", "do_v_e"])
    >>> ser.str.rfind("_")
    0   4
    1   4
    2   4
    dtype: int64
    """

    @Appender(
        _shared_docs["find"]
        % {
            "side": "lowest",
            "method": "find",
            "also": "rfind : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def find(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_find(sub, start, end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["find"]
        % {
            "side": "highest",
            "method": "rfind",
            "also": "find : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rfind(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_rfind(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def normalize(self, form):
        """
        Return the Unicode normal form for the strings in the Series/Index.

        For more information on the forms, see the
        :func:`unicodedata.normalize`.

        Parameters
        ----------
        form : {'NFC', 'NFKC', 'NFD', 'NFKD'}
            Unicode form.

        Returns
        -------
        Series/Index of objects

        Examples
        --------
        >>> ser = pd.Series(['ñ'])
        >>> ser.str.normalize('NFC') == ser.str.normalize('NFD')
        0   False
        dtype: bool
        """
        result = self._data.array._str_normalize(form)
        return self._wrap_result(result)

    _shared_docs[
        "index"
    ] = """
    Return %(side)s indexes in each string in Series/Index.

    Each of the returned indexes corresponds to the position where the
    substring is fully contained between [start:end]. This is the same
    as ``str.%(similar)s`` except instead of returning -1, it raises a
    ValueError when the substring is not found. Equivalent to standard
    ``str.%(method)s``.

    Parameters
    ----------
    sub : str
        Substring being searched.
    start : int
        Left edge index.
    end : int
        Right edge index.

    Returns
    -------
    Series or Index of object

    See Also
    --------
    %(also)s

    Examples
    --------
    For Series.str.index:

    >>> ser = pd.Series(["horse", "eagle", "donkey"])
    >>> ser.str.index("e")
    0   4
    1   0
    2   4
    dtype: int64

    For Series.str.rindex:

    >>> ser = pd.Series(["Deer", "eagle", "Sheep"])
    >>> ser.str.rindex("e")
    0   2
    1   4
    2   3
    dtype: int64
    """

    @Appender(
        _shared_docs["index"]
        % {
            "side": "lowest",
            "similar": "find",
            "method": "index",
            "also": "rindex : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def index(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["index"]
        % {
            "side": "highest",
            "similar": "rfind",
            "method": "rindex",
            "also": "index : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rindex(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def len(self):
        """
        Compute the length of each element in the Series/Index.

        The element may be a sequence (such as a string, tuple or list) or a collection
        (such as a dictionary).

        Returns
        -------
        Series or Index of int
            A Series or Index of integer values indicating the length of each
            element in the Series or Index.

        See Also
        --------
        str.len : Python built-in function returning the length of an object.
        Series.size : Returns the length of the Series.

        Examples
        --------
        Returns the length (number of characters) in a string. Returns the
        number of entries for dictionaries, lists or tuples.

        >>> s = pd.Series(['dog',
        ...                 '',
        ...                 5,
        ...                 {'foo' : 'bar'},
        ...                 [2, 3, 5, 7],
        ...                 ('one', 'two', 'three')])
        >>> s
        0                  dog
        1
        2                    5
        3       {'foo': 'bar'}
        4         [2, 3, 5, 7]
        5    (one, two, three)
        dtype: object
        >>> s.str.len()
        0    3.0
        1    0.0
        2    NaN
        3    1.0
        4    4.0
        5    3.0
        dtype: float64
        """
        result = self._data.array._str_len()
        return self._wrap_result(result, returns_string=False)

    _shared_docs[
        "casemethods"
    ] = """
    Convert strings in the Series/Index to %(type)s.
    %(version)s
    Equivalent to :meth:`str.%(method)s`.

    Returns
    -------
    Series or Index of object

    See Also
    --------
    Series.str.lower : Converts all characters to lowercase.
    Series.str.upper : Converts all characters to uppercase.
    Series.str.title : Converts first character of each word to uppercase and
        remaining to lowercase.
    Series.str.capitalize : Converts first character to uppercase and
        remaining to lowercase.
    Series.str.swapcase : Converts uppercase to lowercase and lowercase to
        uppercase.
    Series.str.casefold: Removes all case distinctions in the string.

    Examples
    --------
    >>> s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
    >>> s
    0                 lower
    1              CAPITALS
    2    this is a sentence
    3              SwApCaSe
    dtype: object

    >>> s.str.lower()
    0                 lower
    1              capitals
    2    this is a sentence
    3              swapcase
    dtype: object

    >>> s.str.upper()
    0                 LOWER
    1              CAPITALS
    2    THIS IS A SENTENCE
    3              SWAPCASE
    dtype: object

    >>> s.str.title()
    0                 Lower
    1              Capitals
    2    This Is A Sentence
    3              Swapcase
    dtype: object

    >>> s.str.capitalize()
    0                 Lower
    1              Capitals
    2    This is a sentence
    3              Swapcase
    dtype: object

    >>> s.str.swapcase()
    0                 LOWER
    1              capitals
    2    THIS IS A SENTENCE
    3              sWaPcAsE
    dtype: object
    """
    # Types:
    #   cases:
    #       upper, lower, title, capitalize, swapcase, casefold
    #   boolean:
    #     isalpha, isnumeric isalnum isdigit isdecimal isspace islower isupper istitle
    # _doc_args holds dict of strings to use in substituting casemethod docs
    _doc_args: dict[str, dict[str, str]] = {}
    _doc_args["lower"] = {"type": "lowercase", "method": "lower", "version": ""}
    _doc_args["upper"] = {"type": "uppercase", "method": "upper", "version": ""}
    _doc_args["title"] = {"type": "titlecase", "method": "title", "version": ""}
    _doc_args["capitalize"] = {
        "type": "be capitalized",
        "method": "capitalize",
        "version": "",
    }
    _doc_args["swapcase"] = {
        "type": "be swapcased",
        "method": "swapcase",
        "version": "",
    }
    _doc_args["casefold"] = {
        "type": "be casefolded",
        "method": "casefold",
        "version": "",
    }

    @Appender(_shared_docs["casemethods"] % _doc_args["lower"])
    @forbid_nonstring_types(["bytes"])
    def lower(self):
        result = self._data.array._str_lower()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["upper"])
    @forbid_nonstring_types(["bytes"])
    def upper(self):
        result = self._data.array._str_upper()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["title"])
    @forbid_nonstring_types(["bytes"])
    def title(self):
        result = self._data.array._str_title()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["capitalize"])
    @forbid_nonstring_types(["bytes"])
    def capitalize(self):
        result = self._data.array._str_capitalize()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["swapcase"])
    @forbid_nonstring_types(["bytes"])
    def swapcase(self):
        result = self._data.array._str_swapcase()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["casefold"])
    @forbid_nonstring_types(["bytes"])
    def casefold(self):
        result = self._data.array._str_casefold()
        return self._wrap_result(result)

    _shared_docs[
        "ismethods"
    ] = """
    Check whether all characters in each string are %(type)s.

    This is equivalent to running the Python string method
    :meth:`str.%(method)s` for each element of the Series/Index. If a string
    has zero characters, ``False`` is returned for that check.

    Returns
    -------
    Series or Index of bool
        Series or Index of boolean values with the same length as the original
        Series/Index.

    See Also
    --------
    Series.str.isalpha : Check whether all characters are alphabetic.
    Series.str.isnumeric : Check whether all characters are numeric.
    Series.str.isalnum : Check whether all characters are alphanumeric.
    Series.str.isdigit : Check whether all characters are digits.
    Series.str.isdecimal : Check whether all characters are decimal.
    Series.str.isspace : Check whether all characters are whitespace.
    Series.str.islower : Check whether all characters are lowercase.
    Series.str.isupper : Check whether all characters are uppercase.
    Series.str.istitle : Check whether all characters are titlecase.

    Examples
    --------
    **Checks for Alphabetic and Numeric Characters**

    >>> s1 = pd.Series(['one', 'one1', '1', ''])

    >>> s1.str.isalpha()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    >>> s1.str.isnumeric()
    0    False
    1    False
    2     True
    3    False
    dtype: bool

    >>> s1.str.isalnum()
    0     True
    1     True
    2     True
    3    False
    dtype: bool

    Note that checks against characters mixed with any additional punctuation
    or whitespace will evaluate to false for an alphanumeric check.

    >>> s2 = pd.Series(['A B', '1.5', '3,000'])
    >>> s2.str.isalnum()
    0    False
    1    False
    2    False
    dtype: bool

    **More Detailed Checks for Numeric Characters**

    There are several different but overlapping sets of numeric characters that
    can be checked for.

    >>> s3 = pd.Series(['23', '³', '⅕', ''])

    The ``s3.str.isdecimal`` method checks for characters used to form numbers
    in base 10.

    >>> s3.str.isdecimal()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    The ``s.str.isdigit`` method is the same as ``s3.str.isdecimal`` but also
    includes special digits, like superscripted and subscripted digits in
    unicode.

    >>> s3.str.isdigit()
    0     True
    1     True
    2    False
    3    False
    dtype: bool

    The ``s.str.isnumeric`` method is the same as ``s3.str.isdigit`` but also
    includes other characters that can represent quantities such as unicode
    fractions.

    >>> s3.str.isnumeric()
    0     True
    1     True
    2     True
    3    False
    dtype: bool

    **Checks for Whitespace**

    >>> s4 = pd.Series([' ', '\\t\\r\\n ', ''])
    >>> s4.str.isspace()
    0     True
    1     True
    2    False
    dtype: bool

    **Checks for Character Case**

    >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])

    >>> s5.str.islower()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    >>> s5.str.isupper()
    0    False
    1    False
    2     True
    3    False
    dtype: bool

    The ``s5.str.istitle`` method checks for whether all words are in title
    case (whether only the first letter of each word is capitalized). Words are
    assumed to be as any sequence of non-numeric characters separated by
    whitespace characters.

    >>> s5.str.istitle()
    0    False
    1     True
    2    False
    3    False
    dtype: bool
    """
    _doc_args["isalnum"] = {"type": "alphanumeric", "method": "isalnum"}
    _doc_args["isalpha"] = {"type": "alphabetic", "method": "isalpha"}
    _doc_args["isdigit"] = {"type": "digits", "method": "isdigit"}
    _doc_args["isspace"] = {"type": "whitespace", "method": "isspace"}
    _doc_args["islower"] = {"type": "lowercase", "method": "islower"}
    _doc_args["isupper"] = {"type": "uppercase", "method": "isupper"}
    _doc_args["istitle"] = {"type": "titlecase", "method": "istitle"}
    _doc_args["isnumeric"] = {"type": "numeric", "method": "isnumeric"}
    _doc_args["isdecimal"] = {"type": "decimal", "method": "isdecimal"}
    # force _noarg_wrapper return type with dtype=np.dtype(bool) (GH 29624)

    isalnum = _map_and_wrap(
        "isalnum", docstring=_shared_docs["ismethods"] % _doc_args["isalnum"]
    )
    isalpha = _map_and_wrap(
        "isalpha", docstring=_shared_docs["ismethods"] % _doc_args["isalpha"]
    )
    isdigit = _map_and_wrap(
        "isdigit", docstring=_shared_docs["ismethods"] % _doc_args["isdigit"]
    )
    isspace = _map_and_wrap(
        "isspace", docstring=_shared_docs["ismethods"] % _doc_args["isspace"]
    )
    islower = _map_and_wrap(
        "islower", docstring=_shared_docs["ismethods"] % _doc_args["islower"]
    )
    isupper = _map_and_wrap(
        "isupper", docstring=_shared_docs["ismethods"] % _doc_args["isupper"]
    )
    istitle = _map_and_wrap(
        "istitle", docstring=_shared_docs["ismethods"] % _doc_args["istitle"]
    )
    isnumeric = _map_and_wrap(
        "isnumeric", docstring=_shared_docs["ismethods"] % _doc_args["isnumeric"]
    )
    isdecimal = _map_and_wrap(
        "isdecimal", docstring=_shared_docs["ismethods"] % _doc_args["isdecimal"]
    )


def cat_safe(list_of_columns: list[npt.NDArray[np.object_]], sep: str):
    """
    Auxiliary function for :meth:`str.cat`.

    Same signature as cat_core, but handles TypeErrors in concatenation, which
    happen if the arrays in list_of columns have the wrong dtypes or content.

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    nd.array
        The concatenation of list_of_columns with sep.
    """
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        # if there are any non-string values (wrong dtype or hidden behind
        # object dtype), np.sum will fail; catch and return with better message
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ["string", "empty"]:
                raise TypeError(
                    "Concatenation requires list-likes containing only "
                    "strings (or missing values). Offending values found in "
                    f"column {dtype}"
                ) from None
    return result


def cat_core(list_of_columns: list, sep: str):
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    nd.array
        The concatenation of list_of_columns with sep.
    """
    if sep == "":
        # no need to interleave sep if it is empty
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)


def _result_dtype(arr):
    # workaround #27953
    # ideally we just pass `dtype=arr.dtype` unconditionally, but this fails
    # when the list of values is empty.
    from pandas.core.arrays.string_ import StringDtype

    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object


def _get_single_group_name(regex: re.Pattern) -> Hashable:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None


def _get_group_names(regex: re.Pattern) -> list[Hashable]:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : compiled regex

    Returns
    -------
    list of column labels
    """
    names = {v: k for k, v in regex.groupindex.items()}
    return [names.get(1 + i, i) for i in range(regex.groups)]


def str_extractall(arr, pat, flags: int = 0) -> DataFrame:
    regex = re.compile(pat, flags=flags)
    # the regex must contain capture groups.
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")

    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)

    columns = _get_group_names(regex)
    match_list = []
    index_list = []
    is_mi = arr.index.nlevels > 1

    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)

            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = [np.nan if group == "" else group for group in match_tuple]
                match_list.append(na_tuple)
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)

    from pandas import MultiIndex

    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ["match"])
    dtype = _result_dtype(arr)

    result = arr._constructor_expanddim(
        match_list, index=index, columns=columns, dtype=dtype
    )
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\convert_generation.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""
This converts GPT2 or T5 model to onnx with beam search operator.

Example 1: convert gpt2 model with beam search:
    python convert_generation.py -m gpt2 --output gpt2_beam_search.onnx

Example 2: convert gpt2 model with beam search containing specific cuda optimizations:
    python convert_generation.py -m gpt2 --output gpt2_beam_search.onnx --use_gpu               \
        --past_present_share_buffer --use_decoder_masked_attention

Example 3: convert gpt2 model with beam search with mixed precision and enable SkipLayerNorm strict mode:
    python convert_generation.py -m gpt2 --output gpt2_beam_search.onnx --use_gpu -p fp16 --use_sln_strict_mode

Example 4: convert T5 model with beam search in two steps:
    python -m models.t5.convert_to_onnx -m t5-small
    python convert_generation.py -m t5-small --model_type t5             \
        --decoder_onnx ./onnx_models/t5-small_decoder.onnx               \
        --encoder_decoder_init_onnx ./onnx_models/t5-small_encoder.onnx  \
        --output ./onnx_models/t5_small_beam_search.onnx

Example 5: convert T5 model with beam search. All in one step:
    python convert_generation.py -m t5-small --model_type t5 --output t5_small_beam_search.onnx

Example 6: convert T5 model with beam search containing specific cuda optimizations. All in one step:
    python convert_generation.py -m t5-small --model_type t5 --output t5_small_beam_search.onnx   \
        --use_gpu --past_present_share_buffer --use_decoder_masked_attention

Example 7: convert MT5 model with external data file like mt5-base-beamsearch.onnx.data in below example.
    python convert_generation.py -m google/mt5-base --model_type mt5 --output mt5-base-beamsearch.onnx -e

Example 8: convert gpt2 model with greedy search:
    python convert_generation.py -m gpt2 --output gpt2_greedy_search.onnx --num_beams 1 --num_return_sequences 1

Example 9: convert gpt2 model with sampling:
    python convert_generation.py -m gpt2 --output gpt2_sampling.onnx --num_beams 1 --num_return_sequences 1 --top_p 0.6
"""

import argparse
import logging
import math
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from benchmark_helper import Precision, setup_logger
from fusion_utils import NumpyHelper
from onnx import GraphProto, ModelProto, TensorProto
from onnx_model import OnnxModel
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    MT5Config,
    MT5ForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
)
from onnxruntime.transformers.models.gpt2.convert_to_onnx import (
    main as convert_gpt2_to_onnx,
)
from onnxruntime.transformers.models.gpt2.gpt2_helper import PRETRAINED_GPT2_MODELS
from onnxruntime.transformers.models.t5.convert_to_onnx import (
    export_onnx_models as export_t5_onnx_models,
)
from onnxruntime.transformers.models.t5.t5_helper import (
    PRETRAINED_MT5_MODELS,
    PRETRAINED_T5_MODELS,
)

logger = logging.getLogger("")


class GenerationType(Enum):
    BEAMSEARCH = "beam_search"
    GREEDYSEARCH = "greedy_search"
    SAMPLING = "sampling"

    def __str__(self):
        return self.value


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments

    Args:
        argv (Optional[List[str]], optional): _description_. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    input_group = parser.add_argument_group("Input options")

    input_group.add_argument(
        "-m",
        "--model_name_or_path",
        required=True,
        type=str,
        help="Pytorch model checkpoint path, or pretrained model name in the list: "
        + ", ".join(PRETRAINED_GPT2_MODELS + PRETRAINED_T5_MODELS + PRETRAINED_MT5_MODELS),
    )

    input_group.add_argument(
        "--model_type",
        required=False,
        type=str,
        default="gpt2",
        choices=["gpt2", "t5", "mt5"],
        help="Model type (default is gpt2) in the list: " + ", ".join(["gpt2", "t5", "mt5"]),
    )

    input_group.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    input_group.add_argument(
        "--decoder_onnx",
        required=False,
        type=str,
        default="",
        help="Path of onnx model for decoder. Specify it when you have exported the model.",
    )

    input_group.add_argument(
        "--encoder_decoder_init_onnx",
        required=False,
        type=str,
        default="",
        help="Path of ONNX model for encoder and decoder initialization. Specify it when you have exported the model.",
    )

    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Print more information",
    )
    parser.set_defaults(verbose=False)

    output_group = parser.add_argument_group("Output options")

    output_group.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output path for onnx model with beam search.",
    )

    output_group.add_argument(
        "-p",
        "--precision",
        required=False,
        type=str,
        default=Precision.FLOAT32.value,
        choices=[Precision.FLOAT32.value, Precision.FLOAT16.value],
        help="Precision of model to run. fp32 for full precision, fp16 for half or mixed precision",
    )

    output_group.add_argument(
        "-b",
        "--op_block_list",
        required=False,
        nargs="*",
        default=["auto"],
        help="Disable certain onnx operators when exporting model to onnx format. When using default"
        'value for gpt2 type of model fp16 precision, it will be set to ["Add", "LayerNormalization",'
        ' "SkipLayerNormalization", "FastGelu"]. Other situation, it will be set to []',
    )

    output_group.add_argument(
        "-e",
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="save external data for model > 2G",
    )
    output_group.set_defaults(use_external_data_format=False)

    output_group.add_argument(
        "-s",
        "--run_shape_inference",
        required=False,
        action="store_true",
        help="run shape inference",
    )
    output_group.set_defaults(run_shape_inference=False)

    output_group.add_argument(
        "-dpvs",
        "--disable_pad_vocab_size",
        required=False,
        action="store_true",
        help="Do not pad logits MatMul weight to be a multiple of 8 along the dimension where dim value is"
        " the vocab size. The logits MatMul may hence be of poor performance for fp16 precision.",
    )
    output_group.set_defaults(disable_pad_vocab_size=False)

    output_group.add_argument(
        "-dsgd",
        "--disable_separate_gpt2_decoder_for_init_run",
        required=False,
        action="store_true",
        help="Do not create separate decoder subgraphs for initial and remaining runs. This does not allow "
        "for optimizations based on sequence lengths in each subgraph",
    )
    output_group.set_defaults(disable_separate_gpt2_decoder_for_init_run=False)

    output_group.add_argument(
        "-i",
        "--disable_shared_initializers",
        required=False,
        action="store_true",
        help="do not share initializers in encoder and decoder for T5 or in the init decoder and decoder for "
        "GPT2. It will increase memory usage of t5/mt5/gpt2 models.",
    )
    output_group.set_defaults(disable_shared_initializers=False)

    output_group.add_argument(
        "--encoder_decoder_init",
        required=False,
        action="store_true",
        help="Add decoder initialization to encoder for T5 model. This is legacy format that will be deprecated.",
    )
    output_group.set_defaults(encoder_decoder_init=False)

    model_group = parser.add_argument_group("Beam search parameters that stored in the output model")

    model_group.add_argument(
        "--output_sequences_scores",
        required=False,
        action="store_true",
        help="output sequences scores",
    )
    model_group.set_defaults(output_sequences_scores=False)

    model_group.add_argument(
        "--output_token_scores",
        required=False,
        action="store_true",
        help="output token scores",
    )
    model_group.set_defaults(output_token_scores=False)

    model_group.add_argument("--early_stopping", required=False, action="store_true")
    model_group.set_defaults(early_stopping=False)

    model_group.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        required=False,
        default=0,
        help="No repeat ngram size",
    )

    model_group.add_argument(
        "--vocab_mask",
        required=False,
        action="store_true",
        help="Enable vocab_mask. This mask applies only to every generated token to filter some bad words.",
    )
    model_group.set_defaults(vocab_mask=False)

    model_group.add_argument(
        "--past_present_share_buffer",
        required=False,
        action="store_true",
        help="Use shared buffer for past and present, currently work for gpt2 greedy/sampling search.",
    )
    model_group.set_defaults(past_present_share_buffer=False)

    model_group.add_argument(
        "--use_decoder_masked_attention",
        required=False,
        action="store_true",
        help="Uses `DecoderMaskedSelfAttention` or `DecoderMaskedMultiHeadAttention` to optimize the decoding Attention computation. "
        "Must be used with `past_present_share_buffer`. Currently, only Attention head sizes of 32, 64 and 128 are supported.",
    )
    model_group.set_defaults(use_decoder_masked_attention=False)

    model_group.add_argument(
        "--prefix_vocab_mask",
        required=False,
        action="store_true",
        help="Enable prefix_vocab_mask. This mask can be used to filter bad words in the first generated token only",
    )
    model_group.set_defaults(prefix_vocab_mask=False)

    model_group.add_argument(
        "--custom_attention_mask",
        required=False,
        action="store_true",
        help="Enable custom_attention_mask. This mask can be used to replace default encoder attention mask",
    )
    model_group.set_defaults(custom_attention_mask=False)

    model_group.add_argument(
        "--presence_mask",
        required=False,
        action="store_true",
        help="Presence mask for custom sampling",
    )
    model_group.set_defaults(presence_mask=False)

    model_group.add_argument(
        "--seed",
        required=False,
        action="store_true",
        help="Random seed for sampling op",
    )
    model_group.set_defaults(seed=False)

    beam_parameters_group = parser.add_argument_group(
        "Beam search parameters not stored in the output model, for testing parity and performance"
    )

    beam_parameters_group.add_argument("--min_length", type=int, required=False, default=1, help="Min sequence length")

    beam_parameters_group.add_argument("--max_length", type=int, required=False, default=50, help="Max sequence length")

    beam_parameters_group.add_argument("--num_beams", type=int, required=False, default=4, help="Beam size")

    beam_parameters_group.add_argument(
        "--num_return_sequences",
        type=int,
        required=False,
        default=1,
        help="Number of return sequence <= num_beams",
    )

    beam_parameters_group.add_argument(
        "--length_penalty",
        type=float,
        required=False,
        default=1,
        help="Positive. >1 to penalize and <1 to encourage short sentence.",
    )

    beam_parameters_group.add_argument(
        "--repetition_penalty",
        type=float,
        required=False,
        default=1,
        help="Positive. >1 to penalize and <1 to encourage.",
    )

    beam_parameters_group.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="The value used to module the next token probabilities.",
    )

    beam_parameters_group.add_argument(
        "--top_p",
        type=float,
        required=False,
        default=1.0,
        help="Top P for sampling",
    )

    beam_parameters_group.add_argument(
        "--filter_value",
        type=float,
        required=False,
        default=-float("Inf"),
        help="Filter value for Top P sampling",
    )

    beam_parameters_group.add_argument(
        "--min_tokens_to_keep",
        type=int,
        required=False,
        default=1,
        help="Minimum number of tokens we keep per batch example in the output.",
    )

    beam_parameters_group.add_argument(
        "--presence_penalty",
        type=float,
        required=False,
        default=0.0,
        help="presence penalty for custom sampling.",
    )

    beam_parameters_group.add_argument(
        "--custom",
        type=int,
        required=False,
        default=0,
        help="If 1 customized top P logic is applied",
    )

    beam_parameters_group.add_argument(
        "--vocab_size",
        type=int,
        required=False,
        default=-1,
        help="Vocab_size of the underlying model used to decide the shape of vocab mask",
    )

    beam_parameters_group.add_argument(
        "--eos_token_id",
        type=int,
        required=False,
        default=-1,
        help="custom eos_token_id for generating model with existing onnx encoder/decoder",
    )

    beam_parameters_group.add_argument(
        "--pad_token_id",
        type=int,
        required=False,
        default=-1,
        help="custom pad_token_id for generating model with existing onnx encoder/decoder",
    )

    test_group = parser.add_argument_group("Other options for testing parity and performance")

    test_group.add_argument(
        "--use_sln_strict_mode",
        required=False,
        action="store_true",
        help="Enable strict mode for SLN in CUDA provider. This ensures a better accuracy but will be slower.",
    )
    test_group.set_defaults(use_sln_strict_mode=False)

    test_group.add_argument(
        "--use_gpu",
        required=False,
        action="store_true",
        help="use GPU for inference. Required for fp16.",
    )
    test_group.set_defaults(use_gpu=False)

    test_group.add_argument(
        "--disable_parity",
        required=False,
        action="store_true",
        help="do not run parity test",
    )
    test_group.set_defaults(disable_parity=False)

    test_group.add_argument(
        "--disable_perf_test",
        required=False,
        action="store_true",
        help="do not run perf test",
    )
    test_group.set_defaults(disable_perf_test=False)

    test_group.add_argument(
        "--torch_performance",
        required=False,
        action="store_true",
        help="test PyTorch performance",
    )
    test_group.set_defaults(torch_performance=False)

    test_group.add_argument(
        "--total_runs",
        required=False,
        type=int,
        default=1,
        help="Number of times of inference for latency measurement",
    )

    test_group.add_argument(
        "--save_test_data",
        required=False,
        action="store_true",
        help="save test data for onnxruntime_perf_test tool",
    )
    test_group.set_defaults(save_test_data=False)

    args = parser.parse_args(argv)

    return args


def gpt2_to_onnx(args: argparse.Namespace):
    """Convert GPT-2 model to onnx

    Args:
        args (argparse.Namespace): arguments parsed from command line
    """
    model_name = args.model_name_or_path

    arguments = [
        "--model_name_or_path",
        model_name,
        "--output",
        args.decoder_onnx,
        "--optimize_onnx",
        "--precision",
        args.precision,
        "--test_runs",
        "1",
        "--test_cases",
        "10",
        "--overwrite",  # Overwrite onnx file if existed
    ]
    if args.cache_dir:
        arguments.extend(["--cache_dir", args.cache_dir])
    if args.use_gpu:
        arguments.append("--use_gpu")
    if args.use_external_data_format:
        arguments.append("--use_external_data_format")

    if len(args.op_block_list):
        arguments.extend(["--op_block_list"])
        arguments.extend(args.op_block_list)

    if args.precision == Precision.FLOAT16.value:
        assert args.use_gpu, "fp16 or mixed precision model cannot run in CPU. Please add --use_gpu"
        # TODO(tianleiwu): Use auto mixed precision for fp16 conversion: arguments.append('--auto_mixed_precision')
        #       Need change cuda kernel to support a combination of fp32 logits and fp16 past state.
        #       Currently logits and past state shall be same data type.

    if args.verbose:
        logger.info(f"arguments for convert_to_onnx:{arguments}")

    convert_gpt2_to_onnx(argv=arguments)


def t5_to_onnx(args: argparse.Namespace):
    """Convert T5 model to onnx

    Args:
        args (argparse.Namespace): arguments parsed from command line
    """
    paths = export_t5_onnx_models(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        output_dir=Path(args.output).parent,
        use_gpu=args.use_gpu,
        use_external_data_format=args.use_external_data_format,
        optimize_onnx=(args.precision != Precision.FLOAT16.value),
        precision=args.precision,
        verbose=False,
        use_decoder_start_token=False,
        overwrite=True,
        disable_auto_mixed_precision=False,
        use_int32_inputs=True,
        model_type=args.model_type,
        encoder_decoder_init=args.encoder_decoder_init,
        force_fp16_io=(args.precision == Precision.FLOAT16.value),  # required by BeamSearch op implementation.
    )

    logger.debug(f"onnx model for encoder: {paths[0]}")
    logger.debug(f"onnx model for decoder: {paths[1]}")
    args.encoder_decoder_init_onnx = paths[0]
    args.decoder_onnx = paths[1]


def shape_inference(onnx_path: str, use_external_data_format: bool = True):
    """Shape inference on an onnx file, which will be overwritten.

    Args:
        onnx_path (str): Path of onnx model
        use_external_data_format(bool): output tensors to external data or not.
    """
    # Run symbolic shape inference to walk around ORT shape inference issue for subgraph.
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    model = onnx.load_model(onnx_path, load_external_data=True)
    out = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=False)
    if out:
        OnnxModel.save(out, onnx_path, save_as_external_data=use_external_data_format)
    else:
        logger.warning("Failed to run symbolic shape inference on the model.")


def pad_weights_of_logits_matmul(onnx_path: str, use_external_data_format: bool = True) -> bool:
    """Pad the logits MatMul weight in the provided decoder model, which will be overwritten.

    Args:
        onnx_path (str): Path of onnx model
        use_external_data_format(bool): output tensors to external data or not.
    """
    decoder_model_proto = onnx.load_model(onnx_path, load_external_data=True)

    logits_output_name = decoder_model_proto.graph.output[0].name

    decoder_model = OnnxModel(decoder_model_proto)

    output_name_to_node = decoder_model.output_name_to_node()
    assert logits_output_name in output_name_to_node

    matmul_node = output_name_to_node[logits_output_name]
    # Sanity check - the logits need to be produced by a MatMul node
    if matmul_node.op_type != "MatMul":
        return False

    # The logits MatMul weight MUST be an initializer (or)
    # it MUST be flowing through a Transpose whose input is
    # an initializer
    pad_along_axis_1 = True
    logits_weight = decoder_model.get_initializer(matmul_node.input[1])
    if logits_weight is None:
        transpose_before_matmul = decoder_model.match_parent(matmul_node, "Transpose", 1)

        if transpose_before_matmul is None:
            return False

        logits_weight = decoder_model.get_initializer(transpose_before_matmul.input[0])

        if logits_weight is None:
            return False

        pad_along_axis_1 = False

    # The logits MatMul weight MUST be fp16
    if logits_weight.data_type != TensorProto.DataType.FLOAT16:
        return False

    # The logits MatMul weight MUST be 2-dimensional
    if len(logits_weight.dims) != 2:
        return False

    # Pad and over-write the initializer (if needed)
    actual_vocab_size = logits_weight.dims[1]

    if (actual_vocab_size % 8) == 0:
        # Already "padded"
        return True

    padded_vocab_size = math.ceil(actual_vocab_size / 8) * 8
    padding = padded_vocab_size - actual_vocab_size

    # TODO(hasesh): Handle cases where the fp16 data is stored in the
    # non-raw data field
    if logits_weight.raw_data:
        if pad_along_axis_1:
            padding_data = np.zeros((logits_weight.dims[0], padding), dtype=np.float16)
            weight_with_padding = np.concatenate((NumpyHelper.to_array(logits_weight), padding_data), axis=1)
            logits_weight.dims[1] = padded_vocab_size
        else:
            padding_data = np.zeros((padding, logits_weight.dims[1]), dtype=np.float16)
            weight_with_padding = np.concatenate((NumpyHelper.to_array(logits_weight), padding_data), axis=0)
            logits_weight.dims[0] = padded_vocab_size

        logits_weight.raw_data = weight_with_padding.tobytes()
    else:
        return False

    # Save the model
    OnnxModel.save(decoder_model_proto, onnx_path, save_as_external_data=use_external_data_format)
    return True


def create_ort_session(model_path: str, use_gpu: bool, use_sln_strict_mode: bool) -> InferenceSession:
    """Create OnnxRuntime session.

    Args:
        model_path (str): onnx model path
        use_gpu (bool): use GPU or not
        use_sln_strict_mode (bool): use strict mode for skip layer normalization or not

    Raises:
        RuntimeError: CUDAExecutionProvider is not available when --use_gpu is specified.

    Returns:
        onnxruntime.InferenceSession: The created session.
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    if use_gpu:
        if "CUDAExecutionProvider" not in get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is not available for --use_gpu!")
        else:
            logger.info("use CUDAExecutionProvider")
        if use_sln_strict_mode:
            cuda_provider_options = {"enable_skip_layer_norm_strict_mode": True}
            provider_options = {"CUDAExecutionProvider": cuda_provider_options}
            execution_providers = [
                (name, provider_options[name]) if name in provider_options else name for name in execution_providers
            ]

    ort_session = InferenceSession(model_path, sess_options, providers=execution_providers)
    return ort_session


def verify_gpt2_subgraph(graph: onnx.GraphProto, precision: Precision):
    """Verify GPT-2 subgraph

    Args:
        graph (onnx.GraphProto): onnx graph of GPT-2
        precision (Precision): Precision (FLOAT16 or FLOAT32) of the model.

    Raises:
        ValueError: Number of inputs not expected.
        ValueError: Input name is not expected.
        ValueError: Input data type is not expected.
        ValueError: Number of outputs not expected.
        ValueError: Output name is not expected.
        ValueError: Output data type is not expected.
    """
    is_float16 = precision == Precision.FLOAT16.value

    input_count = len(graph.input)
    layer_count = input_count - 3
    assert layer_count >= 1

    expected_inputs = ["input_ids", "position_ids", "attention_mask"] + [f"past_{i}" for i in range(layer_count)]
    if len(graph.input) != len(expected_inputs):
        raise ValueError(f"Number of inputs expected to be {len(expected_inputs)}. Got {len(graph.input)}")

    for i, expected_input in enumerate(expected_inputs):
        if graph.input[i].name != expected_input:
            raise ValueError(f"Input {i} is expected to be {expected_input}. Got {graph.input[i].name}")

        expected_type = TensorProto.INT32
        if i >= 3:
            expected_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT

        input_type = graph.input[i].type.tensor_type.elem_type
        if input_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {input_type}")
    logger.info("Verifying GPT-2 graph inputs: name and data type are good.")

    expected_outputs = ["logits"] + [f"present_{i}" for i in range(layer_count)]
    if len(graph.output) != len(expected_outputs):
        raise ValueError(f"Number of outputs expected to be {len(expected_outputs)}. Got {len(graph.output)}")

    for i, expected_output in enumerate(expected_outputs):
        if graph.output[i].name != expected_output:
            raise ValueError(f"Output {i} is expected to be {expected_output}. Got {graph.output[i].name}")

        expected_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT
        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {output_type}")
    logger.info("Verifying GPT-2 graph outputs: name and data type are good.")

    # TODO(tianleiwu): verify shapes of inputs and outputs.
    return


def verify_t5_decoder_subgraph(graph: onnx.GraphProto, precision: Precision):
    """Verify T5 decoder subgraph

    Args:
        graph (onnx.GraphProto): onnx graph of T5 decoder
        precision (Precision): Precision (FLOAT16 or FLOAT32) of the model.

    Raises:
        ValueError: Number of inputs not expected.
        ValueError: Input name is not expected.
        ValueError: Input data type is not expected.
        ValueError: Number of outputs not expected.
        ValueError: Output name is not expected.
        ValueError: Output data type is not expected.
    """
    is_float16 = precision == Precision.FLOAT16.value
    float_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT

    input_count = len(graph.input)
    layer_count = (input_count - 2) // 4
    assert layer_count >= 1

    # Expect inputs:
    #   input_ids: int32 (B, 1)
    #   encoder_attention_mask: int32 (B, encode_sequence_length)

    #   past_key_self_0: (B, num_heads, past_decode_sequence_length, head_size)
    #   past_value_self_0: (B, num_heads, past_decode_sequence_length, head_size)
    #   ... (for each self attention layer)

    #   past_key_cross_0: (B, num_heads, encode_sequence_length, head_size)
    #   past_value_cross_0: (B, num_heads, encode_sequence_length, head_size)
    #   ... (for each cross attention layer)

    # TODO: encoder_hidden_states is optional
    expected_inputs = ["input_ids", "encoder_attention_mask"]
    for i in range(layer_count):
        expected_inputs.append(f"past_key_self_{i}")
        expected_inputs.append(f"past_value_self_{i}")
    for i in range(layer_count):
        expected_inputs.append(f"past_key_cross_{i}")
        expected_inputs.append(f"past_value_cross_{i}")

    if len(graph.input) != len(expected_inputs):
        raise ValueError(f"Number of inputs expected to be {len(expected_inputs)}. Got {len(graph.input)}")

    for i, expected_input in enumerate(expected_inputs):
        if graph.input[i].name != expected_input:
            raise ValueError(f"Input {i} is expected to be {expected_input}. Got {graph.input[i].name}")

        expected_type = TensorProto.INT32 if i < 2 else float_type
        input_type = graph.input[i].type.tensor_type.elem_type
        if input_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {input_type}")

    # Expect outputs:
    #   logits:               (B, 1, vocab_size)
    #   present_key_self_0:   (B, num_heads, past_decode_sequence_length + 1, head_size)
    #   present_value_self_0: (B, num_heads, past_decode_sequence_length + 1, head_size)
    #                     ... (for each self attention layer)
    expected_outputs = ["logits"]
    for i in range(layer_count):
        expected_outputs.append(f"present_key_self_{i}")
        expected_outputs.append(f"present_value_self_{i}")

    if len(graph.output) != len(expected_outputs):
        raise ValueError(f"Number of outputs expected to be {len(expected_outputs)}. Got {len(graph.output)}")

    for i, expected_output in enumerate(expected_outputs):
        if graph.output[i].name != expected_output:
            raise ValueError(f"Output {i} is expected to be {expected_output}. Got {graph.output[i].name}")
        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != float_type:
            raise ValueError(f"Output {i} is expected to have onnx data type {float_type}. Got {output_type}")


def verify_t5_encoder_decoder_init_subgraph(graph: onnx.GraphProto, precision: Precision):
    """Verify T5 decoder subgraph

    Args:
        graph (onnx.GraphProto): onnx graph of T5 decoder
        precision (Precision): Precision (FLOAT16 or FLOAT32) of the model.

    Raises:
        ValueError: Number of inputs not expected.
        ValueError: Input name is not expected.
        ValueError: Input data type is not expected.
        ValueError: Number of outputs not expected.
        ValueError: Output name is not expected.
        ValueError: Output data type is not expected.
    """
    is_float16 = precision == Precision.FLOAT16.value
    new_format = "cross" in graph.output[0].name

    # Expect 3 inputs:
    #   encoder_input_ids:      int32 (B, encode_sequence_length)
    #   encoder_attention_mask: int32 (B, encode_sequence_length)
    #   decoder_input_ids:      int32 (B, 1)
    expected_inputs = [
        "encoder_input_ids",
        "encoder_attention_mask",
        "decoder_input_ids",
    ]
    if new_format:
        expected_inputs = expected_inputs[:2]
    if len(graph.input) != len(expected_inputs):
        raise ValueError(f"Number of inputs expected to be {len(expected_inputs)}. Got {len(graph.input)}")

    for i, expected_input in enumerate(expected_inputs):
        if graph.input[i].name != expected_input:
            raise ValueError(f"Input {i} is expected to be {expected_input}. Got {graph.input[i].name}")

        expected_type = TensorProto.INT32
        input_type = graph.input[i].type.tensor_type.elem_type
        if input_type != expected_type:
            raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {input_type}")

    if new_format:
        assert len(graph.output) % 2 == 0
        layer_count = len(graph.output) // 2
        assert layer_count >= 1

        # Expected outputs:
        #   present_key_cross_0:   (B, num_heads, encode_sequence_length, head_size)
        #   present_value_cross_0: (B, num_heads, encode_sequence_length, head_size)
        #                      ... (for each cross attention layer)
        expected_outputs = []
        for i in range(layer_count):
            expected_outputs.append(f"present_key_cross_{i}")
            expected_outputs.append(f"present_value_cross_{i}")
    else:
        logger.warning("This format is deprecated. Please export T5 encoder in new format with only cross outputs.")
        assert (len(graph.output) - 2) % 4 == 0
        layer_count = (len(graph.output) - 2) // 4
        assert layer_count >= 1

        # Expected outputs:
        #   logits:                (B, 1, vocab_size)
        #   encoder_hidden_states: (B, encode_sequence_length, encoder_hidden_size)
        #   present_key_self_0:    (B, num_heads, 1, head_size)
        #   present_value_self_0:  (B, num_heads, 1, head_size)
        #                      ... (for each self attention layer)
        #   present_key_cross_0:   (B, num_heads, encode_sequence_length, head_size)
        #   present_value_cross_0: (B, num_heads, encode_sequence_length, head_size)
        #                      ... (for each cross attention layer)
        expected_outputs = ["logits", "encoder_hidden_states"]
        for i in range(layer_count):
            expected_outputs.append(f"present_key_self_{i}")
            expected_outputs.append(f"present_value_self_{i}")
        for i in range(layer_count):
            expected_outputs.append(f"present_key_cross_{i}")
            expected_outputs.append(f"present_value_cross_{i}")

    if len(graph.output) != len(expected_outputs):
        raise ValueError(f"Number of outputs expected to be {len(expected_outputs)}. Got {len(graph.output)}")

    for i, expected_output in enumerate(expected_outputs):
        if graph.output[i].name != expected_output:
            raise ValueError(f"Output {i} is expected to be {expected_output}. Got {graph.output[i].name}")

        expected_type = TensorProto.FLOAT16 if is_float16 else TensorProto.FLOAT
        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != expected_type:
            raise ValueError(f"Output {i} is expected to have onnx data type {expected_type}. Got {output_type}")

    logger.info("T5 encoder graph verified: name and data type of inputs and outputs are good.")


def remove_shared_initializers(
    graph1: GraphProto,
    graph2: GraphProto,
    shared_prefix: str = "shared_",
    min_elements: int = 1024,
    signature_cache1: dict | None = None,
    signature_cache2: dict | None = None,
):
    """Remove initializers with same value from two graphs.

    Args:
        graph1 (GraphProto): the first graph to process
        graph2 (GraphProto): the second graph to process
        shared_prefix (str): add prefix to the shared initializers among two graphs
        min_elements (int, optional): minimal number of elements for initializers to be considered. Defaults to 1024.
        signature_cache1 (dict): Optional dictionary to store data signatures of tensors in graph1 in order to speed up comparison
        signature_cache2 (dict): Optional dictionary to store data signatures of tensors in graph2 in order to speed up comparison
    """

    mapping_initializers_1 = {}
    mapping_initializers_2 = {}
    shared_initializers_1 = []
    shared_initializers_2 = []
    shared_initializers_names = []

    for initializer1 in graph1.initializer:
        if not (initializer1.dims and sum(initializer1.dims) >= min_elements):
            continue

        for initializer2 in graph2.initializer:
            if not (initializer2.dims and sum(initializer2.dims) >= min_elements):
                continue

            if OnnxModel.has_same_value(initializer1, initializer2, signature_cache1, signature_cache2):
                mapping_initializers_1[initializer1.name] = shared_prefix + initializer2.name
                shared_initializers_1.append(initializer1)

                if initializer2.name not in mapping_initializers_2:
                    shared_name = shared_prefix + initializer2.name
                    mapping_initializers_2[initializer2.name] = shared_name
                    shared_initializers_2.append(initializer2)
                    shared_initializers_names.append(shared_name)
                break

    logger.debug(f"shared initializers:{shared_initializers_names}")

    # Make sure new name does not exist in graph 1
    for node in graph1.node:
        for j in range(len(node.input)):
            if node.input[j] in shared_initializers_names:
                raise RuntimeError(f"name is found in graph 1: {node.input[j]}")

    # Make sure new name does not exist in graph 2
    for node in graph2.node:
        for j in range(len(node.input)):
            if node.input[j] in shared_initializers_names:
                raise RuntimeError(f"name is found in graph 2: {node.input[j]}")

    # Remove shared initializers from graph 2
    for initializer in shared_initializers_2:
        graph2.initializer.remove(initializer)

    # Rename value info for old names in graph 2
    for value_info in graph2.value_info:
        if value_info.name in mapping_initializers_2:
            value_info.name = mapping_initializers_2[value_info.name]

    # Rename nodes inputs in graph 2:
    for node in graph2.node:
        for j in range(len(node.input)):
            if node.input[j] in mapping_initializers_2:
                new_name = mapping_initializers_2[node.input[j]]
                logger.debug(f"graph 2 rename node {node.name} input {j} from {node.input[j]} to {new_name}")
                node.input[j] = new_name

    #  Remove shared initializers from graph 1
    for initializer in shared_initializers_1:
        graph1.initializer.remove(initializer)

    # Rename value info for old names in graph 1
    for value_info in graph1.value_info:
        if value_info.name in mapping_initializers_1:
            value_info.name = mapping_initializers_1[value_info.name]

    # Rename nodes inputs in graph 1:
    for node in graph1.node:
        for j in range(len(node.input)):
            if node.input[j] in mapping_initializers_1:
                new_name = mapping_initializers_1[node.input[j]]
                logger.debug(f"graph 1 rename node {node.name} input {j} from {node.input[j]} to {new_name}")
                node.input[j] = new_name

    # Rename shared initializers in graph 2
    for initializer in shared_initializers_2:
        initializer.name = mapping_initializers_2[initializer.name]

    for initializer in shared_initializers_2:
        shape = onnx.numpy_helper.to_array(initializer).shape
        value_info = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, shape)
        # Need add value_info for initializers moved to parent graph. Otherwise, ORT will fail.
        graph1.value_info.append(value_info)
        graph2.value_info.append(value_info)

    return shared_initializers_2


def get_shared_initializers(encoder_model: ModelProto, decoder_model: ModelProto):
    encoder = OnnxModel(encoder_model)
    decoder = OnnxModel(decoder_model)
    encoder.add_prefix_to_names("e_")
    decoder.add_prefix_to_names("d_")
    signature_cache1, signature_cache2 = {}, {}
    encoder.remove_duplicated_initializer(signature_cache1)
    decoder.remove_duplicated_initializer(signature_cache2)
    initializers = remove_shared_initializers(
        decoder.model.graph,
        encoder.model.graph,
        shared_prefix="s_",
        signature_cache1=signature_cache1,
        signature_cache2=signature_cache2,
    )
    return initializers


def move_initializers(
    graph: GraphProto,
    min_elements: int = 1024,
) -> list[TensorProto]:
    """Remove initializers of a graph, when they have number of elements larger than a threshold.

    Args:
        graph (GraphProto): the graph.
        min_elements (int, optional): minimal number of elements for initializers to be considered. Defaults to 1024.

    Returns:
        List[TensorProto]: initializers that are removed from the graph.
    """
    moved_initializers = []
    for tensor in graph.initializer:
        if not (tensor.dims and sum(tensor.dims) >= min_elements):
            continue
        moved_initializers.append(tensor)

    for initializer in moved_initializers:
        graph.initializer.remove(initializer)

    # Add type info, otherwise ORT will raise error: "input arg (*) does not have type information set by parent node."
    for initializer in moved_initializers:
        shape = onnx.numpy_helper.to_array(initializer).shape
        value_info = onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, shape)
        graph.value_info.append(value_info)

    return moved_initializers


def _attribute_to_pair(attribute):
    """
    Convert attribute to kwarg format for use with onnx.helper.make_node.
        :parameter attribute: attribute in AttributeProto format.
        :return: attribute in {key: value} format.
    """
    if attribute.type == 0:
        raise ValueError(f"attribute {attribute.name} does not have type specified.")

    # Based on attribute type definitions from AttributeProto
    # definition in https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    if attribute.type == 1:
        value = attribute.f
    elif attribute.type == 2:
        value = attribute.i
    elif attribute.type == 3:
        value = attribute.s
    elif attribute.type == 4:
        value = attribute.t
    elif attribute.type == 5:
        value = attribute.g
    elif attribute.type == 6:
        value = attribute.floats
    elif attribute.type == 7:
        value = attribute.ints
    elif attribute.type == 8:
        value = attribute.strings
    elif attribute.type == 9:
        value = attribute.tensors
    elif attribute.type == 10:
        value = attribute.graphs
    else:
        raise ValueError(f"attribute {attribute.name} has unsupported type {attribute.type}.")

    return (attribute.name, value)


def kwargs_of(node):
    kwargs = {}
    for attr in node.attribute:
        (key, value) = _attribute_to_pair(attr)
        kwargs.update({key: value})
    if node.domain:
        kwargs.update({"domain": node.domain})
    return kwargs


def shape_of(vi):
    return tuple([d.dim_param if (d.dim_param) else d.dim_value for d in vi.type.tensor_type.shape.dim])


def update_decoder_subgraph_past_present_share_buffer(subg: GraphProto):
    input_past_0 = 3
    output_past_0 = 1
    new_inputs = []
    for i, vi in enumerate(subg.input):
        if i >= input_past_0:
            shape = shape_of(vi)
            vi = onnx.helper.make_tensor_value_info(  # noqa: PLW2901
                vi.name,
                elem_type=vi.type.tensor_type.elem_type,
                shape=[shape[0], shape[1], shape[2], "max_seq_len", shape[4]],
            )
        new_inputs.extend([vi])
    new_inputs.extend([onnx.helper.make_tensor_value_info("past_sequence_length", onnx.TensorProto.INT32, shape=[1])])
    subg.ClearField("input")
    subg.input.extend(new_inputs)

    new_outputs = []
    for i, vi in enumerate(subg.output):
        if i >= output_past_0:
            shape = shape_of(vi)
            vi = onnx.helper.make_tensor_value_info(  # noqa: PLW2901
                vi.name,
                elem_type=vi.type.tensor_type.elem_type,
                shape=[shape[0], shape[1], shape[2], "max_seq_len", shape[4]],
            )
        new_outputs.extend([vi])
    subg.ClearField("output")
    subg.output.extend(new_outputs)

    new_nodes = []
    for node in subg.node:
        new_node = node
        if node.op_type == "Attention":
            kwargs = kwargs_of(node)
            kwargs.update({"past_present_share_buffer": 1})
            nis = []
            nis.extend(node.input)
            while len(nis) < 6:
                nis.extend([""])
            if len(nis) < 7:
                nis.extend(["past_sequence_length"])
            new_node = onnx.helper.make_node("Attention", nis, node.output, name=node.name, **kwargs)
        new_nodes.extend([new_node])
    subg.ClearField("node")
    subg.node.extend(new_nodes)
    return subg


def update_decoder_subgraph_use_decoder_masked_attention(
    subg: GraphProto, is_beam_search: bool, switch_attention: bool
) -> bool:
    """Update the Attention nodes to DecoderMaskedSelfAttention.

    Args:
        subg (GraphProto): GraphProto of the decoder subgraph
        is_beam_search (bool): Boolean specifying if the sampling algo is BeamSearch
        switch_attention (bool): Boolean specifying if `Attention` is to be switched with `DecoderMaskedSelfAttention`
    """
    if is_beam_search:
        new_inputs = []
        for _i, vi in enumerate(subg.input):
            new_inputs.extend([vi])

        # Add 2 BeamSearch specific inputs
        new_inputs.extend([onnx.helper.make_tensor_value_info("beam_width", onnx.TensorProto.INT32, shape=[1])])
        new_inputs.extend(
            [
                onnx.helper.make_tensor_value_info(
                    "cache_indirection",
                    onnx.TensorProto.INT32,
                    shape=["batch_size", "beam_width", "max_seq_len"],
                )
            ]
        )
        subg.ClearField("input")
        subg.input.extend(new_inputs)

    if switch_attention:
        decoder_masked_attention_supported_attr = [
            "past_present_share_buffer",
            "num_heads",
            "scale",
            "mask_filter_value",
            "domain",
        ]

        new_nodes = []
        for node in subg.node:
            if node.op_type == "Attention":
                kwargs = kwargs_of(node)
                for k in kwargs.copy():
                    # The Attention operator does not support different qkv hidden sizes when past/present
                    # input/output exists (GPT2 model). Hence, we should never run into this.
                    # But, if we do, do not go ahead with the optimization.
                    if k == "qkv_hidden_sizes":
                        return False

                    if k not in decoder_masked_attention_supported_attr:
                        # Log the fact that we are removing certain attributes from the node
                        # We don't need to log it for "unidirectional" as we are aware that
                        # decoding attention kernels are unidirectional by definition.
                        if k != "unidirectional":
                            logger.warning(
                                f"Removing attribute: {k} from Attention node while switching to DecoderMaskedSelfAttention"
                            )

                        del kwargs[k]

                nis = []
                nis.extend(node.input)

                # Add 2 BeamSearch specific inputs
                if is_beam_search:
                    while len(nis) < 7:
                        nis.extend([""])
                    if len(nis) < 8:
                        nis.extend(["beam_width"])
                    if len(nis) < 9:
                        nis.extend(["cache_indirection"])

                node = onnx.helper.make_node(  # noqa: PLW2901
                    "DecoderMaskedSelfAttention",
                    nis,
                    node.output,
                    name=node.name,
                    **kwargs,
                )
            new_nodes.extend([node])
        subg.ClearField("node")
        subg.node.extend(new_nodes)

    return True


def find_past_seq_len_usage(subg: GraphProto):
    """Correct graph which originally use dim of past_seq_len from input_ids's shape which is fixed to max_seq_len after
       shared past/present buffer

    Args:
        subg (GraphProto): GraphProto of the decoder subgraph
    return:
        tensor_names_to_rename : set of tensor names which is equal to past_sequence_length
        nodes_to_remove : list of node to remove
    """
    tensor_names_to_rename = set()
    nodes_to_remove = []

    graph_input_names = {inp.name: index for index, inp in enumerate(subg.input)}

    input_name_to_nodes = {}
    output_name_to_node = {}
    for node in subg.node:
        for input_name in node.input:
            if input_name:
                if input_name not in input_name_to_nodes:
                    input_name_to_nodes[input_name] = [node]
                else:
                    input_name_to_nodes[input_name].append(node)
        for output_name in node.output:
            if output_name:
                output_name_to_node[output_name] = node

    for node in subg.node:
        # find "past_key_self_0 --> [Transpose(past_key_self_0) --> Reshape(past_key_self_0)] --> Shape(past_key_self_0) --> Gather(*, 2)"
        # where [Transpose(past_key_self_0) --> Reshape(past_key_self_0)] may or may not exist
        if node.op_type == "Gather":
            if not node.input[1] or not node.input[0]:
                continue

            # Find Gather node's index value
            shape_tensor_name, shape_index_name = (node.input[0], node.input[1])
            ini_gather_indices = None
            if "Constant_" in shape_index_name:
                # If shape_index_name refers to a Constant node
                for const_node in subg.node:
                    if const_node.op_type == "Constant" and const_node.output[0] == shape_index_name:
                        ini_gather_indices = const_node.attribute[0].t
                        break
            else:
                # If shape_index_name refers to an initializer
                for tensor in subg.initializer:
                    if tensor.name == shape_index_name:
                        ini_gather_indices = tensor
                        break
            if ini_gather_indices is None:
                continue
            gather_indices_arr = onnx.numpy_helper.to_array(ini_gather_indices)

            if (
                gather_indices_arr.size == 1
                and gather_indices_arr.item() in {1, 2}
                and node.input[0] in output_name_to_node
            ):
                shape_node = output_name_to_node[shape_tensor_name]
                if not (shape_node.op_type == "Shape" and shape_node.input[0]):
                    continue

                if (
                    shape_node.input[0] in graph_input_names
                    and (
                        shape_node.input[0].startswith("past_key_self_")
                        or shape_node.input[0].startswith("past_value_self_")
                    )
                    and gather_indices_arr.item() == 2
                ):
                    # "past_key_self_0 --> Shape(past_key_self_0) --> Gather(*, 2)"
                    tensor_names_to_rename.add(node.output[0])
                    nodes_to_remove.append(node)
                    if len(input_name_to_nodes[shape_node.output[0]]) == 1:
                        nodes_to_remove.append(shape_node)
                        continue

                if shape_node.input[0] not in output_name_to_node:
                    continue
                reshape_node = output_name_to_node[shape_node.input[0]]
                if not (reshape_node.op_type == "Reshape" and reshape_node.input[0]):
                    continue
                transpose_node = output_name_to_node[reshape_node.input[0]]
                if not (transpose_node.op_type == "Transpose" and transpose_node.input[0]):
                    continue

                if (
                    transpose_node.input[0] in graph_input_names
                    and (
                        transpose_node.input[0].startswith("past_key_self_")
                        or transpose_node.input[0].startswith("past_value_self_")
                    )
                    and gather_indices_arr.item() == 1
                ):
                    # "past_key_self_0 --> Transpose(past_key_self_0) --> Reshape(past_key_self_0) --> Shape(past_key_self_0) --> Gather(*, 2)"
                    tensor_names_to_rename.add(node.output[0])
                    nodes_to_remove.extend([node, shape_node, reshape_node])
                    if len(input_name_to_nodes[transpose_node.output[0]]) == 1:
                        nodes_to_remove.append(transpose_node)
                        continue

    return tensor_names_to_rename, nodes_to_remove


def add_cache_indirection_to_mha(model: OnnxModel, past_seq_len_name: str):
    # Add past_sequence_length and cache_indirection as inputs to all MultiHeadAttention ops and as inputs to model
    cache_indirection_name = "cache_indirection"
    mha_nodes = list(filter(lambda node: node.op_type == "MultiHeadAttention", model.model.graph.node))
    for node in mha_nodes:
        # MHA op takes the following potential inputs:
        # query, key, value, bias, key_padding_mask, add_qk, past_key, past_value
        while len(node.input) < 8:
            node.input.append("")
        node.input.append(past_seq_len_name)
        node.input.append(cache_indirection_name)

    model.model.graph.input.append(
        onnx.helper.make_tensor_value_info(
            cache_indirection_name, TensorProto.INT32, shape=["batch_size", "beam_width", "max_sequence_length"]
        ),
    )
    model.topological_sort()
    return model


def add_output_qk_to_mha(model: OnnxModel, dtype: int = 0, skip_node_idxs: list[int] = []):  # noqa: B006
    # Add output_qk as output to MultiHeadAttention ops and as outputs to model
    output_qk_basename = "output_cross_qk"
    output_qks = []
    mha_nodes = list(filter(lambda node: node.op_type == "MultiHeadAttention", model.model.graph.node))
    for idx, node in enumerate(mha_nodes):
        # Skip MHA nodes where output_qk does not need to be added
        if idx in skip_node_idxs:
            continue

        # Get `num_heads` attribute from MHA
        num_heads = 0
        for att in node.attribute:
            if att.name == "num_heads":
                num_heads = att.i
                break

        # Get dtype for `output_qk` based on MHA bias if not provided
        output_qk_dtype = dtype
        if output_qk_dtype == 0:
            for i in model.model.graph.initializer:
                if i.name == node.input[3]:
                    output_qk_dtype = i.data_type
                    break

        # Get `target_sequence_length` attribute from 4D input for key if it's a constant
        target_sequence_length = "target_sequence_length"
        for i in model.model.graph.input:
            if i.name == node.input[1]:
                target_sequence_length = i.type.tensor_type.shape.dim[2].dim_value
                break

        # MHA op takes the following potential outputs:
        # output, present_key, present_value
        while len(node.output) < 3:
            node.output.append("")

        output_qk_name = f"{output_qk_basename}_{idx // 2}"
        node.output.append(output_qk_name)
        output_qks.append(
            onnx.helper.make_tensor_value_info(
                output_qk_name,
                output_qk_dtype,
                shape=["batch_size", num_heads, "sequence_length", target_sequence_length],
            ),
        )

    model.model.graph.output.extend(output_qks)
    model.topological_sort()
    return model


def fix_past_sequence_length(model: ModelProto):
    # Modify total_sequence_length = past_sequence_length + curr_sequence_length subgraph to calculate
    # past_sequence_length from the new `past_sequence_length` input of size 1D and type int32 instead of
    # from `past_key_self_0` since DecoderMaskedMultiHeadAttention (DMMHA) uses buffer sharing and
    # `past_key_self_0.shape[2] = max_sequence_length` instead of `past_key_self_0.shape[2] = past_sequence_length`
    # when buffer sharing is enabled
    #
    # Before:
    #
    #   input_ids      past_key_self_0
    #       |                 |
    #     Shape             Shape
    #       |                 |
    #     Gather            Gather
    #     (idx=1)           (idx=2)
    #       |                 |    \
    #       +--------+--------+    Unsqueeze
    #                |
    #               Add
    #
    # After:
    #
    #   input_ids    past_sequence_length (1D)
    #       |                 |
    #     Shape            Squeeze
    #       |                 |
    #     Gather             Cast
    #     (idx=1)           (int64)
    #       |                 |    \
    #       +--------+--------+    Unsqueeze
    #                |
    #               Add

    node = list(filter(lambda n: n.op_type == "LayerNormalization", model.model.graph.node))[0]  # noqa: RUF015

    base_path = model.match_parent_path(
        node,
        ["Add", "Slice"],
        [0, 1],
    )
    if base_path is None:
        return

    left_path = model.match_parent_path(
        base_path[-1],
        ["Unsqueeze", "Add", "Gather", "Shape"],
        [2, 0, 0, 0],
    )
    right_path = model.match_parent_path(
        base_path[-1],
        ["Unsqueeze", "Gather", "Shape"],
        [1, 0, 0],
    )
    long_right_path = model.match_parent_path(
        base_path[-1],
        ["Unsqueeze", "Gather", "Shape", "Reshape", "Transpose"],
        [1, 0, 0, 0, 0],
    )
    if left_path is None or right_path is None or left_path[-2:] != right_path[-2:]:
        return

    # Remove `past_key_self_0 --> [Transpose --> Reshape] --> Shape --> Gather` connection
    # where `Transpose --> Reshape` part may or may not exist. The OpenAI implementation of
    # Whisper has an extra `Transpose --> Reshape` connection to remove.
    constant_node = list(filter(lambda n: n.output[0] == left_path[-2].input[1], model.model.graph.node))[0]  # noqa: RUF015
    model.model.graph.node.remove(left_path[-2])
    model.model.graph.node.remove(left_path[-1])
    model.model.graph.node.remove(constant_node)
    if long_right_path is not None:
        # Remove `Transpose --> Reshape` part
        model.model.graph.node.remove(long_right_path[-2])
        model.model.graph.node.remove(long_right_path[-1])

    # Add `past_sequence_length` as model input
    past_seq_len_name = "past_sequence_length"
    model.model.graph.input.append(
        onnx.helper.make_tensor_value_info(past_seq_len_name, TensorProto.INT32, shape=[1]),
    )

    # Add `past_sequence_length --> Squeeze --> Cast` connection
    past_seq_len_int32 = "past_seq_len_int32"
    past_seq_len_int64 = "past_seq_len_int64"

    squeeze_node = onnx.helper.make_node(
        "Squeeze",
        inputs=[past_seq_len_name],
        outputs=[past_seq_len_int32],
        name=model.create_node_name("Squeeze"),
    )
    squeeze_output = onnx.helper.make_tensor_value_info(past_seq_len_int32, TensorProto.INT32, shape=[])
    cast_node = onnx.helper.make_node(
        "Cast",
        inputs=[past_seq_len_int32],
        outputs=[past_seq_len_int64],
        name=model.create_node_name("Cast"),
        to=TensorProto.INT64,
    )
    cast_output = onnx.helper.make_tensor_value_info(past_seq_len_int64, TensorProto.INT64, shape=[])

    model.model.graph.value_info.extend([squeeze_output, cast_output])

    # Add `past_seq_len_int64` as an input name to existing nodes
    left_path[1].input[0] = past_seq_len_int64
    right_path[0].input[0] = past_seq_len_int64

    # Add new nodes to graph
    model.model.graph.node.extend([squeeze_node, cast_node])
    model.topological_sort()
    return model, past_seq_len_name


def replace_mha_with_dmmha(model: OnnxModel, past_seq_len_name: str):
    # Add `beam_width` and `cache_indirection` as model inputs
    beam_width = "beam_width"
    cache_indirection = "cache_indirection"

    model.model.graph.input.extend(
        [
            onnx.helper.make_tensor_value_info(beam_width, TensorProto.INT32, shape=[1]),
            onnx.helper.make_tensor_value_info(
                cache_indirection, TensorProto.INT32, shape=["batch_size", "beam_width", "max_sequence_length"]
            ),
        ]
    )

    # Replace all `MultiHeadAttention` nodes with `DecoderMaskedMultiHeadAttention` nodes
    mha_nodes = list(filter(lambda node: node.op_type == "MultiHeadAttention", model.model.graph.node))
    for idx, node in enumerate(mha_nodes):
        # Get `num_heads` attribute from MHA
        num_heads = 0
        for att in node.attribute:
            if att.name == "num_heads":
                num_heads = att.i
                break

        # Make Q*K outputs for cross-attention layers, which happen every alternative layer
        qk_output_name = f"output_cross_qk_{idx // 2}"
        qk_output = onnx.helper.make_tensor_value_info(
            qk_output_name, TensorProto.FLOAT, shape=["batch_size", num_heads, 1, "encode_sequence_length / 2"]
        )
        if idx % 2 == 1:
            model.model.graph.output.append(qk_output)

        # Make DMMHA node
        dmmha_node = onnx.helper.make_node(
            "DecoderMaskedMultiHeadAttention",
            inputs=[
                node.input[0],  # query
                node.input[1],  # key
                node.input[2],  # value
                "",  # mask_index
                "",  # relative_position_bias
                node.input[6] if len(node.input) > 4 else "",  # past_key
                node.input[7] if len(node.input) > 4 else "",  # past_value
                past_seq_len_name,  # past_sequence_length
                beam_width,  # beam_width
                cache_indirection,  # cache_indirection
                node.input[3],  # bias
            ],
            outputs=[
                node.output[0],  # output
                node.output[1] if len(node.input) > 4 else "",  # present_key
                node.output[2] if len(node.input) > 4 else "",  # present_value
                qk_output_name if idx % 2 == 1 else "",  # output_cross_qk
            ],
            name=node.name.replace("MultiHeadAttention", "DecoderMaskedMultiHeadAttention"),
            domain="com.microsoft",
            num_heads=num_heads,
            output_qk=(idx % 2),
            past_present_share_buffer=1,
        )
        if idx % 2 == 0:
            # Remove empty string for output_cross_qk, which happens every alternative layer
            dmmha_node.output.remove("")

        model.model.graph.node.remove(node)
        model.model.graph.node.extend([dmmha_node])

    model.topological_sort()
    return model


def replace_mha_with_gqa(
    model: OnnxModel,
    attn_mask: str,
    kv_num_heads: int = 0,
    world_size: int = 1,
    window_size: int = -1,
):
    # Insert attention_mask subgraph to calculate shared inputs for all GroupQueryAttention nodes
    #
    #                attention_mask
    #               /              \
    #          ReduceSum          Shape
    #              |                |
    #             Sub             Gather
    #              |                |
    #          seqlens_k   total_sequence_length
    #              |                |
    #        Cast to int32    Cast to int32

    model.add_initializer(
        onnx.helper.make_tensor(
            name="one",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[1],
        )
    )
    reduce_sum_node = onnx.helper.make_node(
        "ReduceSum",
        inputs=[attn_mask, "one"],
        outputs=[attn_mask + "_row_sums"],
        name=model.create_node_name("ReduceSum"),
    )
    sub_node = onnx.helper.make_node(
        "Sub",
        inputs=[attn_mask + "_row_sums", "one"],
        outputs=["seqlens_k_int64"],
        name=model.create_node_name("Sub"),
    )
    seqlen_k_cast_node = onnx.helper.make_node(
        "Cast",
        inputs=["seqlens_k_int64"],
        outputs=["seqlens_k"],
        name=model.create_node_name("Cast"),
        to=TensorProto.INT32,
    )
    shape_node = onnx.helper.make_node(
        "Shape",
        inputs=[attn_mask],
        outputs=[attn_mask + "_shape"],
        name=model.create_node_name("Shape"),
    )
    gather_node = onnx.helper.make_node(
        "Gather",
        inputs=[attn_mask + "_shape", "one"],
        outputs=["total_seq_len_int64"],
        name=model.create_node_name("Gather"),
        axis=0,
    )
    total_seqlen_cast_node = onnx.helper.make_node(
        "Cast",
        inputs=["total_seq_len_int64"],
        outputs=["total_seq_len"],
        name=model.create_node_name("Cast"),
        to=TensorProto.INT32,
    )
    model.model.graph.node.extend(
        [
            reduce_sum_node,
            sub_node,
            seqlen_k_cast_node,
            shape_node,
            gather_node,
            total_seqlen_cast_node,
        ]
    )

    # Replace MultiHeadAttention with GroupQueryAttention
    #
    # When replacing, fuse the following subgraph:
    #
    #                 root_input
    #               /     |      \
    #         MatMul    MatMul    MatMul
    #           |         |         |
    #          Add       Add       Add      (optional Adds)
    #           |         |         |
    #         RotEmb    RotEmb      |
    #            \        |        /
    #             MultiHeadAttention
    #
    # to this new subgraph:
    #
    #                 root_input
    #                     |
    #                PackedMatMul           (if possible)
    #                     |
    #                 PackedAdd             (if possible)
    #                     |
    #             GroupQueryAttention
    #

    mha_nodes = list(filter(lambda node: node.op_type == "MultiHeadAttention", model.model.graph.node))
    for idx, node in enumerate(mha_nodes):
        # Detect Q path to MHA
        q_path_1 = model.match_parent_path(node, ["RotaryEmbedding", "Add", "MatMul"], [0, 0, 0])
        q_path_2 = model.match_parent_path(node, ["RotaryEmbedding", "MatMul"], [0, 0])

        q_rotary, q_add, q_matmul = None, None, None
        if q_path_1 is not None:
            q_rotary, q_add, q_matmul = q_path_1
        elif q_path_2 is not None:
            q_rotary, q_matmul = q_path_2

        # Detect K path to MHA
        k_path_1 = model.match_parent_path(node, ["RotaryEmbedding", "Add", "MatMul"], [1, 0, 0])
        k_path_2 = model.match_parent_path(node, ["RotaryEmbedding", "MatMul"], [1, 0])

        k_rotary, k_add, k_matmul = None, None, None
        if k_path_1 is not None:
            k_rotary, k_add, k_matmul = k_path_1
        elif k_path_2 is not None:
            k_rotary, k_matmul = k_path_2

        # Detect V path to MHA
        v_path_1 = model.match_parent_path(node, ["Add", "MatMul"], [2, 0])
        v_path_2 = model.match_parent_path(node, ["MatMul"], [2])

        v_add, v_matmul = None, None
        if v_path_1 is not None:
            v_add, v_matmul = v_path_1
        elif v_path_2 is not None:
            v_matmul = v_path_2[0]

        # Get `interleaved` attribute from RotaryEmbedding
        interleaved = 0
        if q_rotary is not None and k_rotary is not None:
            for att in q_rotary.attribute:
                if att.name == "interleaved":
                    interleaved = att.i

        # Get `num_heads` attribute from MHA
        num_heads = 0
        for att in node.attribute:
            if att.name == "num_heads":
                num_heads = att.i

        # Check if root_input to Q/K/V paths is the same
        root_input_is_same = q_matmul.input[0] == k_matmul.input[0] and k_matmul.input[0] == v_matmul.input[0]

        # Check if Q/K/V paths all have bias or all don't have bias
        all_paths_have_bias = q_add is not None and k_add is not None and v_add is not None
        all_paths_have_no_bias = q_add is None and k_add is None and v_add is None

        # Make PackedMatMul node if possible
        q_input_to_attention, k_input_to_attention, v_input_to_attention = "", "", ""
        if root_input_is_same and (all_paths_have_bias or all_paths_have_no_bias):
            qw = NumpyHelper.to_array(model.get_initializer(q_matmul.input[1]))
            kw = NumpyHelper.to_array(model.get_initializer(k_matmul.input[1]))
            vw = NumpyHelper.to_array(model.get_initializer(v_matmul.input[1]))

            dim = qw.shape[-1]
            qkv_weight = np.stack((qw, kw, vw), axis=1).reshape(dim, 3 * dim)
            qkv_weight = onnx.numpy_helper.from_array(qkv_weight, name=f"QKV_Weight_{idx}")
            model.add_initializer(qkv_weight)

            packed_matmul_node = onnx.helper.make_node(
                "MatMul",
                inputs=[q_matmul.input[0], qkv_weight.name],
                outputs=[f"{qkv_weight.name}_output"],
                name=model.create_node_name("MatMul"),
            )
            model.model.graph.node.extend([packed_matmul_node])
            model.model.graph.node.remove(q_matmul)
            model.model.graph.node.remove(k_matmul)
            model.model.graph.node.remove(v_matmul)
            q_input_to_attention = packed_matmul_node.output[0]

            # Make PackedAdd node if possible
            if all_paths_have_bias:
                qb = NumpyHelper.to_array(model.get_initializer(q_add.input[1]))
                kb = NumpyHelper.to_array(model.get_initializer(k_add.input[1]))
                vb = NumpyHelper.to_array(model.get_initializer(v_add.input[1]))

                dim = qb.shape[-1]
                qkv_bias = np.stack((qb, kb, vb), axis=0).reshape(3 * dim)
                qkv_bias = onnx.numpy_helper.from_array(qkv_bias, name=f"QKV_Bias_{idx}")
                model.add_initializer(qkv_bias)
                packed_add_node = onnx.helper.make_node(
                    "Add",
                    inputs=[packed_matmul_node.output[0], qkv_bias.name],
                    outputs=[f"{qkv_bias.name}_output"],
                )
                model.model.graph.node.extend([packed_add_node])
                model.model.graph.node.remove(q_add)
                model.model.graph.node.remove(k_add)
                model.model.graph.node.remove(v_add)
                q_input_to_attention = packed_add_node.output[0]

        else:
            q_input_to_attention = q_matmul.output[0]
            k_input_to_attention = k_matmul.output[0]
            v_input_to_attention = v_matmul.output[0]

        # Make GQA node
        gqa_node = onnx.helper.make_node(
            "GroupQueryAttention",
            inputs=[
                q_input_to_attention,  # query
                k_input_to_attention,  # key
                v_input_to_attention,  # value
                node.input[6],  # past_key
                node.input[7],  # past_value
                seqlen_k_cast_node.output[0],  # seqlens_k (for attention mask)
                total_seqlen_cast_node.output[0],  # total_seq_len (for attention mask)
                (q_rotary.input[2] if q_rotary is not None else ""),  # cos_cache (for rotary embeddings)
                (q_rotary.input[3] if q_rotary is not None else ""),  # sin_cache (for rotary embeddings)
            ],
            outputs=node.output,
            name=node.name.replace("MultiHeadAttention", "GroupQueryAttention"),
            domain="com.microsoft",
            num_heads=num_heads // world_size,
            kv_num_heads=(num_heads // world_size if kv_num_heads == 0 else kv_num_heads // world_size),
            local_window_size=window_size,
            do_rotary=int(q_rotary is not None and k_rotary is not None),
            rotary_interleaved=interleaved,
        )
        model.model.graph.node.remove(node)
        model.model.graph.node.extend([gqa_node])

        if q_rotary is not None:
            model.model.graph.node.remove(q_rotary)
        if k_rotary is not None:
            model.model.graph.node.remove(k_rotary)

    return model


def update_decoder_subgraph_output_cross_attention(subg: GraphProto):
    input_self_past_0 = 1
    # w/wo attention mask, w/wo hidden_state
    graph_input_names = [gi.name for gi in subg.input]
    while input_self_past_0 < 3 and not graph_input_names[input_self_past_0].startswith("past"):
        input_self_past_0 += 1
    output_self_present_0 = 1

    num_layers = (len(subg.output) - output_self_present_0) // 2
    input_cross_past_0 = 2 * num_layers + input_self_past_0
    past_key_cross_inputs = {subg.input[layer * 2 + input_cross_past_0].name: layer for layer in range(num_layers)}
    print(f"    -- past_key_cross_inputs = {past_key_cross_inputs}")

    input_past_key_cross_0_shape = shape_of(subg.input[input_cross_past_0])
    print(f"past_key_cross_0_shape is {input_past_key_cross_0_shape}")
    batch_size_dim = input_past_key_cross_0_shape[0]
    num_heads_dim = input_past_key_cross_0_shape[1]
    cross_seq_len_dim = input_past_key_cross_0_shape[2]

    num_layer_output_qk = 0
    for node in subg.node:
        if (node.op_type == "DecoderMaskedMultiHeadAttention") and (node.input[1] in past_key_cross_inputs):
            print(f"    -- add cross QK output from: node: {node.name} with output: {node.output}")
            num_layer_output_qk += 1
            layer = past_key_cross_inputs[node.input[1]]
            cross_attention_out_name = f"output_cross_qk_{layer}"
            appended_names = [""] * (3 - len(node.output))
            appended_names.append(cross_attention_out_name)
            node.output.extend(appended_names)
            node.attribute.extend([onnx.helper.make_attribute("output_qk", 1)])

            cross_attention = onnx.helper.make_tensor_value_info(
                cross_attention_out_name,
                TensorProto.FLOAT,
                [batch_size_dim, num_heads_dim, 1, cross_seq_len_dim],
            )
            subg.output.extend([cross_attention])
    if num_layer_output_qk != num_layers:
        raise ValueError(f"Did not add cross QK for all layers{num_layers} vs {num_layer_output_qk}")


def update_decoder_subgraph_share_buffer_and_use_decoder_masked_mha(subg: ModelProto):
    input_self_past_0 = 1
    # w/wo attention mask, w/wo hidden_state
    graph_input_names = [gi.name for gi in subg.input]
    while input_self_past_0 < 3 and not graph_input_names[input_self_past_0].startswith("past"):
        input_self_past_0 += 1
    output_self_past_0 = 1

    num_layers = int((len(subg.input) - input_self_past_0) / 4)
    input_cross_past_0 = 2 * num_layers + input_self_past_0

    new_nodes = []
    old_nodes = []
    for node in subg.node:
        if node.op_type == "MultiHeadAttention":
            old_nodes.extend([node])

    # If not all the MultiHeadAttention nodes are fused, this optimization is not applicable
    if len(old_nodes) < num_layers:
        return False

    # Redirect the RelativePositionBias node's input from past_key_self_0.shape[2] to past_sequence_length.
    # There is only one RelativePositionBias node in T5 decoder subgraph.
    rel_pos_bias_node = None
    for node in subg.node:
        if node.op_type == "RelativePositionBias":
            rel_pos_bias_node = node
            break

    decoder_masked_attention_supported_attr = [
        "past_present_share_buffer",
        "num_heads",
        "scale",
        "mask_filter_value",
        "domain",
    ]

    target_squeezed_past_seq_name = "past_sequence_length_squeezed_int64"
    tensor_names_to_rename, nodes_to_remove = find_past_seq_len_usage(subg)
    if len(tensor_names_to_rename) > 0:
        for name_to_rename in tensor_names_to_rename:
            print(f"Found tensor name `{name_to_rename}` to be renamed to `{target_squeezed_past_seq_name}`")
        for nr in nodes_to_remove:
            print(f"Found node to remove: type = {nr.op_type}, name = {nr.name}")

        squeeze_node = onnx.helper.make_node(
            "Squeeze",
            ["past_sequence_length"],
            ["past_sequence_length_squeezed"],
            name="node_past_sequence_length_squeeze",
        )
        cast_node = onnx.helper.make_node(
            "Cast",
            ["past_sequence_length_squeezed"],
            [target_squeezed_past_seq_name],
            name="node_past_sequence_length_squeeze_cast",
            to=TensorProto.INT64,
        )
        new_nodes.extend([squeeze_node, cast_node])

    for node in subg.node:
        if len(node.output) > 0 and rel_pos_bias_node is not None and node.output[0] == rel_pos_bias_node.input[1]:
            cast_node = onnx.helper.make_node(
                "Cast",
                ["past_sequence_length"],
                ["past_sequence_length_int64"],
                name="past_sequence_length_cast",
                to=TensorProto.INT64,
            )
            node.input[1] = cast_node.output[0]
            new_nodes.extend([cast_node])

        if node.op_type == "MultiHeadAttention":
            kwargs = kwargs_of(node)
            for k in kwargs.copy():
                if k not in decoder_masked_attention_supported_attr:
                    del kwargs[k]

            # note: This logic only apply to T5 model where there is no bias in Attention node.
            nis = [
                node.input[0],  # query
                node.input[1],  # key
                node.input[2],  # value
            ]

            nis.extend([node.input[4] if len(node.input) > 4 else ""])  # 2D mask
            nis.extend([node.input[5] if len(node.input) > 5 else ""])  # attention_bias
            nis.extend([node.input[6] if len(node.input) > 6 else ""])  # past_key
            nis.extend([node.input[7] if len(node.input) > 7 else ""])  # past_value
            nis.extend(["past_sequence_length"])  # past_sequence_length
            nis.extend(["beam_width"])  # beam_width
            nis.extend(["cache_indirection"])  # cache_indirection
            nis.extend([node.input[3] if len(node.input) > 3 else ""])  # bias

            kwargs["past_present_share_buffer"] = 1

            node = onnx.helper.make_node(  # noqa: PLW2901
                "DecoderMaskedMultiHeadAttention",
                nis,
                node.output,
                name=node.name,
                **kwargs,
            )

        if node not in nodes_to_remove:
            for index, name in enumerate(node.input):
                if name in tensor_names_to_rename:
                    node.input[index] = target_squeezed_past_seq_name
            new_nodes.extend([node])

    subg.ClearField("node")
    subg.node.extend(new_nodes)
    orig_input_names = [inp.name for inp in subg.input]

    new_inputs = []
    for i, vi in enumerate(subg.input):
        if i >= input_self_past_0 and i < input_cross_past_0:
            shape = shape_of(vi)
            vi = onnx.helper.make_tensor_value_info(  # noqa: PLW2901
                vi.name,
                elem_type=vi.type.tensor_type.elem_type,
                shape=[shape[0], shape[1], "max_seq_len", shape[3]],
            )
        new_inputs.extend([vi])
    if "past_sequence_length" not in orig_input_names:
        new_inputs.extend(
            [onnx.helper.make_tensor_value_info("past_sequence_length", onnx.TensorProto.INT32, shape=[1])]
        )
    if "beam_width" not in orig_input_names:
        new_inputs.extend([onnx.helper.make_tensor_value_info("beam_width", onnx.TensorProto.INT32, shape=[1])])
    if "cache_indirection" not in orig_input_names:
        new_inputs.extend(
            [
                onnx.helper.make_tensor_value_info(
                    "cache_indirection",
                    onnx.TensorProto.INT32,
                    shape=["batch_size", "beam_width", "max_seq_len"],
                )
            ]
        )
    subg.ClearField("input")
    subg.input.extend(new_inputs)

    new_outputs = []
    for i, vi in enumerate(subg.output):
        if i >= output_self_past_0:
            shape = shape_of(vi)
            vi = onnx.helper.make_tensor_value_info(  # noqa: PLW2901
                vi.name,
                elem_type=vi.type.tensor_type.elem_type,
                shape=[shape[0], shape[1], "max_seq_len", shape[3]],
            )
        new_outputs.extend([vi])
    subg.ClearField("output")
    subg.output.extend(new_outputs)

    return True


def pack_qkv_for_decoder_masked_mha(model_proto: ModelProto):
    onnx_model = OnnxModel(model_proto)
    output_name_to_node = onnx_model.output_name_to_node()

    nodes_to_add = []
    nodes_to_remove = []
    for node in onnx_model.nodes():
        if node.op_type == "DecoderMaskedMultiHeadAttention":
            if "past_key_cross" in node.input[1] and "past_value_cross" in node.input[2]:
                continue
            q_matmul = output_name_to_node[node.input[0]]
            k_matmul = output_name_to_node[node.input[1]]
            v_matmul = output_name_to_node[node.input[2]]

            q_weight = onnx_model.get_initializer(q_matmul.input[1])
            k_weight = onnx_model.get_initializer(k_matmul.input[1])
            v_weight = onnx_model.get_initializer(v_matmul.input[1])
            if not (q_weight and k_weight and v_weight):
                return False

            qw = NumpyHelper.to_array(q_weight)
            kw = NumpyHelper.to_array(k_weight)
            vw = NumpyHelper.to_array(v_weight)

            qkv_weight = np.concatenate([qw, kw, vw], axis=1)

            matmul_node_name = onnx_model.create_node_name("MatMul", name_prefix="MatMul_QKV")
            weight = onnx.helper.make_tensor(
                name=matmul_node_name + "_weight",
                data_type=(TensorProto.FLOAT if q_weight.data_type == 1 else TensorProto.FLOAT16),
                dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
                vals=qkv_weight.flatten().tolist(),
            )

            model_proto.graph.initializer.extend([weight])

            matmul_node = onnx.helper.make_node(
                "MatMul",
                inputs=[q_matmul.input[0], matmul_node_name + "_weight"],
                outputs=[matmul_node_name + "_out"],
                name=matmul_node_name,
            )

            node.input[0] = matmul_node.output[0]
            node.input[1] = ""
            node.input[2] = ""

            nodes_to_add.extend([matmul_node])
            nodes_to_remove.extend([q_matmul, k_matmul, v_matmul])

    onnx_model.add_nodes(nodes_to_add)
    onnx_model.remove_nodes(nodes_to_remove)
    onnx_model.update_graph()

    onnx_model.topological_sort()

    return True


def update_input_shapes_for_gpt2_decoder_model(decoder_onnx_path: str, use_external_data_format: bool = True):
    """Update the input shapes for the inputs "input_ids" and "position_ids" and make the sequence length dim value 1 for each of them.
       The decoder model will be over-written.

    Args:
        decoder_onnx_path (str): Path of GPT-2 decoder onnx model
        use_external_data_format(bool): output tensors to external data or not.
    """

    decoder_model_proto = onnx.load_model(decoder_onnx_path, load_external_data=True)
    for i in range(len(decoder_model_proto.graph.input)):
        if (
            decoder_model_proto.graph.input[i].name == "input_ids"
            or decoder_model_proto.graph.input[i].name == "position_ids"
        ):
            shape_dim_proto = decoder_model_proto.graph.input[i].type.tensor_type.shape.dim[1]

            # Clear any existing dim_param first
            if shape_dim_proto.HasField("dim_param"):
                shape_dim_proto.Clear()

            # Update dim_value to be 1
            shape_dim_proto.dim_value = 1

    OnnxModel.save(
        decoder_model_proto,
        decoder_onnx_path,
        save_as_external_data=use_external_data_format,
    )
    return True


def generate_gpt2_init_decoder(
    decoder_onnx_path: str,
    init_decoder_onnx_path: str,
    use_external_data_format: bool = True,
) -> bool:
    """Generates the initial decoder GPT2 subgraph and saves it for downstream use.
       The initial decoder model will be saved to init_decoder_onnx_path.

    Args:
        decoder_onnx_path (str): Path of GPT-2 decoder onnx model
        init_decoder_onnx_path (str): Path of GPT-2 init decoder onnx model
        use_external_data_format(bool): output tensors to external data or not.
    """
    init_decoder_model_proto = onnx.load_model(decoder_onnx_path, load_external_data=True)

    logits_output_name = init_decoder_model_proto.graph.output[0].name

    gpt2_init_decoder_model = OnnxModel(init_decoder_model_proto)

    output_name_to_node = gpt2_init_decoder_model.output_name_to_node()
    assert logits_output_name in output_name_to_node

    logits_matmul_node = output_name_to_node[logits_output_name]

    # Sanity check - the logits need to be produced by a MatMul node
    if logits_matmul_node.op_type != "MatMul":
        return False

    # Try to find the last residual Add
    # For fp16, there are Casts along the way

    # Normalization Node is : LayerNormalization
    logits_matmul_to_residual_add_path = gpt2_init_decoder_model.match_parent_path(
        logits_matmul_node,
        [
            "Cast",
            "LayerNormalization",
            "Add",
            "Add",
            "Cast",
            "MatMul",
            "Cast",
            "FastGelu",
            "Cast",
            "MatMul",
            "Cast",
            "LayerNormalization",
            "Add",
        ],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )

    # Normalization Node is : SkipLayerNormalization
    if logits_matmul_to_residual_add_path is None:
        logits_matmul_to_residual_add_path = gpt2_init_decoder_model.match_parent_path(
            logits_matmul_node,
            [
                "Cast",
                "SkipLayerNormalization",
                "Cast",
                "MatMul",
                "Cast",
                "FastGelu",
                "Cast",
                "MatMul",
                "Cast",
                "SkipLayerNormalization",
            ],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        )

    # Try without the Casts before and after the MatMuls
    if logits_matmul_to_residual_add_path is None:
        # Normalization Node is : LayerNormalization
        logits_matmul_to_residual_add_path = gpt2_init_decoder_model.match_parent_path(
            logits_matmul_node,
            [
                "LayerNormalization",
                "Add",
                "Add",
                "MatMul",
                "FastGelu",
                "MatMul",
                "LayerNormalization",
                "Add",
            ],
            [0, 0, 1, 0, 0, 0, 0, 0],
        )

        # Normalization Node is : SkipLayerNormalization
        if logits_matmul_to_residual_add_path is None:
            logits_matmul_to_residual_add_path = gpt2_init_decoder_model.match_parent_path(
                logits_matmul_node,
                [
                    "SkipLayerNormalization",
                    "MatMul",
                    "FastGelu",
                    "MatMul",
                    "SkipLayerNormalization",
                ],
                [0, 1, 0, 0, 0],
            )

    # TODO(hasesh): Are there more permutations to try before returning ?
    if logits_matmul_to_residual_add_path is None:
        return False

    residual_add_node = logits_matmul_to_residual_add_path[-1]

    # If the last node in the pattern is SkipLayerNormalization, we need to adjust our pattern searches accordingly
    is_skiplayernorm_path = residual_add_node.op_type == "SkipLayerNormalization"

    # Regular LayerNormalization path
    if not is_skiplayernorm_path:
        residual_add_to_attention_parent_index = 0
        residual_add_to_attention_path = gpt2_init_decoder_model.match_parent_path(
            residual_add_node,
            ["Add", "Cast", "MatMul", "Attention"],
            [residual_add_to_attention_parent_index, 0, 0, 0],
        )

        # Try other parent index of the residual Add node
        if residual_add_to_attention_path is None:
            residual_add_to_attention_parent_index = 1
            residual_add_to_attention_path = gpt2_init_decoder_model.match_parent_path(
                residual_add_node,
                ["Add", "Cast", "MatMul", "Attention"],
                [residual_add_to_attention_parent_index, 0, 0, 0],
            )

        # Try without the Casts before and after the MatMuls
        if residual_add_to_attention_path is None:
            residual_add_to_attention_parent_index = 0
            residual_add_to_attention_path = gpt2_init_decoder_model.match_parent_path(
                residual_add_node,
                ["Add", "MatMul", "Attention"],
                [residual_add_to_attention_parent_index, 0, 0],
            )

        # Try without the Casts before and after the MatMuls and other parent index of the residual Add node
        if residual_add_to_attention_path is None:
            residual_add_to_attention_parent_index = 1
            residual_add_to_attention_path = gpt2_init_decoder_model.match_parent_path(
                residual_add_node,
                ["Add", "MatMul", "Attention"],
                [residual_add_to_attention_parent_index, 0, 0],
            )

    # SkipLayerNormalization path
    else:
        residual_add_to_attention_parent_index = 0
        residual_add_to_attention_path = gpt2_init_decoder_model.match_parent_path(
            residual_add_node,
            ["Cast", "MatMul", "Attention"],
            [residual_add_to_attention_parent_index, 0, 0],
        )

        # Try other parent index of the residual Add node
        if residual_add_to_attention_path is None:
            residual_add_to_attention_parent_index = 1
            residual_add_to_attention_path = gpt2_init_decoder_model.match_parent_path(
                residual_add_node,
                ["Cast", "MatMul", "Attention"],
                [residual_add_to_attention_parent_index, 0, 0],
            )

        # Try without the Casts before and after the MatMuls
        if residual_add_to_attention_path is None:
            residual_add_to_attention_parent_index = 0
            residual_add_to_attention_path = gpt2_init_decoder_model.match_parent_path(
                residual_add_node,
                ["MatMul", "Attention"],
                [residual_add_to_attention_parent_index, 0],
            )

        # Try without the Casts before and after the MatMuls and other parent index of the residual Add node
        if residual_add_to_attention_path is None:
            residual_add_to_attention_parent_index = 1
            residual_add_to_attention_path = gpt2_init_decoder_model.match_parent_path(
                residual_add_node,
                ["MatMul", "Attention"],
                [residual_add_to_attention_parent_index, 0],
            )

    # TODO(hasesh): Are there more permutations to try before returning ?
    if residual_add_to_attention_path is None:
        return False

    residual_add_to_add_parent_index = 0 if residual_add_to_attention_parent_index == 1 else 1

    # Regular LayerNormalization path
    if not is_skiplayernorm_path:
        add_before_residual_add = gpt2_init_decoder_model.match_parent(
            residual_add_node, "Add", residual_add_to_add_parent_index
        )

    # SkipLayerNormalization path
    else:
        add_before_residual_add = gpt2_init_decoder_model.match_parent(
            residual_add_node,
            "SkipLayerNormalization",
            residual_add_to_add_parent_index,
        )

    if add_before_residual_add is None:
        return False

    attention = residual_add_to_attention_path[-1]
    matmul_after_attention = residual_add_to_attention_path[-2]

    slice_starts = onnx.helper.make_tensor(
        name="SliceLastTokenStarts",
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[-1],
    )

    slice_ends = onnx.helper.make_tensor(
        name="SliceLastTokenEnds",
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[-2],
    )

    slice_axes = onnx.helper.make_tensor(
        name="SliceLastTokenAxes",
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[1],
    )

    slice_steps = onnx.helper.make_tensor(
        name="SliceLastTokenSteps",
        data_type=TensorProto.INT32,
        dims=[1],
        vals=[-1],
    )

    gpt2_init_decoder_model.add_initializer(slice_starts)
    gpt2_init_decoder_model.add_initializer(slice_ends)
    gpt2_init_decoder_model.add_initializer(slice_axes)
    gpt2_init_decoder_model.add_initializer(slice_steps)

    # Add Slice node to the graph such that it consumes the output of Attention
    slice_0_output_name = "edge_modified_" + attention.output[0]
    slice_node_0 = onnx.helper.make_node(
        "Slice",
        inputs=[
            attention.output[0],
            "SliceLastTokenStarts",
            "SliceLastTokenEnds",
            "SliceLastTokenAxes",
            "SliceLastTokenSteps",
        ],
        outputs=[slice_0_output_name],
        name=gpt2_init_decoder_model.create_node_name("Slice", "GatherLastToken_0_"),
    )

    # Add Slice node to the graph such that it consumes the output of Add before the residual Add
    # If the 'Add' output is produced by a SkipLayerNormalization node, then adjust its output
    # index appropriately
    add_before_residual_add_output = (
        add_before_residual_add.output[0] if not is_skiplayernorm_path else add_before_residual_add.output[3]
    )

    slice_1_output_name = "edge_modified_" + add_before_residual_add.output[0]
    slice_node_1 = onnx.helper.make_node(
        "Slice",
        inputs=[
            add_before_residual_add_output,
            "SliceLastTokenStarts",
            "SliceLastTokenEnds",
            "SliceLastTokenAxes",
            "SliceLastTokenSteps",
        ],
        outputs=[slice_1_output_name],
        name=gpt2_init_decoder_model.create_node_name("Slice", "GatherLastToken_1_"),
    )

    # Add the 2 Slice nodes
    gpt2_init_decoder_model.add_node(slice_node_0)
    gpt2_init_decoder_model.add_node(slice_node_1)

    # Adjust the input(s) to the nodes consuming the outputs of the added Slice nodes
    gpt2_init_decoder_model.replace_node_input(matmul_after_attention, attention.output[0], slice_0_output_name)
    gpt2_init_decoder_model.replace_node_input(residual_add_node, add_before_residual_add_output, slice_1_output_name)

    # Topologically sort the updated graph
    gpt2_init_decoder_model.topological_sort()

    # Save the init decoder model
    OnnxModel.save(
        init_decoder_model_proto,
        init_decoder_onnx_path,
        save_as_external_data=use_external_data_format,
    )
    return True


def make_dim_proto_numeric_t5(model, config):
    """Make dim_proto numeric.

    Args:
        model: T5 encoder and decoder model.
        config: T5 config.
    """
    sequence_length = str(1)
    num_heads = str(config.num_heads)
    hidden_size = str(config.d_model)
    head_size = str(config.d_kv)

    for tensor in model.graph.output:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            if dim_proto.HasField("dim_param") and dim_proto.dim_param in [
                sequence_length,
                num_heads,
                hidden_size,
                head_size,
            ]:
                dim_value = int(dim_proto.dim_param)
                dim_proto.Clear()
                dim_proto.dim_value = dim_value

    for tensor in model.graph.input:
        for dim_proto in tensor.type.tensor_type.shape.dim:
            if dim_proto.HasField("dim_param") and dim_proto.dim_param in [
                sequence_length,
                num_heads,
                hidden_size,
                head_size,
            ]:
                dim_value = int(dim_proto.dim_param)
                dim_proto.Clear()
                dim_proto.dim_value = dim_value


def convert_generation_model(
    args: argparse.Namespace,
    generation_type: GenerationType = GenerationType.BEAMSEARCH,
):
    """Convert model according to command line arguments.

    Args:
        args (argparse.Namespace): arguments parsed from command line
    """
    is_gpt2: bool = args.model_type == "gpt2"
    is_beamsearch: bool = generation_type == GenerationType.BEAMSEARCH
    is_greedysearch: bool = generation_type == GenerationType.GREEDYSEARCH
    is_sampling: bool = generation_type == GenerationType.SAMPLING
    past_present_share_buffer: bool = args.past_present_share_buffer

    logger.info(f"**** past_present_share_buffer={past_present_share_buffer}")
    if len(args.op_block_list) == 1 and args.op_block_list[0] == "auto":
        if is_gpt2 and args.precision == Precision.FLOAT16.value:
            args.op_block_list = [
                "Add",
                "LayerNormalization",
                "SkipLayerNormalization",
                "FastGelu",
            ]
            logger.info(f"**** Setting op_block_list to {args.op_block_list}")
            logger.info("**** use --op_block_list if you want to override the block operator list.")
        else:
            args.op_block_list = []

    if is_greedysearch or is_sampling:
        if not is_gpt2:
            raise NotImplementedError("Currently only gpt2 with greedy search/sampling is supported")
        if args.output_sequences_scores:
            raise NotImplementedError("output_sequences_scores currently is not supported in greedy search/sampling")
        if args.output_token_scores:
            raise NotImplementedError("output_token_scores currently is not supported in greedy search/sampling")

    # For BeamSearch, sharing buffers for past and present states is only supported
    # when using `use_decoder_masked_attention`
    if past_present_share_buffer and is_beamsearch and not args.use_decoder_masked_attention:
        raise ValueError(
            "`use_decoder_masked_attention` MUST be turned on to use `past_present_share_buffer` in case of BeamSearch"
        )

    # For any kind of sampling, using decoder masked multihead attention is only supported
    # when using `past_present_share_buffer`
    if args.use_decoder_masked_attention and not past_present_share_buffer:
        raise ValueError("`past_present_share_buffer` MUST be turned on to use `use_decoder_masked_attention`")

    # For any kind of sampling, using decoder masked multihead attention is only supported
    # on GPUs
    if args.use_decoder_masked_attention and not args.use_gpu:
        raise ValueError("`use_decoder_masked_attention` option is only supported on GPUs")

    if is_gpt2:
        if args.decoder_onnx and os.path.exists(args.decoder_onnx):
            logger.info(f"skip convert_to_onnx since path existed: {args.decoder_onnx}")
        else:
            if not args.decoder_onnx:
                onnx_filename = f"{args.model_name_or_path}_past_{args.precision}.onnx"
                args.decoder_onnx = Path(Path(args.output).parent, onnx_filename).as_posix()

            logger.info(f"Convert GPT model {args.model_name_or_path} to onnx {args.decoder_onnx} ...")
            gpt2_to_onnx(args)
    else:  # t5 or mt5
        if args.decoder_onnx and args.encoder_decoder_init_onnx:
            logger.info(
                f"skip convert_to_onnx since paths specified: {args.decoder_onnx} and {args.encoder_decoder_init_onnx}"
            )
        else:
            logger.info(f"Convert model {args.model_name_or_path} to onnx ...")
            t5_to_onnx(args)

    # We only want to pad the logits MatMul weight in the decoder for fp16 models.
    # The inherent assumption is that fp16 models run on GPU for which all
    # dims need to be a multiple of 8 to leverage tensor cores.
    # NOTE: We currently only support padding the MatMul logits weight for GPT2 GreedySearch/BeamSearch.
    # This can be expanded to other models/decoding strategies later
    logits_matmul_weight_padded = False
    if (
        not args.disable_pad_vocab_size
        and args.precision == Precision.FLOAT16.value
        and is_gpt2
        and (is_beamsearch or is_greedysearch or is_sampling)
    ):
        logger.info(
            f"Pad logits MatMul weights for optimal MatMul perf in fp16 on {args.decoder_onnx}. "
            "The file will be overwritten."
        )
        logits_matmul_weight_padded = pad_weights_of_logits_matmul(args.decoder_onnx, args.use_external_data_format)
        if not logits_matmul_weight_padded:
            logger.warning(
                "Tried and failed to pad logits MatMul weights. Performance may be sub-optimal for this MatMul"
            )

    gpt2_init_decoder_generated = False
    gpt2_init_decoder_onnx_path = None
    if (
        not args.disable_separate_gpt2_decoder_for_init_run
        and is_gpt2
        and (is_beamsearch or is_greedysearch or is_sampling)
    ):
        logger.info(f"Creating an initial run GPT2 decoder from {args.decoder_onnx}. ")

        gpt2_init_decoder_onnx_filename = f"gpt2_init_past_{args.precision}.onnx"

        gpt2_init_decoder_onnx_path = Path(Path(args.output).parent, gpt2_init_decoder_onnx_filename).as_posix()

        gpt2_init_decoder_generated = generate_gpt2_init_decoder(
            args.decoder_onnx,
            gpt2_init_decoder_onnx_path,
            args.use_external_data_format,
        )

        if not gpt2_init_decoder_generated:
            logger.warning(
                "Tried and failed to generate the init decoder GPT2 model. "
                "Performance may be sub-optimal for the initial decoding run"
            )

        # Update the graph input shapes for the non-initial decoder model to account
        # for the fact that the sequence length will always be 1
        if gpt2_init_decoder_generated and not update_input_shapes_for_gpt2_decoder_model(
            args.decoder_onnx, args.use_external_data_format
        ):
            # Can't proceed further - better to raise an exception
            raise ValueError("Could not update the input shapes for the non-initial decoder subgraph.")

    # If the user explicitly requests running shape inference or if we padded/mutated
    # weight(s)/input shape(s) in the decoder, we want to run shape inference to capture the new
    # shapes
    if logits_matmul_weight_padded or args.run_shape_inference or gpt2_init_decoder_generated:
        logger.info(f"Run symbolic shape inference on {args.decoder_onnx}. The file will be overwritten.")
        shape_inference(args.decoder_onnx, args.use_external_data_format)
        if gpt2_init_decoder_generated:
            logger.info(f"Run symbolic shape inference on {gpt2_init_decoder_onnx_path}. The file will be overwritten.")
            shape_inference(gpt2_init_decoder_onnx_path, args.use_external_data_format)

    if is_gpt2:
        config = GPT2Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    elif args.model_type == "t5":
        config = T5Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = MT5Config.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    if args.verbose:
        logger.info(f"Config={config}")

    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id if is_gpt2 else config.pad_token_id
    vocab_size = config.vocab_size

    # if vocab_size is given in parameters use that.
    if args.vocab_size != -1:
        vocab_size = args.vocab_size

    if args.eos_token_id != -1:
        eos_token_id = args.eos_token_id
    if args.pad_token_id != -1:
        pad_token_id = args.pad_token_id

    decoder_model = onnx.load_model(args.decoder_onnx, load_external_data=True)
    decoder_model.graph.name = f"{args.model_type} decoder"

    gpt2_init_decoder_model = None
    if args.model_type == "gpt2":
        verify_gpt2_subgraph(decoder_model.graph, args.precision)

        # If we generated the init decoder model, verify that as well
        if gpt2_init_decoder_generated:
            gpt2_init_decoder_model = onnx.load_model(gpt2_init_decoder_onnx_path, load_external_data=True)
            gpt2_init_decoder_model.graph.name = f"{args.model_type} init decoder"
            verify_gpt2_subgraph(gpt2_init_decoder_model.graph, args.precision)
    else:
        verify_t5_decoder_subgraph(decoder_model.graph, args.precision)

    inputs = None
    if is_beamsearch:
        inputs = [
            "input_ids",
            "max_length",
            "min_length",
            "num_beams",
            "num_return_sequences",
            "length_penalty",
            "repetition_penalty",
        ]
    elif is_greedysearch or is_sampling:
        inputs = [
            "input_ids",
            "max_length",
            "min_length",
            "repetition_penalty",
        ]

    if args.vocab_mask:
        inputs.append("vocab_mask")
    else:
        inputs.append("")

    if args.prefix_vocab_mask:
        inputs.append("prefix_vocab_mask")
    else:
        inputs.append("")

    if args.custom_attention_mask:
        inputs.append("attention_mask")
    else:
        inputs.append("")

    if is_sampling:
        if args.custom and args.presence_mask:
            inputs.append("presence_mask")
        else:
            inputs.append("")

        if args.seed:
            inputs.append("seed")

    outputs = ["sequences"]
    if args.output_sequences_scores:
        outputs.append("sequences_scores")

    if args.output_token_scores:
        assert args.output_sequences_scores, "--output_token_scores requires --output_sequences_scores"
        outputs.append("scores")

    node = None
    if is_beamsearch:
        node = onnx.helper.make_node(
            "BeamSearch",
            inputs=inputs,
            outputs=outputs,
            name=f"BeamSearch_{args.model_type}",
        )
    elif is_greedysearch:
        node = onnx.helper.make_node(
            "GreedySearch",
            inputs=inputs,
            outputs=outputs,
            name=f"GreedySearch_{args.model_type}",
        )
    elif is_sampling:
        node = onnx.helper.make_node(
            "Sampling",
            inputs=inputs,
            outputs=outputs,
            name=f"Sampling_{args.model_type}",
        )

    node.domain = "com.microsoft"

    attr_to_extend = None
    if is_beamsearch:
        attr_to_extend = [
            onnx.helper.make_attribute("eos_token_id", eos_token_id),
            onnx.helper.make_attribute("pad_token_id", pad_token_id),
            onnx.helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
            onnx.helper.make_attribute("early_stopping", 1 if args.early_stopping else 0),
            onnx.helper.make_attribute("model_type", 0 if args.model_type == "gpt2" else 1),
        ]
    elif is_greedysearch:
        attr_to_extend = [
            onnx.helper.make_attribute("eos_token_id", eos_token_id),
            onnx.helper.make_attribute("pad_token_id", pad_token_id),
            onnx.helper.make_attribute("model_type", 0 if args.model_type == "gpt2" else 1),
            onnx.helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
        ]
    elif is_sampling:
        attr_to_extend = [
            onnx.helper.make_attribute("eos_token_id", eos_token_id),
            onnx.helper.make_attribute("pad_token_id", pad_token_id),
            onnx.helper.make_attribute("model_type", 0 if args.model_type == "gpt2" else 1),
            onnx.helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
            onnx.helper.make_attribute("temperature", args.temperature),
            onnx.helper.make_attribute("top_p", args.top_p),
            onnx.helper.make_attribute("filter_value", args.filter_value),
            onnx.helper.make_attribute("min_tokens_to_keep", args.min_tokens_to_keep),
            onnx.helper.make_attribute("custom", args.custom),
            onnx.helper.make_attribute("presence_penalty", args.presence_penalty),
        ]

    # Explicitly pass in the vocab size via an attribute
    if logits_matmul_weight_padded:
        attr_to_extend.extend([onnx.helper.make_attribute("vocab_size", vocab_size)])

    node.attribute.extend(attr_to_extend)

    initializers = []

    if args.model_type in ["t5", "mt5"]:
        if args.run_shape_inference:
            logger.info(f"Symbolic shape inference on {args.encoder_decoder_init_onnx}. The file will be overwritten.")
            shape_inference(args.encoder_decoder_init_onnx, args.use_external_data_format)
        encoder_model = onnx.load_model(args.encoder_decoder_init_onnx, load_external_data=True)
        suffix = "encoder" if len(encoder_model.graph.input) == 2 else "encoder and decoder init"
        encoder_model.graph.name = f"{args.model_type} {suffix}"
        verify_t5_encoder_decoder_init_subgraph(encoder_model.graph, args.precision)

        make_dim_proto_numeric_t5(encoder_model, config)
        make_dim_proto_numeric_t5(decoder_model, config)

        # Update decoder subgraph in preparation to use past present share buffer
        if past_present_share_buffer:
            if not args.use_decoder_masked_attention:
                raise ValueError("past_present_share_buffer is only supported with use_decoder_masked_attention")

            logger.info(
                "*****update t5 decoder subgraph to share past/present buffer and use decoder_masked_multihead_attention*****"
            )
            if update_decoder_subgraph_share_buffer_and_use_decoder_masked_mha(decoder_model.graph):
                logger.info("*****update t5 decoder subgraph successfully!!!*****")
            else:
                logger.info("*****DecoderMaskedMultiHeadAttention is not applied to T5 decoder*****")

            if pack_qkv_for_decoder_masked_mha(decoder_model):
                logger.info("*****pack qkv for decoder masked mha successfully!!!*****")
            else:
                logger.info("*****pack qkv for decoder masked mha failed!!!*****")

        if not args.disable_shared_initializers:
            # Unique shared initializers from the decoder and decoder_init could reduce memory usage in inference.
            initializers = get_shared_initializers(encoder_model, decoder_model)
            logger.info(
                f"{len(initializers)} shared initializers ({[i.name for i in initializers]}) in encoder and decoder subgraphs are moved to the main graph"
            )

            # TODO(tianleiwu): investigate the following which causes error in inference
            # Move initializer from subgraph to main graph could reduce memory usage in inference.
            # moved_initializers = move_initializers(encoder_model.graph)
            # logger.info(
            #     f"{len(moved_initializers)} initializers ({[i.name for i in moved_initializers]}) from the encoder are moved to the main graph"
            # )
            # initializers.extend(moved_initializers)

        assert config.decoder_start_token_id >= 0, "decoder_start_token_id should be >= 0"

        node.attribute.extend(
            [
                onnx.helper.make_attribute("encoder", encoder_model.graph),
                onnx.helper.make_attribute("decoder", decoder_model.graph),
                onnx.helper.make_attribute("decoder_start_token_id", config.decoder_start_token_id),
            ]
        )
    else:
        if gpt2_init_decoder_generated:
            # Move shared initializers (shared between init decoder and decoder models) to the main
            # graph and remove them from these models
            if not args.disable_shared_initializers:
                # Unique shared initializers from the decoder and decoder_init could reduce memory usage in inference.
                initializers = get_shared_initializers(gpt2_init_decoder_model, decoder_model)
                logger.info(
                    f"{len(initializers)} shared initializers ({[i.name for i in initializers]}) in decoder and init decoder subgraphs are moved to the main graph"
                )

            # Update init decoder subgraph in preparation to use past present share buffer
            if past_present_share_buffer:
                logger.info("*****update init decoder subgraph to make past and present share buffer******************")
                update_decoder_subgraph_past_present_share_buffer(gpt2_init_decoder_model.graph)

            # Update init decoder subgraph in preparation to use DecoderMaskedSelfAttention
            # NOTE: Even if we will not use DecoderMaskedSelfAttention in the init decoder subgraph
            # it makes the runtime changes cleaner if we keep both the init decoder and decoder subgraphs
            # same in terms of the subgraph inputs.
            if args.use_decoder_masked_attention and not update_decoder_subgraph_use_decoder_masked_attention(
                gpt2_init_decoder_model.graph, is_beamsearch, False
            ):
                raise ValueError("Could not update the init decoder subgraph to use DecoderMaskedSelfAttention")

            node.attribute.append(onnx.helper.make_attribute("init_decoder", gpt2_init_decoder_model.graph))
        else:
            # Move initializer from subgraph to main graph could reduce memory usage in inference.
            initializers = move_initializers(decoder_model.graph)
            logger.info(f"{len(initializers)} initializers from the decoder are moved to the main graph")

        # Update decoder subgraph in preparation to use past present share buffer
        if past_present_share_buffer:
            logger.info("*****update decoder subgraph to make past and present share buffer******************")
            update_decoder_subgraph_past_present_share_buffer(decoder_model.graph)

        # Update decoder subgraph in preparation to use DecoderMaskedSelfAttention
        if args.use_decoder_masked_attention and not update_decoder_subgraph_use_decoder_masked_attention(
            decoder_model.graph, is_beamsearch, True
        ):
            raise ValueError("Could not update the decoder subgraph to use DecoderMaskedSelfAttention")

        node.attribute.append(onnx.helper.make_attribute("decoder", decoder_model.graph))

    # graph inputs
    input_ids = onnx.helper.make_tensor_value_info("input_ids", TensorProto.INT32, ["batch_size", "sequence_length"])
    max_length = onnx.helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
    min_length = onnx.helper.make_tensor_value_info("min_length", TensorProto.INT32, [1])
    num_beams = onnx.helper.make_tensor_value_info("num_beams", TensorProto.INT32, [1])
    num_return_sequences = onnx.helper.make_tensor_value_info("num_return_sequences", TensorProto.INT32, [1])
    length_penalty = onnx.helper.make_tensor_value_info("length_penalty", TensorProto.FLOAT, [1])
    repetition_penalty = onnx.helper.make_tensor_value_info("repetition_penalty", TensorProto.FLOAT, [1])

    graph_inputs = None
    if is_beamsearch:
        graph_inputs = [
            input_ids,
            max_length,
            min_length,
            num_beams,
            num_return_sequences,
            length_penalty,
            repetition_penalty,
        ]
    elif is_greedysearch or is_sampling:
        graph_inputs = [
            input_ids,
            max_length,
            min_length,
            repetition_penalty,
        ]

    if args.vocab_mask:
        vocab_mask = onnx.helper.make_tensor_value_info("vocab_mask", TensorProto.INT32, [vocab_size])
        graph_inputs.append(vocab_mask)

    if args.prefix_vocab_mask:
        prefix_vocab_mask = onnx.helper.make_tensor_value_info(
            "prefix_vocab_mask", TensorProto.INT32, ["batch_size", vocab_size]
        )
        graph_inputs.append(prefix_vocab_mask)

    if args.custom_attention_mask:
        attention_mask = onnx.helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT32, ["batch_size", "sequence_length"]
        )
        graph_inputs.append(attention_mask)

    if args.custom and args.presence_mask:
        presence_mask = onnx.helper.make_tensor_value_info(
            "presence_mask", TensorProto.INT32, ["batch_size", vocab_size]
        )
        graph_inputs.append(presence_mask)

    if is_sampling and args.seed:
        seed = onnx.helper.make_tensor_value_info("seed", TensorProto.INT32, [1])
        graph_inputs.append(seed)

    # graph outputs
    sequences = None
    if is_beamsearch:
        sequences = onnx.helper.make_tensor_value_info(
            "sequences",
            TensorProto.INT32,
            ["batch_size", "num_return_sequences", "max_length"],
        )
    elif is_greedysearch or is_sampling:
        sequences = onnx.helper.make_tensor_value_info(
            "sequences",
            TensorProto.INT32,
            ["batch_size", "max_length"],
        )

    graph_outputs = [sequences]

    if args.output_sequences_scores:
        sequences_scores = onnx.helper.make_tensor_value_info(
            "sequences_scores",
            TensorProto.FLOAT,
            ["batch_size", "num_return_sequences"],
        )
        graph_outputs.append(sequences_scores)

    if args.output_token_scores:
        scores = onnx.helper.make_tensor_value_info(
            "scores",
            TensorProto.FLOAT,
            ["max_length - sequence_length", "batch_size", "num_beams", vocab_size],
        )
        graph_outputs.append(scores)

    new_graph = onnx.helper.make_graph(
        [node],
        (f"{args.model_type} beam search" if not is_greedysearch else f"{args.model_type} greedy search"),
        graph_inputs,
        graph_outputs,
        initializers,
    )

    # Create the model
    new_model = onnx.helper.make_model(
        new_graph,
        producer_name="onnxruntime.transformers",
        opset_imports=decoder_model.opset_import,
    )

    # TODO(tianleiwu): move shared initializers from T5 encoder and decoder subgraphs to parent graph to save memory.
    if args.use_external_data_format:
        from packaging import version

        if version.parse(onnx.__version__) < version.parse("1.12.0"):
            logger.warning("Require onnx >= 1.12 to save large (>2GB) model!")

        OnnxModel.save(
            new_model,
            args.output,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
        )
    else:
        onnx.save(new_model, args.output)
    logger.info(f"model save to {args.output}")


def test_torch_performance(
    args: argparse.Namespace,
    model: GPT2LMHeadModel | T5ForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
    bad_words_ids: list[list[int]],
) -> dict[str, Any]:
    """Test PyTorch performance of text generation.

    Args:
        args (argparse.Namespace): arguments parsed from command line
        model (Union[GPT2LMHeadModel, T5ForConditionalGeneration]): PyTorch model
        input_ids (torch.Tensor): input_ids
        attention_mask (torch.Tensor): Attention mask
        eos_token_id (int): EOS token ID
        pad_token_id (int): Padding token ID
        bad_words_ids (List[List[int]]): Words shall not be generated.

    Raises:
        RuntimeError: PyTorch with CUDA is not available for --use_gpu

    Returns:
        Dict[str, Any]: A dictionary with string with metric name, and value can be integer or string.
    """
    if args.use_gpu and not torch.cuda.is_available():
        raise RuntimeError("Please install PyTorch with Cuda for testing gpu performance.")

    if args.precision == Precision.FLOAT16.value:
        model.half()

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.to(device)

    torch.set_grad_enabled(False)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    torch_latency = []
    for _ in range(args.total_runs):
        start = time.time()
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
            return_dict_in_generate=True,
            output_scores=args.output_sequences_scores or args.output_token_scores,
        )
        torch_latency.append(time.time() - start)
    batch_size = input_ids.shape[0]
    from benchmark_helper import get_latency_result

    return get_latency_result(torch_latency, batch_size)


def create_attention_mask(input_ids, pad_token_id):
    attention_mask = np.ones(input_ids.shape, dtype=np.int32)
    for i in range(input_ids.shape[0]):
        abs_pos = 0
        for j in range(input_ids.shape[1]):
            if input_ids[i][j] == pad_token_id and abs_pos == 0:
                attention_mask[i][j] = 0
            else:
                abs_pos += 1
    return attention_mask


def test_gpt_model(
    args: argparse.Namespace,
    sentences: list[str] | None = None,
    is_greedy: bool = False,
):
    """Test GPT-2 model

    Args:
        args (argparse.Namespace): arguments parsed from command line
        sentences (Optional[List[str]], optional): input text. Defaults to None.

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """
    assert args.model_type == "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Use different length sentences to test batching
    if sentences is None:
        sentences = [
            "The product is released",
            "I enjoy walking in the park",
            "Test best way to invest",
        ]

    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    bad_words = "walk in park"
    bad_words_ids = tokenizer.encode(bad_words, add_prefix_space=True)
    bad_words_ids = [[word_id] for word_id in bad_words_ids]  # Convert to list of list
    if args.vocab_mask:
        logger.debug("bad_words_ids", bad_words_ids)  # noqa: PLE1205
    else:
        bad_words_ids = []

    config = model.config
    eos_token_id = config.eos_token_id
    pad_token_id = config.eos_token_id
    vocab_size = config.vocab_size

    torch_decoded_sequences = []
    beam_outputs = None
    if not args.disable_parity:
        print("-" * 50)
        print("Test PyTorch model and beam search with huggingface transformers...")
        beam_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
            return_dict_in_generate=True,
            output_scores=args.output_sequences_scores or args.output_token_scores,
        )
        print("input_ids", input_ids)
        print("huggingface transformers outputs:")
        print("sequences", beam_outputs.sequences)
        if args.output_sequences_scores:
            print("sequences_scores", beam_outputs.sequences_scores)
        if args.output_token_scores:
            print("scores", beam_outputs.scores)
        for i, sequence in enumerate(beam_outputs.sequences):
            decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=True)
            torch_decoded_sequences.append(decoded_sequence)
            print(f"{i}: {decoded_sequence}")

    print("-" * 50)
    print("Testing beam search with onnxruntime...")

    if is_greedy:
        inputs = {
            "input_ids": input_ids.cpu().numpy().astype(np.int32),
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
        }
    else:
        inputs = {
            "input_ids": input_ids.cpu().numpy().astype(np.int32),
            "max_length": np.array([args.max_length], dtype=np.int32),
            "min_length": np.array([args.min_length], dtype=np.int32),
            "num_beams": np.array([args.num_beams], dtype=np.int32),
            "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
            "length_penalty": np.array([args.length_penalty], dtype=np.float32),
            "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
        }

    if args.vocab_mask:
        vocab_mask = np.ones((vocab_size), dtype=np.int32)
        if args.vocab_mask:
            for bad_word_id in bad_words_ids:
                vocab_mask[bad_word_id] = 0
        inputs["vocab_mask"] = vocab_mask

    if args.custom_attention_mask:
        inputs["attention_mask"] = create_attention_mask(input_ids, pad_token_id)

    batch_size = input_ids.shape[0]
    if args.prefix_vocab_mask:
        logger.info("Use prefix vocab mask with all ones in ORT, but no corresponding setting for Torch model.")
        prefix_vocab_mask = np.ones((batch_size, vocab_size), dtype=np.int32)
        inputs["prefix_vocab_mask"] = prefix_vocab_mask

    if args.save_test_data:
        test_data_dir = Path(args.output).parent.as_posix()
        logger.debug("test_data_dir", test_data_dir)  # noqa: PLE1205
        from bert_test_data import output_test_data

        logger.info(f"Saving test_data to {test_data_dir}/test_data_set_* ...")

        all_inputs = [inputs]
        for i, inputs in enumerate(all_inputs):
            dir = os.path.join(test_data_dir, "test_data_set_" + str(i))
            output_test_data(dir, inputs)

    logger.debug("ORT inputs", inputs)  # noqa: PLE1205

    if args.disable_perf_test:
        return

    logger.debug("Creating ort session......")
    ort_session = create_ort_session(args.output, args.use_gpu, args.use_sln_strict_mode)

    logger.debug("Run ort session......")
    result = ort_session.run(None, inputs)

    # Test performance
    latency = []
    for _ in range(args.total_runs):
        start = time.time()
        _ = ort_session.run(None, inputs)
        latency.append(time.time() - start)

    from benchmark_helper import get_latency_result

    batch_size = input_ids.shape[0]
    output = get_latency_result(latency, batch_size)

    print("ORT outputs:")
    sequences = result[0]
    print("sequences", sequences)
    if args.output_sequences_scores:
        print("sequences_scores", result[1])
    if args.output_token_scores:
        print("scores", result[2])

    if is_greedy:
        (batch_size, max_length) = sequences.shape
        ort_decoded_sequences = []
        for i in range(batch_size):
            decoded_sequence = tokenizer.decode(sequences[i], skip_special_tokens=True)
            ort_decoded_sequences.append(decoded_sequence)
            print(f"batch {i} sequence: {decoded_sequence}")
    else:
        (batch_size, num_sequences, max_length) = sequences.shape
        ort_decoded_sequences = []
        for i in range(batch_size):
            for j in range(num_sequences):
                decoded_sequence = tokenizer.decode(sequences[i][j], skip_special_tokens=True)
                ort_decoded_sequences.append(decoded_sequence)
                print(f"batch {i} sequence {j}: {decoded_sequence}")

    if beam_outputs:
        torch_sequences = beam_outputs.sequences.reshape(batch_size, args.num_return_sequences, -1)
        ort_sequences = torch.LongTensor(sequences)
        print("-" * 50)
        print("Torch Sequences:")
        print(torch_sequences)
        print(torch_decoded_sequences)
        print("-" * 50)
        print("ORT Sequences:")
        print(ort_sequences)
        print(ort_decoded_sequences)
        print("-" * 50)
        # Compare the generated text instead of word IDs since ORT pads to max sequence length but Torch not.
        is_same = torch_decoded_sequences == ort_decoded_sequences
        print("Torch and ORT result is", "same" if is_same else "different")
        output["parity"] = is_same

    if args.torch_performance:
        torch_latency_output = test_torch_performance(
            args,
            model,
            input_ids,
            attention_mask,
            eos_token_id,
            pad_token_id,
            bad_words_ids,
        )
        print("Torch Latency", torch_latency_output)

    print("ORT", output)

    return output


def test_t5_model(args: argparse.Namespace, sentences: list[str] | None = None):
    """Test T5 or MT5 model

    Args:
        args (argparse.Namespace): arguments parsed from command line
        sentences (Optional[List[str]], optional): input text. Defaults to None.

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """
    assert args.model_type in ["t5", "mt5"]

    if args.prefix_vocab_mask:
        logger.debug("Skipping parity test as prefix vocab mask is not implemented by Hugging Face")
        return None

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"

    if args.model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
    else:
        model = MT5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )

    # Use different length sentences to test batching
    if sentences is None:
        sentences = [
            "translate English to French: The product is released",
            "summarize: research continues to show that pets bring real health benefits to their owners. Having a dog around can lead to lower levels of stress for both adults and kids.",
            # "summarize: I enjoy walking in the park. It makes my mind feel calm and refreshed. "
            # + "I enjoy looking at the trees, flowers, and wildlife around me, and listening to sound from natural.",
        ]

    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    bad_words = "walk in park"
    bad_words_ids = tokenizer.encode(bad_words)[:-1]  # exclude the last token (EOS)
    bad_words_ids = [[word_id] for word_id in bad_words_ids]  # Convert to list of list
    if args.vocab_mask:
        logger.debug("bad_words_ids", bad_words_ids)  # noqa: PLE1205
    else:
        bad_words_ids = []

    config = model.config
    eos_token_id = config.eos_token_id
    pad_token_id = config.pad_token_id
    vocab_size = config.vocab_size
    logger.debug(f"eos_token_id:{eos_token_id}, pad_token_id:{pad_token_id}, vocab_size:{vocab_size}")

    torch_decoded_sequences = []
    if not args.disable_parity:
        print("-" * 50)
        print("Test PyTorch model and beam search with huggingface transformers...")
        beam_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            bad_words_ids=bad_words_ids if bad_words_ids else None,
            return_dict_in_generate=True,
            output_scores=args.output_sequences_scores or args.output_token_scores,
        )

        print("input_ids", input_ids)
        print("huggingface transformers outputs:")
        print("sequences", beam_outputs.sequences)
        if args.output_sequences_scores:
            print("sequences_scores", beam_outputs.sequences_scores)
        if args.output_token_scores:
            print("scores", beam_outputs.scores)
        for i, sequence in enumerate(beam_outputs.sequences):
            decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=True)
            torch_decoded_sequences.append(decoded_sequence)
            print(f"{i}: {decoded_sequence}")

    print("-" * 50)
    print("Testing beam search with onnxruntime...")

    vocab_mask = np.ones((vocab_size), dtype=np.int32)
    if args.vocab_mask:
        for bad_word_id in bad_words_ids:
            vocab_mask[bad_word_id] = 0

    inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int32),
        "max_length": np.array([args.max_length], dtype=np.int32),
        "min_length": np.array([args.min_length], dtype=np.int32),
        "num_beams": np.array([args.num_beams], dtype=np.int32),
        "num_return_sequences": np.array([args.num_return_sequences], dtype=np.int32),
        "length_penalty": np.array([args.length_penalty], dtype=np.float32),
        "repetition_penalty": np.array([args.repetition_penalty], dtype=np.float32),
    }

    if args.vocab_mask:
        inputs["vocab_mask"] = vocab_mask

    if args.custom_attention_mask:
        inputs["attention_mask"] = create_attention_mask(input_ids, pad_token_id)

    if args.save_test_data:
        test_data_dir = Path(args.output).parent.as_posix()
        logger.debug("test_data_dir", test_data_dir)  # noqa: PLE1205
        from bert_test_data import output_test_data

        all_inputs = [inputs]
        for i, inputs in enumerate(all_inputs):
            dir = os.path.join(test_data_dir, "test_data_set_" + str(i))
            output_test_data(dir, inputs)

    logger.debug("ORT inputs", inputs)  # noqa: PLE1205

    ort_session = create_ort_session(args.output, args.use_gpu, args.use_sln_strict_mode)

    # Test performance
    latency = []
    for _ in range(args.total_runs):
        start = time.time()
        result = ort_session.run(None, inputs)
        latency.append(time.time() - start)
    batch_size = input_ids.shape[0]
    from benchmark_helper import get_latency_result

    output = get_latency_result(latency, batch_size)

    print("ORT outputs:")
    sequences = result[0]
    print("sequences", sequences)
    if args.output_sequences_scores:
        print("sequences_scores", result[1])
    if args.output_token_scores:
        print("scores", result[2])

    (batch_size, num_sequences, max_length) = sequences.shape
    ort_decoded_sequences = []
    for i in range(batch_size):
        for j in range(num_sequences):
            decoded_sequence = tokenizer.decode(sequences[i][j], skip_special_tokens=True)
            ort_decoded_sequences.append(decoded_sequence)
            print(f"batch {i} sequence {j}: {decoded_sequence}")

    if not args.disable_parity:
        torch_sequences = beam_outputs.sequences.reshape(batch_size, args.num_return_sequences, -1)
        ort_sequences = torch.LongTensor(sequences)
        print("-" * 50)
        print("Torch Sequences:")
        print(torch_sequences)
        print(torch_decoded_sequences)
        print("-" * 50)
        print("ORT Sequences:")
        print(ort_sequences)
        print(ort_decoded_sequences)
        print("-" * 50)
        # Compare the generated text instead of word IDs since ORT pads to max sequence length but Torch not.
        is_same = torch_decoded_sequences == ort_decoded_sequences
        print("Torch and ORT result is ", "same" if is_same else "different")
        output["parity"] = is_same

    if args.torch_performance:
        torch_latency_output = test_torch_performance(
            args,
            model,
            input_ids,
            attention_mask,
            eos_token_id,
            pad_token_id,
            bad_words_ids,
        )
        print("Torch Latency", torch_latency_output)

    print("ORT", output)
    return output


def main(argv: list[str] | None = None, sentences: list[str] | None = None):
    """Main entry function

    Args:
        argv (Optional[List[str]], optional): _description_. Defaults to None.
        sentences (Optional[List[str]], optional): input text. Defaults to None.

    Raises:
        ValueError: Path does not exist: --encoder_decoder_init_onnx
        ValueError: Path does not exist: --decoder_onnx
        ValueError: --decoder_onnx and --encoder_decoder_init_onnx are not used together for T5

    Returns:
        Union[Dict[str, Any], None]: A dictionary with string with metric name, and value can be integer or string.
    """

    args = parse_arguments(argv)
    setup_logger(args.verbose)

    if args.model_type in ["t5", "mt5"]:
        if args.encoder_decoder_init_onnx and not os.path.exists(args.encoder_decoder_init_onnx):
            raise ValueError(f"Path does not exist: --encoder_decoder_init_onnx {args.encoder_decoder_init_onnx}")
        if args.decoder_onnx and not os.path.exists(args.decoder_onnx):
            raise ValueError(f"Path does not exist: --decoder_onnx {args.decoder_onnx}")
        if (args.encoder_decoder_init_onnx and not args.decoder_onnx) or (
            args.decoder_onnx and not args.encoder_decoder_init_onnx
        ):
            raise ValueError("--decoder_onnx shall use together with --encoder_decoder_init_onnx")

    is_greedy = args.num_beams == 1 and args.num_return_sequences == 1

    if args.model_type == "gpt2" and is_greedy:
        if args.top_p > 0.0 and args.top_p < 1.0:
            convert_generation_model(args, GenerationType.SAMPLING)
            logger.info(
                "The test for gpt2_sampling onnx model is limited to non-custom model with small top_p(e.g <=0.01) value. The result should be the same as gpt2 greedy search."
            )
            if args.top_p > 0.01 or args.custom or args.seed:
                return
        else:
            convert_generation_model(args, GenerationType.GREEDYSEARCH)
    else:
        convert_generation_model(args)

    logger.info("start testing model...")
    if args.model_type in ["t5", "mt5"]:
        result = test_t5_model(args, sentences=sentences)
    else:
        result = test_gpt_model(args, sentences=sentences, is_greedy=is_greedy)

    if result:
        if args.use_external_data_format:
            logger.info(f"Output files: {args.output}, {args.output}.data")
        else:
            logger.info(f"Output file: {args.output}")

    return result


if __name__ == "__main__":
    main()