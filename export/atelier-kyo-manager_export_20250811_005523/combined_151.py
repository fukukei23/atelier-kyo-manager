
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\oracle\provision.py ===
# dialects/oracle/provision.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from ... import create_engine
from ... import exc
from ... import inspect
from ...engine import url as sa_url
from ...testing.provision import configure_follower
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_post_tables
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import follower_url_from_main
from ...testing.provision import log
from ...testing.provision import post_configure_engine
from ...testing.provision import run_reap_dbs
from ...testing.provision import set_default_schema_on_connection
from ...testing.provision import stop_test_class_outside_fixtures
from ...testing.provision import temp_table_keyword_args
from ...testing.provision import update_db_opts


@create_db.for_db("oracle")
def _oracle_create_db(cfg, eng, ident):
    # NOTE: make sure you've run "ALTER DATABASE default tablespace users" or
    # similar, so that the default tablespace is not "system"; reflection will
    # fail otherwise
    with eng.begin() as conn:
        conn.exec_driver_sql("create user %s identified by xe" % ident)
        conn.exec_driver_sql("create user %s_ts1 identified by xe" % ident)
        conn.exec_driver_sql("create user %s_ts2 identified by xe" % ident)
        conn.exec_driver_sql("grant dba to %s" % (ident,))
        conn.exec_driver_sql("grant unlimited tablespace to %s" % ident)
        conn.exec_driver_sql("grant unlimited tablespace to %s_ts1" % ident)
        conn.exec_driver_sql("grant unlimited tablespace to %s_ts2" % ident)
        # these are needed to create materialized views
        conn.exec_driver_sql("grant create table to %s" % ident)
        conn.exec_driver_sql("grant create table to %s_ts1" % ident)
        conn.exec_driver_sql("grant create table to %s_ts2" % ident)


@configure_follower.for_db("oracle")
def _oracle_configure_follower(config, ident):
    config.test_schema = "%s_ts1" % ident
    config.test_schema_2 = "%s_ts2" % ident


def _ora_drop_ignore(conn, dbname):
    try:
        conn.exec_driver_sql("drop user %s cascade" % dbname)
        log.info("Reaped db: %s", dbname)
        return True
    except exc.DatabaseError as err:
        log.warning("couldn't drop db: %s", err)
        return False


@drop_all_schema_objects_pre_tables.for_db("oracle")
def _ora_drop_all_schema_objects_pre_tables(cfg, eng):
    _purge_recyclebin(eng)
    _purge_recyclebin(eng, cfg.test_schema)


@drop_all_schema_objects_post_tables.for_db("oracle")
def _ora_drop_all_schema_objects_post_tables(cfg, eng):
    with eng.begin() as conn:
        for syn in conn.dialect._get_synonyms(conn, None, None, None):
            conn.exec_driver_sql(f"drop synonym {syn['synonym_name']}")

        for syn in conn.dialect._get_synonyms(
            conn, cfg.test_schema, None, None
        ):
            conn.exec_driver_sql(
                f"drop synonym {cfg.test_schema}.{syn['synonym_name']}"
            )

        for tmp_table in inspect(conn).get_temp_table_names():
            conn.exec_driver_sql(f"drop table {tmp_table}")


@drop_db.for_db("oracle")
def _oracle_drop_db(cfg, eng, ident):
    with eng.begin() as conn:
        # cx_Oracle seems to occasionally leak open connections when a large
        # suite it run, even if we confirm we have zero references to
        # connection objects.
        # while there is a "kill session" command in Oracle Database,
        # it unfortunately does not release the connection sufficiently.
        _ora_drop_ignore(conn, ident)
        _ora_drop_ignore(conn, "%s_ts1" % ident)
        _ora_drop_ignore(conn, "%s_ts2" % ident)


@stop_test_class_outside_fixtures.for_db("oracle")
def _ora_stop_test_class_outside_fixtures(config, db, cls):
    try:
        _purge_recyclebin(db)
    except exc.DatabaseError as err:
        log.warning("purge recyclebin command failed: %s", err)

    # clear statement cache on all connections that were used
    # https://github.com/oracle/python-cx_Oracle/issues/519

    for cx_oracle_conn in _all_conns:
        try:
            sc = cx_oracle_conn.stmtcachesize
        except db.dialect.dbapi.InterfaceError:
            # connection closed
            pass
        else:
            cx_oracle_conn.stmtcachesize = 0
            cx_oracle_conn.stmtcachesize = sc
    _all_conns.clear()


def _purge_recyclebin(eng, schema=None):
    with eng.begin() as conn:
        if schema is None:
            # run magic command to get rid of identity sequences
            # https://floo.bar/2019/11/29/drop-the-underlying-sequence-of-an-identity-column/  # noqa: E501
            conn.exec_driver_sql("purge recyclebin")
        else:
            # per user: https://community.oracle.com/tech/developers/discussion/2255402/how-to-clear-dba-recyclebin-for-a-particular-user  # noqa: E501
            for owner, object_name, type_ in conn.exec_driver_sql(
                "select owner, object_name,type from "
                "dba_recyclebin where owner=:schema and type='TABLE'",
                {"schema": conn.dialect.denormalize_name(schema)},
            ).all():
                conn.exec_driver_sql(f'purge {type_} {owner}."{object_name}"')


_all_conns = set()


@post_configure_engine.for_db("oracle")
def _oracle_post_configure_engine(url, engine, follower_ident):
    from sqlalchemy import event

    @event.listens_for(engine, "checkout")
    def checkout(dbapi_con, con_record, con_proxy):
        _all_conns.add(dbapi_con)

    @event.listens_for(engine, "checkin")
    def checkin(dbapi_connection, connection_record):
        # work around cx_Oracle issue:
        # https://github.com/oracle/python-cx_Oracle/issues/530
        # invalidate oracle connections that had 2pc set up
        if "cx_oracle_xid" in connection_record.info:
            connection_record.invalidate()


@run_reap_dbs.for_db("oracle")
def _reap_oracle_dbs(url, idents):
    log.info("db reaper connecting to %r", url)
    eng = create_engine(url)
    with eng.begin() as conn:
        log.info("identifiers in file: %s", ", ".join(idents))

        to_reap = conn.exec_driver_sql(
            "select u.username from all_users u where username "
            "like 'TEST_%' and not exists (select username "
            "from v$session where username=u.username)"
        )
        all_names = {username.lower() for (username,) in to_reap}
        to_drop = set()
        for name in all_names:
            if name.endswith("_ts1") or name.endswith("_ts2"):
                continue
            elif name in idents:
                to_drop.add(name)
                if "%s_ts1" % name in all_names:
                    to_drop.add("%s_ts1" % name)
                if "%s_ts2" % name in all_names:
                    to_drop.add("%s_ts2" % name)

        dropped = total = 0
        for total, username in enumerate(to_drop, 1):
            if _ora_drop_ignore(conn, username):
                dropped += 1
        log.info(
            "Dropped %d out of %d stale databases detected", dropped, total
        )


@follower_url_from_main.for_db("oracle")
def _oracle_follower_url_from_main(url, ident):
    url = sa_url.make_url(url)
    return url.set(username=ident, password="xe")


@temp_table_keyword_args.for_db("oracle")
def _oracle_temp_table_keyword_args(cfg, eng):
    return {
        "prefixes": ["GLOBAL TEMPORARY"],
        "oracle_on_commit": "PRESERVE ROWS",
    }


@set_default_schema_on_connection.for_db("oracle")
def _oracle_set_default_schema_on_connection(
    cfg, dbapi_connection, schema_name
):
    cursor = dbapi_connection.cursor()
    cursor.execute("ALTER SESSION SET CURRENT_SCHEMA=%s" % schema_name)
    cursor.close()


@update_db_opts.for_db("oracle")
def _update_db_opts(db_url, db_opts, options):
    """Set database options (db_opts) for a test database that we created."""
    if (
        options.oracledb_thick_mode
        and sa_url.make_url(db_url).get_driver_name() == "oracledb"
    ):
        db_opts["thick_mode"] = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\diagonal.py ===
from sympy.core.sympify import _sympify

from sympy.matrices.expressions import MatrixExpr
from sympy.core import S, Eq, Ge
from sympy.core.mul import Mul
from sympy.functions.special.tensor_functions import KroneckerDelta


class DiagonalMatrix(MatrixExpr):
    """DiagonalMatrix(M) will create a matrix expression that
    behaves as though all off-diagonal elements,
    `M[i, j]` where `i != j`, are zero.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalMatrix, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> D = DiagonalMatrix(MatrixSymbol('x', 2, 3))
    >>> D[1, 2]
    0
    >>> D[1, 1]
    x[1, 1]

    The length of the diagonal -- the lesser of the two dimensions of `M` --
    is accessed through the `diagonal_length` property:

    >>> D.diagonal_length
    2
    >>> DiagonalMatrix(MatrixSymbol('x', n + 1, n)).diagonal_length
    n

    When one of the dimensions is symbolic the other will be treated as
    though it is smaller:

    >>> tall = DiagonalMatrix(MatrixSymbol('x', n, 3))
    >>> tall.diagonal_length
    3
    >>> tall[10, 1]
    0

    When the size of the diagonal is not known, a value of None will
    be returned:

    >>> DiagonalMatrix(MatrixSymbol('x', n, m)).diagonal_length is None
    True

    """
    arg = property(lambda self: self.args[0])

    shape = property(lambda self: self.arg.shape)  # type:ignore

    @property
    def diagonal_length(self):
        r, c = self.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        elif r.is_Integer and not c.is_Integer:
            m = r
        elif c.is_Integer and not r.is_Integer:
            m = c
        elif r == c:
            m = r
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m

    def _entry(self, i, j, **kwargs):
        if self.diagonal_length is not None:
            if Ge(i, self.diagonal_length) is S.true:
                return S.Zero
            elif Ge(j, self.diagonal_length) is S.true:
                return S.Zero
        eq = Eq(i, j)
        if eq is S.true:
            return self.arg[i, i]
        elif eq is S.false:
            return S.Zero
        return self.arg[i, j]*KroneckerDelta(i, j)


class DiagonalOf(MatrixExpr):
    """DiagonalOf(M) will create a matrix expression that
    is equivalent to the diagonal of `M`, represented as
    a single column matrix.

    Examples
    ========

    >>> from sympy import MatrixSymbol, DiagonalOf, Symbol
    >>> n = Symbol('n', integer=True)
    >>> m = Symbol('m', integer=True)
    >>> x = MatrixSymbol('x', 2, 3)
    >>> diag = DiagonalOf(x)
    >>> diag.shape
    (2, 1)

    The diagonal can be addressed like a matrix or vector and will
    return the corresponding element of the original matrix:

    >>> diag[1, 0] == diag[1] == x[1, 1]
    True

    The length of the diagonal -- the lesser of the two dimensions of `M` --
    is accessed through the `diagonal_length` property:

    >>> diag.diagonal_length
    2
    >>> DiagonalOf(MatrixSymbol('x', n + 1, n)).diagonal_length
    n

    When only one of the dimensions is symbolic the other will be
    treated as though it is smaller:

    >>> dtall = DiagonalOf(MatrixSymbol('x', n, 3))
    >>> dtall.diagonal_length
    3

    When the size of the diagonal is not known, a value of None will
    be returned:

    >>> DiagonalOf(MatrixSymbol('x', n, m)).diagonal_length is None
    True

    """
    arg = property(lambda self: self.args[0])
    @property
    def shape(self):
        r, c = self.arg.shape
        if r.is_Integer and c.is_Integer:
            m = min(r, c)
        elif r.is_Integer and not c.is_Integer:
            m = r
        elif c.is_Integer and not r.is_Integer:
            m = c
        elif r == c:
            m = r
        else:
            try:
                m = min(r, c)
            except TypeError:
                m = None
        return m, S.One

    @property
    def diagonal_length(self):
        return self.shape[0]

    def _entry(self, i, j, **kwargs):
        return self.arg._entry(i, i, **kwargs)


class DiagMatrix(MatrixExpr):
    """
    Turn a vector into a diagonal matrix.
    """
    def __new__(cls, vector):
        vector = _sympify(vector)
        obj = MatrixExpr.__new__(cls, vector)
        shape = vector.shape
        dim = shape[1] if shape[0] == 1 else shape[0]
        if vector.shape[0] != 1:
            obj._iscolumn = True
        else:
            obj._iscolumn = False
        obj._shape = (dim, dim)
        obj._vector = vector
        return obj

    @property
    def shape(self):
        return self._shape

    def _entry(self, i, j, **kwargs):
        if self._iscolumn:
            result = self._vector._entry(i, 0, **kwargs)
        else:
            result = self._vector._entry(0, j, **kwargs)
        if i != j:
            result *= KroneckerDelta(i, j)
        return result

    def _eval_transpose(self):
        return self

    def as_explicit(self):
        from sympy.matrices.dense import diag
        return diag(*list(self._vector.as_explicit()))

    def doit(self, **hints):
        from sympy.assumptions import ask, Q
        from sympy.matrices.expressions.matmul import MatMul
        from sympy.matrices.expressions.transpose import Transpose
        from sympy.matrices.dense import eye
        from sympy.matrices.matrixbase import MatrixBase
        vector = self._vector
        # This accounts for shape (1, 1) and identity matrices, among others:
        if ask(Q.diagonal(vector)):
            return vector
        if isinstance(vector, MatrixBase):
            ret = eye(max(vector.shape))
            for i in range(ret.shape[0]):
                ret[i, i] = vector[i]
            return type(vector)(ret)
        if vector.is_MatMul:
            matrices = [arg for arg in vector.args if arg.is_Matrix]
            scalars = [arg for arg in vector.args if arg not in matrices]
            if scalars:
                return Mul.fromiter(scalars)*DiagMatrix(MatMul.fromiter(matrices).doit()).doit()
        if isinstance(vector, Transpose):
            vector = vector.arg
        return DiagMatrix(vector)


def diagonalize_vector(vector):
    return DiagMatrix(vector).doit()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\realfield.py ===
"""Implementation of :class:`RealField` class. """


from sympy.external.gmpy import SYMPY_INTS, MPQ
from sympy.core.numbers import Float
from sympy.polys.domains.field import Field
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.domains.characteristiczero import CharacteristicZero
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public

from mpmath import MPContext
from mpmath.libmp import to_rational as _mpmath_to_rational


def to_rational(s, max_denom, limit=True):

    p, q = _mpmath_to_rational(s._mpf_)

    # Needed for GROUND_TYPES=flint if gmpy2 is installed because mpmath's
    # to_rational() function returns a gmpy2.mpz instance and if MPQ is
    # flint.fmpq then MPQ(p, q) will fail.
    p = int(p)
    q = int(q)

    if not limit or q <= max_denom:
        return p, q

    p0, q0, p1, q1 = 0, 1, 1, 0
    n, d = p, q

    while True:
        a = n//d
        q2 = q0 + a*q1
        if q2 > max_denom:
            break
        p0, q0, p1, q1 = p1, q1, p0 + a*p1, q2
        n, d = d, n - a*d

    k = (max_denom - q0)//q1

    number = MPQ(p, q)
    bound1 = MPQ(p0 + k*p1, q0 + k*q1)
    bound2 = MPQ(p1, q1)

    if not bound2 or not bound1:
        return p, q
    elif abs(bound2 - number) <= abs(bound1 - number):
        return bound2.numerator, bound2.denominator
    else:
        return bound1.numerator, bound1.denominator


@public
class RealField(Field, CharacteristicZero, SimpleDomain):
    """Real numbers up to the given precision. """

    rep = 'RR'

    is_RealField = is_RR = True

    is_Exact = False
    is_Numerical = True
    is_PID = False

    has_assoc_Ring = False
    has_assoc_Field = True

    _default_precision = 53

    @property
    def has_default_precision(self):
        return self.precision == self._default_precision

    @property
    def precision(self):
        return self._context.prec

    @property
    def dps(self):
        return self._context.dps

    @property
    def tolerance(self):
        return self._tolerance

    def __init__(self, prec=None, dps=None, tol=None):
        # XXX: The tol parameter is ignored but is kept for now for backwards
        # compatibility.

        context = MPContext()

        if prec is None and dps is None:
            context.prec = self._default_precision
        elif dps is None:
            context.prec = prec
        elif prec is None:
            context.dps = dps
        else:
            raise TypeError("Cannot set both prec and dps")

        self._context = context

        self._dtype = context.mpf
        self.zero = self.dtype(0)
        self.one = self.dtype(1)

        # Only max_denom here is used for anything and is only used for
        # to_rational.
        self._max_denom = max(2**context.prec // 200, 99)
        self._tolerance = self.one / self._max_denom

    @property
    def tp(self):
        # XXX: Domain treats tp as an alias of dtype. Here we need to two
        # separate things: dtype is a callable to make/convert instances.
        # We use tp with isinstance to check if an object is an instance
        # of the domain already.
        return self._dtype

    def dtype(self, arg):
        # XXX: This is needed because mpmath does not recognise fmpz.
        # It might be better to add conversion routines to mpmath and if that
        # happens then this can be removed.
        if isinstance(arg, SYMPY_INTS):
            arg = int(arg)
        return self._dtype(arg)

    def __eq__(self, other):
        return isinstance(other, RealField) and self.precision == other.precision

    def __hash__(self):
        return hash((self.__class__.__name__, self._dtype, self.precision))

    def to_sympy(self, element):
        """Convert ``element`` to SymPy number. """
        return Float(element, self.dps)

    def from_sympy(self, expr):
        """Convert SymPy's number to ``dtype``. """
        number = expr.evalf(n=self.dps)

        if number.is_Number:
            return self.dtype(number)
        else:
            raise CoercionFailed("expected real number, got %s" % expr)

    def from_ZZ(self, element, base):
        return self.dtype(element)

    def from_ZZ_python(self, element, base):
        return self.dtype(element)

    def from_ZZ_gmpy(self, element, base):
        return self.dtype(int(element))

    # XXX: We need to convert the denominators to int here because mpmath does
    # not recognise mpz. Ideally mpmath would handle this and if it changed to
    # do so then the calls to int here could be removed.

    def from_QQ(self, element, base):
        return self.dtype(element.numerator) / int(element.denominator)

    def from_QQ_python(self, element, base):
        return self.dtype(element.numerator) / int(element.denominator)

    def from_QQ_gmpy(self, element, base):
        return self.dtype(int(element.numerator)) / int(element.denominator)

    def from_AlgebraicField(self, element, base):
        return self.from_sympy(base.to_sympy(element).evalf(self.dps))

    def from_RealField(self, element, base):
        return self.dtype(element)

    def from_ComplexField(self, element, base):
        if not element.imag:
            return self.dtype(element.real)

    def to_rational(self, element, limit=True):
        """Convert a real number to rational number. """
        return to_rational(element, self._max_denom, limit=limit)

    def get_ring(self):
        """Returns a ring associated with ``self``. """
        return self

    def get_exact(self):
        """Returns an exact domain associated with ``self``. """
        from sympy.polys.domains import QQ
        return QQ

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return self.one

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a*b

    def almosteq(self, a, b, tolerance=None):
        """Check if ``a`` and ``b`` are almost equal. """
        return self._context.almosteq(a, b, tolerance)

    def is_square(self, a):
        """Returns ``True`` if ``a >= 0`` and ``False`` otherwise. """
        return a >= 0

    def exsqrt(self, a):
        """Non-negative square root for ``a >= 0`` and ``None`` otherwise.

        Explanation
        ===========
        The square root may be slightly inaccurate due to floating point
        rounding error.
        """
        return a ** 0.5 if a >= 0 else None


RR = RealField()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\templating.py ===
from __future__ import annotations

import typing as t

from jinja2 import BaseLoader
from jinja2 import Environment as BaseEnvironment
from jinja2 import Template
from jinja2 import TemplateNotFound

from .globals import _cv_app
from .globals import _cv_request
from .globals import current_app
from .globals import request
from .helpers import stream_with_context
from .signals import before_render_template
from .signals import template_rendered

if t.TYPE_CHECKING:  # pragma: no cover
    from .app import Flask
    from .sansio.app import App
    from .sansio.scaffold import Scaffold


def _default_template_ctx_processor() -> dict[str, t.Any]:
    """Default template context processor.  Injects `request`,
    `session` and `g`.
    """
    appctx = _cv_app.get(None)
    reqctx = _cv_request.get(None)
    rv: dict[str, t.Any] = {}
    if appctx is not None:
        rv["g"] = appctx.g
    if reqctx is not None:
        rv["request"] = reqctx.request
        rv["session"] = reqctx.session
    return rv


class Environment(BaseEnvironment):
    """Works like a regular Jinja2 environment but has some additional
    knowledge of how Flask's blueprint works so that it can prepend the
    name of the blueprint to referenced templates if necessary.
    """

    def __init__(self, app: App, **options: t.Any) -> None:
        if "loader" not in options:
            options["loader"] = app.create_global_jinja_loader()
        BaseEnvironment.__init__(self, **options)
        self.app = app


class DispatchingJinjaLoader(BaseLoader):
    """A loader that looks for templates in the application and all
    the blueprint folders.
    """

    def __init__(self, app: App) -> None:
        self.app = app

    def get_source(
        self, environment: BaseEnvironment, template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        if self.app.config["EXPLAIN_TEMPLATE_LOADING"]:
            return self._get_source_explained(environment, template)
        return self._get_source_fast(environment, template)

    def _get_source_explained(
        self, environment: BaseEnvironment, template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        attempts = []
        rv: tuple[str, str | None, t.Callable[[], bool] | None] | None
        trv: None | (tuple[str, str | None, t.Callable[[], bool] | None]) = None

        for srcobj, loader in self._iter_loaders(template):
            try:
                rv = loader.get_source(environment, template)
                if trv is None:
                    trv = rv
            except TemplateNotFound:
                rv = None
            attempts.append((loader, srcobj, rv))

        from .debughelpers import explain_template_loading_attempts

        explain_template_loading_attempts(self.app, template, attempts)

        if trv is not None:
            return trv
        raise TemplateNotFound(template)

    def _get_source_fast(
        self, environment: BaseEnvironment, template: str
    ) -> tuple[str, str | None, t.Callable[[], bool] | None]:
        for _srcobj, loader in self._iter_loaders(template):
            try:
                return loader.get_source(environment, template)
            except TemplateNotFound:
                continue
        raise TemplateNotFound(template)

    def _iter_loaders(self, template: str) -> t.Iterator[tuple[Scaffold, BaseLoader]]:
        loader = self.app.jinja_loader
        if loader is not None:
            yield self.app, loader

        for blueprint in self.app.iter_blueprints():
            loader = blueprint.jinja_loader
            if loader is not None:
                yield blueprint, loader

    def list_templates(self) -> list[str]:
        result = set()
        loader = self.app.jinja_loader
        if loader is not None:
            result.update(loader.list_templates())

        for blueprint in self.app.iter_blueprints():
            loader = blueprint.jinja_loader
            if loader is not None:
                for template in loader.list_templates():
                    result.add(template)

        return list(result)


def _render(app: Flask, template: Template, context: dict[str, t.Any]) -> str:
    app.update_template_context(context)
    before_render_template.send(
        app, _async_wrapper=app.ensure_sync, template=template, context=context
    )
    rv = template.render(context)
    template_rendered.send(
        app, _async_wrapper=app.ensure_sync, template=template, context=context
    )
    return rv


def render_template(
    template_name_or_list: str | Template | list[str | Template],
    **context: t.Any,
) -> str:
    """Render a template by name with the given context.

    :param template_name_or_list: The name of the template to render. If
        a list is given, the first name to exist will be rendered.
    :param context: The variables to make available in the template.
    """
    app = current_app._get_current_object()  # type: ignore[attr-defined]
    template = app.jinja_env.get_or_select_template(template_name_or_list)
    return _render(app, template, context)


def render_template_string(source: str, **context: t.Any) -> str:
    """Render a template from the given source string with the given
    context.

    :param source: The source code of the template to render.
    :param context: The variables to make available in the template.
    """
    app = current_app._get_current_object()  # type: ignore[attr-defined]
    template = app.jinja_env.from_string(source)
    return _render(app, template, context)


def _stream(
    app: Flask, template: Template, context: dict[str, t.Any]
) -> t.Iterator[str]:
    app.update_template_context(context)
    before_render_template.send(
        app, _async_wrapper=app.ensure_sync, template=template, context=context
    )

    def generate() -> t.Iterator[str]:
        yield from template.generate(context)
        template_rendered.send(
            app, _async_wrapper=app.ensure_sync, template=template, context=context
        )

    rv = generate()

    # If a request context is active, keep it while generating.
    if request:
        rv = stream_with_context(rv)

    return rv


def stream_template(
    template_name_or_list: str | Template | list[str | Template],
    **context: t.Any,
) -> t.Iterator[str]:
    """Render a template by name with the given context as a stream.
    This returns an iterator of strings, which can be used as a
    streaming response from a view.

    :param template_name_or_list: The name of the template to render. If
        a list is given, the first name to exist will be rendered.
    :param context: The variables to make available in the template.

    .. versionadded:: 2.2
    """
    app = current_app._get_current_object()  # type: ignore[attr-defined]
    template = app.jinja_env.get_or_select_template(template_name_or_list)
    return _stream(app, template, context)


def stream_template_string(source: str, **context: t.Any) -> t.Iterator[str]:
    """Render a template from the given source string with the given
    context as a stream. This returns an iterator of strings, which can
    be used as a streaming response from a view.

    :param source: The source code of the template to render.
    :param context: The variables to make available in the template.

    .. versionadded:: 2.2
    """
    app = current_app._get_current_object()  # type: ignore[attr-defined]
    template = app.jinja_env.from_string(source)
    return _stream(app, template, context)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\prefixes.py ===
"""
Module defining unit prefixe class and some constants.

Constant dict for SI and binary prefixes are defined as PREFIXES and
BIN_PREFIXES.
"""
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.core.singleton import S

class Prefix(Expr):
    """
    This class represent prefixes, with their name, symbol and factor.

    Prefixes are used to create derived units from a given unit. They should
    always be encapsulated into units.

    The factor is constructed from a base (default is 10) to some power, and
    it gives the total multiple or fraction. For example the kilometer km
    is constructed from the meter (factor 1) and the kilo (10 to the power 3,
    i.e. 1000). The base can be changed to allow e.g. binary prefixes.

    A prefix multiplied by something will always return the product of this
    other object times the factor, except if the other object:

    - is a prefix and they can be combined into a new prefix;
    - defines multiplication with prefixes (which is the case for the Unit
      class).
    """
    _op_priority = 13.0
    is_commutative = True

    def __new__(cls, name, abbrev, exponent, base=sympify(10), latex_repr=None):

        name = sympify(name)
        abbrev = sympify(abbrev)
        exponent = sympify(exponent)
        base = sympify(base)

        obj = Expr.__new__(cls, name, abbrev, exponent, base)
        obj._name = name
        obj._abbrev = abbrev
        obj._scale_factor = base**exponent
        obj._exponent = exponent
        obj._base = base
        obj._latex_repr = latex_repr
        return obj

    @property
    def name(self):
        return self._name

    @property
    def abbrev(self):
        return self._abbrev

    @property
    def scale_factor(self):
        return self._scale_factor

    def _latex(self, printer):
        if self._latex_repr is None:
            return r'\text{%s}' % self._abbrev
        return self._latex_repr

    @property
    def base(self):
        return self._base

    def __str__(self):
        return str(self._abbrev)

    def __repr__(self):
        if self.base == 10:
            return "Prefix(%r, %r, %r)" % (
                str(self.name), str(self.abbrev), self._exponent)
        else:
            return "Prefix(%r, %r, %r, %r)" % (
                str(self.name), str(self.abbrev), self._exponent, self.base)

    def __mul__(self, other):
        from sympy.physics.units import Quantity
        if not isinstance(other, (Quantity, Prefix)):
            return super().__mul__(other)

        fact = self.scale_factor * other.scale_factor

        if isinstance(other, Prefix):
            if fact == 1:
                return S.One
            # simplify prefix
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == fact:
                    return PREFIXES[p]
            return fact

        return self.scale_factor * other

    def __truediv__(self, other):
        if not hasattr(other, "scale_factor"):
            return super().__truediv__(other)

        fact = self.scale_factor / other.scale_factor

        if fact == 1:
            return S.One
        elif isinstance(other, Prefix):
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == fact:
                    return PREFIXES[p]
            return fact

        return self.scale_factor / other

    def __rtruediv__(self, other):
        if other == 1:
            for p in PREFIXES:
                if PREFIXES[p].scale_factor == 1 / self.scale_factor:
                    return PREFIXES[p]
        return other / self.scale_factor


def prefix_unit(unit, prefixes):
    """
    Return a list of all units formed by unit and the given prefixes.

    You can use the predefined PREFIXES or BIN_PREFIXES, but you can also
    pass as argument a subdict of them if you do not want all prefixed units.

        >>> from sympy.physics.units.prefixes import (PREFIXES,
        ...                                                 prefix_unit)
        >>> from sympy.physics.units import m
        >>> pref = {"m": PREFIXES["m"], "c": PREFIXES["c"], "d": PREFIXES["d"]}
        >>> prefix_unit(m, pref)  # doctest: +SKIP
        [millimeter, centimeter, decimeter]
    """

    from sympy.physics.units.quantities import Quantity
    from sympy.physics.units import UnitSystem

    prefixed_units = []

    for prefix in prefixes.values():
        quantity = Quantity(
                "%s%s" % (prefix.name, unit.name),
                abbrev=("%s%s" % (prefix.abbrev, unit.abbrev)),
                is_prefixed=True,
           )
        UnitSystem._quantity_dimensional_equivalence_map_global[quantity] = unit
        UnitSystem._quantity_scale_factors_global[quantity] = (prefix.scale_factor, unit)
        prefixed_units.append(quantity)

    return prefixed_units


yotta = Prefix('yotta', 'Y', 24)
zetta = Prefix('zetta', 'Z', 21)
exa = Prefix('exa', 'E', 18)
peta = Prefix('peta', 'P', 15)
tera = Prefix('tera', 'T', 12)
giga = Prefix('giga', 'G', 9)
mega = Prefix('mega', 'M', 6)
kilo = Prefix('kilo', 'k', 3)
hecto = Prefix('hecto', 'h', 2)
deca = Prefix('deca', 'da', 1)
deci = Prefix('deci', 'd', -1)
centi = Prefix('centi', 'c', -2)
milli = Prefix('milli', 'm', -3)
micro = Prefix('micro', 'mu', -6, latex_repr=r"\mu")
nano = Prefix('nano', 'n', -9)
pico = Prefix('pico', 'p', -12)
femto = Prefix('femto', 'f', -15)
atto = Prefix('atto', 'a', -18)
zepto = Prefix('zepto', 'z', -21)
yocto = Prefix('yocto', 'y', -24)


# https://physics.nist.gov/cuu/Units/prefixes.html
PREFIXES = {
    'Y': yotta,
    'Z': zetta,
    'E': exa,
    'P': peta,
    'T': tera,
    'G': giga,
    'M': mega,
    'k': kilo,
    'h': hecto,
    'da': deca,
    'd': deci,
    'c': centi,
    'm': milli,
    'mu': micro,
    'n': nano,
    'p': pico,
    'f': femto,
    'a': atto,
    'z': zepto,
    'y': yocto,
}


kibi = Prefix('kibi', 'Y', 10, 2)
mebi = Prefix('mebi', 'Y', 20, 2)
gibi = Prefix('gibi', 'Y', 30, 2)
tebi = Prefix('tebi', 'Y', 40, 2)
pebi = Prefix('pebi', 'Y', 50, 2)
exbi = Prefix('exbi', 'Y', 60, 2)


# https://physics.nist.gov/cuu/Units/binary.html
BIN_PREFIXES = {
    'Ki': kibi,
    'Mi': mebi,
    'Gi': gibi,
    'Ti': tebi,
    'Pi': pebi,
    'Ei': exbi,
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\debug\console.py ===
from __future__ import annotations

import code
import sys
import typing as t
from contextvars import ContextVar
from types import CodeType

from markupsafe import escape

from .repr import debug_repr
from .repr import dump
from .repr import helper

_stream: ContextVar[HTMLStringO] = ContextVar("werkzeug.debug.console.stream")
_ipy: ContextVar[_InteractiveConsole] = ContextVar("werkzeug.debug.console.ipy")


class HTMLStringO:
    """A StringO version that HTML escapes on write."""

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def seek(self, n: int, mode: int = 0) -> None:
        pass

    def readline(self) -> str:
        if len(self._buffer) == 0:
            return ""
        ret = self._buffer[0]
        del self._buffer[0]
        return ret

    def reset(self) -> str:
        val = "".join(self._buffer)
        del self._buffer[:]
        return val

    def _write(self, x: str) -> None:
        self._buffer.append(x)

    def write(self, x: str) -> None:
        self._write(escape(x))

    def writelines(self, x: t.Iterable[str]) -> None:
        self._write(escape("".join(x)))


class ThreadedStream:
    """Thread-local wrapper for sys.stdout for the interactive console."""

    @staticmethod
    def push() -> None:
        if not isinstance(sys.stdout, ThreadedStream):
            sys.stdout = t.cast(t.TextIO, ThreadedStream())

        _stream.set(HTMLStringO())

    @staticmethod
    def fetch() -> str:
        try:
            stream = _stream.get()
        except LookupError:
            return ""

        return stream.reset()

    @staticmethod
    def displayhook(obj: object) -> None:
        try:
            stream = _stream.get()
        except LookupError:
            return _displayhook(obj)  # type: ignore

        # stream._write bypasses escaping as debug_repr is
        # already generating HTML for us.
        if obj is not None:
            _ipy.get().locals["_"] = obj
            stream._write(debug_repr(obj))

    def __setattr__(self, name: str, value: t.Any) -> None:
        raise AttributeError(f"read only attribute {name}")

    def __dir__(self) -> list[str]:
        return dir(sys.__stdout__)

    def __getattribute__(self, name: str) -> t.Any:
        try:
            stream = _stream.get()
        except LookupError:
            stream = sys.__stdout__  # type: ignore[assignment]

        return getattr(stream, name)

    def __repr__(self) -> str:
        return repr(sys.__stdout__)


# add the threaded stream as display hook
_displayhook = sys.displayhook
sys.displayhook = ThreadedStream.displayhook


class _ConsoleLoader:
    def __init__(self) -> None:
        self._storage: dict[int, str] = {}

    def register(self, code: CodeType, source: str) -> None:
        self._storage[id(code)] = source
        # register code objects of wrapped functions too.
        for var in code.co_consts:
            if isinstance(var, CodeType):
                self._storage[id(var)] = source

    def get_source_by_code(self, code: CodeType) -> str | None:
        try:
            return self._storage[id(code)]
        except KeyError:
            return None


class _InteractiveConsole(code.InteractiveInterpreter):
    locals: dict[str, t.Any]

    def __init__(self, globals: dict[str, t.Any], locals: dict[str, t.Any]) -> None:
        self.loader = _ConsoleLoader()
        locals = {
            **globals,
            **locals,
            "dump": dump,
            "help": helper,
            "__loader__": self.loader,
        }
        super().__init__(locals)
        original_compile = self.compile

        def compile(source: str, filename: str, symbol: str) -> CodeType | None:
            code = original_compile(source, filename, symbol)

            if code is not None:
                self.loader.register(code, source)

            return code

        self.compile = compile  # type: ignore[assignment]
        self.more = False
        self.buffer: list[str] = []

    def runsource(self, source: str, **kwargs: t.Any) -> str:  # type: ignore
        source = f"{source.rstrip()}\n"
        ThreadedStream.push()
        prompt = "... " if self.more else ">>> "
        try:
            source_to_eval = "".join(self.buffer + [source])
            if super().runsource(source_to_eval, "<debugger>", "single"):
                self.more = True
                self.buffer.append(source)
            else:
                self.more = False
                del self.buffer[:]
        finally:
            output = ThreadedStream.fetch()
        return f"{prompt}{escape(source)}{output}"

    def runcode(self, code: CodeType) -> None:
        try:
            exec(code, self.locals)
        except Exception:
            self.showtraceback()

    def showtraceback(self) -> None:
        from .tbtools import DebugTraceback

        exc = t.cast(BaseException, sys.exc_info()[1])
        te = DebugTraceback(exc, skip=1)
        sys.stdout._write(te.render_traceback_html())  # type: ignore

    def showsyntaxerror(self, filename: str | None = None) -> None:
        from .tbtools import DebugTraceback

        exc = t.cast(BaseException, sys.exc_info()[1])
        te = DebugTraceback(exc, skip=4)
        sys.stdout._write(te.render_traceback_html())  # type: ignore

    def write(self, data: str) -> None:
        sys.stdout.write(data)


class Console:
    """An interactive console."""

    def __init__(
        self,
        globals: dict[str, t.Any] | None = None,
        locals: dict[str, t.Any] | None = None,
    ) -> None:
        if locals is None:
            locals = {}
        if globals is None:
            globals = {}
        self._ipy = _InteractiveConsole(globals, locals)

    def eval(self, code: str) -> str:
        _ipy.set(self._ipy)
        old_sys_stdout = sys.stdout
        try:
            return self._ipy.runsource(code)
        finally:
            sys.stdout = old_sys_stdout

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\plot_controller.py ===
from pyglet.window import key
from pyglet.window.mouse import LEFT, RIGHT, MIDDLE
from sympy.plotting.pygletplot.util import get_direction_vectors, get_basis_vectors


class PlotController:

    normal_mouse_sensitivity = 4.0
    modified_mouse_sensitivity = 1.0

    normal_key_sensitivity = 160.0
    modified_key_sensitivity = 40.0

    keymap = {
        key.LEFT: 'left',
        key.A: 'left',
        key.NUM_4: 'left',

        key.RIGHT: 'right',
        key.D: 'right',
        key.NUM_6: 'right',

        key.UP: 'up',
        key.W: 'up',
        key.NUM_8: 'up',

        key.DOWN: 'down',
        key.S: 'down',
        key.NUM_2: 'down',

        key.Z: 'rotate_z_neg',
        key.NUM_1: 'rotate_z_neg',

        key.C: 'rotate_z_pos',
        key.NUM_3: 'rotate_z_pos',

        key.Q: 'spin_left',
        key.NUM_7: 'spin_left',
        key.E: 'spin_right',
        key.NUM_9: 'spin_right',

        key.X: 'reset_camera',
        key.NUM_5: 'reset_camera',

        key.NUM_ADD: 'zoom_in',
        key.PAGEUP: 'zoom_in',
        key.R: 'zoom_in',

        key.NUM_SUBTRACT: 'zoom_out',
        key.PAGEDOWN: 'zoom_out',
        key.F: 'zoom_out',

        key.RSHIFT: 'modify_sensitivity',
        key.LSHIFT: 'modify_sensitivity',

        key.F1: 'rot_preset_xy',
        key.F2: 'rot_preset_xz',
        key.F3: 'rot_preset_yz',
        key.F4: 'rot_preset_perspective',

        key.F5: 'toggle_axes',
        key.F6: 'toggle_axe_colors',

        key.F8: 'save_image'
    }

    def __init__(self, window, *, invert_mouse_zoom=False, **kwargs):
        self.invert_mouse_zoom = invert_mouse_zoom
        self.window = window
        self.camera = window.camera
        self.action = {
            # Rotation around the view Y (up) vector
            'left': False,
            'right': False,
            # Rotation around the view X vector
            'up': False,
            'down': False,
            # Rotation around the view Z vector
            'spin_left': False,
            'spin_right': False,
            # Rotation around the model Z vector
            'rotate_z_neg': False,
            'rotate_z_pos': False,
            # Reset to the default rotation
            'reset_camera': False,
            # Performs camera z-translation
            'zoom_in': False,
            'zoom_out': False,
            # Use alternative sensitivity (speed)
            'modify_sensitivity': False,
            # Rotation presets
            'rot_preset_xy': False,
            'rot_preset_xz': False,
            'rot_preset_yz': False,
            'rot_preset_perspective': False,
            # axes
            'toggle_axes': False,
            'toggle_axe_colors': False,
            # screenshot
            'save_image': False
        }

    def update(self, dt):
        z = 0
        if self.action['zoom_out']:
            z -= 1
        if self.action['zoom_in']:
            z += 1
        if z != 0:
            self.camera.zoom_relative(z/10.0, self.get_key_sensitivity()/10.0)

        dx, dy, dz = 0, 0, 0
        if self.action['left']:
            dx -= 1
        if self.action['right']:
            dx += 1
        if self.action['up']:
            dy -= 1
        if self.action['down']:
            dy += 1
        if self.action['spin_left']:
            dz += 1
        if self.action['spin_right']:
            dz -= 1

        if not self.is_2D():
            if dx != 0:
                self.camera.euler_rotate(dx*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[1]))
            if dy != 0:
                self.camera.euler_rotate(dy*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[0]))
            if dz != 0:
                self.camera.euler_rotate(dz*dt*self.get_key_sensitivity(),
                                         *(get_direction_vectors()[2]))
        else:
            self.camera.mouse_translate(0, 0, dx*dt*self.get_key_sensitivity(),
                                        -dy*dt*self.get_key_sensitivity())

        rz = 0
        if self.action['rotate_z_neg'] and not self.is_2D():
            rz -= 1
        if self.action['rotate_z_pos'] and not self.is_2D():
            rz += 1

        if rz != 0:
            self.camera.euler_rotate(rz*dt*self.get_key_sensitivity(),
                                     *(get_basis_vectors()[2]))

        if self.action['reset_camera']:
            self.camera.reset()

        if self.action['rot_preset_xy']:
            self.camera.set_rot_preset('xy')
        if self.action['rot_preset_xz']:
            self.camera.set_rot_preset('xz')
        if self.action['rot_preset_yz']:
            self.camera.set_rot_preset('yz')
        if self.action['rot_preset_perspective']:
            self.camera.set_rot_preset('perspective')

        if self.action['toggle_axes']:
            self.action['toggle_axes'] = False
            self.camera.axes.toggle_visible()

        if self.action['toggle_axe_colors']:
            self.action['toggle_axe_colors'] = False
            self.camera.axes.toggle_colors()

        if self.action['save_image']:
            self.action['save_image'] = False
            self.window.plot.saveimage()

        return True

    def get_mouse_sensitivity(self):
        if self.action['modify_sensitivity']:
            return self.modified_mouse_sensitivity
        else:
            return self.normal_mouse_sensitivity

    def get_key_sensitivity(self):
        if self.action['modify_sensitivity']:
            return self.modified_key_sensitivity
        else:
            return self.normal_key_sensitivity

    def on_key_press(self, symbol, modifiers):
        if symbol in self.keymap:
            self.action[self.keymap[symbol]] = True

    def on_key_release(self, symbol, modifiers):
        if symbol in self.keymap:
            self.action[self.keymap[symbol]] = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & LEFT:
            if self.is_2D():
                self.camera.mouse_translate(x, y, dx, dy)
            else:
                self.camera.spherical_rotate((x - dx, y - dy), (x, y),
                                             self.get_mouse_sensitivity())
        if buttons & MIDDLE:
            self.camera.zoom_relative([1, -1][self.invert_mouse_zoom]*dy,
                                      self.get_mouse_sensitivity()/20.0)
        if buttons & RIGHT:
            self.camera.mouse_translate(x, y, dx, dy)

    def on_mouse_scroll(self, x, y, dx, dy):
        self.camera.zoom_relative([1, -1][self.invert_mouse_zoom]*dy,
                                  self.get_mouse_sensitivity())

    def is_2D(self):
        functions = self.window.plot._functions
        for i in functions:
            if len(functions[i].i_vars) > 1 or len(functions[i].d_vars) > 2:
                return False
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\cffi\setuptools_ext.py ===
import os
import sys

try:
    basestring
except NameError:
    # Python 3.x
    basestring = str

def error(msg):
    from cffi._shimmed_dist_utils import DistutilsSetupError
    raise DistutilsSetupError(msg)


def execfile(filename, glob):
    # We use execfile() (here rewritten for Python 3) instead of
    # __import__() to load the build script.  The problem with
    # a normal import is that in some packages, the intermediate
    # __init__.py files may already try to import the file that
    # we are generating.
    with open(filename) as f:
        src = f.read()
    src += '\n'      # Python 2.6 compatibility
    code = compile(src, filename, 'exec')
    exec(code, glob, glob)


def add_cffi_module(dist, mod_spec):
    from cffi.api import FFI

    if not isinstance(mod_spec, basestring):
        error("argument to 'cffi_modules=...' must be a str or a list of str,"
              " not %r" % (type(mod_spec).__name__,))
    mod_spec = str(mod_spec)
    try:
        build_file_name, ffi_var_name = mod_spec.split(':')
    except ValueError:
        error("%r must be of the form 'path/build.py:ffi_variable'" %
              (mod_spec,))
    if not os.path.exists(build_file_name):
        ext = ''
        rewritten = build_file_name.replace('.', '/') + '.py'
        if os.path.exists(rewritten):
            ext = ' (rewrite cffi_modules to [%r])' % (
                rewritten + ':' + ffi_var_name,)
        error("%r does not name an existing file%s" % (build_file_name, ext))

    mod_vars = {'__name__': '__cffi__', '__file__': build_file_name}
    execfile(build_file_name, mod_vars)

    try:
        ffi = mod_vars[ffi_var_name]
    except KeyError:
        error("%r: object %r not found in module" % (mod_spec,
                                                     ffi_var_name))
    if not isinstance(ffi, FFI):
        ffi = ffi()      # maybe it's a function instead of directly an ffi
    if not isinstance(ffi, FFI):
        error("%r is not an FFI instance (got %r)" % (mod_spec,
                                                      type(ffi).__name__))
    if not hasattr(ffi, '_assigned_source'):
        error("%r: the set_source() method was not called" % (mod_spec,))
    module_name, source, source_extension, kwds = ffi._assigned_source
    if ffi._windows_unicode:
        kwds = kwds.copy()
        ffi._apply_windows_unicode(kwds)

    if source is None:
        _add_py_module(dist, ffi, module_name)
    else:
        _add_c_module(dist, ffi, module_name, source, source_extension, kwds)

def _set_py_limited_api(Extension, kwds):
    """
    Add py_limited_api to kwds if setuptools >= 26 is in use.
    Do not alter the setting if it already exists.
    Setuptools takes care of ignoring the flag on Python 2 and PyPy.

    CPython itself should ignore the flag in a debugging version
    (by not listing .abi3.so in the extensions it supports), but
    it doesn't so far, creating troubles.  That's why we check
    for "not hasattr(sys, 'gettotalrefcount')" (the 2.7 compatible equivalent
    of 'd' not in sys.abiflags). (http://bugs.python.org/issue28401)

    On Windows, with CPython <= 3.4, it's better not to use py_limited_api
    because virtualenv *still* doesn't copy PYTHON3.DLL on these versions.
    Recently (2020) we started shipping only >= 3.5 wheels, though.  So
    we'll give it another try and set py_limited_api on Windows >= 3.5.
    """
    from cffi import recompiler

    if ('py_limited_api' not in kwds and not hasattr(sys, 'gettotalrefcount')
            and recompiler.USE_LIMITED_API):
        import setuptools
        try:
            setuptools_major_version = int(setuptools.__version__.partition('.')[0])
            if setuptools_major_version >= 26:
                kwds['py_limited_api'] = True
        except ValueError:  # certain development versions of setuptools
            # If we don't know the version number of setuptools, we
            # try to set 'py_limited_api' anyway.  At worst, we get a
            # warning.
            kwds['py_limited_api'] = True
    return kwds

def _add_c_module(dist, ffi, module_name, source, source_extension, kwds):
    # We are a setuptools extension. Need this build_ext for py_limited_api.
    from setuptools.command.build_ext import build_ext
    from cffi._shimmed_dist_utils import Extension, log, mkpath
    from cffi import recompiler

    allsources = ['$PLACEHOLDER']
    allsources.extend(kwds.pop('sources', []))
    kwds = _set_py_limited_api(Extension, kwds)
    ext = Extension(name=module_name, sources=allsources, **kwds)

    def make_mod(tmpdir, pre_run=None):
        c_file = os.path.join(tmpdir, module_name + source_extension)
        log.info("generating cffi module %r" % c_file)
        mkpath(tmpdir)
        # a setuptools-only, API-only hook: called with the "ext" and "ffi"
        # arguments just before we turn the ffi into C code.  To use it,
        # subclass the 'distutils.command.build_ext.build_ext' class and
        # add a method 'def pre_run(self, ext, ffi)'.
        if pre_run is not None:
            pre_run(ext, ffi)
        updated = recompiler.make_c_source(ffi, module_name, source, c_file)
        if not updated:
            log.info("already up-to-date")
        return c_file

    if dist.ext_modules is None:
        dist.ext_modules = []
    dist.ext_modules.append(ext)

    base_class = dist.cmdclass.get('build_ext', build_ext)
    class build_ext_make_mod(base_class):
        def run(self):
            if ext.sources[0] == '$PLACEHOLDER':
                pre_run = getattr(self, 'pre_run', None)
                ext.sources[0] = make_mod(self.build_temp, pre_run)
            base_class.run(self)
    dist.cmdclass['build_ext'] = build_ext_make_mod
    # NB. multiple runs here will create multiple 'build_ext_make_mod'
    # classes.  Even in this case the 'build_ext' command should be
    # run once; but just in case, the logic above does nothing if
    # called again.


def _add_py_module(dist, ffi, module_name):
    from setuptools.command.build_py import build_py
    from setuptools.command.build_ext import build_ext
    from cffi._shimmed_dist_utils import log, mkpath
    from cffi import recompiler

    def generate_mod(py_file):
        log.info("generating cffi module %r" % py_file)
        mkpath(os.path.dirname(py_file))
        updated = recompiler.make_py_source(ffi, module_name, py_file)
        if not updated:
            log.info("already up-to-date")

    base_class = dist.cmdclass.get('build_py', build_py)
    class build_py_make_mod(base_class):
        def run(self):
            base_class.run(self)
            module_path = module_name.split('.')
            module_path[-1] += '.py'
            generate_mod(os.path.join(self.build_lib, *module_path))
        def get_source_files(self):
            # This is called from 'setup.py sdist' only.  Exclude
            # the generate .py module in this case.
            saved_py_modules = self.py_modules
            try:
                if saved_py_modules:
                    self.py_modules = [m for m in saved_py_modules
                                         if m != module_name]
                return base_class.get_source_files(self)
            finally:
                self.py_modules = saved_py_modules
    dist.cmdclass['build_py'] = build_py_make_mod

    # distutils and setuptools have no notion I could find of a
    # generated python module.  If we don't add module_name to
    # dist.py_modules, then things mostly work but there are some
    # combination of options (--root and --record) that will miss
    # the module.  So we add it here, which gives a few apparently
    # harmless warnings about not finding the file outside the
    # build directory.
    # Then we need to hack more in get_source_files(); see above.
    if dist.py_modules is None:
        dist.py_modules = []
    dist.py_modules.append(module_name)

    # the following is only for "build_ext -i"
    base_class_2 = dist.cmdclass.get('build_ext', build_ext)
    class build_ext_make_mod(base_class_2):
        def run(self):
            base_class_2.run(self)
            if self.inplace:
                # from get_ext_fullpath() in distutils/command/build_ext.py
                module_path = module_name.split('.')
                package = '.'.join(module_path[:-1])
                build_py = self.get_finalized_command('build_py')
                package_dir = build_py.get_package_dir(package)
                file_name = module_path[-1] + '.py'
                generate_mod(os.path.join(package_dir, file_name))
    dist.cmdclass['build_ext'] = build_ext_make_mod

def cffi_modules(dist, attr, value):
    assert attr == 'cffi_modules'
    if isinstance(value, basestring):
        value = [value]

    for cffi_module in value:
        add_cffi_module(dist, cffi_module)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_qordered_matmul.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionQOrderedMatMul(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "QOrderedMatMul", "MatMul")

    def fuse(self, node, input_name_to_nodes: dict, output_name_to_node: dict):
        matmul_children = self.model.get_children(node, input_name_to_nodes)

        # Should only have 1 child - Bias Add
        if len(matmul_children) != 1 or matmul_children[0].op_type != "Add":
            return

        bias_add_node = matmul_children[0]

        # Atleast one of the inputs to Bias Add node must be a constant
        bias_add_node_index = 0
        if (
            self.model.get_constant_value(bias_add_node.input[0]) is None
            and self.model.get_constant_value(bias_add_node.input[1]) is None
        ):
            return

        if self.model.get_constant_value(bias_add_node.input[0]) is None:
            bias_add_node_index = 1

        bias_add_children = self.model.get_children(bias_add_node, input_name_to_nodes)

        if len(bias_add_children) != 1:
            return

        bias_add_child = bias_add_children[0]

        # Bias Add can have another Add downstream (Residual Add layer)
        residual_add_node = None

        downstream_quantize_node = None

        if bias_add_child.op_type == "Add":
            residual_add_node = bias_add_child

            residual_add_children = self.model.get_children(residual_add_node, input_name_to_nodes)

            if len(residual_add_children) != 1 or residual_add_children[0].op_type != "QuantizeLinear":
                return

            downstream_quantize_node = residual_add_children[0]

        elif bias_add_child.op_type == "QuantizeLinear":
            downstream_quantize_node = bias_add_child

        else:
            return

        # Make sure the downstream QuantizeLinear has the proper zero points and scales
        if not FusionUtils.check_qdq_node_for_fusion(downstream_quantize_node, self.model):
            return

        # The first input to MatMul should flow through a DequantizeLinear node
        first_path_id, first_input_parent_nodes, _ = self.model.match_parent_paths(
            node,
            [(["DequantizeLinear"], [0])],
            output_name_to_node,
        )

        # If Attention is not fused, this is the pattern to look for
        # leading upto the MatMul
        reshape_node_0 = None
        transpose_node_0 = None
        if first_path_id < 0:
            first_path_id, first_input_parent_nodes, _ = self.model.match_parent_paths(
                node,
                [(["Reshape", "Transpose", "DequantizeLinear", "QuantizeLinear"], [0, 0, 0, 0])],
                output_name_to_node,
            )

            if first_path_id < 0:
                return

            reshape_node_0 = first_input_parent_nodes[0]
            transpose_node_0 = first_input_parent_nodes[1]
            dequantize_node_0 = first_input_parent_nodes[2]
        else:
            dequantize_node_0 = first_input_parent_nodes[0]

        # Make sure the upstream DequantizeLinear-0 has the proper zero points and scales
        if not FusionUtils.check_qdq_node_for_fusion(dequantize_node_0, self.model):
            return

        # The second input to MatMul should flow through a DequantizeLinear node
        dequantize_node_1 = None
        is_weight_transpose_required = True

        weight_path_id, weight_nodes, _ = self.model.match_parent_paths(
            node,
            [(["DequantizeLinear", "QuantizeLinear", "Transpose", "DequantizeLinear"], [1, 0, 0, 0])],
            output_name_to_node,
        )

        if weight_path_id < 0:
            weight_path_id, weight_nodes, _ = self.model.match_parent_paths(
                node,
                [(["DequantizeLinear"], [1])],
                output_name_to_node,
            )

            if weight_path_id < 0:
                return

            dequantize_node_1 = weight_nodes[0]
        else:
            is_weight_transpose_required = False
            dequantize_node_1 = weight_nodes[3]

        # Check if weight 'B' is a constant
        if self.model.get_constant_value(dequantize_node_1.input[0]) is None:
            return

        # Make sure the upstream DequantizeLinear-1 has the proper zero points and scales
        # Per-channel scales are supported for weights alone
        if not FusionUtils.check_qdq_node_for_fusion(dequantize_node_1, self.model, False):
            return

        # Make sure the upstream flow into the Residual Add node flows through a DQ node
        residual_add_dequantize_node = None

        if residual_add_node is not None:
            residual_path_id, residual_input_parent_nodes, _ = self.model.match_parent_paths(
                residual_add_node,
                [
                    (["DequantizeLinear"], [1]),
                ],
                output_name_to_node,
            )

            if residual_path_id < 0:
                return

            residual_add_dequantize_node = residual_input_parent_nodes[0]

        # Make sure the upstream DequantizeLinear to the Residual Add has the proper zero points and scales
        if residual_add_dequantize_node is not None and not FusionUtils.check_qdq_node_for_fusion(
            residual_add_dequantize_node, self.model
        ):
            return

        # Subgraph nodes to be fused
        subgraph_nodes = [node, bias_add_node]  # MatMul + Bias Add

        if residual_add_node is not None:
            subgraph_nodes.extend([residual_add_node])  # Residual Add

        subgraph_nodes.extend(weight_nodes)
        subgraph_nodes.extend([downstream_quantize_node])  # Downstream Q node

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes, downstream_quantize_node.output, input_name_to_nodes, output_name_to_node
        ):
            logger.debug("It is not safe to fuse QOrderedMatMul node. Skip")
            return

        # Deal with the case where-in the Attention subgraph is not fused
        if transpose_node_0 is not None:
            self.model.replace_node_input(transpose_node_0, transpose_node_0.input[0], dequantize_node_0.input[0])

        # Make inputs
        fused_node_inputs = [
            reshape_node_0.output[0] if reshape_node_0 is not None else dequantize_node_0.input[0],
            dequantize_node_0.input[1],
            dequantize_node_1.input[0],
            dequantize_node_1.input[1],
            downstream_quantize_node.input[1],
            bias_add_node.input[bias_add_node_index],
        ]

        if residual_add_node is not None:
            fused_node_inputs.append(residual_add_dequantize_node.input[0])
            fused_node_inputs.append(residual_add_dequantize_node.input[1])

        # The MatMul weight 'B' and 'bias' need some post-processing
        # Transpose weight 'B' from order ROW to order COL
        # This offline transpose is needed only while using the CUDA EP
        # TODO: Make this fusion logic EP-agnostic ?
        if is_weight_transpose_required:
            weight_tensor = self.model.get_initializer(dequantize_node_1.input[0])
            FusionUtils.transpose_2d_int8_tensor(weight_tensor)

        fused_node = helper.make_node(
            "QOrderedMatMul",
            inputs=fused_node_inputs,
            outputs=[downstream_quantize_node.output[0]],
            name=self.model.create_node_name("QOrderedMatMul", name_prefix="QOrderedMatMul"),
        )

        fused_node.attribute.extend([helper.make_attribute("order_A", 1)])
        fused_node.attribute.extend([helper.make_attribute("order_B", 0)])
        fused_node.attribute.extend([helper.make_attribute("order_Y", 1)])

        fused_node.domain = "com.microsoft"

        self.nodes_to_remove.extend(subgraph_nodes)
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\contrib\socks.py ===
# -*- coding: utf-8 -*-
"""
This module contains provisional support for SOCKS proxies from within
urllib3. This module supports SOCKS4, SOCKS4A (an extension of SOCKS4), and
SOCKS5. To enable its functionality, either install PySocks or install this
module with the ``socks`` extra.

The SOCKS implementation supports the full range of urllib3 features. It also
supports the following SOCKS features:

- SOCKS4A (``proxy_url='socks4a://...``)
- SOCKS4 (``proxy_url='socks4://...``)
- SOCKS5 with remote DNS (``proxy_url='socks5h://...``)
- SOCKS5 with local DNS (``proxy_url='socks5://...``)
- Usernames and passwords for the SOCKS proxy

.. note::
   It is recommended to use ``socks5h://`` or ``socks4a://`` schemes in
   your ``proxy_url`` to ensure that DNS resolution is done from the remote
   server instead of client-side when connecting to a domain name.

SOCKS4 supports IPv4 and domain names with the SOCKS4A extension. SOCKS5
supports IPv4, IPv6, and domain names.

When connecting to a SOCKS4 proxy the ``username`` portion of the ``proxy_url``
will be sent as the ``userid`` section of the SOCKS request:

.. code-block:: python

    proxy_url="socks4a://<userid>@proxy-host"

When connecting to a SOCKS5 proxy the ``username`` and ``password`` portion
of the ``proxy_url`` will be sent as the username/password to authenticate
with the proxy:

.. code-block:: python

    proxy_url="socks5h://<username>:<password>@proxy-host"

"""
from __future__ import absolute_import

try:
    import socks
except ImportError:
    import warnings

    from ..exceptions import DependencyWarning

    warnings.warn(
        (
            "SOCKS support in urllib3 requires the installation of optional "
            "dependencies: specifically, PySocks.  For more information, see "
            "https://urllib3.readthedocs.io/en/1.26.x/contrib.html#socks-proxies"
        ),
        DependencyWarning,
    )
    raise

from socket import error as SocketError
from socket import timeout as SocketTimeout

from ..connection import HTTPConnection, HTTPSConnection
from ..connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from ..exceptions import ConnectTimeoutError, NewConnectionError
from ..poolmanager import PoolManager
from ..util.url import parse_url

try:
    import ssl
except ImportError:
    ssl = None


class SOCKSConnection(HTTPConnection):
    """
    A plain-text HTTP connection that connects via a SOCKS proxy.
    """

    def __init__(self, *args, **kwargs):
        self._socks_options = kwargs.pop("_socks_options")
        super(SOCKSConnection, self).__init__(*args, **kwargs)

    def _new_conn(self):
        """
        Establish a new connection via the SOCKS proxy.
        """
        extra_kw = {}
        if self.source_address:
            extra_kw["source_address"] = self.source_address

        if self.socket_options:
            extra_kw["socket_options"] = self.socket_options

        try:
            conn = socks.create_connection(
                (self.host, self.port),
                proxy_type=self._socks_options["socks_version"],
                proxy_addr=self._socks_options["proxy_host"],
                proxy_port=self._socks_options["proxy_port"],
                proxy_username=self._socks_options["username"],
                proxy_password=self._socks_options["password"],
                proxy_rdns=self._socks_options["rdns"],
                timeout=self.timeout,
                **extra_kw
            )

        except SocketTimeout:
            raise ConnectTimeoutError(
                self,
                "Connection to %s timed out. (connect timeout=%s)"
                % (self.host, self.timeout),
            )

        except socks.ProxyError as e:
            # This is fragile as hell, but it seems to be the only way to raise
            # useful errors here.
            if e.socket_err:
                error = e.socket_err
                if isinstance(error, SocketTimeout):
                    raise ConnectTimeoutError(
                        self,
                        "Connection to %s timed out. (connect timeout=%s)"
                        % (self.host, self.timeout),
                    )
                else:
                    raise NewConnectionError(
                        self, "Failed to establish a new connection: %s" % error
                    )
            else:
                raise NewConnectionError(
                    self, "Failed to establish a new connection: %s" % e
                )

        except SocketError as e:  # Defensive: PySocks should catch all these.
            raise NewConnectionError(
                self, "Failed to establish a new connection: %s" % e
            )

        return conn


# We don't need to duplicate the Verified/Unverified distinction from
# urllib3/connection.py here because the HTTPSConnection will already have been
# correctly set to either the Verified or Unverified form by that module. This
# means the SOCKSHTTPSConnection will automatically be the correct type.
class SOCKSHTTPSConnection(SOCKSConnection, HTTPSConnection):
    pass


class SOCKSHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = SOCKSConnection


class SOCKSHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = SOCKSHTTPSConnection


class SOCKSProxyManager(PoolManager):
    """
    A version of the urllib3 ProxyManager that routes connections via the
    defined SOCKS proxy.
    """

    pool_classes_by_scheme = {
        "http": SOCKSHTTPConnectionPool,
        "https": SOCKSHTTPSConnectionPool,
    }

    def __init__(
        self,
        proxy_url,
        username=None,
        password=None,
        num_pools=10,
        headers=None,
        **connection_pool_kw
    ):
        parsed = parse_url(proxy_url)

        if username is None and password is None and parsed.auth is not None:
            split = parsed.auth.split(":")
            if len(split) == 2:
                username, password = split
        if parsed.scheme == "socks5":
            socks_version = socks.PROXY_TYPE_SOCKS5
            rdns = False
        elif parsed.scheme == "socks5h":
            socks_version = socks.PROXY_TYPE_SOCKS5
            rdns = True
        elif parsed.scheme == "socks4":
            socks_version = socks.PROXY_TYPE_SOCKS4
            rdns = False
        elif parsed.scheme == "socks4a":
            socks_version = socks.PROXY_TYPE_SOCKS4
            rdns = True
        else:
            raise ValueError("Unable to determine SOCKS version from %s" % proxy_url)

        self.proxy_url = proxy_url

        socks_options = {
            "socks_version": socks_version,
            "proxy_host": parsed.host,
            "proxy_port": parsed.port,
            "username": username,
            "password": password,
            "rdns": rdns,
        }
        connection_pool_kw["_socks_options"] = socks_options

        super(SOCKSProxyManager, self).__init__(
            num_pools, headers, **connection_pool_kw
        )

        self.pool_classes_by_scheme = SOCKSProxyManager.pool_classes_by_scheme

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\json\provider.py ===
from __future__ import annotations

import dataclasses
import decimal
import json
import typing as t
import uuid
import weakref
from datetime import date

from werkzeug.http import http_date

if t.TYPE_CHECKING:  # pragma: no cover
    from werkzeug.sansio.response import Response

    from ..sansio.app import App


class JSONProvider:
    """A standard set of JSON operations for an application. Subclasses
    of this can be used to customize JSON behavior or use different
    JSON libraries.

    To implement a provider for a specific library, subclass this base
    class and implement at least :meth:`dumps` and :meth:`loads`. All
    other methods have default implementations.

    To use a different provider, either subclass ``Flask`` and set
    :attr:`~flask.Flask.json_provider_class` to a provider class, or set
    :attr:`app.json <flask.Flask.json>` to an instance of the class.

    :param app: An application instance. This will be stored as a
        :class:`weakref.proxy` on the :attr:`_app` attribute.

    .. versionadded:: 2.2
    """

    def __init__(self, app: App) -> None:
        self._app: App = weakref.proxy(app)

    def dumps(self, obj: t.Any, **kwargs: t.Any) -> str:
        """Serialize data as JSON.

        :param obj: The data to serialize.
        :param kwargs: May be passed to the underlying JSON library.
        """
        raise NotImplementedError

    def dump(self, obj: t.Any, fp: t.IO[str], **kwargs: t.Any) -> None:
        """Serialize data as JSON and write to a file.

        :param obj: The data to serialize.
        :param fp: A file opened for writing text. Should use the UTF-8
            encoding to be valid JSON.
        :param kwargs: May be passed to the underlying JSON library.
        """
        fp.write(self.dumps(obj, **kwargs))

    def loads(self, s: str | bytes, **kwargs: t.Any) -> t.Any:
        """Deserialize data as JSON.

        :param s: Text or UTF-8 bytes.
        :param kwargs: May be passed to the underlying JSON library.
        """
        raise NotImplementedError

    def load(self, fp: t.IO[t.AnyStr], **kwargs: t.Any) -> t.Any:
        """Deserialize data as JSON read from a file.

        :param fp: A file opened for reading text or UTF-8 bytes.
        :param kwargs: May be passed to the underlying JSON library.
        """
        return self.loads(fp.read(), **kwargs)

    def _prepare_response_obj(
        self, args: tuple[t.Any, ...], kwargs: dict[str, t.Any]
    ) -> t.Any:
        if args and kwargs:
            raise TypeError("app.json.response() takes either args or kwargs, not both")

        if not args and not kwargs:
            return None

        if len(args) == 1:
            return args[0]

        return args or kwargs

    def response(self, *args: t.Any, **kwargs: t.Any) -> Response:
        """Serialize the given arguments as JSON, and return a
        :class:`~flask.Response` object with the ``application/json``
        mimetype.

        The :func:`~flask.json.jsonify` function calls this method for
        the current application.

        Either positional or keyword arguments can be given, not both.
        If no arguments are given, ``None`` is serialized.

        :param args: A single value to serialize, or multiple values to
            treat as a list to serialize.
        :param kwargs: Treat as a dict to serialize.
        """
        obj = self._prepare_response_obj(args, kwargs)
        return self._app.response_class(self.dumps(obj), mimetype="application/json")


def _default(o: t.Any) -> t.Any:
    if isinstance(o, date):
        return http_date(o)

    if isinstance(o, (decimal.Decimal, uuid.UUID)):
        return str(o)

    if dataclasses and dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)  # type: ignore[arg-type]

    if hasattr(o, "__html__"):
        return str(o.__html__())

    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


class DefaultJSONProvider(JSONProvider):
    """Provide JSON operations using Python's built-in :mod:`json`
    library. Serializes the following additional data types:

    -   :class:`datetime.datetime` and :class:`datetime.date` are
        serialized to :rfc:`822` strings. This is the same as the HTTP
        date format.
    -   :class:`uuid.UUID` is serialized to a string.
    -   :class:`dataclasses.dataclass` is passed to
        :func:`dataclasses.asdict`.
    -   :class:`~markupsafe.Markup` (or any object with a ``__html__``
        method) will call the ``__html__`` method to get a string.
    """

    default: t.Callable[[t.Any], t.Any] = staticmethod(_default)  # type: ignore[assignment]
    """Apply this function to any object that :meth:`json.dumps` does
    not know how to serialize. It should return a valid JSON type or
    raise a ``TypeError``.
    """

    ensure_ascii = True
    """Replace non-ASCII characters with escape sequences. This may be
    more compatible with some clients, but can be disabled for better
    performance and size.
    """

    sort_keys = True
    """Sort the keys in any serialized dicts. This may be useful for
    some caching situations, but can be disabled for better performance.
    When enabled, keys must all be strings, they are not converted
    before sorting.
    """

    compact: bool | None = None
    """If ``True``, or ``None`` out of debug mode, the :meth:`response`
    output will not add indentation, newlines, or spaces. If ``False``,
    or ``None`` in debug mode, it will use a non-compact representation.
    """

    mimetype = "application/json"
    """The mimetype set in :meth:`response`."""

    def dumps(self, obj: t.Any, **kwargs: t.Any) -> str:
        """Serialize data as JSON to a string.

        Keyword arguments are passed to :func:`json.dumps`. Sets some
        parameter defaults from the :attr:`default`,
        :attr:`ensure_ascii`, and :attr:`sort_keys` attributes.

        :param obj: The data to serialize.
        :param kwargs: Passed to :func:`json.dumps`.
        """
        kwargs.setdefault("default", self.default)
        kwargs.setdefault("ensure_ascii", self.ensure_ascii)
        kwargs.setdefault("sort_keys", self.sort_keys)
        return json.dumps(obj, **kwargs)

    def loads(self, s: str | bytes, **kwargs: t.Any) -> t.Any:
        """Deserialize data as JSON from a string or bytes.

        :param s: Text or UTF-8 bytes.
        :param kwargs: Passed to :func:`json.loads`.
        """
        return json.loads(s, **kwargs)

    def response(self, *args: t.Any, **kwargs: t.Any) -> Response:
        """Serialize the given arguments as JSON, and return a
        :class:`~flask.Response` object with it. The response mimetype
        will be "application/json" and can be changed with
        :attr:`mimetype`.

        If :attr:`compact` is ``False`` or debug mode is enabled, the
        output will be formatted to be easier to read.

        Either positional or keyword arguments can be given, not both.
        If no arguments are given, ``None`` is serialized.

        :param args: A single value to serialize, or multiple values to
            treat as a list to serialize.
        :param kwargs: Treat as a dict to serialize.
        """
        obj = self._prepare_response_obj(args, kwargs)
        dump_args: dict[str, t.Any] = {}

        if (self.compact is None and self._app.debug) or self.compact is False:
            dump_args.setdefault("indent", 2)
        else:
            dump_args.setdefault("separators", (",", ":"))

        return self._app.response_class(
            f"{self.dumps(obj, **dump_args)}\n", mimetype=self.mimetype
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\fft\__init__.py ===
"""
Discrete Fourier Transform
==========================

.. currentmodule:: numpy.fft

The SciPy module `scipy.fft` is a more comprehensive superset
of `numpy.fft`, which includes only a basic set of routines.

Standard FFTs
-------------

.. autosummary::
   :toctree: generated/

   fft       Discrete Fourier transform.
   ifft      Inverse discrete Fourier transform.
   fft2      Discrete Fourier transform in two dimensions.
   ifft2     Inverse discrete Fourier transform in two dimensions.
   fftn      Discrete Fourier transform in N-dimensions.
   ifftn     Inverse discrete Fourier transform in N dimensions.

Real FFTs
---------

.. autosummary::
   :toctree: generated/

   rfft      Real discrete Fourier transform.
   irfft     Inverse real discrete Fourier transform.
   rfft2     Real discrete Fourier transform in two dimensions.
   irfft2    Inverse real discrete Fourier transform in two dimensions.
   rfftn     Real discrete Fourier transform in N dimensions.
   irfftn    Inverse real discrete Fourier transform in N dimensions.

Hermitian FFTs
--------------

.. autosummary::
   :toctree: generated/

   hfft      Hermitian discrete Fourier transform.
   ihfft     Inverse Hermitian discrete Fourier transform.

Helper routines
---------------

.. autosummary::
   :toctree: generated/

   fftfreq   Discrete Fourier Transform sample frequencies.
   rfftfreq  DFT sample frequencies (for usage with rfft, irfft).
   fftshift  Shift zero-frequency component to center of spectrum.
   ifftshift Inverse of fftshift.


Background information
----------------------

Fourier analysis is fundamentally a method for expressing a function as a
sum of periodic components, and for recovering the function from those
components.  When both the function and its Fourier transform are
replaced with discretized counterparts, it is called the discrete Fourier
transform (DFT).  The DFT has become a mainstay of numerical computing in
part because of a very fast algorithm for computing it, called the Fast
Fourier Transform (FFT), which was known to Gauss (1805) and was brought
to light in its current form by Cooley and Tukey [CT]_.  Press et al. [NR]_
provide an accessible introduction to Fourier analysis and its
applications.

Because the discrete Fourier transform separates its input into
components that contribute at discrete frequencies, it has a great number
of applications in digital signal processing, e.g., for filtering, and in
this context the discretized input to the transform is customarily
referred to as a *signal*, which exists in the *time domain*.  The output
is called a *spectrum* or *transform* and exists in the *frequency
domain*.

Implementation details
----------------------

There are many ways to define the DFT, varying in the sign of the
exponent, normalization, etc.  In this implementation, the DFT is defined
as

.. math::
   A_k =  \\sum_{m=0}^{n-1} a_m \\exp\\left\\{-2\\pi i{mk \\over n}\\right\\}
   \\qquad k = 0,\\ldots,n-1.

The DFT is in general defined for complex inputs and outputs, and a
single-frequency component at linear frequency :math:`f` is
represented by a complex exponential
:math:`a_m = \\exp\\{2\\pi i\\,f m\\Delta t\\}`, where :math:`\\Delta t`
is the sampling interval.

The values in the result follow so-called "standard" order: If ``A =
fft(a, n)``, then ``A[0]`` contains the zero-frequency term (the sum of
the signal), which is always purely real for real inputs. Then ``A[1:n/2]``
contains the positive-frequency terms, and ``A[n/2+1:]`` contains the
negative-frequency terms, in order of decreasingly negative frequency.
For an even number of input points, ``A[n/2]`` represents both positive and
negative Nyquist frequency, and is also purely real for real input.  For
an odd number of input points, ``A[(n-1)/2]`` contains the largest positive
frequency, while ``A[(n+1)/2]`` contains the largest negative frequency.
The routine ``np.fft.fftfreq(n)`` returns an array giving the frequencies
of corresponding elements in the output.  The routine
``np.fft.fftshift(A)`` shifts transforms and their frequencies to put the
zero-frequency components in the middle, and ``np.fft.ifftshift(A)`` undoes
that shift.

When the input `a` is a time-domain signal and ``A = fft(a)``, ``np.abs(A)``
is its amplitude spectrum and ``np.abs(A)**2`` is its power spectrum.
The phase spectrum is obtained by ``np.angle(A)``.

The inverse DFT is defined as

.. math::
   a_m = \\frac{1}{n}\\sum_{k=0}^{n-1}A_k\\exp\\left\\{2\\pi i{mk\\over n}\\right\\}
   \\qquad m = 0,\\ldots,n-1.

It differs from the forward transform by the sign of the exponential
argument and the default normalization by :math:`1/n`.

Type Promotion
--------------

`numpy.fft` promotes ``float32`` and ``complex64`` arrays to ``float64`` and
``complex128`` arrays respectively. For an FFT implementation that does not
promote input arrays, see `scipy.fftpack`.

Normalization
-------------

The argument ``norm`` indicates which direction of the pair of direct/inverse
transforms is scaled and with what normalization factor.
The default normalization (``"backward"``) has the direct (forward) transforms
unscaled and the inverse (backward) transforms scaled by :math:`1/n`. It is
possible to obtain unitary transforms by setting the keyword argument ``norm``
to ``"ortho"`` so that both direct and inverse transforms are scaled by
:math:`1/\\sqrt{n}`. Finally, setting the keyword argument ``norm`` to
``"forward"`` has the direct transforms scaled by :math:`1/n` and the inverse
transforms unscaled (i.e. exactly opposite to the default ``"backward"``).
`None` is an alias of the default option ``"backward"`` for backward
compatibility.

Real and Hermitian transforms
-----------------------------

When the input is purely real, its transform is Hermitian, i.e., the
component at frequency :math:`f_k` is the complex conjugate of the
component at frequency :math:`-f_k`, which means that for real
inputs there is no information in the negative frequency components that
is not already available from the positive frequency components.
The family of `rfft` functions is
designed to operate on real inputs, and exploits this symmetry by
computing only the positive frequency components, up to and including the
Nyquist frequency.  Thus, ``n`` input points produce ``n/2+1`` complex
output points.  The inverses of this family assumes the same symmetry of
its input, and for an output of ``n`` points uses ``n/2+1`` input points.

Correspondingly, when the spectrum is purely real, the signal is
Hermitian.  The `hfft` family of functions exploits this symmetry by
using ``n/2+1`` complex points in the input (time) domain for ``n`` real
points in the frequency domain.

In higher dimensions, FFTs are used, e.g., for image analysis and
filtering.  The computational efficiency of the FFT means that it can
also be a faster way to compute large convolutions, using the property
that a convolution in the time domain is equivalent to a point-by-point
multiplication in the frequency domain.

Higher dimensions
-----------------

In two dimensions, the DFT is defined as

.. math::
   A_{kl} =  \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1}
   a_{mn}\\exp\\left\\{-2\\pi i \\left({mk\\over M}+{nl\\over N}\\right)\\right\\}
   \\qquad k = 0, \\ldots, M-1;\\quad l = 0, \\ldots, N-1,

which extends in the obvious way to higher dimensions, and the inverses
in higher dimensions also extend in the same way.

References
----------

.. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
        machine calculation of complex Fourier series," *Math. Comput.*
        19: 297-301.

.. [NR] Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P.,
        2007, *Numerical Recipes: The Art of Scientific Computing*, ch.
        12-13.  Cambridge Univ. Press, Cambridge, UK.

Examples
--------

For examples, see the various functions.

"""

# TODO: `numpy.fft.helper`` was deprecated in NumPy 2.0. It should
# be deleted once downstream libraries move to `numpy.fft`.
from . import _helper, _pocketfft, helper
from ._helper import *
from ._pocketfft import *

__all__ = _pocketfft.__all__.copy()  # noqa: PLE0605
__all__ += _helper.__all__

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\qft.py ===
"""An implementation of qubits and gates acting on them.

Todo:

* Update docstrings.
* Update tests.
* Implement apply using decompose.
* Implement represent using decompose or something smarter. For this to
  work we first have to implement represent for SWAP.
* Decide if we want upper index to be inclusive in the constructor.
* Fix the printing of Rk gates in plotting.
"""

from sympy.core.expr import Expr
from sympy.core.numbers import (I, Integer, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.functions import sqrt

from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError, QExpr
from sympy.matrices import eye
from sympy.physics.quantum.tensorproduct import matrix_tensor_product

from sympy.physics.quantum.gate import (
    Gate, HadamardGate, SwapGate, OneQubitGate, CGate, PhaseGate, TGate, ZGate
)

from sympy.functions.elementary.complexes import sign

__all__ = [
    'QFT',
    'IQFT',
    'RkGate',
    'Rk'
]

#-----------------------------------------------------------------------------
# Fourier stuff
#-----------------------------------------------------------------------------


class RkGate(OneQubitGate):
    """This is the R_k gate of the QTF."""
    gate_name = 'Rk'
    gate_name_latex = 'R'

    def __new__(cls, *args):
        if len(args) != 2:
            raise QuantumError(
                'Rk gates only take two arguments, got: %r' % args
            )
        # For small k, Rk gates simplify to other gates, using these
        # substitutions give us familiar results for the QFT for small numbers
        # of qubits.
        target = args[0]
        k = args[1]
        if k == 1:
            return ZGate(target)
        elif k == 2:
            return PhaseGate(target)
        elif k == 3:
            return TGate(target)
        args = cls._eval_args(args)
        inst = Expr.__new__(cls, *args)
        inst.hilbert_space = cls._eval_hilbert_space(args)
        return inst

    @classmethod
    def _eval_args(cls, args):
        # Fall back to this, because Gate._eval_args assumes that args is
        # all targets and can't contain duplicates.
        return QExpr._eval_args(args)

    @property
    def k(self):
        return self.label[1]

    @property
    def targets(self):
        return self.label[:1]

    @property
    def gate_name_plot(self):
        return r'$%s_%s$' % (self.gate_name_latex, str(self.k))

    def get_target_matrix(self, format='sympy'):
        if format == 'sympy':
            return Matrix([[1, 0], [0, exp(sign(self.k)*Integer(2)*pi*I/(Integer(2)**abs(self.k)))]])
        raise NotImplementedError(
            'Invalid format for the R_k gate: %r' % format)


Rk = RkGate


class Fourier(Gate):
    """Superclass of Quantum Fourier and Inverse Quantum Fourier Gates."""

    @classmethod
    def _eval_args(self, args):
        if len(args) != 2:
            raise QuantumError(
                'QFT/IQFT only takes two arguments, got: %r' % args
            )
        if args[0] >= args[1]:
            raise QuantumError("Start must be smaller than finish")
        return Gate._eval_args(args)

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        """
            Represents the (I)QFT In the Z Basis
        """
        nqubits = options.get('nqubits', 0)
        if nqubits == 0:
            raise QuantumError(
                'The number of qubits must be given as nqubits.')
        if nqubits < self.min_qubits:
            raise QuantumError(
                'The number of qubits %r is too small for the gate.' % nqubits
            )
        size = self.size
        omega = self.omega

        #Make a matrix that has the basic Fourier Transform Matrix
        arrayFT = [[omega**(
            i*j % size)/sqrt(size) for i in range(size)] for j in range(size)]
        matrixFT = Matrix(arrayFT)

        #Embed the FT Matrix in a higher space, if necessary
        if self.label[0] != 0:
            matrixFT = matrix_tensor_product(eye(2**self.label[0]), matrixFT)
        if self.min_qubits < nqubits:
            matrixFT = matrix_tensor_product(
                matrixFT, eye(2**(nqubits - self.min_qubits)))

        return matrixFT

    @property
    def targets(self):
        return range(self.label[0], self.label[1])

    @property
    def min_qubits(self):
        return self.label[1]

    @property
    def size(self):
        """Size is the size of the QFT matrix"""
        return 2**(self.label[1] - self.label[0])

    @property
    def omega(self):
        return Symbol('omega')


class QFT(Fourier):
    """The forward quantum Fourier transform."""

    gate_name = 'QFT'
    gate_name_latex = 'QFT'

    def decompose(self):
        """Decomposes QFT into elementary gates."""
        start = self.label[0]
        finish = self.label[1]
        circuit = 1
        for level in reversed(range(start, finish)):
            circuit = HadamardGate(level)*circuit
            for i in range(level - start):
                circuit = CGate(level - i - 1, RkGate(level, i + 2))*circuit
        for i in range((finish - start)//2):
            circuit = SwapGate(i + start, finish - i - 1)*circuit
        return circuit

    def _apply_operator_Qubit(self, qubits, **options):
        return qapply(self.decompose()*qubits)

    def _eval_inverse(self):
        return IQFT(*self.args)

    @property
    def omega(self):
        return exp(2*pi*I/self.size)


class IQFT(Fourier):
    """The inverse quantum Fourier transform."""

    gate_name = 'IQFT'
    gate_name_latex = '{QFT^{-1}}'

    def decompose(self):
        """Decomposes IQFT into elementary gates."""
        start = self.args[0]
        finish = self.args[1]
        circuit = 1
        for i in range((finish - start)//2):
            circuit = SwapGate(i + start, finish - i - 1)*circuit
        for level in range(start, finish):
            for i in reversed(range(level - start)):
                circuit = CGate(level - i - 1, RkGate(level, -i - 2))*circuit
            circuit = HadamardGate(level)*circuit
        return circuit

    def _eval_inverse(self):
        return QFT(*self.args)

    @property
    def omega(self):
        return exp(-2*pi*I/self.size)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\locations\_sysconfig.py ===
import logging
import os
import sys
import sysconfig
import typing

from pip._internal.exceptions import InvalidSchemeCombination, UserInstallationInvalid
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.virtualenv import running_under_virtualenv

from .base import change_root, get_major_minor_version, is_osx_framework

logger = logging.getLogger(__name__)


# Notes on _infer_* functions.
# Unfortunately ``get_default_scheme()`` didn't exist before 3.10, so there's no
# way to ask things like "what is the '_prefix' scheme on this platform". These
# functions try to answer that with some heuristics while accounting for ad-hoc
# platforms not covered by CPython's default sysconfig implementation. If the
# ad-hoc implementation does not fully implement sysconfig, we'll fall back to
# a POSIX scheme.

_AVAILABLE_SCHEMES = set(sysconfig.get_scheme_names())

_PREFERRED_SCHEME_API = getattr(sysconfig, "get_preferred_scheme", None)


def _should_use_osx_framework_prefix() -> bool:
    """Check for Apple's ``osx_framework_library`` scheme.

    Python distributed by Apple's Command Line Tools has this special scheme
    that's used when:

    * This is a framework build.
    * We are installing into the system prefix.

    This does not account for ``pip install --prefix`` (also means we're not
    installing to the system prefix), which should use ``posix_prefix``, but
    logic here means ``_infer_prefix()`` outputs ``osx_framework_library``. But
    since ``prefix`` is not available for ``sysconfig.get_default_scheme()``,
    which is the stdlib replacement for ``_infer_prefix()``, presumably Apple
    wouldn't be able to magically switch between ``osx_framework_library`` and
    ``posix_prefix``. ``_infer_prefix()`` returning ``osx_framework_library``
    means its behavior is consistent whether we use the stdlib implementation
    or our own, and we deal with this special case in ``get_scheme()`` instead.
    """
    return (
        "osx_framework_library" in _AVAILABLE_SCHEMES
        and not running_under_virtualenv()
        and is_osx_framework()
    )


def _infer_prefix() -> str:
    """Try to find a prefix scheme for the current platform.

    This tries:

    * A special ``osx_framework_library`` for Python distributed by Apple's
      Command Line Tools, when not running in a virtual environment.
    * Implementation + OS, used by PyPy on Windows (``pypy_nt``).
    * Implementation without OS, used by PyPy on POSIX (``pypy``).
    * OS + "prefix", used by CPython on POSIX (``posix_prefix``).
    * Just the OS name, used by CPython on Windows (``nt``).

    If none of the above works, fall back to ``posix_prefix``.
    """
    if _PREFERRED_SCHEME_API:
        return _PREFERRED_SCHEME_API("prefix")
    if _should_use_osx_framework_prefix():
        return "osx_framework_library"
    implementation_suffixed = f"{sys.implementation.name}_{os.name}"
    if implementation_suffixed in _AVAILABLE_SCHEMES:
        return implementation_suffixed
    if sys.implementation.name in _AVAILABLE_SCHEMES:
        return sys.implementation.name
    suffixed = f"{os.name}_prefix"
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    if os.name in _AVAILABLE_SCHEMES:  # On Windows, prefx is just called "nt".
        return os.name
    return "posix_prefix"


def _infer_user() -> str:
    """Try to find a user scheme for the current platform."""
    if _PREFERRED_SCHEME_API:
        return _PREFERRED_SCHEME_API("user")
    if is_osx_framework() and not running_under_virtualenv():
        suffixed = "osx_framework_user"
    else:
        suffixed = f"{os.name}_user"
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    if "posix_user" not in _AVAILABLE_SCHEMES:  # User scheme unavailable.
        raise UserInstallationInvalid()
    return "posix_user"


def _infer_home() -> str:
    """Try to find a home for the current platform."""
    if _PREFERRED_SCHEME_API:
        return _PREFERRED_SCHEME_API("home")
    suffixed = f"{os.name}_home"
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    return "posix_home"


# Update these keys if the user sets a custom home.
_HOME_KEYS = [
    "installed_base",
    "base",
    "installed_platbase",
    "platbase",
    "prefix",
    "exec_prefix",
]
if sysconfig.get_config_var("userbase") is not None:
    _HOME_KEYS.append("userbase")


def get_scheme(
    dist_name: str,
    user: bool = False,
    home: typing.Optional[str] = None,
    root: typing.Optional[str] = None,
    isolated: bool = False,
    prefix: typing.Optional[str] = None,
) -> Scheme:
    """
    Get the "scheme" corresponding to the input parameters.

    :param dist_name: the name of the package to retrieve the scheme for, used
        in the headers scheme path
    :param user: indicates to use the "user" scheme
    :param home: indicates to use the "home" scheme
    :param root: root under which other directories are re-based
    :param isolated: ignored, but kept for distutils compatibility (where
        this controls whether the user-site pydistutils.cfg is honored)
    :param prefix: indicates to use the "prefix" scheme and provides the
        base directory for the same
    """
    if user and prefix:
        raise InvalidSchemeCombination("--user", "--prefix")
    if home and prefix:
        raise InvalidSchemeCombination("--home", "--prefix")

    if home is not None:
        scheme_name = _infer_home()
    elif user:
        scheme_name = _infer_user()
    else:
        scheme_name = _infer_prefix()

    # Special case: When installing into a custom prefix, use posix_prefix
    # instead of osx_framework_library. See _should_use_osx_framework_prefix()
    # docstring for details.
    if prefix is not None and scheme_name == "osx_framework_library":
        scheme_name = "posix_prefix"

    if home is not None:
        variables = {k: home for k in _HOME_KEYS}
    elif prefix is not None:
        variables = {k: prefix for k in _HOME_KEYS}
    else:
        variables = {}

    paths = sysconfig.get_paths(scheme=scheme_name, vars=variables)

    # Logic here is very arbitrary, we're doing it for compatibility, don't ask.
    # 1. Pip historically uses a special header path in virtual environments.
    # 2. If the distribution name is not known, distutils uses 'UNKNOWN'. We
    #    only do the same when not running in a virtual environment because
    #    pip's historical header path logic (see point 1) did not do this.
    if running_under_virtualenv():
        if user:
            base = variables.get("userbase", sys.prefix)
        else:
            base = variables.get("base", sys.prefix)
        python_xy = f"python{get_major_minor_version()}"
        paths["include"] = os.path.join(base, "include", "site", python_xy)
    elif not dist_name:
        dist_name = "UNKNOWN"

    scheme = Scheme(
        platlib=paths["platlib"],
        purelib=paths["purelib"],
        headers=os.path.join(paths["include"], dist_name),
        scripts=paths["scripts"],
        data=paths["data"],
    )
    if root is not None:
        converted_keys = {}
        for key in SCHEME_KEYS:
            converted_keys[key] = change_root(root, getattr(scheme, key))
        scheme = Scheme(**converted_keys)
    return scheme


def get_bin_prefix() -> str:
    # Forcing to use /usr/local/bin for standard macOS framework installs.
    if sys.platform[:6] == "darwin" and sys.prefix[:16] == "/System/Library/":
        return "/usr/local/bin"
    return sysconfig.get_paths()["scripts"]


def get_purelib() -> str:
    return sysconfig.get_paths()["purelib"]


def get_platlib() -> str:
    return sysconfig.get_paths()["platlib"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\token.py ===
"""
    pygments.token
    ~~~~~~~~~~~~~~

    Basic token types and the standard tokens.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""


class _TokenType(tuple):
    parent = None

    def split(self):
        buf = []
        node = self
        while node is not None:
            buf.append(node)
            node = node.parent
        buf.reverse()
        return buf

    def __init__(self, *args):
        # no need to call super.__init__
        self.subtypes = set()

    def __contains__(self, val):
        return self is val or (
            type(val) is self.__class__ and
            val[:len(self)] == self
        )

    def __getattr__(self, val):
        if not val or not val[0].isupper():
            return tuple.__getattribute__(self, val)
        new = _TokenType(self + (val,))
        setattr(self, val, new)
        self.subtypes.add(new)
        new.parent = self
        return new

    def __repr__(self):
        return 'Token' + (self and '.' or '') + '.'.join(self)

    def __copy__(self):
        # These instances are supposed to be singletons
        return self

    def __deepcopy__(self, memo):
        # These instances are supposed to be singletons
        return self


Token = _TokenType()

# Special token types
Text = Token.Text
Whitespace = Text.Whitespace
Escape = Token.Escape
Error = Token.Error
# Text that doesn't belong to this lexer (e.g. HTML in PHP)
Other = Token.Other

# Common token types for source code
Keyword = Token.Keyword
Name = Token.Name
Literal = Token.Literal
String = Literal.String
Number = Literal.Number
Punctuation = Token.Punctuation
Operator = Token.Operator
Comment = Token.Comment

# Generic types for non-source code
Generic = Token.Generic

# String and some others are not direct children of Token.
# alias them:
Token.Token = Token
Token.String = String
Token.Number = Number


def is_token_subtype(ttype, other):
    """
    Return True if ``ttype`` is a subtype of ``other``.

    exists for backwards compatibility. use ``ttype in other`` now.
    """
    return ttype in other


def string_to_tokentype(s):
    """
    Convert a string into a token type::

        >>> string_to_token('String.Double')
        Token.Literal.String.Double
        >>> string_to_token('Token.Literal.Number')
        Token.Literal.Number
        >>> string_to_token('')
        Token

    Tokens that are already tokens are returned unchanged:

        >>> string_to_token(String)
        Token.Literal.String
    """
    if isinstance(s, _TokenType):
        return s
    if not s:
        return Token
    node = Token
    for item in s.split('.'):
        node = getattr(node, item)
    return node


# Map standard token types to short names, used in CSS class naming.
# If you add a new item, please be sure to run this file to perform
# a consistency check for duplicate values.
STANDARD_TYPES = {
    Token:                         '',

    Text:                          '',
    Whitespace:                    'w',
    Escape:                        'esc',
    Error:                         'err',
    Other:                         'x',

    Keyword:                       'k',
    Keyword.Constant:              'kc',
    Keyword.Declaration:           'kd',
    Keyword.Namespace:             'kn',
    Keyword.Pseudo:                'kp',
    Keyword.Reserved:              'kr',
    Keyword.Type:                  'kt',

    Name:                          'n',
    Name.Attribute:                'na',
    Name.Builtin:                  'nb',
    Name.Builtin.Pseudo:           'bp',
    Name.Class:                    'nc',
    Name.Constant:                 'no',
    Name.Decorator:                'nd',
    Name.Entity:                   'ni',
    Name.Exception:                'ne',
    Name.Function:                 'nf',
    Name.Function.Magic:           'fm',
    Name.Property:                 'py',
    Name.Label:                    'nl',
    Name.Namespace:                'nn',
    Name.Other:                    'nx',
    Name.Tag:                      'nt',
    Name.Variable:                 'nv',
    Name.Variable.Class:           'vc',
    Name.Variable.Global:          'vg',
    Name.Variable.Instance:        'vi',
    Name.Variable.Magic:           'vm',

    Literal:                       'l',
    Literal.Date:                  'ld',

    String:                        's',
    String.Affix:                  'sa',
    String.Backtick:               'sb',
    String.Char:                   'sc',
    String.Delimiter:              'dl',
    String.Doc:                    'sd',
    String.Double:                 's2',
    String.Escape:                 'se',
    String.Heredoc:                'sh',
    String.Interpol:               'si',
    String.Other:                  'sx',
    String.Regex:                  'sr',
    String.Single:                 's1',
    String.Symbol:                 'ss',

    Number:                        'm',
    Number.Bin:                    'mb',
    Number.Float:                  'mf',
    Number.Hex:                    'mh',
    Number.Integer:                'mi',
    Number.Integer.Long:           'il',
    Number.Oct:                    'mo',

    Operator:                      'o',
    Operator.Word:                 'ow',

    Punctuation:                   'p',
    Punctuation.Marker:            'pm',

    Comment:                       'c',
    Comment.Hashbang:              'ch',
    Comment.Multiline:             'cm',
    Comment.Preproc:               'cp',
    Comment.PreprocFile:           'cpf',
    Comment.Single:                'c1',
    Comment.Special:               'cs',

    Generic:                       'g',
    Generic.Deleted:               'gd',
    Generic.Emph:                  'ge',
    Generic.Error:                 'gr',
    Generic.Heading:               'gh',
    Generic.Inserted:              'gi',
    Generic.Output:                'go',
    Generic.Prompt:                'gp',
    Generic.Strong:                'gs',
    Generic.Subheading:            'gu',
    Generic.EmphStrong:            'ges',
    Generic.Traceback:             'gt',
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\range.py ===
from __future__ import annotations

import collections.abc as cabc
import typing as t
from datetime import datetime

if t.TYPE_CHECKING:
    import typing_extensions as te

T = t.TypeVar("T")


class IfRange:
    """Very simple object that represents the `If-Range` header in parsed
    form.  It will either have neither a etag or date or one of either but
    never both.

    .. versionadded:: 0.7
    """

    def __init__(self, etag: str | None = None, date: datetime | None = None):
        #: The etag parsed and unquoted.  Ranges always operate on strong
        #: etags so the weakness information is not necessary.
        self.etag = etag
        #: The date in parsed format or `None`.
        self.date = date

    def to_header(self) -> str:
        """Converts the object back into an HTTP header."""
        if self.date is not None:
            return http.http_date(self.date)
        if self.etag is not None:
            return http.quote_etag(self.etag)
        return ""

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {str(self)!r}>"


class Range:
    """Represents a ``Range`` header. All methods only support only
    bytes as the unit. Stores a list of ranges if given, but the methods
    only work if only one range is provided.

    :raise ValueError: If the ranges provided are invalid.

    .. versionchanged:: 0.15
        The ranges passed in are validated.

    .. versionadded:: 0.7
    """

    def __init__(
        self, units: str, ranges: cabc.Sequence[tuple[int, int | None]]
    ) -> None:
        #: The units of this range.  Usually "bytes".
        self.units = units
        #: A list of ``(begin, end)`` tuples for the range header provided.
        #: The ranges are non-inclusive.
        self.ranges = ranges

        for start, end in ranges:
            if start is None or (end is not None and (start < 0 or start >= end)):
                raise ValueError(f"{(start, end)} is not a valid range.")

    def range_for_length(self, length: int | None) -> tuple[int, int] | None:
        """If the range is for bytes, the length is not None and there is
        exactly one range and it is satisfiable it returns a ``(start, stop)``
        tuple, otherwise `None`.
        """
        if self.units != "bytes" or length is None or len(self.ranges) != 1:
            return None
        start, end = self.ranges[0]
        if end is None:
            end = length
            if start < 0:
                start += length
        if http.is_byte_range_valid(start, end, length):
            return start, min(end, length)
        return None

    def make_content_range(self, length: int | None) -> ContentRange | None:
        """Creates a :class:`~werkzeug.datastructures.ContentRange` object
        from the current range and given content length.
        """
        rng = self.range_for_length(length)
        if rng is not None:
            return ContentRange(self.units, rng[0], rng[1], length)
        return None

    def to_header(self) -> str:
        """Converts the object back into an HTTP header."""
        ranges = []
        for begin, end in self.ranges:
            if end is None:
                ranges.append(f"{begin}-" if begin >= 0 else str(begin))
            else:
                ranges.append(f"{begin}-{end - 1}")
        return f"{self.units}={','.join(ranges)}"

    def to_content_range_header(self, length: int | None) -> str | None:
        """Converts the object into `Content-Range` HTTP header,
        based on given length
        """
        range = self.range_for_length(length)
        if range is not None:
            return f"{self.units} {range[0]}-{range[1] - 1}/{length}"
        return None

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {str(self)!r}>"


class _CallbackProperty(t.Generic[T]):
    def __set_name__(self, owner: type[ContentRange], name: str) -> None:
        self.attr = f"_{name}"

    @t.overload
    def __get__(self, instance: None, owner: None) -> te.Self: ...
    @t.overload
    def __get__(self, instance: ContentRange, owner: type[ContentRange]) -> T: ...
    def __get__(
        self, instance: ContentRange | None, owner: type[ContentRange] | None
    ) -> te.Self | T:
        if instance is None:
            return self

        return instance.__dict__[self.attr]  # type: ignore[no-any-return]

    def __set__(self, instance: ContentRange, value: T) -> None:
        instance.__dict__[self.attr] = value

        if instance.on_update is not None:
            instance.on_update(instance)


class ContentRange:
    """Represents the content range header.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        units: str | None,
        start: int | None,
        stop: int | None,
        length: int | None = None,
        on_update: cabc.Callable[[ContentRange], None] | None = None,
    ) -> None:
        self.on_update = on_update
        self.set(start, stop, length, units)

    #: The units to use, usually "bytes"
    units: str | None = _CallbackProperty()  # type: ignore[assignment]
    #: The start point of the range or `None`.
    start: int | None = _CallbackProperty()  # type: ignore[assignment]
    #: The stop point of the range (non-inclusive) or `None`.  Can only be
    #: `None` if also start is `None`.
    stop: int | None = _CallbackProperty()  # type: ignore[assignment]
    #: The length of the range or `None`.
    length: int | None = _CallbackProperty()  # type: ignore[assignment]

    def set(
        self,
        start: int | None,
        stop: int | None,
        length: int | None = None,
        units: str | None = "bytes",
    ) -> None:
        """Simple method to update the ranges."""
        assert http.is_byte_range_valid(start, stop, length), "Bad range provided"
        self._units: str | None = units
        self._start: int | None = start
        self._stop: int | None = stop
        self._length: int | None = length
        if self.on_update is not None:
            self.on_update(self)

    def unset(self) -> None:
        """Sets the units to `None` which indicates that the header should
        no longer be used.
        """
        self.set(None, None, units=None)

    def to_header(self) -> str:
        if self._units is None:
            return ""
        if self._length is None:
            length: str | int = "*"
        else:
            length = self._length
        if self._start is None:
            return f"{self._units} */{length}"
        return f"{self._units} {self._start}-{self._stop - 1}/{length}"  # type: ignore[operator]

    def __bool__(self) -> bool:
        return self._units is not None

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {str(self)!r}>"


# circular dependencies
from .. import http

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\calculus\polynomials.py ===
from ..libmp.backend import xrange
from .calculus import defun

#----------------------------------------------------------------------------#
#                                Polynomials                                 #
#----------------------------------------------------------------------------#

# XXX: extra precision
@defun
def polyval(ctx, coeffs, x, derivative=False):
    r"""
    Given coefficients `[c_n, \ldots, c_2, c_1, c_0]` and a number `x`,
    :func:`~mpmath.polyval` evaluates the polynomial

    .. math ::

        P(x) = c_n x^n + \ldots + c_2 x^2 + c_1 x + c_0.

    If *derivative=True* is set, :func:`~mpmath.polyval` simultaneously
    evaluates `P(x)` with the derivative, `P'(x)`, and returns the
    tuple `(P(x), P'(x))`.

        >>> from mpmath import *
        >>> mp.pretty = True
        >>> polyval([3, 0, 2], 0.5)
        2.75
        >>> polyval([3, 0, 2], 0.5, derivative=True)
        (2.75, 3.0)

    The coefficients and the evaluation point may be any combination
    of real or complex numbers.
    """
    if not coeffs:
        return ctx.zero
    p = ctx.convert(coeffs[0])
    q = ctx.zero
    for c in coeffs[1:]:
        if derivative:
            q = p + x*q
        p = c + x*p
    if derivative:
        return p, q
    else:
        return p

@defun
def polyroots(ctx, coeffs, maxsteps=50, cleanup=True, extraprec=10,
        error=False, roots_init=None):
    """
    Computes all roots (real or complex) of a given polynomial.

    The roots are returned as a sorted list, where real roots appear first
    followed by complex conjugate roots as adjacent elements. The polynomial
    should be given as a list of coefficients, in the format used by
    :func:`~mpmath.polyval`. The leading coefficient must be nonzero.

    With *error=True*, :func:`~mpmath.polyroots` returns a tuple *(roots, err)*
    where *err* is an estimate of the maximum error among the computed roots.

    **Examples**

    Finding the three real roots of `x^3 - x^2 - 14x + 24`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> nprint(polyroots([1,-1,-14,24]), 4)
        [-4.0, 2.0, 3.0]

    Finding the two complex conjugate roots of `4x^2 + 3x + 2`, with an
    error estimate::

        >>> roots, err = polyroots([4,3,2], error=True)
        >>> for r in roots:
        ...     print(r)
        ...
        (-0.375 + 0.59947894041409j)
        (-0.375 - 0.59947894041409j)
        >>>
        >>> err
        2.22044604925031e-16
        >>>
        >>> polyval([4,3,2], roots[0])
        (2.22044604925031e-16 + 0.0j)
        >>> polyval([4,3,2], roots[1])
        (2.22044604925031e-16 + 0.0j)

    The following example computes all the 5th roots of unity; that is,
    the roots of `x^5 - 1`::

        >>> mp.dps = 20
        >>> for r in polyroots([1, 0, 0, 0, 0, -1]):
        ...     print(r)
        ...
        1.0
        (-0.8090169943749474241 + 0.58778525229247312917j)
        (-0.8090169943749474241 - 0.58778525229247312917j)
        (0.3090169943749474241 + 0.95105651629515357212j)
        (0.3090169943749474241 - 0.95105651629515357212j)

    **Precision and conditioning**

    The roots are computed to the current working precision accuracy. If this
    accuracy cannot be achieved in ``maxsteps`` steps, then a
    ``NoConvergence`` exception is raised. The algorithm internally is using
    the current working precision extended by ``extraprec``. If
    ``NoConvergence`` was raised, that is caused either by not having enough
    extra precision to achieve convergence (in which case increasing
    ``extraprec`` should fix the problem) or too low ``maxsteps`` (in which
    case increasing ``maxsteps`` should fix the problem), or a combination of
    both.

    The user should always do a convergence study with regards to
    ``extraprec`` to ensure accurate results. It is possible to get
    convergence to a wrong answer with too low ``extraprec``.

    Provided there are no repeated roots, :func:`~mpmath.polyroots` can
    typically compute all roots of an arbitrary polynomial to high precision::

        >>> mp.dps = 60
        >>> for r in polyroots([1, 0, -10, 0, 1]):
        ...     print(r)
        ...
        -3.14626436994197234232913506571557044551247712918732870123249
        -0.317837245195782244725757617296174288373133378433432554879127
        0.317837245195782244725757617296174288373133378433432554879127
        3.14626436994197234232913506571557044551247712918732870123249
        >>>
        >>> sqrt(3) + sqrt(2)
        3.14626436994197234232913506571557044551247712918732870123249
        >>> sqrt(3) - sqrt(2)
        0.317837245195782244725757617296174288373133378433432554879127

    **Algorithm**

    :func:`~mpmath.polyroots` implements the Durand-Kerner method [1], which
    uses complex arithmetic to locate all roots simultaneously.
    The Durand-Kerner method can be viewed as approximately performing
    simultaneous Newton iteration for all the roots. In particular,
    the convergence to simple roots is quadratic, just like Newton's
    method.

    Although all roots are internally calculated using complex arithmetic, any
    root found to have an imaginary part smaller than the estimated numerical
    error is truncated to a real number (small real parts are also chopped).
    Real roots are placed first in the returned list, sorted by value. The
    remaining complex roots are sorted by their real parts so that conjugate
    roots end up next to each other.

    **References**

    1. http://en.wikipedia.org/wiki/Durand-Kerner_method

    """
    if len(coeffs) <= 1:
        if not coeffs or not coeffs[0]:
            raise ValueError("Input to polyroots must not be the zero polynomial")
        # Constant polynomial with no roots
        return []

    orig = ctx.prec
    tol = +ctx.eps
    with ctx.extraprec(extraprec):
        deg = len(coeffs) - 1
        # Must be monic
        lead = ctx.convert(coeffs[0])
        if lead == 1:
            coeffs = [ctx.convert(c) for c in coeffs]
        else:
            coeffs = [c/lead for c in coeffs]
        f = lambda x: ctx.polyval(coeffs, x)
        if roots_init is None:
            roots = [ctx.mpc((0.4+0.9j)**n) for n in xrange(deg)]
        else:
            roots = [None]*deg;
            deg_init = min(deg, len(roots_init))
            roots[:deg_init] = list(roots_init[:deg_init])
            roots[deg_init:] = [ctx.mpc((0.4+0.9j)**n) for n
                                in xrange(deg_init,deg)]
        err = [ctx.one for n in xrange(deg)]
        # Durand-Kerner iteration until convergence
        for step in xrange(maxsteps):
            if abs(max(err)) < tol:
                break
            for i in xrange(deg):
                p = roots[i]
                x = f(p)
                for j in range(deg):
                    if i != j:
                        try:
                            x /= (p-roots[j])
                        except ZeroDivisionError:
                            continue
                roots[i] = p - x
                err[i] = abs(x)
        if abs(max(err)) >= tol:
            raise ctx.NoConvergence("Didn't converge in maxsteps=%d steps." \
                    % maxsteps)
        # Remove small real or imaginary parts
        if cleanup:
            for i in xrange(deg):
                if abs(roots[i]) < tol:
                    roots[i] = ctx.zero
                elif abs(ctx._im(roots[i])) < tol:
                    roots[i] = roots[i].real
                elif abs(ctx._re(roots[i])) < tol:
                    roots[i] = roots[i].imag * 1j
        roots.sort(key=lambda x: (abs(ctx._im(x)), ctx._re(x)))
    if error:
        err = max(err)
        err = max(err, ctx.ldexp(1, -orig+1))
        return [+r for r in roots], +err
    else:
        return [+r for r in roots]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\random\__init__.py ===
"""
========================
Random Number Generation
========================

Use ``default_rng()`` to create a `Generator` and call its methods.

=============== =========================================================
Generator
--------------- ---------------------------------------------------------
Generator       Class implementing all of the random number distributions
default_rng     Default constructor for ``Generator``
=============== =========================================================

============================================= ===
BitGenerator Streams that work with Generator
--------------------------------------------- ---
MT19937
PCG64
PCG64DXSM
Philox
SFC64
============================================= ===

============================================= ===
Getting entropy to initialize a BitGenerator
--------------------------------------------- ---
SeedSequence
============================================= ===


Legacy
------

For backwards compatibility with previous versions of numpy before 1.17, the
various aliases to the global `RandomState` methods are left alone and do not
use the new `Generator` API.

==================== =========================================================
Utility functions
-------------------- ---------------------------------------------------------
random               Uniformly distributed floats over ``[0, 1)``
bytes                Uniformly distributed random bytes.
permutation          Randomly permute a sequence / generate a random sequence.
shuffle              Randomly permute a sequence in place.
choice               Random sample from 1-D array.
==================== =========================================================

==================== =========================================================
Compatibility
functions - removed
in the new API
-------------------- ---------------------------------------------------------
rand                 Uniformly distributed values.
randn                Normally distributed values.
ranf                 Uniformly distributed floating point numbers.
random_integers      Uniformly distributed integers in a given range.
                     (deprecated, use ``integers(..., closed=True)`` instead)
random_sample        Alias for `random_sample`
randint              Uniformly distributed integers in a given range
seed                 Seed the legacy random number generator.
==================== =========================================================

==================== =========================================================
Univariate
distributions
-------------------- ---------------------------------------------------------
beta                 Beta distribution over ``[0, 1]``.
binomial             Binomial distribution.
chisquare            :math:`\\chi^2` distribution.
exponential          Exponential distribution.
f                    F (Fisher-Snedecor) distribution.
gamma                Gamma distribution.
geometric            Geometric distribution.
gumbel               Gumbel distribution.
hypergeometric       Hypergeometric distribution.
laplace              Laplace distribution.
logistic             Logistic distribution.
lognormal            Log-normal distribution.
logseries            Logarithmic series distribution.
negative_binomial    Negative binomial distribution.
noncentral_chisquare Non-central chi-square distribution.
noncentral_f         Non-central F distribution.
normal               Normal / Gaussian distribution.
pareto               Pareto distribution.
poisson              Poisson distribution.
power                Power distribution.
rayleigh             Rayleigh distribution.
triangular           Triangular distribution.
uniform              Uniform distribution.
vonmises             Von Mises circular distribution.
wald                 Wald (inverse Gaussian) distribution.
weibull              Weibull distribution.
zipf                 Zipf's distribution over ranked data.
==================== =========================================================

==================== ==========================================================
Multivariate
distributions
-------------------- ----------------------------------------------------------
dirichlet            Multivariate generalization of Beta distribution.
multinomial          Multivariate generalization of the binomial distribution.
multivariate_normal  Multivariate generalization of the normal distribution.
==================== ==========================================================

==================== =========================================================
Standard
distributions
-------------------- ---------------------------------------------------------
standard_cauchy      Standard Cauchy-Lorentz distribution.
standard_exponential Standard exponential distribution.
standard_gamma       Standard Gamma distribution.
standard_normal      Standard normal distribution.
standard_t           Standard Student's t-distribution.
==================== =========================================================

==================== =========================================================
Internal functions
-------------------- ---------------------------------------------------------
get_state            Get tuple representing internal state of generator.
set_state            Set state of generator.
==================== =========================================================


"""
__all__ = [
    'beta',
    'binomial',
    'bytes',
    'chisquare',
    'choice',
    'dirichlet',
    'exponential',
    'f',
    'gamma',
    'geometric',
    'get_state',
    'gumbel',
    'hypergeometric',
    'laplace',
    'logistic',
    'lognormal',
    'logseries',
    'multinomial',
    'multivariate_normal',
    'negative_binomial',
    'noncentral_chisquare',
    'noncentral_f',
    'normal',
    'pareto',
    'permutation',
    'poisson',
    'power',
    'rand',
    'randint',
    'randn',
    'random',
    'random_integers',
    'random_sample',
    'ranf',
    'rayleigh',
    'sample',
    'seed',
    'set_state',
    'shuffle',
    'standard_cauchy',
    'standard_exponential',
    'standard_gamma',
    'standard_normal',
    'standard_t',
    'triangular',
    'uniform',
    'vonmises',
    'wald',
    'weibull',
    'zipf',
]

# add these for module-freeze analysis (like PyInstaller)
from . import _bounded_integers, _common, _pickle
from ._generator import Generator, default_rng
from ._mt19937 import MT19937
from ._pcg64 import PCG64, PCG64DXSM
from ._philox import Philox
from ._sfc64 import SFC64
from .bit_generator import BitGenerator, SeedSequence
from .mtrand import *

__all__ += ['Generator', 'RandomState', 'SeedSequence', 'MT19937',
            'Philox', 'PCG64', 'PCG64DXSM', 'SFC64', 'default_rng',
            'BitGenerator']


def __RandomState_ctor():
    """Return a RandomState instance.

    This function exists solely to assist (un)pickling.

    Note that the state of the RandomState returned here is irrelevant, as this
    function's entire purpose is to return a newly allocated RandomState whose
    state pickle can set.  Consequently the RandomState returned by this function
    is a freshly allocated copy with a seed=0.

    See https://github.com/numpy/numpy/issues/4763 for a detailed discussion

    """
    return RandomState(seed=0)


from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_char_codes.py ===
from typing import Literal

_BoolCodes = Literal[
    "bool", "bool_",
    "?", "|?", "=?", "<?", ">?",
    "b1", "|b1", "=b1", "<b1", ">b1",
]  # fmt: skip

_UInt8Codes = Literal["uint8", "u1", "|u1", "=u1", "<u1", ">u1"]
_UInt16Codes = Literal["uint16", "u2", "|u2", "=u2", "<u2", ">u2"]
_UInt32Codes = Literal["uint32", "u4", "|u4", "=u4", "<u4", ">u4"]
_UInt64Codes = Literal["uint64", "u8", "|u8", "=u8", "<u8", ">u8"]

_Int8Codes = Literal["int8", "i1", "|i1", "=i1", "<i1", ">i1"]
_Int16Codes = Literal["int16", "i2", "|i2", "=i2", "<i2", ">i2"]
_Int32Codes = Literal["int32", "i4", "|i4", "=i4", "<i4", ">i4"]
_Int64Codes = Literal["int64", "i8", "|i8", "=i8", "<i8", ">i8"]

_Float16Codes = Literal["float16", "f2", "|f2", "=f2", "<f2", ">f2"]
_Float32Codes = Literal["float32", "f4", "|f4", "=f4", "<f4", ">f4"]
_Float64Codes = Literal["float64", "f8", "|f8", "=f8", "<f8", ">f8"]

_Complex64Codes = Literal["complex64", "c8", "|c8", "=c8", "<c8", ">c8"]
_Complex128Codes = Literal["complex128", "c16", "|c16", "=c16", "<c16", ">c16"]

_ByteCodes = Literal["byte", "b", "|b", "=b", "<b", ">b"]
_ShortCodes = Literal["short", "h", "|h", "=h", "<h", ">h"]
_IntCCodes = Literal["intc", "i", "|i", "=i", "<i", ">i"]
_IntPCodes = Literal["intp", "int", "int_", "n", "|n", "=n", "<n", ">n"]
_LongCodes = Literal["long", "l", "|l", "=l", "<l", ">l"]
_IntCodes = _IntPCodes
_LongLongCodes = Literal["longlong", "q", "|q", "=q", "<q", ">q"]

_UByteCodes = Literal["ubyte", "B", "|B", "=B", "<B", ">B"]
_UShortCodes = Literal["ushort", "H", "|H", "=H", "<H", ">H"]
_UIntCCodes = Literal["uintc", "I", "|I", "=I", "<I", ">I"]
_UIntPCodes = Literal["uintp", "uint", "N", "|N", "=N", "<N", ">N"]
_ULongCodes = Literal["ulong", "L", "|L", "=L", "<L", ">L"]
_UIntCodes = _UIntPCodes
_ULongLongCodes = Literal["ulonglong", "Q", "|Q", "=Q", "<Q", ">Q"]

_HalfCodes = Literal["half", "e", "|e", "=e", "<e", ">e"]
_SingleCodes = Literal["single", "f", "|f", "=f", "<f", ">f"]
_DoubleCodes = Literal["double", "float", "d", "|d", "=d", "<d", ">d"]
_LongDoubleCodes = Literal["longdouble", "g", "|g", "=g", "<g", ">g"]

_CSingleCodes = Literal["csingle", "F", "|F", "=F", "<F", ">F"]
_CDoubleCodes = Literal["cdouble", "complex", "D", "|D", "=D", "<D", ">D"]
_CLongDoubleCodes = Literal["clongdouble", "G", "|G", "=G", "<G", ">G"]

_StrCodes = Literal["str", "str_", "unicode", "U", "|U", "=U", "<U", ">U"]
_BytesCodes = Literal["bytes", "bytes_", "S", "|S", "=S", "<S", ">S"]
_VoidCodes = Literal["void", "V", "|V", "=V", "<V", ">V"]
_ObjectCodes = Literal["object", "object_", "O", "|O", "=O", "<O", ">O"]

_DT64Codes = Literal[
    "datetime64", "|datetime64", "=datetime64",
    "<datetime64", ">datetime64",
    "datetime64[Y]", "|datetime64[Y]", "=datetime64[Y]",
    "<datetime64[Y]", ">datetime64[Y]",
    "datetime64[M]", "|datetime64[M]", "=datetime64[M]",
    "<datetime64[M]", ">datetime64[M]",
    "datetime64[W]", "|datetime64[W]", "=datetime64[W]",
    "<datetime64[W]", ">datetime64[W]",
    "datetime64[D]", "|datetime64[D]", "=datetime64[D]",
    "<datetime64[D]", ">datetime64[D]",
    "datetime64[h]", "|datetime64[h]", "=datetime64[h]",
    "<datetime64[h]", ">datetime64[h]",
    "datetime64[m]", "|datetime64[m]", "=datetime64[m]",
    "<datetime64[m]", ">datetime64[m]",
    "datetime64[s]", "|datetime64[s]", "=datetime64[s]",
    "<datetime64[s]", ">datetime64[s]",
    "datetime64[ms]", "|datetime64[ms]", "=datetime64[ms]",
    "<datetime64[ms]", ">datetime64[ms]",
    "datetime64[us]", "|datetime64[us]", "=datetime64[us]",
    "<datetime64[us]", ">datetime64[us]",
    "datetime64[ns]", "|datetime64[ns]", "=datetime64[ns]",
    "<datetime64[ns]", ">datetime64[ns]",
    "datetime64[ps]", "|datetime64[ps]", "=datetime64[ps]",
    "<datetime64[ps]", ">datetime64[ps]",
    "datetime64[fs]", "|datetime64[fs]", "=datetime64[fs]",
    "<datetime64[fs]", ">datetime64[fs]",
    "datetime64[as]", "|datetime64[as]", "=datetime64[as]",
    "<datetime64[as]", ">datetime64[as]",
    "M", "|M", "=M", "<M", ">M",
    "M8", "|M8", "=M8", "<M8", ">M8",
    "M8[Y]", "|M8[Y]", "=M8[Y]", "<M8[Y]", ">M8[Y]",
    "M8[M]", "|M8[M]", "=M8[M]", "<M8[M]", ">M8[M]",
    "M8[W]", "|M8[W]", "=M8[W]", "<M8[W]", ">M8[W]",
    "M8[D]", "|M8[D]", "=M8[D]", "<M8[D]", ">M8[D]",
    "M8[h]", "|M8[h]", "=M8[h]", "<M8[h]", ">M8[h]",
    "M8[m]", "|M8[m]", "=M8[m]", "<M8[m]", ">M8[m]",
    "M8[s]", "|M8[s]", "=M8[s]", "<M8[s]", ">M8[s]",
    "M8[ms]", "|M8[ms]", "=M8[ms]", "<M8[ms]", ">M8[ms]",
    "M8[us]", "|M8[us]", "=M8[us]", "<M8[us]", ">M8[us]",
    "M8[ns]", "|M8[ns]", "=M8[ns]", "<M8[ns]", ">M8[ns]",
    "M8[ps]", "|M8[ps]", "=M8[ps]", "<M8[ps]", ">M8[ps]",
    "M8[fs]", "|M8[fs]", "=M8[fs]", "<M8[fs]", ">M8[fs]",
    "M8[as]", "|M8[as]", "=M8[as]", "<M8[as]", ">M8[as]",
]
_TD64Codes = Literal[
    "timedelta64", "|timedelta64", "=timedelta64",
    "<timedelta64", ">timedelta64",
    "timedelta64[Y]", "|timedelta64[Y]", "=timedelta64[Y]",
    "<timedelta64[Y]", ">timedelta64[Y]",
    "timedelta64[M]", "|timedelta64[M]", "=timedelta64[M]",
    "<timedelta64[M]", ">timedelta64[M]",
    "timedelta64[W]", "|timedelta64[W]", "=timedelta64[W]",
    "<timedelta64[W]", ">timedelta64[W]",
    "timedelta64[D]", "|timedelta64[D]", "=timedelta64[D]",
    "<timedelta64[D]", ">timedelta64[D]",
    "timedelta64[h]", "|timedelta64[h]", "=timedelta64[h]",
    "<timedelta64[h]", ">timedelta64[h]",
    "timedelta64[m]", "|timedelta64[m]", "=timedelta64[m]",
    "<timedelta64[m]", ">timedelta64[m]",
    "timedelta64[s]", "|timedelta64[s]", "=timedelta64[s]",
    "<timedelta64[s]", ">timedelta64[s]",
    "timedelta64[ms]", "|timedelta64[ms]", "=timedelta64[ms]",
    "<timedelta64[ms]", ">timedelta64[ms]",
    "timedelta64[us]", "|timedelta64[us]", "=timedelta64[us]",
    "<timedelta64[us]", ">timedelta64[us]",
    "timedelta64[ns]", "|timedelta64[ns]", "=timedelta64[ns]",
    "<timedelta64[ns]", ">timedelta64[ns]",
    "timedelta64[ps]", "|timedelta64[ps]", "=timedelta64[ps]",
    "<timedelta64[ps]", ">timedelta64[ps]",
    "timedelta64[fs]", "|timedelta64[fs]", "=timedelta64[fs]",
    "<timedelta64[fs]", ">timedelta64[fs]",
    "timedelta64[as]", "|timedelta64[as]", "=timedelta64[as]",
    "<timedelta64[as]", ">timedelta64[as]",
    "m", "|m", "=m", "<m", ">m",
    "m8", "|m8", "=m8", "<m8", ">m8",
    "m8[Y]", "|m8[Y]", "=m8[Y]", "<m8[Y]", ">m8[Y]",
    "m8[M]", "|m8[M]", "=m8[M]", "<m8[M]", ">m8[M]",
    "m8[W]", "|m8[W]", "=m8[W]", "<m8[W]", ">m8[W]",
    "m8[D]", "|m8[D]", "=m8[D]", "<m8[D]", ">m8[D]",
    "m8[h]", "|m8[h]", "=m8[h]", "<m8[h]", ">m8[h]",
    "m8[m]", "|m8[m]", "=m8[m]", "<m8[m]", ">m8[m]",
    "m8[s]", "|m8[s]", "=m8[s]", "<m8[s]", ">m8[s]",
    "m8[ms]", "|m8[ms]", "=m8[ms]", "<m8[ms]", ">m8[ms]",
    "m8[us]", "|m8[us]", "=m8[us]", "<m8[us]", ">m8[us]",
    "m8[ns]", "|m8[ns]", "=m8[ns]", "<m8[ns]", ">m8[ns]",
    "m8[ps]", "|m8[ps]", "=m8[ps]", "<m8[ps]", ">m8[ps]",
    "m8[fs]", "|m8[fs]", "=m8[fs]", "<m8[fs]", ">m8[fs]",
    "m8[as]", "|m8[as]", "=m8[as]", "<m8[as]", ">m8[as]",
]

# NOTE: `StringDType' has no scalar type, and therefore has no name that can
# be passed to the `dtype` constructor
_StringCodes = Literal["T", "|T", "=T", "<T", ">T"]

# NOTE: Nested literals get flattened and de-duplicated at runtime, which isn't
# the case for a `Union` of `Literal`s.
# So even though they're equivalent when type-checking, they differ at runtime.
# Another advantage of nesting, is that they always have a "flat"
# `Literal.__args__`, which is a tuple of *literally* all its literal values.

_UnsignedIntegerCodes = Literal[
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _UIntCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _ULongCodes,
    _ULongLongCodes,
]
_SignedIntegerCodes = Literal[
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _IntCodes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _LongCodes,
    _LongLongCodes,
]
_FloatingCodes = Literal[
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes
]
_ComplexFloatingCodes = Literal[
    _Complex64Codes,
    _Complex128Codes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
]
_IntegerCodes = Literal[_UnsignedIntegerCodes, _SignedIntegerCodes]
_InexactCodes = Literal[_FloatingCodes, _ComplexFloatingCodes]
_NumberCodes = Literal[_IntegerCodes, _InexactCodes]

_CharacterCodes = Literal[_StrCodes, _BytesCodes]
_FlexibleCodes = Literal[_VoidCodes, _CharacterCodes]

_GenericCodes = Literal[
    _BoolCodes,
    _NumberCodes,
    _FlexibleCodes,
    _DT64Codes,
    _TD64Codes,
    _ObjectCodes,
    # TODO: add `_StringCodes` once it has a scalar type
    # _StringCodes,
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\computation\align.py ===
"""
Core eval alignment algorithms.
"""
from __future__ import annotations

from functools import (
    partial,
    wraps,
)
from typing import (
    TYPE_CHECKING,
    Callable,
)
import warnings

import numpy as np

from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation.common import result_type_many

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import F

    from pandas.core.generic import NDFrame
    from pandas.core.indexes.api import Index


def _align_core_single_unary_op(
    term,
) -> tuple[partial | type[NDFrame], dict[str, Index] | None]:
    typ: partial | type[NDFrame]
    axes: dict[str, Index] | None = None

    if isinstance(term.value, np.ndarray):
        typ = partial(np.asanyarray, dtype=term.value.dtype)
    else:
        typ = type(term.value)
        if hasattr(term.value, "axes"):
            axes = _zip_axes_from_type(typ, term.value.axes)

    return typ, axes


def _zip_axes_from_type(
    typ: type[NDFrame], new_axes: Sequence[Index]
) -> dict[str, Index]:
    return {name: new_axes[i] for i, name in enumerate(typ._AXIS_ORDERS)}


def _any_pandas_objects(terms) -> bool:
    """
    Check a sequence of terms for instances of PandasObject.
    """
    return any(isinstance(term.value, PandasObject) for term in terms)


def _filter_special_cases(f) -> Callable[[F], F]:
    @wraps(f)
    def wrapper(terms):
        # single unary operand
        if len(terms) == 1:
            return _align_core_single_unary_op(terms[0])

        term_values = (term.value for term in terms)

        # we don't have any pandas objects
        if not _any_pandas_objects(terms):
            return result_type_many(*term_values), None

        return f(terms)

    return wrapper


@_filter_special_cases
def _align_core(terms):
    term_index = [i for i, term in enumerate(terms) if hasattr(term.value, "axes")]
    term_dims = [terms[i].value.ndim for i in term_index]

    from pandas import Series

    ndims = Series(dict(zip(term_index, term_dims)))

    # initial axes are the axes of the largest-axis'd term
    biggest = terms[ndims.idxmax()].value
    typ = biggest._constructor
    axes = biggest.axes
    naxes = len(axes)
    gt_than_one_axis = naxes > 1

    for value in (terms[i].value for i in term_index):
        is_series = isinstance(value, ABCSeries)
        is_series_and_gt_one_axis = is_series and gt_than_one_axis

        for axis, items in enumerate(value.axes):
            if is_series_and_gt_one_axis:
                ax, itm = naxes - 1, value.index
            else:
                ax, itm = axis, items

            if not axes[ax].is_(itm):
                axes[ax] = axes[ax].union(itm)

    for i, ndim in ndims.items():
        for axis, items in zip(range(ndim), axes):
            ti = terms[i].value

            if hasattr(ti, "reindex"):
                transpose = isinstance(ti, ABCSeries) and naxes > 1
                reindexer = axes[naxes - 1] if transpose else items

                term_axis_size = len(ti.axes[axis])
                reindexer_size = len(reindexer)

                ordm = np.log10(max(1, abs(reindexer_size - term_axis_size)))
                if ordm >= 1 and reindexer_size >= 10000:
                    w = (
                        f"Alignment difference on axis {axis} is larger "
                        f"than an order of magnitude on term {repr(terms[i].name)}, "
                        f"by more than {ordm:.4g}; performance may suffer."
                    )
                    warnings.warn(
                        w, category=PerformanceWarning, stacklevel=find_stack_level()
                    )

                obj = ti.reindex(reindexer, axis=axis, copy=False)
                terms[i].update(obj)

        terms[i].update(terms[i].value.values)

    return typ, _zip_axes_from_type(typ, axes)


def align_terms(terms):
    """
    Align a set of terms.
    """
    try:
        # flatten the parse tree (a nested list, really)
        terms = list(com.flatten(terms))
    except TypeError:
        # can't iterate so it must just be a constant or single variable
        if isinstance(terms.value, (ABCSeries, ABCDataFrame)):
            typ = type(terms.value)
            return typ, _zip_axes_from_type(typ, terms.value.axes)
        return np.result_type(terms.type), None

    # if all resolved variables are numeric scalars
    if all(term.is_scalar for term in terms):
        return result_type_many(*(term.value for term in terms)).type, None

    # perform the main alignment
    typ, axes = _align_core(terms)
    return typ, axes


def reconstruct_object(typ, obj, axes, dtype):
    """
    Reconstruct an object given its type, raw value, and possibly empty
    (None) axes.

    Parameters
    ----------
    typ : object
        A type
    obj : object
        The value to use in the type constructor
    axes : dict
        The axes to use to construct the resulting pandas object

    Returns
    -------
    ret : typ
        An object of type ``typ`` with the value `obj` and possible axes
        `axes`.
    """
    try:
        typ = typ.type
    except AttributeError:
        pass

    res_t = np.result_type(obj.dtype, dtype)

    if not isinstance(typ, partial) and issubclass(typ, PandasObject):
        return typ(obj, dtype=res_t, **axes)

    # special case for pathological things like ~True/~False
    if hasattr(res_t, "type") and typ == np.bool_ and res_t != np.bool_:
        ret_value = res_t.type(obj)
    else:
        ret_value = typ(obj).astype(res_t)
        # The condition is to distinguish 0-dim array (returned in case of
        # scalar) and 1 element array
        # e.g. np.array(0) and np.array([0])
        if (
            len(obj.shape) == 1
            and len(obj) == 1
            and not isinstance(ret_value, np.ndarray)
        ):
            ret_value = np.array([ret_value]).astype(res_t)

    return ret_value

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\keysyms\ironpython_keysyms.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#       Copyright (C) 2003-2006 Gary Bishop.
#       Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#       Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************


import System

from .common import KeyPress, make_KeyPress_from_keydescr, validkey

c32 = System.ConsoleKey
Shift = System.ConsoleModifiers.Shift
Control = System.ConsoleModifiers.Control
Alt = System.ConsoleModifiers.Alt
# table for translating virtual keys to X windows key symbols
code2sym_map = {
    # c32.CANCEL:    'Cancel',
    c32.Backspace: "BackSpace",
    c32.Tab: "Tab",
    c32.Clear: "Clear",
    c32.Enter: "Return",
    # c32.Shift:     'Shift_L',
    # c32.Control:   'Control_L',
    # c32.Menu:      'Alt_L',
    c32.Pause: "Pause",
    # c32.Capital:   'Caps_Lock',
    c32.Escape: "Escape",
    # c32.Space:     'space',
    c32.PageUp: "Prior",
    c32.PageDown: "Next",
    c32.End: "End",
    c32.Home: "Home",
    c32.LeftArrow: "Left",
    c32.UpArrow: "Up",
    c32.RightArrow: "Right",
    c32.DownArrow: "Down",
    c32.Select: "Select",
    c32.Print: "Print",
    c32.Execute: "Execute",
    # c32.Snapshot:  'Snapshot',
    c32.Insert: "Insert",
    c32.Delete: "Delete",
    c32.Help: "Help",
    c32.F1: "F1",
    c32.F2: "F2",
    c32.F3: "F3",
    c32.F4: "F4",
    c32.F5: "F5",
    c32.F6: "F6",
    c32.F7: "F7",
    c32.F8: "F8",
    c32.F9: "F9",
    c32.F10: "F10",
    c32.F11: "F11",
    c32.F12: "F12",
    c32.F13: "F13",
    c32.F14: "F14",
    c32.F15: "F15",
    c32.F16: "F16",
    c32.F17: "F17",
    c32.F18: "F18",
    c32.F19: "F19",
    c32.F20: "F20",
    c32.F21: "F21",
    c32.F22: "F22",
    c32.F23: "F23",
    c32.F24: "F24",
    # c32.Numlock:    'Num_Lock,',
    # c32.Scroll:     'Scroll_Lock',
    # c32.Apps:       'VK_APPS',
    # c32.ProcesskeY: 'VK_PROCESSKEY',
    # c32.Attn:       'VK_ATTN',
    # c32.Crsel:      'VK_CRSEL',
    # c32.Exsel:      'VK_EXSEL',
    # c32.Ereof:      'VK_EREOF',
    # c32.Play:       'VK_PLAY',
    # c32.Zoom:       'VK_ZOOM',
    # c32.Noname:     'VK_NONAME',
    # c32.Pa1:        'VK_PA1',
    c32.OemClear: "VK_OEM_CLEAR",
    c32.NumPad0: "NUMPAD0",
    c32.NumPad1: "NUMPAD1",
    c32.NumPad2: "NUMPAD2",
    c32.NumPad3: "NUMPAD3",
    c32.NumPad4: "NUMPAD4",
    c32.NumPad5: "NUMPAD5",
    c32.NumPad6: "NUMPAD6",
    c32.NumPad7: "NUMPAD7",
    c32.NumPad8: "NUMPAD8",
    c32.NumPad9: "NUMPAD9",
    c32.Divide: "Divide",
    c32.Multiply: "Multiply",
    c32.Add: "Add",
    c32.Subtract: "Subtract",
    c32.Decimal: "VK_DECIMAL",
}

# function to handle the mapping


def make_keysym(keycode):
    try:
        sym = code2sym_map[keycode]
    except KeyError:
        sym = ""
    return sym


sym2code_map = {}
for code, sym in code2sym_map.items():
    sym2code_map[sym.lower()] = code


def key_text_to_keyinfo(keytext):
    """Convert a GNU readline style textual description of a key to keycode with modifiers"""
    if keytext.startswith('"'):  # "
        return keyseq_to_keyinfo(keytext[1:-1])
    else:
        return keyname_to_keyinfo(keytext)


def char_to_keyinfo(char, control=False, meta=False, shift=False):
    vk = ord(char)
    if vk & 0xFFFF == 0xFFFF:
        print('VkKeyScan("%s") = %x' % (char, vk))
        raise ValueError("bad key")
    if vk & 0x100:
        shift = True
    if vk & 0x200:
        control = True
    if vk & 0x400:
        meta = True
    return (control, meta, shift, vk & 0xFF)


def keyname_to_keyinfo(keyname):
    control = False
    meta = False
    shift = False

    while True:
        lkeyname = keyname.lower()
        if lkeyname.startswith("control-"):
            control = True
            keyname = keyname[8:]
        elif lkeyname.startswith("ctrl-"):
            control = True
            keyname = keyname[5:]
        elif lkeyname.startswith("meta-"):
            meta = True
            keyname = keyname[5:]
        elif lkeyname.startswith("alt-"):
            meta = True
            keyname = keyname[4:]
        elif lkeyname.startswith("shift-"):
            shift = True
            keyname = keyname[6:]
        else:
            if len(keyname) > 1:
                return (control, meta, shift, sym2code_map.get(keyname.lower(), " "))
            else:
                return char_to_keyinfo(keyname, control, meta, shift)


def keyseq_to_keyinfo(keyseq):
    res = []
    control = False
    meta = False
    shift = False

    while True:
        if keyseq.startswith("\\C-"):
            control = True
            keyseq = keyseq[3:]
        elif keyseq.startswith("\\M-"):
            meta = True
            keyseq = keyseq[3:]
        elif keyseq.startswith("\\e"):
            res.append(char_to_keyinfo("\033", control, meta, shift))
            control = meta = shift = False
            keyseq = keyseq[2:]
        elif len(keyseq) >= 1:
            res.append(char_to_keyinfo(keyseq[0], control, meta, shift))
            control = meta = shift = False
            keyseq = keyseq[1:]
        else:
            return res[0]


def make_keyinfo(keycode, state):
    control = False
    meta = False
    shift = False
    return (control, meta, shift, keycode)


def make_KeyPress(char, state, keycode):

    shift = bool(int(state) & int(Shift))
    control = bool(int(state) & int(Control))
    meta = bool(int(state) & int(Alt))
    keyname = code2sym_map.get(keycode, "").lower()
    if control and meta:  # equivalent to altgr so clear flags
        control = False
        meta = False
    elif control:
        char = str(keycode)
    return KeyPress(char, shift, control, meta, keyname)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\background_service.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: BackgroundService (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker


class ServiceName(enum.Enum):
    '''
    The Background Service that will be associated with the commands/events.
    Every Background Service operates independently, but they share the same
    API.
    '''
    BACKGROUND_FETCH = "backgroundFetch"
    BACKGROUND_SYNC = "backgroundSync"
    PUSH_MESSAGING = "pushMessaging"
    NOTIFICATIONS = "notifications"
    PAYMENT_HANDLER = "paymentHandler"
    PERIODIC_BACKGROUND_SYNC = "periodicBackgroundSync"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class EventMetadata:
    '''
    A key-value pair for additional event information to pass along.
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
class BackgroundServiceEvent:
    #: Timestamp of the event (in seconds).
    timestamp: network.TimeSinceEpoch

    #: The origin this event belongs to.
    origin: str

    #: The Service Worker ID that initiated the event.
    service_worker_registration_id: service_worker.RegistrationID

    #: The Background Service this event belongs to.
    service: ServiceName

    #: A description of the event.
    event_name: str

    #: An identifier that groups related events together.
    instance_id: str

    #: A list of event-specific information.
    event_metadata: typing.List[EventMetadata]

    #: Storage key this event belongs to.
    storage_key: str

    def to_json(self):
        json = dict()
        json['timestamp'] = self.timestamp.to_json()
        json['origin'] = self.origin
        json['serviceWorkerRegistrationId'] = self.service_worker_registration_id.to_json()
        json['service'] = self.service.to_json()
        json['eventName'] = self.event_name
        json['instanceId'] = self.instance_id
        json['eventMetadata'] = [i.to_json() for i in self.event_metadata]
        json['storageKey'] = self.storage_key
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            timestamp=network.TimeSinceEpoch.from_json(json['timestamp']),
            origin=str(json['origin']),
            service_worker_registration_id=service_worker.RegistrationID.from_json(json['serviceWorkerRegistrationId']),
            service=ServiceName.from_json(json['service']),
            event_name=str(json['eventName']),
            instance_id=str(json['instanceId']),
            event_metadata=[EventMetadata.from_json(i) for i in json['eventMetadata']],
            storage_key=str(json['storageKey']),
        )


def start_observing(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables event updates for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.startObserving',
        'params': params,
    }
    json = yield cmd_dict


def stop_observing(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables event updates for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.stopObserving',
        'params': params,
    }
    json = yield cmd_dict


def set_recording(
        should_record: bool,
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set the recording state for the service.

    :param should_record:
    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['shouldRecord'] = should_record
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.setRecording',
        'params': params,
    }
    json = yield cmd_dict


def clear_events(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all stored data for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.clearEvents',
        'params': params,
    }
    json = yield cmd_dict


@event_class('BackgroundService.recordingStateChanged')
@dataclass
class RecordingStateChanged:
    '''
    Called when the recording state for the service has been updated.
    '''
    is_recording: bool
    service: ServiceName

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RecordingStateChanged:
        return cls(
            is_recording=bool(json['isRecording']),
            service=ServiceName.from_json(json['service'])
        )


@event_class('BackgroundService.backgroundServiceEventReceived')
@dataclass
class BackgroundServiceEventReceived:
    '''
    Called with all existing backgroundServiceEvents when enabled, and all new
    events afterwards if enabled and recording.
    '''
    background_service_event: BackgroundServiceEvent

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackgroundServiceEventReceived:
        return cls(
            background_service_event=BackgroundServiceEvent.from_json(json['backgroundServiceEvent'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\background_service.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: BackgroundService (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker


class ServiceName(enum.Enum):
    '''
    The Background Service that will be associated with the commands/events.
    Every Background Service operates independently, but they share the same
    API.
    '''
    BACKGROUND_FETCH = "backgroundFetch"
    BACKGROUND_SYNC = "backgroundSync"
    PUSH_MESSAGING = "pushMessaging"
    NOTIFICATIONS = "notifications"
    PAYMENT_HANDLER = "paymentHandler"
    PERIODIC_BACKGROUND_SYNC = "periodicBackgroundSync"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class EventMetadata:
    '''
    A key-value pair for additional event information to pass along.
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
class BackgroundServiceEvent:
    #: Timestamp of the event (in seconds).
    timestamp: network.TimeSinceEpoch

    #: The origin this event belongs to.
    origin: str

    #: The Service Worker ID that initiated the event.
    service_worker_registration_id: service_worker.RegistrationID

    #: The Background Service this event belongs to.
    service: ServiceName

    #: A description of the event.
    event_name: str

    #: An identifier that groups related events together.
    instance_id: str

    #: A list of event-specific information.
    event_metadata: typing.List[EventMetadata]

    #: Storage key this event belongs to.
    storage_key: str

    def to_json(self):
        json = dict()
        json['timestamp'] = self.timestamp.to_json()
        json['origin'] = self.origin
        json['serviceWorkerRegistrationId'] = self.service_worker_registration_id.to_json()
        json['service'] = self.service.to_json()
        json['eventName'] = self.event_name
        json['instanceId'] = self.instance_id
        json['eventMetadata'] = [i.to_json() for i in self.event_metadata]
        json['storageKey'] = self.storage_key
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            timestamp=network.TimeSinceEpoch.from_json(json['timestamp']),
            origin=str(json['origin']),
            service_worker_registration_id=service_worker.RegistrationID.from_json(json['serviceWorkerRegistrationId']),
            service=ServiceName.from_json(json['service']),
            event_name=str(json['eventName']),
            instance_id=str(json['instanceId']),
            event_metadata=[EventMetadata.from_json(i) for i in json['eventMetadata']],
            storage_key=str(json['storageKey']),
        )


def start_observing(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables event updates for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.startObserving',
        'params': params,
    }
    json = yield cmd_dict


def stop_observing(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables event updates for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.stopObserving',
        'params': params,
    }
    json = yield cmd_dict


def set_recording(
        should_record: bool,
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set the recording state for the service.

    :param should_record:
    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['shouldRecord'] = should_record
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.setRecording',
        'params': params,
    }
    json = yield cmd_dict


def clear_events(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all stored data for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.clearEvents',
        'params': params,
    }
    json = yield cmd_dict


@event_class('BackgroundService.recordingStateChanged')
@dataclass
class RecordingStateChanged:
    '''
    Called when the recording state for the service has been updated.
    '''
    is_recording: bool
    service: ServiceName

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RecordingStateChanged:
        return cls(
            is_recording=bool(json['isRecording']),
            service=ServiceName.from_json(json['service'])
        )


@event_class('BackgroundService.backgroundServiceEventReceived')
@dataclass
class BackgroundServiceEventReceived:
    '''
    Called with all existing backgroundServiceEvents when enabled, and all new
    events afterwards if enabled and recording.
    '''
    background_service_event: BackgroundServiceEvent

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackgroundServiceEventReceived:
        return cls(
            background_service_event=BackgroundServiceEvent.from_json(json['backgroundServiceEvent'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\background_service.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: BackgroundService (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
from . import service_worker


class ServiceName(enum.Enum):
    '''
    The Background Service that will be associated with the commands/events.
    Every Background Service operates independently, but they share the same
    API.
    '''
    BACKGROUND_FETCH = "backgroundFetch"
    BACKGROUND_SYNC = "backgroundSync"
    PUSH_MESSAGING = "pushMessaging"
    NOTIFICATIONS = "notifications"
    PAYMENT_HANDLER = "paymentHandler"
    PERIODIC_BACKGROUND_SYNC = "periodicBackgroundSync"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class EventMetadata:
    '''
    A key-value pair for additional event information to pass along.
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
class BackgroundServiceEvent:
    #: Timestamp of the event (in seconds).
    timestamp: network.TimeSinceEpoch

    #: The origin this event belongs to.
    origin: str

    #: The Service Worker ID that initiated the event.
    service_worker_registration_id: service_worker.RegistrationID

    #: The Background Service this event belongs to.
    service: ServiceName

    #: A description of the event.
    event_name: str

    #: An identifier that groups related events together.
    instance_id: str

    #: A list of event-specific information.
    event_metadata: typing.List[EventMetadata]

    #: Storage key this event belongs to.
    storage_key: str

    def to_json(self):
        json = dict()
        json['timestamp'] = self.timestamp.to_json()
        json['origin'] = self.origin
        json['serviceWorkerRegistrationId'] = self.service_worker_registration_id.to_json()
        json['service'] = self.service.to_json()
        json['eventName'] = self.event_name
        json['instanceId'] = self.instance_id
        json['eventMetadata'] = [i.to_json() for i in self.event_metadata]
        json['storageKey'] = self.storage_key
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            timestamp=network.TimeSinceEpoch.from_json(json['timestamp']),
            origin=str(json['origin']),
            service_worker_registration_id=service_worker.RegistrationID.from_json(json['serviceWorkerRegistrationId']),
            service=ServiceName.from_json(json['service']),
            event_name=str(json['eventName']),
            instance_id=str(json['instanceId']),
            event_metadata=[EventMetadata.from_json(i) for i in json['eventMetadata']],
            storage_key=str(json['storageKey']),
        )


def start_observing(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables event updates for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.startObserving',
        'params': params,
    }
    json = yield cmd_dict


def stop_observing(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables event updates for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.stopObserving',
        'params': params,
    }
    json = yield cmd_dict


def set_recording(
        should_record: bool,
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set the recording state for the service.

    :param should_record:
    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['shouldRecord'] = should_record
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.setRecording',
        'params': params,
    }
    json = yield cmd_dict


def clear_events(
        service: ServiceName
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears all stored data for the service.

    :param service:
    '''
    params: T_JSON_DICT = dict()
    params['service'] = service.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'BackgroundService.clearEvents',
        'params': params,
    }
    json = yield cmd_dict


@event_class('BackgroundService.recordingStateChanged')
@dataclass
class RecordingStateChanged:
    '''
    Called when the recording state for the service has been updated.
    '''
    is_recording: bool
    service: ServiceName

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RecordingStateChanged:
        return cls(
            is_recording=bool(json['isRecording']),
            service=ServiceName.from_json(json['service'])
        )


@event_class('BackgroundService.backgroundServiceEventReceived')
@dataclass
class BackgroundServiceEventReceived:
    '''
    Called with all existing backgroundServiceEvents when enabled, and all new
    events afterwards if enabled and recording.
    '''
    background_service_event: BackgroundServiceEvent

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BackgroundServiceEventReceived:
        return cls(
            background_service_event=BackgroundServiceEvent.from_json(json['backgroundServiceEvent'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\connectors\asyncio.py ===
# connectors/asyncio.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

"""generic asyncio-adapted versions of DBAPI connection and cursor"""

from __future__ import annotations

import collections

from ..engine import AdaptedConnection
from ..util.concurrency import asyncio
from ..util.concurrency import await_fallback
from ..util.concurrency import await_only


class AsyncAdapt_dbapi_cursor:
    server_side = False
    __slots__ = (
        "_adapt_connection",
        "_connection",
        "await_",
        "_cursor",
        "_rows",
    )

    def __init__(self, adapt_connection):
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self.await_ = adapt_connection.await_

        cursor = self._connection.cursor()
        self._cursor = self._aenter_cursor(cursor)

        if not self.server_side:
            self._rows = collections.deque()

    def _aenter_cursor(self, cursor):
        return self.await_(cursor.__aenter__())

    @property
    def description(self):
        return self._cursor.description

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def arraysize(self):
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value):
        self._cursor.arraysize = value

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    def close(self):
        # note we aren't actually closing the cursor here,
        # we are just letting GC do it.  see notes in aiomysql dialect
        self._rows.clear()

    def execute(self, operation, parameters=None):
        return self.await_(self._execute_async(operation, parameters))

    def executemany(self, operation, seq_of_parameters):
        return self.await_(
            self._executemany_async(operation, seq_of_parameters)
        )

    async def _execute_async(self, operation, parameters):
        async with self._adapt_connection._execute_mutex:
            result = await self._cursor.execute(operation, parameters or ())

            if self._cursor.description and not self.server_side:
                self._rows = collections.deque(await self._cursor.fetchall())
            return result

    async def _executemany_async(self, operation, seq_of_parameters):
        async with self._adapt_connection._execute_mutex:
            return await self._cursor.executemany(operation, seq_of_parameters)

    def nextset(self):
        self.await_(self._cursor.nextset())
        if self._cursor.description and not self.server_side:
            self._rows = collections.deque(
                self.await_(self._cursor.fetchall())
            )

    def setinputsizes(self, *inputsizes):
        # NOTE: this is overrridden in aioodbc due to
        # see https://github.com/aio-libs/aioodbc/issues/451
        # right now

        return self.await_(self._cursor.setinputsizes(*inputsizes))

    def __iter__(self):
        while self._rows:
            yield self._rows.popleft()

    def fetchone(self):
        if self._rows:
            return self._rows.popleft()
        else:
            return None

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize
        rr = self._rows
        return [rr.popleft() for _ in range(min(size, len(rr)))]

    def fetchall(self):
        retval = list(self._rows)
        self._rows.clear()
        return retval


class AsyncAdapt_dbapi_ss_cursor(AsyncAdapt_dbapi_cursor):
    __slots__ = ()
    server_side = True

    def __init__(self, adapt_connection):
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self.await_ = adapt_connection.await_

        cursor = self._connection.cursor()

        self._cursor = self.await_(cursor.__aenter__())

    def close(self):
        if self._cursor is not None:
            self.await_(self._cursor.close())
            self._cursor = None

    def fetchone(self):
        return self.await_(self._cursor.fetchone())

    def fetchmany(self, size=None):
        return self.await_(self._cursor.fetchmany(size=size))

    def fetchall(self):
        return self.await_(self._cursor.fetchall())

    def __iter__(self):
        iterator = self._cursor.__aiter__()
        while True:
            try:
                yield self.await_(iterator.__anext__())
            except StopAsyncIteration:
                break


class AsyncAdapt_dbapi_connection(AdaptedConnection):
    _cursor_cls = AsyncAdapt_dbapi_cursor
    _ss_cursor_cls = AsyncAdapt_dbapi_ss_cursor

    await_ = staticmethod(await_only)
    __slots__ = ("dbapi", "_execute_mutex")

    def __init__(self, dbapi, connection):
        self.dbapi = dbapi
        self._connection = connection
        self._execute_mutex = asyncio.Lock()

    def ping(self, reconnect):
        return self.await_(self._connection.ping(reconnect))

    def add_output_converter(self, *arg, **kw):
        self._connection.add_output_converter(*arg, **kw)

    def character_set_name(self):
        return self._connection.character_set_name()

    @property
    def autocommit(self):
        return self._connection.autocommit

    @autocommit.setter
    def autocommit(self, value):
        # https://github.com/aio-libs/aioodbc/issues/448
        # self._connection.autocommit = value

        self._connection._conn.autocommit = value

    def cursor(self, server_side=False):
        if server_side:
            return self._ss_cursor_cls(self)
        else:
            return self._cursor_cls(self)

    def rollback(self):
        self.await_(self._connection.rollback())

    def commit(self):
        self.await_(self._connection.commit())

    def close(self):
        self.await_(self._connection.close())


class AsyncAdaptFallback_dbapi_connection(AsyncAdapt_dbapi_connection):
    __slots__ = ()

    await_ = staticmethod(await_fallback)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\fields\numeric.py ===
import decimal

from wtforms import widgets
from wtforms.fields.core import Field
from wtforms.utils import unset_value

__all__ = (
    "IntegerField",
    "DecimalField",
    "FloatField",
    "IntegerRangeField",
    "DecimalRangeField",
)


class LocaleAwareNumberField(Field):
    """
    Base class for implementing locale-aware number parsing.

    Locale-aware numbers require the 'babel' package to be present.
    """

    def __init__(
        self,
        label=None,
        validators=None,
        use_locale=False,
        number_format=None,
        **kwargs,
    ):
        super().__init__(label, validators, **kwargs)
        self.use_locale = use_locale
        if use_locale:
            self.number_format = number_format
            self.locale = kwargs["_form"].meta.locales[0]
            self._init_babel()

    def _init_babel(self):
        try:
            from babel import numbers

            self.babel_numbers = numbers
        except ImportError as exc:
            raise ImportError(
                "Using locale-aware decimals requires the babel library."
            ) from exc

    def _parse_decimal(self, value):
        return self.babel_numbers.parse_decimal(value, self.locale)

    def _format_decimal(self, value):
        return self.babel_numbers.format_decimal(value, self.number_format, self.locale)


class IntegerField(Field):
    """
    A text field, except all input is coerced to an integer.  Erroneous input
    is ignored and will not be accepted as a value.
    """

    widget = widgets.NumberInput()

    def __init__(self, label=None, validators=None, **kwargs):
        super().__init__(label, validators, **kwargs)

    def _value(self):
        if self.raw_data:
            return self.raw_data[0]
        if self.data is not None:
            return str(self.data)
        return ""

    def process_data(self, value):
        if value is None or value is unset_value:
            self.data = None
            return

        try:
            self.data = int(value)
        except (ValueError, TypeError) as exc:
            self.data = None
            raise ValueError(self.gettext("Not a valid integer value.")) from exc

    def process_formdata(self, valuelist):
        if not valuelist:
            return

        try:
            self.data = int(valuelist[0])
        except ValueError as exc:
            self.data = None
            raise ValueError(self.gettext("Not a valid integer value.")) from exc


class DecimalField(LocaleAwareNumberField):
    """
    A text field which displays and coerces data of the `decimal.Decimal` type.

    :param places:
        How many decimal places to quantize the value to for display on form.
        If unset, use 2 decimal places.
        If explicitely set to `None`, does not quantize value.
    :param rounding:
        How to round the value during quantize, for example
        `decimal.ROUND_UP`. If unset, uses the rounding value from the
        current thread's context.
    :param use_locale:
        If True, use locale-based number formatting. Locale-based number
        formatting requires the 'babel' package.
    :param number_format:
        Optional number format for locale. If omitted, use the default decimal
        format for the locale.
    """

    widget = widgets.NumberInput(step="any")

    def __init__(
        self, label=None, validators=None, places=unset_value, rounding=None, **kwargs
    ):
        super().__init__(label, validators, **kwargs)
        if self.use_locale and (places is not unset_value or rounding is not None):
            raise TypeError(
                "When using locale-aware numbers, 'places' and 'rounding' are ignored."
            )

        if places is unset_value:
            places = 2
        self.places = places
        self.rounding = rounding

    def _value(self):
        if self.raw_data:
            return self.raw_data[0]

        if self.data is None:
            return ""

        if self.use_locale:
            return str(self._format_decimal(self.data))

        if self.places is None:
            return str(self.data)

        if not hasattr(self.data, "quantize"):
            # If for some reason, data is a float or int, then format
            # as we would for floats using string formatting.
            format = "%%0.%df" % self.places
            return format % self.data

        exp = decimal.Decimal(".1") ** self.places
        if self.rounding is None:
            quantized = self.data.quantize(exp)
        else:
            quantized = self.data.quantize(exp, rounding=self.rounding)
        return str(quantized)

    def process_formdata(self, valuelist):
        if not valuelist:
            return

        try:
            if self.use_locale:
                self.data = self._parse_decimal(valuelist[0])
            else:
                self.data = decimal.Decimal(valuelist[0])
        except (decimal.InvalidOperation, ValueError) as exc:
            self.data = None
            raise ValueError(self.gettext("Not a valid decimal value.")) from exc


class FloatField(Field):
    """
    A text field, except all input is coerced to an float.  Erroneous input
    is ignored and will not be accepted as a value.
    """

    widget = widgets.TextInput()

    def __init__(self, label=None, validators=None, **kwargs):
        super().__init__(label, validators, **kwargs)

    def _value(self):
        if self.raw_data:
            return self.raw_data[0]
        if self.data is not None:
            return str(self.data)
        return ""

    def process_formdata(self, valuelist):
        if not valuelist:
            return

        try:
            self.data = float(valuelist[0])
        except ValueError as exc:
            self.data = None
            raise ValueError(self.gettext("Not a valid float value.")) from exc


class IntegerRangeField(IntegerField):
    """
    Represents an ``<input type="range">``.
    """

    widget = widgets.RangeInput()


class DecimalRangeField(DecimalField):
    """
    Represents an ``<input type="range">``.
    """

    widget = widgets.RangeInput(step="any")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\naming.py ===
# sql/naming.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Establish constraint and index naming conventions.


"""

from __future__ import annotations

import re

from . import events  # noqa
from .base import _NONE_NAME
from .elements import conv as conv
from .schema import CheckConstraint
from .schema import Column
from .schema import Constraint
from .schema import ForeignKeyConstraint
from .schema import Index
from .schema import PrimaryKeyConstraint
from .schema import Table
from .schema import UniqueConstraint
from .. import event
from .. import exc


class ConventionDict:
    def __init__(self, const, table, convention):
        self.const = const
        self._is_fk = isinstance(const, ForeignKeyConstraint)
        self.table = table
        self.convention = convention
        self._const_name = const.name

    def _key_table_name(self):
        return self.table.name

    def _column_X(self, idx, attrname):
        if self._is_fk:
            try:
                fk = self.const.elements[idx]
            except IndexError:
                return ""
            else:
                return getattr(fk.parent, attrname)
        else:
            cols = list(self.const.columns)
            try:
                col = cols[idx]
            except IndexError:
                return ""
            else:
                return getattr(col, attrname)

    def _key_constraint_name(self):
        if self._const_name in (None, _NONE_NAME):
            raise exc.InvalidRequestError(
                "Naming convention including "
                "%(constraint_name)s token requires that "
                "constraint is explicitly named."
            )
        if not isinstance(self._const_name, conv):
            self.const.name = None
        return self._const_name

    def _key_column_X_key(self, idx):
        # note this method was missing before
        # [ticket:3989], meaning tokens like ``%(column_0_key)s`` weren't
        # working even though documented.
        return self._column_X(idx, "key")

    def _key_column_X_name(self, idx):
        return self._column_X(idx, "name")

    def _key_column_X_label(self, idx):
        return self._column_X(idx, "_ddl_label")

    def _key_referred_table_name(self):
        fk = self.const.elements[0]
        refs = fk.target_fullname.split(".")
        if len(refs) == 3:
            refschema, reftable, refcol = refs
        else:
            reftable, refcol = refs
        return reftable

    def _key_referred_column_X_name(self, idx):
        fk = self.const.elements[idx]
        # note that before [ticket:3989], this method was returning
        # the specification for the :class:`.ForeignKey` itself, which normally
        # would be using the ``.key`` of the column, not the name.
        return fk.column.name

    def __getitem__(self, key):
        if key in self.convention:
            return self.convention[key](self.const, self.table)
        elif hasattr(self, "_key_%s" % key):
            return getattr(self, "_key_%s" % key)()
        else:
            col_template = re.match(r".*_?column_(\d+)(_?N)?_.+", key)
            if col_template:
                idx = col_template.group(1)
                multiples = col_template.group(2)

                if multiples:
                    if self._is_fk:
                        elems = self.const.elements
                    else:
                        elems = list(self.const.columns)
                    tokens = []
                    for idx, elem in enumerate(elems):
                        attr = "_key_" + key.replace("0" + multiples, "X")
                        try:
                            tokens.append(getattr(self, attr)(idx))
                        except AttributeError:
                            raise KeyError(key)
                    sep = "_" if multiples.startswith("_") else ""
                    return sep.join(tokens)
                else:
                    attr = "_key_" + key.replace(idx, "X")
                    idx = int(idx)
                    if hasattr(self, attr):
                        return getattr(self, attr)(idx)
        raise KeyError(key)


_prefix_dict = {
    Index: "ix",
    PrimaryKeyConstraint: "pk",
    CheckConstraint: "ck",
    UniqueConstraint: "uq",
    ForeignKeyConstraint: "fk",
}


def _get_convention(dict_, key):
    for super_ in key.__mro__:
        if super_ in _prefix_dict and _prefix_dict[super_] in dict_:
            return dict_[_prefix_dict[super_]]
        elif super_ in dict_:
            return dict_[super_]
    else:
        return None


def _constraint_name_for_table(const, table):
    metadata = table.metadata
    convention = _get_convention(metadata.naming_convention, type(const))

    if isinstance(const.name, conv):
        return const.name
    elif (
        convention is not None
        and not isinstance(const.name, conv)
        and (
            const.name is None
            or "constraint_name" in convention
            or const.name is _NONE_NAME
        )
    ):
        return conv(
            convention
            % ConventionDict(const, table, metadata.naming_convention)
        )
    elif convention is _NONE_NAME:
        return None


@event.listens_for(
    PrimaryKeyConstraint, "_sa_event_column_added_to_pk_constraint"
)
def _column_added_to_pk_constraint(pk_constraint, col):
    if pk_constraint._implicit_generated:
        # only operate upon the "implicit" pk constraint for now,
        # as we have to force the name to None to reset it.  the
        # "implicit" constraint will only have a naming convention name
        # if at all.
        table = pk_constraint.table
        pk_constraint.name = None
        newname = _constraint_name_for_table(pk_constraint, table)
        if newname:
            pk_constraint.name = newname


@event.listens_for(Constraint, "after_parent_attach")
@event.listens_for(Index, "after_parent_attach")
def _constraint_name(const, table):
    if isinstance(table, Column):
        # this path occurs for a CheckConstraint linked to a Column

        # for column-attached constraint, set another event
        # to link the column attached to the table as this constraint
        # associated with the table.
        event.listen(
            table,
            "after_parent_attach",
            lambda col, table: _constraint_name(const, table),
        )

    elif isinstance(table, Table):
        if isinstance(const.name, conv) or const.name is _NONE_NAME:
            return

        newname = _constraint_name_for_table(const, table)
        if newname:
            const.name = newname

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\relation\binrel.py ===
"""
General binary relations.
"""
from typing import Optional

from sympy.core.singleton import S
from sympy.assumptions import AppliedPredicate, ask, Predicate, Q  # type: ignore
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.boolalg import conjuncts, Not

__all__ = ["BinaryRelation", "AppliedBinaryRelation"]


class BinaryRelation(Predicate):
    """
    Base class for all binary relational predicates.

    Explanation
    ===========

    Binary relation takes two arguments and returns ``AppliedBinaryRelation``
    instance. To evaluate it to boolean value, use :obj:`~.ask()` or
    :obj:`~.refine()` function.

    You can add support for new types by registering the handler to dispatcher.
    See :obj:`~.Predicate()` for more information about predicate dispatching.

    Examples
    ========

    Applying and evaluating to boolean value:

    >>> from sympy import Q, ask, sin, cos
    >>> from sympy.abc import x
    >>> Q.eq(sin(x)**2+cos(x)**2, 1)
    Q.eq(sin(x)**2 + cos(x)**2, 1)
    >>> ask(_)
    True

    You can define a new binary relation by subclassing and dispatching.
    Here, we define a relation $R$ such that $x R y$ returns true if
    $x = y + 1$.

    >>> from sympy import ask, Number, Q
    >>> from sympy.assumptions import BinaryRelation
    >>> class MyRel(BinaryRelation):
    ...     name = "R"
    ...     is_reflexive = False
    >>> Q.R = MyRel()
    >>> @Q.R.register(Number, Number)
    ... def _(n1, n2, assumptions):
    ...     return ask(Q.zero(n1 - n2 - 1), assumptions)
    >>> Q.R(2, 1)
    Q.R(2, 1)

    Now, we can use ``ask()`` to evaluate it to boolean value.

    >>> ask(Q.R(2, 1))
    True
    >>> ask(Q.R(1, 2))
    False

    ``Q.R`` returns ``False`` with minimum cost if two arguments have same
    structure because it is antireflexive relation [1] by
    ``is_reflexive = False``.

    >>> ask(Q.R(x, x))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Reflexive_relation
    """

    is_reflexive: Optional[bool] = None
    is_symmetric: Optional[bool] = None

    def __call__(self, *args):
        if not len(args) == 2:
            raise ValueError("Binary relation takes two arguments, but got %s." % len(args))
        return AppliedBinaryRelation(self, *args)

    @property
    def reversed(self):
        if self.is_symmetric:
            return self
        return None

    @property
    def negated(self):
        return None

    def _compare_reflexive(self, lhs, rhs):
        # quick exit for structurally same arguments
        # do not check != here because it cannot catch the
        # equivalent arguments with different structures.

        # reflexivity does not hold to NaN
        if lhs is S.NaN or rhs is S.NaN:
            return None

        reflexive = self.is_reflexive
        if reflexive is None:
            pass
        elif reflexive and (lhs == rhs):
            return True
        elif not reflexive and (lhs == rhs):
            return False
        return None

    def eval(self, args, assumptions=True):
        # quick exit for structurally same arguments
        ret = self._compare_reflexive(*args)
        if ret is not None:
            return ret

        # don't perform simplify on args here. (done by AppliedBinaryRelation._eval_ask)
        # evaluate by multipledispatch
        lhs, rhs = args
        ret = self.handler(lhs, rhs, assumptions=assumptions)
        if ret is not None:
            return ret

        # check reversed order if the relation is reflexive
        if self.is_reflexive:
            types = (type(lhs), type(rhs))
            if self.handler.dispatch(*types) is not self.handler.dispatch(*reversed(types)):
                ret = self.handler(rhs, lhs, assumptions=assumptions)

        return ret


class AppliedBinaryRelation(AppliedPredicate):
    """
    The class of expressions resulting from applying ``BinaryRelation``
    to the arguments.

    """

    @property
    def lhs(self):
        """The left-hand side of the relation."""
        return self.arguments[0]

    @property
    def rhs(self):
        """The right-hand side of the relation."""
        return self.arguments[1]

    @property
    def reversed(self):
        """
        Try to return the relationship with sides reversed.
        """
        revfunc = self.function.reversed
        if revfunc is None:
            return self
        return revfunc(self.rhs, self.lhs)

    @property
    def reversedsign(self):
        """
        Try to return the relationship with signs reversed.
        """
        revfunc = self.function.reversed
        if revfunc is None:
            return self
        if not any(side.kind is BooleanKind for side in self.arguments):
            return revfunc(-self.lhs, -self.rhs)
        return self

    @property
    def negated(self):
        neg_rel = self.function.negated
        if neg_rel is None:
            return Not(self, evaluate=False)
        return neg_rel(*self.arguments)

    def _eval_ask(self, assumptions):
        conj_assumps = set()
        binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
        for a in conjuncts(assumptions):
            if a.func in binrelpreds:
                conj_assumps.add(binrelpreds[type(a)](*a.args))
            else:
                conj_assumps.add(a)

        # After CNF in assumptions module is modified to take polyadic
        # predicate, this will be removed
        if any(rel in conj_assumps for rel in (self, self.reversed)):
            return True
        neg_rels = (self.negated, self.reversed.negated, Not(self, evaluate=False),
            Not(self.reversed, evaluate=False))
        if any(rel in conj_assumps for rel in neg_rels):
            return False

        # evaluation using multipledispatching
        ret = self.function.eval(self.arguments, assumptions)
        if ret is not None:
            return ret

        # simplify the args and try again
        args = tuple(a.simplify() for a in self.arguments)
        return self.function.eval(args, assumptions)

    def __bool__(self):
        ret = ask(self)
        if ret is None:
            raise TypeError("Cannot determine truth value of %s" % self)
        return ret

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\dispersion.py ===
from sympy.core import S
from sympy.polys import Poly


def dispersionset(p, q=None, *gens, **args):
    r"""Compute the *dispersion set* of two polynomials.

    For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
    and `\deg g > 0` the dispersion set `\operatorname{J}(f, g)` is defined as:

    .. math::
        \operatorname{J}(f, g)
        & := \{a \in \mathbb{N}_0 | \gcd(f(x), g(x+a)) \neq 1\} \\
        &  = \{a \in \mathbb{N}_0 | \deg \gcd(f(x), g(x+a)) \geq 1\}

    For a single polynomial one defines `\operatorname{J}(f) := \operatorname{J}(f, f)`.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.polys.dispersion import dispersion, dispersionset
    >>> from sympy.abc import x

    Dispersion set and dispersion of a simple polynomial:

    >>> fp = poly((x - 3)*(x + 3), x)
    >>> sorted(dispersionset(fp))
    [0, 6]
    >>> dispersion(fp)
    6

    Note that the definition of the dispersion is not symmetric:

    >>> fp = poly(x**4 - 3*x**2 + 1, x)
    >>> gp = fp.shift(-3)
    >>> sorted(dispersionset(fp, gp))
    [2, 3, 4]
    >>> dispersion(fp, gp)
    4
    >>> sorted(dispersionset(gp, fp))
    []
    >>> dispersion(gp, fp)
    -oo

    Computing the dispersion also works over field extensions:

    >>> from sympy import sqrt
    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    >>> sorted(dispersionset(fp, gp))
    [2]
    >>> sorted(dispersionset(gp, fp))
    [1, 4]

    We can even perform the computations for polynomials
    having symbolic coefficients:

    >>> from sympy.abc import a
    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    >>> sorted(dispersionset(fp))
    [0, 1]

    See Also
    ========

    dispersion

    References
    ==========

    .. [1] [ManWright94]_
    .. [2] [Koepf98]_
    .. [3] [Abramov71]_
    .. [4] [Man93]_
    """
    # Check for valid input
    same = False if q is not None else True
    if same:
        q = p

    p = Poly(p, *gens, **args)
    q = Poly(q, *gens, **args)

    if not p.is_univariate or not q.is_univariate:
        raise ValueError("Polynomials need to be univariate")

    # The generator
    if not p.gen == q.gen:
        raise ValueError("Polynomials must have the same generator")
    gen = p.gen

    # We define the dispersion of constant polynomials to be zero
    if p.degree() < 1 or q.degree() < 1:
        return {0}

    # Factor p and q over the rationals
    fp = p.factor_list()
    fq = q.factor_list() if not same else fp

    # Iterate over all pairs of factors
    J = set()
    for s, unused in fp[1]:
        for t, unused in fq[1]:
            m = s.degree()
            n = t.degree()
            if n != m:
                continue
            an = s.LC()
            bn = t.LC()
            if not (an - bn).is_zero:
                continue
            # Note that the roles of `s` and `t` below are switched
            # w.r.t. the original paper. This is for consistency
            # with the description in the book of W. Koepf.
            anm1 = s.coeff_monomial(gen**(m-1))
            bnm1 = t.coeff_monomial(gen**(n-1))
            alpha = (anm1 - bnm1) / S(n*bn)
            if not alpha.is_integer:
                continue
            if alpha < 0 or alpha in J:
                continue
            if n > 1 and not (s - t.shift(alpha)).is_zero:
                continue
            J.add(alpha)

    return J


def dispersion(p, q=None, *gens, **args):
    r"""Compute the *dispersion* of polynomials.

    For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
    and `\deg g > 0` the dispersion `\operatorname{dis}(f, g)` is defined as:

    .. math::
        \operatorname{dis}(f, g)
        & := \max\{ J(f,g) \cup \{0\} \} \\
        &  = \max\{ \{a \in \mathbb{N} | \gcd(f(x), g(x+a)) \neq 1\} \cup \{0\} \}

    and for a single polynomial `\operatorname{dis}(f) := \operatorname{dis}(f, f)`.
    Note that we make the definition `\max\{\} := -\infty`.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.polys.dispersion import dispersion, dispersionset
    >>> from sympy.abc import x

    Dispersion set and dispersion of a simple polynomial:

    >>> fp = poly((x - 3)*(x + 3), x)
    >>> sorted(dispersionset(fp))
    [0, 6]
    >>> dispersion(fp)
    6

    Note that the definition of the dispersion is not symmetric:

    >>> fp = poly(x**4 - 3*x**2 + 1, x)
    >>> gp = fp.shift(-3)
    >>> sorted(dispersionset(fp, gp))
    [2, 3, 4]
    >>> dispersion(fp, gp)
    4
    >>> sorted(dispersionset(gp, fp))
    []
    >>> dispersion(gp, fp)
    -oo

    The maximum of an empty set is defined to be `-\infty`
    as seen in this example.

    Computing the dispersion also works over field extensions:

    >>> from sympy import sqrt
    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    >>> sorted(dispersionset(fp, gp))
    [2]
    >>> sorted(dispersionset(gp, fp))
    [1, 4]

    We can even perform the computations for polynomials
    having symbolic coefficients:

    >>> from sympy.abc import a
    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    >>> sorted(dispersionset(fp))
    [0, 1]

    See Also
    ========

    dispersionset

    References
    ==========

    .. [1] [ManWright94]_
    .. [2] [Koepf98]_
    .. [3] [Abramov71]_
    .. [4] [Man93]_
    """
    J = dispersionset(p, q, *gens, **args)
    if not J:
        # Definition for maximum of empty set
        j = S.NegativeInfinity
    else:
        j = max(J)
    return j

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\comments\comment_sheet.py ===
# Copyright (c) 2010-2024 openpyxl

## Incomplete!
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Integer,
    Set,
    String,
    Bool,
)
from openpyxl.descriptors.excel import Guid, ExtensionList
from openpyxl.descriptors.sequence import NestedSequence

from openpyxl.utils.indexed_list import IndexedList
from openpyxl.xml.constants import SHEET_MAIN_NS

from openpyxl.cell.text import Text
from .author import AuthorList
from .comments import Comment
from .shape_writer import ShapeWriter


class Properties(Serialisable):

    locked = Bool(allow_none=True)
    defaultSize = Bool(allow_none=True)
    _print = Bool(allow_none=True)
    disabled = Bool(allow_none=True)
    uiObject = Bool(allow_none=True)
    autoFill = Bool(allow_none=True)
    autoLine = Bool(allow_none=True)
    altText = String(allow_none=True)
    textHAlign = Set(values=(['left', 'center', 'right', 'justify', 'distributed']))
    textVAlign = Set(values=(['top', 'center', 'bottom', 'justify', 'distributed']))
    lockText = Bool(allow_none=True)
    justLastX = Bool(allow_none=True)
    autoScale = Bool(allow_none=True)
    rowHidden = Bool(allow_none=True)
    colHidden = Bool(allow_none=True)
    # anchor = Typed(expected_type=ObjectAnchor, )

    __elements__ = ('anchor',)

    def __init__(self,
                 locked=None,
                 defaultSize=None,
                 _print=None,
                 disabled=None,
                 uiObject=None,
                 autoFill=None,
                 autoLine=None,
                 altText=None,
                 textHAlign=None,
                 textVAlign=None,
                 lockText=None,
                 justLastX=None,
                 autoScale=None,
                 rowHidden=None,
                 colHidden=None,
                 anchor=None,
                ):
        self.locked = locked
        self.defaultSize = defaultSize
        self._print = _print
        self.disabled = disabled
        self.uiObject = uiObject
        self.autoFill = autoFill
        self.autoLine = autoLine
        self.altText = altText
        self.textHAlign = textHAlign
        self.textVAlign = textVAlign
        self.lockText = lockText
        self.justLastX = justLastX
        self.autoScale = autoScale
        self.rowHidden = rowHidden
        self.colHidden = colHidden
        self.anchor = anchor


class CommentRecord(Serialisable):

    tagname = "comment"

    ref = String()
    authorId = Integer()
    guid = Guid(allow_none=True)
    shapeId = Integer(allow_none=True)
    text = Typed(expected_type=Text)
    commentPr = Typed(expected_type=Properties, allow_none=True)
    author = String(allow_none=True)

    __elements__ = ('text', 'commentPr')
    __attrs__ = ('ref', 'authorId', 'guid', 'shapeId')

    def __init__(self,
                 ref="",
                 authorId=0,
                 guid=None,
                 shapeId=0,
                 text=None,
                 commentPr=None,
                 author=None,
                 height=79,
                 width=144
                ):
        self.ref = ref
        self.authorId = authorId
        self.guid = guid
        self.shapeId = shapeId
        if text is None:
            text = Text()
        self.text = text
        self.commentPr = commentPr
        self.author = author
        self.height = height
        self.width = width


    @classmethod
    def from_cell(cls, cell):
        """
        Class method to convert cell comment
        """
        comment = cell._comment
        ref = cell.coordinate
        self = cls(ref=ref, author=comment.author)
        self.text.t = comment.content
        self.height = comment.height
        self.width = comment.width
        return self


    @property
    def content(self):
        """
        Remove all inline formatting and stuff
        """
        return self.text.content


class CommentSheet(Serialisable):

    tagname = "comments"

    authors = Typed(expected_type=AuthorList)
    commentList = NestedSequence(expected_type=CommentRecord, count=0)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    _id = None
    _path = "/xl/comments/comment{0}.xml"
    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.comments+xml"
    _rel_type = "comments"
    _rel_id = None

    __elements__ = ('authors', 'commentList')

    def __init__(self,
                 authors=None,
                 commentList=None,
                 extLst=None,
                ):
        self.authors = authors
        self.commentList = commentList


    def to_tree(self):
        tree = super().to_tree()
        tree.set("xmlns", SHEET_MAIN_NS)
        return tree


    @property
    def comments(self):
        """
        Return a dictionary of comments keyed by coord
        """
        authors = self.authors.author

        for c in self.commentList:
            yield c.ref, Comment(c.content, authors[c.authorId], c.height, c.width)


    @classmethod
    def from_comments(cls, comments):
        """
        Create a comment sheet from a list of comments for a particular worksheet
        """
        authors = IndexedList()

        # dedupe authors and get indexes
        for comment in comments:
            comment.authorId = authors.add(comment.author)

        return cls(authors=AuthorList(authors), commentList=comments)


    def write_shapes(self, vml=None):
        """
        Create the VML for comments
        """
        sw = ShapeWriter(self.comments)
        return sw.write(vml)


    @property
    def path(self):
        """
        Return path within the archive
        """
        return self._path.format(self._id)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_cte.py ===
# testing/suite/test_cte.py
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
from ... import ForeignKey
from ... import Integer
from ... import select
from ... import String
from ... import testing


class CTETest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ("ctes",)

    run_inserts = "each"
    run_deletes = "each"

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "some_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(50)),
            Column("parent_id", ForeignKey("some_table.id")),
        )

        Table(
            "some_other_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(50)),
            Column("parent_id", Integer),
        )

    @classmethod
    def insert_data(cls, connection):
        connection.execute(
            cls.tables.some_table.insert(),
            [
                {"id": 1, "data": "d1", "parent_id": None},
                {"id": 2, "data": "d2", "parent_id": 1},
                {"id": 3, "data": "d3", "parent_id": 1},
                {"id": 4, "data": "d4", "parent_id": 3},
                {"id": 5, "data": "d5", "parent_id": 3},
            ],
        )

    def test_select_nonrecursive_round_trip(self, connection):
        some_table = self.tables.some_table

        cte = (
            select(some_table)
            .where(some_table.c.data.in_(["d2", "d3", "d4"]))
            .cte("some_cte")
        )
        result = connection.execute(
            select(cte.c.data).where(cte.c.data.in_(["d4", "d5"]))
        )
        eq_(result.fetchall(), [("d4",)])

    def test_select_recursive_round_trip(self, connection):
        some_table = self.tables.some_table

        cte = (
            select(some_table)
            .where(some_table.c.data.in_(["d2", "d3", "d4"]))
            .cte("some_cte", recursive=True)
        )

        cte_alias = cte.alias("c1")
        st1 = some_table.alias()
        # note that SQL Server requires this to be UNION ALL,
        # can't be UNION
        cte = cte.union_all(
            select(st1).where(st1.c.id == cte_alias.c.parent_id)
        )
        result = connection.execute(
            select(cte.c.data)
            .where(cte.c.data != "d2")
            .order_by(cte.c.data.desc())
        )
        eq_(
            result.fetchall(),
            [("d4",), ("d3",), ("d3",), ("d1",), ("d1",), ("d1",)],
        )

    def test_insert_from_select_round_trip(self, connection):
        some_table = self.tables.some_table
        some_other_table = self.tables.some_other_table

        cte = (
            select(some_table)
            .where(some_table.c.data.in_(["d2", "d3", "d4"]))
            .cte("some_cte")
        )
        connection.execute(
            some_other_table.insert().from_select(
                ["id", "data", "parent_id"], select(cte)
            )
        )
        eq_(
            connection.execute(
                select(some_other_table).order_by(some_other_table.c.id)
            ).fetchall(),
            [(2, "d2", 1), (3, "d3", 1), (4, "d4", 3)],
        )

    @testing.requires.ctes_with_update_delete
    @testing.requires.update_from
    def test_update_from_round_trip(self, connection):
        some_table = self.tables.some_table
        some_other_table = self.tables.some_other_table

        connection.execute(
            some_other_table.insert().from_select(
                ["id", "data", "parent_id"], select(some_table)
            )
        )

        cte = (
            select(some_table)
            .where(some_table.c.data.in_(["d2", "d3", "d4"]))
            .cte("some_cte")
        )
        connection.execute(
            some_other_table.update()
            .values(parent_id=5)
            .where(some_other_table.c.data == cte.c.data)
        )
        eq_(
            connection.execute(
                select(some_other_table).order_by(some_other_table.c.id)
            ).fetchall(),
            [
                (1, "d1", None),
                (2, "d2", 5),
                (3, "d3", 5),
                (4, "d4", 5),
                (5, "d5", 3),
            ],
        )

    @testing.requires.ctes_with_update_delete
    @testing.requires.delete_from
    def test_delete_from_round_trip(self, connection):
        some_table = self.tables.some_table
        some_other_table = self.tables.some_other_table

        connection.execute(
            some_other_table.insert().from_select(
                ["id", "data", "parent_id"], select(some_table)
            )
        )

        cte = (
            select(some_table)
            .where(some_table.c.data.in_(["d2", "d3", "d4"]))
            .cte("some_cte")
        )
        connection.execute(
            some_other_table.delete().where(
                some_other_table.c.data == cte.c.data
            )
        )
        eq_(
            connection.execute(
                select(some_other_table).order_by(some_other_table.c.id)
            ).fetchall(),
            [(1, "d1", None), (5, "d5", 3)],
        )

    @testing.requires.ctes_with_update_delete
    def test_delete_scalar_subq_round_trip(self, connection):
        some_table = self.tables.some_table
        some_other_table = self.tables.some_other_table

        connection.execute(
            some_other_table.insert().from_select(
                ["id", "data", "parent_id"], select(some_table)
            )
        )

        cte = (
            select(some_table)
            .where(some_table.c.data.in_(["d2", "d3", "d4"]))
            .cte("some_cte")
        )
        connection.execute(
            some_other_table.delete().where(
                some_other_table.c.data
                == select(cte.c.data)
                .where(cte.c.id == some_other_table.c.id)
                .scalar_subquery()
            )
        )
        eq_(
            connection.execute(
                select(some_other_table).order_by(some_other_table.c.id)
            ).fetchall(),
            [(1, "d1", None), (5, "d5", 3)],
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\__init__.py ===
"""
Python HTTP library with thread-safe connection pooling, file post support, user friendly, and more
"""

from __future__ import annotations

# Set default logging handler to avoid "No handler found" warnings.
import logging
import sys
import typing
import warnings
from logging import NullHandler

from . import exceptions
from ._base_connection import _TYPE_BODY
from ._collections import HTTPHeaderDict
from ._version import __version__
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, connection_from_url
from .filepost import _TYPE_FIELDS, encode_multipart_formdata
from .poolmanager import PoolManager, ProxyManager, proxy_from_url
from .response import BaseHTTPResponse, HTTPResponse
from .util.request import make_headers
from .util.retry import Retry
from .util.timeout import Timeout

# Ensure that Python is compiled with OpenSSL 1.1.1+
# If the 'ssl' module isn't available at all that's
# fine, we only care if the module is available.
try:
    import ssl
except ImportError:
    pass
else:
    if not ssl.OPENSSL_VERSION.startswith("OpenSSL "):  # Defensive:
        warnings.warn(
            "urllib3 v2 only supports OpenSSL 1.1.1+, currently "
            f"the 'ssl' module is compiled with {ssl.OPENSSL_VERSION!r}. "
            "See: https://github.com/urllib3/urllib3/issues/3020",
            exceptions.NotOpenSSLWarning,
        )
    elif ssl.OPENSSL_VERSION_INFO < (1, 1, 1):  # Defensive:
        raise ImportError(
            "urllib3 v2 only supports OpenSSL 1.1.1+, currently "
            f"the 'ssl' module is compiled with {ssl.OPENSSL_VERSION!r}. "
            "See: https://github.com/urllib3/urllib3/issues/2168"
        )

__author__ = "Andrey Petrov (andrey.petrov@shazow.net)"
__license__ = "MIT"
__version__ = __version__

__all__ = (
    "HTTPConnectionPool",
    "HTTPHeaderDict",
    "HTTPSConnectionPool",
    "PoolManager",
    "ProxyManager",
    "HTTPResponse",
    "Retry",
    "Timeout",
    "add_stderr_logger",
    "connection_from_url",
    "disable_warnings",
    "encode_multipart_formdata",
    "make_headers",
    "proxy_from_url",
    "request",
    "BaseHTTPResponse",
)

logging.getLogger(__name__).addHandler(NullHandler())


def add_stderr_logger(
    level: int = logging.DEBUG,
) -> logging.StreamHandler[typing.TextIO]:
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging.

    Returns the handler after adding it.
    """
    # This method needs to be in this __init__.py to get the __name__ correct
    # even if urllib3 is vendored within another package.
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


# ... Clean up.
del NullHandler


# All warning filters *must* be appended unless you're really certain that they
# shouldn't be: otherwise, it's very hard for users to use most Python
# mechanisms to silence them.
# SecurityWarning's always go off by default.
warnings.simplefilter("always", exceptions.SecurityWarning, append=True)
# InsecurePlatformWarning's don't vary between requests, so we keep it default.
warnings.simplefilter("default", exceptions.InsecurePlatformWarning, append=True)


def disable_warnings(category: type[Warning] = exceptions.HTTPWarning) -> None:
    """
    Helper for quickly disabling all urllib3 warnings.
    """
    warnings.simplefilter("ignore", category)


_DEFAULT_POOL = PoolManager()


def request(
    method: str,
    url: str,
    *,
    body: _TYPE_BODY | None = None,
    fields: _TYPE_FIELDS | None = None,
    headers: typing.Mapping[str, str] | None = None,
    preload_content: bool | None = True,
    decode_content: bool | None = True,
    redirect: bool | None = True,
    retries: Retry | bool | int | None = None,
    timeout: Timeout | float | int | None = 3,
    json: typing.Any | None = None,
) -> BaseHTTPResponse:
    """
    A convenience, top-level request method. It uses a module-global ``PoolManager`` instance.
    Therefore, its side effects could be shared across dependencies relying on it.
    To avoid side effects create a new ``PoolManager`` instance and use it instead.
    The method does not accept low-level ``**urlopen_kw`` keyword arguments.

    :param method:
        HTTP request method (such as GET, POST, PUT, etc.)

    :param url:
        The URL to perform the request on.

    :param body:
        Data to send in the request body, either :class:`str`, :class:`bytes`,
        an iterable of :class:`str`/:class:`bytes`, or a file-like object.

    :param fields:
        Data to encode and send in the request body.

    :param headers:
        Dictionary of custom headers to send, such as User-Agent,
        If-None-Match, etc.

    :param bool preload_content:
        If True, the response's body will be preloaded into memory.

    :param bool decode_content:
        If True, will attempt to decode the body based on the
        'content-encoding' header.

    :param redirect:
        If True, automatically handle redirects (status codes 301, 302,
        303, 307, 308). Each redirect counts as a retry. Disabling retries
        will disable redirect, too.

    :param retries:
        Configure the number of retries to allow before raising a
        :class:`~urllib3.exceptions.MaxRetryError` exception.

        If ``None`` (default) will retry 3 times, see ``Retry.DEFAULT``. Pass a
        :class:`~urllib3.util.retry.Retry` object for fine-grained control
        over different types of retries.
        Pass an integer number to retry connection errors that many times,
        but no other types of errors. Pass zero to never retry.

        If ``False``, then retries are disabled and any exception is raised
        immediately. Also, instead of raising a MaxRetryError on redirects,
        the redirect response will be returned.

    :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

    :param timeout:
        If specified, overrides the default timeout for this one
        request. It may be a float (in seconds) or an instance of
        :class:`urllib3.util.Timeout`.

    :param json:
        Data to encode and send as JSON with UTF-encoded in the request body.
        The ``"Content-Type"`` header will be set to ``"application/json"``
        unless specified otherwise.
    """

    return _DEFAULT_POOL.request(
        method,
        url,
        body=body,
        fields=fields,
        headers=headers,
        preload_content=preload_content,
        decode_content=decode_content,
        redirect=redirect,
        retries=retries,
        timeout=timeout,
        json=json,
    )


if sys.platform == "emscripten":
    from .contrib.emscripten import inject_into_urllib3  # noqa: 401

    inject_into_urllib3()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\_internal.py ===
from __future__ import annotations

import logging
import re
import sys
import typing as t
from datetime import datetime
from datetime import timezone

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment

    from .wrappers.request import Request

_logger: logging.Logger | None = None


class _Missing:
    def __repr__(self) -> str:
        return "no value"

    def __reduce__(self) -> str:
        return "_missing"


_missing = _Missing()


def _wsgi_decoding_dance(s: str) -> str:
    return s.encode("latin1").decode(errors="replace")


def _wsgi_encoding_dance(s: str) -> str:
    return s.encode().decode("latin1")


def _get_environ(obj: WSGIEnvironment | Request) -> WSGIEnvironment:
    env = getattr(obj, "environ", obj)
    assert isinstance(
        env, dict
    ), f"{type(obj).__name__!r} is not a WSGI environment (has to be a dict)"
    return env


def _has_level_handler(logger: logging.Logger) -> bool:
    """Check if there is a handler in the logging chain that will handle
    the given logger's effective level.
    """
    level = logger.getEffectiveLevel()
    current = logger

    while current:
        if any(handler.level <= level for handler in current.handlers):
            return True

        if not current.propagate:
            break

        current = current.parent  # type: ignore

    return False


class _ColorStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """On Windows, wrap stream with Colorama for ANSI style support."""

    def __init__(self) -> None:
        try:
            import colorama
        except ImportError:
            stream = None
        else:
            stream = colorama.AnsiToWin32(sys.stderr)

        super().__init__(stream)


def _log(type: str, message: str, *args: t.Any, **kwargs: t.Any) -> None:
    """Log a message to the 'werkzeug' logger.

    The logger is created the first time it is needed. If there is no
    level set, it is set to :data:`logging.INFO`. If there is no handler
    for the logger's effective level, a :class:`logging.StreamHandler`
    is added.
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger("werkzeug")

        if _logger.level == logging.NOTSET:
            _logger.setLevel(logging.INFO)

        if not _has_level_handler(_logger):
            _logger.addHandler(_ColorStreamHandler())

    getattr(_logger, type)(message.rstrip(), *args, **kwargs)


@t.overload
def _dt_as_utc(dt: None) -> None: ...


@t.overload
def _dt_as_utc(dt: datetime) -> datetime: ...


def _dt_as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return dt

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)

    return dt


_TAccessorValue = t.TypeVar("_TAccessorValue")


class _DictAccessorProperty(t.Generic[_TAccessorValue]):
    """Baseclass for `environ_property` and `header_property`."""

    read_only = False

    def __init__(
        self,
        name: str,
        default: _TAccessorValue | None = None,
        load_func: t.Callable[[str], _TAccessorValue] | None = None,
        dump_func: t.Callable[[_TAccessorValue], str] | None = None,
        read_only: bool | None = None,
        doc: str | None = None,
    ) -> None:
        self.name = name
        self.default = default
        self.load_func = load_func
        self.dump_func = dump_func
        if read_only is not None:
            self.read_only = read_only
        self.__doc__ = doc

    def lookup(self, instance: t.Any) -> t.MutableMapping[str, t.Any]:
        raise NotImplementedError

    @t.overload
    def __get__(
        self, instance: None, owner: type
    ) -> _DictAccessorProperty[_TAccessorValue]: ...

    @t.overload
    def __get__(self, instance: t.Any, owner: type) -> _TAccessorValue: ...

    def __get__(
        self, instance: t.Any | None, owner: type
    ) -> _TAccessorValue | _DictAccessorProperty[_TAccessorValue]:
        if instance is None:
            return self

        storage = self.lookup(instance)

        if self.name not in storage:
            return self.default  # type: ignore

        value = storage[self.name]

        if self.load_func is not None:
            try:
                return self.load_func(value)
            except (ValueError, TypeError):
                return self.default  # type: ignore

        return value  # type: ignore

    def __set__(self, instance: t.Any, value: _TAccessorValue) -> None:
        if self.read_only:
            raise AttributeError("read only property")

        if self.dump_func is not None:
            self.lookup(instance)[self.name] = self.dump_func(value)
        else:
            self.lookup(instance)[self.name] = value

    def __delete__(self, instance: t.Any) -> None:
        if self.read_only:
            raise AttributeError("read only property")

        self.lookup(instance).pop(self.name, None)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


_plain_int_re = re.compile(r"-?\d+", re.ASCII)


def _plain_int(value: str) -> int:
    """Parse an int only if it is only ASCII digits and ``-``.

    This disallows ``+``, ``_``, and non-ASCII digits, which are accepted by ``int`` but
    are not allowed in HTTP header values.

    Any leading or trailing whitespace is stripped
    """
    value = value.strip()
    if _plain_int_re.fullmatch(value) is None:
        raise ValueError

    return int(value)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\pickle.py ===
""" pickle compat """
from __future__ import annotations

import pickle
from typing import (
    TYPE_CHECKING,
    Any,
)
import warnings

from pandas.compat import pickle_compat as pc
from pandas.util._decorators import doc

from pandas.core.shared_docs import _shared_docs

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        ReadPickleBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import (
        DataFrame,
        Series,
    )


@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "filepath_or_buffer",
)
def to_pickle(
    obj: Any,
    filepath_or_buffer: FilePath | WriteBuffer[bytes],
    compression: CompressionOptions = "infer",
    protocol: int = pickle.HIGHEST_PROTOCOL,
    storage_options: StorageOptions | None = None,
) -> None:
    """
    Pickle (serialize) object to file.

    Parameters
    ----------
    obj : any object
        Any python object.
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``write()`` function.
        Also accepts URL. URL has to be of S3 or GCS.
    {compression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    protocol : int
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL (see [1], paragraph 12.1.2). The possible
        values for this parameter depend on the version of Python. For Python
        2.x, possible values are 0, 1, 2. For Python>=3.0, 3 is a valid value.
        For Python >= 3.4, 4 is a valid value. A negative value for the
        protocol parameter is equivalent to setting its value to
        HIGHEST_PROTOCOL.

    {storage_options}

        .. [1] https://docs.python.org/3/library/pickle.html

    See Also
    --------
    read_pickle : Load pickled pandas object (or any object) from file.
    DataFrame.to_hdf : Write DataFrame to an HDF5 file.
    DataFrame.to_sql : Write DataFrame to a SQL database.
    DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

    Examples
    --------
    >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})  # doctest: +SKIP
    >>> original_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")  # doctest: +SKIP

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
    >>> unpickled_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    """  # noqa: E501
    if protocol < 0:
        protocol = pickle.HIGHEST_PROTOCOL

    with get_handle(
        filepath_or_buffer,
        "wb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:
        # letting pickle write directly to the buffer is more memory-efficient
        pickle.dump(obj, handles.handle, protocol=protocol)


@doc(
    storage_options=_shared_docs["storage_options"],
    decompression_options=_shared_docs["decompression_options"] % "filepath_or_buffer",
)
def read_pickle(
    filepath_or_buffer: FilePath | ReadPickleBuffer,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions | None = None,
) -> DataFrame | Series:
    """
    Load pickled pandas object (or any object) from file.

    .. warning::

       Loading pickled data received from untrusted sources can be
       unsafe. See `here <https://docs.python.org/3/library/pickle.html>`__.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a binary ``readlines()`` function.
        Also accepts URL. URL is not limited to S3 and GCS.

    {decompression_options}

        .. versionchanged:: 1.4.0 Zstandard support.

    {storage_options}

    Returns
    -------
    same type as object stored in file

    See Also
    --------
    DataFrame.to_pickle : Pickle (serialize) DataFrame object to file.
    Series.to_pickle : Pickle (serialize) Series object to file.
    read_hdf : Read HDF5 file into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.
    read_parquet : Load a parquet object, returning a DataFrame.

    Notes
    -----
    read_pickle is only guaranteed to be backwards compatible to pandas 0.20.3
    provided the object was serialized with to_pickle.

    Examples
    --------
    >>> original_df = pd.DataFrame(
    ...     {{"foo": range(5), "bar": range(5, 10)}}
    ...    )  # doctest: +SKIP
    >>> original_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    >>> pd.to_pickle(original_df, "./dummy.pkl")  # doctest: +SKIP

    >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
    >>> unpickled_df  # doctest: +SKIP
       foo  bar
    0    0    5
    1    1    6
    2    2    7
    3    3    8
    4    4    9
    """
    excs_to_catch = (AttributeError, ImportError, ModuleNotFoundError, TypeError)
    with get_handle(
        filepath_or_buffer,
        "rb",
        compression=compression,
        is_text=False,
        storage_options=storage_options,
    ) as handles:
        # 1) try standard library Pickle
        # 2) try pickle_compat (older pandas version) to handle subclass changes
        # 3) try pickle_compat with latin-1 encoding upon a UnicodeDecodeError

        try:
            # TypeError for Cython complaints about object.__new__ vs Tick.__new__
            try:
                with warnings.catch_warnings(record=True):
                    # We want to silence any warnings about, e.g. moved modules.
                    warnings.simplefilter("ignore", Warning)
                    return pickle.load(handles.handle)
            except excs_to_catch:
                # e.g.
                #  "No module named 'pandas.core.sparse.series'"
                #  "Can't get attribute '__nat_unpickle' on <module 'pandas._libs.tslib"
                return pc.load(handles.handle, encoding=None)
        except UnicodeDecodeError:
            # e.g. can occur for files written in py27; see GH#28645 and GH#31988
            return pc.load(handles.handle, encoding="latin-1")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\network\lazy_wheel.py ===
"""Lazy ZIP over HTTP"""

__all__ = ["HTTPRangeRequestUnsupported", "dist_from_wheel_url"]

from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Generator, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile

from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response

from pip._internal.metadata import BaseDistribution, MemoryWheel, get_wheel_distribution
from pip._internal.network.session import PipSession
from pip._internal.network.utils import HEADERS, raise_for_status, response_chunks


class HTTPRangeRequestUnsupported(Exception):
    pass


def dist_from_wheel_url(name: str, url: str, session: PipSession) -> BaseDistribution:
    """Return a distribution object from the given wheel URL.

    This uses HTTP range requests to only fetch the portion of the wheel
    containing metadata, just enough for the object to be constructed.
    If such requests are not supported, HTTPRangeRequestUnsupported
    is raised.
    """
    with LazyZipOverHTTP(url, session) as zf:
        # For read-only ZIP files, ZipFile only needs methods read,
        # seek, seekable and tell, not the whole IO protocol.
        wheel = MemoryWheel(zf.name, zf)  # type: ignore
        # After context manager exit, wheel.name
        # is an invalid file by intention.
        return get_wheel_distribution(wheel, canonicalize_name(name))


class LazyZipOverHTTP:
    """File-like object mapped to a ZIP file over HTTP.

    This uses HTTP range requests to lazily fetch the file's content,
    which is supposed to be fed to ZipFile.  If such requests are not
    supported by the server, raise HTTPRangeRequestUnsupported
    during initialization.
    """

    def __init__(
        self, url: str, session: PipSession, chunk_size: int = CONTENT_CHUNK_SIZE
    ) -> None:
        head = session.head(url, headers=HEADERS)
        raise_for_status(head)
        assert head.status_code == 200
        self._session, self._url, self._chunk_size = session, url, chunk_size
        self._length = int(head.headers["Content-Length"])
        self._file = NamedTemporaryFile()
        self.truncate(self._length)
        self._left: List[int] = []
        self._right: List[int] = []
        if "bytes" not in head.headers.get("Accept-Ranges", "none"):
            raise HTTPRangeRequestUnsupported("range request is not supported")
        self._check_zip()

    @property
    def mode(self) -> str:
        """Opening mode, which is always rb."""
        return "rb"

    @property
    def name(self) -> str:
        """Path to the underlying file."""
        return self._file.name

    def seekable(self) -> bool:
        """Return whether random access is supported, which is True."""
        return True

    def close(self) -> None:
        """Close the file."""
        self._file.close()

    @property
    def closed(self) -> bool:
        """Whether the file is closed."""
        return self._file.closed

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the object and return them.

        As a convenience, if size is unspecified or -1,
        all bytes until EOF are returned.  Fewer than
        size bytes may be returned if EOF is reached.
        """
        download_size = max(size, self._chunk_size)
        start, length = self.tell(), self._length
        stop = length if size < 0 else min(start + download_size, length)
        start = max(0, stop - download_size)
        self._download(start, stop - 1)
        return self._file.read(size)

    def readable(self) -> bool:
        """Return whether the file is readable, which is True."""
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        """Change stream position and return the new absolute position.

        Seek to offset relative position indicated by whence:
        * 0: Start of stream (the default).  pos should be >= 0;
        * 1: Current position - pos may be negative;
        * 2: End of stream - pos usually negative.
        """
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        """Return the current position."""
        return self._file.tell()

    def truncate(self, size: Optional[int] = None) -> int:
        """Resize the stream to the given size in bytes.

        If size is unspecified resize to the current position.
        The current stream position isn't changed.

        Return the new file size.
        """
        return self._file.truncate(size)

    def writable(self) -> bool:
        """Return False."""
        return False

    def __enter__(self) -> "LazyZipOverHTTP":
        self._file.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._file.__exit__(*exc)

    @contextmanager
    def _stay(self) -> Generator[None, None, None]:
        """Return a context manager keeping the position.

        At the end of the block, seek back to original position.
        """
        pos = self.tell()
        try:
            yield
        finally:
            self.seek(pos)

    def _check_zip(self) -> None:
        """Check and download until the file is a valid ZIP."""
        end = self._length - 1
        for start in reversed(range(0, end, self._chunk_size)):
            self._download(start, end)
            with self._stay():
                try:
                    # For read-only ZIP files, ZipFile only needs
                    # methods read, seek, seekable and tell.
                    ZipFile(self)
                except BadZipFile:
                    pass
                else:
                    break

    def _stream_response(
        self, start: int, end: int, base_headers: Dict[str, str] = HEADERS
    ) -> Response:
        """Return HTTP response to a range request from start to end."""
        headers = base_headers.copy()
        headers["Range"] = f"bytes={start}-{end}"
        # TODO: Get range requests to be correctly cached
        headers["Cache-Control"] = "no-cache"
        return self._session.get(self._url, headers=headers, stream=True)

    def _merge(
        self, start: int, end: int, left: int, right: int
    ) -> Generator[Tuple[int, int], None, None]:
        """Return a generator of intervals to be fetched.

        Args:
            start (int): Start of needed interval
            end (int): End of needed interval
            left (int): Index of first overlapping downloaded data
            right (int): Index after last overlapping downloaded data
        """
        lslice, rslice = self._left[left:right], self._right[left:right]
        i = start = min([start] + lslice[:1])
        end = max([end] + rslice[-1:])
        for j, k in zip(lslice, rslice):
            if j > i:
                yield i, j - 1
            i = k + 1
        if i <= end:
            yield i, end
        self._left[left:right], self._right[left:right] = [start], [end]

    def _download(self, start: int, end: int) -> None:
        """Download bytes from start to end inclusively."""
        with self._stay():
            left = bisect_left(self._right, start)
            right = bisect_right(self._left, end)
            for start, end in self._merge(start, end, left, right):
                response = self._stream_response(start, end)
                response.raise_for_status()
                self.seek(start)
                for chunk in response_chunks(response, self._chunk_size):
                    self._file.write(chunk)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\cache.py ===
""" Caching facility for SymPy """
from importlib import import_module
from typing import Callable

class _cache(list):
    """ List of cached functions """

    def print_cache(self):
        """print cache info"""

        for item in self:
            name = item.__name__
            myfunc = item
            while hasattr(myfunc, '__wrapped__'):
                if hasattr(myfunc, 'cache_info'):
                    info = myfunc.cache_info()
                    break
                else:
                    myfunc = myfunc.__wrapped__
            else:
                info = None

            print(name, info)

    def clear_cache(self):
        """clear cache content"""
        for item in self:
            myfunc = item
            while hasattr(myfunc, '__wrapped__'):
                if hasattr(myfunc, 'cache_clear'):
                    myfunc.cache_clear()
                    break
                else:
                    myfunc = myfunc.__wrapped__


# global cache registry:
CACHE = _cache()
# make clear and print methods available
print_cache = CACHE.print_cache
clear_cache = CACHE.clear_cache

from functools import lru_cache, wraps

def __cacheit(maxsize):
    """caching decorator.

        important: the result of cached function must be *immutable*


        Examples
        ========

        >>> from sympy import cacheit
        >>> @cacheit
        ... def f(a, b):
        ...    return a+b

        >>> @cacheit
        ... def f(a, b): # noqa: F811
        ...    return [a, b] # <-- WRONG, returns mutable object

        to force cacheit to check returned results mutability and consistency,
        set environment variable SYMPY_USE_CACHE to 'debug'
    """
    def func_wrapper(func):
        cfunc = lru_cache(maxsize, typed=True)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                retval = cfunc(*args, **kwargs)
            except TypeError as e:
                if not e.args or not e.args[0].startswith('unhashable type:'):
                    raise
                retval = func(*args, **kwargs)
            return retval

        wrapper.cache_info = cfunc.cache_info
        wrapper.cache_clear = cfunc.cache_clear

        CACHE.append(wrapper)
        return wrapper

    return func_wrapper
########################################


def __cacheit_nocache(func):
    return func


def __cacheit_debug(maxsize):
    """cacheit + code to check cache consistency"""
    def func_wrapper(func):
        cfunc = __cacheit(maxsize)(func)

        @wraps(func)
        def wrapper(*args, **kw_args):
            # always call function itself and compare it with cached version
            r1 = func(*args, **kw_args)
            r2 = cfunc(*args, **kw_args)

            # try to see if the result is immutable
            #
            # this works because:
            #
            # hash([1,2,3])         -> raise TypeError
            # hash({'a':1, 'b':2})  -> raise TypeError
            # hash((1,[2,3]))       -> raise TypeError
            #
            # hash((1,2,3))         -> just computes the hash
            hash(r1), hash(r2)

            # also see if returned values are the same
            if r1 != r2:
                raise RuntimeError("Returned values are not the same")
            return r1
        return wrapper
    return func_wrapper


def _getenv(key, default=None):
    from os import getenv
    return getenv(key, default)

# SYMPY_USE_CACHE=yes/no/debug
USE_CACHE = _getenv('SYMPY_USE_CACHE', 'yes').lower()
# SYMPY_CACHE_SIZE=some_integer/None
# special cases :
#  SYMPY_CACHE_SIZE=0    -> No caching
#  SYMPY_CACHE_SIZE=None -> Unbounded caching
scs = _getenv('SYMPY_CACHE_SIZE', '1000')
if scs.lower() == 'none':
    SYMPY_CACHE_SIZE = None
else:
    try:
        SYMPY_CACHE_SIZE = int(scs)
    except ValueError:
        raise RuntimeError(
            'SYMPY_CACHE_SIZE must be a valid integer or None. ' + \
            'Got: %s' % SYMPY_CACHE_SIZE)

if USE_CACHE == 'no':
    cacheit = __cacheit_nocache
elif USE_CACHE == 'yes':
    cacheit = __cacheit(SYMPY_CACHE_SIZE)
elif USE_CACHE == 'debug':
    cacheit = __cacheit_debug(SYMPY_CACHE_SIZE)   # a lot slower
else:
    raise RuntimeError(
        'unrecognized value for SYMPY_USE_CACHE: %s' % USE_CACHE)


def cached_property(func):
    '''Decorator to cache property method'''
    attrname = '__' + func.__name__
    _cached_property_sentinel = object()
    def propfunc(self):
        val = getattr(self, attrname, _cached_property_sentinel)
        if val is _cached_property_sentinel:
            val = func(self)
            setattr(self, attrname, val)
        return val
    return property(propfunc)


def lazy_function(module : str, name : str) -> Callable:
    """Create a lazy proxy for a function in a module.

    The module containing the function is not imported until the function is used.

    """
    func = None

    def _get_function():
        nonlocal func
        if func is None:
            func = getattr(import_module(module), name)
        return func

    # The metaclass is needed so that help() shows the docstring
    class LazyFunctionMeta(type):
        @property
        def __doc__(self):
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

    class LazyFunction(metaclass=LazyFunctionMeta):
        def __call__(self, *args, **kwargs):
            # inline get of function for performance gh-23832
            nonlocal func
            if func is None:
                func = getattr(import_module(module), name)
            return func(*args, **kwargs)

        @property
        def __doc__(self):
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

        def __str__(self):
            return _get_function().__str__()

        def __repr__(self):
            return f"<{__class__.__name__} object at 0x{id(self):x}>: wrapping '{module}.{name}'"

    return LazyFunction()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_skiplayernorm.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionSkipLayerNormalization(Fusion):
    """
    Fuse Add + LayerNormalization into one node: SkipLayerNormalization
    Note: This fusion does not check the input shape of Add and LayerNormalization.
    """

    def __init__(
        self,
        model: OnnxModel,
        fused_op_type: str = "SkipLayerNormalization",
        search_op_types: str = "LayerNormalization",
        shape_infer: bool = True,
    ):
        super().__init__(model, fused_op_type, search_op_types)
        if shape_infer:
            # Update shape inference is needed since other fusions might add new edge which does not have shape info yet.
            self.shape_infer_helper = self.model.infer_runtime_shape({"batch_size": 4, "seq_len": 7}, update=True)
            if self.shape_infer_helper is None:
                # TODO(tianleiwu): support subgraph in shape inference or add broadcasting in SkipLayerNormalization op.
                logger.warning("symbolic shape inference disabled or failed.")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        add = self.model.get_parent(node, 0, output_name_to_node)

        # In some models there is input_ids->gather->add->LayerNorm and one of input of the
        # add node is initializer with fixed shape which should not be fused into SkipLayerNorm
        if add is None or add.op_type != "Add":
            return

        # The number of inputs of add should be 2
        if len(add.input) != 2:
            return

        for add_input in add.input:
            if self.model.get_initializer(add_input) is not None:
                return

        # To avoid an Add node have two children of LayerNormalization, we shall only fuse one SkipLayerNormalization
        if add in self.nodes_to_remove:
            return

        # Root Mean Square Layer Normalization
        simplified = node.op_type == "SimplifiedLayerNormalization"

        if hasattr(self, "shape_infer_helper"):
            if self.shape_infer_helper is not None:
                if (
                    self.shape_infer_helper.get_edge_shape(add.input[0])
                    and len(self.shape_infer_helper.get_edge_shape(add.input[0])) != 3
                ):
                    logger.debug("skip SkipLayerNormalization fusion since shape of input %s is not 3D", add.input[0])
                    return

                # TODO(tianleiwu): support broadcasting Skip shape (1, sequence_length, hidden_size) or (sequence_length, hidden_size)
                if not self.shape_infer_helper.compare_shape(add.input[0], add.input[1]):
                    logger.debug(
                        "skip SkipLayerNormalization fusion since shape of inputs (%s, %s) are not same",
                        add.input[0],
                        add.input[1],
                    )
                    return
            else:
                logger.debug("skip SkipLayerNormalization fusion since symbolic shape inference failed")
                return

        gather_path = self.model.match_parent_path(add, ["Gather"], [None])
        if gather_path is not None and self.model.find_graph_input(gather_path[0].input[1]) is None:
            if self.model.match_parent_path(gather_path[0], ["ConstantOfShape"], [1]) is None:
                return

        # This means that the residual Add before the LayerNormalization produces an output
        # that is consumed by some other nodes or graph output other than the LayerNormalization itself
        # We can still go ahead with the SkipLayerNormalization fusion but we need to
        # preserve the output of Add and that needs to be produced by SkipLayerNormalization.
        add_has_graph_output = self.model.find_graph_output(add.output[0]) is not None
        residual_add_has_multiple_consumers = (
            add_has_graph_output or len(self.model.get_children(add, input_name_to_nodes)) > 1
        )

        outputs_to_keep = node.output

        if residual_add_has_multiple_consumers:
            outputs_to_keep.extend([add.output[0]])

        outputs = [node.output[0]]

        # Skip the other optional outputs of SkipLayerNormalization before adding the Add's output
        if residual_add_has_multiple_consumers:
            outputs.extend(["", "", add.output[0]])

        if self.model.is_safe_to_fuse_nodes([add, node], outputs_to_keep, input_name_to_nodes, output_name_to_node):
            self.nodes_to_remove.extend([add, node])

            inputs = (
                [add.input[0], add.input[1], node.input[1], node.input[2]]
                if not simplified
                else [add.input[0], add.input[1], node.input[1]]
            )
            normalize_node = helper.make_node(
                self.fused_op_type,
                inputs=inputs,
                outputs=outputs,
                name=self.model.create_node_name(self.fused_op_type, name_prefix="SkipLayerNorm"),
            )
            normalize_node.domain = "com.microsoft"

            # Pass attribute "epsilon" from layernorm node to SkipLayerNormalization
            for att in node.attribute:
                if att.name == "epsilon":
                    normalize_node.attribute.extend([att])

            # Set default epsilon if no epsilon exists from layernorm
            if len(normalize_node.attribute) == 0:
                normalize_node.attribute.extend([helper.make_attribute("epsilon", 1.0e-12)])

            self.nodes_to_add.append(normalize_node)
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name


class FusionBiasSkipLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "SkipLayerNormalization", "SkipLayerNormalization", "add bias")

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        if len(node.input) != 4:
            return

        return_indice = []
        nodes = self.model.match_parent_path(node, ["Add", "MatMul"], [None, None], output_name_to_node, return_indice)
        if nodes is not None:
            (add, _matmul) = nodes
        else:
            # In case of fp16, we could have a Cast between the MatMul and the bias Add
            return_indice = []
            nodes = self.model.match_parent_path(
                node, ["Add", "Cast", "MatMul"], [None, None, None], output_name_to_node, return_indice
            )
            if nodes is not None:
                (add, _cast, _matmul) = nodes
            else:
                return

        assert len(return_indice) == 2 or len(return_indice) == 3
        add_input_index = return_indice[0]
        if add_input_index >= 2:
            return
        sln_input = add.input[return_indice[1]]
        bias_input = add.input[1 - return_indice[1]]
        skip_input = node.input[1 - add_input_index]

        # bias should be one dimension
        initializer = self.model.get_initializer(bias_input)
        if initializer is None:
            return
        bias_weight = NumpyHelper.to_array(initializer)
        if bias_weight is None:
            logger.debug("Bias weight not found")
            return
        if len(bias_weight.shape) != 1:
            logger.debug("Bias weight is not 1D")
            return

        subgraph_nodes = [node, add]
        if not self.model.is_safe_to_fuse_nodes(subgraph_nodes, node.output, input_name_to_nodes, output_name_to_node):
            logger.debug("Skip fusing SkipLayerNormalization with Bias since it is not safe")
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        inputs = [
            sln_input,
            skip_input,
            node.input[2],
            node.input[3],
            bias_input,
        ]
        new_node = helper.make_node(
            "SkipLayerNormalization",
            inputs=inputs,
            outputs=node.output,
            name=self.model.create_node_name("SkipLayerNormalization", "SkipLayerNorm_AddBias_"),
        )
        new_node.domain = "com.microsoft"

        # Pass attribute "epsilon" from skiplayernorm node to skiplayernorm(add bias)
        for att in node.attribute:
            if att.name == "epsilon":
                new_node.attribute.extend([att])

        # Set default epsilon if no epsilon exists from skiplayernorm
        if len(new_node.attribute) == 0:
            new_node.attribute.extend([helper.make_attribute("epsilon", 1.0e-12)])

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\dependency_groups\_implementation.py ===
from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping

from pip._vendor.packaging.requirements import Requirement


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _normalize_group_names(
    dependency_groups: Mapping[str, str | Mapping[str, str]],
) -> Mapping[str, str | Mapping[str, str]]:
    original_names: dict[str, list[str]] = {}
    normalized_groups = {}

    for group_name, value in dependency_groups.items():
        normed_group_name = _normalize_name(group_name)
        original_names.setdefault(normed_group_name, []).append(group_name)
        normalized_groups[normed_group_name] = value

    errors = []
    for normed_name, names in original_names.items():
        if len(names) > 1:
            errors.append(f"{normed_name} ({', '.join(names)})")
    if errors:
        raise ValueError(f"Duplicate dependency group names: {', '.join(errors)}")

    return normalized_groups


@dataclasses.dataclass
class DependencyGroupInclude:
    include_group: str


class CyclicDependencyError(ValueError):
    """
    An error representing the detection of a cycle.
    """

    def __init__(self, requested_group: str, group: str, include_group: str) -> None:
        self.requested_group = requested_group
        self.group = group
        self.include_group = include_group

        if include_group == group:
            reason = f"{group} includes itself"
        else:
            reason = f"{include_group} -> {group}, {group} -> {include_group}"
        super().__init__(
            "Cyclic dependency group include while resolving "
            f"{requested_group}: {reason}"
        )


class DependencyGroupResolver:
    """
    A resolver for Dependency Group data.

    This class handles caching, name normalization, cycle detection, and other
    parsing requirements. There are only two public methods for exploring the data:
    ``lookup()`` and ``resolve()``.

    :param dependency_groups: A mapping, as provided via pyproject
        ``[dependency-groups]``.
    """

    def __init__(
        self,
        dependency_groups: Mapping[str, str | Mapping[str, str]],
    ) -> None:
        if not isinstance(dependency_groups, Mapping):
            raise TypeError("Dependency Groups table is not a mapping")
        self.dependency_groups = _normalize_group_names(dependency_groups)
        # a map of group names to parsed data
        self._parsed_groups: dict[
            str, tuple[Requirement | DependencyGroupInclude, ...]
        ] = {}
        # a map of group names to their ancestors, used for cycle detection
        self._include_graph_ancestors: dict[str, tuple[str, ...]] = {}
        # a cache of completed resolutions to Requirement lists
        self._resolve_cache: dict[str, tuple[Requirement, ...]] = {}

    def lookup(self, group: str) -> tuple[Requirement | DependencyGroupInclude, ...]:
        """
        Lookup a group name, returning the parsed dependency data for that group.
        This will not resolve includes.

        :param group: the name of the group to lookup

        :raises ValueError: if the data does not appear to be valid dependency group
            data
        :raises TypeError: if the data is not a string
        :raises LookupError: if group name is absent
        :raises packaging.requirements.InvalidRequirement: if a specifier is not valid
        """
        if not isinstance(group, str):
            raise TypeError("Dependency group name is not a str")
        group = _normalize_name(group)
        return self._parse_group(group)

    def resolve(self, group: str) -> tuple[Requirement, ...]:
        """
        Resolve a dependency group to a list of requirements.

        :param group: the name of the group to resolve

        :raises TypeError: if the inputs appear to be the wrong types
        :raises ValueError: if the data does not appear to be valid dependency group
            data
        :raises LookupError: if group name is absent
        :raises packaging.requirements.InvalidRequirement: if a specifier is not valid
        """
        if not isinstance(group, str):
            raise TypeError("Dependency group name is not a str")
        group = _normalize_name(group)
        return self._resolve(group, group)

    def _parse_group(
        self, group: str
    ) -> tuple[Requirement | DependencyGroupInclude, ...]:
        # short circuit -- never do the work twice
        if group in self._parsed_groups:
            return self._parsed_groups[group]

        if group not in self.dependency_groups:
            raise LookupError(f"Dependency group '{group}' not found")

        raw_group = self.dependency_groups[group]
        if not isinstance(raw_group, list):
            raise TypeError(f"Dependency group '{group}' is not a list")

        elements: list[Requirement | DependencyGroupInclude] = []
        for item in raw_group:
            if isinstance(item, str):
                # packaging.requirements.Requirement parsing ensures that this is a
                # valid PEP 508 Dependency Specifier
                # raises InvalidRequirement on failure
                elements.append(Requirement(item))
            elif isinstance(item, dict):
                if tuple(item.keys()) != ("include-group",):
                    raise ValueError(f"Invalid dependency group item: {item}")

                include_group = next(iter(item.values()))
                elements.append(DependencyGroupInclude(include_group=include_group))
            else:
                raise ValueError(f"Invalid dependency group item: {item}")

        self._parsed_groups[group] = tuple(elements)
        return self._parsed_groups[group]

    def _resolve(self, group: str, requested_group: str) -> tuple[Requirement, ...]:
        """
        This is a helper for cached resolution to strings.

        :param group: The name of the group to resolve.
        :param requested_group: The group which was used in the original, user-facing
            request.
        """
        if group in self._resolve_cache:
            return self._resolve_cache[group]

        parsed = self._parse_group(group)

        resolved_group = []
        for item in parsed:
            if isinstance(item, Requirement):
                resolved_group.append(item)
            elif isinstance(item, DependencyGroupInclude):
                include_group = _normalize_name(item.include_group)
                if include_group in self._include_graph_ancestors.get(group, ()):
                    raise CyclicDependencyError(
                        requested_group, group, item.include_group
                    )
                self._include_graph_ancestors[include_group] = (
                    *self._include_graph_ancestors.get(group, ()),
                    group,
                )
                resolved_group.extend(self._resolve(include_group, requested_group))
            else:  # unreachable
                raise NotImplementedError(
                    f"Invalid dependency group item after parse: {item}"
                )

        self._resolve_cache[group] = tuple(resolved_group)
        return self._resolve_cache[group]


def resolve(
    dependency_groups: Mapping[str, str | Mapping[str, str]], /, *groups: str
) -> tuple[str, ...]:
    """
    Resolve a dependency group to a tuple of requirements, as strings.

    :param dependency_groups: the parsed contents of the ``[dependency-groups]`` table
        from ``pyproject.toml``
    :param groups: the name of the group(s) to resolve

    :raises TypeError: if the inputs appear to be the wrong types
    :raises ValueError: if the data does not appear to be valid dependency group data
    :raises LookupError: if group name is absent
    :raises packaging.requirements.InvalidRequirement: if a specifier is not valid
    """
    resolver = DependencyGroupResolver(dependency_groups)
    return tuple(str(r) for group in groups for r in resolver.resolve(group))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\resolvelib\structs.py ===
from __future__ import annotations

import itertools
from collections import namedtuple
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Sequence,
    TypeVar,
    Union,
)

KT = TypeVar("KT")  # Identifier.
RT = TypeVar("RT")  # Requirement.
CT = TypeVar("CT")  # Candidate.

Matches = Union[Iterable[CT], Callable[[], Iterable[CT]]]

if TYPE_CHECKING:
    from .resolvers.criterion import Criterion

    class RequirementInformation(NamedTuple, Generic[RT, CT]):
        requirement: RT
        parent: CT | None

    class State(NamedTuple, Generic[RT, CT, KT]):
        """Resolution state in a round."""

        mapping: dict[KT, CT]
        criteria: dict[KT, Criterion[RT, CT]]
        backtrack_causes: list[RequirementInformation[RT, CT]]

else:
    RequirementInformation = namedtuple(
        "RequirementInformation", ["requirement", "parent"]
    )
    State = namedtuple("State", ["mapping", "criteria", "backtrack_causes"])


class DirectedGraph(Generic[KT]):
    """A graph structure with directed edges."""

    def __init__(self) -> None:
        self._vertices: set[KT] = set()
        self._forwards: dict[KT, set[KT]] = {}  # <key> -> Set[<key>]
        self._backwards: dict[KT, set[KT]] = {}  # <key> -> Set[<key>]

    def __iter__(self) -> Iterator[KT]:
        return iter(self._vertices)

    def __len__(self) -> int:
        return len(self._vertices)

    def __contains__(self, key: KT) -> bool:
        return key in self._vertices

    def copy(self) -> DirectedGraph[KT]:
        """Return a shallow copy of this graph."""
        other = type(self)()
        other._vertices = set(self._vertices)
        other._forwards = {k: set(v) for k, v in self._forwards.items()}
        other._backwards = {k: set(v) for k, v in self._backwards.items()}
        return other

    def add(self, key: KT) -> None:
        """Add a new vertex to the graph."""
        if key in self._vertices:
            raise ValueError("vertex exists")
        self._vertices.add(key)
        self._forwards[key] = set()
        self._backwards[key] = set()

    def remove(self, key: KT) -> None:
        """Remove a vertex from the graph, disconnecting all edges from/to it."""
        self._vertices.remove(key)
        for f in self._forwards.pop(key):
            self._backwards[f].remove(key)
        for t in self._backwards.pop(key):
            self._forwards[t].remove(key)

    def connected(self, f: KT, t: KT) -> bool:
        return f in self._backwards[t] and t in self._forwards[f]

    def connect(self, f: KT, t: KT) -> None:
        """Connect two existing vertices.

        Nothing happens if the vertices are already connected.
        """
        if t not in self._vertices:
            raise KeyError(t)
        self._forwards[f].add(t)
        self._backwards[t].add(f)

    def iter_edges(self) -> Iterator[tuple[KT, KT]]:
        for f, children in self._forwards.items():
            for t in children:
                yield f, t

    def iter_children(self, key: KT) -> Iterator[KT]:
        return iter(self._forwards[key])

    def iter_parents(self, key: KT) -> Iterator[KT]:
        return iter(self._backwards[key])


class IteratorMapping(Mapping[KT, Iterator[CT]], Generic[RT, CT, KT]):
    def __init__(
        self,
        mapping: Mapping[KT, RT],
        accessor: Callable[[RT], Iterable[CT]],
        appends: Mapping[KT, Iterable[CT]] | None = None,
    ) -> None:
        self._mapping = mapping
        self._accessor = accessor
        self._appends: Mapping[KT, Iterable[CT]] = appends or {}

    def __repr__(self) -> str:
        return "IteratorMapping({!r}, {!r}, {!r})".format(
            self._mapping,
            self._accessor,
            self._appends,
        )

    def __bool__(self) -> bool:
        return bool(self._mapping or self._appends)

    def __contains__(self, key: object) -> bool:
        return key in self._mapping or key in self._appends

    def __getitem__(self, k: KT) -> Iterator[CT]:
        try:
            v = self._mapping[k]
        except KeyError:
            return iter(self._appends[k])
        return itertools.chain(self._accessor(v), self._appends.get(k, ()))

    def __iter__(self) -> Iterator[KT]:
        more = (k for k in self._appends if k not in self._mapping)
        return itertools.chain(self._mapping, more)

    def __len__(self) -> int:
        more = sum(1 for k in self._appends if k not in self._mapping)
        return len(self._mapping) + more


class _FactoryIterableView(Iterable[RT]):
    """Wrap an iterator factory returned by `find_matches()`.

    Calling `iter()` on this class would invoke the underlying iterator
    factory, making it a "collection with ordering" that can be iterated
    through multiple times, but lacks random access methods presented in
    built-in Python sequence types.
    """

    def __init__(self, factory: Callable[[], Iterable[RT]]) -> None:
        self._factory = factory
        self._iterable: Iterable[RT] | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self)})"

    def __bool__(self) -> bool:
        try:
            next(iter(self))
        except StopIteration:
            return False
        return True

    def __iter__(self) -> Iterator[RT]:
        iterable = self._factory() if self._iterable is None else self._iterable
        self._iterable, current = itertools.tee(iterable)
        return current


class _SequenceIterableView(Iterable[RT]):
    """Wrap an iterable returned by find_matches().

    This is essentially just a proxy to the underlying sequence that provides
    the same interface as `_FactoryIterableView`.
    """

    def __init__(self, sequence: Sequence[RT]):
        self._sequence = sequence

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._sequence})"

    def __bool__(self) -> bool:
        return bool(self._sequence)

    def __iter__(self) -> Iterator[RT]:
        return iter(self._sequence)


def build_iter_view(matches: Matches[CT]) -> Iterable[CT]:
    """Build an iterable view from the value returned by `find_matches()`."""
    if callable(matches):
        return _FactoryIterableView(matches)
    if not isinstance(matches, Sequence):
        matches = list(matches)
    return _SequenceIterableView(matches)


IterableView = Iterable

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\particle.py ===
from sympy import S
from sympy.physics.vector import cross, dot
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.inertia import inertia_of_point_mass
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Particle']


class Particle(BodyBase):
    """A particle.

    Explanation
    ===========

    Particles have a non-zero mass and lack spatial extension; they take up no
    space.

    Values need to be supplied on initialization, but can be changed later.

    Parameters
    ==========

    name : str
        Name of particle
    point : Point
        A physics/mechanics Point which represents the position, velocity, and
        acceleration of this Particle
    mass : Sympifyable
        A SymPy expression representing the Particle's mass
    potential_energy : Sympifyable
        The potential energy of the Particle.

    Examples
    ========

    >>> from sympy.physics.mechanics import Particle, Point
    >>> from sympy import Symbol
    >>> po = Point('po')
    >>> m = Symbol('m')
    >>> pa = Particle('pa', po, m)
    >>> # Or you could change these later
    >>> pa.mass = m
    >>> pa.point = po

    """
    point = BodyBase.masscenter

    def __init__(self, name, point=None, mass=None):
        super().__init__(name, point, mass)

    def linear_momentum(self, frame):
        """Linear momentum of the particle.

        Explanation
        ===========

        The linear momentum L, of a particle P, with respect to frame N is
        given by:

        L = m * v

        where m is the mass of the particle, and v is the velocity of the
        particle in the frame N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v = dynamicsymbols('m v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> A = Particle('A', P, m)
        >>> P.set_vel(N, v * N.x)
        >>> A.linear_momentum(N)
        m*v*N.x

        """

        return self.mass * self.point.vel(frame)

    def angular_momentum(self, point, frame):
        """Angular momentum of the particle about the point.

        Explanation
        ===========

        The angular momentum H, about some point O of a particle, P, is given
        by:

        ``H = cross(r, m * v)``

        where r is the position vector from point O to the particle P, m is
        the mass of the particle, and v is the velocity of the particle in
        the inertial frame, N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum of the particle is desired.

        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r = dynamicsymbols('m v r')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> A = O.locatenew('A', r * N.x)
        >>> P = Particle('P', A, m)
        >>> P.point.set_vel(N, v * N.y)
        >>> P.angular_momentum(O, N)
        m*r*v*N.z

        """

        return cross(self.point.pos_from(point),
                     self.mass * self.point.vel(frame))

    def kinetic_energy(self, frame):
        """Kinetic energy of the particle.

        Explanation
        ===========

        The kinetic energy, T, of a particle, P, is given by:

        ``T = 1/2 (dot(m * v, v))``

        where m is the mass of particle P, and v is the velocity of the
        particle in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The Particle's velocity is typically defined with respect to
            an inertial frame but any relevant frame in which the velocity is
            known can be supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy import symbols
        >>> m, v, r = symbols('m v r')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> P = Particle('P', O, m)
        >>> P.point.set_vel(N, v * N.y)
        >>> P.kinetic_energy(N)
        m*v**2/2

        """

        return S.Half * self.mass * dot(self.point.vel(frame),
                                        self.point.vel(frame))

    def set_potential_energy(self, scalar):
        sympy_deprecation_warning(
            """
The sympy.physics.mechanics.Particle.set_potential_energy()
method is deprecated. Instead use

    P.potential_energy = scalar
            """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-set-potential-energy",
        )
        self.potential_energy = scalar

    def parallel_axis(self, point, frame):
        """Returns an inertia dyadic of the particle with respect to another
        point and frame.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the particle expressed about the provided
            point and frame.

        """
        return inertia_of_point_mass(self.mass, self.point.pos_from(point),
                                     frame)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\plot_modes.py ===
from sympy.utilities.lambdify import lambdify
from sympy.core.numbers import pi
from sympy.functions import sin, cos
from sympy.plotting.pygletplot.plot_curve import PlotCurve
from sympy.plotting.pygletplot.plot_surface import PlotSurface

from math import sin as p_sin
from math import cos as p_cos


def float_vec3(f):
    def inner(*args):
        v = f(*args)
        return float(v[0]), float(v[1]), float(v[2])
    return inner


class Cartesian2D(PlotCurve):
    i_vars, d_vars = 'x', 'y'
    intervals = [[-5, 5, 100]]
    aliases = ['cartesian']
    is_default = True

    def _get_sympy_evaluator(self):
        fy = self.d_vars[0]
        x = self.t_interval.v

        @float_vec3
        def e(_x):
            return (_x, fy.subs(x, _x), 0.0)
        return e

    def _get_lambda_evaluator(self):
        fy = self.d_vars[0]
        x = self.t_interval.v
        return lambdify([x], [x, fy, 0.0])


class Cartesian3D(PlotSurface):
    i_vars, d_vars = 'xy', 'z'
    intervals = [[-1, 1, 40], [-1, 1, 40]]
    aliases = ['cartesian', 'monge']
    is_default = True

    def _get_sympy_evaluator(self):
        fz = self.d_vars[0]
        x = self.u_interval.v
        y = self.v_interval.v

        @float_vec3
        def e(_x, _y):
            return (_x, _y, fz.subs(x, _x).subs(y, _y))
        return e

    def _get_lambda_evaluator(self):
        fz = self.d_vars[0]
        x = self.u_interval.v
        y = self.v_interval.v
        return lambdify([x, y], [x, y, fz])


class ParametricCurve2D(PlotCurve):
    i_vars, d_vars = 't', 'xy'
    intervals = [[0, 2*pi, 100]]
    aliases = ['parametric']
    is_default = True

    def _get_sympy_evaluator(self):
        fx, fy = self.d_vars
        t = self.t_interval.v

        @float_vec3
        def e(_t):
            return (fx.subs(t, _t), fy.subs(t, _t), 0.0)
        return e

    def _get_lambda_evaluator(self):
        fx, fy = self.d_vars
        t = self.t_interval.v
        return lambdify([t], [fx, fy, 0.0])


class ParametricCurve3D(PlotCurve):
    i_vars, d_vars = 't', 'xyz'
    intervals = [[0, 2*pi, 100]]
    aliases = ['parametric']
    is_default = True

    def _get_sympy_evaluator(self):
        fx, fy, fz = self.d_vars
        t = self.t_interval.v

        @float_vec3
        def e(_t):
            return (fx.subs(t, _t), fy.subs(t, _t), fz.subs(t, _t))
        return e

    def _get_lambda_evaluator(self):
        fx, fy, fz = self.d_vars
        t = self.t_interval.v
        return lambdify([t], [fx, fy, fz])


class ParametricSurface(PlotSurface):
    i_vars, d_vars = 'uv', 'xyz'
    intervals = [[-1, 1, 40], [-1, 1, 40]]
    aliases = ['parametric']
    is_default = True

    def _get_sympy_evaluator(self):
        fx, fy, fz = self.d_vars
        u = self.u_interval.v
        v = self.v_interval.v

        @float_vec3
        def e(_u, _v):
            return (fx.subs(u, _u).subs(v, _v),
                    fy.subs(u, _u).subs(v, _v),
                    fz.subs(u, _u).subs(v, _v))
        return e

    def _get_lambda_evaluator(self):
        fx, fy, fz = self.d_vars
        u = self.u_interval.v
        v = self.v_interval.v
        return lambdify([u, v], [fx, fy, fz])


class Polar(PlotCurve):
    i_vars, d_vars = 't', 'r'
    intervals = [[0, 2*pi, 100]]
    aliases = ['polar']
    is_default = False

    def _get_sympy_evaluator(self):
        fr = self.d_vars[0]
        t = self.t_interval.v

        def e(_t):
            _r = float(fr.subs(t, _t))
            return (_r*p_cos(_t), _r*p_sin(_t), 0.0)
        return e

    def _get_lambda_evaluator(self):
        fr = self.d_vars[0]
        t = self.t_interval.v
        fx, fy = fr*cos(t), fr*sin(t)
        return lambdify([t], [fx, fy, 0.0])


class Cylindrical(PlotSurface):
    i_vars, d_vars = 'th', 'r'
    intervals = [[0, 2*pi, 40], [-1, 1, 20]]
    aliases = ['cylindrical', 'polar']
    is_default = False

    def _get_sympy_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        h = self.v_interval.v

        def e(_t, _h):
            _r = float(fr.subs(t, _t).subs(h, _h))
            return (_r*p_cos(_t), _r*p_sin(_t), _h)
        return e

    def _get_lambda_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        h = self.v_interval.v
        fx, fy = fr*cos(t), fr*sin(t)
        return lambdify([t, h], [fx, fy, h])


class Spherical(PlotSurface):
    i_vars, d_vars = 'tp', 'r'
    intervals = [[0, 2*pi, 40], [0, pi, 20]]
    aliases = ['spherical']
    is_default = False

    def _get_sympy_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        p = self.v_interval.v

        def e(_t, _p):
            _r = float(fr.subs(t, _t).subs(p, _p))
            return (_r*p_cos(_t)*p_sin(_p),
                    _r*p_sin(_t)*p_sin(_p),
                    _r*p_cos(_p))
        return e

    def _get_lambda_evaluator(self):
        fr = self.d_vars[0]
        t = self.u_interval.v
        p = self.v_interval.v
        fx = fr * cos(t) * sin(p)
        fy = fr * sin(t) * sin(p)
        fz = fr * cos(p)
        return lambdify([t, p], [fx, fy, fz])

Cartesian2D._register()
Cartesian3D._register()
ParametricCurve2D._register()
ParametricCurve3D._register()
ParametricSurface._register()
Polar._register()
Cylindrical._register()
Spherical._register()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\datastructures\file_storage.py ===
from __future__ import annotations

import collections.abc as cabc
import mimetypes
import os
import typing as t
from io import BytesIO
from os import fsdecode
from os import fspath

from .._internal import _plain_int
from .headers import Headers
from .structures import MultiDict


class FileStorage:
    """The :class:`FileStorage` class is a thin wrapper over incoming files.
    It is used by the request object to represent uploaded files.  All the
    attributes of the wrapper stream are proxied by the file storage so
    it's possible to do ``storage.read()`` instead of the long form
    ``storage.stream.read()``.
    """

    def __init__(
        self,
        stream: t.IO[bytes] | None = None,
        filename: str | None = None,
        name: str | None = None,
        content_type: str | None = None,
        content_length: int | None = None,
        headers: Headers | None = None,
    ):
        self.name = name
        self.stream = stream or BytesIO()

        # If no filename is provided, attempt to get the filename from
        # the stream object. Python names special streams like
        # ``<stderr>`` with angular brackets, skip these streams.
        if filename is None:
            filename = getattr(stream, "name", None)

            if filename is not None:
                filename = fsdecode(filename)

            if filename and filename[0] == "<" and filename[-1] == ">":
                filename = None
        else:
            filename = fsdecode(filename)

        self.filename = filename

        if headers is None:
            headers = Headers()
        self.headers = headers
        if content_type is not None:
            headers["Content-Type"] = content_type
        if content_length is not None:
            headers["Content-Length"] = str(content_length)

    def _parse_content_type(self) -> None:
        if not hasattr(self, "_parsed_content_type"):
            self._parsed_content_type = http.parse_options_header(self.content_type)

    @property
    def content_type(self) -> str | None:
        """The content-type sent in the header.  Usually not available"""
        return self.headers.get("content-type")

    @property
    def content_length(self) -> int:
        """The content-length sent in the header.  Usually not available"""
        if "content-length" in self.headers:
            try:
                return _plain_int(self.headers["content-length"])
            except ValueError:
                pass

        return 0

    @property
    def mimetype(self) -> str:
        """Like :attr:`content_type`, but without parameters (eg, without
        charset, type etc.) and always lowercase.  For example if the content
        type is ``text/HTML; charset=utf-8`` the mimetype would be
        ``'text/html'``.

        .. versionadded:: 0.7
        """
        self._parse_content_type()
        return self._parsed_content_type[0].lower()

    @property
    def mimetype_params(self) -> dict[str, str]:
        """The mimetype parameters as dict.  For example if the content
        type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.7
        """
        self._parse_content_type()
        return self._parsed_content_type[1]

    def save(
        self, dst: str | os.PathLike[str] | t.IO[bytes], buffer_size: int = 16384
    ) -> None:
        """Save the file to a destination path or file object.  If the
        destination is a file object you have to close it yourself after the
        call.  The buffer size is the number of bytes held in memory during
        the copy process.  It defaults to 16KB.

        For secure file saving also have a look at :func:`secure_filename`.

        :param dst: a filename, :class:`os.PathLike`, or open file
            object to write to.
        :param buffer_size: Passed as the ``length`` parameter of
            :func:`shutil.copyfileobj`.

        .. versionchanged:: 1.0
            Supports :mod:`pathlib`.
        """
        from shutil import copyfileobj

        close_dst = False

        if hasattr(dst, "__fspath__"):
            dst = fspath(dst)

        if isinstance(dst, str):
            dst = open(dst, "wb")
            close_dst = True

        try:
            copyfileobj(self.stream, dst, buffer_size)
        finally:
            if close_dst:
                dst.close()

    def close(self) -> None:
        """Close the underlying file if possible."""
        try:
            self.stream.close()
        except Exception:
            pass

    def __bool__(self) -> bool:
        return bool(self.filename)

    def __getattr__(self, name: str) -> t.Any:
        try:
            return getattr(self.stream, name)
        except AttributeError:
            # SpooledTemporaryFile on Python < 3.11 doesn't implement IOBase,
            # get the attribute from its backing file instead.
            if hasattr(self.stream, "_file"):
                return getattr(self.stream._file, name)
            raise

    def __iter__(self) -> cabc.Iterator[bytes]:
        return iter(self.stream)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.filename!r} ({self.content_type!r})>"


class FileMultiDict(MultiDict[str, FileStorage]):
    """A special :class:`MultiDict` that has convenience methods to add
    files to it.  This is used for :class:`EnvironBuilder` and generally
    useful for unittesting.

    .. versionadded:: 0.5
    """

    def add_file(
        self,
        name: str,
        file: str | os.PathLike[str] | t.IO[bytes] | FileStorage,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> None:
        """Adds a new file to the dict.  `file` can be a file name or
        a :class:`file`-like or a :class:`FileStorage` object.

        :param name: the name of the field.
        :param file: a filename or :class:`file`-like object
        :param filename: an optional filename
        :param content_type: an optional content type
        """
        if isinstance(file, FileStorage):
            self.add(name, file)
            return

        if isinstance(file, (str, os.PathLike)):
            if filename is None:
                filename = os.fspath(file)

            file_obj: t.IO[bytes] = open(file, "rb")
        else:
            file_obj = file  # type: ignore[assignment]

        if filename and content_type is None:
            content_type = (
                mimetypes.guess_type(filename)[0] or "application/octet-stream"
            )

        self.add(name, FileStorage(file_obj, filename, name, content_type))


# circular dependencies
from .. import http

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\sam2\mask_decoder.py ===
# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import warnings

import torch
from image_encoder import SAM2ImageEncoder, random_sam2_input_image
from prompt_encoder import SAM2PromptEncoder
from sam2.modeling.sam2_base import SAM2Base
from torch import nn

logger = logging.getLogger(__name__)


class SAM2MaskDecoder(nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        multimask_output: bool,
        dynamic_multimask_via_stability: bool = True,
    ) -> None:
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.multimask_output = multimask_output
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability

    @torch.no_grad()
    def forward(
        self,
        image_features_0: torch.Tensor,
        image_features_1: torch.Tensor,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
    ):
        """
        Decode masks from image and prompt embeddings. Only support H=W=1024.

        Args:
            image_features_0 (torch.Tensor): [1, 32, H/4, W/4]. high resolution features of level 0 from image encoder.
            image_features_1 (torch.Tensor): [1, 64, H/8, W/8]. high resolution features of level 1 from image encoder.
            image_embeddings (torch.Tensor): [1, 256, H/16, W/16]. image embedding from image encoder.
            image_pe (torch.Tensor): [1, 256, H/16, W/16]. image positional encoding.
            sparse_embeddings (torch.Tensor): [L, P+1, 256], embedding for points and boxes.
            dense_embeddings (torch.Tensor):  [L, 256, H/16, W/16]. embedding for input masks.

        Returns:
            low_res_masks (torch.Tensor, optional): [1, M, H/4, W/4]. low resolution masks.
            iou_predictions (torch.Tensor): [1, M]. scores for M masks.
        """
        low_res_masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            repeat_image=sparse_embeddings.shape[0] > 1,  # batch mode
            high_res_features=[image_features_0, image_features_1],
        )

        if self.multimask_output:
            low_res_masks = low_res_masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        elif self.dynamic_multimask_via_stability:
            # When outputting a single mask, if the stability score from the current single-mask
            # output (based on output token 0) falls below a threshold, we instead select from
            # multi-mask outputs (based on output token 1~3) the mask with the highest predicted IoU score.
            low_res_masks, iou_predictions = self.mask_decoder._dynamic_multimask_via_stability(
                low_res_masks, iou_predictions
            )
        else:
            low_res_masks = low_res_masks[:, 0:1, :, :]
            iou_predictions = iou_predictions[:, 0:1]

        return low_res_masks, iou_predictions


def export_mask_decoder_onnx(
    sam2_model: SAM2Base,
    onnx_model_path: str,
    multimask_output: bool,
    dynamic_multimask_via_stability: bool = True,
    verbose=False,
):
    sam2_prompt_encoder = SAM2PromptEncoder(sam2_model).cpu()

    image = random_sam2_input_image()
    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()
    image_features_0, image_features_1, image_embeddings = sam2_encoder(image)
    logger.info("image_features_0.shape: %s", image_features_0.shape)
    logger.info("image_features_1.shape: %s", image_features_1.shape)
    logger.info("image_embeddings.shape: %s", image_embeddings.shape)

    # encode an random prompt
    num_labels = 2
    num_points = 3
    point_coords = torch.randint(low=0, high=1024, size=(num_labels, num_points, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(num_labels, num_points), dtype=torch.float)
    input_masks = torch.zeros(num_labels, 1, 256, 256, dtype=torch.float)
    has_input_masks = torch.ones(1, dtype=torch.float)

    sparse_embeddings, dense_embeddings, image_pe = sam2_prompt_encoder(
        point_coords, point_labels, input_masks, has_input_masks
    )

    logger.info("sparse_embeddings.shape: %s", sparse_embeddings.shape)
    logger.info("dense_embeddings.shape: %s", dense_embeddings.shape)
    logger.info("image_pe.shape: %s", image_pe.shape)

    sam2_mask_decoder = SAM2MaskDecoder(sam2_model, multimask_output, dynamic_multimask_via_stability)
    inputs = (image_features_0, image_features_1, image_embeddings, image_pe, sparse_embeddings, dense_embeddings)
    low_res_masks, iou_predictions = sam2_mask_decoder(*inputs)
    logger.info("low_res_masks.shape: %s", low_res_masks.shape)
    logger.info("iou_predictions.shape: %s", iou_predictions.shape)

    with warnings.catch_warnings():
        if not verbose:
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            sam2_mask_decoder,
            inputs,
            onnx_model_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=[
                "image_features_0",
                "image_features_1",
                "image_embeddings",
                "image_pe",
                "sparse_embeddings",
                "dense_embeddings",
            ],
            output_names=["low_res_masks", "iou_predictions"],
            dynamic_axes={
                "sparse_embeddings": {0: "num_labels", 1: "num_points+1"},
                "dense_embeddings": {0: "num_labels"},
                "low_res_masks": {0: "num_labels"},
                "iou_predictions": {0: "num_labels"},
            },
        )

    print("mask decoder onnx model saved to", onnx_model_path)


def test_mask_decoder_onnx(
    sam2_model: SAM2Base,
    onnx_model_path: str,
    multimask_output: bool,
    dynamic_multimask_via_stability: bool,
):
    sam2_prompt_encoder = SAM2PromptEncoder(sam2_model).cpu()

    image = random_sam2_input_image()
    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()
    image_features_0, image_features_1, image_embeddings = sam2_encoder(image)

    num_labels = 1
    num_points = 5
    point_coords = torch.randint(low=0, high=1024, size=(num_labels, num_points, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(num_labels, num_points), dtype=torch.float)
    input_masks = torch.rand(num_labels, 1, 256, 256, dtype=torch.float)
    has_input_masks = torch.ones(1, dtype=torch.float)

    sparse_embeddings, dense_embeddings, image_pe = sam2_prompt_encoder(
        point_coords, point_labels, input_masks, has_input_masks
    )

    sam2_mask_decoder = SAM2MaskDecoder(sam2_model, multimask_output, dynamic_multimask_via_stability)
    inputs = (image_features_0, image_features_1, image_embeddings, image_pe, sparse_embeddings, dense_embeddings)
    low_res_masks, iou_predictions = sam2_mask_decoder(*inputs)

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    logger.info("input_names: %s", input_names)

    model_outputs = ort_session.get_outputs()
    output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    logger.info("output_names: %s", output_names)

    outputs = ort_session.run(
        output_names,
        {
            "image_features_0": image_features_0.numpy(),
            "image_features_1": image_features_1.numpy(),
            "image_embeddings": image_embeddings.numpy(),
            "image_pe": image_pe.numpy(),
            "sparse_embeddings": sparse_embeddings.numpy(),
            "dense_embeddings": dense_embeddings.numpy(),
        },
    )

    for i, output_name in enumerate(output_names):
        logger.info("output %s shape: %s", output_name, outputs[i].shape)

    ort_low_res_masks, ort_iou_predictions = outputs
    torch.testing.assert_close(low_res_masks, torch.tensor(ort_low_res_masks), atol=5e-3, rtol=1e-4)
    torch.testing.assert_close(iou_predictions, torch.tensor(ort_iou_predictions), atol=5e-3, rtol=1e-4)
    print(f"onnx model has been verified: {onnx_model_path}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\testing\print_coercion_tables.py ===
#!/usr/bin/env python3
"""Prints type-coercion tables for the built-in NumPy types

"""
from collections import namedtuple

import numpy as np
from numpy._core.numerictypes import obj2sctype


# Generic object that can be added, but doesn't do anything else
class GenericObject:
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    dtype = np.dtype('O')

def print_cancast_table(ntypes):
    print('X', end=' ')
    for char in ntypes:
        print(char, end=' ')
    print()
    for row in ntypes:
        print(row, end=' ')
        for col in ntypes:
            if np.can_cast(row, col, "equiv"):
                cast = "#"
            elif np.can_cast(row, col, "safe"):
                cast = "="
            elif np.can_cast(row, col, "same_kind"):
                cast = "~"
            elif np.can_cast(row, col, "unsafe"):
                cast = "."
            else:
                cast = " "
            print(cast, end=' ')
        print()

def print_coercion_table(ntypes, inputfirstvalue, inputsecondvalue, firstarray,
                         use_promote_types=False):
    print('+', end=' ')
    for char in ntypes:
        print(char, end=' ')
    print()
    for row in ntypes:
        if row == 'O':
            rowtype = GenericObject
        else:
            rowtype = obj2sctype(row)

        print(row, end=' ')
        for col in ntypes:
            if col == 'O':
                coltype = GenericObject
            else:
                coltype = obj2sctype(col)
            try:
                if firstarray:
                    rowvalue = np.array([rowtype(inputfirstvalue)], dtype=rowtype)
                else:
                    rowvalue = rowtype(inputfirstvalue)
                colvalue = coltype(inputsecondvalue)
                if use_promote_types:
                    char = np.promote_types(rowvalue.dtype, colvalue.dtype).char
                else:
                    value = np.add(rowvalue, colvalue)
                    if isinstance(value, np.ndarray):
                        char = value.dtype.char
                    else:
                        char = np.dtype(type(value)).char
            except ValueError:
                char = '!'
            except OverflowError:
                char = '@'
            except TypeError:
                char = '#'
            print(char, end=' ')
        print()


def print_new_cast_table(*, can_cast=True, legacy=False, flags=False):
    """Prints new casts, the values given are default "can-cast" values, not
    actual ones.
    """
    from numpy._core._multiarray_tests import get_all_cast_information

    cast_table = {
        -1: " ",
        0: "#",  # No cast (classify as equivalent here)
        1: "#",  # equivalent casting
        2: "=",  # safe casting
        3: "~",  # same-kind casting
        4: ".",  # unsafe casting
    }
    flags_table = {
        0: "▗", 7: "█",
        1: "▚", 2: "▐", 4: "▄",
                3: "▜", 5: "▙",
                        6: "▟",
    }

    cast_info = namedtuple("cast_info", ["can_cast", "legacy", "flags"])
    no_cast_info = cast_info(" ", " ", " ")

    casts = get_all_cast_information()
    table = {}
    dtypes = set()
    for cast in casts:
        dtypes.add(cast["from"])
        dtypes.add(cast["to"])

        if cast["from"] not in table:
            table[cast["from"]] = {}
        to_dict = table[cast["from"]]

        can_cast = cast_table[cast["casting"]]
        legacy = "L" if cast["legacy"] else "."
        flags = 0
        if cast["requires_pyapi"]:
            flags |= 1
        if cast["supports_unaligned"]:
            flags |= 2
        if cast["no_floatingpoint_errors"]:
            flags |= 4

        flags = flags_table[flags]
        to_dict[cast["to"]] = cast_info(can_cast=can_cast, legacy=legacy, flags=flags)

    # The np.dtype(x.type) is a bit strange, because dtype classes do
    # not expose much yet.
    types = np.typecodes["All"]

    def sorter(x):
        # This is a bit weird hack, to get a table as close as possible to
        # the one printing all typecodes (but expecting user-dtypes).
        dtype = np.dtype(x.type)
        try:
            indx = types.index(dtype.char)
        except ValueError:
            indx = np.inf
        return (indx, dtype.char)

    dtypes = sorted(dtypes, key=sorter)

    def print_table(field="can_cast"):
        print('X', end=' ')
        for dt in dtypes:
            print(np.dtype(dt.type).char, end=' ')
        print()
        for from_dt in dtypes:
            print(np.dtype(from_dt.type).char, end=' ')
            row = table.get(from_dt, {})
            for to_dt in dtypes:
                print(getattr(row.get(to_dt, no_cast_info), field), end=' ')
            print()

    if can_cast:
        # Print the actual table:
        print()
        print("Casting: # is equivalent, = is safe, ~ is same-kind, and . is unsafe")
        print()
        print_table("can_cast")

    if legacy:
        print()
        print("L denotes a legacy cast . a non-legacy one.")
        print()
        print_table("legacy")

    if flags:
        print()
        print(f"{flags_table[0]}: no flags, "
              f"{flags_table[1]}: PyAPI, "
              f"{flags_table[2]}: supports unaligned, "
              f"{flags_table[4]}: no-float-errors")
        print()
        print_table("flags")


if __name__ == '__main__':
    print("can cast")
    print_cancast_table(np.typecodes['All'])
    print()
    print("In these tables, ValueError is '!', OverflowError is '@', TypeError is '#'")
    print()
    print("scalar + scalar")
    print_coercion_table(np.typecodes['All'], 0, 0, False)
    print()
    print("scalar + neg scalar")
    print_coercion_table(np.typecodes['All'], 0, -1, False)
    print()
    print("array + scalar")
    print_coercion_table(np.typecodes['All'], 0, 0, True)
    print()
    print("array + neg scalar")
    print_coercion_table(np.typecodes['All'], 0, -1, True)
    print()
    print("promote_types")
    print_coercion_table(np.typecodes['All'], 0, 0, False, True)
    print("New casting type promotion:")
    print_new_cast_table(can_cast=True, legacy=True, flags=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\compat\__init__.py ===
"""
compat
======

Cross-compatible functions for different versions of Python.

Other items:
* platform checker
"""
from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING

from pandas.compat._constants import (
    IS64,
    ISMUSL,
    PY310,
    PY311,
    PY312,
    PYPY,
)
import pandas.compat.compressors
from pandas.compat.numpy import is_numpy_dev
from pandas.compat.pyarrow import (
    HAS_PYARROW,
    pa_version_under10p1,
    pa_version_under11p0,
    pa_version_under13p0,
    pa_version_under14p0,
    pa_version_under14p1,
    pa_version_under16p0,
    pa_version_under17p0,
    pa_version_under18p0,
    pa_version_under19p0,
    pa_version_under20p0,
)

if TYPE_CHECKING:
    from pandas._typing import F


def set_function_name(f: F, name: str, cls: type) -> F:
    """
    Bind the name/qualname attributes of the function.
    """
    f.__name__ = name
    f.__qualname__ = f"{cls.__name__}.{name}"
    f.__module__ = cls.__module__
    return f


def is_platform_little_endian() -> bool:
    """
    Checking if the running platform is little endian.

    Returns
    -------
    bool
        True if the running platform is little endian.
    """
    return sys.byteorder == "little"


def is_platform_windows() -> bool:
    """
    Checking if the running platform is windows.

    Returns
    -------
    bool
        True if the running platform is windows.
    """
    return sys.platform in ["win32", "cygwin"]


def is_platform_linux() -> bool:
    """
    Checking if the running platform is linux.

    Returns
    -------
    bool
        True if the running platform is linux.
    """
    return sys.platform == "linux"


def is_platform_mac() -> bool:
    """
    Checking if the running platform is mac.

    Returns
    -------
    bool
        True if the running platform is mac.
    """
    return sys.platform == "darwin"


def is_platform_arm() -> bool:
    """
    Checking if the running platform use ARM architecture.

    Returns
    -------
    bool
        True if the running platform uses ARM architecture.
    """
    return platform.machine() in ("arm64", "aarch64") or platform.machine().startswith(
        "armv"
    )


def is_platform_power() -> bool:
    """
    Checking if the running platform use Power architecture.

    Returns
    -------
    bool
        True if the running platform uses ARM architecture.
    """
    return platform.machine() in ("ppc64", "ppc64le")


def is_ci_environment() -> bool:
    """
    Checking if running in a continuous integration environment by checking
    the PANDAS_CI environment variable.

    Returns
    -------
    bool
        True if the running in a continuous integration environment.
    """
    return os.environ.get("PANDAS_CI", "0") == "1"


def get_lzma_file() -> type[pandas.compat.compressors.LZMAFile]:
    """
    Importing the `LZMAFile` class from the `lzma` module.

    Returns
    -------
    class
        The `LZMAFile` class from the `lzma` module.

    Raises
    ------
    RuntimeError
        If the `lzma` module was not imported correctly, or didn't exist.
    """
    if not pandas.compat.compressors.has_lzma:
        raise RuntimeError(
            "lzma module not available. "
            "A Python re-install with the proper dependencies, "
            "might be required to solve this issue."
        )
    return pandas.compat.compressors.LZMAFile


def get_bz2_file() -> type[pandas.compat.compressors.BZ2File]:
    """
    Importing the `BZ2File` class from the `bz2` module.

    Returns
    -------
    class
        The `BZ2File` class from the `bz2` module.

    Raises
    ------
    RuntimeError
        If the `bz2` module was not imported correctly, or didn't exist.
    """
    if not pandas.compat.compressors.has_bz2:
        raise RuntimeError(
            "bz2 module not available. "
            "A Python re-install with the proper dependencies, "
            "might be required to solve this issue."
        )
    return pandas.compat.compressors.BZ2File


__all__ = [
    "is_numpy_dev",
    "pa_version_under10p1",
    "pa_version_under11p0",
    "pa_version_under13p0",
    "pa_version_under14p0",
    "pa_version_under14p1",
    "pa_version_under16p0",
    "pa_version_under17p0",
    "pa_version_under18p0",
    "pa_version_under19p0",
    "pa_version_under20p0",
    "HAS_PYARROW",
    "IS64",
    "ISMUSL",
    "PY310",
    "PY311",
    "PY312",
    "PYPY",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\_ranges.py ===
"""
Helper functions to generate range-like data for DatetimeArray
(and possibly TimedeltaArray/PeriodArray)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.lib import i8max
from pandas._libs.tslibs import (
    BaseOffset,
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    iNaT,
)

if TYPE_CHECKING:
    from pandas._typing import npt


def generate_regular_range(
    start: Timestamp | Timedelta | None,
    end: Timestamp | Timedelta | None,
    periods: int | None,
    freq: BaseOffset,
    unit: str = "ns",
) -> npt.NDArray[np.intp]:
    """
    Generate a range of dates or timestamps with the spans between dates
    described by the given `freq` DateOffset.

    Parameters
    ----------
    start : Timedelta, Timestamp or None
        First point of produced date range.
    end : Timedelta, Timestamp or None
        Last point of produced date range.
    periods : int or None
        Number of periods in produced date range.
    freq : Tick
        Describes space between dates in produced date range.
    unit : str, default "ns"
        The resolution the output is meant to represent.

    Returns
    -------
    ndarray[np.int64]
        Representing the given resolution.
    """
    istart = start._value if start is not None else None
    iend = end._value if end is not None else None
    freq.nanos  # raises if non-fixed frequency
    td = Timedelta(freq)
    b: int
    e: int
    try:
        td = td.as_unit(unit, round_ok=False)
    except ValueError as err:
        raise ValueError(
            f"freq={freq} is incompatible with unit={unit}. "
            "Use a lower freq or a higher unit instead."
        ) from err
    stride = int(td._value)

    if periods is None and istart is not None and iend is not None:
        b = istart
        # cannot just use e = Timestamp(end) + 1 because arange breaks when
        # stride is too large, see GH10887
        e = b + (iend - b) // stride * stride + stride // 2 + 1
    elif istart is not None and periods is not None:
        b = istart
        e = _generate_range_overflow_safe(b, periods, stride, side="start")
    elif iend is not None and periods is not None:
        e = iend + stride
        b = _generate_range_overflow_safe(e, periods, stride, side="end")
    else:
        raise ValueError(
            "at least 'start' or 'end' should be specified if a 'period' is given."
        )

    with np.errstate(over="raise"):
        # If the range is sufficiently large, np.arange may overflow
        #  and incorrectly return an empty array if not caught.
        try:
            values = np.arange(b, e, stride, dtype=np.int64)
        except FloatingPointError:
            xdr = [b]
            while xdr[-1] != e:
                xdr.append(xdr[-1] + stride)
            values = np.array(xdr[:-1], dtype=np.int64)
    return values


def _generate_range_overflow_safe(
    endpoint: int, periods: int, stride: int, side: str = "start"
) -> int:
    """
    Calculate the second endpoint for passing to np.arange, checking
    to avoid an integer overflow.  Catch OverflowError and re-raise
    as OutOfBoundsDatetime.

    Parameters
    ----------
    endpoint : int
        nanosecond timestamp of the known endpoint of the desired range
    periods : int
        number of periods in the desired range
    stride : int
        nanoseconds between periods in the desired range
    side : {'start', 'end'}
        which end of the range `endpoint` refers to

    Returns
    -------
    other_end : int

    Raises
    ------
    OutOfBoundsDatetime
    """
    # GH#14187 raise instead of incorrectly wrapping around
    assert side in ["start", "end"]

    i64max = np.uint64(i8max)
    msg = f"Cannot generate range with {side}={endpoint} and periods={periods}"

    with np.errstate(over="raise"):
        # if periods * strides cannot be multiplied within the *uint64* bounds,
        #  we cannot salvage the operation by recursing, so raise
        try:
            addend = np.uint64(periods) * np.uint64(np.abs(stride))
        except FloatingPointError as err:
            raise OutOfBoundsDatetime(msg) from err

    if np.abs(addend) <= i64max:
        # relatively easy case without casting concerns
        return _generate_range_overflow_safe_signed(endpoint, periods, stride, side)

    elif (endpoint > 0 and side == "start" and stride > 0) or (
        endpoint < 0 < stride and side == "end"
    ):
        # no chance of not-overflowing
        raise OutOfBoundsDatetime(msg)

    elif side == "end" and endpoint - stride <= i64max < endpoint:
        # in _generate_regular_range we added `stride` thereby overflowing
        #  the bounds.  Adjust to fix this.
        return _generate_range_overflow_safe(
            endpoint - stride, periods - 1, stride, side
        )

    # split into smaller pieces
    mid_periods = periods // 2
    remaining = periods - mid_periods
    assert 0 < remaining < periods, (remaining, periods, endpoint, stride)

    midpoint = int(_generate_range_overflow_safe(endpoint, mid_periods, stride, side))
    return _generate_range_overflow_safe(midpoint, remaining, stride, side)


def _generate_range_overflow_safe_signed(
    endpoint: int, periods: int, stride: int, side: str
) -> int:
    """
    A special case for _generate_range_overflow_safe where `periods * stride`
    can be calculated without overflowing int64 bounds.
    """
    assert side in ["start", "end"]
    if side == "end":
        stride *= -1

    with np.errstate(over="raise"):
        addend = np.int64(periods) * np.int64(stride)
        try:
            # easy case with no overflows
            result = np.int64(endpoint) + addend
            if result == iNaT:
                # Putting this into a DatetimeArray/TimedeltaArray
                #  would incorrectly be interpreted as NaT
                raise OverflowError
            return int(result)
        except (FloatingPointError, OverflowError):
            # with endpoint negative and addend positive we risk
            #  FloatingPointError; with reversed signed we risk OverflowError
            pass

        # if stride and endpoint had opposite signs, then endpoint + addend
        #  should never overflow.  so they must have the same signs
        assert (stride > 0 and endpoint >= 0) or (stride < 0 and endpoint <= 0)

        if stride > 0:
            # watch out for very special case in which we just slightly
            #  exceed implementation bounds, but when passing the result to
            #  np.arange will get a result slightly within the bounds

            uresult = np.uint64(endpoint) + np.uint64(addend)
            i64max = np.uint64(i8max)
            assert uresult > i64max
            if uresult <= i64max + np.uint64(stride):
                return int(uresult)

    raise OutOfBoundsDatetime(
        f"Cannot generate range with {side}={endpoint} and periods={periods}"
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\sparse\scipy_sparse.py ===
"""
Interaction with scipy.sparse matrices.

Currently only includes to_coo helpers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas._libs import lib

from pandas.core.dtypes.missing import notna

from pandas.core.algorithms import factorize
from pandas.core.indexes.api import MultiIndex
from pandas.core.series import Series

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import scipy.sparse

    from pandas._typing import (
        IndexLabel,
        npt,
    )


def _check_is_partition(parts: Iterable, whole: Iterable):
    whole = set(whole)
    parts = [set(x) for x in parts]
    if set.intersection(*parts) != set():
        raise ValueError("Is not a partition because intersection is not null.")
    if set.union(*parts) != whole:
        raise ValueError("Is not a partition because union is not the whole.")


def _levels_to_axis(
    ss,
    levels: tuple[int] | list[int],
    valid_ilocs: npt.NDArray[np.intp],
    sort_labels: bool = False,
) -> tuple[npt.NDArray[np.intp], list[IndexLabel]]:
    """
    For a MultiIndexed sparse Series `ss`, return `ax_coords` and `ax_labels`,
    where `ax_coords` are the coordinates along one of the two axes of the
    destination sparse matrix, and `ax_labels` are the labels from `ss`' Index
    which correspond to these coordinates.

    Parameters
    ----------
    ss : Series
    levels : tuple/list
    valid_ilocs : numpy.ndarray
        Array of integer positions of valid values for the sparse matrix in ss.
    sort_labels : bool, default False
        Sort the axis labels before forming the sparse matrix. When `levels`
        refers to a single level, set to True for a faster execution.

    Returns
    -------
    ax_coords : numpy.ndarray (axis coordinates)
    ax_labels : list (axis labels)
    """
    # Since the labels are sorted in `Index.levels`, when we wish to sort and
    # there is only one level of the MultiIndex for this axis, the desired
    # output can be obtained in the following simpler, more efficient way.
    if sort_labels and len(levels) == 1:
        ax_coords = ss.index.codes[levels[0]][valid_ilocs]
        ax_labels = ss.index.levels[levels[0]]

    else:
        levels_values = lib.fast_zip(
            [ss.index.get_level_values(lvl).to_numpy() for lvl in levels]
        )
        codes, ax_labels = factorize(levels_values, sort=sort_labels)
        ax_coords = codes[valid_ilocs]

    ax_labels = ax_labels.tolist()
    return ax_coords, ax_labels


def _to_ijv(
    ss,
    row_levels: tuple[int] | list[int] = (0,),
    column_levels: tuple[int] | list[int] = (1,),
    sort_labels: bool = False,
) -> tuple[
    np.ndarray,
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    list[IndexLabel],
    list[IndexLabel],
]:
    """
    For an arbitrary MultiIndexed sparse Series return (v, i, j, ilabels,
    jlabels) where (v, (i, j)) is suitable for passing to scipy.sparse.coo
    constructor, and ilabels and jlabels are the row and column labels
    respectively.

    Parameters
    ----------
    ss : Series
    row_levels : tuple/list
    column_levels : tuple/list
    sort_labels : bool, default False
        Sort the row and column labels before forming the sparse matrix.
        When `row_levels` and/or `column_levels` refer to a single level,
        set to `True` for a faster execution.

    Returns
    -------
    values : numpy.ndarray
        Valid values to populate a sparse matrix, extracted from
        ss.
    i_coords : numpy.ndarray (row coordinates of the values)
    j_coords : numpy.ndarray (column coordinates of the values)
    i_labels : list (row labels)
    j_labels : list (column labels)
    """
    # index and column levels must be a partition of the index
    _check_is_partition([row_levels, column_levels], range(ss.index.nlevels))
    # From the sparse Series, get the integer indices and data for valid sparse
    # entries.
    sp_vals = ss.array.sp_values
    na_mask = notna(sp_vals)
    values = sp_vals[na_mask]
    valid_ilocs = ss.array.sp_index.indices[na_mask]

    i_coords, i_labels = _levels_to_axis(
        ss, row_levels, valid_ilocs, sort_labels=sort_labels
    )

    j_coords, j_labels = _levels_to_axis(
        ss, column_levels, valid_ilocs, sort_labels=sort_labels
    )

    return values, i_coords, j_coords, i_labels, j_labels


def sparse_series_to_coo(
    ss: Series,
    row_levels: Iterable[int] = (0,),
    column_levels: Iterable[int] = (1,),
    sort_labels: bool = False,
) -> tuple[scipy.sparse.coo_matrix, list[IndexLabel], list[IndexLabel]]:
    """
    Convert a sparse Series to a scipy.sparse.coo_matrix using index
    levels row_levels, column_levels as the row and column
    labels respectively. Returns the sparse_matrix, row and column labels.
    """
    import scipy.sparse

    if ss.index.nlevels < 2:
        raise ValueError("to_coo requires MultiIndex with nlevels >= 2.")
    if not ss.index.is_unique:
        raise ValueError(
            "Duplicate index entries are not allowed in to_coo transformation."
        )

    # to keep things simple, only rely on integer indexing (not labels)
    row_levels = [ss.index._get_level_number(x) for x in row_levels]
    column_levels = [ss.index._get_level_number(x) for x in column_levels]

    v, i, j, rows, columns = _to_ijv(
        ss, row_levels=row_levels, column_levels=column_levels, sort_labels=sort_labels
    )
    sparse_matrix = scipy.sparse.coo_matrix(
        (v, (i, j)), shape=(len(rows), len(columns))
    )
    return sparse_matrix, rows, columns


def coo_to_sparse_series(
    A: scipy.sparse.coo_matrix, dense_index: bool = False
) -> Series:
    """
    Convert a scipy.sparse.coo_matrix to a Series with type sparse.

    Parameters
    ----------
    A : scipy.sparse.coo_matrix
    dense_index : bool, default False

    Returns
    -------
    Series

    Raises
    ------
    TypeError if A is not a coo_matrix
    """
    from pandas import SparseDtype

    try:
        ser = Series(A.data, MultiIndex.from_arrays((A.row, A.col)), copy=False)
    except AttributeError as err:
        raise TypeError(
            f"Expected coo_matrix. Got {type(A).__name__} instead."
        ) from err
    ser = ser.sort_index()
    ser = ser.astype(SparseDtype(ser.dtype))
    if dense_index:
        ind = MultiIndex.from_product([A.row, A.col])
        ser = ser.reindex(ind)
    return ser