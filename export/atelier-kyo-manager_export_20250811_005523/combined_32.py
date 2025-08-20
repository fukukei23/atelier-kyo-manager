
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\base.py ===
# engine/base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
"""Defines :class:`_engine.Connection` and :class:`_engine.Engine`.

"""
from __future__ import annotations

import contextlib
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from .interfaces import BindTyping
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPICursor
from .interfaces import ExceptionContext
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .interfaces import IsolationLevel
from .util import _distill_params_20
from .util import _distill_raw_params
from .util import TransactionalContext
from .. import exc
from .. import inspection
from .. import log
from .. import util
from ..sql import compiler
from ..sql import util as sql_util

if typing.TYPE_CHECKING:
    from . import CursorResult
    from . import ScalarResult
    from .interfaces import _AnyExecuteParams
    from .interfaces import _AnyMultiExecuteParams
    from .interfaces import _CoreAnyExecuteParams
    from .interfaces import _CoreMultiExecuteParams
    from .interfaces import _CoreSingleExecuteParams
    from .interfaces import _DBAPIAnyExecuteParams
    from .interfaces import _DBAPISingleExecuteParams
    from .interfaces import _ExecuteOptions
    from .interfaces import CompiledCacheType
    from .interfaces import CoreExecuteOptionsParameter
    from .interfaces import Dialect
    from .interfaces import SchemaTranslateMapType
    from .reflection import Inspector  # noqa
    from .url import URL
    from ..event import dispatcher
    from ..log import _EchoFlagType
    from ..pool import _ConnectionFairy
    from ..pool import Pool
    from ..pool import PoolProxiedConnection
    from ..sql import Executable
    from ..sql._typing import _InfoType
    from ..sql.compiler import Compiled
    from ..sql.ddl import ExecutableDDLElement
    from ..sql.ddl import InvokeDDLBase
    from ..sql.functions import FunctionElement
    from ..sql.schema import DefaultGenerator
    from ..sql.schema import HasSchemaAttr
    from ..sql.schema import SchemaVisitable
    from ..sql.selectable import TypedReturnsRows


_T = TypeVar("_T", bound=Any)
_EMPTY_EXECUTION_OPTS: _ExecuteOptions = util.EMPTY_DICT
NO_OPTIONS: Mapping[str, Any] = util.EMPTY_DICT


class Connection(ConnectionEventsTarget, inspection.Inspectable["Inspector"]):
    """Provides high-level functionality for a wrapped DB-API connection.

    The :class:`_engine.Connection` object is procured by calling the
    :meth:`_engine.Engine.connect` method of the :class:`_engine.Engine`
    object, and provides services for execution of SQL statements as well
    as transaction control.

    The Connection object is **not** thread-safe. While a Connection can be
    shared among threads using properly synchronized access, it is still
    possible that the underlying DBAPI connection may not support shared
    access between threads. Check the DBAPI documentation for details.

    The Connection object represents a single DBAPI connection checked out
    from the connection pool. In this state, the connection pool has no
    affect upon the connection, including its expiration or timeout state.
    For the connection pool to properly manage connections, connections
    should be returned to the connection pool (i.e. ``connection.close()``)
    whenever the connection is not in use.

    .. index::
      single: thread safety; Connection

    """

    dialect: Dialect
    dispatch: dispatcher[ConnectionEventsTarget]

    _sqla_logger_namespace = "sqlalchemy.engine.Connection"

    # used by sqlalchemy.engine.util.TransactionalContext
    _trans_context_manager: Optional[TransactionalContext] = None

    # legacy as of 2.0, should be eventually deprecated and
    # removed.  was used in the "pre_ping" recipe that's been in the docs
    # a long time
    should_close_with_result = False

    _dbapi_connection: Optional[PoolProxiedConnection]

    _execution_options: _ExecuteOptions

    _transaction: Optional[RootTransaction]
    _nested_transaction: Optional[NestedTransaction]

    def __init__(
        self,
        engine: Engine,
        connection: Optional[PoolProxiedConnection] = None,
        _has_events: Optional[bool] = None,
        _allow_revalidate: bool = True,
        _allow_autobegin: bool = True,
    ):
        """Construct a new Connection."""
        self.engine = engine
        self.dialect = dialect = engine.dialect

        if connection is None:
            try:
                self._dbapi_connection = engine.raw_connection()
            except dialect.loaded_dbapi.Error as err:
                Connection._handle_dbapi_exception_noconnection(
                    err, dialect, engine
                )
                raise
        else:
            self._dbapi_connection = connection

        self._transaction = self._nested_transaction = None
        self.__savepoint_seq = 0
        self.__in_begin = False

        self.__can_reconnect = _allow_revalidate
        self._allow_autobegin = _allow_autobegin
        self._echo = self.engine._should_log_info()

        if _has_events is None:
            # if _has_events is sent explicitly as False,
            # then don't join the dispatch of the engine; we don't
            # want to handle any of the engine's events in that case.
            self.dispatch = self.dispatch._join(engine.dispatch)
        self._has_events = _has_events or (
            _has_events is None and engine._has_events
        )

        self._execution_options = engine._execution_options

        if self._has_events or self.engine._has_events:
            self.dispatch.engine_connect(self)

    # this can be assigned differently via
    # characteristics.LoggingTokenCharacteristic
    _message_formatter: Any = None

    def _log_info(self, message: str, *arg: Any, **kw: Any) -> None:
        fmt = self._message_formatter

        if fmt:
            message = fmt(message)

        if log.STACKLEVEL:
            kw["stacklevel"] = 1 + log.STACKLEVEL_OFFSET

        self.engine.logger.info(message, *arg, **kw)

    def _log_debug(self, message: str, *arg: Any, **kw: Any) -> None:
        fmt = self._message_formatter

        if fmt:
            message = fmt(message)

        if log.STACKLEVEL:
            kw["stacklevel"] = 1 + log.STACKLEVEL_OFFSET

        self.engine.logger.debug(message, *arg, **kw)

    @property
    def _schema_translate_map(self) -> Optional[SchemaTranslateMapType]:
        schema_translate_map: Optional[SchemaTranslateMapType] = (
            self._execution_options.get("schema_translate_map", None)
        )

        return schema_translate_map

    def schema_for_object(self, obj: HasSchemaAttr) -> Optional[str]:
        """Return the schema name for the given schema item taking into
        account current schema translate map.

        """

        name = obj.schema
        schema_translate_map: Optional[SchemaTranslateMapType] = (
            self._execution_options.get("schema_translate_map", None)
        )

        if (
            schema_translate_map
            and name in schema_translate_map
            and obj._use_schema_map
        ):
            return schema_translate_map[name]
        else:
            return name

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        self.close()

    @overload
    def execution_options(
        self,
        *,
        compiled_cache: Optional[CompiledCacheType] = ...,
        logging_token: str = ...,
        isolation_level: IsolationLevel = ...,
        no_parameters: bool = False,
        stream_results: bool = False,
        max_row_buffer: int = ...,
        yield_per: int = ...,
        insertmanyvalues_page_size: int = ...,
        schema_translate_map: Optional[SchemaTranslateMapType] = ...,
        preserve_rowcount: bool = False,
        **opt: Any,
    ) -> Connection: ...

    @overload
    def execution_options(self, **opt: Any) -> Connection: ...

    def execution_options(self, **opt: Any) -> Connection:
        r"""Set non-SQL options for the connection which take effect
        during execution.

        This method modifies this :class:`_engine.Connection` **in-place**;
        the return value is the same :class:`_engine.Connection` object
        upon which the method is called.   Note that this is in contrast
        to the behavior of the ``execution_options`` methods on other
        objects such as :meth:`_engine.Engine.execution_options` and
        :meth:`_sql.Executable.execution_options`.  The rationale is that many
        such execution options necessarily modify the state of the base
        DBAPI connection in any case so there is no feasible means of
        keeping the effect of such an option localized to a "sub" connection.

        .. versionchanged:: 2.0  The :meth:`_engine.Connection.execution_options`
           method, in contrast to other objects with this method, modifies
           the connection in-place without creating copy of it.

        As discussed elsewhere, the :meth:`_engine.Connection.execution_options`
        method accepts any arbitrary parameters including user defined names.
        All parameters given are consumable in a number of ways including
        by using the :meth:`_engine.Connection.get_execution_options` method.
        See the examples at :meth:`_sql.Executable.execution_options`
        and :meth:`_engine.Engine.execution_options`.

        The keywords that are currently recognized by SQLAlchemy itself
        include all those listed under :meth:`.Executable.execution_options`,
        as well as others that are specific to :class:`_engine.Connection`.

        :param compiled_cache: Available on: :class:`_engine.Connection`,
          :class:`_engine.Engine`.

          A dictionary where :class:`.Compiled` objects
          will be cached when the :class:`_engine.Connection`
          compiles a clause
          expression into a :class:`.Compiled` object.  This dictionary will
          supersede the statement cache that may be configured on the
          :class:`_engine.Engine` itself.   If set to None, caching
          is disabled, even if the engine has a configured cache size.

          Note that the ORM makes use of its own "compiled" caches for
          some operations, including flush operations.  The caching
          used by the ORM internally supersedes a cache dictionary
          specified here.

        :param logging_token: Available on: :class:`_engine.Connection`,
          :class:`_engine.Engine`, :class:`_sql.Executable`.

          Adds the specified string token surrounded by brackets in log
          messages logged by the connection, i.e. the logging that's enabled
          either via the :paramref:`_sa.create_engine.echo` flag or via the
          ``logging.getLogger("sqlalchemy.engine")`` logger. This allows a
          per-connection or per-sub-engine token to be available which is
          useful for debugging concurrent connection scenarios.

          .. versionadded:: 1.4.0b2

          .. seealso::

            :ref:`dbengine_logging_tokens` - usage example

            :paramref:`_sa.create_engine.logging_name` - adds a name to the
            name used by the Python logger object itself.

        :param isolation_level: Available on: :class:`_engine.Connection`,
          :class:`_engine.Engine`.

          Set the transaction isolation level for the lifespan of this
          :class:`_engine.Connection` object.
          Valid values include those string
          values accepted by the :paramref:`_sa.create_engine.isolation_level`
          parameter passed to :func:`_sa.create_engine`.  These levels are
          semi-database specific; see individual dialect documentation for
          valid levels.

          The isolation level option applies the isolation level by emitting
          statements on the DBAPI connection, and **necessarily affects the
          original Connection object overall**. The isolation level will remain
          at the given setting until explicitly changed, or when the DBAPI
          connection itself is :term:`released` to the connection pool, i.e. the
          :meth:`_engine.Connection.close` method is called, at which time an
          event handler will emit additional statements on the DBAPI connection
          in order to revert the isolation level change.

          .. note:: The ``isolation_level`` execution option may only be
             established before the :meth:`_engine.Connection.begin` method is
             called, as well as before any SQL statements are emitted which
             would otherwise trigger "autobegin", or directly after a call to
             :meth:`_engine.Connection.commit` or
             :meth:`_engine.Connection.rollback`. A database cannot change the
             isolation level on a transaction in progress.

          .. note:: The ``isolation_level`` execution option is implicitly
             reset if the :class:`_engine.Connection` is invalidated, e.g. via
             the :meth:`_engine.Connection.invalidate` method, or if a
             disconnection error occurs. The new connection produced after the
             invalidation will **not** have the selected isolation level
             re-applied to it automatically.

          .. seealso::

                :ref:`dbapi_autocommit`

                :meth:`_engine.Connection.get_isolation_level`
                - view current actual level

        :param no_parameters: Available on: :class:`_engine.Connection`,
          :class:`_sql.Executable`.

          When ``True``, if the final parameter
          list or dictionary is totally empty, will invoke the
          statement on the cursor as ``cursor.execute(statement)``,
          not passing the parameter collection at all.
          Some DBAPIs such as psycopg2 and mysql-python consider
          percent signs as significant only when parameters are
          present; this option allows code to generate SQL
          containing percent signs (and possibly other characters)
          that is neutral regarding whether it's executed by the DBAPI
          or piped into a script that's later invoked by
          command line tools.

        :param stream_results: Available on: :class:`_engine.Connection`,
          :class:`_sql.Executable`.

          Indicate to the dialect that results should be "streamed" and not
          pre-buffered, if possible.  For backends such as PostgreSQL, MySQL
          and MariaDB, this indicates the use of a "server side cursor" as
          opposed to a client side cursor.  Other backends such as that of
          Oracle Database may already use server side cursors by default.

          The usage of
          :paramref:`_engine.Connection.execution_options.stream_results` is
          usually combined with setting a fixed number of rows to to be fetched
          in batches, to allow for efficient iteration of database rows while
          at the same time not loading all result rows into memory at once;
          this can be configured on a :class:`_engine.Result` object using the
          :meth:`_engine.Result.yield_per` method, after execution has
          returned a new :class:`_engine.Result`.   If
          :meth:`_engine.Result.yield_per` is not used,
          the :paramref:`_engine.Connection.execution_options.stream_results`
          mode of operation will instead use a dynamically sized buffer
          which buffers sets of rows at a time, growing on each batch
          based on a fixed growth size up until a limit which may
          be configured using the
          :paramref:`_engine.Connection.execution_options.max_row_buffer`
          parameter.

          When using the ORM to fetch ORM mapped objects from a result,
          :meth:`_engine.Result.yield_per` should always be used with
          :paramref:`_engine.Connection.execution_options.stream_results`,
          so that the ORM does not fetch all rows into new ORM objects at once.

          For typical use, the
          :paramref:`_engine.Connection.execution_options.yield_per` execution
          option should be preferred, which sets up both
          :paramref:`_engine.Connection.execution_options.stream_results` and
          :meth:`_engine.Result.yield_per` at once. This option is supported
          both at a core level by :class:`_engine.Connection` as well as by the
          ORM :class:`_engine.Session`; the latter is described at
          :ref:`orm_queryguide_yield_per`.

          .. seealso::

            :ref:`engine_stream_results` - background on
            :paramref:`_engine.Connection.execution_options.stream_results`

            :paramref:`_engine.Connection.execution_options.max_row_buffer`

            :paramref:`_engine.Connection.execution_options.yield_per`

            :ref:`orm_queryguide_yield_per` - in the :ref:`queryguide_toplevel`
            describing the ORM version of ``yield_per``

        :param max_row_buffer: Available on: :class:`_engine.Connection`,
          :class:`_sql.Executable`.  Sets a maximum
          buffer size to use when the
          :paramref:`_engine.Connection.execution_options.stream_results`
          execution option is used on a backend that supports server side
          cursors.  The default value if not specified is 1000.

          .. seealso::

            :paramref:`_engine.Connection.execution_options.stream_results`

            :ref:`engine_stream_results`


        :param yield_per: Available on: :class:`_engine.Connection`,
          :class:`_sql.Executable`.  Integer value applied which will
          set the :paramref:`_engine.Connection.execution_options.stream_results`
          execution option and invoke :meth:`_engine.Result.yield_per`
          automatically at once.  Allows equivalent functionality as
          is present when using this parameter with the ORM.

          .. versionadded:: 1.4.40

          .. seealso::

            :ref:`engine_stream_results` - background and examples
            on using server side cursors with Core.

            :ref:`orm_queryguide_yield_per` - in the :ref:`queryguide_toplevel`
            describing the ORM version of ``yield_per``

        :param insertmanyvalues_page_size: Available on: :class:`_engine.Connection`,
            :class:`_engine.Engine`. Number of rows to format into an
            INSERT statement when the statement uses "insertmanyvalues" mode,
            which is a paged form of bulk insert that is used for many backends
            when using :term:`executemany` execution typically in conjunction
            with RETURNING. Defaults to 1000. May also be modified on a
            per-engine basis using the
            :paramref:`_sa.create_engine.insertmanyvalues_page_size` parameter.

            .. versionadded:: 2.0

            .. seealso::

                :ref:`engine_insertmanyvalues`

        :param schema_translate_map: Available on: :class:`_engine.Connection`,
          :class:`_engine.Engine`, :class:`_sql.Executable`.

          A dictionary mapping schema names to schema names, that will be
          applied to the :paramref:`_schema.Table.schema` element of each
          :class:`_schema.Table`
          encountered when SQL or DDL expression elements
          are compiled into strings; the resulting schema name will be
          converted based on presence in the map of the original name.

          .. seealso::

            :ref:`schema_translating`

        :param preserve_rowcount: Boolean; when True, the ``cursor.rowcount``
          attribute will be unconditionally memoized within the result and
          made available via the :attr:`.CursorResult.rowcount` attribute.
          Normally, this attribute is only preserved for UPDATE and DELETE
          statements.  Using this option, the DBAPIs rowcount value can
          be accessed for other kinds of statements such as INSERT and SELECT,
          to the degree that the DBAPI supports these statements.  See
          :attr:`.CursorResult.rowcount` for notes regarding the behavior
          of this attribute.

          .. versionadded:: 2.0.28

        .. seealso::

            :meth:`_engine.Engine.execution_options`

            :meth:`.Executable.execution_options`

            :meth:`_engine.Connection.get_execution_options`

            :ref:`orm_queryguide_execution_options` - documentation on all
            ORM-specific execution options

        """  # noqa
        if self._has_events or self.engine._has_events:
            self.dispatch.set_connection_execution_options(self, opt)
        self._execution_options = self._execution_options.union(opt)
        self.dialect.set_connection_execution_options(self, opt)
        return self

    def get_execution_options(self) -> _ExecuteOptions:
        """Get the non-SQL options which will take effect during execution.

        .. versionadded:: 1.3

        .. seealso::

            :meth:`_engine.Connection.execution_options`
        """
        return self._execution_options

    @property
    def _still_open_and_dbapi_connection_is_valid(self) -> bool:
        pool_proxied_connection = self._dbapi_connection
        return (
            pool_proxied_connection is not None
            and pool_proxied_connection.is_valid
        )

    @property
    def closed(self) -> bool:
        """Return True if this connection is closed."""

        return self._dbapi_connection is None and not self.__can_reconnect

    @property
    def invalidated(self) -> bool:
        """Return True if this connection was invalidated.

        This does not indicate whether or not the connection was
        invalidated at the pool level, however

        """

        # prior to 1.4, "invalid" was stored as a state independent of
        # "closed", meaning an invalidated connection could be "closed",
        # the _dbapi_connection would be None and closed=True, yet the
        # "invalid" flag would stay True.  This meant that there were
        # three separate states (open/valid, closed/valid, closed/invalid)
        # when there is really no reason for that; a connection that's
        # "closed" does not need to be "invalid".  So the state is now
        # represented by the two facts alone.

        pool_proxied_connection = self._dbapi_connection
        return pool_proxied_connection is None and self.__can_reconnect

    @property
    def connection(self) -> PoolProxiedConnection:
        """The underlying DB-API connection managed by this Connection.

        This is a SQLAlchemy connection-pool proxied connection
        which then has the attribute
        :attr:`_pool._ConnectionFairy.dbapi_connection` that refers to the
        actual driver connection.

        .. seealso::


            :ref:`dbapi_connections`

        """

        if self._dbapi_connection is None:
            try:
                return self._revalidate_connection()
            except (exc.PendingRollbackError, exc.ResourceClosedError):
                raise
            except BaseException as e:
                self._handle_dbapi_exception(e, None, None, None, None)
        else:
            return self._dbapi_connection

    def get_isolation_level(self) -> IsolationLevel:
        """Return the current **actual** isolation level that's present on
        the database within the scope of this connection.

        This attribute will perform a live SQL operation against the database
        in order to procure the current isolation level, so the value returned
        is the actual level on the underlying DBAPI connection regardless of
        how this state was set. This will be one of the four actual isolation
        modes ``READ UNCOMMITTED``, ``READ COMMITTED``, ``REPEATABLE READ``,
        ``SERIALIZABLE``. It will **not** include the ``AUTOCOMMIT`` isolation
        level setting. Third party dialects may also feature additional
        isolation level settings.

        .. note::  This method **will not report** on the ``AUTOCOMMIT``
          isolation level, which is a separate :term:`dbapi` setting that's
          independent of **actual** isolation level.  When ``AUTOCOMMIT`` is
          in use, the database connection still has a "traditional" isolation
          mode in effect, that is typically one of the four values
          ``READ UNCOMMITTED``, ``READ COMMITTED``, ``REPEATABLE READ``,
          ``SERIALIZABLE``.

        Compare to the :attr:`_engine.Connection.default_isolation_level`
        accessor which returns the isolation level that is present on the
        database at initial connection time.

        .. seealso::

            :attr:`_engine.Connection.default_isolation_level`
            - view default level

            :paramref:`_sa.create_engine.isolation_level`
            - set per :class:`_engine.Engine` isolation level

            :paramref:`.Connection.execution_options.isolation_level`
            - set per :class:`_engine.Connection` isolation level

        """
        dbapi_connection = self.connection.dbapi_connection
        assert dbapi_connection is not None
        try:
            return self.dialect.get_isolation_level(dbapi_connection)
        except BaseException as e:
            self._handle_dbapi_exception(e, None, None, None, None)

    @property
    def default_isolation_level(self) -> Optional[IsolationLevel]:
        """The initial-connection time isolation level associated with the
        :class:`_engine.Dialect` in use.

        This value is independent of the
        :paramref:`.Connection.execution_options.isolation_level` and
        :paramref:`.Engine.execution_options.isolation_level` execution
        options, and is determined by the :class:`_engine.Dialect` when the
        first connection is created, by performing a SQL query against the
        database for the current isolation level before any additional commands
        have been emitted.

        Calling this accessor does not invoke any new SQL queries.

        .. seealso::

            :meth:`_engine.Connection.get_isolation_level`
            - view current actual isolation level

            :paramref:`_sa.create_engine.isolation_level`
            - set per :class:`_engine.Engine` isolation level

            :paramref:`.Connection.execution_options.isolation_level`
            - set per :class:`_engine.Connection` isolation level

        """
        return self.dialect.default_isolation_level

    def _invalid_transaction(self) -> NoReturn:
        raise exc.PendingRollbackError(
            "Can't reconnect until invalid %stransaction is rolled "
            "back.  Please rollback() fully before proceeding"
            % ("savepoint " if self._nested_transaction is not None else ""),
            code="8s2b",
        )

    def _revalidate_connection(self) -> PoolProxiedConnection:
        if self.__can_reconnect and self.invalidated:
            if self._transaction is not None:
                self._invalid_transaction()
            self._dbapi_connection = self.engine.raw_connection()
            return self._dbapi_connection
        raise exc.ResourceClosedError("This Connection is closed")

    @property
    def info(self) -> _InfoType:
        """Info dictionary associated with the underlying DBAPI connection
        referred to by this :class:`_engine.Connection`, allowing user-defined
        data to be associated with the connection.

        The data here will follow along with the DBAPI connection including
        after it is returned to the connection pool and used again
        in subsequent instances of :class:`_engine.Connection`.

        """

        return self.connection.info

    def invalidate(self, exception: Optional[BaseException] = None) -> None:
        """Invalidate the underlying DBAPI connection associated with
        this :class:`_engine.Connection`.

        An attempt will be made to close the underlying DBAPI connection
        immediately; however if this operation fails, the error is logged
        but not raised.  The connection is then discarded whether or not
        close() succeeded.

        Upon the next use (where "use" typically means using the
        :meth:`_engine.Connection.execute` method or similar),
        this :class:`_engine.Connection` will attempt to
        procure a new DBAPI connection using the services of the
        :class:`_pool.Pool` as a source of connectivity (e.g.
        a "reconnection").

        If a transaction was in progress (e.g. the
        :meth:`_engine.Connection.begin` method has been called) when
        :meth:`_engine.Connection.invalidate` method is called, at the DBAPI
        level all state associated with this transaction is lost, as
        the DBAPI connection is closed.  The :class:`_engine.Connection`
        will not allow a reconnection to proceed until the
        :class:`.Transaction` object is ended, by calling the
        :meth:`.Transaction.rollback` method; until that point, any attempt at
        continuing to use the :class:`_engine.Connection` will raise an
        :class:`~sqlalchemy.exc.InvalidRequestError`.
        This is to prevent applications from accidentally
        continuing an ongoing transactional operations despite the
        fact that the transaction has been lost due to an
        invalidation.

        The :meth:`_engine.Connection.invalidate` method,
        just like auto-invalidation,
        will at the connection pool level invoke the
        :meth:`_events.PoolEvents.invalidate` event.

        :param exception: an optional ``Exception`` instance that's the
         reason for the invalidation.  is passed along to event handlers
         and logging functions.

        .. seealso::

            :ref:`pool_connection_invalidation`

        """

        if self.invalidated:
            return

        if self.closed:
            raise exc.ResourceClosedError("This Connection is closed")

        if self._still_open_and_dbapi_connection_is_valid:
            pool_proxied_connection = self._dbapi_connection
            assert pool_proxied_connection is not None
            pool_proxied_connection.invalidate(exception)

        self._dbapi_connection = None

    def detach(self) -> None:
        """Detach the underlying DB-API connection from its connection pool.

        E.g.::

            with engine.connect() as conn:
                conn.detach()
                conn.execute(text("SET search_path TO schema1, schema2"))

                # work with connection

            # connection is fully closed (since we used "with:", can
            # also call .close())

        This :class:`_engine.Connection` instance will remain usable.
        When closed
        (or exited from a context manager context as above),
        the DB-API connection will be literally closed and not
        returned to its originating pool.

        This method can be used to insulate the rest of an application
        from a modified state on a connection (such as a transaction
        isolation level or similar).

        """

        if self.closed:
            raise exc.ResourceClosedError("This Connection is closed")

        pool_proxied_connection = self._dbapi_connection
        if pool_proxied_connection is None:
            raise exc.InvalidRequestError(
                "Can't detach an invalidated Connection"
            )
        pool_proxied_connection.detach()

    def _autobegin(self) -> None:
        if self._allow_autobegin and not self.__in_begin:
            self.begin()

    def begin(self) -> RootTransaction:
        """Begin a transaction prior to autobegin occurring.

        E.g.::

            with engine.connect() as conn:
                with conn.begin() as trans:
                    conn.execute(table.insert(), {"username": "sandy"})

        The returned object is an instance of :class:`_engine.RootTransaction`.
        This object represents the "scope" of the transaction,
        which completes when either the :meth:`_engine.Transaction.rollback`
        or :meth:`_engine.Transaction.commit` method is called; the object
        also works as a context manager as illustrated above.

        The :meth:`_engine.Connection.begin` method begins a
        transaction that normally will be begun in any case when the connection
        is first used to execute a statement.  The reason this method might be
        used would be to invoke the :meth:`_events.ConnectionEvents.begin`
        event at a specific time, or to organize code within the scope of a
        connection checkout in terms of context managed blocks, such as::

            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(...)
                    conn.execute(...)

                with conn.begin():
                    conn.execute(...)
                    conn.execute(...)

        The above code is not  fundamentally any different in its behavior than
        the following code  which does not use
        :meth:`_engine.Connection.begin`; the below style is known
        as "commit as you go" style::

            with engine.connect() as conn:
                conn.execute(...)
                conn.execute(...)
                conn.commit()

                conn.execute(...)
                conn.execute(...)
                conn.commit()

        From a database point of view, the :meth:`_engine.Connection.begin`
        method does not emit any SQL or change the state of the underlying
        DBAPI connection in any way; the Python DBAPI does not have any
        concept of explicit transaction begin.

        .. seealso::

            :ref:`tutorial_working_with_transactions` - in the
            :ref:`unified_tutorial`

            :meth:`_engine.Connection.begin_nested` - use a SAVEPOINT

            :meth:`_engine.Connection.begin_twophase` -
            use a two phase /XID transaction

            :meth:`_engine.Engine.begin` - context manager available from
            :class:`_engine.Engine`

        """
        if self._transaction is None:
            self._transaction = RootTransaction(self)
            return self._transaction
        else:
            raise exc.InvalidRequestError(
                "This connection has already initialized a SQLAlchemy "
                "Transaction() object via begin() or autobegin; can't "
                "call begin() here unless rollback() or commit() "
                "is called first."
            )

    def begin_nested(self) -> NestedTransaction:
        """Begin a nested transaction (i.e. SAVEPOINT) and return a transaction
        handle that controls the scope of the SAVEPOINT.

        E.g.::

            with engine.begin() as connection:
                with connection.begin_nested():
                    connection.execute(table.insert(), {"username": "sandy"})

        The returned object is an instance of
        :class:`_engine.NestedTransaction`, which includes transactional
        methods :meth:`_engine.NestedTransaction.commit` and
        :meth:`_engine.NestedTransaction.rollback`; for a nested transaction,
        these methods correspond to the operations "RELEASE SAVEPOINT <name>"
        and "ROLLBACK TO SAVEPOINT <name>". The name of the savepoint is local
        to the :class:`_engine.NestedTransaction` object and is generated
        automatically. Like any other :class:`_engine.Transaction`, the
        :class:`_engine.NestedTransaction` may be used as a context manager as
        illustrated above which will "release" or "rollback" corresponding to
        if the operation within the block were successful or raised an
        exception.

        Nested transactions require SAVEPOINT support in the underlying
        database, else the behavior is undefined. SAVEPOINT is commonly used to
        run operations within a transaction that may fail, while continuing the
        outer transaction. E.g.::

            from sqlalchemy import exc

            with engine.begin() as connection:
                trans = connection.begin_nested()
                try:
                    connection.execute(table.insert(), {"username": "sandy"})
                    trans.commit()
                except exc.IntegrityError:  # catch for duplicate username
                    trans.rollback()  # rollback to savepoint

                # outer transaction continues
                connection.execute(...)

        If :meth:`_engine.Connection.begin_nested` is called without first
        calling :meth:`_engine.Connection.begin` or
        :meth:`_engine.Engine.begin`, the :class:`_engine.Connection` object
        will "autobegin" the outer transaction first. This outer transaction
        may be committed using "commit-as-you-go" style, e.g.::

            with engine.connect() as connection:  # begin() wasn't called

                with connection.begin_nested():  # will auto-"begin()" first
                    connection.execute(...)
                # savepoint is released

                connection.execute(...)

                # explicitly commit outer transaction
                connection.commit()

                # can continue working with connection here

        .. versionchanged:: 2.0

            :meth:`_engine.Connection.begin_nested` will now participate
            in the connection "autobegin" behavior that is new as of
            2.0 / "future" style connections in 1.4.

        .. seealso::

            :meth:`_engine.Connection.begin`

            :ref:`session_begin_nested` - ORM support for SAVEPOINT

        """
        if self._transaction is None:
            self._autobegin()

        return NestedTransaction(self)

    def begin_twophase(self, xid: Optional[Any] = None) -> TwoPhaseTransaction:
        """Begin a two-phase or XA transaction and return a transaction
        handle.

        The returned object is an instance of :class:`.TwoPhaseTransaction`,
        which in addition to the methods provided by
        :class:`.Transaction`, also provides a
        :meth:`~.TwoPhaseTransaction.prepare` method.

        :param xid: the two phase transaction id.  If not supplied, a
          random id will be generated.

        .. seealso::

            :meth:`_engine.Connection.begin`

            :meth:`_engine.Connection.begin_twophase`

        """

        if self._transaction is not None:
            raise exc.InvalidRequestError(
                "Cannot start a two phase transaction when a transaction "
                "is already in progress."
            )
        if xid is None:
            xid = self.engine.dialect.create_xid()
        return TwoPhaseTransaction(self, xid)

    def commit(self) -> None:
        """Commit the transaction that is currently in progress.

        This method commits the current transaction if one has been started.
        If no transaction was started, the method has no effect, assuming
        the connection is in a non-invalidated state.

        A transaction is begun on a :class:`_engine.Connection` automatically
        whenever a statement is first executed, or when the
        :meth:`_engine.Connection.begin` method is called.

        .. note:: The :meth:`_engine.Connection.commit` method only acts upon
          the primary database transaction that is linked to the
          :class:`_engine.Connection` object.  It does not operate upon a
          SAVEPOINT that would have been invoked from the
          :meth:`_engine.Connection.begin_nested` method; for control of a
          SAVEPOINT, call :meth:`_engine.NestedTransaction.commit` on the
          :class:`_engine.NestedTransaction` that is returned by the
          :meth:`_engine.Connection.begin_nested` method itself.


        """
        if self._transaction:
            self._transaction.commit()

    def rollback(self) -> None:
        """Roll back the transaction that is currently in progress.

        This method rolls back the current transaction if one has been started.
        If no transaction was started, the method has no effect.  If a
        transaction was started and the connection is in an invalidated state,
        the transaction is cleared using this method.

        A transaction is begun on a :class:`_engine.Connection` automatically
        whenever a statement is first executed, or when the
        :meth:`_engine.Connection.begin` method is called.

        .. note:: The :meth:`_engine.Connection.rollback` method only acts
          upon the primary database transaction that is linked to the
          :class:`_engine.Connection` object.  It does not operate upon a
          SAVEPOINT that would have been invoked from the
          :meth:`_engine.Connection.begin_nested` method; for control of a
          SAVEPOINT, call :meth:`_engine.NestedTransaction.rollback` on the
          :class:`_engine.NestedTransaction` that is returned by the
          :meth:`_engine.Connection.begin_nested` method itself.


        """
        if self._transaction:
            self._transaction.rollback()

    def recover_twophase(self) -> List[Any]:
        return self.engine.dialect.do_recover_twophase(self)

    def rollback_prepared(self, xid: Any, recover: bool = False) -> None:
        self.engine.dialect.do_rollback_twophase(self, xid, recover=recover)

    def commit_prepared(self, xid: Any, recover: bool = False) -> None:
        self.engine.dialect.do_commit_twophase(self, xid, recover=recover)

    def in_transaction(self) -> bool:
        """Return True if a transaction is in progress."""
        return self._transaction is not None and self._transaction.is_active

    def in_nested_transaction(self) -> bool:
        """Return True if a transaction is in progress."""
        return (
            self._nested_transaction is not None
            and self._nested_transaction.is_active
        )

    def _is_autocommit_isolation(self) -> bool:
        opt_iso = self._execution_options.get("isolation_level", None)
        return bool(
            opt_iso == "AUTOCOMMIT"
            or (
                opt_iso is None
                and self.engine.dialect._on_connect_isolation_level
                == "AUTOCOMMIT"
            )
        )

    def _get_required_transaction(self) -> RootTransaction:
        trans = self._transaction
        if trans is None:
            raise exc.InvalidRequestError("connection is not in a transaction")
        return trans

    def _get_required_nested_transaction(self) -> NestedTransaction:
        trans = self._nested_transaction
        if trans is None:
            raise exc.InvalidRequestError(
                "connection is not in a nested transaction"
            )
        return trans

    def get_transaction(self) -> Optional[RootTransaction]:
        """Return the current root transaction in progress, if any.

        .. versionadded:: 1.4

        """

        return self._transaction

    def get_nested_transaction(self) -> Optional[NestedTransaction]:
        """Return the current nested transaction in progress, if any.

        .. versionadded:: 1.4

        """
        return self._nested_transaction

    def _begin_impl(self, transaction: RootTransaction) -> None:
        if self._echo:
            if self._is_autocommit_isolation():
                self._log_info(
                    "BEGIN (implicit; DBAPI should not BEGIN due to "
                    "autocommit mode)"
                )
            else:
                self._log_info("BEGIN (implicit)")

        self.__in_begin = True

        if self._has_events or self.engine._has_events:
            self.dispatch.begin(self)

        try:
            self.engine.dialect.do_begin(self.connection)
        except BaseException as e:
            self._handle_dbapi_exception(e, None, None, None, None)
        finally:
            self.__in_begin = False

    def _rollback_impl(self) -> None:
        if self._has_events or self.engine._has_events:
            self.dispatch.rollback(self)

        if self._still_open_and_dbapi_connection_is_valid:
            if self._echo:
                if self._is_autocommit_isolation():
                    self._log_info(
                        "ROLLBACK using DBAPI connection.rollback(), "
                        "DBAPI should ignore due to autocommit mode"
                    )
                else:
                    self._log_info("ROLLBACK")
            try:
                self.engine.dialect.do_rollback(self.connection)
            except BaseException as e:
                self._handle_dbapi_exception(e, None, None, None, None)

    def _commit_impl(self) -> None:
        if self._has_events or self.engine._has_events:
            self.dispatch.commit(self)

        if self._echo:
            if self._is_autocommit_isolation():
                self._log_info(
                    "COMMIT using DBAPI connection.commit(), "
                    "DBAPI should ignore due to autocommit mode"
                )
            else:
                self._log_info("COMMIT")
        try:
            self.engine.dialect.do_commit(self.connection)
        except BaseException as e:
            self._handle_dbapi_exception(e, None, None, None, None)

    def _savepoint_impl(self, name: Optional[str] = None) -> str:
        if self._has_events or self.engine._has_events:
            self.dispatch.savepoint(self, name)

        if name is None:
            self.__savepoint_seq += 1
            name = "sa_savepoint_%s" % self.__savepoint_seq
        self.engine.dialect.do_savepoint(self, name)
        return name

    def _rollback_to_savepoint_impl(self, name: str) -> None:
        if self._has_events or self.engine._has_events:
            self.dispatch.rollback_savepoint(self, name, None)

        if self._still_open_and_dbapi_connection_is_valid:
            self.engine.dialect.do_rollback_to_savepoint(self, name)

    def _release_savepoint_impl(self, name: str) -> None:
        if self._has_events or self.engine._has_events:
            self.dispatch.release_savepoint(self, name, None)

        self.engine.dialect.do_release_savepoint(self, name)

    def _begin_twophase_impl(self, transaction: TwoPhaseTransaction) -> None:
        if self._echo:
            self._log_info("BEGIN TWOPHASE (implicit)")
        if self._has_events or self.engine._has_events:
            self.dispatch.begin_twophase(self, transaction.xid)

        self.__in_begin = True
        try:
            self.engine.dialect.do_begin_twophase(self, transaction.xid)
        except BaseException as e:
            self._handle_dbapi_exception(e, None, None, None, None)
        finally:
            self.__in_begin = False

    def _prepare_twophase_impl(self, xid: Any) -> None:
        if self._has_events or self.engine._has_events:
            self.dispatch.prepare_twophase(self, xid)

        assert isinstance(self._transaction, TwoPhaseTransaction)
        try:
            self.engine.dialect.do_prepare_twophase(self, xid)
        except BaseException as e:
            self._handle_dbapi_exception(e, None, None, None, None)

    def _rollback_twophase_impl(self, xid: Any, is_prepared: bool) -> None:
        if self._has_events or self.engine._has_events:
            self.dispatch.rollback_twophase(self, xid, is_prepared)

        if self._still_open_and_dbapi_connection_is_valid:
            assert isinstance(self._transaction, TwoPhaseTransaction)
            try:
                self.engine.dialect.do_rollback_twophase(
                    self, xid, is_prepared
                )
            except BaseException as e:
                self._handle_dbapi_exception(e, None, None, None, None)

    def _commit_twophase_impl(self, xid: Any, is_prepared: bool) -> None:
        if self._has_events or self.engine._has_events:
            self.dispatch.commit_twophase(self, xid, is_prepared)

        assert isinstance(self._transaction, TwoPhaseTransaction)
        try:
            self.engine.dialect.do_commit_twophase(self, xid, is_prepared)
        except BaseException as e:
            self._handle_dbapi_exception(e, None, None, None, None)

    def close(self) -> None:
        """Close this :class:`_engine.Connection`.

        This results in a release of the underlying database
        resources, that is, the DBAPI connection referenced
        internally. The DBAPI connection is typically restored
        back to the connection-holding :class:`_pool.Pool` referenced
        by the :class:`_engine.Engine` that produced this
        :class:`_engine.Connection`. Any transactional state present on
        the DBAPI connection is also unconditionally released via
        the DBAPI connection's ``rollback()`` method, regardless
        of any :class:`.Transaction` object that may be
        outstanding with regards to this :class:`_engine.Connection`.

        This has the effect of also calling :meth:`_engine.Connection.rollback`
        if any transaction is in place.

        After :meth:`_engine.Connection.close` is called, the
        :class:`_engine.Connection` is permanently in a closed state,
        and will allow no further operations.

        """

        if self._transaction:
            self._transaction.close()
            skip_reset = True
        else:
            skip_reset = False

        if self._dbapi_connection is not None:
            conn = self._dbapi_connection

            # as we just closed the transaction, close the connection
            # pool connection without doing an additional reset
            if skip_reset:
                cast("_ConnectionFairy", conn)._close_special(
                    transaction_reset=True
                )
            else:
                conn.close()

            # There is a slight chance that conn.close() may have
            # triggered an invalidation here in which case
            # _dbapi_connection would already be None, however usually
            # it will be non-None here and in a "closed" state.
            self._dbapi_connection = None
        self.__can_reconnect = False

    @overload
    def scalar(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> Optional[_T]: ...

    @overload
    def scalar(
        self,
        statement: Executable,
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> Any: ...

    def scalar(
        self,
        statement: Executable,
        parameters: Optional[_CoreSingleExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> Any:
        r"""Executes a SQL statement construct and returns a scalar object.

        This method is shorthand for invoking the
        :meth:`_engine.Result.scalar` method after invoking the
        :meth:`_engine.Connection.execute` method.  Parameters are equivalent.

        :return: a scalar Python value representing the first column of the
         first row returned.

        """
        distilled_parameters = _distill_params_20(parameters)
        try:
            meth = statement._execute_on_scalar
        except AttributeError as err:
            raise exc.ObjectNotExecutableError(statement) from err
        else:
            return meth(
                self,
                distilled_parameters,
                execution_options or NO_OPTIONS,
            )

    @overload
    def scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> ScalarResult[_T]: ...

    @overload
    def scalars(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> ScalarResult[Any]: ...

    def scalars(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> ScalarResult[Any]:
        """Executes and returns a scalar result set, which yields scalar values
        from the first column of each row.

        This method is equivalent to calling :meth:`_engine.Connection.execute`
        to receive a :class:`_result.Result` object, then invoking the
        :meth:`_result.Result.scalars` method to produce a
        :class:`_result.ScalarResult` instance.

        :return: a :class:`_result.ScalarResult`

        .. versionadded:: 1.4.24

        """

        return self.execute(
            statement, parameters, execution_options=execution_options
        ).scalars()

    @overload
    def execute(
        self,
        statement: TypedReturnsRows[_T],
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> CursorResult[_T]: ...

    @overload
    def execute(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> CursorResult[Any]: ...

    def execute(
        self,
        statement: Executable,
        parameters: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> CursorResult[Any]:
        r"""Executes a SQL statement construct and returns a
        :class:`_engine.CursorResult`.

        :param statement: The statement to be executed.  This is always
         an object that is in both the :class:`_expression.ClauseElement` and
         :class:`_expression.Executable` hierarchies, including:

         * :class:`_expression.Select`
         * :class:`_expression.Insert`, :class:`_expression.Update`,
           :class:`_expression.Delete`
         * :class:`_expression.TextClause` and
           :class:`_expression.TextualSelect`
         * :class:`_schema.DDL` and objects which inherit from
           :class:`_schema.ExecutableDDLElement`

        :param parameters: parameters which will be bound into the statement.
         This may be either a dictionary of parameter names to values,
         or a mutable sequence (e.g. a list) of dictionaries.  When a
         list of dictionaries is passed, the underlying statement execution
         will make use of the DBAPI ``cursor.executemany()`` method.
         When a single dictionary is passed, the DBAPI ``cursor.execute()``
         method will be used.

        :param execution_options: optional dictionary of execution options,
         which will be associated with the statement execution.  This
         dictionary can provide a subset of the options that are accepted
         by :meth:`_engine.Connection.execution_options`.

        :return: a :class:`_engine.Result` object.

        """
        distilled_parameters = _distill_params_20(parameters)
        try:
            meth = statement._execute_on_connection
        except AttributeError as err:
            raise exc.ObjectNotExecutableError(statement) from err
        else:
            return meth(
                self,
                distilled_parameters,
                execution_options or NO_OPTIONS,
            )

    def _execute_function(
        self,
        func: FunctionElement[Any],
        distilled_parameters: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> CursorResult[Any]:
        """Execute a sql.FunctionElement object."""

        return self._execute_clauseelement(
            func.select(), distilled_parameters, execution_options
        )

    def _execute_default(
        self,
        default: DefaultGenerator,
        distilled_parameters: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> Any:
        """Execute a schema.ColumnDefault object."""

        execution_options = self._execution_options.merge_with(
            execution_options
        )

        event_multiparams: Optional[_CoreMultiExecuteParams]
        event_params: Optional[_CoreAnyExecuteParams]

        # note for event handlers, the "distilled parameters" which is always
        # a list of dicts is broken out into separate "multiparams" and
        # "params" collections, which allows the handler to distinguish
        # between an executemany and execute style set of parameters.
        if self._has_events or self.engine._has_events:
            (
                default,
                distilled_parameters,
                event_multiparams,
                event_params,
            ) = self._invoke_before_exec_event(
                default, distilled_parameters, execution_options
            )
        else:
            event_multiparams = event_params = None

        try:
            conn = self._dbapi_connection
            if conn is None:
                conn = self._revalidate_connection()

            dialect = self.dialect
            ctx = dialect.execution_ctx_cls._init_default(
                dialect, self, conn, execution_options
            )
        except (exc.PendingRollbackError, exc.ResourceClosedError):
            raise
        except BaseException as e:
            self._handle_dbapi_exception(e, None, None, None, None)

        ret = ctx._exec_default(None, default, None)

        if self._has_events or self.engine._has_events:
            self.dispatch.after_execute(
                self,
                default,
                event_multiparams,
                event_params,
                execution_options,
                ret,
            )

        return ret

    def _execute_ddl(
        self,
        ddl: ExecutableDDLElement,
        distilled_parameters: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> CursorResult[Any]:
        """Execute a schema.DDL object."""

        exec_opts = ddl._execution_options.merge_with(
            self._execution_options, execution_options
        )

        event_multiparams: Optional[_CoreMultiExecuteParams]
        event_params: Optional[_CoreSingleExecuteParams]

        if self._has_events or self.engine._has_events:
            (
                ddl,
                distilled_parameters,
                event_multiparams,
                event_params,
            ) = self._invoke_before_exec_event(
                ddl, distilled_parameters, exec_opts
            )
        else:
            event_multiparams = event_params = None

        schema_translate_map = exec_opts.get("schema_translate_map", None)

        dialect = self.dialect

        compiled = ddl.compile(
            dialect=dialect, schema_translate_map=schema_translate_map
        )
        ret = self._execute_context(
            dialect,
            dialect.execution_ctx_cls._init_ddl,
            compiled,
            None,
            exec_opts,
            compiled,
        )
        if self._has_events or self.engine._has_events:
            self.dispatch.after_execute(
                self,
                ddl,
                event_multiparams,
                event_params,
                exec_opts,
                ret,
            )
        return ret

    def _invoke_before_exec_event(
        self,
        elem: Any,
        distilled_params: _CoreMultiExecuteParams,
        execution_options: _ExecuteOptions,
    ) -> Tuple[
        Any,
        _CoreMultiExecuteParams,
        _CoreMultiExecuteParams,
        _CoreSingleExecuteParams,
    ]:
        event_multiparams: _CoreMultiExecuteParams
        event_params: _CoreSingleExecuteParams

        if len(distilled_params) == 1:
            event_multiparams, event_params = [], distilled_params[0]
        else:
            event_multiparams, event_params = distilled_params, {}

        for fn in self.dispatch.before_execute:
            elem, event_multiparams, event_params = fn(
                self,
                elem,
                event_multiparams,
                event_params,
                execution_options,
            )

        if event_multiparams:
            distilled_params = list(event_multiparams)
            if event_params:
                raise exc.InvalidRequestError(
                    "Event handler can't return non-empty multiparams "
                    "and params at the same time"
                )
        elif event_params:
            distilled_params = [event_params]
        else:
            distilled_params = []

        return elem, distilled_params, event_multiparams, event_params

    def _execute_clauseelement(
        self,
        elem: Executable,
        distilled_parameters: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter,
    ) -> CursorResult[Any]:
        """Execute a sql.ClauseElement object."""

        execution_options = elem._execution_options.merge_with(
            self._execution_options, execution_options
        )

        has_events = self._has_events or self.engine._has_events
        if has_events:
            (
                elem,
                distilled_parameters,
                event_multiparams,
                event_params,
            ) = self._invoke_before_exec_event(
                elem, distilled_parameters, execution_options
            )

        if distilled_parameters:
            # ensure we don't retain a link to the view object for keys()
            # which links to the values, which we don't want to cache
            keys = sorted(distilled_parameters[0])
            for_executemany = len(distilled_parameters) > 1
        else:
            keys = []
            for_executemany = False

        dialect = self.dialect

        schema_translate_map = execution_options.get(
            "schema_translate_map", None
        )

        compiled_cache: Optional[CompiledCacheType] = execution_options.get(
            "compiled_cache", self.engine._compiled_cache
        )

        compiled_sql, extracted_params, cache_hit = elem._compile_w_cache(
            dialect=dialect,
            compiled_cache=compiled_cache,
            column_keys=keys,
            for_executemany=for_executemany,
            schema_translate_map=schema_translate_map,
            linting=self.dialect.compiler_linting | compiler.WARN_LINTING,
        )
        ret = self._execute_context(
            dialect,
            dialect.execution_ctx_cls._init_compiled,
            compiled_sql,
            distilled_parameters,
            execution_options,
            compiled_sql,
            distilled_parameters,
            elem,
            extracted_params,
            cache_hit=cache_hit,
        )
        if has_events:
            self.dispatch.after_execute(
                self,
                elem,
                event_multiparams,
                event_params,
                execution_options,
                ret,
            )
        return ret

    def _execute_compiled(
        self,
        compiled: Compiled,
        distilled_parameters: _CoreMultiExecuteParams,
        execution_options: CoreExecuteOptionsParameter = _EMPTY_EXECUTION_OPTS,
    ) -> CursorResult[Any]:
        """Execute a sql.Compiled object.

        TODO: why do we have this?   likely deprecate or remove

        """

        execution_options = compiled.execution_options.merge_with(
            self._execution_options, execution_options
        )

        if self._has_events or self.engine._has_events:
            (
                compiled,
                distilled_parameters,
                event_multiparams,
                event_params,
            ) = self._invoke_before_exec_event(
                compiled, distilled_parameters, execution_options
            )

        dialect = self.dialect

        ret = self._execute_context(
            dialect,
            dialect.execution_ctx_cls._init_compiled,
            compiled,
            distilled_parameters,
            execution_options,
            compiled,
            distilled_parameters,
            None,
            None,
        )
        if self._has_events or self.engine._has_events:
            self.dispatch.after_execute(
                self,
                compiled,
                event_multiparams,
                event_params,
                execution_options,
                ret,
            )
        return ret

    def exec_driver_sql(
        self,
        statement: str,
        parameters: Optional[_DBAPIAnyExecuteParams] = None,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
    ) -> CursorResult[Any]:
        r"""Executes a string SQL statement on the DBAPI cursor directly,
        without any SQL compilation steps.

        This can be used to pass any string directly to the
        ``cursor.execute()`` method of the DBAPI in use.

        :param statement: The statement str to be executed.   Bound parameters
         must use the underlying DBAPI's paramstyle, such as "qmark",
         "pyformat", "format", etc.

        :param parameters: represent bound parameter values to be used in the
         execution.  The format is one of:   a dictionary of named parameters,
         a tuple of positional parameters, or a list containing either
         dictionaries or tuples for multiple-execute support.

        :return: a :class:`_engine.CursorResult`.

         E.g. multiple dictionaries::


             conn.exec_driver_sql(
                 "INSERT INTO table (id, value) VALUES (%(id)s, %(value)s)",
                 [{"id": 1, "value": "v1"}, {"id": 2, "value": "v2"}],
             )

         Single dictionary::

             conn.exec_driver_sql(
                 "INSERT INTO table (id, value) VALUES (%(id)s, %(value)s)",
                 dict(id=1, value="v1"),
             )

         Single tuple::

             conn.exec_driver_sql(
                 "INSERT INTO table (id, value) VALUES (?, ?)", (1, "v1")
             )

         .. note:: The :meth:`_engine.Connection.exec_driver_sql` method does
             not participate in the
             :meth:`_events.ConnectionEvents.before_execute` and
             :meth:`_events.ConnectionEvents.after_execute` events.   To
             intercept calls to :meth:`_engine.Connection.exec_driver_sql`, use
             :meth:`_events.ConnectionEvents.before_cursor_execute` and
             :meth:`_events.ConnectionEvents.after_cursor_execute`.

         .. seealso::

            :pep:`249`

        """

        distilled_parameters = _distill_raw_params(parameters)

        execution_options = self._execution_options.merge_with(
            execution_options
        )

        dialect = self.dialect
        ret = self._execute_context(
            dialect,
            dialect.execution_ctx_cls._init_statement,
            statement,
            None,
            execution_options,
            statement,
            distilled_parameters,
        )

        return ret

    def _execute_context(
        self,
        dialect: Dialect,
        constructor: Callable[..., ExecutionContext],
        statement: Union[str, Compiled],
        parameters: Optional[_AnyMultiExecuteParams],
        execution_options: _ExecuteOptions,
        *args: Any,
        **kw: Any,
    ) -> CursorResult[Any]:
        """Create an :class:`.ExecutionContext` and execute, returning
        a :class:`_engine.CursorResult`."""

        if execution_options:
            yp = execution_options.get("yield_per", None)
            if yp:
                execution_options = execution_options.union(
                    {"stream_results": True, "max_row_buffer": yp}
                )
        try:
            conn = self._dbapi_connection
            if conn is None:
                conn = self._revalidate_connection()

            context = constructor(
                dialect, self, conn, execution_options, *args, **kw
            )
        except (exc.PendingRollbackError, exc.ResourceClosedError):
            raise
        except BaseException as e:
            self._handle_dbapi_exception(
                e, str(statement), parameters, None, None
            )

        if (
            self._transaction
            and not self._transaction.is_active
            or (
                self._nested_transaction
                and not self._nested_transaction.is_active
            )
        ):
            self._invalid_transaction()

        elif self._trans_context_manager:
            TransactionalContext._trans_ctx_check(self)

        if self._transaction is None:
            self._autobegin()

        context.pre_exec()

        if context.execute_style is ExecuteStyle.INSERTMANYVALUES:
            return self._exec_insertmany_context(dialect, context)
        else:
            return self._exec_single_context(
                dialect, context, statement, parameters
            )

    def _exec_single_context(
        self,
        dialect: Dialect,
        context: ExecutionContext,
        statement: Union[str, Compiled],
        parameters: Optional[_AnyMultiExecuteParams],
    ) -> CursorResult[Any]:
        """continue the _execute_context() method for a single DBAPI
        cursor.execute() or cursor.executemany() call.

        """
        if dialect.bind_typing is BindTyping.SETINPUTSIZES:
            generic_setinputsizes = context._prepare_set_input_sizes()

            if generic_setinputsizes:
                try:
                    dialect.do_set_input_sizes(
                        context.cursor, generic_setinputsizes, context
                    )
                except BaseException as e:
                    self._handle_dbapi_exception(
                        e, str(statement), parameters, None, context
                    )

        cursor, str_statement, parameters = (
            context.cursor,
            context.statement,
            context.parameters,
        )

        effective_parameters: Optional[_AnyExecuteParams]

        if not context.executemany:
            effective_parameters = parameters[0]
        else:
            effective_parameters = parameters

        if self._has_events or self.engine._has_events:
            for fn in self.dispatch.before_cursor_execute:
                str_statement, effective_parameters = fn(
                    self,
                    cursor,
                    str_statement,
                    effective_parameters,
                    context,
                    context.executemany,
                )

        if self._echo:
            self._log_info(str_statement)

            stats = context._get_cache_stats()

            if not self.engine.hide_parameters:
                self._log_info(
                    "[%s] %r",
                    stats,
                    sql_util._repr_params(
                        effective_parameters,
                        batches=10,
                        ismulti=context.executemany,
                    ),
                )
            else:
                self._log_info(
                    "[%s] [SQL parameters hidden due to hide_parameters=True]",
                    stats,
                )

        evt_handled: bool = False
        try:
            if context.execute_style is ExecuteStyle.EXECUTEMANY:
                effective_parameters = cast(
                    "_CoreMultiExecuteParams", effective_parameters
                )
                if self.dialect._has_events:
                    for fn in self.dialect.dispatch.do_executemany:
                        if fn(
                            cursor,
                            str_statement,
                            effective_parameters,
                            context,
                        ):
                            evt_handled = True
                            break
                if not evt_handled:
                    self.dialect.do_executemany(
                        cursor,
                        str_statement,
                        effective_parameters,
                        context,
                    )
            elif not effective_parameters and context.no_parameters:
                if self.dialect._has_events:
                    for fn in self.dialect.dispatch.do_execute_no_params:
                        if fn(cursor, str_statement, context):
                            evt_handled = True
                            break
                if not evt_handled:
                    self.dialect.do_execute_no_params(
                        cursor, str_statement, context
                    )
            else:
                effective_parameters = cast(
                    "_CoreSingleExecuteParams", effective_parameters
                )
                if self.dialect._has_events:
                    for fn in self.dialect.dispatch.do_execute:
                        if fn(
                            cursor,
                            str_statement,
                            effective_parameters,
                            context,
                        ):
                            evt_handled = True
                            break
                if not evt_handled:
                    self.dialect.do_execute(
                        cursor, str_statement, effective_parameters, context
                    )

            if self._has_events or self.engine._has_events:
                self.dispatch.after_cursor_execute(
                    self,
                    cursor,
                    str_statement,
                    effective_parameters,
                    context,
                    context.executemany,
                )

            context.post_exec()

            result = context._setup_result_proxy()

        except BaseException as e:
            self._handle_dbapi_exception(
                e, str_statement, effective_parameters, cursor, context
            )

        return result

    def _exec_insertmany_context(
        self,
        dialect: Dialect,
        context: ExecutionContext,
    ) -> CursorResult[Any]:
        """continue the _execute_context() method for an "insertmanyvalues"
        operation, which will invoke DBAPI
        cursor.execute() one or more times with individual log and
        event hook calls.

        """

        if dialect.bind_typing is BindTyping.SETINPUTSIZES:
            generic_setinputsizes = context._prepare_set_input_sizes()
        else:
            generic_setinputsizes = None

        cursor, str_statement, parameters = (
            context.cursor,
            context.statement,
            context.parameters,
        )

        effective_parameters = parameters

        engine_events = self._has_events or self.engine._has_events
        if self.dialect._has_events:
            do_execute_dispatch: Iterable[Any] = (
                self.dialect.dispatch.do_execute
            )
        else:
            do_execute_dispatch = ()

        if self._echo:
            stats = context._get_cache_stats() + " (insertmanyvalues)"

        preserve_rowcount = context.execution_options.get(
            "preserve_rowcount", False
        )
        rowcount = 0

        for imv_batch in dialect._deliver_insertmanyvalues_batches(
            self,
            cursor,
            str_statement,
            effective_parameters,
            generic_setinputsizes,
            context,
        ):
            if imv_batch.processed_setinputsizes:
                try:
                    dialect.do_set_input_sizes(
                        context.cursor,
                        imv_batch.processed_setinputsizes,
                        context,
                    )
                except BaseException as e:
                    self._handle_dbapi_exception(
                        e,
                        sql_util._long_statement(imv_batch.replaced_statement),
                        imv_batch.replaced_parameters,
                        None,
                        context,
                        is_sub_exec=True,
                    )

            sub_stmt = imv_batch.replaced_statement
            sub_params = imv_batch.replaced_parameters

            if engine_events:
                for fn in self.dispatch.before_cursor_execute:
                    sub_stmt, sub_params = fn(
                        self,
                        cursor,
                        sub_stmt,
                        sub_params,
                        context,
                        True,
                    )

            if self._echo:
                self._log_info(sql_util._long_statement(sub_stmt))

                imv_stats = f""" {imv_batch.batchnum}/{
                            imv_batch.total_batches
                } ({
                    'ordered'
                    if imv_batch.rows_sorted else 'unordered'
                }{
                    '; batch not supported'
                    if imv_batch.is_downgraded
                    else ''
                })"""

                if imv_batch.batchnum == 1:
                    stats += imv_stats
                else:
                    stats = f"insertmanyvalues{imv_stats}"

                if not self.engine.hide_parameters:
                    self._log_info(
                        "[%s] %r",
                        stats,
                        sql_util._repr_params(
                            sub_params,
                            batches=10,
                            ismulti=False,
                        ),
                    )
                else:
                    self._log_info(
                        "[%s] [SQL parameters hidden due to "
                        "hide_parameters=True]",
                        stats,
                    )

            try:
                for fn in do_execute_dispatch:
                    if fn(
                        cursor,
                        sub_stmt,
                        sub_params,
                        context,
                    ):
                        break
                else:
                    dialect.do_execute(
                        cursor,
                        sub_stmt,
                        sub_params,
                        context,
                    )

            except BaseException as e:
                self._handle_dbapi_exception(
                    e,
                    sql_util._long_statement(sub_stmt),
                    sub_params,
                    cursor,
                    context,
                    is_sub_exec=True,
                )

            if engine_events:
                self.dispatch.after_cursor_execute(
                    self,
                    cursor,
                    str_statement,
                    effective_parameters,
                    context,
                    context.executemany,
                )

            if preserve_rowcount:
                rowcount += imv_batch.current_batch_size

        try:
            context.post_exec()

            if preserve_rowcount:
                context._rowcount = rowcount  # type: ignore[attr-defined]

            result = context._setup_result_proxy()

        except BaseException as e:
            self._handle_dbapi_exception(
                e, str_statement, effective_parameters, cursor, context
            )

        return result

    def _cursor_execute(
        self,
        cursor: DBAPICursor,
        statement: str,
        parameters: _DBAPISingleExecuteParams,
        context: Optional[ExecutionContext] = None,
    ) -> None:
        """Execute a statement + params on the given cursor.

        Adds appropriate logging and exception handling.

        This method is used by DefaultDialect for special-case
        executions, such as for sequences and column defaults.
        The path of statement execution in the majority of cases
        terminates at _execute_context().

        """
        if self._has_events or self.engine._has_events:
            for fn in self.dispatch.before_cursor_execute:
                statement, parameters = fn(
                    self, cursor, statement, parameters, context, False
                )

        if self._echo:
            self._log_info(statement)
            self._log_info("[raw sql] %r", parameters)
        try:
            for fn in (
                ()
                if not self.dialect._has_events
                else self.dialect.dispatch.do_execute
            ):
                if fn(cursor, statement, parameters, context):
                    break
            else:
                self.dialect.do_execute(cursor, statement, parameters, context)
        except BaseException as e:
            self._handle_dbapi_exception(
                e, statement, parameters, cursor, context
            )

        if self._has_events or self.engine._has_events:
            self.dispatch.after_cursor_execute(
                self, cursor, statement, parameters, context, False
            )

    def _safe_close_cursor(self, cursor: DBAPICursor) -> None:
        """Close the given cursor, catching exceptions
        and turning into log warnings.

        """
        try:
            cursor.close()
        except Exception:
            # log the error through the connection pool's logger.
            self.engine.pool.logger.error(
                "Error closing cursor", exc_info=True
            )

    _reentrant_error = False
    _is_disconnect = False

    def _handle_dbapi_exception(
        self,
        e: BaseException,
        statement: Optional[str],
        parameters: Optional[_AnyExecuteParams],
        cursor: Optional[DBAPICursor],
        context: Optional[ExecutionContext],
        is_sub_exec: bool = False,
    ) -> NoReturn:
        exc_info = sys.exc_info()

        is_exit_exception = util.is_exit_exception(e)

        if not self._is_disconnect:
            self._is_disconnect = (
                isinstance(e, self.dialect.loaded_dbapi.Error)
                and not self.closed
                and self.dialect.is_disconnect(
                    e,
                    self._dbapi_connection if not self.invalidated else None,
                    cursor,
                )
            ) or (is_exit_exception and not self.closed)

        invalidate_pool_on_disconnect = not is_exit_exception

        ismulti: bool = (
            not is_sub_exec and context.executemany
            if context is not None
            else False
        )
        if self._reentrant_error:
            raise exc.DBAPIError.instance(
                statement,
                parameters,
                e,
                self.dialect.loaded_dbapi.Error,
                hide_parameters=self.engine.hide_parameters,
                dialect=self.dialect,
                ismulti=ismulti,
            ).with_traceback(exc_info[2]) from e
        self._reentrant_error = True
        try:
            # non-DBAPI error - if we already got a context,
            # or there's no string statement, don't wrap it
            should_wrap = isinstance(e, self.dialect.loaded_dbapi.Error) or (
                statement is not None
                and context is None
                and not is_exit_exception
            )

            if should_wrap:
                sqlalchemy_exception = exc.DBAPIError.instance(
                    statement,
                    parameters,
                    cast(Exception, e),
                    self.dialect.loaded_dbapi.Error,
                    hide_parameters=self.engine.hide_parameters,
                    connection_invalidated=self._is_disconnect,
                    dialect=self.dialect,
                    ismulti=ismulti,
                )
            else:
                sqlalchemy_exception = None

            newraise = None

            if (self.dialect._has_events) and not self._execution_options.get(
                "skip_user_error_events", False
            ):
                ctx = ExceptionContextImpl(
                    e,
                    sqlalchemy_exception,
                    self.engine,
                    self.dialect,
                    self,
                    cursor,
                    statement,
                    parameters,
                    context,
                    self._is_disconnect,
                    invalidate_pool_on_disconnect,
                    False,
                )

                for fn in self.dialect.dispatch.handle_error:
                    try:
                        # handler returns an exception;
                        # call next handler in a chain
                        per_fn = fn(ctx)
                        if per_fn is not None:
                            ctx.chained_exception = newraise = per_fn
                    except Exception as _raised:
                        # handler raises an exception - stop processing
                        newraise = _raised
                        break

                if self._is_disconnect != ctx.is_disconnect:
                    self._is_disconnect = ctx.is_disconnect
                    if sqlalchemy_exception:
                        sqlalchemy_exception.connection_invalidated = (
                            ctx.is_disconnect
                        )

                # set up potentially user-defined value for
                # invalidate pool.
                invalidate_pool_on_disconnect = (
                    ctx.invalidate_pool_on_disconnect
                )

            if should_wrap and context:
                context.handle_dbapi_exception(e)

            if not self._is_disconnect:
                if cursor:
                    self._safe_close_cursor(cursor)
                # "autorollback" was mostly relevant in 1.x series.
                # It's very unlikely to reach here, as the connection
                # does autobegin so when we are here, we are usually
                # in an explicit / semi-explicit transaction.
                # however we have a test which manufactures this
                # scenario in any case using an event handler.
                # test/engine/test_execute.py-> test_actual_autorollback
                if not self.in_transaction():
                    self._rollback_impl()

            if newraise:
                raise newraise.with_traceback(exc_info[2]) from e
            elif should_wrap:
                assert sqlalchemy_exception is not None
                raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
            else:
                assert exc_info[1] is not None
                raise exc_info[1].with_traceback(exc_info[2])
        finally:
            del self._reentrant_error
            if self._is_disconnect:
                del self._is_disconnect
                if not self.invalidated:
                    dbapi_conn_wrapper = self._dbapi_connection
                    assert dbapi_conn_wrapper is not None
                    if invalidate_pool_on_disconnect:
                        self.engine.pool._invalidate(dbapi_conn_wrapper, e)
                    self.invalidate(e)

    @classmethod
    def _handle_dbapi_exception_noconnection(
        cls,
        e: BaseException,
        dialect: Dialect,
        engine: Optional[Engine] = None,
        is_disconnect: Optional[bool] = None,
        invalidate_pool_on_disconnect: bool = True,
        is_pre_ping: bool = False,
    ) -> NoReturn:
        exc_info = sys.exc_info()

        if is_disconnect is None:
            is_disconnect = isinstance(
                e, dialect.loaded_dbapi.Error
            ) and dialect.is_disconnect(e, None, None)

        should_wrap = isinstance(e, dialect.loaded_dbapi.Error)

        if should_wrap:
            sqlalchemy_exception = exc.DBAPIError.instance(
                None,
                None,
                cast(Exception, e),
                dialect.loaded_dbapi.Error,
                hide_parameters=(
                    engine.hide_parameters if engine is not None else False
                ),
                connection_invalidated=is_disconnect,
                dialect=dialect,
            )
        else:
            sqlalchemy_exception = None

        newraise = None

        if dialect._has_events:
            ctx = ExceptionContextImpl(
                e,
                sqlalchemy_exception,
                engine,
                dialect,
                None,
                None,
                None,
                None,
                None,
                is_disconnect,
                invalidate_pool_on_disconnect,
                is_pre_ping,
            )
            for fn in dialect.dispatch.handle_error:
                try:
                    # handler returns an exception;
                    # call next handler in a chain
                    per_fn = fn(ctx)
                    if per_fn is not None:
                        ctx.chained_exception = newraise = per_fn
                except Exception as _raised:
                    # handler raises an exception - stop processing
                    newraise = _raised
                    break

            if sqlalchemy_exception and is_disconnect != ctx.is_disconnect:
                sqlalchemy_exception.connection_invalidated = ctx.is_disconnect

        if newraise:
            raise newraise.with_traceback(exc_info[2]) from e
        elif should_wrap:
            assert sqlalchemy_exception is not None
            raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
        else:
            assert exc_info[1] is not None
            raise exc_info[1].with_traceback(exc_info[2])

    def _run_ddl_visitor(
        self,
        visitorcallable: Type[InvokeDDLBase],
        element: SchemaVisitable,
        **kwargs: Any,
    ) -> None:
        """run a DDL visitor.

        This method is only here so that the MockConnection can change the
        options given to the visitor so that "checkfirst" is skipped.

        """
        visitorcallable(
            dialect=self.dialect, connection=self, **kwargs
        ).traverse_single(element)


class ExceptionContextImpl(ExceptionContext):
    """Implement the :class:`.ExceptionContext` interface."""

    __slots__ = (
        "connection",
        "engine",
        "dialect",
        "cursor",
        "statement",
        "parameters",
        "original_exception",
        "sqlalchemy_exception",
        "chained_exception",
        "execution_context",
        "is_disconnect",
        "invalidate_pool_on_disconnect",
        "is_pre_ping",
    )

    def __init__(
        self,
        exception: BaseException,
        sqlalchemy_exception: Optional[exc.StatementError],
        engine: Optional[Engine],
        dialect: Dialect,
        connection: Optional[Connection],
        cursor: Optional[DBAPICursor],
        statement: Optional[str],
        parameters: Optional[_DBAPIAnyExecuteParams],
        context: Optional[ExecutionContext],
        is_disconnect: bool,
        invalidate_pool_on_disconnect: bool,
        is_pre_ping: bool,
    ):
        self.engine = engine
        self.dialect = dialect
        self.connection = connection
        self.sqlalchemy_exception = sqlalchemy_exception
        self.original_exception = exception
        self.execution_context = context
        self.statement = statement
        self.parameters = parameters
        self.is_disconnect = is_disconnect
        self.invalidate_pool_on_disconnect = invalidate_pool_on_disconnect
        self.is_pre_ping = is_pre_ping


class Transaction(TransactionalContext):
    """Represent a database transaction in progress.

    The :class:`.Transaction` object is procured by
    calling the :meth:`_engine.Connection.begin` method of
    :class:`_engine.Connection`::

        from sqlalchemy import create_engine

        engine = create_engine("postgresql+psycopg2://scott:tiger@localhost/test")
        connection = engine.connect()
        trans = connection.begin()
        connection.execute(text("insert into x (a, b) values (1, 2)"))
        trans.commit()

    The object provides :meth:`.rollback` and :meth:`.commit`
    methods in order to control transaction boundaries.  It
    also implements a context manager interface so that
    the Python ``with`` statement can be used with the
    :meth:`_engine.Connection.begin` method::

        with connection.begin():
            connection.execute(text("insert into x (a, b) values (1, 2)"))

    The Transaction object is **not** threadsafe.

    .. seealso::

        :meth:`_engine.Connection.begin`

        :meth:`_engine.Connection.begin_twophase`

        :meth:`_engine.Connection.begin_nested`

    .. index::
      single: thread safety; Transaction
    """  # noqa

    __slots__ = ()

    _is_root: bool = False
    is_active: bool
    connection: Connection

    def __init__(self, connection: Connection):
        raise NotImplementedError()

    @property
    def _deactivated_from_connection(self) -> bool:
        """True if this transaction is totally deactivated from the connection
        and therefore can no longer affect its state.

        """
        raise NotImplementedError()

    def _do_close(self) -> None:
        raise NotImplementedError()

    def _do_rollback(self) -> None:
        raise NotImplementedError()

    def _do_commit(self) -> None:
        raise NotImplementedError()

    @property
    def is_valid(self) -> bool:
        return self.is_active and not self.connection.invalidated

    def close(self) -> None:
        """Close this :class:`.Transaction`.

        If this transaction is the base transaction in a begin/commit
        nesting, the transaction will rollback().  Otherwise, the
        method returns.

        This is used to cancel a Transaction without affecting the scope of
        an enclosing transaction.

        """
        try:
            self._do_close()
        finally:
            assert not self.is_active

    def rollback(self) -> None:
        """Roll back this :class:`.Transaction`.

        The implementation of this may vary based on the type of transaction in
        use:

        * For a simple database transaction (e.g. :class:`.RootTransaction`),
          it corresponds to a ROLLBACK.

        * For a :class:`.NestedTransaction`, it corresponds to a
          "ROLLBACK TO SAVEPOINT" operation.

        * For a :class:`.TwoPhaseTransaction`, DBAPI-specific methods for two
          phase transactions may be used.


        """
        try:
            self._do_rollback()
        finally:
            assert not self.is_active

    def commit(self) -> None:
        """Commit this :class:`.Transaction`.

        The implementation of this may vary based on the type of transaction in
        use:

        * For a simple database transaction (e.g. :class:`.RootTransaction`),
          it corresponds to a COMMIT.

        * For a :class:`.NestedTransaction`, it corresponds to a
          "RELEASE SAVEPOINT" operation.

        * For a :class:`.TwoPhaseTransaction`, DBAPI-specific methods for two
          phase transactions may be used.

        """
        try:
            self._do_commit()
        finally:
            assert not self.is_active

    def _get_subject(self) -> Connection:
        return self.connection

    def _transaction_is_active(self) -> bool:
        return self.is_active

    def _transaction_is_closed(self) -> bool:
        return not self._deactivated_from_connection

    def _rollback_can_be_called(self) -> bool:
        # for RootTransaction / NestedTransaction, it's safe to call
        # rollback() even if the transaction is deactive and no warnings
        # will be emitted.  tested in
        # test_transaction.py -> test_no_rollback_in_deactive(?:_savepoint)?
        return True


class RootTransaction(Transaction):
    """Represent the "root" transaction on a :class:`_engine.Connection`.

    This corresponds to the current "BEGIN/COMMIT/ROLLBACK" that's occurring
    for the :class:`_engine.Connection`. The :class:`_engine.RootTransaction`
    is created by calling upon the :meth:`_engine.Connection.begin` method, and
    remains associated with the :class:`_engine.Connection` throughout its
    active span. The current :class:`_engine.RootTransaction` in use is
    accessible via the :attr:`_engine.Connection.get_transaction` method of
    :class:`_engine.Connection`.

    In :term:`2.0 style` use, the :class:`_engine.Connection` also employs
    "autobegin" behavior that will create a new
    :class:`_engine.RootTransaction` whenever a connection in a
    non-transactional state is used to emit commands on the DBAPI connection.
    The scope of the :class:`_engine.RootTransaction` in 2.0 style
    use can be controlled using the :meth:`_engine.Connection.commit` and
    :meth:`_engine.Connection.rollback` methods.


    """

    _is_root = True

    __slots__ = ("connection", "is_active")

    def __init__(self, connection: Connection):
        assert connection._transaction is None
        if connection._trans_context_manager:
            TransactionalContext._trans_ctx_check(connection)
        self.connection = connection
        self._connection_begin_impl()
        connection._transaction = self

        self.is_active = True

    def _deactivate_from_connection(self) -> None:
        if self.is_active:
            assert self.connection._transaction is self
            self.is_active = False

        elif self.connection._transaction is not self:
            util.warn("transaction already deassociated from connection")

    @property
    def _deactivated_from_connection(self) -> bool:
        return self.connection._transaction is not self

    def _connection_begin_impl(self) -> None:
        self.connection._begin_impl(self)

    def _connection_rollback_impl(self) -> None:
        self.connection._rollback_impl()

    def _connection_commit_impl(self) -> None:
        self.connection._commit_impl()

    def _close_impl(self, try_deactivate: bool = False) -> None:
        try:
            if self.is_active:
                self._connection_rollback_impl()

            if self.connection._nested_transaction:
                self.connection._nested_transaction._cancel()
        finally:
            if self.is_active or try_deactivate:
                self._deactivate_from_connection()
            if self.connection._transaction is self:
                self.connection._transaction = None

        assert not self.is_active
        assert self.connection._transaction is not self

    def _do_close(self) -> None:
        self._close_impl()

    def _do_rollback(self) -> None:
        self._close_impl(try_deactivate=True)

    def _do_commit(self) -> None:
        if self.is_active:
            assert self.connection._transaction is self

            try:
                self._connection_commit_impl()
            finally:
                # whether or not commit succeeds, cancel any
                # nested transactions, make this transaction "inactive"
                # and remove it as a reset agent
                if self.connection._nested_transaction:
                    self.connection._nested_transaction._cancel()

                self._deactivate_from_connection()

            # ...however only remove as the connection's current transaction
            # if commit succeeded.  otherwise it stays on so that a rollback
            # needs to occur.
            self.connection._transaction = None
        else:
            if self.connection._transaction is self:
                self.connection._invalid_transaction()
            else:
                raise exc.InvalidRequestError("This transaction is inactive")

        assert not self.is_active
        assert self.connection._transaction is not self


class NestedTransaction(Transaction):
    """Represent a 'nested', or SAVEPOINT transaction.

    The :class:`.NestedTransaction` object is created by calling the
    :meth:`_engine.Connection.begin_nested` method of
    :class:`_engine.Connection`.

    When using :class:`.NestedTransaction`, the semantics of "begin" /
    "commit" / "rollback" are as follows:

    * the "begin" operation corresponds to the "BEGIN SAVEPOINT" command, where
      the savepoint is given an explicit name that is part of the state
      of this object.

    * The :meth:`.NestedTransaction.commit` method corresponds to a
      "RELEASE SAVEPOINT" operation, using the savepoint identifier associated
      with this :class:`.NestedTransaction`.

    * The :meth:`.NestedTransaction.rollback` method corresponds to a
      "ROLLBACK TO SAVEPOINT" operation, using the savepoint identifier
      associated with this :class:`.NestedTransaction`.

    The rationale for mimicking the semantics of an outer transaction in
    terms of savepoints so that code may deal with a "savepoint" transaction
    and an "outer" transaction in an agnostic way.

    .. seealso::

        :ref:`session_begin_nested` - ORM version of the SAVEPOINT API.

    """

    __slots__ = ("connection", "is_active", "_savepoint", "_previous_nested")

    _savepoint: str

    def __init__(self, connection: Connection):
        assert connection._transaction is not None
        if connection._trans_context_manager:
            TransactionalContext._trans_ctx_check(connection)
        self.connection = connection
        self._savepoint = self.connection._savepoint_impl()
        self.is_active = True
        self._previous_nested = connection._nested_transaction
        connection._nested_transaction = self

    def _deactivate_from_connection(self, warn: bool = True) -> None:
        if self.connection._nested_transaction is self:
            self.connection._nested_transaction = self._previous_nested
        elif warn:
            util.warn(
                "nested transaction already deassociated from connection"
            )

    @property
    def _deactivated_from_connection(self) -> bool:
        return self.connection._nested_transaction is not self

    def _cancel(self) -> None:
        # called by RootTransaction when the outer transaction is
        # committed, rolled back, or closed to cancel all savepoints
        # without any action being taken
        self.is_active = False
        self._deactivate_from_connection()
        if self._previous_nested:
            self._previous_nested._cancel()

    def _close_impl(
        self, deactivate_from_connection: bool, warn_already_deactive: bool
    ) -> None:
        try:
            if (
                self.is_active
                and self.connection._transaction
                and self.connection._transaction.is_active
            ):
                self.connection._rollback_to_savepoint_impl(self._savepoint)
        finally:
            self.is_active = False

            if deactivate_from_connection:
                self._deactivate_from_connection(warn=warn_already_deactive)

        assert not self.is_active
        if deactivate_from_connection:
            assert self.connection._nested_transaction is not self

    def _do_close(self) -> None:
        self._close_impl(True, False)

    def _do_rollback(self) -> None:
        self._close_impl(True, True)

    def _do_commit(self) -> None:
        if self.is_active:
            try:
                self.connection._release_savepoint_impl(self._savepoint)
            finally:
                # nested trans becomes inactive on failed release
                # unconditionally.  this prevents it from trying to
                # emit SQL when it rolls back.
                self.is_active = False

            # but only de-associate from connection if it succeeded
            self._deactivate_from_connection()
        else:
            if self.connection._nested_transaction is self:
                self.connection._invalid_transaction()
            else:
                raise exc.InvalidRequestError(
                    "This nested transaction is inactive"
                )


class TwoPhaseTransaction(RootTransaction):
    """Represent a two-phase transaction.

    A new :class:`.TwoPhaseTransaction` object may be procured
    using the :meth:`_engine.Connection.begin_twophase` method.

    The interface is the same as that of :class:`.Transaction`
    with the addition of the :meth:`prepare` method.

    """

    __slots__ = ("xid", "_is_prepared")

    xid: Any

    def __init__(self, connection: Connection, xid: Any):
        self._is_prepared = False
        self.xid = xid
        super().__init__(connection)

    def prepare(self) -> None:
        """Prepare this :class:`.TwoPhaseTransaction`.

        After a PREPARE, the transaction can be committed.

        """
        if not self.is_active:
            raise exc.InvalidRequestError("This transaction is inactive")
        self.connection._prepare_twophase_impl(self.xid)
        self._is_prepared = True

    def _connection_begin_impl(self) -> None:
        self.connection._begin_twophase_impl(self)

    def _connection_rollback_impl(self) -> None:
        self.connection._rollback_twophase_impl(self.xid, self._is_prepared)

    def _connection_commit_impl(self) -> None:
        self.connection._commit_twophase_impl(self.xid, self._is_prepared)


class Engine(
    ConnectionEventsTarget, log.Identified, inspection.Inspectable["Inspector"]
):
    """
    Connects a :class:`~sqlalchemy.pool.Pool` and
    :class:`~sqlalchemy.engine.interfaces.Dialect` together to provide a
    source of database connectivity and behavior.

    An :class:`_engine.Engine` object is instantiated publicly using the
    :func:`~sqlalchemy.create_engine` function.

    .. seealso::

        :doc:`/core/engines`

        :ref:`connections_toplevel`

    """

    dispatch: dispatcher[ConnectionEventsTarget]

    _compiled_cache: Optional[CompiledCacheType]

    _execution_options: _ExecuteOptions = _EMPTY_EXECUTION_OPTS
    _has_events: bool = False
    _connection_cls: Type[Connection] = Connection
    _sqla_logger_namespace: str = "sqlalchemy.engine.Engine"
    _is_future: bool = False

    _schema_translate_map: Optional[SchemaTranslateMapType] = None
    _option_cls: Type[OptionEngine]

    dialect: Dialect
    pool: Pool
    url: URL
    hide_parameters: bool

    def __init__(
        self,
        pool: Pool,
        dialect: Dialect,
        url: URL,
        logging_name: Optional[str] = None,
        echo: Optional[_EchoFlagType] = None,
        query_cache_size: int = 500,
        execution_options: Optional[Mapping[str, Any]] = None,
        hide_parameters: bool = False,
    ):
        self.pool = pool
        self.url = url
        self.dialect = dialect
        if logging_name:
            self.logging_name = logging_name
        self.echo = echo
        self.hide_parameters = hide_parameters
        if query_cache_size != 0:
            self._compiled_cache = util.LRUCache(
                query_cache_size, size_alert=self._lru_size_alert
            )
        else:
            self._compiled_cache = None
        log.instance_logger(self, echoflag=echo)
        if execution_options:
            self.update_execution_options(**execution_options)

    def _lru_size_alert(self, cache: util.LRUCache[Any, Any]) -> None:
        if self._should_log_info():
            self.logger.info(
                "Compiled cache size pruning from %d items to %d.  "
                "Increase cache size to reduce the frequency of pruning.",
                len(cache),
                cache.capacity,
            )

    @property
    def engine(self) -> Engine:
        """Returns this :class:`.Engine`.

        Used for legacy schemes that accept :class:`.Connection` /
        :class:`.Engine` objects within the same variable.

        """
        return self

    def clear_compiled_cache(self) -> None:
        """Clear the compiled cache associated with the dialect.

        This applies **only** to the built-in cache that is established
        via the :paramref:`_engine.create_engine.query_cache_size` parameter.
        It will not impact any dictionary caches that were passed via the
        :paramref:`.Connection.execution_options.compiled_cache` parameter.

        .. versionadded:: 1.4

        """
        if self._compiled_cache:
            self._compiled_cache.clear()

    def update_execution_options(self, **opt: Any) -> None:
        r"""Update the default execution_options dictionary
        of this :class:`_engine.Engine`.

        The given keys/values in \**opt are added to the
        default execution options that will be used for
        all connections.  The initial contents of this dictionary
        can be sent via the ``execution_options`` parameter
        to :func:`_sa.create_engine`.

        .. seealso::

            :meth:`_engine.Connection.execution_options`

            :meth:`_engine.Engine.execution_options`

        """
        self.dispatch.set_engine_execution_options(self, opt)
        self._execution_options = self._execution_options.union(opt)
        self.dialect.set_engine_execution_options(self, opt)

    @overload
    def execution_options(
        self,
        *,
        compiled_cache: Optional[CompiledCacheType] = ...,
        logging_token: str = ...,
        isolation_level: IsolationLevel = ...,
        insertmanyvalues_page_size: int = ...,
        schema_translate_map: Optional[SchemaTranslateMapType] = ...,
        **opt: Any,
    ) -> OptionEngine: ...

    @overload
    def execution_options(self, **opt: Any) -> OptionEngine: ...

    def execution_options(self, **opt: Any) -> OptionEngine:
        """Return a new :class:`_engine.Engine` that will provide
        :class:`_engine.Connection` objects with the given execution options.

        The returned :class:`_engine.Engine` remains related to the original
        :class:`_engine.Engine` in that it shares the same connection pool and
        other state:

        * The :class:`_pool.Pool` used by the new :class:`_engine.Engine`
          is the
          same instance.  The :meth:`_engine.Engine.dispose`
          method will replace
          the connection pool instance for the parent engine as well
          as this one.
        * Event listeners are "cascaded" - meaning, the new
          :class:`_engine.Engine`
          inherits the events of the parent, and new events can be associated
          with the new :class:`_engine.Engine` individually.
        * The logging configuration and logging_name is copied from the parent
          :class:`_engine.Engine`.

        The intent of the :meth:`_engine.Engine.execution_options` method is
        to implement schemes where multiple :class:`_engine.Engine`
        objects refer to the same connection pool, but are differentiated
        by options that affect some execution-level behavior for each
        engine.    One such example is breaking into separate "reader" and
        "writer" :class:`_engine.Engine` instances, where one
        :class:`_engine.Engine`
        has a lower :term:`isolation level` setting configured or is even
        transaction-disabled using "autocommit".  An example of this
        configuration is at :ref:`dbapi_autocommit_multiple`.

        Another example is one that
        uses a custom option ``shard_id`` which is consumed by an event
        to change the current schema on a database connection::

            from sqlalchemy import event
            from sqlalchemy.engine import Engine

            primary_engine = create_engine("mysql+mysqldb://")
            shard1 = primary_engine.execution_options(shard_id="shard1")
            shard2 = primary_engine.execution_options(shard_id="shard2")

            shards = {"default": "base", "shard_1": "db1", "shard_2": "db2"}


            @event.listens_for(Engine, "before_cursor_execute")
            def _switch_shard(conn, cursor, stmt, params, context, executemany):
                shard_id = conn.get_execution_options().get("shard_id", "default")
                current_shard = conn.info.get("current_shard", None)

                if current_shard != shard_id:
                    cursor.execute("use %s" % shards[shard_id])
                    conn.info["current_shard"] = shard_id

        The above recipe illustrates two :class:`_engine.Engine` objects that
        will each serve as factories for :class:`_engine.Connection` objects
        that have pre-established "shard_id" execution options present. A
        :meth:`_events.ConnectionEvents.before_cursor_execute` event handler
        then interprets this execution option to emit a MySQL ``use`` statement
        to switch databases before a statement execution, while at the same
        time keeping track of which database we've established using the
        :attr:`_engine.Connection.info` dictionary.

        .. seealso::

            :meth:`_engine.Connection.execution_options`
            - update execution options
            on a :class:`_engine.Connection` object.

            :meth:`_engine.Engine.update_execution_options`
            - update the execution
            options for a given :class:`_engine.Engine` in place.

            :meth:`_engine.Engine.get_execution_options`


        """  # noqa: E501
        return self._option_cls(self, opt)

    def get_execution_options(self) -> _ExecuteOptions:
        """Get the non-SQL options which will take effect during execution.

        .. versionadded: 1.3

        .. seealso::

            :meth:`_engine.Engine.execution_options`
        """
        return self._execution_options

    @property
    def name(self) -> str:
        """String name of the :class:`~sqlalchemy.engine.interfaces.Dialect`
        in use by this :class:`Engine`.

        """

        return self.dialect.name

    @property
    def driver(self) -> str:
        """Driver name of the :class:`~sqlalchemy.engine.interfaces.Dialect`
        in use by this :class:`Engine`.

        """

        return self.dialect.driver

    echo = log.echo_property()

    def __repr__(self) -> str:
        return "Engine(%r)" % (self.url,)

    def dispose(self, close: bool = True) -> None:
        """Dispose of the connection pool used by this
        :class:`_engine.Engine`.

        A new connection pool is created immediately after the old one has been
        disposed. The previous connection pool is disposed either actively, by
        closing out all currently checked-in connections in that pool, or
        passively, by losing references to it but otherwise not closing any
        connections. The latter strategy is more appropriate for an initializer
        in a forked Python process.

        :param close: if left at its default of ``True``, has the
         effect of fully closing all **currently checked in**
         database connections.  Connections that are still checked out
         will **not** be closed, however they will no longer be associated
         with this :class:`_engine.Engine`,
         so when they are closed individually, eventually the
         :class:`_pool.Pool` which they are associated with will
         be garbage collected and they will be closed out fully, if
         not already closed on checkin.

         If set to ``False``, the previous connection pool is de-referenced,
         and otherwise not touched in any way.

        .. versionadded:: 1.4.33  Added the :paramref:`.Engine.dispose.close`
            parameter to allow the replacement of a connection pool in a child
            process without interfering with the connections used by the parent
            process.


        .. seealso::

            :ref:`engine_disposal`

            :ref:`pooling_multiprocessing`

        """
        if close:
            self.pool.dispose()
        self.pool = self.pool.recreate()
        self.dispatch.engine_disposed(self)

    @contextlib.contextmanager
    def _optional_conn_ctx_manager(
        self, connection: Optional[Connection] = None
    ) -> Iterator[Connection]:
        if connection is None:
            with self.connect() as conn:
                yield conn
        else:
            yield connection

    @contextlib.contextmanager
    def begin(self) -> Iterator[Connection]:
        """Return a context manager delivering a :class:`_engine.Connection`
        with a :class:`.Transaction` established.

        E.g.::

            with engine.begin() as conn:
                conn.execute(text("insert into table (x, y, z) values (1, 2, 3)"))
                conn.execute(text("my_special_procedure(5)"))

        Upon successful operation, the :class:`.Transaction`
        is committed.  If an error is raised, the :class:`.Transaction`
        is rolled back.

        .. seealso::

            :meth:`_engine.Engine.connect` - procure a
            :class:`_engine.Connection` from
            an :class:`_engine.Engine`.

            :meth:`_engine.Connection.begin` - start a :class:`.Transaction`
            for a particular :class:`_engine.Connection`.

        """  # noqa: E501
        with self.connect() as conn:
            with conn.begin():
                yield conn

    def _run_ddl_visitor(
        self,
        visitorcallable: Type[InvokeDDLBase],
        element: SchemaVisitable,
        **kwargs: Any,
    ) -> None:
        with self.begin() as conn:
            conn._run_ddl_visitor(visitorcallable, element, **kwargs)

    def connect(self) -> Connection:
        """Return a new :class:`_engine.Connection` object.

        The :class:`_engine.Connection` acts as a Python context manager, so
        the typical use of this method looks like::

            with engine.connect() as connection:
                connection.execute(text("insert into table values ('foo')"))
                connection.commit()

        Where above, after the block is completed, the connection is "closed"
        and its underlying DBAPI resources are returned to the connection pool.
        This also has the effect of rolling back any transaction that
        was explicitly begun or was begun via autobegin, and will
        emit the :meth:`_events.ConnectionEvents.rollback` event if one was
        started and is still in progress.

        .. seealso::

            :meth:`_engine.Engine.begin`

        """

        return self._connection_cls(self)

    def raw_connection(self) -> PoolProxiedConnection:
        """Return a "raw" DBAPI connection from the connection pool.

        The returned object is a proxied version of the DBAPI
        connection object used by the underlying driver in use.
        The object will have all the same behavior as the real DBAPI
        connection, except that its ``close()`` method will result in the
        connection being returned to the pool, rather than being closed
        for real.

        This method provides direct DBAPI connection access for
        special situations when the API provided by
        :class:`_engine.Connection`
        is not needed.   When a :class:`_engine.Connection` object is already
        present, the DBAPI connection is available using
        the :attr:`_engine.Connection.connection` accessor.

        .. seealso::

            :ref:`dbapi_connections`

        """
        return self.pool.connect()


class OptionEngineMixin(log.Identified):
    _sa_propagate_class_events = False

    dispatch: dispatcher[ConnectionEventsTarget]
    _compiled_cache: Optional[CompiledCacheType]
    dialect: Dialect
    pool: Pool
    url: URL
    hide_parameters: bool
    echo: log.echo_property

    def __init__(
        self, proxied: Engine, execution_options: CoreExecuteOptionsParameter
    ):
        self._proxied = proxied
        self.url = proxied.url
        self.dialect = proxied.dialect
        self.logging_name = proxied.logging_name
        self.echo = proxied.echo
        self._compiled_cache = proxied._compiled_cache
        self.hide_parameters = proxied.hide_parameters
        log.instance_logger(self, echoflag=self.echo)

        # note: this will propagate events that are assigned to the parent
        # engine after this OptionEngine is created.   Since we share
        # the events of the parent we also disallow class-level events
        # to apply to the OptionEngine class directly.
        #
        # the other way this can work would be to transfer existing
        # events only, using:
        # self.dispatch._update(proxied.dispatch)
        #
        # that might be more appropriate however it would be a behavioral
        # change for logic that assigns events to the parent engine and
        # would like it to take effect for the already-created sub-engine.
        self.dispatch = self.dispatch._join(proxied.dispatch)

        self._execution_options = proxied._execution_options
        self.update_execution_options(**execution_options)

    def update_execution_options(self, **opt: Any) -> None:
        raise NotImplementedError()

    if not typing.TYPE_CHECKING:
        # https://github.com/python/typing/discussions/1095

        @property
        def pool(self) -> Pool:
            return self._proxied.pool

        @pool.setter
        def pool(self, pool: Pool) -> None:
            self._proxied.pool = pool

        @property
        def _has_events(self) -> bool:
            return self._proxied._has_events or self.__dict__.get(
                "_has_events", False
            )

        @_has_events.setter
        def _has_events(self, value: bool) -> None:
            self.__dict__["_has_events"] = value


class OptionEngine(OptionEngineMixin, Engine):
    def update_execution_options(self, **opt: Any) -> None:
        Engine.update_execution_options(self, **opt)


Engine._option_cls = OptionEngine

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\crypto\crypto.py ===
"""
This file contains some classical ciphers and routines
implementing a linear-feedback shift register (LFSR)
and the Diffie-Hellman key exchange.

.. warning::

   This module is intended for educational purposes only. Do not use the
   functions in this module for real cryptographic applications. If you wish
   to encrypt real data, we recommend using something like the `cryptography
   <https://cryptography.io/en/latest/>`_ module.

"""

from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import string
import warnings

from itertools import cycle

from sympy.external.gmpy import GROUND_TYPES
from sympy.core import Symbol
from sympy.core.numbers import Rational
from sympy.core.random import _randrange, _randint
from sympy.external.gmpy import gcd, invert
from sympy.functions.combinatorial.numbers import (totient as _euler,
                                                   reduced_totient as _carmichael)
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset
from sympy.utilities.decorator import doctest_depends_on


if GROUND_TYPES == 'flint':
    __doctest_skip__ = ['lfsr_sequence']


class NonInvertibleCipherWarning(RuntimeWarning):
    """A warning raised if the cipher is not invertible."""
    def __init__(self, msg):
        self.fullMessage = msg

    def __str__(self):
        return '\n\t' + self.fullMessage

    def warn(self, stacklevel=3):
        warnings.warn(self, stacklevel=stacklevel)


def AZ(s=None):
    """Return the letters of ``s`` in uppercase. In case more than
    one string is passed, each of them will be processed and a list
    of upper case strings will be returned.

    Examples
    ========

    >>> from sympy.crypto.crypto import AZ
    >>> AZ('Hello, world!')
    'HELLOWORLD'
    >>> AZ('Hello, world!'.split())
    ['HELLO', 'WORLD']

    See Also
    ========

    check_and_join

    """
    if not s:
        return uppercase
    t = isinstance(s, str)
    if t:
        s = [s]
    rv = [check_and_join(i.upper().split(), uppercase, filter=True)
        for i in s]
    if t:
        return rv[0]
    return rv

bifid5 = AZ().replace('J', '')
bifid6 = AZ() + string.digits
bifid10 = printable


def padded_key(key, symbols):
    """Return a string of the distinct characters of ``symbols`` with
    those of ``key`` appearing first. A ValueError is raised if
    a) there are duplicate characters in ``symbols`` or
    b) there are characters in ``key`` that are  not in ``symbols``.

    Examples
    ========

    >>> from sympy.crypto.crypto import padded_key
    >>> padded_key('PUPPY', 'OPQRSTUVWXY')
    'PUYOQRSTVWX'
    >>> padded_key('RSA', 'ARTIST')
    Traceback (most recent call last):
    ...
    ValueError: duplicate characters in symbols: T

    """
    syms = list(uniq(symbols))
    if len(syms) != len(symbols):
        extra = ''.join(sorted({
            i for i in symbols if symbols.count(i) > 1}))
        raise ValueError('duplicate characters in symbols: %s' % extra)
    extra = set(key) - set(syms)
    if extra:
        raise ValueError(
            'characters in key but not symbols: %s' % ''.join(
            sorted(extra)))
    key0 = ''.join(list(uniq(key)))
    # remove from syms characters in key0
    return key0 + translate(''.join(syms), None, key0)


def check_and_join(phrase, symbols=None, filter=None):
    """
    Joins characters of ``phrase`` and if ``symbols`` is given, raises
    an error if any character in ``phrase`` is not in ``symbols``.

    Parameters
    ==========

    phrase
        String or list of strings to be returned as a string.

    symbols
        Iterable of characters allowed in ``phrase``.

        If ``symbols`` is ``None``, no checking is performed.

    Examples
    ========

    >>> from sympy.crypto.crypto import check_and_join
    >>> check_and_join('a phrase')
    'a phrase'
    >>> check_and_join('a phrase'.upper().split())
    'APHRASE'
    >>> check_and_join('a phrase!'.upper().split(), 'ARE', filter=True)
    'ARAE'
    >>> check_and_join('a phrase!'.upper().split(), 'ARE')
    Traceback (most recent call last):
    ...
    ValueError: characters in phrase but not symbols: "!HPS"

    """
    rv = ''.join(''.join(phrase))
    if symbols is not None:
        symbols = check_and_join(symbols)
        missing = ''.join(sorted(set(rv) - set(symbols)))
        if missing:
            if not filter:
                raise ValueError(
                    'characters in phrase but not symbols: "%s"' % missing)
            rv = translate(rv, None, missing)
    return rv


def _prep(msg, key, alp, default=None):
    if not alp:
        if not default:
            alp = AZ()
            msg = AZ(msg)
            key = AZ(key)
        else:
            alp = default
    else:
        alp = ''.join(alp)
    key = check_and_join(key, alp, filter=True)
    msg = check_and_join(msg, alp, filter=True)
    return msg, key, alp


def cycle_list(k, n):
    """
    Returns the elements of the list ``range(n)`` shifted to the
    left by ``k`` (so the list starts with ``k`` (mod ``n``)).

    Examples
    ========

    >>> from sympy.crypto.crypto import cycle_list
    >>> cycle_list(3, 10)
    [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]

    """
    k = k % n
    return list(range(k, n)) + list(range(k))


######## shift cipher examples ############


def encipher_shift(msg, key, symbols=None):
    """
    Performs shift cipher encryption on plaintext msg, and returns the
    ciphertext.

    Parameters
    ==========

    key : int
        The secret key.

    msg : str
        Plaintext of upper-case letters.

    Returns
    =======

    str
        Ciphertext of upper-case letters.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_shift, decipher_shift
    >>> msg = "GONAVYBEATARMY"
    >>> ct = encipher_shift(msg, 1); ct
    'HPOBWZCFBUBSNZ'

    To decipher the shifted text, change the sign of the key:

    >>> encipher_shift(ct, -1)
    'GONAVYBEATARMY'

    There is also a convenience function that does this with the
    original key:

    >>> decipher_shift(ct, 1)
    'GONAVYBEATARMY'

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L1`` of
               corresponding integers.
            2. Compute from the list ``L1`` a new list ``L2``, given by
               adding ``(k mod 26)`` to each element in ``L1``.
            3. Compute from the list ``L2`` a string ``ct`` of
               corresponding letters.

    The shift cipher is also called the Caesar cipher, after
    Julius Caesar, who, according to Suetonius, used it with a
    shift of three to protect messages of military significance.
    Caesar's nephew Augustus reportedly used a similar cipher, but
    with a right shift of 1.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Caesar_cipher
    .. [2] https://mathworld.wolfram.com/CaesarsMethod.html

    See Also
    ========

    decipher_shift

    """
    msg, _, A = _prep(msg, '', symbols)
    shift = len(A) - key % len(A)
    key = A[shift:] + A[:shift]
    return translate(msg, key, A)


def decipher_shift(msg, key, symbols=None):
    """
    Return the text by shifting the characters of ``msg`` to the
    left by the amount given by ``key``.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_shift, decipher_shift
    >>> msg = "GONAVYBEATARMY"
    >>> ct = encipher_shift(msg, 1); ct
    'HPOBWZCFBUBSNZ'

    To decipher the shifted text, change the sign of the key:

    >>> encipher_shift(ct, -1)
    'GONAVYBEATARMY'

    Or use this function with the original key:

    >>> decipher_shift(ct, 1)
    'GONAVYBEATARMY'

    """
    return encipher_shift(msg, -key, symbols)

def encipher_rot13(msg, symbols=None):
    """
    Performs the ROT13 encryption on a given plaintext ``msg``.

    Explanation
    ===========

    ROT13 is a substitution cipher which substitutes each letter
    in the plaintext message for the letter furthest away from it
    in the English alphabet.

    Equivalently, it is just a Caeser (shift) cipher with a shift
    key of 13 (midway point of the alphabet).

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/ROT13

    See Also
    ========

    decipher_rot13
    encipher_shift

    """
    return encipher_shift(msg, 13, symbols)

def decipher_rot13(msg, symbols=None):
    """
    Performs the ROT13 decryption on a given plaintext ``msg``.

    Explanation
    ============

    ``decipher_rot13`` is equivalent to ``encipher_rot13`` as both
    ``decipher_shift`` with a key of 13 and ``encipher_shift`` key with a
    key of 13 will return the same results. Nonetheless,
    ``decipher_rot13`` has nonetheless been explicitly defined here for
    consistency.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_rot13, decipher_rot13
    >>> msg = 'GONAVYBEATARMY'
    >>> ciphertext = encipher_rot13(msg);ciphertext
    'TBANILORNGNEZL'
    >>> decipher_rot13(ciphertext)
    'GONAVYBEATARMY'
    >>> encipher_rot13(msg) == decipher_rot13(msg)
    True
    >>> msg == decipher_rot13(ciphertext)
    True

    """
    return decipher_shift(msg, 13, symbols)

######## affine cipher examples ############


def encipher_affine(msg, key, symbols=None, _inverse=False):
    r"""
    Performs the affine cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Explanation
    ===========

    Encryption is based on the map `x \rightarrow ax+b` (mod `N`)
    where ``N`` is the number of characters in the alphabet.
    Decryption is based on the map `x \rightarrow cx+d` (mod `N`),
    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).
    In particular, for the map to be invertible, we need
    `\mathrm{gcd}(a, N) = 1` and an error will be raised if this is
    not true.

    Parameters
    ==========

    msg : str
        Characters that appear in ``symbols``.

    a, b : int, int
        A pair integers, with ``gcd(a, N) = 1`` (the secret key).

    symbols
        String of characters (default = uppercase letters).

        When no symbols are given, ``msg`` is converted to upper case
        letters and all other characters are ignored.

    Returns
    =======

    ct
        String of characters (the ciphertext message)

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L1`` of
               corresponding integers.
            2. Compute from the list ``L1`` a new list ``L2``, given by
               replacing ``x`` by ``a*x + b (mod N)``, for each element
               ``x`` in ``L1``.
            3. Compute from the list ``L2`` a string ``ct`` of
               corresponding letters.

    This is a straightforward generalization of the shift cipher with
    the added complexity of requiring 2 characters to be deciphered in
    order to recover the key.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Affine_cipher

    See Also
    ========

    decipher_affine

    """
    msg, _, A = _prep(msg, '', symbols)
    N = len(A)
    a, b = key
    assert gcd(a, N) == 1
    if _inverse:
        c = invert(a, N)
        d = -b*c
        a, b = c, d
    B = ''.join([A[(a*i + b) % N] for i in range(N)])
    return translate(msg, A, B)


def decipher_affine(msg, key, symbols=None):
    r"""
    Return the deciphered text that was made from the mapping,
    `x \rightarrow ax+b` (mod `N`), where ``N`` is the
    number of characters in the alphabet. Deciphering is done by
    reciphering with a new key: `x \rightarrow cx+d` (mod `N`),
    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_affine, decipher_affine
    >>> msg = "GO NAVY BEAT ARMY"
    >>> key = (3, 1)
    >>> encipher_affine(msg, key)
    'TROBMVENBGBALV'
    >>> decipher_affine(_, key)
    'GONAVYBEATARMY'

    See Also
    ========

    encipher_affine

    """
    return encipher_affine(msg, key, symbols, _inverse=True)


def encipher_atbash(msg, symbols=None):
    r"""
    Enciphers a given ``msg`` into its Atbash ciphertext and returns it.

    Explanation
    ===========

    Atbash is a substitution cipher originally used to encrypt the Hebrew
    alphabet. Atbash works on the principle of mapping each alphabet to its
    reverse / counterpart (i.e. a would map to z, b to y etc.)

    Atbash is functionally equivalent to the affine cipher with ``a = 25``
    and ``b = 25``

    See Also
    ========

    decipher_atbash

    """
    return encipher_affine(msg, (25, 25), symbols)


def decipher_atbash(msg, symbols=None):
    r"""
    Deciphers a given ``msg`` using Atbash cipher and returns it.

    Explanation
    ===========

    ``decipher_atbash`` is functionally equivalent to ``encipher_atbash``.
    However, it has still been added as a separate function to maintain
    consistency.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_atbash, decipher_atbash
    >>> msg = 'GONAVYBEATARMY'
    >>> encipher_atbash(msg)
    'TLMZEBYVZGZINB'
    >>> decipher_atbash(msg)
    'TLMZEBYVZGZINB'
    >>> encipher_atbash(msg) == decipher_atbash(msg)
    True
    >>> msg == encipher_atbash(encipher_atbash(msg))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Atbash

    See Also
    ========

    encipher_atbash

    """
    return decipher_affine(msg, (25, 25), symbols)

#################### substitution cipher ###########################


def encipher_substitution(msg, old, new=None):
    r"""
    Returns the ciphertext obtained by replacing each character that
    appears in ``old`` with the corresponding character in ``new``.
    If ``old`` is a mapping, then new is ignored and the replacements
    defined by ``old`` are used.

    Explanation
    ===========

    This is a more general than the affine cipher in that the key can
    only be recovered by determining the mapping for each symbol.
    Though in practice, once a few symbols are recognized the mappings
    for other characters can be quickly guessed.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_substitution, AZ
    >>> old = 'OEYAG'
    >>> new = '034^6'
    >>> msg = AZ("go navy! beat army!")
    >>> ct = encipher_substitution(msg, old, new); ct
    '60N^V4B3^T^RM4'

    To decrypt a substitution, reverse the last two arguments:

    >>> encipher_substitution(ct, new, old)
    'GONAVYBEATARMY'

    In the special case where ``old`` and ``new`` are a permutation of
    order 2 (representing a transposition of characters) their order
    is immaterial:

    >>> old = 'NAVY'
    >>> new = 'ANYV'
    >>> encipher = lambda x: encipher_substitution(x, old, new)
    >>> encipher('NAVY')
    'ANYV'
    >>> encipher(_)
    'NAVY'

    The substitution cipher, in general, is a method
    whereby "units" (not necessarily single characters) of plaintext
    are replaced with ciphertext according to a regular system.

    >>> ords = dict(zip('abc', ['\\%i' % ord(i) for i in 'abc']))
    >>> print(encipher_substitution('abc', ords))
    \97\98\99

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Substitution_cipher

    """
    return translate(msg, old, new)


######################################################################
#################### Vigenere cipher examples ########################
######################################################################

def encipher_vigenere(msg, key, symbols=None):
    """
    Performs the Vigenere cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_vigenere, AZ
    >>> key = "encrypt"
    >>> msg = "meet me on monday"
    >>> encipher_vigenere(msg, key)
    'QRGKKTHRZQEBPR'

    Section 1 of the Kryptos sculpture at the CIA headquarters
    uses this cipher and also changes the order of the
    alphabet [2]_. Here is the first line of that section of
    the sculpture:

    >>> from sympy.crypto.crypto import decipher_vigenere, padded_key
    >>> alp = padded_key('KRYPTOS', AZ())
    >>> key = 'PALIMPSEST'
    >>> msg = 'EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJ'
    >>> decipher_vigenere(msg, key, alp)
    'BETWEENSUBTLESHADINGANDTHEABSENC'

    Explanation
    ===========

    The Vigenere cipher is named after Blaise de Vigenere, a sixteenth
    century diplomat and cryptographer, by a historical accident.
    Vigenere actually invented a different and more complicated cipher.
    The so-called *Vigenere cipher* was actually invented
    by Giovan Batista Belaso in 1553.

    This cipher was used in the 1800's, for example, during the American
    Civil War. The Confederacy used a brass cipher disk to implement the
    Vigenere cipher (now on display in the NSA Museum in Fort
    Meade) [1]_.

    The Vigenere cipher is a generalization of the shift cipher.
    Whereas the shift cipher shifts each letter by the same amount
    (that amount being the key of the shift cipher) the Vigenere
    cipher shifts a letter by an amount determined by the key (which is
    a word or phrase known only to the sender and receiver).

    For example, if the key was a single letter, such as "C", then the
    so-called Vigenere cipher is actually a shift cipher with a
    shift of `2` (since "C" is the 2nd letter of the alphabet, if
    you start counting at `0`). If the key was a word with two
    letters, such as "CA", then the so-called Vigenere cipher will
    shift letters in even positions by `2` and letters in odd positions
    are left alone (shifted by `0`, since "A" is the 0th letter, if
    you start counting at `0`).


    ALGORITHM:

        INPUT:

            ``msg``: string of characters that appear in ``symbols``
            (the plaintext)

            ``key``: a string of characters that appear in ``symbols``
            (the secret key)

            ``symbols``: a string of letters defining the alphabet


        OUTPUT:

            ``ct``: string of characters (the ciphertext message)

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``key`` a list ``L1`` of
               corresponding integers. Let ``n1 = len(L1)``.
            2. Compute from the string ``msg`` a list ``L2`` of
               corresponding integers. Let ``n2 = len(L2)``.
            3. Break ``L2`` up sequentially into sublists of size
               ``n1``; the last sublist may be smaller than ``n1``
            4. For each of these sublists ``L`` of ``L2``, compute a
               new list ``C`` given by ``C[i] = L[i] + L1[i] (mod N)``
               to the ``i``-th element in the sublist, for each ``i``.
            5. Assemble these lists ``C`` by concatenation into a new
               list of length ``n2``.
            6. Compute from the new list a string ``ct`` of
               corresponding letters.

    Once it is known that the key is, say, `n` characters long,
    frequency analysis can be applied to every `n`-th letter of
    the ciphertext to determine the plaintext. This method is
    called *Kasiski examination* (although it was first discovered
    by Babbage). If they key is as long as the message and is
    comprised of randomly selected characters -- a one-time pad -- the
    message is theoretically unbreakable.

    The cipher Vigenere actually discovered is an "auto-key" cipher
    described as follows.

    ALGORITHM:

        INPUT:

          ``key``: a string of letters (the secret key)

          ``msg``: string of letters (the plaintext message)

        OUTPUT:

          ``ct``: string of upper-case letters (the ciphertext message)

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L2`` of
               corresponding integers. Let ``n2 = len(L2)``.
            2. Let ``n1`` be the length of the key. Append to the
               string ``key`` the first ``n2 - n1`` characters of
               the plaintext message. Compute from this string (also of
               length ``n2``) a list ``L1`` of integers corresponding
               to the letter numbers in the first step.
            3. Compute a new list ``C`` given by
               ``C[i] = L1[i] + L2[i] (mod N)``.
            4. Compute from the new list a string ``ct`` of letters
               corresponding to the new integers.

    To decipher the auto-key ciphertext, the key is used to decipher
    the first ``n1`` characters and then those characters become the
    key to  decipher the next ``n1`` characters, etc...:

    >>> m = AZ('go navy, beat army! yes you can'); m
    'GONAVYBEATARMYYESYOUCAN'
    >>> key = AZ('gold bug'); n1 = len(key); n2 = len(m)
    >>> auto_key = key + m[:n2 - n1]; auto_key
    'GOLDBUGGONAVYBEATARMYYE'
    >>> ct = encipher_vigenere(m, auto_key); ct
    'MCYDWSHKOGAMKZCELYFGAYR'
    >>> n1 = len(key)
    >>> pt = []
    >>> while ct:
    ...     part, ct = ct[:n1], ct[n1:]
    ...     pt.append(decipher_vigenere(part, key))
    ...     key = pt[-1]
    ...
    >>> ''.join(pt) == m
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Vigenere_cipher
    .. [2] https://web.archive.org/web/20071116100808/https://filebox.vt.edu/users/batman/kryptos.html
       (short URL: https://goo.gl/ijr22d)

    """
    msg, key, A = _prep(msg, key, symbols)
    map = {c: i for i, c in enumerate(A)}
    key = [map[c] for c in key]
    N = len(map)
    k = len(key)
    rv = []
    for i, m in enumerate(msg):
        rv.append(A[(map[m] + key[i % k]) % N])
    rv = ''.join(rv)
    return rv


def decipher_vigenere(msg, key, symbols=None):
    """
    Decode using the Vigenere cipher.

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_vigenere
    >>> key = "encrypt"
    >>> ct = "QRGK kt HRZQE BPR"
    >>> decipher_vigenere(ct, key)
    'MEETMEONMONDAY'

    """
    msg, key, A = _prep(msg, key, symbols)
    map = {c: i for i, c in enumerate(A)}
    N = len(A)   # normally, 26
    K = [map[c] for c in key]
    n = len(K)
    C = [map[c] for c in msg]
    rv = ''.join([A[(-K[i % n] + c) % N] for i, c in enumerate(C)])
    return rv


#################### Hill cipher  ########################


def encipher_hill(msg, key, symbols=None, pad="Q"):
    r"""
    Return the Hill cipher encryption of ``msg``.

    Explanation
    ===========

    The Hill cipher [1]_, invented by Lester S. Hill in the 1920's [2]_,
    was the first polygraphic cipher in which it was practical
    (though barely) to operate on more than three symbols at once.
    The following discussion assumes an elementary knowledge of
    matrices.

    First, each letter is first encoded as a number starting with 0.
    Suppose your message `msg` consists of `n` capital letters, with no
    spaces. This may be regarded an `n`-tuple M of elements of
    `Z_{26}` (if the letters are those of the English alphabet). A key
    in the Hill cipher is a `k x k` matrix `K`, all of whose entries
    are in `Z_{26}`, such that the matrix `K` is invertible (i.e., the
    linear transformation `K: Z_{N}^k \rightarrow Z_{N}^k`
    is one-to-one).


    Parameters
    ==========

    msg
        Plaintext message of `n` upper-case letters.

    key
        A `k \times k` invertible matrix `K`, all of whose entries are
        in `Z_{26}` (or whatever number of symbols are being used).

    pad
        Character (default "Q") to use to make length of text be a
        multiple of ``k``.

    Returns
    =======

    ct
        Ciphertext of upper-case letters.

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L`` of
               corresponding integers. Let ``n = len(L)``.
            2. Break the list ``L`` up into ``t = ceiling(n/k)``
               sublists ``L_1``, ..., ``L_t`` of size ``k`` (with
               the last list "padded" to ensure its size is
               ``k``).
            3. Compute new list ``C_1``, ..., ``C_t`` given by
               ``C[i] = K*L_i`` (arithmetic is done mod N), for each
               ``i``.
            4. Concatenate these into a list ``C = C_1 + ... + C_t``.
            5. Compute from ``C`` a string ``ct`` of corresponding
               letters. This has length ``k*t``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hill_cipher
    .. [2] Lester S. Hill, Cryptography in an Algebraic Alphabet,
       The American Mathematical Monthly Vol.36, June-July 1929,
       pp.306-312.

    See Also
    ========

    decipher_hill

    """
    assert key.is_square
    assert len(pad) == 1
    msg, pad, A = _prep(msg, pad, symbols)
    map = {c: i for i, c in enumerate(A)}
    P = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(P)
    m, r = divmod(n, k)
    if r:
        P = P + [map[pad]]*(k - r)
        m += 1
    rv = ''.join([A[c % N] for j in range(m) for c in
        list(key*Matrix(k, 1, [P[i]
        for i in range(k*j, k*(j + 1))]))])
    return rv


def decipher_hill(msg, key, symbols=None):
    """
    Deciphering is the same as enciphering but using the inverse of the
    key matrix.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_hill, decipher_hill
    >>> from sympy import Matrix

    >>> key = Matrix([[1, 2], [3, 5]])
    >>> encipher_hill("meet me on monday", key)
    'UEQDUEODOCTCWQ'
    >>> decipher_hill(_, key)
    'MEETMEONMONDAY'

    When the length of the plaintext (stripped of invalid characters)
    is not a multiple of the key dimension, extra characters will
    appear at the end of the enciphered and deciphered text. In order to
    decipher the text, those characters must be included in the text to
    be deciphered. In the following, the key has a dimension of 4 but
    the text is 2 short of being a multiple of 4 so two characters will
    be added.

    >>> key = Matrix([[1, 1, 1, 2], [0, 1, 1, 0],
    ...               [2, 2, 3, 4], [1, 1, 0, 1]])
    >>> msg = "ST"
    >>> encipher_hill(msg, key)
    'HJEB'
    >>> decipher_hill(_, key)
    'STQQ'
    >>> encipher_hill(msg, key, pad="Z")
    'ISPK'
    >>> decipher_hill(_, key)
    'STZZ'

    If the last two characters of the ciphertext were ignored in
    either case, the wrong plaintext would be recovered:

    >>> decipher_hill("HD", key)
    'ORMV'
    >>> decipher_hill("IS", key)
    'UIKY'

    See Also
    ========

    encipher_hill

    """
    assert key.is_square
    msg, _, A = _prep(msg, '', symbols)
    map = {c: i for i, c in enumerate(A)}
    C = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(C)
    m, r = divmod(n, k)
    if r:
        C = C + [0]*(k - r)
        m += 1
    key_inv = key.inv_mod(N)
    rv = ''.join([A[p % N] for j in range(m) for p in
        list(key_inv*Matrix(
        k, 1, [C[i] for i in range(k*j, k*(j + 1))]))])
    return rv


#################### Bifid cipher  ########################


def encipher_bifid(msg, key, symbols=None):
    r"""
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    This is the version of the Bifid cipher that uses an `n \times n`
    Polybius square.

    Parameters
    ==========

    msg
        Plaintext string.

    key
        Short string for key.

        Duplicate characters are ignored and then it is padded with the
        characters in ``symbols`` that were not in the short key.

    symbols
        `n \times n` characters defining the alphabet.

        (default is string.printable)

    Returns
    =======

    ciphertext
        Ciphertext using Bifid5 cipher without spaces.

    See Also
    ========

    decipher_bifid, encipher_bifid5, encipher_bifid6

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bifid_cipher

    """
    msg, key, A = _prep(msg, key, symbols, bifid10)
    long_key = ''.join(uniq(key)) or A

    n = len(A)**.5
    if n != int(n):
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))
    N = int(n)
    if len(long_key) < N**2:
        long_key = list(long_key) + [x for x in A if x not in long_key]

    # the fractionalization
    row_col = {ch: divmod(i, N) for i, ch in enumerate(long_key)}
    r, c = zip(*[row_col[x] for x in msg])
    rc = r + c
    ch = {i: ch for ch, i in row_col.items()}
    rv = ''.join(ch[i] for i in zip(rc[::2], rc[1::2]))
    return rv


def decipher_bifid(msg, key, symbols=None):
    r"""
    Performs the Bifid cipher decryption on ciphertext ``msg``, and
    returns the plaintext.

    This is the version of the Bifid cipher that uses the `n \times n`
    Polybius square.

    Parameters
    ==========

    msg
        Ciphertext string.

    key
        Short string for key.

        Duplicate characters are ignored and then it is padded with the
        characters in symbols that were not in the short key.

    symbols
        `n \times n` characters defining the alphabet.

        (default=string.printable, a `10 \times 10` matrix)

    Returns
    =======

    deciphered
        Deciphered text.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_bifid, decipher_bifid, AZ)

    Do an encryption using the bifid5 alphabet:

    >>> alp = AZ().replace('J', '')
    >>> ct = AZ("meet me on monday!")
    >>> key = AZ("gold bug")
    >>> encipher_bifid(ct, key, alp)
    'IEILHHFSTSFQYE'

    When entering the text or ciphertext, spaces are ignored so it
    can be formatted as desired. Re-entering the ciphertext from the
    preceding, putting 4 characters per line and padding with an extra
    J, does not cause problems for the deciphering:

    >>> decipher_bifid('''
    ... IEILH
    ... HFSTS
    ... FQYEJ''', key, alp)
    'MEETMEONMONDAY'

    When no alphabet is given, all 100 printable characters will be
    used:

    >>> key = ''
    >>> encipher_bifid('hello world!', key)
    'bmtwmg-bIo*w'
    >>> decipher_bifid(_, key)
    'hello world!'

    If the key is changed, a different encryption is obtained:

    >>> key = 'gold bug'
    >>> encipher_bifid('hello world!', 'gold_bug')
    'hg2sfuei7t}w'

    And if the key used to decrypt the message is not exact, the
    original text will not be perfectly obtained:

    >>> decipher_bifid(_, 'gold pug')
    'heldo~wor6d!'

    """
    msg, _, A = _prep(msg, '', symbols, bifid10)
    long_key = ''.join(uniq(key)) or A

    n = len(A)**.5
    if n != int(n):
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))
    N = int(n)
    if len(long_key) < N**2:
        long_key = list(long_key) + [x for x in A if x not in long_key]

    # the reverse fractionalization
    row_col = {
        ch: divmod(i, N) for i, ch in enumerate(long_key)}
    rc = [i for c in msg for i in row_col[c]]
    n = len(msg)
    rc = zip(*(rc[:n], rc[n:]))
    ch = {i: ch for ch, i in row_col.items()}
    rv = ''.join(ch[i] for i in rc)
    return rv


def bifid_square(key):
    """Return characters of ``key`` arranged in a square.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...    bifid_square, AZ, padded_key, bifid5)
    >>> bifid_square(AZ().replace('J', ''))
    Matrix([
    [A, B, C, D, E],
    [F, G, H, I, K],
    [L, M, N, O, P],
    [Q, R, S, T, U],
    [V, W, X, Y, Z]])

    >>> bifid_square(padded_key(AZ('gold bug!'), bifid5))
    Matrix([
    [G, O, L, D, B],
    [U, A, C, E, F],
    [H, I, K, M, N],
    [P, Q, R, S, T],
    [V, W, X, Y, Z]])

    See Also
    ========

    padded_key

    """
    A = ''.join(uniq(''.join(key)))
    n = len(A)**.5
    if n != int(n):
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))
    n = int(n)
    f = lambda i, j: Symbol(A[n*i + j])
    rv = Matrix(n, n, f)
    return rv


def encipher_bifid5(msg, key):
    r"""
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Explanation
    ===========

    This is the version of the Bifid cipher that uses the `5 \times 5`
    Polybius square. The letter "J" is ignored so it must be replaced
    with something else (traditionally an "I") before encryption.

    ALGORITHM: (5x5 case)

        STEPS:
            0. Create the `5 \times 5` Polybius square ``S`` associated
               to ``key`` as follows:

                a) moving from left-to-right, top-to-bottom,
                   place the letters of the key into a `5 \times 5`
                   matrix,
                b) if the key has less than 25 letters, add the
                   letters of the alphabet not in the key until the
                   `5 \times 5` square is filled.

            1. Create a list ``P`` of pairs of numbers which are the
               coordinates in the Polybius square of the letters in
               ``msg``.
            2. Let ``L1`` be the list of all first coordinates of ``P``
               (length of ``L1 = n``), let ``L2`` be the list of all
               second coordinates of ``P`` (so the length of ``L2``
               is also ``n``).
            3. Let ``L`` be the concatenation of ``L1`` and ``L2``
               (length ``L = 2*n``), except that consecutive numbers
               are paired ``(L[2*i], L[2*i + 1])``. You can regard
               ``L`` as a list of pairs of length ``n``.
            4. Let ``C`` be the list of all letters which are of the
               form ``S[i, j]``, for all ``(i, j)`` in ``L``. As a
               string, this is the ciphertext of ``msg``.

    Parameters
    ==========

    msg : str
        Plaintext string.

        Converted to upper case and filtered of anything but all letters
        except J.

    key
        Short string for key; non-alphabetic letters, J and duplicated
        characters are ignored and then, if the length is less than 25
        characters, it is padded with other letters of the alphabet
        (in alphabetical order).

    Returns
    =======

    ct
        Ciphertext (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_bifid5, decipher_bifid5)

    "J" will be omitted unless it is replaced with something else:

    >>> round_trip = lambda m, k: \
    ...     decipher_bifid5(encipher_bifid5(m, k), k)
    >>> key = 'a'
    >>> msg = "JOSIE"
    >>> round_trip(msg, key)
    'OSIE'
    >>> round_trip(msg.replace("J", "I"), key)
    'IOSIE'
    >>> j = "QIQ"
    >>> round_trip(msg.replace("J", j), key).replace(j, "J")
    'JOSIE'


    Notes
    =====

    The Bifid cipher was invented around 1901 by Felix Delastelle.
    It is a *fractional substitution* cipher, where letters are
    replaced by pairs of symbols from a smaller alphabet. The
    cipher uses a `5 \times 5` square filled with some ordering of the
    alphabet, except that "J" is replaced with "I" (this is a so-called
    Polybius square; there is a `6 \times 6` analog if you add back in
    "J" and also append onto the usual 26 letter alphabet, the digits
    0, 1, ..., 9).
    According to Helen Gaines' book *Cryptanalysis*, this type of cipher
    was used in the field by the German Army during World War I.

    See Also
    ========

    decipher_bifid5, encipher_bifid

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid5)
    key = padded_key(key, bifid5)
    return encipher_bifid(msg, '', key)


def decipher_bifid5(msg, key):
    r"""
    Return the Bifid cipher decryption of ``msg``.

    Explanation
    ===========

    This is the version of the Bifid cipher that uses the `5 \times 5`
    Polybius square; the letter "J" is ignored unless a ``key`` of
    length 25 is used.

    Parameters
    ==========

    msg
        Ciphertext string.

    key
        Short string for key; duplicated characters are ignored and if
        the length is less then 25 characters, it will be padded with
        other letters from the alphabet omitting "J".
        Non-alphabetic characters are ignored.

    Returns
    =======

    plaintext
        Plaintext from Bifid5 cipher (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_bifid5, decipher_bifid5
    >>> key = "gold bug"
    >>> encipher_bifid5('meet me on friday', key)
    'IEILEHFSTSFXEE'
    >>> encipher_bifid5('meet me on monday', key)
    'IEILHHFSTSFQYE'
    >>> decipher_bifid5(_, key)
    'MEETMEONMONDAY'

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid5)
    key = padded_key(key, bifid5)
    return decipher_bifid(msg, '', key)


def bifid5_square(key=None):
    r"""
    5x5 Polybius square.

    Produce the Polybius square for the `5 \times 5` Bifid cipher.

    Examples
    ========

    >>> from sympy.crypto.crypto import bifid5_square
    >>> bifid5_square("gold bug")
    Matrix([
    [G, O, L, D, B],
    [U, A, C, E, F],
    [H, I, K, M, N],
    [P, Q, R, S, T],
    [V, W, X, Y, Z]])

    """
    if not key:
        key = bifid5
    else:
        _, key, _ = _prep('', key.upper(), None, bifid5)
        key = padded_key(key, bifid5)
    return bifid_square(key)


def encipher_bifid6(msg, key):
    r"""
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    This is the version of the Bifid cipher that uses the `6 \times 6`
    Polybius square.

    Parameters
    ==========

    msg
        Plaintext string (digits okay).

    key
        Short string for key (digits okay).

        If ``key`` is less than 36 characters long, the square will be
        filled with letters A through Z and digits 0 through 9.

    Returns
    =======

    ciphertext
        Ciphertext from Bifid cipher (all caps, no spaces).

    See Also
    ========

    decipher_bifid6, encipher_bifid

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid6)
    key = padded_key(key, bifid6)
    return encipher_bifid(msg, '', key)


def decipher_bifid6(msg, key):
    r"""
    Performs the Bifid cipher decryption on ciphertext ``msg``, and
    returns the plaintext.

    This is the version of the Bifid cipher that uses the `6 \times 6`
    Polybius square.

    Parameters
    ==========

    msg
        Ciphertext string (digits okay); converted to upper case

    key
        Short string for key (digits okay).

        If ``key`` is less than 36 characters long, the square will be
        filled with letters A through Z and digits 0 through 9.
        All letters are converted to uppercase.

    Returns
    =======

    plaintext
        Plaintext from Bifid cipher (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_bifid6, decipher_bifid6
    >>> key = "gold bug"
    >>> encipher_bifid6('meet me on monday at 8am', key)
    'KFKLJJHF5MMMKTFRGPL'
    >>> decipher_bifid6(_, key)
    'MEETMEONMONDAYAT8AM'

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid6)
    key = padded_key(key, bifid6)
    return decipher_bifid(msg, '', key)


def bifid6_square(key=None):
    r"""
    6x6 Polybius square.

    Produces the Polybius square for the `6 \times 6` Bifid cipher.
    Assumes alphabet of symbols is "A", ..., "Z", "0", ..., "9".

    Examples
    ========

    >>> from sympy.crypto.crypto import bifid6_square
    >>> key = "gold bug"
    >>> bifid6_square(key)
    Matrix([
    [G, O, L, D, B, U],
    [A, C, E, F, H, I],
    [J, K, M, N, P, Q],
    [R, S, T, V, W, X],
    [Y, Z, 0, 1, 2, 3],
    [4, 5, 6, 7, 8, 9]])

    """
    if not key:
        key = bifid6
    else:
        _, key, _ = _prep('', key.upper(), None, bifid6)
        key = padded_key(key, bifid6)
    return bifid_square(key)


#################### RSA  #############################

def _decipher_rsa_crt(i, d, factors):
    """Decipher RSA using chinese remainder theorem from the information
    of the relatively-prime factors of the modulus.

    Parameters
    ==========

    i : integer
        Ciphertext

    d : integer
        The exponent component.

    factors : list of relatively-prime integers
        The integers given must be coprime and the product must equal
        the modulus component of the original RSA key.

    Examples
    ========

    How to decrypt RSA with CRT:

    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key
    >>> primes = [61, 53]
    >>> e = 17
    >>> args = primes + [e]
    >>> puk = rsa_public_key(*args)
    >>> prk = rsa_private_key(*args)

    >>> from sympy.crypto.crypto import encipher_rsa, _decipher_rsa_crt
    >>> msg = 65
    >>> crt_primes = primes
    >>> encrypted = encipher_rsa(msg, puk)
    >>> decrypted = _decipher_rsa_crt(encrypted, prk[1], primes)
    >>> decrypted
    65
    """
    moduluses = [pow(i, d, p) for p in factors]

    result = crt(factors, moduluses)
    if not result:
        raise ValueError("CRT failed")
    return result[0]


def _rsa_key(*args, public=True, private=True, totient='Euler', index=None, multipower=None):
    r"""A private subroutine to generate RSA key

    Parameters
    ==========

    public, private : bool, optional
        Flag to generate either a public key, a private key.

    totient : 'Euler' or 'Carmichael'
        Different notation used for totient.

    multipower : bool, optional
        Flag to bypass warning for multipower RSA.
    """

    if len(args) < 2:
        return False

    if totient not in ('Euler', 'Carmichael'):
        raise ValueError(
            "The argument totient={} should either be " \
            "'Euler', 'Carmichalel'." \
            .format(totient))

    if totient == 'Euler':
        _totient = _euler
    else:
        _totient = _carmichael

    if index is not None:
        index = as_int(index)
        if totient != 'Carmichael':
            raise ValueError(
                "Setting the 'index' keyword argument requires totient"
                "notation to be specified as 'Carmichael'.")

    primes, e = args[:-1], args[-1]

    if not all(isprime(p) for p in primes):
        new_primes = []
        for i in primes:
            new_primes.extend(factorint(i, multiple=True))
        primes = new_primes

    n = reduce(lambda i, j: i*j, primes)

    tally = multiset(primes)
    if all(v == 1 for v in tally.values()):
        phi = int(_totient(tally))

    else:
        if not multipower:
            NonInvertibleCipherWarning(
                'Non-distinctive primes found in the factors {}. '
                'The cipher may not be decryptable for some numbers '
                'in the complete residue system Z[{}], but the cipher '
                'can still be valid if you restrict the domain to be '
                'the reduced residue system Z*[{}]. You can pass '
                'the flag multipower=True if you want to suppress this '
                'warning.'
                .format(primes, n, n)
                # stacklevel=4 because most users will call a function that
                # calls this function
                ).warn(stacklevel=4)
        phi = int(_totient(tally))

    if gcd(e, phi) == 1:
        if public and not private:
            if isinstance(index, int):
                e = e % phi
                e += index * phi
            return n, e

        if private and not public:
            d = invert(e, phi)
            if isinstance(index, int):
                d += index * phi
            return n, d

    return False


def rsa_public_key(*args, **kwargs):
    r"""Return the RSA *public key* pair, `(n, e)`

    Parameters
    ==========

    args : naturals
        If specified as `p, q, e` where `p` and `q` are distinct primes
        and `e` is a desired public exponent of the RSA, `n = p q` and
        `e` will be verified against the totient
        `\phi(n)` (Euler totient) or `\lambda(n)` (Carmichael totient)
        to be `\gcd(e, \phi(n)) = 1` or `\gcd(e, \lambda(n)) = 1`.

        If specified as `p_1, p_2, \dots, p_n, e` where
        `p_1, p_2, \dots, p_n` are specified as primes,
        and `e` is specified as a desired public exponent of the RSA,
        it will be able to form a multi-prime RSA, which is a more
        generalized form of the popular 2-prime RSA.

        It can also be possible to form a single-prime RSA by specifying
        the argument as `p, e`, which can be considered a trivial case
        of a multiprime RSA.

        Furthermore, it can be possible to form a multi-power RSA by
        specifying two or more pairs of the primes to be same.
        However, unlike the two-distinct prime RSA or multi-prime
        RSA, not every numbers in the complete residue system
        (`\mathbb{Z}_n`) will be decryptable since the mapping
        `\mathbb{Z}_{n} \rightarrow \mathbb{Z}_{n}`
        will not be bijective.
        (Only except for the trivial case when
        `e = 1`
        or more generally,

        .. math::
            e \in \left \{ 1 + k \lambda(n)
            \mid k \in \mathbb{Z} \land k \geq 0 \right \}

        when RSA reduces to the identity.)
        However, the RSA can still be decryptable for the numbers in the
        reduced residue system (`\mathbb{Z}_n^{\times}`), since the
        mapping
        `\mathbb{Z}_{n}^{\times} \rightarrow \mathbb{Z}_{n}^{\times}`
        can still be bijective.

        If you pass a non-prime integer to the arguments
        `p_1, p_2, \dots, p_n`, the particular number will be
        prime-factored and it will become either a multi-prime RSA or a
        multi-power RSA in its canonical form, depending on whether the
        product equals its radical or not.
        `p_1 p_2 \dots p_n = \text{rad}(p_1 p_2 \dots p_n)`

    totient : bool, optional
        If ``'Euler'``, it uses Euler's totient `\phi(n)` which is
        :meth:`sympy.functions.combinatorial.numbers.totient` in SymPy.

        If ``'Carmichael'``, it uses Carmichael's totient `\lambda(n)`
        which is :meth:`sympy.functions.combinatorial.numbers.reduced_totient` in SymPy.

        Unlike private key generation, this is a trivial keyword for
        public key generation because
        `\gcd(e, \phi(n)) = 1 \iff \gcd(e, \lambda(n)) = 1`.

    index : nonnegative integer, optional
        Returns an arbitrary solution of a RSA public key at the index
        specified at `0, 1, 2, \dots`. This parameter needs to be
        specified along with ``totient='Carmichael'``.

        Similarly to the non-uniquenss of a RSA private key as described
        in the ``index`` parameter documentation in
        :meth:`rsa_private_key`, RSA public key is also not unique and
        there is an infinite number of RSA public exponents which
        can behave in the same manner.

        From any given RSA public exponent `e`, there are can be an
        another RSA public exponent `e + k \lambda(n)` where `k` is an
        integer, `\lambda` is a Carmichael's totient function.

        However, considering only the positive cases, there can be
        a principal solution of a RSA public exponent `e_0` in
        `0 < e_0 < \lambda(n)`, and all the other solutions
        can be canonicalzed in a form of `e_0 + k \lambda(n)`.

        ``index`` specifies the `k` notation to yield any possible value
        an RSA public key can have.

        An example of computing any arbitrary RSA public key:

        >>> from sympy.crypto.crypto import rsa_public_key
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=0)
        (3233, 17)
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=1)
        (3233, 797)
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=2)
        (3233, 1577)

    multipower : bool, optional
        Any pair of non-distinct primes found in the RSA specification
        will restrict the domain of the cryptosystem, as noted in the
        explanation of the parameter ``args``.

        SymPy RSA key generator may give a warning before dispatching it
        as a multi-power RSA, however, you can disable the warning if
        you pass ``True`` to this keyword.

    Returns
    =======

    (n, e) : int, int
        `n` is a product of any arbitrary number of primes given as
        the argument.

        `e` is relatively prime (coprime) to the Euler totient
        `\phi(n)`.

    False
        Returned if less than two arguments are given, or `e` is
        not relatively prime to the modulus.

    Examples
    ========

    >>> from sympy.crypto.crypto import rsa_public_key

    A public key of a two-prime RSA:

    >>> p, q, e = 3, 5, 7
    >>> rsa_public_key(p, q, e)
    (15, 7)
    >>> rsa_public_key(p, q, 30)
    False

    A public key of a multiprime RSA:

    >>> primes = [2, 3, 5, 7, 11, 13]
    >>> e = 7
    >>> args = primes + [e]
    >>> rsa_public_key(*args)
    (30030, 7)

    Notes
    =====

    Although the RSA can be generalized over any modulus `n`, using
    two large primes had became the most popular specification because a
    product of two large primes is usually the hardest to factor
    relatively to the digits of `n` can have.

    However, it may need further understanding of the time complexities
    of each prime-factoring algorithms to verify the claim.

    See Also
    ========

    rsa_private_key
    encipher_rsa
    decipher_rsa

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/RSA_%28cryptosystem%29

    .. [2] https://cacr.uwaterloo.ca/techreports/2006/cacr2006-16.pdf

    .. [3] https://link.springer.com/content/pdf/10.1007/BFb0055738.pdf

    .. [4] https://www.itiis.org/digital-library/manuscript/1381
    """
    return _rsa_key(*args, public=True, private=False, **kwargs)


def rsa_private_key(*args, **kwargs):
    r"""Return the RSA *private key* pair, `(n, d)`

    Parameters
    ==========

    args : naturals
        The keyword is identical to the ``args`` in
        :meth:`rsa_public_key`.

    totient : bool, optional
        If ``'Euler'``, it uses Euler's totient convention `\phi(n)`
        which is :meth:`sympy.functions.combinatorial.numbers.totient` in SymPy.

        If ``'Carmichael'``, it uses Carmichael's totient convention
        `\lambda(n)` which is
        :meth:`sympy.functions.combinatorial.numbers.reduced_totient` in SymPy.

        There can be some output differences for private key generation
        as examples below.

        Example using Euler's totient:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Euler')
        (3233, 2753)

        Example using Carmichael's totient:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Carmichael')
        (3233, 413)

    index : nonnegative integer, optional
        Returns an arbitrary solution of a RSA private key at the index
        specified at `0, 1, 2, \dots`. This parameter needs to be
        specified along with ``totient='Carmichael'``.

        RSA private exponent is a non-unique solution of
        `e d \mod \lambda(n) = 1` and it is possible in any form of
        `d + k \lambda(n)`, where `d` is an another
        already-computed private exponent, and `\lambda` is a
        Carmichael's totient function, and `k` is any integer.

        However, considering only the positive cases, there can be
        a principal solution of a RSA private exponent `d_0` in
        `0 < d_0 < \lambda(n)`, and all the other solutions
        can be canonicalzed in a form of `d_0 + k \lambda(n)`.

        ``index`` specifies the `k` notation to yield any possible value
        an RSA private key can have.

        An example of computing any arbitrary RSA private key:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=0)
        (3233, 413)
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=1)
        (3233, 1193)
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=2)
        (3233, 1973)

    multipower : bool, optional
        The keyword is identical to the ``multipower`` in
        :meth:`rsa_public_key`.

    Returns
    =======

    (n, d) : int, int
        `n` is a product of any arbitrary number of primes given as
        the argument.

        `d` is the inverse of `e` (mod `\phi(n)`) where `e` is the
        exponent given, and `\phi` is a Euler totient.

    False
        Returned if less than two arguments are given, or `e` is
        not relatively prime to the totient of the modulus.

    Examples
    ========

    >>> from sympy.crypto.crypto import rsa_private_key

    A private key of a two-prime RSA:

    >>> p, q, e = 3, 5, 7
    >>> rsa_private_key(p, q, e)
    (15, 7)
    >>> rsa_private_key(p, q, 30)
    False

    A private key of a multiprime RSA:

    >>> primes = [2, 3, 5, 7, 11, 13]
    >>> e = 7
    >>> args = primes + [e]
    >>> rsa_private_key(*args)
    (30030, 823)

    See Also
    ========

    rsa_public_key
    encipher_rsa
    decipher_rsa

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/RSA_%28cryptosystem%29

    .. [2] https://cacr.uwaterloo.ca/techreports/2006/cacr2006-16.pdf

    .. [3] https://link.springer.com/content/pdf/10.1007/BFb0055738.pdf

    .. [4] https://www.itiis.org/digital-library/manuscript/1381
    """
    return _rsa_key(*args, public=False, private=True, **kwargs)


def _encipher_decipher_rsa(i, key, factors=None):
    n, d = key
    if not factors:
        return pow(i, d, n)

    def _is_coprime_set(l):
        is_coprime_set = True
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                if gcd(l[i], l[j]) != 1:
                    is_coprime_set = False
                    break
        return is_coprime_set

    prod = reduce(lambda i, j: i*j, factors)
    if prod == n and _is_coprime_set(factors):
        return _decipher_rsa_crt(i, d, factors)
    return _encipher_decipher_rsa(i, key, factors=None)


def encipher_rsa(i, key, factors=None):
    r"""Encrypt the plaintext with RSA.

    Parameters
    ==========

    i : integer
        The plaintext to be encrypted for.

    key : (n, e) where n, e are integers
        `n` is the modulus of the key and `e` is the exponent of the
        key. The encryption is computed by `i^e \bmod n`.

        The key can either be a public key or a private key, however,
        the message encrypted by a public key can only be decrypted by
        a private key, and vice versa, as RSA is an asymmetric
        cryptography system.

    factors : list of coprime integers
        This is identical to the keyword ``factors`` in
        :meth:`decipher_rsa`.

    Notes
    =====

    Some specifications may make the RSA not cryptographically
    meaningful.

    For example, `0`, `1` will remain always same after taking any
    number of exponentiation, thus, should be avoided.

    Furthermore, if `i^e < n`, `i` may easily be figured out by taking
    `e` th root.

    And also, specifying the exponent as `1` or in more generalized form
    as `1 + k \lambda(n)` where `k` is an nonnegative integer,
    `\lambda` is a carmichael totient, the RSA becomes an identity
    mapping.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_rsa
    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key

    Public Key Encryption:

    >>> p, q, e = 3, 5, 7
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> encipher_rsa(msg, puk)
    3

    Private Key Encryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> msg = 12
    >>> encipher_rsa(msg, prk)
    3

    Encryption using chinese remainder theorem:

    >>> encipher_rsa(msg, prk, factors=[p, q])
    3
    """
    return _encipher_decipher_rsa(i, key, factors=factors)


def decipher_rsa(i, key, factors=None):
    r"""Decrypt the ciphertext with RSA.

    Parameters
    ==========

    i : integer
        The ciphertext to be decrypted for.

    key : (n, d) where n, d are integers
        `n` is the modulus of the key and `d` is the exponent of the
        key. The decryption is computed by `i^d \bmod n`.

        The key can either be a public key or a private key, however,
        the message encrypted by a public key can only be decrypted by
        a private key, and vice versa, as RSA is an asymmetric
        cryptography system.

    factors : list of coprime integers
        As the modulus `n` created from RSA key generation is composed
        of arbitrary prime factors
        `n = {p_1}^{k_1}{p_2}^{k_2}\dots{p_n}^{k_n}` where
        `p_1, p_2, \dots, p_n` are distinct primes and
        `k_1, k_2, \dots, k_n` are positive integers, chinese remainder
        theorem can be used to compute `i^d \bmod n` from the
        fragmented modulo operations like

        .. math::
            i^d \bmod {p_1}^{k_1}, i^d \bmod {p_2}^{k_2}, \dots,
            i^d \bmod {p_n}^{k_n}

        or like

        .. math::
            i^d \bmod {p_1}^{k_1}{p_2}^{k_2},
            i^d \bmod {p_3}^{k_3}, \dots ,
            i^d \bmod {p_n}^{k_n}

        as long as every moduli does not share any common divisor each
        other.

        The raw primes used in generating the RSA key pair can be a good
        option.

        Note that the speed advantage of using this is only viable for
        very large cases (Like 2048-bit RSA keys) since the
        overhead of using pure Python implementation of
        :meth:`sympy.ntheory.modular.crt` may overcompensate the
        theoretical speed advantage.

    Notes
    =====

    See the ``Notes`` section in the documentation of
    :meth:`encipher_rsa`

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_rsa, encipher_rsa
    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key

    Public Key Encryption and Decryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> new_msg = encipher_rsa(msg, prk)
    >>> new_msg
    3
    >>> decipher_rsa(new_msg, puk)
    12

    Private Key Encryption and Decryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> new_msg = encipher_rsa(msg, puk)
    >>> new_msg
    3
    >>> decipher_rsa(new_msg, prk)
    12

    Decryption using chinese remainder theorem:

    >>> decipher_rsa(new_msg, prk, factors=[p, q])
    12

    See Also
    ========

    encipher_rsa
    """
    return _encipher_decipher_rsa(i, key, factors=factors)


#################### kid krypto (kid RSA) #############################


def kid_rsa_public_key(a, b, A, B):
    r"""
    Kid RSA is a version of RSA useful to teach grade school children
    since it does not involve exponentiation.

    Explanation
    ===========

    Alice wants to talk to Bob. Bob generates keys as follows.
    Key generation:

    * Select positive integers `a, b, A, B` at random.
    * Compute `M = a b - 1`, `e = A M + a`, `d = B M + b`,
      `n = (e d - 1)//M`.
    * The *public key* is `(n, e)`. Bob sends these to Alice.
    * The *private key* is `(n, d)`, which Bob keeps secret.

    Encryption: If `p` is the plaintext message then the
    ciphertext is `c = p e \pmod n`.

    Decryption: If `c` is the ciphertext message then the
    plaintext is `p = c d \pmod n`.

    Examples
    ========

    >>> from sympy.crypto.crypto import kid_rsa_public_key
    >>> a, b, A, B = 3, 4, 5, 6
    >>> kid_rsa_public_key(a, b, A, B)
    (369, 58)

    """
    M = a*b - 1
    e = A*M + a
    d = B*M + b
    n = (e*d - 1)//M
    return n, e


def kid_rsa_private_key(a, b, A, B):
    """
    Compute `M = a b - 1`, `e = A M + a`, `d = B M + b`,
    `n = (e d - 1) / M`. The *private key* is `d`, which Bob
    keeps secret.

    Examples
    ========

    >>> from sympy.crypto.crypto import kid_rsa_private_key
    >>> a, b, A, B = 3, 4, 5, 6
    >>> kid_rsa_private_key(a, b, A, B)
    (369, 70)

    """
    M = a*b - 1
    e = A*M + a
    d = B*M + b
    n = (e*d - 1)//M
    return n, d


def encipher_kid_rsa(msg, key):
    """
    Here ``msg`` is the plaintext and ``key`` is the public key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_kid_rsa, kid_rsa_public_key)
    >>> msg = 200
    >>> a, b, A, B = 3, 4, 5, 6
    >>> key = kid_rsa_public_key(a, b, A, B)
    >>> encipher_kid_rsa(msg, key)
    161

    """
    n, e = key
    return (msg*e) % n


def decipher_kid_rsa(msg, key):
    """
    Here ``msg`` is the plaintext and ``key`` is the private key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     kid_rsa_public_key, kid_rsa_private_key,
    ...     decipher_kid_rsa, encipher_kid_rsa)
    >>> a, b, A, B = 3, 4, 5, 6
    >>> d = kid_rsa_private_key(a, b, A, B)
    >>> msg = 200
    >>> pub = kid_rsa_public_key(a, b, A, B)
    >>> pri = kid_rsa_private_key(a, b, A, B)
    >>> ct = encipher_kid_rsa(msg, pub)
    >>> decipher_kid_rsa(ct, pri)
    200

    """
    n, d = key
    return (msg*d) % n


#################### Morse Code ######################################

morse_char = {
    ".-": "A", "-...": "B",
    "-.-.": "C", "-..": "D",
    ".": "E", "..-.": "F",
    "--.": "G", "....": "H",
    "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N",
    "---": "O", ".--.": "P",
    "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T",
    "..-": "U", "...-": "V",
    ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1",
    "..---": "2", "...--": "3",
    "....-": "4", ".....": "5",
    "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",",
    "---...": ":", "-.-.-.": ";",
    "..--..": "?", "-....-": "-",
    "..--.-": "_", "-.--.": "(",
    "-.--.-": ")", ".----.": "'",
    "-...-": "=", ".-.-.": "+",
    "-..-.": "/", ".--.-.": "@",
    "...-..-": "$", "-.-.--": "!"}
char_morse = {v: k for k, v in morse_char.items()}


def encode_morse(msg, sep='|', mapping=None):
    """
    Encodes a plaintext into popular Morse Code with letters
    separated by ``sep`` and words by a double ``sep``.

    Examples
    ========

    >>> from sympy.crypto.crypto import encode_morse
    >>> msg = 'ATTACK RIGHT FLANK'
    >>> encode_morse(msg)
    '.-|-|-|.-|-.-.|-.-||.-.|..|--.|....|-||..-.|.-..|.-|-.|-.-'

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morse_code

    """

    mapping = mapping or char_morse
    assert sep not in mapping
    word_sep = 2*sep
    mapping[" "] = word_sep
    suffix = msg and msg[-1] in whitespace

    # normalize whitespace
    msg = (' ' if word_sep else '').join(msg.split())
    # omit unmapped chars
    chars = set(''.join(msg.split()))
    ok = set(mapping.keys())
    msg = translate(msg, None, ''.join(chars - ok))

    morsestring = []
    words = msg.split()
    for word in words:
        morseword = []
        for letter in word:
            morseletter = mapping[letter]
            morseword.append(morseletter)

        word = sep.join(morseword)
        morsestring.append(word)

    return word_sep.join(morsestring) + (word_sep if suffix else '')


def decode_morse(msg, sep='|', mapping=None):
    """
    Decodes a Morse Code with letters separated by ``sep``
    (default is '|') and words by `word_sep` (default is '||)
    into plaintext.

    Examples
    ========

    >>> from sympy.crypto.crypto import decode_morse
    >>> mc = '--|---|...-|.||.|.-|...|-'
    >>> decode_morse(mc)
    'MOVE EAST'

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morse_code

    """

    mapping = mapping or morse_char
    word_sep = 2*sep
    characterstring = []
    words = msg.strip(word_sep).split(word_sep)
    for word in words:
        letters = word.split(sep)
        chars = [mapping[c] for c in letters]
        word = ''.join(chars)
        characterstring.append(word)
    rv = " ".join(characterstring)
    return rv


#################### LFSRs  ##########################################


@doctest_depends_on(ground_types=['python', 'gmpy'])
def lfsr_sequence(key, fill, n):
    r"""
    This function creates an LFSR sequence.

    Parameters
    ==========

    key : list
        A list of finite field elements, `[c_0, c_1, \ldots, c_k].`

    fill : list
        The list of the initial terms of the LFSR sequence,
        `[x_0, x_1, \ldots, x_k].`

    n
        Number of terms of the sequence that the function returns.

    Returns
    =======

    L
        The LFSR sequence defined by
        `x_{n+1} = c_k x_n + \ldots + c_0 x_{n-k}`, for
        `n \leq k`.

    Notes
    =====

    S. Golomb [G]_ gives a list of three statistical properties a
    sequence of numbers `a = \{a_n\}_{n=1}^\infty`,
    `a_n \in \{0,1\}`, should display to be considered
    "random". Define the autocorrelation of `a` to be

    .. math::

        C(k) = C(k,a) = \lim_{N\rightarrow \infty} {1\over N}\sum_{n=1}^N (-1)^{a_n + a_{n+k}}.

    In the case where `a` is periodic with period
    `P` then this reduces to

    .. math::

        C(k) = {1\over P}\sum_{n=1}^P (-1)^{a_n + a_{n+k}}.

    Assume `a` is periodic with period `P`.

    - balance:

      .. math::

        \left|\sum_{n=1}^P(-1)^{a_n}\right| \leq 1.

    - low autocorrelation:

       .. math::

         C(k) = \left\{ \begin{array}{cc} 1,& k = 0,\\ \epsilon, & k \ne 0. \end{array} \right.

      (For sequences satisfying these first two properties, it is known
      that `\epsilon = -1/P` must hold.)

    - proportional runs property: In each period, half the runs have
      length `1`, one-fourth have length `2`, etc.
      Moreover, there are as many runs of `1`'s as there are of
      `0`'s.

    Examples
    ========

    >>> from sympy.crypto.crypto import lfsr_sequence
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> lfsr_sequence(key, fill, 10)
    [1 mod 2, 1 mod 2, 0 mod 2, 1 mod 2, 0 mod 2,
    1 mod 2, 1 mod 2, 0 mod 2, 0 mod 2, 1 mod 2]

    References
    ==========

    .. [G] Solomon Golomb, Shift register sequences, Aegean Park Press,
       Laguna Hills, Ca, 1967

    """
    if not isinstance(key, list):
        raise TypeError("key must be a list")
    if not isinstance(fill, list):
        raise TypeError("fill must be a list")
    p = key[0].modulus()
    F = FF(p)
    s = fill
    k = len(fill)
    L = []
    for i in range(n):
        s0 = s[:]
        L.append(s[0])
        s = s[1:k]
        x = sum(int(key[i]*s0[i]) for i in range(k))
        s.append(F(x))
    return L       # use [int(x) for x in L] for int version


def lfsr_autocorrelation(L, P, k):
    """
    This function computes the LFSR autocorrelation function.

    Parameters
    ==========

    L
        A periodic sequence of elements of `GF(2)`.
        L must have length larger than P.

    P
        The period of L.

    k : int
        An integer `k` (`0 < k < P`).

    Returns
    =======

    autocorrelation
        The k-th value of the autocorrelation of the LFSR L.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     lfsr_sequence, lfsr_autocorrelation)
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_autocorrelation(s, 15, 7)
    -1/15
    >>> lfsr_autocorrelation(s, 15, 0)
    1

    """
    if not isinstance(L, list):
        raise TypeError("L (=%s) must be a list" % L)
    P = int(P)
    k = int(k)
    L0 = L[:P]     # slices makes a copy
    L1 = L0 + L0[:k]
    L2 = [(-1)**(int(L1[i]) + int(L1[i + k])) for i in range(P)]
    tot = sum(L2)
    return Rational(tot, P)


def lfsr_connection_polynomial(s):
    """
    This function computes the LFSR connection polynomial.

    Parameters
    ==========

    s
        A sequence of elements of even length, with entries in a finite
        field.

    Returns
    =======

    C(x)
        The connection polynomial of a minimal LFSR yielding s.

        This implements the algorithm in section 3 of J. L. Massey's
        article [M]_.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     lfsr_sequence, lfsr_connection_polynomial)
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**4 + x + 1
    >>> fill = [F(1), F(0), F(0), F(1)]
    >>> key = [F(1), F(1), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + 1
    >>> fill = [F(1), F(0), F(1)]
    >>> key = [F(1), F(1), F(0)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + x**2 + 1
    >>> fill = [F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + x + 1

    References
    ==========

    .. [M] James L. Massey, "Shift-Register Synthesis and BCH Decoding."
        IEEE Trans. on Information Theory, vol. 15(1), pp. 122-127,
        Jan 1969.

    """
    # Initialization:
    p = s[0].modulus()
    x = Symbol("x")
    C = 1*x**0
    B = 1*x**0
    m = 1
    b = 1*x**0
    L = 0
    N = 0
    while N < len(s):
        if L > 0:
            dC = Poly(C).degree()
            r = min(L + 1, dC + 1)
            coeffsC = [C.subs(x, 0)] + [C.coeff(x**i)
                for i in range(1, dC + 1)]
            d = (int(s[N]) + sum(coeffsC[i]*int(s[N - i])
                for i in range(1, r))) % p
        if L == 0:
            d = int(s[N])*x**0
        if d == 0:
            m += 1
            N += 1
        if d > 0:
            if 2*L > N:
                C = (C - d*((b**(p - 2)) % p)*x**m*B).expand()
                m += 1
                N += 1
            else:
                T = C
                C = (C - d*((b**(p - 2)) % p)*x**m*B).expand()
                L = N + 1 - L
                m = 1
                b = d
                B = T
                N += 1
    dC = Poly(C).degree()
    coeffsC = [C.subs(x, 0)] + [C.coeff(x**i) for i in range(1, dC + 1)]
    return sum(coeffsC[i] % p*x**i for i in range(dC + 1)
        if coeffsC[i] is not None)


#################### ElGamal  #############################


def elgamal_private_key(digit=10, seed=None):
    r"""
    Return three number tuple as private key.

    Explanation
    ===========

    Elgamal encryption is based on the mathematical problem
    called the Discrete Logarithm Problem (DLP). For example,

    `a^{b} \equiv c \pmod p`

    In general, if ``a`` and ``b`` are known, ``ct`` is easily
    calculated. If ``b`` is unknown, it is hard to use
    ``a`` and ``ct`` to get ``b``.

    Parameters
    ==========

    digit : int
        Minimum number of binary digits for key.

    Returns
    =======

    tuple : (p, r, d)
        p = prime number.

        r = primitive root.

        d = random number.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import elgamal_private_key
    >>> from sympy.ntheory import is_primitive_root, isprime
    >>> a, b, _ = elgamal_private_key()
    >>> isprime(a)
    True
    >>> is_primitive_root(b, a)
    True

    """
    randrange = _randrange(seed)
    p = nextprime(2**digit)
    return p, primitive_root(p), randrange(2, p)


def elgamal_public_key(key):
    r"""
    Return three number tuple as public key.

    Parameters
    ==========

    key : (p, r, e)
        Tuple generated by ``elgamal_private_key``.

    Returns
    =======

    tuple : (p, r, e)
        `e = r**d \bmod p`

        `d` is a random number in private key.

    Examples
    ========

    >>> from sympy.crypto.crypto import elgamal_public_key
    >>> elgamal_public_key((1031, 14, 636))
    (1031, 14, 212)

    """
    p, r, e = key
    return p, r, pow(r, e, p)


def encipher_elgamal(i, key, seed=None):
    r"""
    Encrypt message with public key.

    Explanation
    ===========

    ``i`` is a plaintext message expressed as an integer.
    ``key`` is public key (p, r, e). In order to encrypt
    a message, a random number ``a`` in ``range(2, p)``
    is generated and the encrypted message is returned as
    `c_{1}` and `c_{2}` where:

    `c_{1} \equiv r^{a} \pmod p`

    `c_{2} \equiv m e^{a} \pmod p`

    Parameters
    ==========

    msg
        int of encoded message.

    key
        Public key.

    Returns
    =======

    tuple : (c1, c2)
        Encipher into two number.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_elgamal, elgamal_private_key, elgamal_public_key
    >>> pri = elgamal_private_key(5, seed=[3]); pri
    (37, 2, 3)
    >>> pub = elgamal_public_key(pri); pub
    (37, 2, 8)
    >>> msg = 36
    >>> encipher_elgamal(msg, pub, seed=[3])
    (8, 6)

    """
    p, r, e = key
    if i < 0 or i >= p:
        raise ValueError(
            'Message (%s) should be in range(%s)' % (i, p))
    randrange = _randrange(seed)
    a = randrange(2, p)
    return pow(r, a, p), i*pow(e, a, p) % p


def decipher_elgamal(msg, key):
    r"""
    Decrypt message with private key.

    `msg = (c_{1}, c_{2})`

    `key = (p, r, d)`

    According to extended Eucliden theorem,
    `u c_{1}^{d} + p n = 1`

    `u \equiv 1/{{c_{1}}^d} \pmod p`

    `u c_{2} \equiv \frac{1}{c_{1}^d} c_{2} \equiv \frac{1}{r^{ad}} c_{2} \pmod p`

    `\frac{1}{r^{ad}} m e^a \equiv \frac{1}{r^{ad}} m {r^{d a}} \equiv m \pmod p`

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_elgamal
    >>> from sympy.crypto.crypto import encipher_elgamal
    >>> from sympy.crypto.crypto import elgamal_private_key
    >>> from sympy.crypto.crypto import elgamal_public_key

    >>> pri = elgamal_private_key(5, seed=[3])
    >>> pub = elgamal_public_key(pri); pub
    (37, 2, 8)
    >>> msg = 17
    >>> decipher_elgamal(encipher_elgamal(msg, pub), pri) == msg
    True

    """
    p, _, d = key
    c1, c2 = msg
    u = pow(c1, -d, p)
    return u * c2 % p


################ Diffie-Hellman Key Exchange  #########################

def dh_private_key(digit=10, seed=None):
    r"""
    Return three integer tuple as private key.

    Explanation
    ===========

    Diffie-Hellman key exchange is based on the mathematical problem
    called the Discrete Logarithm Problem (see ElGamal).

    Diffie-Hellman key exchange is divided into the following steps:

    *   Alice and Bob agree on a base that consist of a prime ``p``
        and a primitive root of ``p`` called ``g``
    *   Alice choses a number ``a`` and Bob choses a number ``b`` where
        ``a`` and ``b`` are random numbers in range `[2, p)`. These are
        their private keys.
    *   Alice then publicly sends Bob `g^{a} \pmod p` while Bob sends
        Alice `g^{b} \pmod p`
    *   They both raise the received value to their secretly chosen
        number (``a`` or ``b``) and now have both as their shared key
        `g^{ab} \pmod p`

    Parameters
    ==========

    digit
        Minimum number of binary digits required in key.

    Returns
    =======

    tuple : (p, g, a)
        p = prime number.

        g = primitive root of p.

        a = random number from 2 through p - 1.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import dh_private_key
    >>> from sympy.ntheory import isprime, is_primitive_root
    >>> p, g, _ = dh_private_key()
    >>> isprime(p)
    True
    >>> is_primitive_root(g, p)
    True
    >>> p, g, _ = dh_private_key(5)
    >>> isprime(p)
    True
    >>> is_primitive_root(g, p)
    True

    """
    p = nextprime(2**digit)
    g = primitive_root(p)
    randrange = _randrange(seed)
    a = randrange(2, p)
    return p, g, a


def dh_public_key(key):
    r"""
    Return three number tuple as public key.

    This is the tuple that Alice sends to Bob.

    Parameters
    ==========

    key : (p, g, a)
        A tuple generated by ``dh_private_key``.

    Returns
    =======

    tuple : int, int, int
        A tuple of `(p, g, g^a \mod p)` with `p`, `g` and `a` given as
        parameters.s

    Examples
    ========

    >>> from sympy.crypto.crypto import dh_private_key, dh_public_key
    >>> p, g, a = dh_private_key();
    >>> _p, _g, x = dh_public_key((p, g, a))
    >>> p == _p and g == _g
    True
    >>> x == pow(g, a, p)
    True

    """
    p, g, a = key
    return p, g, pow(g, a, p)


def dh_shared_key(key, b):
    """
    Return an integer that is the shared key.

    This is what Bob and Alice can both calculate using the public
    keys they received from each other and their private keys.

    Parameters
    ==========

    key : (p, g, x)
        Tuple `(p, g, x)` generated by ``dh_public_key``.

    b
        Random number in the range of `2` to `p - 1`
        (Chosen by second key exchange member (Bob)).

    Returns
    =======

    int
        A shared key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     dh_private_key, dh_public_key, dh_shared_key)
    >>> prk = dh_private_key();
    >>> p, g, x = dh_public_key(prk);
    >>> sk = dh_shared_key((p, g, x), 1000)
    >>> sk == pow(x, 1000, p)
    True

    """
    p, _, x = key
    if 1 >= b or b >= p:
        raise ValueError(filldedent('''
            Value of b should be greater 1 and less
            than prime %s.''' % p))

    return pow(x, b, p)


################ Goldwasser-Micali Encryption  #########################


def _legendre(a, p):
    """
    Returns the legendre symbol of a and p
    assuming that p is a prime.

    i.e. 1 if a is a quadratic residue mod p
        -1 if a is not a quadratic residue mod p
         0 if a is divisible by p

    Parameters
    ==========

    a : int
        The number to test.

    p : prime
        The prime to test ``a`` against.

    Returns
    =======

    int
        Legendre symbol (a / p).

    """
    sig = pow(a, (p - 1)//2, p)
    if sig == 1:
        return 1
    elif sig == 0:
        return 0
    else:
        return -1


def _random_coprime_stream(n, seed=None):
    randrange = _randrange(seed)
    while True:
        y = randrange(n)
        if gcd(y, n) == 1:
            yield y


def gm_private_key(p, q, a=None):
    r"""
    Check if ``p`` and ``q`` can be used as private keys for
    the Goldwasser-Micali encryption. The method works
    roughly as follows.

    Explanation
    ===========

    #. Pick two large primes $p$ and $q$.
    #. Call their product $N$.
    #. Given a message as an integer $i$, write $i$ in its bit representation $b_0, \dots, b_n$.
    #. For each $k$,

     if $b_k = 0$:
        let $a_k$ be a random square
        (quadratic residue) modulo $p q$
        such that ``jacobi_symbol(a, p*q) = 1``
     if $b_k = 1$:
        let $a_k$ be a random non-square
        (non-quadratic residue) modulo $p q$
        such that ``jacobi_symbol(a, p*q) = 1``

    returns $\left[a_1, a_2, \dots\right]$

    $b_k$ can be recovered by checking whether or not
    $a_k$ is a residue. And from the $b_k$'s, the message
    can be reconstructed.

    The idea is that, while ``jacobi_symbol(a, p*q)``
    can be easily computed (and when it is equal to $-1$ will
    tell you that $a$ is not a square mod $p q$), quadratic
    residuosity modulo a composite number is hard to compute
    without knowing its factorization.

    Moreover, approximately half the numbers coprime to $p q$ have
    :func:`~.jacobi_symbol` equal to $1$ . And among those, approximately half
    are residues and approximately half are not. This maximizes the
    entropy of the code.

    Parameters
    ==========

    p, q, a
        Initialization variables.

    Returns
    =======

    tuple : (p, q)
        The input value ``p`` and ``q``.

    Raises
    ======

    ValueError
        If ``p`` and ``q`` are not distinct odd primes.

    """
    if p == q:
        raise ValueError("expected distinct primes, "
                         "got two copies of %i" % p)
    elif not isprime(p) or not isprime(q):
        raise ValueError("first two arguments must be prime, "
                         "got %i of %i" % (p, q))
    elif p == 2 or q == 2:
        raise ValueError("first two arguments must not be even, "
                         "got %i of %i" % (p, q))
    return p, q


def gm_public_key(p, q, a=None, seed=None):
    """
    Compute public keys for ``p`` and ``q``.
    Note that in Goldwasser-Micali Encryption,
    public keys are randomly selected.

    Parameters
    ==========

    p, q, a : int, int, int
        Initialization variables.

    Returns
    =======

    tuple : (a, N)
        ``a`` is the input ``a`` if it is not ``None`` otherwise
        some random integer coprime to ``p`` and ``q``.

        ``N`` is the product of ``p`` and ``q``.

    """

    p, q = gm_private_key(p, q)
    N = p * q

    if a is None:
        randrange = _randrange(seed)
        while True:
            a = randrange(N)
            if _legendre(a, p) == _legendre(a, q) == -1:
                break
    else:
        if _legendre(a, p) != -1 or _legendre(a, q) != -1:
            return False
    return (a, N)


def encipher_gm(i, key, seed=None):
    """
    Encrypt integer 'i' using public_key 'key'
    Note that gm uses random encryption.

    Parameters
    ==========

    i : int
        The message to encrypt.

    key : (a, N)
        The public key.

    Returns
    =======

    list : list of int
        The randomized encrypted message.

    """
    if i < 0:
        raise ValueError(
            "message must be a non-negative "
            "integer: got %d instead" % i)
    a, N = key
    bits = []
    while i > 0:
        bits.append(i % 2)
        i //= 2

    gen = _random_coprime_stream(N, seed)
    rev = reversed(bits)
    encode = lambda b: next(gen)**2*pow(a, b) % N
    return [ encode(b) for b in rev ]



def decipher_gm(message, key):
    """
    Decrypt message 'message' using public_key 'key'.

    Parameters
    ==========

    message : list of int
        The randomized encrypted message.

    key : (p, q)
        The private key.

    Returns
    =======

    int
        The encrypted message.

    """
    p, q = key
    res = lambda m, p: _legendre(m, p) > 0
    bits = [res(m, p) * res(m, q) for m in message]
    m = 0
    for b in bits:
        m <<= 1
        m += not b
    return m



########### RailFence Cipher #############

def encipher_railfence(message,rails):
    """
    Performs Railfence Encryption on plaintext and returns ciphertext

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_railfence
    >>> message = "hello world"
    >>> encipher_railfence(message,3)
    'horel ollwd'

    Parameters
    ==========

    message : string, the message to encrypt.
    rails : int, the number of rails.

    Returns
    =======

    The Encrypted string message.

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Rail_fence_cipher

    """
    r = list(range(rails))
    p = cycle(r + r[-2:0:-1])
    return ''.join(sorted(message, key=lambda i: next(p)))


def decipher_railfence(ciphertext,rails):
    """
    Decrypt the message using the given rails

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_railfence
    >>> decipher_railfence("horel ollwd",3)
    'hello world'

    Parameters
    ==========

    message : string, the message to encrypt.
    rails : int, the number of rails.

    Returns
    =======

    The Decrypted string message.

    """
    r = list(range(rails))
    p = cycle(r + r[-2:0:-1])

    idx = sorted(range(len(ciphertext)), key=lambda i: next(p))
    res = [''] * len(ciphertext)
    for i, c in zip(idx, ciphertext):
        res[i] = c
    return ''.join(res)


################ Blum-Goldwasser cryptosystem  #########################

def bg_private_key(p, q):
    """
    Check if p and q can be used as private keys for
    the Blum-Goldwasser cryptosystem.

    Explanation
    ===========

    The three necessary checks for p and q to pass
    so that they can be used as private keys:

        1. p and q must both be prime
        2. p and q must be distinct
        3. p and q must be congruent to 3 mod 4

    Parameters
    ==========

    p, q
        The keys to be checked.

    Returns
    =======

    p, q
        Input values.

    Raises
    ======

    ValueError
        If p and q do not pass the above conditions.

    """

    if not isprime(p) or not isprime(q):
        raise ValueError("the two arguments must be prime, "
                         "got %i and %i" %(p, q))
    elif p == q:
        raise ValueError("the two arguments must be distinct, "
                         "got two copies of %i. " %p)
    elif (p - 3) % 4 != 0 or (q - 3) % 4 != 0:
        raise ValueError("the two arguments must be congruent to 3 mod 4, "
                         "got %i and %i" %(p, q))
    return p, q

def bg_public_key(p, q):
    """
    Calculates public keys from private keys.

    Explanation
    ===========

    The function first checks the validity of
    private keys passed as arguments and
    then returns their product.

    Parameters
    ==========

    p, q
        The private keys.

    Returns
    =======

    N
        The public key.

    """
    p, q = bg_private_key(p, q)
    N = p * q
    return N

def encipher_bg(i, key, seed=None):
    """
    Encrypts the message using public key and seed.

    Explanation
    ===========

    ALGORITHM:
        1. Encodes i as a string of L bits, m.
        2. Select a random element r, where 1 < r < key, and computes
           x = r^2 mod key.
        3. Use BBS pseudo-random number generator to generate L random bits, b,
        using the initial seed as x.
        4. Encrypted message, c_i = m_i XOR b_i, 1 <= i <= L.
        5. x_L = x^(2^L) mod key.
        6. Return (c, x_L)

    Parameters
    ==========

    i
        Message, a non-negative integer

    key
        The public key

    Returns
    =======

    Tuple
        (encrypted_message, x_L)

    Raises
    ======

    ValueError
        If i is negative.

    """

    if i < 0:
        raise ValueError(
            "message must be a non-negative "
            "integer: got %d instead" % i)

    enc_msg = []
    while i > 0:
        enc_msg.append(i % 2)
        i //= 2
    enc_msg.reverse()
    L = len(enc_msg)

    r = _randint(seed)(2, key - 1)
    x = r**2 % key
    x_L = pow(int(x), int(2**L), int(key))

    rand_bits = []
    for _ in range(L):
        rand_bits.append(x % 2)
        x = x**2 % key

    encrypt_msg = [m ^ b for (m, b) in zip(enc_msg, rand_bits)]

    return (encrypt_msg, x_L)

def decipher_bg(message, key):
    """
    Decrypts the message using private keys.

    Explanation
    ===========

    ALGORITHM:
        1. Let, c be the encrypted message, y the second number received,
        and p and q be the private keys.
        2. Compute, r_p = y^((p+1)/4 ^ L) mod p and
        r_q = y^((q+1)/4 ^ L) mod q.
        3. Compute x_0 = (q(q^-1 mod p)r_p + p(p^-1 mod q)r_q) mod N.
        4. From, recompute the bits using the BBS generator, as in the
        encryption algorithm.
        5. Compute original message by XORing c and b.

    Parameters
    ==========

    message
        Tuple of encrypted message and a non-negative integer.

    key
        Tuple of private keys.

    Returns
    =======

    orig_msg
        The original message

    """

    p, q = key
    encrypt_msg, y = message
    public_key = p * q
    L = len(encrypt_msg)
    p_t = ((p + 1)/4)**L
    q_t = ((q + 1)/4)**L
    r_p = pow(int(y), int(p_t), int(p))
    r_q = pow(int(y), int(q_t), int(q))

    x = (q * invert(q, p) * r_p + p * invert(p, q) * r_q) % public_key

    orig_bits = []
    for _ in range(L):
        orig_bits.append(x % 2)
        x = x**2 % public_key

    orig_msg = 0
    for (m, b) in zip(encrypt_msg, orig_bits):
        orig_msg = orig_msg * 2
        orig_msg += (m ^ b)

    return orig_msg

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\descriptor_pb2.py ===
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: google/protobuf/descriptor.proto
# Protobuf Python Version: 6.31.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'google/protobuf/descriptor.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR = _descriptor.FileDescriptor(
    name='google/protobuf/descriptor.proto',
    package='google.protobuf',
    syntax='proto2',
    edition='EDITION_PROTO2',
    serialized_options=b'\n\023com.google.protobufB\020DescriptorProtosH\001Z-google.golang.org/protobuf/types/descriptorpb\370\001\001\242\002\003GPB\252\002\032Google.Protobuf.Reflection',
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n google/protobuf/descriptor.proto\x12\x0fgoogle.protobuf\"[\n\x11\x46ileDescriptorSet\x12\x38\n\x04\x66ile\x18\x01 \x03(\x0b\x32$.google.protobuf.FileDescriptorProtoR\x04\x66ile*\x0c\x08\x80\xec\xca\xff\x01\x10\x81\xec\xca\xff\x01\"\xc5\x05\n\x13\x46ileDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x18\n\x07package\x18\x02 \x01(\tR\x07package\x12\x1e\n\ndependency\x18\x03 \x03(\tR\ndependency\x12+\n\x11public_dependency\x18\n \x03(\x05R\x10publicDependency\x12\'\n\x0fweak_dependency\x18\x0b \x03(\x05R\x0eweakDependency\x12+\n\x11option_dependency\x18\x0f \x03(\tR\x10optionDependency\x12\x43\n\x0cmessage_type\x18\x04 \x03(\x0b\x32 .google.protobuf.DescriptorProtoR\x0bmessageType\x12\x41\n\tenum_type\x18\x05 \x03(\x0b\x32$.google.protobuf.EnumDescriptorProtoR\x08\x65numType\x12\x41\n\x07service\x18\x06 \x03(\x0b\x32\'.google.protobuf.ServiceDescriptorProtoR\x07service\x12\x43\n\textension\x18\x07 \x03(\x0b\x32%.google.protobuf.FieldDescriptorProtoR\textension\x12\x36\n\x07options\x18\x08 \x01(\x0b\x32\x1c.google.protobuf.FileOptionsR\x07options\x12I\n\x10source_code_info\x18\t \x01(\x0b\x32\x1f.google.protobuf.SourceCodeInfoR\x0esourceCodeInfo\x12\x16\n\x06syntax\x18\x0c \x01(\tR\x06syntax\x12\x32\n\x07\x65\x64ition\x18\x0e \x01(\x0e\x32\x18.google.protobuf.EditionR\x07\x65\x64ition\"\xfc\x06\n\x0f\x44\x65scriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12;\n\x05\x66ield\x18\x02 \x03(\x0b\x32%.google.protobuf.FieldDescriptorProtoR\x05\x66ield\x12\x43\n\textension\x18\x06 \x03(\x0b\x32%.google.protobuf.FieldDescriptorProtoR\textension\x12\x41\n\x0bnested_type\x18\x03 \x03(\x0b\x32 .google.protobuf.DescriptorProtoR\nnestedType\x12\x41\n\tenum_type\x18\x04 \x03(\x0b\x32$.google.protobuf.EnumDescriptorProtoR\x08\x65numType\x12X\n\x0f\x65xtension_range\x18\x05 \x03(\x0b\x32/.google.protobuf.DescriptorProto.ExtensionRangeR\x0e\x65xtensionRange\x12\x44\n\noneof_decl\x18\x08 \x03(\x0b\x32%.google.protobuf.OneofDescriptorProtoR\toneofDecl\x12\x39\n\x07options\x18\x07 \x01(\x0b\x32\x1f.google.protobuf.MessageOptionsR\x07options\x12U\n\x0ereserved_range\x18\t \x03(\x0b\x32..google.protobuf.DescriptorProto.ReservedRangeR\rreservedRange\x12#\n\rreserved_name\x18\n \x03(\tR\x0creservedName\x12\x41\n\nvisibility\x18\x0b \x01(\x0e\x32!.google.protobuf.SymbolVisibilityR\nvisibility\x1az\n\x0e\x45xtensionRange\x12\x14\n\x05start\x18\x01 \x01(\x05R\x05start\x12\x10\n\x03\x65nd\x18\x02 \x01(\x05R\x03\x65nd\x12@\n\x07options\x18\x03 \x01(\x0b\x32&.google.protobuf.ExtensionRangeOptionsR\x07options\x1a\x37\n\rReservedRange\x12\x14\n\x05start\x18\x01 \x01(\x05R\x05start\x12\x10\n\x03\x65nd\x18\x02 \x01(\x05R\x03\x65nd\"\xcc\x04\n\x15\x45xtensionRangeOptions\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption\x12Y\n\x0b\x64\x65\x63laration\x18\x02 \x03(\x0b\x32\x32.google.protobuf.ExtensionRangeOptions.DeclarationB\x03\x88\x01\x02R\x0b\x64\x65\x63laration\x12\x37\n\x08\x66\x65\x61tures\x18\x32 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12m\n\x0cverification\x18\x03 \x01(\x0e\x32\x38.google.protobuf.ExtensionRangeOptions.VerificationState:\nUNVERIFIEDB\x03\x88\x01\x02R\x0cverification\x1a\x94\x01\n\x0b\x44\x65\x63laration\x12\x16\n\x06number\x18\x01 \x01(\x05R\x06number\x12\x1b\n\tfull_name\x18\x02 \x01(\tR\x08\x66ullName\x12\x12\n\x04type\x18\x03 \x01(\tR\x04type\x12\x1a\n\x08reserved\x18\x05 \x01(\x08R\x08reserved\x12\x1a\n\x08repeated\x18\x06 \x01(\x08R\x08repeatedJ\x04\x08\x04\x10\x05\"4\n\x11VerificationState\x12\x0f\n\x0b\x44\x45\x43LARATION\x10\x00\x12\x0e\n\nUNVERIFIED\x10\x01*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\xc1\x06\n\x14\x46ieldDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x16\n\x06number\x18\x03 \x01(\x05R\x06number\x12\x41\n\x05label\x18\x04 \x01(\x0e\x32+.google.protobuf.FieldDescriptorProto.LabelR\x05label\x12>\n\x04type\x18\x05 \x01(\x0e\x32*.google.protobuf.FieldDescriptorProto.TypeR\x04type\x12\x1b\n\ttype_name\x18\x06 \x01(\tR\x08typeName\x12\x1a\n\x08\x65xtendee\x18\x02 \x01(\tR\x08\x65xtendee\x12#\n\rdefault_value\x18\x07 \x01(\tR\x0c\x64\x65\x66\x61ultValue\x12\x1f\n\x0boneof_index\x18\t \x01(\x05R\noneofIndex\x12\x1b\n\tjson_name\x18\n \x01(\tR\x08jsonName\x12\x37\n\x07options\x18\x08 \x01(\x0b\x32\x1d.google.protobuf.FieldOptionsR\x07options\x12\'\n\x0fproto3_optional\x18\x11 \x01(\x08R\x0eproto3Optional\"\xb6\x02\n\x04Type\x12\x0f\n\x0bTYPE_DOUBLE\x10\x01\x12\x0e\n\nTYPE_FLOAT\x10\x02\x12\x0e\n\nTYPE_INT64\x10\x03\x12\x0f\n\x0bTYPE_UINT64\x10\x04\x12\x0e\n\nTYPE_INT32\x10\x05\x12\x10\n\x0cTYPE_FIXED64\x10\x06\x12\x10\n\x0cTYPE_FIXED32\x10\x07\x12\r\n\tTYPE_BOOL\x10\x08\x12\x0f\n\x0bTYPE_STRING\x10\t\x12\x0e\n\nTYPE_GROUP\x10\n\x12\x10\n\x0cTYPE_MESSAGE\x10\x0b\x12\x0e\n\nTYPE_BYTES\x10\x0c\x12\x0f\n\x0bTYPE_UINT32\x10\r\x12\r\n\tTYPE_ENUM\x10\x0e\x12\x11\n\rTYPE_SFIXED32\x10\x0f\x12\x11\n\rTYPE_SFIXED64\x10\x10\x12\x0f\n\x0bTYPE_SINT32\x10\x11\x12\x0f\n\x0bTYPE_SINT64\x10\x12\"C\n\x05Label\x12\x12\n\x0eLABEL_OPTIONAL\x10\x01\x12\x12\n\x0eLABEL_REPEATED\x10\x03\x12\x12\n\x0eLABEL_REQUIRED\x10\x02\"c\n\x14OneofDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x37\n\x07options\x18\x02 \x01(\x0b\x32\x1d.google.protobuf.OneofOptionsR\x07options\"\xa6\x03\n\x13\x45numDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12?\n\x05value\x18\x02 \x03(\x0b\x32).google.protobuf.EnumValueDescriptorProtoR\x05value\x12\x36\n\x07options\x18\x03 \x01(\x0b\x32\x1c.google.protobuf.EnumOptionsR\x07options\x12]\n\x0ereserved_range\x18\x04 \x03(\x0b\x32\x36.google.protobuf.EnumDescriptorProto.EnumReservedRangeR\rreservedRange\x12#\n\rreserved_name\x18\x05 \x03(\tR\x0creservedName\x12\x41\n\nvisibility\x18\x06 \x01(\x0e\x32!.google.protobuf.SymbolVisibilityR\nvisibility\x1a;\n\x11\x45numReservedRange\x12\x14\n\x05start\x18\x01 \x01(\x05R\x05start\x12\x10\n\x03\x65nd\x18\x02 \x01(\x05R\x03\x65nd\"\x83\x01\n\x18\x45numValueDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x16\n\x06number\x18\x02 \x01(\x05R\x06number\x12;\n\x07options\x18\x03 \x01(\x0b\x32!.google.protobuf.EnumValueOptionsR\x07options\"\xa7\x01\n\x16ServiceDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12>\n\x06method\x18\x02 \x03(\x0b\x32&.google.protobuf.MethodDescriptorProtoR\x06method\x12\x39\n\x07options\x18\x03 \x01(\x0b\x32\x1f.google.protobuf.ServiceOptionsR\x07options\"\x89\x02\n\x15MethodDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x1d\n\ninput_type\x18\x02 \x01(\tR\tinputType\x12\x1f\n\x0boutput_type\x18\x03 \x01(\tR\noutputType\x12\x38\n\x07options\x18\x04 \x01(\x0b\x32\x1e.google.protobuf.MethodOptionsR\x07options\x12\x30\n\x10\x63lient_streaming\x18\x05 \x01(\x08:\x05\x66\x61lseR\x0f\x63lientStreaming\x12\x30\n\x10server_streaming\x18\x06 \x01(\x08:\x05\x66\x61lseR\x0fserverStreaming\"\xad\t\n\x0b\x46ileOptions\x12!\n\x0cjava_package\x18\x01 \x01(\tR\x0bjavaPackage\x12\x30\n\x14java_outer_classname\x18\x08 \x01(\tR\x12javaOuterClassname\x12\x35\n\x13java_multiple_files\x18\n \x01(\x08:\x05\x66\x61lseR\x11javaMultipleFiles\x12\x44\n\x1djava_generate_equals_and_hash\x18\x14 \x01(\x08\x42\x02\x18\x01R\x19javaGenerateEqualsAndHash\x12:\n\x16java_string_check_utf8\x18\x1b \x01(\x08:\x05\x66\x61lseR\x13javaStringCheckUtf8\x12S\n\x0coptimize_for\x18\t \x01(\x0e\x32).google.protobuf.FileOptions.OptimizeMode:\x05SPEEDR\x0boptimizeFor\x12\x1d\n\ngo_package\x18\x0b \x01(\tR\tgoPackage\x12\x35\n\x13\x63\x63_generic_services\x18\x10 \x01(\x08:\x05\x66\x61lseR\x11\x63\x63GenericServices\x12\x39\n\x15java_generic_services\x18\x11 \x01(\x08:\x05\x66\x61lseR\x13javaGenericServices\x12\x35\n\x13py_generic_services\x18\x12 \x01(\x08:\x05\x66\x61lseR\x11pyGenericServices\x12%\n\ndeprecated\x18\x17 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12.\n\x10\x63\x63_enable_arenas\x18\x1f \x01(\x08:\x04trueR\x0e\x63\x63\x45nableArenas\x12*\n\x11objc_class_prefix\x18$ \x01(\tR\x0fobjcClassPrefix\x12)\n\x10\x63sharp_namespace\x18% \x01(\tR\x0f\x63sharpNamespace\x12!\n\x0cswift_prefix\x18\' \x01(\tR\x0bswiftPrefix\x12(\n\x10php_class_prefix\x18( \x01(\tR\x0ephpClassPrefix\x12#\n\rphp_namespace\x18) \x01(\tR\x0cphpNamespace\x12\x34\n\x16php_metadata_namespace\x18, \x01(\tR\x14phpMetadataNamespace\x12!\n\x0cruby_package\x18- \x01(\tR\x0brubyPackage\x12\x37\n\x08\x66\x65\x61tures\x18\x32 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption\":\n\x0cOptimizeMode\x12\t\n\x05SPEED\x10\x01\x12\r\n\tCODE_SIZE\x10\x02\x12\x10\n\x0cLITE_RUNTIME\x10\x03*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02J\x04\x08*\x10+J\x04\x08&\x10\'R\x14php_generic_services\"\xf4\x03\n\x0eMessageOptions\x12<\n\x17message_set_wire_format\x18\x01 \x01(\x08:\x05\x66\x61lseR\x14messageSetWireFormat\x12L\n\x1fno_standard_descriptor_accessor\x18\x02 \x01(\x08:\x05\x66\x61lseR\x1cnoStandardDescriptorAccessor\x12%\n\ndeprecated\x18\x03 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12\x1b\n\tmap_entry\x18\x07 \x01(\x08R\x08mapEntry\x12V\n&deprecated_legacy_json_field_conflicts\x18\x0b \x01(\x08\x42\x02\x18\x01R\"deprecatedLegacyJsonFieldConflicts\x12\x37\n\x08\x66\x65\x61tures\x18\x0c \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02J\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x06\x10\x07J\x04\x08\x08\x10\tJ\x04\x08\t\x10\n\"\x9d\r\n\x0c\x46ieldOptions\x12\x41\n\x05\x63type\x18\x01 \x01(\x0e\x32#.google.protobuf.FieldOptions.CType:\x06STRINGR\x05\x63type\x12\x16\n\x06packed\x18\x02 \x01(\x08R\x06packed\x12G\n\x06jstype\x18\x06 \x01(\x0e\x32$.google.protobuf.FieldOptions.JSType:\tJS_NORMALR\x06jstype\x12\x19\n\x04lazy\x18\x05 \x01(\x08:\x05\x66\x61lseR\x04lazy\x12.\n\x0funverified_lazy\x18\x0f \x01(\x08:\x05\x66\x61lseR\x0eunverifiedLazy\x12%\n\ndeprecated\x18\x03 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12\x19\n\x04weak\x18\n \x01(\x08:\x05\x66\x61lseR\x04weak\x12(\n\x0c\x64\x65\x62ug_redact\x18\x10 \x01(\x08:\x05\x66\x61lseR\x0b\x64\x65\x62ugRedact\x12K\n\tretention\x18\x11 \x01(\x0e\x32-.google.protobuf.FieldOptions.OptionRetentionR\tretention\x12H\n\x07targets\x18\x13 \x03(\x0e\x32..google.protobuf.FieldOptions.OptionTargetTypeR\x07targets\x12W\n\x10\x65\x64ition_defaults\x18\x14 \x03(\x0b\x32,.google.protobuf.FieldOptions.EditionDefaultR\x0f\x65\x64itionDefaults\x12\x37\n\x08\x66\x65\x61tures\x18\x15 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12U\n\x0f\x66\x65\x61ture_support\x18\x16 \x01(\x0b\x32,.google.protobuf.FieldOptions.FeatureSupportR\x0e\x66\x65\x61tureSupport\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption\x1aZ\n\x0e\x45\x64itionDefault\x12\x32\n\x07\x65\x64ition\x18\x03 \x01(\x0e\x32\x18.google.protobuf.EditionR\x07\x65\x64ition\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value\x1a\x96\x02\n\x0e\x46\x65\x61tureSupport\x12G\n\x12\x65\x64ition_introduced\x18\x01 \x01(\x0e\x32\x18.google.protobuf.EditionR\x11\x65\x64itionIntroduced\x12G\n\x12\x65\x64ition_deprecated\x18\x02 \x01(\x0e\x32\x18.google.protobuf.EditionR\x11\x65\x64itionDeprecated\x12/\n\x13\x64\x65precation_warning\x18\x03 \x01(\tR\x12\x64\x65precationWarning\x12\x41\n\x0f\x65\x64ition_removed\x18\x04 \x01(\x0e\x32\x18.google.protobuf.EditionR\x0e\x65\x64itionRemoved\"/\n\x05\x43Type\x12\n\n\x06STRING\x10\x00\x12\x08\n\x04\x43ORD\x10\x01\x12\x10\n\x0cSTRING_PIECE\x10\x02\"5\n\x06JSType\x12\r\n\tJS_NORMAL\x10\x00\x12\r\n\tJS_STRING\x10\x01\x12\r\n\tJS_NUMBER\x10\x02\"U\n\x0fOptionRetention\x12\x15\n\x11RETENTION_UNKNOWN\x10\x00\x12\x15\n\x11RETENTION_RUNTIME\x10\x01\x12\x14\n\x10RETENTION_SOURCE\x10\x02\"\x8c\x02\n\x10OptionTargetType\x12\x17\n\x13TARGET_TYPE_UNKNOWN\x10\x00\x12\x14\n\x10TARGET_TYPE_FILE\x10\x01\x12\x1f\n\x1bTARGET_TYPE_EXTENSION_RANGE\x10\x02\x12\x17\n\x13TARGET_TYPE_MESSAGE\x10\x03\x12\x15\n\x11TARGET_TYPE_FIELD\x10\x04\x12\x15\n\x11TARGET_TYPE_ONEOF\x10\x05\x12\x14\n\x10TARGET_TYPE_ENUM\x10\x06\x12\x1a\n\x16TARGET_TYPE_ENUM_ENTRY\x10\x07\x12\x17\n\x13TARGET_TYPE_SERVICE\x10\x08\x12\x16\n\x12TARGET_TYPE_METHOD\x10\t*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02J\x04\x08\x04\x10\x05J\x04\x08\x12\x10\x13\"\xac\x01\n\x0cOneofOptions\x12\x37\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\xd1\x02\n\x0b\x45numOptions\x12\x1f\n\x0b\x61llow_alias\x18\x02 \x01(\x08R\nallowAlias\x12%\n\ndeprecated\x18\x03 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12V\n&deprecated_legacy_json_field_conflicts\x18\x06 \x01(\x08\x42\x02\x18\x01R\"deprecatedLegacyJsonFieldConflicts\x12\x37\n\x08\x66\x65\x61tures\x18\x07 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02J\x04\x08\x05\x10\x06\"\xd8\x02\n\x10\x45numValueOptions\x12%\n\ndeprecated\x18\x01 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12\x37\n\x08\x66\x65\x61tures\x18\x02 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12(\n\x0c\x64\x65\x62ug_redact\x18\x03 \x01(\x08:\x05\x66\x61lseR\x0b\x64\x65\x62ugRedact\x12U\n\x0f\x66\x65\x61ture_support\x18\x04 \x01(\x0b\x32,.google.protobuf.FieldOptions.FeatureSupportR\x0e\x66\x65\x61tureSupport\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\xd5\x01\n\x0eServiceOptions\x12\x37\n\x08\x66\x65\x61tures\x18\" \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12%\n\ndeprecated\x18! \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\x99\x03\n\rMethodOptions\x12%\n\ndeprecated\x18! \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12q\n\x11idempotency_level\x18\" \x01(\x0e\x32/.google.protobuf.MethodOptions.IdempotencyLevel:\x13IDEMPOTENCY_UNKNOWNR\x10idempotencyLevel\x12\x37\n\x08\x66\x65\x61tures\x18# \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption\"P\n\x10IdempotencyLevel\x12\x17\n\x13IDEMPOTENCY_UNKNOWN\x10\x00\x12\x13\n\x0fNO_SIDE_EFFECTS\x10\x01\x12\x0e\n\nIDEMPOTENT\x10\x02*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\x9a\x03\n\x13UninterpretedOption\x12\x41\n\x04name\x18\x02 \x03(\x0b\x32-.google.protobuf.UninterpretedOption.NamePartR\x04name\x12)\n\x10identifier_value\x18\x03 \x01(\tR\x0fidentifierValue\x12,\n\x12positive_int_value\x18\x04 \x01(\x04R\x10positiveIntValue\x12,\n\x12negative_int_value\x18\x05 \x01(\x03R\x10negativeIntValue\x12!\n\x0c\x64ouble_value\x18\x06 \x01(\x01R\x0b\x64oubleValue\x12!\n\x0cstring_value\x18\x07 \x01(\x0cR\x0bstringValue\x12\'\n\x0f\x61ggregate_value\x18\x08 \x01(\tR\x0e\x61ggregateValue\x1aJ\n\x08NamePart\x12\x1b\n\tname_part\x18\x01 \x02(\tR\x08namePart\x12!\n\x0cis_extension\x18\x02 \x02(\x08R\x0bisExtension\"\x8e\x0f\n\nFeatureSet\x12\x91\x01\n\x0e\x66ield_presence\x18\x01 \x01(\x0e\x32).google.protobuf.FeatureSet.FieldPresenceB?\x88\x01\x01\x98\x01\x04\x98\x01\x01\xa2\x01\r\x12\x08\x45XPLICIT\x18\x84\x07\xa2\x01\r\x12\x08IMPLICIT\x18\xe7\x07\xa2\x01\r\x12\x08\x45XPLICIT\x18\xe8\x07\xb2\x01\x03\x08\xe8\x07R\rfieldPresence\x12l\n\tenum_type\x18\x02 \x01(\x0e\x32$.google.protobuf.FeatureSet.EnumTypeB)\x88\x01\x01\x98\x01\x06\x98\x01\x01\xa2\x01\x0b\x12\x06\x43LOSED\x18\x84\x07\xa2\x01\t\x12\x04OPEN\x18\xe7\x07\xb2\x01\x03\x08\xe8\x07R\x08\x65numType\x12\x98\x01\n\x17repeated_field_encoding\x18\x03 \x01(\x0e\x32\x31.google.protobuf.FeatureSet.RepeatedFieldEncodingB-\x88\x01\x01\x98\x01\x04\x98\x01\x01\xa2\x01\r\x12\x08\x45XPANDED\x18\x84\x07\xa2\x01\x0b\x12\x06PACKED\x18\xe7\x07\xb2\x01\x03\x08\xe8\x07R\x15repeatedFieldEncoding\x12~\n\x0futf8_validation\x18\x04 \x01(\x0e\x32*.google.protobuf.FeatureSet.Utf8ValidationB)\x88\x01\x01\x98\x01\x04\x98\x01\x01\xa2\x01\t\x12\x04NONE\x18\x84\x07\xa2\x01\x0b\x12\x06VERIFY\x18\xe7\x07\xb2\x01\x03\x08\xe8\x07R\x0eutf8Validation\x12~\n\x10message_encoding\x18\x05 \x01(\x0e\x32+.google.protobuf.FeatureSet.MessageEncodingB&\x88\x01\x01\x98\x01\x04\x98\x01\x01\xa2\x01\x14\x12\x0fLENGTH_PREFIXED\x18\x84\x07\xb2\x01\x03\x08\xe8\x07R\x0fmessageEncoding\x12\x82\x01\n\x0bjson_format\x18\x06 \x01(\x0e\x32&.google.protobuf.FeatureSet.JsonFormatB9\x88\x01\x01\x98\x01\x03\x98\x01\x06\x98\x01\x01\xa2\x01\x17\x12\x12LEGACY_BEST_EFFORT\x18\x84\x07\xa2\x01\n\x12\x05\x41LLOW\x18\xe7\x07\xb2\x01\x03\x08\xe8\x07R\njsonFormat\x12\xab\x01\n\x14\x65nforce_naming_style\x18\x07 \x01(\x0e\x32..google.protobuf.FeatureSet.EnforceNamingStyleBI\x88\x01\x02\x98\x01\x01\x98\x01\x02\x98\x01\x03\x98\x01\x04\x98\x01\x05\x98\x01\x06\x98\x01\x07\x98\x01\x08\x98\x01\t\xa2\x01\x11\x12\x0cSTYLE_LEGACY\x18\x84\x07\xa2\x01\x0e\x12\tSTYLE2024\x18\xe9\x07\xb2\x01\x03\x08\xe9\x07R\x12\x65nforceNamingStyle\x12\xb9\x01\n\x19\x64\x65\x66\x61ult_symbol_visibility\x18\x08 \x01(\x0e\x32\x45.google.protobuf.FeatureSet.VisibilityFeature.DefaultSymbolVisibilityB6\x88\x01\x02\x98\x01\x01\xa2\x01\x0f\x12\nEXPORT_ALL\x18\x84\x07\xa2\x01\x15\x12\x10\x45XPORT_TOP_LEVEL\x18\xe9\x07\xb2\x01\x03\x08\xe9\x07R\x17\x64\x65\x66\x61ultSymbolVisibility\x1a\xa1\x01\n\x11VisibilityFeature\"\x81\x01\n\x17\x44\x65\x66\x61ultSymbolVisibility\x12%\n!DEFAULT_SYMBOL_VISIBILITY_UNKNOWN\x10\x00\x12\x0e\n\nEXPORT_ALL\x10\x01\x12\x14\n\x10\x45XPORT_TOP_LEVEL\x10\x02\x12\r\n\tLOCAL_ALL\x10\x03\x12\n\n\x06STRICT\x10\x04J\x08\x08\x01\x10\x80\x80\x80\x80\x02\"\\\n\rFieldPresence\x12\x1a\n\x16\x46IELD_PRESENCE_UNKNOWN\x10\x00\x12\x0c\n\x08\x45XPLICIT\x10\x01\x12\x0c\n\x08IMPLICIT\x10\x02\x12\x13\n\x0fLEGACY_REQUIRED\x10\x03\"7\n\x08\x45numType\x12\x15\n\x11\x45NUM_TYPE_UNKNOWN\x10\x00\x12\x08\n\x04OPEN\x10\x01\x12\n\n\x06\x43LOSED\x10\x02\"V\n\x15RepeatedFieldEncoding\x12#\n\x1fREPEATED_FIELD_ENCODING_UNKNOWN\x10\x00\x12\n\n\x06PACKED\x10\x01\x12\x0c\n\x08\x45XPANDED\x10\x02\"I\n\x0eUtf8Validation\x12\x1b\n\x17UTF8_VALIDATION_UNKNOWN\x10\x00\x12\n\n\x06VERIFY\x10\x02\x12\x08\n\x04NONE\x10\x03\"\x04\x08\x01\x10\x01\"S\n\x0fMessageEncoding\x12\x1c\n\x18MESSAGE_ENCODING_UNKNOWN\x10\x00\x12\x13\n\x0fLENGTH_PREFIXED\x10\x01\x12\r\n\tDELIMITED\x10\x02\"H\n\nJsonFormat\x12\x17\n\x13JSON_FORMAT_UNKNOWN\x10\x00\x12\t\n\x05\x41LLOW\x10\x01\x12\x16\n\x12LEGACY_BEST_EFFORT\x10\x02\"W\n\x12\x45nforceNamingStyle\x12 \n\x1c\x45NFORCE_NAMING_STYLE_UNKNOWN\x10\x00\x12\r\n\tSTYLE2024\x10\x01\x12\x10\n\x0cSTYLE_LEGACY\x10\x02*\x06\x08\xe8\x07\x10\x8bN*\x06\x08\x8bN\x10\x90N*\x06\x08\x90N\x10\x91NJ\x06\x08\xe7\x07\x10\xe8\x07\"\xef\x03\n\x12\x46\x65\x61tureSetDefaults\x12X\n\x08\x64\x65\x66\x61ults\x18\x01 \x03(\x0b\x32<.google.protobuf.FeatureSetDefaults.FeatureSetEditionDefaultR\x08\x64\x65\x66\x61ults\x12\x41\n\x0fminimum_edition\x18\x04 \x01(\x0e\x32\x18.google.protobuf.EditionR\x0eminimumEdition\x12\x41\n\x0fmaximum_edition\x18\x05 \x01(\x0e\x32\x18.google.protobuf.EditionR\x0emaximumEdition\x1a\xf8\x01\n\x18\x46\x65\x61tureSetEditionDefault\x12\x32\n\x07\x65\x64ition\x18\x03 \x01(\x0e\x32\x18.google.protobuf.EditionR\x07\x65\x64ition\x12N\n\x14overridable_features\x18\x04 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x13overridableFeatures\x12\x42\n\x0e\x66ixed_features\x18\x05 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\rfixedFeaturesJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03R\x08\x66\x65\x61tures\"\xb5\x02\n\x0eSourceCodeInfo\x12\x44\n\x08location\x18\x01 \x03(\x0b\x32(.google.protobuf.SourceCodeInfo.LocationR\x08location\x1a\xce\x01\n\x08Location\x12\x16\n\x04path\x18\x01 \x03(\x05\x42\x02\x10\x01R\x04path\x12\x16\n\x04span\x18\x02 \x03(\x05\x42\x02\x10\x01R\x04span\x12)\n\x10leading_comments\x18\x03 \x01(\tR\x0fleadingComments\x12+\n\x11trailing_comments\x18\x04 \x01(\tR\x10trailingComments\x12:\n\x19leading_detached_comments\x18\x06 \x03(\tR\x17leadingDetachedComments*\x0c\x08\x80\xec\xca\xff\x01\x10\x81\xec\xca\xff\x01\"\xd0\x02\n\x11GeneratedCodeInfo\x12M\n\nannotation\x18\x01 \x03(\x0b\x32-.google.protobuf.GeneratedCodeInfo.AnnotationR\nannotation\x1a\xeb\x01\n\nAnnotation\x12\x16\n\x04path\x18\x01 \x03(\x05\x42\x02\x10\x01R\x04path\x12\x1f\n\x0bsource_file\x18\x02 \x01(\tR\nsourceFile\x12\x14\n\x05\x62\x65gin\x18\x03 \x01(\x05R\x05\x62\x65gin\x12\x10\n\x03\x65nd\x18\x04 \x01(\x05R\x03\x65nd\x12R\n\x08semantic\x18\x05 \x01(\x0e\x32\x36.google.protobuf.GeneratedCodeInfo.Annotation.SemanticR\x08semantic\"(\n\x08Semantic\x12\x08\n\x04NONE\x10\x00\x12\x07\n\x03SET\x10\x01\x12\t\n\x05\x41LIAS\x10\x02*\xa7\x02\n\x07\x45\x64ition\x12\x13\n\x0f\x45\x44ITION_UNKNOWN\x10\x00\x12\x13\n\x0e\x45\x44ITION_LEGACY\x10\x84\x07\x12\x13\n\x0e\x45\x44ITION_PROTO2\x10\xe6\x07\x12\x13\n\x0e\x45\x44ITION_PROTO3\x10\xe7\x07\x12\x11\n\x0c\x45\x44ITION_2023\x10\xe8\x07\x12\x11\n\x0c\x45\x44ITION_2024\x10\xe9\x07\x12\x17\n\x13\x45\x44ITION_1_TEST_ONLY\x10\x01\x12\x17\n\x13\x45\x44ITION_2_TEST_ONLY\x10\x02\x12\x1d\n\x17\x45\x44ITION_99997_TEST_ONLY\x10\x9d\x8d\x06\x12\x1d\n\x17\x45\x44ITION_99998_TEST_ONLY\x10\x9e\x8d\x06\x12\x1d\n\x17\x45\x44ITION_99999_TEST_ONLY\x10\x9f\x8d\x06\x12\x13\n\x0b\x45\x44ITION_MAX\x10\xff\xff\xff\xff\x07*U\n\x10SymbolVisibility\x12\x14\n\x10VISIBILITY_UNSET\x10\x00\x12\x14\n\x10VISIBILITY_LOCAL\x10\x01\x12\x15\n\x11VISIBILITY_EXPORT\x10\x02\x42~\n\x13\x63om.google.protobufB\x10\x44\x65scriptorProtosH\x01Z-google.golang.org/protobuf/types/descriptorpb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1aGoogle.Protobuf.Reflection'
  )
else:
  DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n google/protobuf/descriptor.proto\x12\x0fgoogle.protobuf\"[\n\x11\x46ileDescriptorSet\x12\x38\n\x04\x66ile\x18\x01 \x03(\x0b\x32$.google.protobuf.FileDescriptorProtoR\x04\x66ile*\x0c\x08\x80\xec\xca\xff\x01\x10\x81\xec\xca\xff\x01\"\xc5\x05\n\x13\x46ileDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x18\n\x07package\x18\x02 \x01(\tR\x07package\x12\x1e\n\ndependency\x18\x03 \x03(\tR\ndependency\x12+\n\x11public_dependency\x18\n \x03(\x05R\x10publicDependency\x12\'\n\x0fweak_dependency\x18\x0b \x03(\x05R\x0eweakDependency\x12+\n\x11option_dependency\x18\x0f \x03(\tR\x10optionDependency\x12\x43\n\x0cmessage_type\x18\x04 \x03(\x0b\x32 .google.protobuf.DescriptorProtoR\x0bmessageType\x12\x41\n\tenum_type\x18\x05 \x03(\x0b\x32$.google.protobuf.EnumDescriptorProtoR\x08\x65numType\x12\x41\n\x07service\x18\x06 \x03(\x0b\x32\'.google.protobuf.ServiceDescriptorProtoR\x07service\x12\x43\n\textension\x18\x07 \x03(\x0b\x32%.google.protobuf.FieldDescriptorProtoR\textension\x12\x36\n\x07options\x18\x08 \x01(\x0b\x32\x1c.google.protobuf.FileOptionsR\x07options\x12I\n\x10source_code_info\x18\t \x01(\x0b\x32\x1f.google.protobuf.SourceCodeInfoR\x0esourceCodeInfo\x12\x16\n\x06syntax\x18\x0c \x01(\tR\x06syntax\x12\x32\n\x07\x65\x64ition\x18\x0e \x01(\x0e\x32\x18.google.protobuf.EditionR\x07\x65\x64ition\"\xfc\x06\n\x0f\x44\x65scriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12;\n\x05\x66ield\x18\x02 \x03(\x0b\x32%.google.protobuf.FieldDescriptorProtoR\x05\x66ield\x12\x43\n\textension\x18\x06 \x03(\x0b\x32%.google.protobuf.FieldDescriptorProtoR\textension\x12\x41\n\x0bnested_type\x18\x03 \x03(\x0b\x32 .google.protobuf.DescriptorProtoR\nnestedType\x12\x41\n\tenum_type\x18\x04 \x03(\x0b\x32$.google.protobuf.EnumDescriptorProtoR\x08\x65numType\x12X\n\x0f\x65xtension_range\x18\x05 \x03(\x0b\x32/.google.protobuf.DescriptorProto.ExtensionRangeR\x0e\x65xtensionRange\x12\x44\n\noneof_decl\x18\x08 \x03(\x0b\x32%.google.protobuf.OneofDescriptorProtoR\toneofDecl\x12\x39\n\x07options\x18\x07 \x01(\x0b\x32\x1f.google.protobuf.MessageOptionsR\x07options\x12U\n\x0ereserved_range\x18\t \x03(\x0b\x32..google.protobuf.DescriptorProto.ReservedRangeR\rreservedRange\x12#\n\rreserved_name\x18\n \x03(\tR\x0creservedName\x12\x41\n\nvisibility\x18\x0b \x01(\x0e\x32!.google.protobuf.SymbolVisibilityR\nvisibility\x1az\n\x0e\x45xtensionRange\x12\x14\n\x05start\x18\x01 \x01(\x05R\x05start\x12\x10\n\x03\x65nd\x18\x02 \x01(\x05R\x03\x65nd\x12@\n\x07options\x18\x03 \x01(\x0b\x32&.google.protobuf.ExtensionRangeOptionsR\x07options\x1a\x37\n\rReservedRange\x12\x14\n\x05start\x18\x01 \x01(\x05R\x05start\x12\x10\n\x03\x65nd\x18\x02 \x01(\x05R\x03\x65nd\"\xcc\x04\n\x15\x45xtensionRangeOptions\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption\x12Y\n\x0b\x64\x65\x63laration\x18\x02 \x03(\x0b\x32\x32.google.protobuf.ExtensionRangeOptions.DeclarationB\x03\x88\x01\x02R\x0b\x64\x65\x63laration\x12\x37\n\x08\x66\x65\x61tures\x18\x32 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12m\n\x0cverification\x18\x03 \x01(\x0e\x32\x38.google.protobuf.ExtensionRangeOptions.VerificationState:\nUNVERIFIEDB\x03\x88\x01\x02R\x0cverification\x1a\x94\x01\n\x0b\x44\x65\x63laration\x12\x16\n\x06number\x18\x01 \x01(\x05R\x06number\x12\x1b\n\tfull_name\x18\x02 \x01(\tR\x08\x66ullName\x12\x12\n\x04type\x18\x03 \x01(\tR\x04type\x12\x1a\n\x08reserved\x18\x05 \x01(\x08R\x08reserved\x12\x1a\n\x08repeated\x18\x06 \x01(\x08R\x08repeatedJ\x04\x08\x04\x10\x05\"4\n\x11VerificationState\x12\x0f\n\x0b\x44\x45\x43LARATION\x10\x00\x12\x0e\n\nUNVERIFIED\x10\x01*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\xc1\x06\n\x14\x46ieldDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x16\n\x06number\x18\x03 \x01(\x05R\x06number\x12\x41\n\x05label\x18\x04 \x01(\x0e\x32+.google.protobuf.FieldDescriptorProto.LabelR\x05label\x12>\n\x04type\x18\x05 \x01(\x0e\x32*.google.protobuf.FieldDescriptorProto.TypeR\x04type\x12\x1b\n\ttype_name\x18\x06 \x01(\tR\x08typeName\x12\x1a\n\x08\x65xtendee\x18\x02 \x01(\tR\x08\x65xtendee\x12#\n\rdefault_value\x18\x07 \x01(\tR\x0c\x64\x65\x66\x61ultValue\x12\x1f\n\x0boneof_index\x18\t \x01(\x05R\noneofIndex\x12\x1b\n\tjson_name\x18\n \x01(\tR\x08jsonName\x12\x37\n\x07options\x18\x08 \x01(\x0b\x32\x1d.google.protobuf.FieldOptionsR\x07options\x12\'\n\x0fproto3_optional\x18\x11 \x01(\x08R\x0eproto3Optional\"\xb6\x02\n\x04Type\x12\x0f\n\x0bTYPE_DOUBLE\x10\x01\x12\x0e\n\nTYPE_FLOAT\x10\x02\x12\x0e\n\nTYPE_INT64\x10\x03\x12\x0f\n\x0bTYPE_UINT64\x10\x04\x12\x0e\n\nTYPE_INT32\x10\x05\x12\x10\n\x0cTYPE_FIXED64\x10\x06\x12\x10\n\x0cTYPE_FIXED32\x10\x07\x12\r\n\tTYPE_BOOL\x10\x08\x12\x0f\n\x0bTYPE_STRING\x10\t\x12\x0e\n\nTYPE_GROUP\x10\n\x12\x10\n\x0cTYPE_MESSAGE\x10\x0b\x12\x0e\n\nTYPE_BYTES\x10\x0c\x12\x0f\n\x0bTYPE_UINT32\x10\r\x12\r\n\tTYPE_ENUM\x10\x0e\x12\x11\n\rTYPE_SFIXED32\x10\x0f\x12\x11\n\rTYPE_SFIXED64\x10\x10\x12\x0f\n\x0bTYPE_SINT32\x10\x11\x12\x0f\n\x0bTYPE_SINT64\x10\x12\"C\n\x05Label\x12\x12\n\x0eLABEL_OPTIONAL\x10\x01\x12\x12\n\x0eLABEL_REPEATED\x10\x03\x12\x12\n\x0eLABEL_REQUIRED\x10\x02\"c\n\x14OneofDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x37\n\x07options\x18\x02 \x01(\x0b\x32\x1d.google.protobuf.OneofOptionsR\x07options\"\xa6\x03\n\x13\x45numDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12?\n\x05value\x18\x02 \x03(\x0b\x32).google.protobuf.EnumValueDescriptorProtoR\x05value\x12\x36\n\x07options\x18\x03 \x01(\x0b\x32\x1c.google.protobuf.EnumOptionsR\x07options\x12]\n\x0ereserved_range\x18\x04 \x03(\x0b\x32\x36.google.protobuf.EnumDescriptorProto.EnumReservedRangeR\rreservedRange\x12#\n\rreserved_name\x18\x05 \x03(\tR\x0creservedName\x12\x41\n\nvisibility\x18\x06 \x01(\x0e\x32!.google.protobuf.SymbolVisibilityR\nvisibility\x1a;\n\x11\x45numReservedRange\x12\x14\n\x05start\x18\x01 \x01(\x05R\x05start\x12\x10\n\x03\x65nd\x18\x02 \x01(\x05R\x03\x65nd\"\x83\x01\n\x18\x45numValueDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x16\n\x06number\x18\x02 \x01(\x05R\x06number\x12;\n\x07options\x18\x03 \x01(\x0b\x32!.google.protobuf.EnumValueOptionsR\x07options\"\xa7\x01\n\x16ServiceDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12>\n\x06method\x18\x02 \x03(\x0b\x32&.google.protobuf.MethodDescriptorProtoR\x06method\x12\x39\n\x07options\x18\x03 \x01(\x0b\x32\x1f.google.protobuf.ServiceOptionsR\x07options\"\x89\x02\n\x15MethodDescriptorProto\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x1d\n\ninput_type\x18\x02 \x01(\tR\tinputType\x12\x1f\n\x0boutput_type\x18\x03 \x01(\tR\noutputType\x12\x38\n\x07options\x18\x04 \x01(\x0b\x32\x1e.google.protobuf.MethodOptionsR\x07options\x12\x30\n\x10\x63lient_streaming\x18\x05 \x01(\x08:\x05\x66\x61lseR\x0f\x63lientStreaming\x12\x30\n\x10server_streaming\x18\x06 \x01(\x08:\x05\x66\x61lseR\x0fserverStreaming\"\xad\t\n\x0b\x46ileOptions\x12!\n\x0cjava_package\x18\x01 \x01(\tR\x0bjavaPackage\x12\x30\n\x14java_outer_classname\x18\x08 \x01(\tR\x12javaOuterClassname\x12\x35\n\x13java_multiple_files\x18\n \x01(\x08:\x05\x66\x61lseR\x11javaMultipleFiles\x12\x44\n\x1djava_generate_equals_and_hash\x18\x14 \x01(\x08\x42\x02\x18\x01R\x19javaGenerateEqualsAndHash\x12:\n\x16java_string_check_utf8\x18\x1b \x01(\x08:\x05\x66\x61lseR\x13javaStringCheckUtf8\x12S\n\x0coptimize_for\x18\t \x01(\x0e\x32).google.protobuf.FileOptions.OptimizeMode:\x05SPEEDR\x0boptimizeFor\x12\x1d\n\ngo_package\x18\x0b \x01(\tR\tgoPackage\x12\x35\n\x13\x63\x63_generic_services\x18\x10 \x01(\x08:\x05\x66\x61lseR\x11\x63\x63GenericServices\x12\x39\n\x15java_generic_services\x18\x11 \x01(\x08:\x05\x66\x61lseR\x13javaGenericServices\x12\x35\n\x13py_generic_services\x18\x12 \x01(\x08:\x05\x66\x61lseR\x11pyGenericServices\x12%\n\ndeprecated\x18\x17 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12.\n\x10\x63\x63_enable_arenas\x18\x1f \x01(\x08:\x04trueR\x0e\x63\x63\x45nableArenas\x12*\n\x11objc_class_prefix\x18$ \x01(\tR\x0fobjcClassPrefix\x12)\n\x10\x63sharp_namespace\x18% \x01(\tR\x0f\x63sharpNamespace\x12!\n\x0cswift_prefix\x18\' \x01(\tR\x0bswiftPrefix\x12(\n\x10php_class_prefix\x18( \x01(\tR\x0ephpClassPrefix\x12#\n\rphp_namespace\x18) \x01(\tR\x0cphpNamespace\x12\x34\n\x16php_metadata_namespace\x18, \x01(\tR\x14phpMetadataNamespace\x12!\n\x0cruby_package\x18- \x01(\tR\x0brubyPackage\x12\x37\n\x08\x66\x65\x61tures\x18\x32 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption\":\n\x0cOptimizeMode\x12\t\n\x05SPEED\x10\x01\x12\r\n\tCODE_SIZE\x10\x02\x12\x10\n\x0cLITE_RUNTIME\x10\x03*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02J\x04\x08*\x10+J\x04\x08&\x10\'R\x14php_generic_services\"\xf4\x03\n\x0eMessageOptions\x12<\n\x17message_set_wire_format\x18\x01 \x01(\x08:\x05\x66\x61lseR\x14messageSetWireFormat\x12L\n\x1fno_standard_descriptor_accessor\x18\x02 \x01(\x08:\x05\x66\x61lseR\x1cnoStandardDescriptorAccessor\x12%\n\ndeprecated\x18\x03 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12\x1b\n\tmap_entry\x18\x07 \x01(\x08R\x08mapEntry\x12V\n&deprecated_legacy_json_field_conflicts\x18\x0b \x01(\x08\x42\x02\x18\x01R\"deprecatedLegacyJsonFieldConflicts\x12\x37\n\x08\x66\x65\x61tures\x18\x0c \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02J\x04\x08\x04\x10\x05J\x04\x08\x05\x10\x06J\x04\x08\x06\x10\x07J\x04\x08\x08\x10\tJ\x04\x08\t\x10\n\"\x9d\r\n\x0c\x46ieldOptions\x12\x41\n\x05\x63type\x18\x01 \x01(\x0e\x32#.google.protobuf.FieldOptions.CType:\x06STRINGR\x05\x63type\x12\x16\n\x06packed\x18\x02 \x01(\x08R\x06packed\x12G\n\x06jstype\x18\x06 \x01(\x0e\x32$.google.protobuf.FieldOptions.JSType:\tJS_NORMALR\x06jstype\x12\x19\n\x04lazy\x18\x05 \x01(\x08:\x05\x66\x61lseR\x04lazy\x12.\n\x0funverified_lazy\x18\x0f \x01(\x08:\x05\x66\x61lseR\x0eunverifiedLazy\x12%\n\ndeprecated\x18\x03 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12\x19\n\x04weak\x18\n \x01(\x08:\x05\x66\x61lseR\x04weak\x12(\n\x0c\x64\x65\x62ug_redact\x18\x10 \x01(\x08:\x05\x66\x61lseR\x0b\x64\x65\x62ugRedact\x12K\n\tretention\x18\x11 \x01(\x0e\x32-.google.protobuf.FieldOptions.OptionRetentionR\tretention\x12H\n\x07targets\x18\x13 \x03(\x0e\x32..google.protobuf.FieldOptions.OptionTargetTypeR\x07targets\x12W\n\x10\x65\x64ition_defaults\x18\x14 \x03(\x0b\x32,.google.protobuf.FieldOptions.EditionDefaultR\x0f\x65\x64itionDefaults\x12\x37\n\x08\x66\x65\x61tures\x18\x15 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12U\n\x0f\x66\x65\x61ture_support\x18\x16 \x01(\x0b\x32,.google.protobuf.FieldOptions.FeatureSupportR\x0e\x66\x65\x61tureSupport\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption\x1aZ\n\x0e\x45\x64itionDefault\x12\x32\n\x07\x65\x64ition\x18\x03 \x01(\x0e\x32\x18.google.protobuf.EditionR\x07\x65\x64ition\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value\x1a\x96\x02\n\x0e\x46\x65\x61tureSupport\x12G\n\x12\x65\x64ition_introduced\x18\x01 \x01(\x0e\x32\x18.google.protobuf.EditionR\x11\x65\x64itionIntroduced\x12G\n\x12\x65\x64ition_deprecated\x18\x02 \x01(\x0e\x32\x18.google.protobuf.EditionR\x11\x65\x64itionDeprecated\x12/\n\x13\x64\x65precation_warning\x18\x03 \x01(\tR\x12\x64\x65precationWarning\x12\x41\n\x0f\x65\x64ition_removed\x18\x04 \x01(\x0e\x32\x18.google.protobuf.EditionR\x0e\x65\x64itionRemoved\"/\n\x05\x43Type\x12\n\n\x06STRING\x10\x00\x12\x08\n\x04\x43ORD\x10\x01\x12\x10\n\x0cSTRING_PIECE\x10\x02\"5\n\x06JSType\x12\r\n\tJS_NORMAL\x10\x00\x12\r\n\tJS_STRING\x10\x01\x12\r\n\tJS_NUMBER\x10\x02\"U\n\x0fOptionRetention\x12\x15\n\x11RETENTION_UNKNOWN\x10\x00\x12\x15\n\x11RETENTION_RUNTIME\x10\x01\x12\x14\n\x10RETENTION_SOURCE\x10\x02\"\x8c\x02\n\x10OptionTargetType\x12\x17\n\x13TARGET_TYPE_UNKNOWN\x10\x00\x12\x14\n\x10TARGET_TYPE_FILE\x10\x01\x12\x1f\n\x1bTARGET_TYPE_EXTENSION_RANGE\x10\x02\x12\x17\n\x13TARGET_TYPE_MESSAGE\x10\x03\x12\x15\n\x11TARGET_TYPE_FIELD\x10\x04\x12\x15\n\x11TARGET_TYPE_ONEOF\x10\x05\x12\x14\n\x10TARGET_TYPE_ENUM\x10\x06\x12\x1a\n\x16TARGET_TYPE_ENUM_ENTRY\x10\x07\x12\x17\n\x13TARGET_TYPE_SERVICE\x10\x08\x12\x16\n\x12TARGET_TYPE_METHOD\x10\t*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02J\x04\x08\x04\x10\x05J\x04\x08\x12\x10\x13\"\xac\x01\n\x0cOneofOptions\x12\x37\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\xd1\x02\n\x0b\x45numOptions\x12\x1f\n\x0b\x61llow_alias\x18\x02 \x01(\x08R\nallowAlias\x12%\n\ndeprecated\x18\x03 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12V\n&deprecated_legacy_json_field_conflicts\x18\x06 \x01(\x08\x42\x02\x18\x01R\"deprecatedLegacyJsonFieldConflicts\x12\x37\n\x08\x66\x65\x61tures\x18\x07 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02J\x04\x08\x05\x10\x06\"\xd8\x02\n\x10\x45numValueOptions\x12%\n\ndeprecated\x18\x01 \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12\x37\n\x08\x66\x65\x61tures\x18\x02 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12(\n\x0c\x64\x65\x62ug_redact\x18\x03 \x01(\x08:\x05\x66\x61lseR\x0b\x64\x65\x62ugRedact\x12U\n\x0f\x66\x65\x61ture_support\x18\x04 \x01(\x0b\x32,.google.protobuf.FieldOptions.FeatureSupportR\x0e\x66\x65\x61tureSupport\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\xd5\x01\n\x0eServiceOptions\x12\x37\n\x08\x66\x65\x61tures\x18\" \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12%\n\ndeprecated\x18! \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\x99\x03\n\rMethodOptions\x12%\n\ndeprecated\x18! \x01(\x08:\x05\x66\x61lseR\ndeprecated\x12q\n\x11idempotency_level\x18\" \x01(\x0e\x32/.google.protobuf.MethodOptions.IdempotencyLevel:\x13IDEMPOTENCY_UNKNOWNR\x10idempotencyLevel\x12\x37\n\x08\x66\x65\x61tures\x18# \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x08\x66\x65\x61tures\x12X\n\x14uninterpreted_option\x18\xe7\x07 \x03(\x0b\x32$.google.protobuf.UninterpretedOptionR\x13uninterpretedOption\"P\n\x10IdempotencyLevel\x12\x17\n\x13IDEMPOTENCY_UNKNOWN\x10\x00\x12\x13\n\x0fNO_SIDE_EFFECTS\x10\x01\x12\x0e\n\nIDEMPOTENT\x10\x02*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\x9a\x03\n\x13UninterpretedOption\x12\x41\n\x04name\x18\x02 \x03(\x0b\x32-.google.protobuf.UninterpretedOption.NamePartR\x04name\x12)\n\x10identifier_value\x18\x03 \x01(\tR\x0fidentifierValue\x12,\n\x12positive_int_value\x18\x04 \x01(\x04R\x10positiveIntValue\x12,\n\x12negative_int_value\x18\x05 \x01(\x03R\x10negativeIntValue\x12!\n\x0c\x64ouble_value\x18\x06 \x01(\x01R\x0b\x64oubleValue\x12!\n\x0cstring_value\x18\x07 \x01(\x0cR\x0bstringValue\x12\'\n\x0f\x61ggregate_value\x18\x08 \x01(\tR\x0e\x61ggregateValue\x1aJ\n\x08NamePart\x12\x1b\n\tname_part\x18\x01 \x02(\tR\x08namePart\x12!\n\x0cis_extension\x18\x02 \x02(\x08R\x0bisExtension\"\x8e\x0f\n\nFeatureSet\x12\x91\x01\n\x0e\x66ield_presence\x18\x01 \x01(\x0e\x32).google.protobuf.FeatureSet.FieldPresenceB?\x88\x01\x01\x98\x01\x04\x98\x01\x01\xa2\x01\r\x12\x08\x45XPLICIT\x18\x84\x07\xa2\x01\r\x12\x08IMPLICIT\x18\xe7\x07\xa2\x01\r\x12\x08\x45XPLICIT\x18\xe8\x07\xb2\x01\x03\x08\xe8\x07R\rfieldPresence\x12l\n\tenum_type\x18\x02 \x01(\x0e\x32$.google.protobuf.FeatureSet.EnumTypeB)\x88\x01\x01\x98\x01\x06\x98\x01\x01\xa2\x01\x0b\x12\x06\x43LOSED\x18\x84\x07\xa2\x01\t\x12\x04OPEN\x18\xe7\x07\xb2\x01\x03\x08\xe8\x07R\x08\x65numType\x12\x98\x01\n\x17repeated_field_encoding\x18\x03 \x01(\x0e\x32\x31.google.protobuf.FeatureSet.RepeatedFieldEncodingB-\x88\x01\x01\x98\x01\x04\x98\x01\x01\xa2\x01\r\x12\x08\x45XPANDED\x18\x84\x07\xa2\x01\x0b\x12\x06PACKED\x18\xe7\x07\xb2\x01\x03\x08\xe8\x07R\x15repeatedFieldEncoding\x12~\n\x0futf8_validation\x18\x04 \x01(\x0e\x32*.google.protobuf.FeatureSet.Utf8ValidationB)\x88\x01\x01\x98\x01\x04\x98\x01\x01\xa2\x01\t\x12\x04NONE\x18\x84\x07\xa2\x01\x0b\x12\x06VERIFY\x18\xe7\x07\xb2\x01\x03\x08\xe8\x07R\x0eutf8Validation\x12~\n\x10message_encoding\x18\x05 \x01(\x0e\x32+.google.protobuf.FeatureSet.MessageEncodingB&\x88\x01\x01\x98\x01\x04\x98\x01\x01\xa2\x01\x14\x12\x0fLENGTH_PREFIXED\x18\x84\x07\xb2\x01\x03\x08\xe8\x07R\x0fmessageEncoding\x12\x82\x01\n\x0bjson_format\x18\x06 \x01(\x0e\x32&.google.protobuf.FeatureSet.JsonFormatB9\x88\x01\x01\x98\x01\x03\x98\x01\x06\x98\x01\x01\xa2\x01\x17\x12\x12LEGACY_BEST_EFFORT\x18\x84\x07\xa2\x01\n\x12\x05\x41LLOW\x18\xe7\x07\xb2\x01\x03\x08\xe8\x07R\njsonFormat\x12\xab\x01\n\x14\x65nforce_naming_style\x18\x07 \x01(\x0e\x32..google.protobuf.FeatureSet.EnforceNamingStyleBI\x88\x01\x02\x98\x01\x01\x98\x01\x02\x98\x01\x03\x98\x01\x04\x98\x01\x05\x98\x01\x06\x98\x01\x07\x98\x01\x08\x98\x01\t\xa2\x01\x11\x12\x0cSTYLE_LEGACY\x18\x84\x07\xa2\x01\x0e\x12\tSTYLE2024\x18\xe9\x07\xb2\x01\x03\x08\xe9\x07R\x12\x65nforceNamingStyle\x12\xb9\x01\n\x19\x64\x65\x66\x61ult_symbol_visibility\x18\x08 \x01(\x0e\x32\x45.google.protobuf.FeatureSet.VisibilityFeature.DefaultSymbolVisibilityB6\x88\x01\x02\x98\x01\x01\xa2\x01\x0f\x12\nEXPORT_ALL\x18\x84\x07\xa2\x01\x15\x12\x10\x45XPORT_TOP_LEVEL\x18\xe9\x07\xb2\x01\x03\x08\xe9\x07R\x17\x64\x65\x66\x61ultSymbolVisibility\x1a\xa1\x01\n\x11VisibilityFeature\"\x81\x01\n\x17\x44\x65\x66\x61ultSymbolVisibility\x12%\n!DEFAULT_SYMBOL_VISIBILITY_UNKNOWN\x10\x00\x12\x0e\n\nEXPORT_ALL\x10\x01\x12\x14\n\x10\x45XPORT_TOP_LEVEL\x10\x02\x12\r\n\tLOCAL_ALL\x10\x03\x12\n\n\x06STRICT\x10\x04J\x08\x08\x01\x10\x80\x80\x80\x80\x02\"\\\n\rFieldPresence\x12\x1a\n\x16\x46IELD_PRESENCE_UNKNOWN\x10\x00\x12\x0c\n\x08\x45XPLICIT\x10\x01\x12\x0c\n\x08IMPLICIT\x10\x02\x12\x13\n\x0fLEGACY_REQUIRED\x10\x03\"7\n\x08\x45numType\x12\x15\n\x11\x45NUM_TYPE_UNKNOWN\x10\x00\x12\x08\n\x04OPEN\x10\x01\x12\n\n\x06\x43LOSED\x10\x02\"V\n\x15RepeatedFieldEncoding\x12#\n\x1fREPEATED_FIELD_ENCODING_UNKNOWN\x10\x00\x12\n\n\x06PACKED\x10\x01\x12\x0c\n\x08\x45XPANDED\x10\x02\"I\n\x0eUtf8Validation\x12\x1b\n\x17UTF8_VALIDATION_UNKNOWN\x10\x00\x12\n\n\x06VERIFY\x10\x02\x12\x08\n\x04NONE\x10\x03\"\x04\x08\x01\x10\x01\"S\n\x0fMessageEncoding\x12\x1c\n\x18MESSAGE_ENCODING_UNKNOWN\x10\x00\x12\x13\n\x0fLENGTH_PREFIXED\x10\x01\x12\r\n\tDELIMITED\x10\x02\"H\n\nJsonFormat\x12\x17\n\x13JSON_FORMAT_UNKNOWN\x10\x00\x12\t\n\x05\x41LLOW\x10\x01\x12\x16\n\x12LEGACY_BEST_EFFORT\x10\x02\"W\n\x12\x45nforceNamingStyle\x12 \n\x1c\x45NFORCE_NAMING_STYLE_UNKNOWN\x10\x00\x12\r\n\tSTYLE2024\x10\x01\x12\x10\n\x0cSTYLE_LEGACY\x10\x02*\x06\x08\xe8\x07\x10\x8bN*\x06\x08\x8bN\x10\x90N*\x06\x08\x90N\x10\x91NJ\x06\x08\xe7\x07\x10\xe8\x07\"\xef\x03\n\x12\x46\x65\x61tureSetDefaults\x12X\n\x08\x64\x65\x66\x61ults\x18\x01 \x03(\x0b\x32<.google.protobuf.FeatureSetDefaults.FeatureSetEditionDefaultR\x08\x64\x65\x66\x61ults\x12\x41\n\x0fminimum_edition\x18\x04 \x01(\x0e\x32\x18.google.protobuf.EditionR\x0eminimumEdition\x12\x41\n\x0fmaximum_edition\x18\x05 \x01(\x0e\x32\x18.google.protobuf.EditionR\x0emaximumEdition\x1a\xf8\x01\n\x18\x46\x65\x61tureSetEditionDefault\x12\x32\n\x07\x65\x64ition\x18\x03 \x01(\x0e\x32\x18.google.protobuf.EditionR\x07\x65\x64ition\x12N\n\x14overridable_features\x18\x04 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\x13overridableFeatures\x12\x42\n\x0e\x66ixed_features\x18\x05 \x01(\x0b\x32\x1b.google.protobuf.FeatureSetR\rfixedFeaturesJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03R\x08\x66\x65\x61tures\"\xb5\x02\n\x0eSourceCodeInfo\x12\x44\n\x08location\x18\x01 \x03(\x0b\x32(.google.protobuf.SourceCodeInfo.LocationR\x08location\x1a\xce\x01\n\x08Location\x12\x16\n\x04path\x18\x01 \x03(\x05\x42\x02\x10\x01R\x04path\x12\x16\n\x04span\x18\x02 \x03(\x05\x42\x02\x10\x01R\x04span\x12)\n\x10leading_comments\x18\x03 \x01(\tR\x0fleadingComments\x12+\n\x11trailing_comments\x18\x04 \x01(\tR\x10trailingComments\x12:\n\x19leading_detached_comments\x18\x06 \x03(\tR\x17leadingDetachedComments*\x0c\x08\x80\xec\xca\xff\x01\x10\x81\xec\xca\xff\x01\"\xd0\x02\n\x11GeneratedCodeInfo\x12M\n\nannotation\x18\x01 \x03(\x0b\x32-.google.protobuf.GeneratedCodeInfo.AnnotationR\nannotation\x1a\xeb\x01\n\nAnnotation\x12\x16\n\x04path\x18\x01 \x03(\x05\x42\x02\x10\x01R\x04path\x12\x1f\n\x0bsource_file\x18\x02 \x01(\tR\nsourceFile\x12\x14\n\x05\x62\x65gin\x18\x03 \x01(\x05R\x05\x62\x65gin\x12\x10\n\x03\x65nd\x18\x04 \x01(\x05R\x03\x65nd\x12R\n\x08semantic\x18\x05 \x01(\x0e\x32\x36.google.protobuf.GeneratedCodeInfo.Annotation.SemanticR\x08semantic\"(\n\x08Semantic\x12\x08\n\x04NONE\x10\x00\x12\x07\n\x03SET\x10\x01\x12\t\n\x05\x41LIAS\x10\x02*\xa7\x02\n\x07\x45\x64ition\x12\x13\n\x0f\x45\x44ITION_UNKNOWN\x10\x00\x12\x13\n\x0e\x45\x44ITION_LEGACY\x10\x84\x07\x12\x13\n\x0e\x45\x44ITION_PROTO2\x10\xe6\x07\x12\x13\n\x0e\x45\x44ITION_PROTO3\x10\xe7\x07\x12\x11\n\x0c\x45\x44ITION_2023\x10\xe8\x07\x12\x11\n\x0c\x45\x44ITION_2024\x10\xe9\x07\x12\x17\n\x13\x45\x44ITION_1_TEST_ONLY\x10\x01\x12\x17\n\x13\x45\x44ITION_2_TEST_ONLY\x10\x02\x12\x1d\n\x17\x45\x44ITION_99997_TEST_ONLY\x10\x9d\x8d\x06\x12\x1d\n\x17\x45\x44ITION_99998_TEST_ONLY\x10\x9e\x8d\x06\x12\x1d\n\x17\x45\x44ITION_99999_TEST_ONLY\x10\x9f\x8d\x06\x12\x13\n\x0b\x45\x44ITION_MAX\x10\xff\xff\xff\xff\x07*U\n\x10SymbolVisibility\x12\x14\n\x10VISIBILITY_UNSET\x10\x00\x12\x14\n\x10VISIBILITY_LOCAL\x10\x01\x12\x15\n\x11VISIBILITY_EXPORT\x10\x02\x42~\n\x13\x63om.google.protobufB\x10\x44\x65scriptorProtosH\x01Z-google.golang.org/protobuf/types/descriptorpb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1aGoogle.Protobuf.Reflection')

_globals = globals()
if not _descriptor._USE_C_DESCRIPTORS:
  _EDITION = _descriptor.EnumDescriptor(
    name='Edition',
    full_name='google.protobuf.Edition',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='EDITION_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_LEGACY', index=1, number=900,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_PROTO2', index=2, number=998,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_PROTO3', index=3, number=999,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_2023', index=4, number=1000,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_2024', index=5, number=1001,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_1_TEST_ONLY', index=6, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_2_TEST_ONLY', index=7, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_99997_TEST_ONLY', index=8, number=99997,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_99998_TEST_ONLY', index=9, number=99998,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_99999_TEST_ONLY', index=10, number=99999,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EDITION_MAX', index=11, number=2147483647,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_EDITION)

  _SYMBOLVISIBILITY = _descriptor.EnumDescriptor(
    name='SymbolVisibility',
    full_name='google.protobuf.SymbolVisibility',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='VISIBILITY_UNSET', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='VISIBILITY_LOCAL', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='VISIBILITY_EXPORT', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_SYMBOLVISIBILITY)

  _EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE = _descriptor.EnumDescriptor(
    name='VerificationState',
    full_name='google.protobuf.ExtensionRangeOptions.VerificationState',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='DECLARATION', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='UNVERIFIED', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE)

  _FIELDDESCRIPTORPROTO_TYPE = _descriptor.EnumDescriptor(
    name='Type',
    full_name='google.protobuf.FieldDescriptorProto.Type',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='TYPE_DOUBLE', index=0, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_FLOAT', index=1, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_INT64', index=2, number=3,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_UINT64', index=3, number=4,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_INT32', index=4, number=5,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_FIXED64', index=5, number=6,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_FIXED32', index=6, number=7,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_BOOL', index=7, number=8,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_STRING', index=8, number=9,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_GROUP', index=9, number=10,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_MESSAGE', index=10, number=11,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_BYTES', index=11, number=12,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_UINT32', index=12, number=13,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_ENUM', index=13, number=14,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_SFIXED32', index=14, number=15,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_SFIXED64', index=15, number=16,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_SINT32', index=16, number=17,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TYPE_SINT64', index=17, number=18,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FIELDDESCRIPTORPROTO_TYPE)

  _FIELDDESCRIPTORPROTO_LABEL = _descriptor.EnumDescriptor(
    name='Label',
    full_name='google.protobuf.FieldDescriptorProto.Label',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='LABEL_OPTIONAL', index=0, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='LABEL_REPEATED', index=1, number=3,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='LABEL_REQUIRED', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FIELDDESCRIPTORPROTO_LABEL)

  _FILEOPTIONS_OPTIMIZEMODE = _descriptor.EnumDescriptor(
    name='OptimizeMode',
    full_name='google.protobuf.FileOptions.OptimizeMode',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='SPEED', index=0, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='CODE_SIZE', index=1, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='LITE_RUNTIME', index=2, number=3,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FILEOPTIONS_OPTIMIZEMODE)

  _FIELDOPTIONS_CTYPE = _descriptor.EnumDescriptor(
    name='CType',
    full_name='google.protobuf.FieldOptions.CType',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='STRING', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='CORD', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='STRING_PIECE', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FIELDOPTIONS_CTYPE)

  _FIELDOPTIONS_JSTYPE = _descriptor.EnumDescriptor(
    name='JSType',
    full_name='google.protobuf.FieldOptions.JSType',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='JS_NORMAL', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='JS_STRING', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='JS_NUMBER', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FIELDOPTIONS_JSTYPE)

  _FIELDOPTIONS_OPTIONRETENTION = _descriptor.EnumDescriptor(
    name='OptionRetention',
    full_name='google.protobuf.FieldOptions.OptionRetention',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='RETENTION_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='RETENTION_RUNTIME', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='RETENTION_SOURCE', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FIELDOPTIONS_OPTIONRETENTION)

  _FIELDOPTIONS_OPTIONTARGETTYPE = _descriptor.EnumDescriptor(
    name='OptionTargetType',
    full_name='google.protobuf.FieldOptions.OptionTargetType',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_FILE', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_EXTENSION_RANGE', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_MESSAGE', index=3, number=3,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_FIELD', index=4, number=4,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_ONEOF', index=5, number=5,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_ENUM', index=6, number=6,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_ENUM_ENTRY', index=7, number=7,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_SERVICE', index=8, number=8,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='TARGET_TYPE_METHOD', index=9, number=9,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FIELDOPTIONS_OPTIONTARGETTYPE)

  _METHODOPTIONS_IDEMPOTENCYLEVEL = _descriptor.EnumDescriptor(
    name='IdempotencyLevel',
    full_name='google.protobuf.MethodOptions.IdempotencyLevel',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='IDEMPOTENCY_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='NO_SIDE_EFFECTS', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='IDEMPOTENT', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_METHODOPTIONS_IDEMPOTENCYLEVEL)

  _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY = _descriptor.EnumDescriptor(
    name='DefaultSymbolVisibility',
    full_name='google.protobuf.FeatureSet.VisibilityFeature.DefaultSymbolVisibility',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='DEFAULT_SYMBOL_VISIBILITY_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EXPORT_ALL', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EXPORT_TOP_LEVEL', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='LOCAL_ALL', index=3, number=3,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='STRICT', index=4, number=4,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY)

  _FEATURESET_FIELDPRESENCE = _descriptor.EnumDescriptor(
    name='FieldPresence',
    full_name='google.protobuf.FeatureSet.FieldPresence',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='FIELD_PRESENCE_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EXPLICIT', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='IMPLICIT', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='LEGACY_REQUIRED', index=3, number=3,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FEATURESET_FIELDPRESENCE)

  _FEATURESET_ENUMTYPE = _descriptor.EnumDescriptor(
    name='EnumType',
    full_name='google.protobuf.FeatureSet.EnumType',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='ENUM_TYPE_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='OPEN', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='CLOSED', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FEATURESET_ENUMTYPE)

  _FEATURESET_REPEATEDFIELDENCODING = _descriptor.EnumDescriptor(
    name='RepeatedFieldEncoding',
    full_name='google.protobuf.FeatureSet.RepeatedFieldEncoding',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='REPEATED_FIELD_ENCODING_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='PACKED', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='EXPANDED', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FEATURESET_REPEATEDFIELDENCODING)

  _FEATURESET_UTF8VALIDATION = _descriptor.EnumDescriptor(
    name='Utf8Validation',
    full_name='google.protobuf.FeatureSet.Utf8Validation',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='UTF8_VALIDATION_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='VERIFY', index=1, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='NONE', index=2, number=3,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FEATURESET_UTF8VALIDATION)

  _FEATURESET_MESSAGEENCODING = _descriptor.EnumDescriptor(
    name='MessageEncoding',
    full_name='google.protobuf.FeatureSet.MessageEncoding',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='MESSAGE_ENCODING_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='LENGTH_PREFIXED', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='DELIMITED', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FEATURESET_MESSAGEENCODING)

  _FEATURESET_JSONFORMAT = _descriptor.EnumDescriptor(
    name='JsonFormat',
    full_name='google.protobuf.FeatureSet.JsonFormat',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='JSON_FORMAT_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='ALLOW', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='LEGACY_BEST_EFFORT', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FEATURESET_JSONFORMAT)

  _FEATURESET_ENFORCENAMINGSTYLE = _descriptor.EnumDescriptor(
    name='EnforceNamingStyle',
    full_name='google.protobuf.FeatureSet.EnforceNamingStyle',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='ENFORCE_NAMING_STYLE_UNKNOWN', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='STYLE2024', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='STYLE_LEGACY', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_FEATURESET_ENFORCENAMINGSTYLE)

  _GENERATEDCODEINFO_ANNOTATION_SEMANTIC = _descriptor.EnumDescriptor(
    name='Semantic',
    full_name='google.protobuf.GeneratedCodeInfo.Annotation.Semantic',
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
      _descriptor.EnumValueDescriptor(
        name='NONE', index=0, number=0,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='SET', index=1, number=1,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
      _descriptor.EnumValueDescriptor(
        name='ALIAS', index=2, number=2,
        serialized_options=None,
        type=None,
        create_key=_descriptor._internal_create_key),
    ],
    containing_type=None,
    serialized_options=None,
  )
  _sym_db.RegisterEnumDescriptor(_GENERATEDCODEINFO_ANNOTATION_SEMANTIC)


  _FILEDESCRIPTORSET = _descriptor.Descriptor(
    name='FileDescriptorSet',
    full_name='google.protobuf.FileDescriptorSet',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='file', full_name='google.protobuf.FileDescriptorSet.file', index=0,
        number=1, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='file', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(536000000, 536000001), ],
    oneofs=[
    ],
  )


  _FILEDESCRIPTORPROTO = _descriptor.Descriptor(
    name='FileDescriptorProto',
    full_name='google.protobuf.FileDescriptorProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.FileDescriptorProto.name', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='package', full_name='google.protobuf.FileDescriptorProto.package', index=1,
        number=2, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='package', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='dependency', full_name='google.protobuf.FileDescriptorProto.dependency', index=2,
        number=3, type=9, cpp_type=9, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='dependency', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='public_dependency', full_name='google.protobuf.FileDescriptorProto.public_dependency', index=3,
        number=10, type=5, cpp_type=1, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='publicDependency', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='weak_dependency', full_name='google.protobuf.FileDescriptorProto.weak_dependency', index=4,
        number=11, type=5, cpp_type=1, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='weakDependency', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='option_dependency', full_name='google.protobuf.FileDescriptorProto.option_dependency', index=5,
        number=15, type=9, cpp_type=9, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='optionDependency', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='message_type', full_name='google.protobuf.FileDescriptorProto.message_type', index=6,
        number=4, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='messageType', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='enum_type', full_name='google.protobuf.FileDescriptorProto.enum_type', index=7,
        number=5, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='enumType', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='service', full_name='google.protobuf.FileDescriptorProto.service', index=8,
        number=6, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='service', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='extension', full_name='google.protobuf.FileDescriptorProto.extension', index=9,
        number=7, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='extension', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.FileDescriptorProto.options', index=10,
        number=8, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='source_code_info', full_name='google.protobuf.FileDescriptorProto.source_code_info', index=11,
        number=9, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='sourceCodeInfo', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='syntax', full_name='google.protobuf.FileDescriptorProto.syntax', index=12,
        number=12, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='syntax', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='edition', full_name='google.protobuf.FileDescriptorProto.edition', index=13,
        number=14, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='edition', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _DESCRIPTORPROTO_EXTENSIONRANGE = _descriptor.Descriptor(
    name='ExtensionRange',
    full_name='google.protobuf.DescriptorProto.ExtensionRange',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='start', full_name='google.protobuf.DescriptorProto.ExtensionRange.start', index=0,
        number=1, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='start', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='end', full_name='google.protobuf.DescriptorProto.ExtensionRange.end', index=1,
        number=2, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='end', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.DescriptorProto.ExtensionRange.options', index=2,
        number=3, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _DESCRIPTORPROTO_RESERVEDRANGE = _descriptor.Descriptor(
    name='ReservedRange',
    full_name='google.protobuf.DescriptorProto.ReservedRange',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='start', full_name='google.protobuf.DescriptorProto.ReservedRange.start', index=0,
        number=1, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='start', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='end', full_name='google.protobuf.DescriptorProto.ReservedRange.end', index=1,
        number=2, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='end', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _DESCRIPTORPROTO = _descriptor.Descriptor(
    name='DescriptorProto',
    full_name='google.protobuf.DescriptorProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.DescriptorProto.name', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='field', full_name='google.protobuf.DescriptorProto.field', index=1,
        number=2, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='field', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='extension', full_name='google.protobuf.DescriptorProto.extension', index=2,
        number=6, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='extension', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='nested_type', full_name='google.protobuf.DescriptorProto.nested_type', index=3,
        number=3, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='nestedType', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='enum_type', full_name='google.protobuf.DescriptorProto.enum_type', index=4,
        number=4, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='enumType', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='extension_range', full_name='google.protobuf.DescriptorProto.extension_range', index=5,
        number=5, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='extensionRange', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='oneof_decl', full_name='google.protobuf.DescriptorProto.oneof_decl', index=6,
        number=8, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='oneofDecl', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.DescriptorProto.options', index=7,
        number=7, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='reserved_range', full_name='google.protobuf.DescriptorProto.reserved_range', index=8,
        number=9, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='reservedRange', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='reserved_name', full_name='google.protobuf.DescriptorProto.reserved_name', index=9,
        number=10, type=9, cpp_type=9, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='reservedName', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='visibility', full_name='google.protobuf.DescriptorProto.visibility', index=10,
        number=11, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='visibility', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_DESCRIPTORPROTO_EXTENSIONRANGE, _DESCRIPTORPROTO_RESERVEDRANGE, ],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _EXTENSIONRANGEOPTIONS_DECLARATION = _descriptor.Descriptor(
    name='Declaration',
    full_name='google.protobuf.ExtensionRangeOptions.Declaration',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='number', full_name='google.protobuf.ExtensionRangeOptions.Declaration.number', index=0,
        number=1, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='number', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='full_name', full_name='google.protobuf.ExtensionRangeOptions.Declaration.full_name', index=1,
        number=2, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='fullName', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='type', full_name='google.protobuf.ExtensionRangeOptions.Declaration.type', index=2,
        number=3, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='type', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='reserved', full_name='google.protobuf.ExtensionRangeOptions.Declaration.reserved', index=3,
        number=5, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='reserved', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='repeated', full_name='google.protobuf.ExtensionRangeOptions.Declaration.repeated', index=4,
        number=6, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='repeated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _EXTENSIONRANGEOPTIONS = _descriptor.Descriptor(
    name='ExtensionRangeOptions',
    full_name='google.protobuf.ExtensionRangeOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.ExtensionRangeOptions.uninterpreted_option', index=0,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='declaration', full_name='google.protobuf.ExtensionRangeOptions.declaration', index=1,
        number=2, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\002', json_name='declaration', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.ExtensionRangeOptions.features', index=2,
        number=50, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='verification', full_name='google.protobuf.ExtensionRangeOptions.verification', index=3,
        number=3, type=14, cpp_type=8, label=1,
        has_default_value=True, default_value=1,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\002', json_name='verification', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_EXTENSIONRANGEOPTIONS_DECLARATION, ],
    enum_types=[
      _EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE,
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _FIELDDESCRIPTORPROTO = _descriptor.Descriptor(
    name='FieldDescriptorProto',
    full_name='google.protobuf.FieldDescriptorProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.FieldDescriptorProto.name', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='number', full_name='google.protobuf.FieldDescriptorProto.number', index=1,
        number=3, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='number', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='label', full_name='google.protobuf.FieldDescriptorProto.label', index=2,
        number=4, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=1,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='label', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='type', full_name='google.protobuf.FieldDescriptorProto.type', index=3,
        number=5, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=1,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='type', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='type_name', full_name='google.protobuf.FieldDescriptorProto.type_name', index=4,
        number=6, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='typeName', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='extendee', full_name='google.protobuf.FieldDescriptorProto.extendee', index=5,
        number=2, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='extendee', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='default_value', full_name='google.protobuf.FieldDescriptorProto.default_value', index=6,
        number=7, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='defaultValue', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='oneof_index', full_name='google.protobuf.FieldDescriptorProto.oneof_index', index=7,
        number=9, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='oneofIndex', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='json_name', full_name='google.protobuf.FieldDescriptorProto.json_name', index=8,
        number=10, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='jsonName', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.FieldDescriptorProto.options', index=9,
        number=8, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='proto3_optional', full_name='google.protobuf.FieldDescriptorProto.proto3_optional', index=10,
        number=17, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='proto3Optional', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
      _FIELDDESCRIPTORPROTO_TYPE,
      _FIELDDESCRIPTORPROTO_LABEL,
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _ONEOFDESCRIPTORPROTO = _descriptor.Descriptor(
    name='OneofDescriptorProto',
    full_name='google.protobuf.OneofDescriptorProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.OneofDescriptorProto.name', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.OneofDescriptorProto.options', index=1,
        number=2, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE = _descriptor.Descriptor(
    name='EnumReservedRange',
    full_name='google.protobuf.EnumDescriptorProto.EnumReservedRange',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='start', full_name='google.protobuf.EnumDescriptorProto.EnumReservedRange.start', index=0,
        number=1, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='start', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='end', full_name='google.protobuf.EnumDescriptorProto.EnumReservedRange.end', index=1,
        number=2, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='end', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _ENUMDESCRIPTORPROTO = _descriptor.Descriptor(
    name='EnumDescriptorProto',
    full_name='google.protobuf.EnumDescriptorProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.EnumDescriptorProto.name', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='value', full_name='google.protobuf.EnumDescriptorProto.value', index=1,
        number=2, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='value', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.EnumDescriptorProto.options', index=2,
        number=3, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='reserved_range', full_name='google.protobuf.EnumDescriptorProto.reserved_range', index=3,
        number=4, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='reservedRange', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='reserved_name', full_name='google.protobuf.EnumDescriptorProto.reserved_name', index=4,
        number=5, type=9, cpp_type=9, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='reservedName', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='visibility', full_name='google.protobuf.EnumDescriptorProto.visibility', index=5,
        number=6, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='visibility', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE, ],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _ENUMVALUEDESCRIPTORPROTO = _descriptor.Descriptor(
    name='EnumValueDescriptorProto',
    full_name='google.protobuf.EnumValueDescriptorProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.EnumValueDescriptorProto.name', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='number', full_name='google.protobuf.EnumValueDescriptorProto.number', index=1,
        number=2, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='number', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.EnumValueDescriptorProto.options', index=2,
        number=3, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _SERVICEDESCRIPTORPROTO = _descriptor.Descriptor(
    name='ServiceDescriptorProto',
    full_name='google.protobuf.ServiceDescriptorProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.ServiceDescriptorProto.name', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='method', full_name='google.protobuf.ServiceDescriptorProto.method', index=1,
        number=2, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='method', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.ServiceDescriptorProto.options', index=2,
        number=3, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _METHODDESCRIPTORPROTO = _descriptor.Descriptor(
    name='MethodDescriptorProto',
    full_name='google.protobuf.MethodDescriptorProto',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.MethodDescriptorProto.name', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='input_type', full_name='google.protobuf.MethodDescriptorProto.input_type', index=1,
        number=2, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='inputType', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='output_type', full_name='google.protobuf.MethodDescriptorProto.output_type', index=2,
        number=3, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='outputType', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='options', full_name='google.protobuf.MethodDescriptorProto.options', index=3,
        number=4, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='options', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='client_streaming', full_name='google.protobuf.MethodDescriptorProto.client_streaming', index=4,
        number=5, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='clientStreaming', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='server_streaming', full_name='google.protobuf.MethodDescriptorProto.server_streaming', index=5,
        number=6, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='serverStreaming', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _FILEOPTIONS = _descriptor.Descriptor(
    name='FileOptions',
    full_name='google.protobuf.FileOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='java_package', full_name='google.protobuf.FileOptions.java_package', index=0,
        number=1, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='javaPackage', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='java_outer_classname', full_name='google.protobuf.FileOptions.java_outer_classname', index=1,
        number=8, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='javaOuterClassname', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='java_multiple_files', full_name='google.protobuf.FileOptions.java_multiple_files', index=2,
        number=10, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='javaMultipleFiles', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='java_generate_equals_and_hash', full_name='google.protobuf.FileOptions.java_generate_equals_and_hash', index=3,
        number=20, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\030\001', json_name='javaGenerateEqualsAndHash', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='java_string_check_utf8', full_name='google.protobuf.FileOptions.java_string_check_utf8', index=4,
        number=27, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='javaStringCheckUtf8', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='optimize_for', full_name='google.protobuf.FileOptions.optimize_for', index=5,
        number=9, type=14, cpp_type=8, label=1,
        has_default_value=True, default_value=1,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='optimizeFor', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='go_package', full_name='google.protobuf.FileOptions.go_package', index=6,
        number=11, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='goPackage', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='cc_generic_services', full_name='google.protobuf.FileOptions.cc_generic_services', index=7,
        number=16, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='ccGenericServices', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='java_generic_services', full_name='google.protobuf.FileOptions.java_generic_services', index=8,
        number=17, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='javaGenericServices', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='py_generic_services', full_name='google.protobuf.FileOptions.py_generic_services', index=9,
        number=18, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='pyGenericServices', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='deprecated', full_name='google.protobuf.FileOptions.deprecated', index=10,
        number=23, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='deprecated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='cc_enable_arenas', full_name='google.protobuf.FileOptions.cc_enable_arenas', index=11,
        number=31, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=True,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='ccEnableArenas', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='objc_class_prefix', full_name='google.protobuf.FileOptions.objc_class_prefix', index=12,
        number=36, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='objcClassPrefix', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='csharp_namespace', full_name='google.protobuf.FileOptions.csharp_namespace', index=13,
        number=37, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='csharpNamespace', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='swift_prefix', full_name='google.protobuf.FileOptions.swift_prefix', index=14,
        number=39, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='swiftPrefix', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='php_class_prefix', full_name='google.protobuf.FileOptions.php_class_prefix', index=15,
        number=40, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='phpClassPrefix', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='php_namespace', full_name='google.protobuf.FileOptions.php_namespace', index=16,
        number=41, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='phpNamespace', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='php_metadata_namespace', full_name='google.protobuf.FileOptions.php_metadata_namespace', index=17,
        number=44, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='phpMetadataNamespace', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='ruby_package', full_name='google.protobuf.FileOptions.ruby_package', index=18,
        number=45, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='rubyPackage', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.FileOptions.features', index=19,
        number=50, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.FileOptions.uninterpreted_option', index=20,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
      _FILEOPTIONS_OPTIMIZEMODE,
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _MESSAGEOPTIONS = _descriptor.Descriptor(
    name='MessageOptions',
    full_name='google.protobuf.MessageOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='message_set_wire_format', full_name='google.protobuf.MessageOptions.message_set_wire_format', index=0,
        number=1, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='messageSetWireFormat', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='no_standard_descriptor_accessor', full_name='google.protobuf.MessageOptions.no_standard_descriptor_accessor', index=1,
        number=2, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='noStandardDescriptorAccessor', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='deprecated', full_name='google.protobuf.MessageOptions.deprecated', index=2,
        number=3, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='deprecated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='map_entry', full_name='google.protobuf.MessageOptions.map_entry', index=3,
        number=7, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='mapEntry', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='deprecated_legacy_json_field_conflicts', full_name='google.protobuf.MessageOptions.deprecated_legacy_json_field_conflicts', index=4,
        number=11, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\030\001', json_name='deprecatedLegacyJsonFieldConflicts', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.MessageOptions.features', index=5,
        number=12, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.MessageOptions.uninterpreted_option', index=6,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _FIELDOPTIONS_EDITIONDEFAULT = _descriptor.Descriptor(
    name='EditionDefault',
    full_name='google.protobuf.FieldOptions.EditionDefault',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='edition', full_name='google.protobuf.FieldOptions.EditionDefault.edition', index=0,
        number=3, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='edition', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='value', full_name='google.protobuf.FieldOptions.EditionDefault.value', index=1,
        number=2, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='value', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _FIELDOPTIONS_FEATURESUPPORT = _descriptor.Descriptor(
    name='FeatureSupport',
    full_name='google.protobuf.FieldOptions.FeatureSupport',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='edition_introduced', full_name='google.protobuf.FieldOptions.FeatureSupport.edition_introduced', index=0,
        number=1, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='editionIntroduced', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='edition_deprecated', full_name='google.protobuf.FieldOptions.FeatureSupport.edition_deprecated', index=1,
        number=2, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='editionDeprecated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='deprecation_warning', full_name='google.protobuf.FieldOptions.FeatureSupport.deprecation_warning', index=2,
        number=3, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='deprecationWarning', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='edition_removed', full_name='google.protobuf.FieldOptions.FeatureSupport.edition_removed', index=3,
        number=4, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='editionRemoved', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _FIELDOPTIONS = _descriptor.Descriptor(
    name='FieldOptions',
    full_name='google.protobuf.FieldOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='ctype', full_name='google.protobuf.FieldOptions.ctype', index=0,
        number=1, type=14, cpp_type=8, label=1,
        has_default_value=True, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='ctype', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='packed', full_name='google.protobuf.FieldOptions.packed', index=1,
        number=2, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='packed', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='jstype', full_name='google.protobuf.FieldOptions.jstype', index=2,
        number=6, type=14, cpp_type=8, label=1,
        has_default_value=True, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='jstype', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='lazy', full_name='google.protobuf.FieldOptions.lazy', index=3,
        number=5, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='lazy', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='unverified_lazy', full_name='google.protobuf.FieldOptions.unverified_lazy', index=4,
        number=15, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='unverifiedLazy', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='deprecated', full_name='google.protobuf.FieldOptions.deprecated', index=5,
        number=3, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='deprecated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='weak', full_name='google.protobuf.FieldOptions.weak', index=6,
        number=10, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='weak', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='debug_redact', full_name='google.protobuf.FieldOptions.debug_redact', index=7,
        number=16, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='debugRedact', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='retention', full_name='google.protobuf.FieldOptions.retention', index=8,
        number=17, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='retention', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='targets', full_name='google.protobuf.FieldOptions.targets', index=9,
        number=19, type=14, cpp_type=8, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='targets', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='edition_defaults', full_name='google.protobuf.FieldOptions.edition_defaults', index=10,
        number=20, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='editionDefaults', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.FieldOptions.features', index=11,
        number=21, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='feature_support', full_name='google.protobuf.FieldOptions.feature_support', index=12,
        number=22, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='featureSupport', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.FieldOptions.uninterpreted_option', index=13,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_FIELDOPTIONS_EDITIONDEFAULT, _FIELDOPTIONS_FEATURESUPPORT, ],
    enum_types=[
      _FIELDOPTIONS_CTYPE,
      _FIELDOPTIONS_JSTYPE,
      _FIELDOPTIONS_OPTIONRETENTION,
      _FIELDOPTIONS_OPTIONTARGETTYPE,
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _ONEOFOPTIONS = _descriptor.Descriptor(
    name='OneofOptions',
    full_name='google.protobuf.OneofOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.OneofOptions.features', index=0,
        number=1, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.OneofOptions.uninterpreted_option', index=1,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _ENUMOPTIONS = _descriptor.Descriptor(
    name='EnumOptions',
    full_name='google.protobuf.EnumOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='allow_alias', full_name='google.protobuf.EnumOptions.allow_alias', index=0,
        number=2, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='allowAlias', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='deprecated', full_name='google.protobuf.EnumOptions.deprecated', index=1,
        number=3, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='deprecated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='deprecated_legacy_json_field_conflicts', full_name='google.protobuf.EnumOptions.deprecated_legacy_json_field_conflicts', index=2,
        number=6, type=8, cpp_type=7, label=1,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\030\001', json_name='deprecatedLegacyJsonFieldConflicts', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.EnumOptions.features', index=3,
        number=7, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.EnumOptions.uninterpreted_option', index=4,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _ENUMVALUEOPTIONS = _descriptor.Descriptor(
    name='EnumValueOptions',
    full_name='google.protobuf.EnumValueOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='deprecated', full_name='google.protobuf.EnumValueOptions.deprecated', index=0,
        number=1, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='deprecated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.EnumValueOptions.features', index=1,
        number=2, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='debug_redact', full_name='google.protobuf.EnumValueOptions.debug_redact', index=2,
        number=3, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='debugRedact', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='feature_support', full_name='google.protobuf.EnumValueOptions.feature_support', index=3,
        number=4, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='featureSupport', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.EnumValueOptions.uninterpreted_option', index=4,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _SERVICEOPTIONS = _descriptor.Descriptor(
    name='ServiceOptions',
    full_name='google.protobuf.ServiceOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.ServiceOptions.features', index=0,
        number=34, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='deprecated', full_name='google.protobuf.ServiceOptions.deprecated', index=1,
        number=33, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='deprecated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.ServiceOptions.uninterpreted_option', index=2,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _METHODOPTIONS = _descriptor.Descriptor(
    name='MethodOptions',
    full_name='google.protobuf.MethodOptions',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='deprecated', full_name='google.protobuf.MethodOptions.deprecated', index=0,
        number=33, type=8, cpp_type=7, label=1,
        has_default_value=True, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='deprecated', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='idempotency_level', full_name='google.protobuf.MethodOptions.idempotency_level', index=1,
        number=34, type=14, cpp_type=8, label=1,
        has_default_value=True, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='idempotencyLevel', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='features', full_name='google.protobuf.MethodOptions.features', index=2,
        number=35, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='features', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='uninterpreted_option', full_name='google.protobuf.MethodOptions.uninterpreted_option', index=3,
        number=999, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='uninterpretedOption', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
      _METHODOPTIONS_IDEMPOTENCYLEVEL,
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 536870912), ],
    oneofs=[
    ],
  )


  _UNINTERPRETEDOPTION_NAMEPART = _descriptor.Descriptor(
    name='NamePart',
    full_name='google.protobuf.UninterpretedOption.NamePart',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name_part', full_name='google.protobuf.UninterpretedOption.NamePart.name_part', index=0,
        number=1, type=9, cpp_type=9, label=2,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='namePart', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='is_extension', full_name='google.protobuf.UninterpretedOption.NamePart.is_extension', index=1,
        number=2, type=8, cpp_type=7, label=2,
        has_default_value=False, default_value=False,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='isExtension', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _UNINTERPRETEDOPTION = _descriptor.Descriptor(
    name='UninterpretedOption',
    full_name='google.protobuf.UninterpretedOption',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='name', full_name='google.protobuf.UninterpretedOption.name', index=0,
        number=2, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='name', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='identifier_value', full_name='google.protobuf.UninterpretedOption.identifier_value', index=1,
        number=3, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='identifierValue', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='positive_int_value', full_name='google.protobuf.UninterpretedOption.positive_int_value', index=2,
        number=4, type=4, cpp_type=4, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='positiveIntValue', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='negative_int_value', full_name='google.protobuf.UninterpretedOption.negative_int_value', index=3,
        number=5, type=3, cpp_type=2, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='negativeIntValue', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='double_value', full_name='google.protobuf.UninterpretedOption.double_value', index=4,
        number=6, type=1, cpp_type=5, label=1,
        has_default_value=False, default_value=float(0),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='doubleValue', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='string_value', full_name='google.protobuf.UninterpretedOption.string_value', index=5,
        number=7, type=12, cpp_type=9, label=1,
        has_default_value=False, default_value=b"",
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='stringValue', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='aggregate_value', full_name='google.protobuf.UninterpretedOption.aggregate_value', index=6,
        number=8, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='aggregateValue', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_UNINTERPRETEDOPTION_NAMEPART, ],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _FEATURESET_VISIBILITYFEATURE = _descriptor.Descriptor(
    name='VisibilityFeature',
    full_name='google.protobuf.FeatureSet.VisibilityFeature',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
      _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY,
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _FEATURESET = _descriptor.Descriptor(
    name='FeatureSet',
    full_name='google.protobuf.FeatureSet',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='field_presence', full_name='google.protobuf.FeatureSet.field_presence', index=0,
        number=1, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\001\230\001\004\230\001\001\242\001\r\022\010EXPLICIT\030\204\007\242\001\r\022\010IMPLICIT\030\347\007\242\001\r\022\010EXPLICIT\030\350\007\262\001\003\010\350\007', json_name='fieldPresence', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='enum_type', full_name='google.protobuf.FeatureSet.enum_type', index=1,
        number=2, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\001\230\001\006\230\001\001\242\001\013\022\006CLOSED\030\204\007\242\001\t\022\004OPEN\030\347\007\262\001\003\010\350\007', json_name='enumType', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='repeated_field_encoding', full_name='google.protobuf.FeatureSet.repeated_field_encoding', index=2,
        number=3, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\001\230\001\004\230\001\001\242\001\r\022\010EXPANDED\030\204\007\242\001\013\022\006PACKED\030\347\007\262\001\003\010\350\007', json_name='repeatedFieldEncoding', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='utf8_validation', full_name='google.protobuf.FeatureSet.utf8_validation', index=3,
        number=4, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\001\230\001\004\230\001\001\242\001\t\022\004NONE\030\204\007\242\001\013\022\006VERIFY\030\347\007\262\001\003\010\350\007', json_name='utf8Validation', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='message_encoding', full_name='google.protobuf.FeatureSet.message_encoding', index=4,
        number=5, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\001\230\001\004\230\001\001\242\001\024\022\017LENGTH_PREFIXED\030\204\007\262\001\003\010\350\007', json_name='messageEncoding', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='json_format', full_name='google.protobuf.FeatureSet.json_format', index=5,
        number=6, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\001\230\001\003\230\001\006\230\001\001\242\001\027\022\022LEGACY_BEST_EFFORT\030\204\007\242\001\n\022\005ALLOW\030\347\007\262\001\003\010\350\007', json_name='jsonFormat', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='enforce_naming_style', full_name='google.protobuf.FeatureSet.enforce_naming_style', index=6,
        number=7, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\002\230\001\001\230\001\002\230\001\003\230\001\004\230\001\005\230\001\006\230\001\007\230\001\010\230\001\t\242\001\021\022\014STYLE_LEGACY\030\204\007\242\001\016\022\tSTYLE2024\030\351\007\262\001\003\010\351\007', json_name='enforceNamingStyle', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='default_symbol_visibility', full_name='google.protobuf.FeatureSet.default_symbol_visibility', index=7,
        number=8, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\210\001\002\230\001\001\242\001\017\022\nEXPORT_ALL\030\204\007\242\001\025\022\020EXPORT_TOP_LEVEL\030\351\007\262\001\003\010\351\007', json_name='defaultSymbolVisibility', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_FEATURESET_VISIBILITYFEATURE, ],
    enum_types=[
      _FEATURESET_FIELDPRESENCE,
      _FEATURESET_ENUMTYPE,
      _FEATURESET_REPEATEDFIELDENCODING,
      _FEATURESET_UTF8VALIDATION,
      _FEATURESET_MESSAGEENCODING,
      _FEATURESET_JSONFORMAT,
      _FEATURESET_ENFORCENAMINGSTYLE,
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(1000, 9995), (9995, 10000), (10000, 10001), ],
    oneofs=[
    ],
  )


  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT = _descriptor.Descriptor(
    name='FeatureSetEditionDefault',
    full_name='google.protobuf.FeatureSetDefaults.FeatureSetEditionDefault',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='edition', full_name='google.protobuf.FeatureSetDefaults.FeatureSetEditionDefault.edition', index=0,
        number=3, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='edition', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='overridable_features', full_name='google.protobuf.FeatureSetDefaults.FeatureSetEditionDefault.overridable_features', index=1,
        number=4, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='overridableFeatures', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='fixed_features', full_name='google.protobuf.FeatureSetDefaults.FeatureSetEditionDefault.fixed_features', index=2,
        number=5, type=11, cpp_type=10, label=1,
        has_default_value=False, default_value=None,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='fixedFeatures', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _FEATURESETDEFAULTS = _descriptor.Descriptor(
    name='FeatureSetDefaults',
    full_name='google.protobuf.FeatureSetDefaults',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='defaults', full_name='google.protobuf.FeatureSetDefaults.defaults', index=0,
        number=1, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='defaults', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='minimum_edition', full_name='google.protobuf.FeatureSetDefaults.minimum_edition', index=1,
        number=4, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='minimumEdition', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='maximum_edition', full_name='google.protobuf.FeatureSetDefaults.maximum_edition', index=2,
        number=5, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='maximumEdition', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT, ],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )


  _SOURCECODEINFO_LOCATION = _descriptor.Descriptor(
    name='Location',
    full_name='google.protobuf.SourceCodeInfo.Location',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='path', full_name='google.protobuf.SourceCodeInfo.Location.path', index=0,
        number=1, type=5, cpp_type=1, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\020\001', json_name='path', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='span', full_name='google.protobuf.SourceCodeInfo.Location.span', index=1,
        number=2, type=5, cpp_type=1, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\020\001', json_name='span', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='leading_comments', full_name='google.protobuf.SourceCodeInfo.Location.leading_comments', index=2,
        number=3, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='leadingComments', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='trailing_comments', full_name='google.protobuf.SourceCodeInfo.Location.trailing_comments', index=3,
        number=4, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='trailingComments', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='leading_detached_comments', full_name='google.protobuf.SourceCodeInfo.Location.leading_detached_comments', index=4,
        number=6, type=9, cpp_type=9, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='leadingDetachedComments', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _SOURCECODEINFO = _descriptor.Descriptor(
    name='SourceCodeInfo',
    full_name='google.protobuf.SourceCodeInfo',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='location', full_name='google.protobuf.SourceCodeInfo.location', index=0,
        number=1, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='location', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_SOURCECODEINFO_LOCATION, ],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=True,
    extension_ranges=[(536000000, 536000001), ],
    oneofs=[
    ],
  )


  _GENERATEDCODEINFO_ANNOTATION = _descriptor.Descriptor(
    name='Annotation',
    full_name='google.protobuf.GeneratedCodeInfo.Annotation',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='path', full_name='google.protobuf.GeneratedCodeInfo.Annotation.path', index=0,
        number=1, type=5, cpp_type=1, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=b'\020\001', json_name='path', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='source_file', full_name='google.protobuf.GeneratedCodeInfo.Annotation.source_file', index=1,
        number=2, type=9, cpp_type=9, label=1,
        has_default_value=False, default_value=b"".decode('utf-8'),
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='sourceFile', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='begin', full_name='google.protobuf.GeneratedCodeInfo.Annotation.begin', index=2,
        number=3, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='begin', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='end', full_name='google.protobuf.GeneratedCodeInfo.Annotation.end', index=3,
        number=4, type=5, cpp_type=1, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='end', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
      _descriptor.FieldDescriptor(
        name='semantic', full_name='google.protobuf.GeneratedCodeInfo.Annotation.semantic', index=4,
        number=5, type=14, cpp_type=8, label=1,
        has_default_value=False, default_value=0,
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='semantic', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
      _GENERATEDCODEINFO_ANNOTATION_SEMANTIC,
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _GENERATEDCODEINFO = _descriptor.Descriptor(
    name='GeneratedCodeInfo',
    full_name='google.protobuf.GeneratedCodeInfo',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
      _descriptor.FieldDescriptor(
        name='annotation', full_name='google.protobuf.GeneratedCodeInfo.annotation', index=0,
        number=1, type=11, cpp_type=10, label=3,
        has_default_value=False, default_value=[],
        message_type=None, enum_type=None, containing_type=None,
        is_extension=False, extension_scope=None,
        serialized_options=None, json_name='annotation', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[_GENERATEDCODEINFO_ANNOTATION, ],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    extension_ranges=[],
    oneofs=[
    ],
  )

  _FILEDESCRIPTORSET.fields_by_name['file'].message_type = _FILEDESCRIPTORPROTO
  _FILEDESCRIPTORPROTO.fields_by_name['message_type'].message_type = _DESCRIPTORPROTO
  _FILEDESCRIPTORPROTO.fields_by_name['enum_type'].message_type = _ENUMDESCRIPTORPROTO
  _FILEDESCRIPTORPROTO.fields_by_name['service'].message_type = _SERVICEDESCRIPTORPROTO
  _FILEDESCRIPTORPROTO.fields_by_name['extension'].message_type = _FIELDDESCRIPTORPROTO
  _FILEDESCRIPTORPROTO.fields_by_name['options'].message_type = _FILEOPTIONS
  _FILEDESCRIPTORPROTO.fields_by_name['source_code_info'].message_type = _SOURCECODEINFO
  _FILEDESCRIPTORPROTO.fields_by_name['edition'].enum_type = _EDITION
  _DESCRIPTORPROTO_EXTENSIONRANGE.fields_by_name['options'].message_type = _EXTENSIONRANGEOPTIONS
  _DESCRIPTORPROTO_EXTENSIONRANGE.containing_type = _DESCRIPTORPROTO
  _DESCRIPTORPROTO_RESERVEDRANGE.containing_type = _DESCRIPTORPROTO
  _DESCRIPTORPROTO.fields_by_name['field'].message_type = _FIELDDESCRIPTORPROTO
  _DESCRIPTORPROTO.fields_by_name['extension'].message_type = _FIELDDESCRIPTORPROTO
  _DESCRIPTORPROTO.fields_by_name['nested_type'].message_type = _DESCRIPTORPROTO
  _DESCRIPTORPROTO.fields_by_name['enum_type'].message_type = _ENUMDESCRIPTORPROTO
  _DESCRIPTORPROTO.fields_by_name['extension_range'].message_type = _DESCRIPTORPROTO_EXTENSIONRANGE
  _DESCRIPTORPROTO.fields_by_name['oneof_decl'].message_type = _ONEOFDESCRIPTORPROTO
  _DESCRIPTORPROTO.fields_by_name['options'].message_type = _MESSAGEOPTIONS
  _DESCRIPTORPROTO.fields_by_name['reserved_range'].message_type = _DESCRIPTORPROTO_RESERVEDRANGE
  _DESCRIPTORPROTO.fields_by_name['visibility'].enum_type = _SYMBOLVISIBILITY
  _EXTENSIONRANGEOPTIONS_DECLARATION.containing_type = _EXTENSIONRANGEOPTIONS
  _EXTENSIONRANGEOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _EXTENSIONRANGEOPTIONS.fields_by_name['declaration'].message_type = _EXTENSIONRANGEOPTIONS_DECLARATION
  _EXTENSIONRANGEOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _EXTENSIONRANGEOPTIONS.fields_by_name['verification'].enum_type = _EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE
  _EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE.containing_type = _EXTENSIONRANGEOPTIONS
  _FIELDDESCRIPTORPROTO.fields_by_name['label'].enum_type = _FIELDDESCRIPTORPROTO_LABEL
  _FIELDDESCRIPTORPROTO.fields_by_name['type'].enum_type = _FIELDDESCRIPTORPROTO_TYPE
  _FIELDDESCRIPTORPROTO.fields_by_name['options'].message_type = _FIELDOPTIONS
  _FIELDDESCRIPTORPROTO_TYPE.containing_type = _FIELDDESCRIPTORPROTO
  _FIELDDESCRIPTORPROTO_LABEL.containing_type = _FIELDDESCRIPTORPROTO
  _ONEOFDESCRIPTORPROTO.fields_by_name['options'].message_type = _ONEOFOPTIONS
  _ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE.containing_type = _ENUMDESCRIPTORPROTO
  _ENUMDESCRIPTORPROTO.fields_by_name['value'].message_type = _ENUMVALUEDESCRIPTORPROTO
  _ENUMDESCRIPTORPROTO.fields_by_name['options'].message_type = _ENUMOPTIONS
  _ENUMDESCRIPTORPROTO.fields_by_name['reserved_range'].message_type = _ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE
  _ENUMDESCRIPTORPROTO.fields_by_name['visibility'].enum_type = _SYMBOLVISIBILITY
  _ENUMVALUEDESCRIPTORPROTO.fields_by_name['options'].message_type = _ENUMVALUEOPTIONS
  _SERVICEDESCRIPTORPROTO.fields_by_name['method'].message_type = _METHODDESCRIPTORPROTO
  _SERVICEDESCRIPTORPROTO.fields_by_name['options'].message_type = _SERVICEOPTIONS
  _METHODDESCRIPTORPROTO.fields_by_name['options'].message_type = _METHODOPTIONS
  _FILEOPTIONS.fields_by_name['optimize_for'].enum_type = _FILEOPTIONS_OPTIMIZEMODE
  _FILEOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _FILEOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _FILEOPTIONS_OPTIMIZEMODE.containing_type = _FILEOPTIONS
  _MESSAGEOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _MESSAGEOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _FIELDOPTIONS_EDITIONDEFAULT.fields_by_name['edition'].enum_type = _EDITION
  _FIELDOPTIONS_EDITIONDEFAULT.containing_type = _FIELDOPTIONS
  _FIELDOPTIONS_FEATURESUPPORT.fields_by_name['edition_introduced'].enum_type = _EDITION
  _FIELDOPTIONS_FEATURESUPPORT.fields_by_name['edition_deprecated'].enum_type = _EDITION
  _FIELDOPTIONS_FEATURESUPPORT.fields_by_name['edition_removed'].enum_type = _EDITION
  _FIELDOPTIONS_FEATURESUPPORT.containing_type = _FIELDOPTIONS
  _FIELDOPTIONS.fields_by_name['ctype'].enum_type = _FIELDOPTIONS_CTYPE
  _FIELDOPTIONS.fields_by_name['jstype'].enum_type = _FIELDOPTIONS_JSTYPE
  _FIELDOPTIONS.fields_by_name['retention'].enum_type = _FIELDOPTIONS_OPTIONRETENTION
  _FIELDOPTIONS.fields_by_name['targets'].enum_type = _FIELDOPTIONS_OPTIONTARGETTYPE
  _FIELDOPTIONS.fields_by_name['edition_defaults'].message_type = _FIELDOPTIONS_EDITIONDEFAULT
  _FIELDOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _FIELDOPTIONS.fields_by_name['feature_support'].message_type = _FIELDOPTIONS_FEATURESUPPORT
  _FIELDOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _FIELDOPTIONS_CTYPE.containing_type = _FIELDOPTIONS
  _FIELDOPTIONS_JSTYPE.containing_type = _FIELDOPTIONS
  _FIELDOPTIONS_OPTIONRETENTION.containing_type = _FIELDOPTIONS
  _FIELDOPTIONS_OPTIONTARGETTYPE.containing_type = _FIELDOPTIONS
  _ONEOFOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _ONEOFOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _ENUMOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _ENUMOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _ENUMVALUEOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _ENUMVALUEOPTIONS.fields_by_name['feature_support'].message_type = _FIELDOPTIONS_FEATURESUPPORT
  _ENUMVALUEOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _SERVICEOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _SERVICEOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _METHODOPTIONS.fields_by_name['idempotency_level'].enum_type = _METHODOPTIONS_IDEMPOTENCYLEVEL
  _METHODOPTIONS.fields_by_name['features'].message_type = _FEATURESET
  _METHODOPTIONS.fields_by_name['uninterpreted_option'].message_type = _UNINTERPRETEDOPTION
  _METHODOPTIONS_IDEMPOTENCYLEVEL.containing_type = _METHODOPTIONS
  _UNINTERPRETEDOPTION_NAMEPART.containing_type = _UNINTERPRETEDOPTION
  _UNINTERPRETEDOPTION.fields_by_name['name'].message_type = _UNINTERPRETEDOPTION_NAMEPART
  _FEATURESET_VISIBILITYFEATURE.containing_type = _FEATURESET
  _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY.containing_type = _FEATURESET_VISIBILITYFEATURE
  _FEATURESET.fields_by_name['field_presence'].enum_type = _FEATURESET_FIELDPRESENCE
  _FEATURESET.fields_by_name['enum_type'].enum_type = _FEATURESET_ENUMTYPE
  _FEATURESET.fields_by_name['repeated_field_encoding'].enum_type = _FEATURESET_REPEATEDFIELDENCODING
  _FEATURESET.fields_by_name['utf8_validation'].enum_type = _FEATURESET_UTF8VALIDATION
  _FEATURESET.fields_by_name['message_encoding'].enum_type = _FEATURESET_MESSAGEENCODING
  _FEATURESET.fields_by_name['json_format'].enum_type = _FEATURESET_JSONFORMAT
  _FEATURESET.fields_by_name['enforce_naming_style'].enum_type = _FEATURESET_ENFORCENAMINGSTYLE
  _FEATURESET.fields_by_name['default_symbol_visibility'].enum_type = _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY
  _FEATURESET_FIELDPRESENCE.containing_type = _FEATURESET
  _FEATURESET_ENUMTYPE.containing_type = _FEATURESET
  _FEATURESET_REPEATEDFIELDENCODING.containing_type = _FEATURESET
  _FEATURESET_UTF8VALIDATION.containing_type = _FEATURESET
  _FEATURESET_MESSAGEENCODING.containing_type = _FEATURESET
  _FEATURESET_JSONFORMAT.containing_type = _FEATURESET
  _FEATURESET_ENFORCENAMINGSTYLE.containing_type = _FEATURESET
  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT.fields_by_name['edition'].enum_type = _EDITION
  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT.fields_by_name['overridable_features'].message_type = _FEATURESET
  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT.fields_by_name['fixed_features'].message_type = _FEATURESET
  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT.containing_type = _FEATURESETDEFAULTS
  _FEATURESETDEFAULTS.fields_by_name['defaults'].message_type = _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT
  _FEATURESETDEFAULTS.fields_by_name['minimum_edition'].enum_type = _EDITION
  _FEATURESETDEFAULTS.fields_by_name['maximum_edition'].enum_type = _EDITION
  _SOURCECODEINFO_LOCATION.containing_type = _SOURCECODEINFO
  _SOURCECODEINFO.fields_by_name['location'].message_type = _SOURCECODEINFO_LOCATION
  _GENERATEDCODEINFO_ANNOTATION.fields_by_name['semantic'].enum_type = _GENERATEDCODEINFO_ANNOTATION_SEMANTIC
  _GENERATEDCODEINFO_ANNOTATION.containing_type = _GENERATEDCODEINFO
  _GENERATEDCODEINFO_ANNOTATION_SEMANTIC.containing_type = _GENERATEDCODEINFO_ANNOTATION
  _GENERATEDCODEINFO.fields_by_name['annotation'].message_type = _GENERATEDCODEINFO_ANNOTATION
  DESCRIPTOR.message_types_by_name['FileDescriptorSet'] = _FILEDESCRIPTORSET
  DESCRIPTOR.message_types_by_name['FileDescriptorProto'] = _FILEDESCRIPTORPROTO
  DESCRIPTOR.message_types_by_name['DescriptorProto'] = _DESCRIPTORPROTO
  DESCRIPTOR.message_types_by_name['ExtensionRangeOptions'] = _EXTENSIONRANGEOPTIONS
  DESCRIPTOR.message_types_by_name['FieldDescriptorProto'] = _FIELDDESCRIPTORPROTO
  DESCRIPTOR.message_types_by_name['OneofDescriptorProto'] = _ONEOFDESCRIPTORPROTO
  DESCRIPTOR.message_types_by_name['EnumDescriptorProto'] = _ENUMDESCRIPTORPROTO
  DESCRIPTOR.message_types_by_name['EnumValueDescriptorProto'] = _ENUMVALUEDESCRIPTORPROTO
  DESCRIPTOR.message_types_by_name['ServiceDescriptorProto'] = _SERVICEDESCRIPTORPROTO
  DESCRIPTOR.message_types_by_name['MethodDescriptorProto'] = _METHODDESCRIPTORPROTO
  DESCRIPTOR.message_types_by_name['FileOptions'] = _FILEOPTIONS
  DESCRIPTOR.message_types_by_name['MessageOptions'] = _MESSAGEOPTIONS
  DESCRIPTOR.message_types_by_name['FieldOptions'] = _FIELDOPTIONS
  DESCRIPTOR.message_types_by_name['OneofOptions'] = _ONEOFOPTIONS
  DESCRIPTOR.message_types_by_name['EnumOptions'] = _ENUMOPTIONS
  DESCRIPTOR.message_types_by_name['EnumValueOptions'] = _ENUMVALUEOPTIONS
  DESCRIPTOR.message_types_by_name['ServiceOptions'] = _SERVICEOPTIONS
  DESCRIPTOR.message_types_by_name['MethodOptions'] = _METHODOPTIONS
  DESCRIPTOR.message_types_by_name['UninterpretedOption'] = _UNINTERPRETEDOPTION
  DESCRIPTOR.message_types_by_name['FeatureSet'] = _FEATURESET
  DESCRIPTOR.message_types_by_name['FeatureSetDefaults'] = _FEATURESETDEFAULTS
  DESCRIPTOR.message_types_by_name['SourceCodeInfo'] = _SOURCECODEINFO
  DESCRIPTOR.message_types_by_name['GeneratedCodeInfo'] = _GENERATEDCODEINFO
  DESCRIPTOR.enum_types_by_name['Edition'] = _EDITION
  DESCRIPTOR.enum_types_by_name['SymbolVisibility'] = _SYMBOLVISIBILITY
  _sym_db.RegisterFileDescriptor(DESCRIPTOR)

  class _ResolvedFeatures:
    def __init__(self, features = None, **kwargs):
      if features:
        for k, v in features.FIELDS.items():
          setattr(self, k, getattr(features, k))
      else:
        for k, v in kwargs.items():
          setattr(self, k, v)
  DESCRIPTOR._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORSET._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORSET.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[8]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[9]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[10]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[11]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[12]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEDESCRIPTORPROTO.fields[13]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[8]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[9]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO.fields[10]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO_EXTENSIONRANGE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO_EXTENSIONRANGE.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO_EXTENSIONRANGE.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO_EXTENSIONRANGE.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO_RESERVEDRANGE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO_RESERVEDRANGE.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _DESCRIPTORPROTO_RESERVEDRANGE.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_DECLARATION._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_DECLARATION.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_DECLARATION.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_DECLARATION.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_DECLARATION.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_DECLARATION.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[8]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[9]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO.fields[10]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ONEOFDESCRIPTORPROTO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ONEOFDESCRIPTORPROTO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ONEOFDESCRIPTORPROTO.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEDESCRIPTORPROTO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEDESCRIPTORPROTO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEDESCRIPTORPROTO.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEDESCRIPTORPROTO.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SERVICEDESCRIPTORPROTO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SERVICEDESCRIPTORPROTO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SERVICEDESCRIPTORPROTO.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SERVICEDESCRIPTORPROTO.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODDESCRIPTORPROTO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODDESCRIPTORPROTO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODDESCRIPTORPROTO.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODDESCRIPTORPROTO.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODDESCRIPTORPROTO.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODDESCRIPTORPROTO.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODDESCRIPTORPROTO.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[8]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[9]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[10]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[11]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[12]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[13]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[14]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[15]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[16]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[17]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[18]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[19]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS.fields[20]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _MESSAGEOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _MESSAGEOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _MESSAGEOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _MESSAGEOPTIONS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _MESSAGEOPTIONS.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _MESSAGEOPTIONS.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _MESSAGEOPTIONS.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _MESSAGEOPTIONS.fields[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[8]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[9]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[10]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[11]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[12]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS.fields[13]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_EDITIONDEFAULT._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_EDITIONDEFAULT.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_EDITIONDEFAULT.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_FEATURESUPPORT._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_FEATURESUPPORT.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_FEATURESUPPORT.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_FEATURESUPPORT.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_FEATURESUPPORT.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ONEOFOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ONEOFOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ONEOFOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMOPTIONS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMOPTIONS.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMOPTIONS.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEOPTIONS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEOPTIONS.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _ENUMVALUEOPTIONS.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SERVICEOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SERVICEOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SERVICEOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SERVICEOPTIONS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION.fields[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION_NAMEPART._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION_NAMEPART.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["LEGACY_REQUIRED"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _UNINTERPRETEDOPTION_NAMEPART.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["LEGACY_REQUIRED"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET.fields[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET.fields[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET.fields[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_VISIBILITYFEATURE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESETDEFAULTS._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESETDEFAULTS.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESETDEFAULTS.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESETDEFAULTS.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SOURCECODEINFO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SOURCECODEINFO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SOURCECODEINFO_LOCATION._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SOURCECODEINFO_LOCATION.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["PACKED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SOURCECODEINFO_LOCATION.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["PACKED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SOURCECODEINFO_LOCATION.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SOURCECODEINFO_LOCATION.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SOURCECODEINFO_LOCATION.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION.fields[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["PACKED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION.fields[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION.fields[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION.fields[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION.fields[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[8]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[9]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[10]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[11]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[12]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[13]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[14]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[15]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[16]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_TYPE.values[17]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_LABEL._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_LABEL.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_LABEL.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDDESCRIPTORPROTO_LABEL.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS_OPTIMIZEMODE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS_OPTIMIZEMODE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS_OPTIMIZEMODE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FILEOPTIONS_OPTIMIZEMODE.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_CTYPE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_CTYPE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_CTYPE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_CTYPE.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_JSTYPE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_JSTYPE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_JSTYPE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_JSTYPE.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONRETENTION._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONRETENTION.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONRETENTION.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONRETENTION.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[8]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FIELDOPTIONS_OPTIONTARGETTYPE.values[9]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS_IDEMPOTENCYLEVEL._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS_IDEMPOTENCYLEVEL.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS_IDEMPOTENCYLEVEL.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _METHODOPTIONS_IDEMPOTENCYLEVEL.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_FIELDPRESENCE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_FIELDPRESENCE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_FIELDPRESENCE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_FIELDPRESENCE.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_FIELDPRESENCE.values[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_ENUMTYPE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_ENUMTYPE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_ENUMTYPE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_ENUMTYPE.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_REPEATEDFIELDENCODING._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_REPEATEDFIELDENCODING.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_REPEATEDFIELDENCODING.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_REPEATEDFIELDENCODING.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_UTF8VALIDATION._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_UTF8VALIDATION.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_UTF8VALIDATION.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_UTF8VALIDATION.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_MESSAGEENCODING._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_MESSAGEENCODING.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_MESSAGEENCODING.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_MESSAGEENCODING.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_JSONFORMAT._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_JSONFORMAT.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_JSONFORMAT.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_JSONFORMAT.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_ENFORCENAMINGSTYLE._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_ENFORCENAMINGSTYLE.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_ENFORCENAMINGSTYLE.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_ENFORCENAMINGSTYLE.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY.values[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY.values[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION_SEMANTIC._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION_SEMANTIC.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION_SEMANTIC.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _GENERATEDCODEINFO_ANNOTATION_SEMANTIC.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[3]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[4]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[5]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[6]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[7]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[8]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[9]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[10]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _EDITION.values[11]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SYMBOLVISIBILITY._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SYMBOLVISIBILITY.values[0]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SYMBOLVISIBILITY.values[1]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
  _SYMBOLVISIBILITY.values[2]._features = _ResolvedFeatures(field_presence=_FEATURESET_FIELDPRESENCE.values_by_name["EXPLICIT"].number,enum_type=_FEATURESET_ENUMTYPE.values_by_name["CLOSED"].number,repeated_field_encoding=_FEATURESET_REPEATEDFIELDENCODING.values_by_name["EXPANDED"].number,utf8_validation=_FEATURESET_UTF8VALIDATION.values_by_name["NONE"].number,message_encoding=_FEATURESET_MESSAGEENCODING.values_by_name["LENGTH_PREFIXED"].number,json_format=_FEATURESET_JSONFORMAT.values_by_name["LEGACY_BEST_EFFORT"].number)
else:
  _builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.descriptor_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\023com.google.protobufB\020DescriptorProtosH\001Z-google.golang.org/protobuf/types/descriptorpb\370\001\001\242\002\003GPB\252\002\032Google.Protobuf.Reflection'
  _globals['_EXTENSIONRANGEOPTIONS'].fields_by_name['declaration']._loaded_options = None
  _globals['_EXTENSIONRANGEOPTIONS'].fields_by_name['declaration']._serialized_options = b'\210\001\002'
  _globals['_EXTENSIONRANGEOPTIONS'].fields_by_name['verification']._loaded_options = None
  _globals['_EXTENSIONRANGEOPTIONS'].fields_by_name['verification']._serialized_options = b'\210\001\002'
  _globals['_FILEOPTIONS'].fields_by_name['java_generate_equals_and_hash']._loaded_options = None
  _globals['_FILEOPTIONS'].fields_by_name['java_generate_equals_and_hash']._serialized_options = b'\030\001'
  _globals['_MESSAGEOPTIONS'].fields_by_name['deprecated_legacy_json_field_conflicts']._loaded_options = None
  _globals['_MESSAGEOPTIONS'].fields_by_name['deprecated_legacy_json_field_conflicts']._serialized_options = b'\030\001'
  _globals['_ENUMOPTIONS'].fields_by_name['deprecated_legacy_json_field_conflicts']._loaded_options = None
  _globals['_ENUMOPTIONS'].fields_by_name['deprecated_legacy_json_field_conflicts']._serialized_options = b'\030\001'
  _globals['_FEATURESET'].fields_by_name['field_presence']._loaded_options = None
  _globals['_FEATURESET'].fields_by_name['field_presence']._serialized_options = b'\210\001\001\230\001\004\230\001\001\242\001\r\022\010EXPLICIT\030\204\007\242\001\r\022\010IMPLICIT\030\347\007\242\001\r\022\010EXPLICIT\030\350\007\262\001\003\010\350\007'
  _globals['_FEATURESET'].fields_by_name['enum_type']._loaded_options = None
  _globals['_FEATURESET'].fields_by_name['enum_type']._serialized_options = b'\210\001\001\230\001\006\230\001\001\242\001\013\022\006CLOSED\030\204\007\242\001\t\022\004OPEN\030\347\007\262\001\003\010\350\007'
  _globals['_FEATURESET'].fields_by_name['repeated_field_encoding']._loaded_options = None
  _globals['_FEATURESET'].fields_by_name['repeated_field_encoding']._serialized_options = b'\210\001\001\230\001\004\230\001\001\242\001\r\022\010EXPANDED\030\204\007\242\001\013\022\006PACKED\030\347\007\262\001\003\010\350\007'
  _globals['_FEATURESET'].fields_by_name['utf8_validation']._loaded_options = None
  _globals['_FEATURESET'].fields_by_name['utf8_validation']._serialized_options = b'\210\001\001\230\001\004\230\001\001\242\001\t\022\004NONE\030\204\007\242\001\013\022\006VERIFY\030\347\007\262\001\003\010\350\007'
  _globals['_FEATURESET'].fields_by_name['message_encoding']._loaded_options = None
  _globals['_FEATURESET'].fields_by_name['message_encoding']._serialized_options = b'\210\001\001\230\001\004\230\001\001\242\001\024\022\017LENGTH_PREFIXED\030\204\007\262\001\003\010\350\007'
  _globals['_FEATURESET'].fields_by_name['json_format']._loaded_options = None
  _globals['_FEATURESET'].fields_by_name['json_format']._serialized_options = b'\210\001\001\230\001\003\230\001\006\230\001\001\242\001\027\022\022LEGACY_BEST_EFFORT\030\204\007\242\001\n\022\005ALLOW\030\347\007\262\001\003\010\350\007'
  _globals['_FEATURESET'].fields_by_name['enforce_naming_style']._loaded_options = None
  _globals['_FEATURESET'].fields_by_name['enforce_naming_style']._serialized_options = b'\210\001\002\230\001\001\230\001\002\230\001\003\230\001\004\230\001\005\230\001\006\230\001\007\230\001\010\230\001\t\242\001\021\022\014STYLE_LEGACY\030\204\007\242\001\016\022\tSTYLE2024\030\351\007\262\001\003\010\351\007'
  _globals['_FEATURESET'].fields_by_name['default_symbol_visibility']._loaded_options = None
  _globals['_FEATURESET'].fields_by_name['default_symbol_visibility']._serialized_options = b'\210\001\002\230\001\001\242\001\017\022\nEXPORT_ALL\030\204\007\242\001\025\022\020EXPORT_TOP_LEVEL\030\351\007\262\001\003\010\351\007'
  _globals['_SOURCECODEINFO_LOCATION'].fields_by_name['path']._loaded_options = None
  _globals['_SOURCECODEINFO_LOCATION'].fields_by_name['path']._serialized_options = b'\020\001'
  _globals['_SOURCECODEINFO_LOCATION'].fields_by_name['span']._loaded_options = None
  _globals['_SOURCECODEINFO_LOCATION'].fields_by_name['span']._serialized_options = b'\020\001'
  _globals['_GENERATEDCODEINFO_ANNOTATION'].fields_by_name['path']._loaded_options = None
  _globals['_GENERATEDCODEINFO_ANNOTATION'].fields_by_name['path']._serialized_options = b'\020\001'
  _globals['_EDITION']._serialized_start=12667
  _globals['_EDITION']._serialized_end=12962
  _globals['_SYMBOLVISIBILITY']._serialized_start=12964
  _globals['_SYMBOLVISIBILITY']._serialized_end=13049
  _globals['_FILEDESCRIPTORSET']._serialized_start=53
  _globals['_FILEDESCRIPTORSET']._serialized_end=144
  _globals['_FILEDESCRIPTORPROTO']._serialized_start=147
  _globals['_FILEDESCRIPTORPROTO']._serialized_end=856
  _globals['_DESCRIPTORPROTO']._serialized_start=859
  _globals['_DESCRIPTORPROTO']._serialized_end=1751
  _globals['_DESCRIPTORPROTO_EXTENSIONRANGE']._serialized_start=1572
  _globals['_DESCRIPTORPROTO_EXTENSIONRANGE']._serialized_end=1694
  _globals['_DESCRIPTORPROTO_RESERVEDRANGE']._serialized_start=1696
  _globals['_DESCRIPTORPROTO_RESERVEDRANGE']._serialized_end=1751
  _globals['_EXTENSIONRANGEOPTIONS']._serialized_start=1754
  _globals['_EXTENSIONRANGEOPTIONS']._serialized_end=2342
  _globals['_EXTENSIONRANGEOPTIONS_DECLARATION']._serialized_start=2129
  _globals['_EXTENSIONRANGEOPTIONS_DECLARATION']._serialized_end=2277
  _globals['_EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE']._serialized_start=2279
  _globals['_EXTENSIONRANGEOPTIONS_VERIFICATIONSTATE']._serialized_end=2331
  _globals['_FIELDDESCRIPTORPROTO']._serialized_start=2345
  _globals['_FIELDDESCRIPTORPROTO']._serialized_end=3178
  _globals['_FIELDDESCRIPTORPROTO_TYPE']._serialized_start=2799
  _globals['_FIELDDESCRIPTORPROTO_TYPE']._serialized_end=3109
  _globals['_FIELDDESCRIPTORPROTO_LABEL']._serialized_start=3111
  _globals['_FIELDDESCRIPTORPROTO_LABEL']._serialized_end=3178
  _globals['_ONEOFDESCRIPTORPROTO']._serialized_start=3180
  _globals['_ONEOFDESCRIPTORPROTO']._serialized_end=3279
  _globals['_ENUMDESCRIPTORPROTO']._serialized_start=3282
  _globals['_ENUMDESCRIPTORPROTO']._serialized_end=3704
  _globals['_ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE']._serialized_start=3645
  _globals['_ENUMDESCRIPTORPROTO_ENUMRESERVEDRANGE']._serialized_end=3704
  _globals['_ENUMVALUEDESCRIPTORPROTO']._serialized_start=3707
  _globals['_ENUMVALUEDESCRIPTORPROTO']._serialized_end=3838
  _globals['_SERVICEDESCRIPTORPROTO']._serialized_start=3841
  _globals['_SERVICEDESCRIPTORPROTO']._serialized_end=4008
  _globals['_METHODDESCRIPTORPROTO']._serialized_start=4011
  _globals['_METHODDESCRIPTORPROTO']._serialized_end=4276
  _globals['_FILEOPTIONS']._serialized_start=4279
  _globals['_FILEOPTIONS']._serialized_end=5476
  _globals['_FILEOPTIONS_OPTIMIZEMODE']._serialized_start=5373
  _globals['_FILEOPTIONS_OPTIMIZEMODE']._serialized_end=5431
  _globals['_MESSAGEOPTIONS']._serialized_start=5479
  _globals['_MESSAGEOPTIONS']._serialized_end=5979
  _globals['_FIELDOPTIONS']._serialized_start=5982
  _globals['_FIELDOPTIONS']._serialized_end=7675
  _globals['_FIELDOPTIONS_EDITIONDEFAULT']._serialized_start=6819
  _globals['_FIELDOPTIONS_EDITIONDEFAULT']._serialized_end=6909
  _globals['_FIELDOPTIONS_FEATURESUPPORT']._serialized_start=6912
  _globals['_FIELDOPTIONS_FEATURESUPPORT']._serialized_end=7190
  _globals['_FIELDOPTIONS_CTYPE']._serialized_start=7192
  _globals['_FIELDOPTIONS_CTYPE']._serialized_end=7239
  _globals['_FIELDOPTIONS_JSTYPE']._serialized_start=7241
  _globals['_FIELDOPTIONS_JSTYPE']._serialized_end=7294
  _globals['_FIELDOPTIONS_OPTIONRETENTION']._serialized_start=7296
  _globals['_FIELDOPTIONS_OPTIONRETENTION']._serialized_end=7381
  _globals['_FIELDOPTIONS_OPTIONTARGETTYPE']._serialized_start=7384
  _globals['_FIELDOPTIONS_OPTIONTARGETTYPE']._serialized_end=7652
  _globals['_ONEOFOPTIONS']._serialized_start=7678
  _globals['_ONEOFOPTIONS']._serialized_end=7850
  _globals['_ENUMOPTIONS']._serialized_start=7853
  _globals['_ENUMOPTIONS']._serialized_end=8190
  _globals['_ENUMVALUEOPTIONS']._serialized_start=8193
  _globals['_ENUMVALUEOPTIONS']._serialized_end=8537
  _globals['_SERVICEOPTIONS']._serialized_start=8540
  _globals['_SERVICEOPTIONS']._serialized_end=8753
  _globals['_METHODOPTIONS']._serialized_start=8756
  _globals['_METHODOPTIONS']._serialized_end=9165
  _globals['_METHODOPTIONS_IDEMPOTENCYLEVEL']._serialized_start=9074
  _globals['_METHODOPTIONS_IDEMPOTENCYLEVEL']._serialized_end=9154
  _globals['_UNINTERPRETEDOPTION']._serialized_start=9168
  _globals['_UNINTERPRETEDOPTION']._serialized_end=9578
  _globals['_UNINTERPRETEDOPTION_NAMEPART']._serialized_start=9504
  _globals['_UNINTERPRETEDOPTION_NAMEPART']._serialized_end=9578
  _globals['_FEATURESET']._serialized_start=9581
  _globals['_FEATURESET']._serialized_end=11515
  _globals['_FEATURESET_VISIBILITYFEATURE']._serialized_start=10760
  _globals['_FEATURESET_VISIBILITYFEATURE']._serialized_end=10921
  _globals['_FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY']._serialized_start=10782
  _globals['_FEATURESET_VISIBILITYFEATURE_DEFAULTSYMBOLVISIBILITY']._serialized_end=10911
  _globals['_FEATURESET_FIELDPRESENCE']._serialized_start=10923
  _globals['_FEATURESET_FIELDPRESENCE']._serialized_end=11015
  _globals['_FEATURESET_ENUMTYPE']._serialized_start=11017
  _globals['_FEATURESET_ENUMTYPE']._serialized_end=11072
  _globals['_FEATURESET_REPEATEDFIELDENCODING']._serialized_start=11074
  _globals['_FEATURESET_REPEATEDFIELDENCODING']._serialized_end=11160
  _globals['_FEATURESET_UTF8VALIDATION']._serialized_start=11162
  _globals['_FEATURESET_UTF8VALIDATION']._serialized_end=11235
  _globals['_FEATURESET_MESSAGEENCODING']._serialized_start=11237
  _globals['_FEATURESET_MESSAGEENCODING']._serialized_end=11320
  _globals['_FEATURESET_JSONFORMAT']._serialized_start=11322
  _globals['_FEATURESET_JSONFORMAT']._serialized_end=11394
  _globals['_FEATURESET_ENFORCENAMINGSTYLE']._serialized_start=11396
  _globals['_FEATURESET_ENFORCENAMINGSTYLE']._serialized_end=11483
  _globals['_FEATURESETDEFAULTS']._serialized_start=11518
  _globals['_FEATURESETDEFAULTS']._serialized_end=12013
  _globals['_FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT']._serialized_start=11765
  _globals['_FEATURESETDEFAULTS_FEATURESETEDITIONDEFAULT']._serialized_end=12013
  _globals['_SOURCECODEINFO']._serialized_start=12016
  _globals['_SOURCECODEINFO']._serialized_end=12325
  _globals['_SOURCECODEINFO_LOCATION']._serialized_start=12105
  _globals['_SOURCECODEINFO_LOCATION']._serialized_end=12311
  _globals['_GENERATEDCODEINFO']._serialized_start=12328
  _globals['_GENERATEDCODEINFO']._serialized_end=12664
  _globals['_GENERATEDCODEINFO_ANNOTATION']._serialized_start=12429
  _globals['_GENERATEDCODEINFO_ANNOTATION']._serialized_end=12664
  _globals['_GENERATEDCODEINFO_ANNOTATION_SEMANTIC']._serialized_start=12624
  _globals['_GENERATEDCODEINFO_ANNOTATION_SEMANTIC']._serialized_end=12664
# @@protoc_insertion_point(module_scope)